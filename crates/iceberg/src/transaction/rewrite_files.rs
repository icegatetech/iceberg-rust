// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{DataFile, FormatVersion, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer, collect_manifests_entries,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// `RewriteFilesAction` is a transaction action that atomically replaces one set of
/// data files in the table with another set, producing a `replace` snapshot.
///
/// This is the building block for compaction. The caller guarantees that the
/// `added` files hold exactly the same live rows as the `removed` files; this
/// action only rewrites table metadata, never row data. The removed files are
/// dropped from the new snapshot (a manifest that contains a removed file is
/// rewritten, re-emitting its surviving entries as `Existing` and preserving their
/// data sequence numbers; an untouched manifest is carried forward unchanged), and
/// the added files are written into a single new manifest. Per-commit manifest
/// rewriting is therefore proportional to the affected manifests, not the whole table.
///
/// The commit is guarded by an optimistic `RefSnapshotIdMatch` requirement on the
/// table's main ref (added by the underlying snapshot producer), so a concurrent
/// commit causes a retryable conflict rather than a lost update.
pub struct RewriteFilesAction {
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    removed_data_files: Vec<DataFile>,
    inherit_summary_keys: Vec<String>,
    // Opt-in assertion that, on a V3 table, every added file guaranteed `_row_id`.
    row_lineage_guaranteed: bool,
}

impl RewriteFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            removed_data_files: vec![],
            inherit_summary_keys: vec![],
            row_lineage_guaranteed: false,
        }
    }

    /// Add data files to the table in this rewrite.
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Remove data files from the table in this rewrite. Each removed file must be
    /// live in the table's current snapshot, otherwise the commit fails.
    pub fn delete_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.removed_data_files.extend(data_files);
        self
    }

    /// Set commit UUID for the snapshot.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set key metadata for manifest files.
    pub fn set_key_metadata(mut self, key_metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(key_metadata);
        self
    }

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }

    /// Carry forward a snapshot-summary property from the snapshot this rewrite
    /// supersedes (its parent — the current snapshot of the base the commit is
    /// produced against) onto the new `replace` snapshot, UNCHANGED.
    ///
    /// The value is resolved at snapshot-production time, so under the
    /// optimistic-concurrency retry it reflects the FRESH base after any racing
    /// commit — never a value captured before the race. This is how a `replace`
    /// (which otherwise drops summary properties) preserves a monotonic marker
    /// such as a write-ahead-log offset without the caller freezing a possibly
    /// stale value. An explicit [`Self::set_snapshot_properties`] entry for the same
    /// key takes precedence over the inherited value.
    pub fn inherit_summary_property(mut self, key: impl Into<String>) -> Self {
        self.inherit_summary_keys.push(key.into());
        self
    }

    /// Confirms the added files already carry the correct `_row_id` and
    /// `_last_updated_sequence_number` for their rows, which unlocks rewrites on
    /// V3 tables.
    ///
    /// On V3 every row keeps a stable identity across a rewrite, but that data
    /// lives inside the added Parquet files — which this action never reads, so
    /// it can neither check nor fix it. It therefore rejects a V3 rewrite by
    /// default; call this to promise the writer produced those columns correctly.
    /// A false promise corrupts row identity, so only call it when it holds. No
    /// effect on V1/V2 tables, which have no row lineage.
    pub fn row_lineage_guaranteed(mut self) -> Self {
        self.row_lineage_guaranteed = true;
        self
    }
}

#[async_trait]
impl TransactionAction for RewriteFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if self.added_data_files.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "RewriteFilesAction requires at least one added data file",
            ));
        }
        if self.removed_data_files.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "RewriteFilesAction requires at least one removed data file",
            ));
        }

        // On a V3 table the added files must physically carry the preserved
        // `_row_id` / `_last_updated_sequence_number` of the rewritten rows — a
        // guarantee that lives in the caller's writer and cannot be checked here.
        // Reject before writing anything unless the caller opted in. A rewrite
        // always has added files (checked above), so this gates every unasserted
        // V3 rewrite; the survivor-side lineage handling below is unaffected and
        // stays available to a compliant caller. V1/V2 have no row lineage.
        if table.metadata().format_version() == FormatVersion::V3 && !self.row_lineage_guaranteed {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "RewriteFilesAction on a V3 table cannot confirm that the added files preserve \
                 each rewritten row's _row_id and _last_updated_sequence_number. Call \
                 row_lineage_guaranteed() to assert the writer materialized both system \
                 columns, or run compaction on a V1/V2 table.",
            ));
        }

        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
        )
        .with_inherit_summary_keys(self.inherit_summary_keys.clone());

        // Validate the added files (content type, partition spec) the same way fast
        // append does. The duplicate-file check is intentionally skipped: the added
        // files are freshly written outputs, and overlap with kept files would be a
        // caller bug surfaced by the row data, not by metadata.
        snapshot_producer.validate_added_data_files()?;

        snapshot_producer
            .commit(
                RewriteFilesOperation {
                    removed_data_files: self.removed_data_files.clone(),
                },
                DefaultManifestProcess,
            )
            .await
    }
}

/// Snapshot operation for [`RewriteFilesAction`]: records `Operation::Replace`,
/// excludes the removed files from the carried-forward manifests, and reports the
/// removed files for snapshot-summary accounting.
struct RewriteFilesOperation {
    removed_data_files: Vec<DataFile>,
}

impl SnapshotProduceOperation for RewriteFilesOperation {
    fn operation(&self) -> Operation {
        Operation::Replace
    }

    fn removed_data_files(&self) -> &[DataFile] {
        &self.removed_data_files
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        // Removals are expressed by excluding the removed files from the
        // carried-forward manifests in `existing_manifest`, not via delete entries.
        Ok(vec![])
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let Some(snapshot) = snapshot_produce.table.metadata().current_snapshot() else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Cannot rewrite files on a table with no current snapshot",
            ));
        };

        let manifest_list = snapshot
            .load_manifest_list(
                snapshot_produce.table.file_io(),
                &snapshot_produce.table.metadata_ref(),
            )
            .await?;

        let removed_file_paths: HashSet<String> = self
            .removed_data_files
            .iter()
            .map(|file| file.file_path().to_string())
            .collect();

        // On a V3 table each live data file's inherited `first_row_id` must be
        // materialized when a manifest is rewritten.
        let row_lineage_required =
            snapshot_produce.table.metadata().format_version() == FormatVersion::V3;

        let mut kept_manifests: Vec<ManifestFile> = Vec::new();
        let mut removed_found: HashSet<String> = HashSet::new();

        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file
                .load_manifest(snapshot_produce.table.file_io())
                .await?;

            // Single pass over the entries: record which removed files this manifest
            // accounts for (checked against the requested set once every manifest has
            // been visited) and whether it is affected at all.
            let mut touches_removed = false;
            for entry in manifest.entries() {
                if entry.is_alive() && removed_file_paths.contains(entry.file_path()) {
                    touches_removed = true;
                    removed_found.insert(entry.file_path().to_string());
                }
            }

            if !touches_removed {
                // No removed file lives in this manifest: carry it forward unchanged.
                kept_manifests.push(manifest_file.clone());
                continue;
            }

            if row_lineage_required && manifest_file.first_row_id.is_none() {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    "Cannot rewrite a V3 data manifest without an assigned first_row_id: \
                     dropping files from it would reassign row lineage",
                ));
            }

            // Re-emit the surviving live entries as `Existing`, materializing each
            // one's inherited `first_row_id` from the source manifest order so the
            // removed files' offsets are preserved.
            let mut survivors: Vec<ManifestEntry> = Vec::new();
            collect_manifests_entries(
                manifest_file,
                manifest.entries(),
                row_lineage_required,
                |entry| entry.is_alive() && !removed_file_paths.contains(entry.file_path()),
                &mut survivors,
            )?;

            // A rewritten manifest with no survivors is dropped entirely.
            if survivors.is_empty() {
                continue;
            }

            // Re-emit the survivors under the manifest's OWN schema and partition
            // spec — the ones its entries were written with. A touched manifest may
            // belong to an older spec than the current default (whose source columns
            // may even be absent from the current schema), so taking the table's
            // default spec / current schema would relabel the survivors under the
            // wrong spec or fail to serialize their historical partition tuples.
            let mut writer = snapshot_produce.new_rewrite_manifest_writer(
                manifest.metadata().schema().clone(),
                manifest.metadata().partition_spec().clone(),
            )?;
            // Smallest materialized `first_row_id` among survivors becomes the
            // rewritten manifest's assigned base on V3, so the manifest-list writer
            // does not reserve a new row-id range. Stays `None` on V1/V2.
            let mut min_row_id: Option<u64> = None;
            for entry in &survivors {
                let snapshot_id = entry.snapshot_id().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Existing manifest entry is missing a snapshot id",
                    )
                })?;
                let sequence_number = entry.sequence_number().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Existing manifest entry is missing a data sequence number",
                    )
                })?;
                if let Some(first_row_id) = entry.data_file().first_row_id() {
                    // `collect_manifests_entries` already rejected negatives; convert
                    // via `try_from` rather than `as u64` so no invalid value can
                    // silently wrap into a bogus manifest base.
                    let first_row_id = u64::try_from(first_row_id).map_err(|_| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "Negative first_row_id in a re-emitted manifest entry",
                        )
                    })?;
                    min_row_id = Some(min_row_id.map_or(first_row_id, |min| min.min(first_row_id)));
                }
                // TODO(med): add deleted files to manifest entry - writer.add_delete_file
                writer.add_existing_file(
                    entry.data_file().clone(),
                    snapshot_id,
                    sequence_number,
                    entry.file_sequence_number,
                )?;
            }

            if row_lineage_required {
                writer.set_first_row_id(min_row_id);
            }
            kept_manifests.push(writer.write_manifest_file().await?);
        }

        if removed_found.len() != removed_file_paths.len() {
            let mut missing: Vec<&str> = removed_file_paths
                .iter()
                .filter(|path| !removed_found.contains(*path))
                .map(String::as_str)
                .collect();
            missing.sort_unstable();
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot rewrite files that are not live in the current snapshot: {}",
                    missing.join(", ")
                ),
            ));
        }

        Ok(kept_manifests)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal,
        Operation, Struct,
    };
    use crate::transaction::tests::{
        make_v2_minimal_table_in_catalog, make_v3_minimal_table_in_catalog,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction, TransactionAction};
    use crate::{ErrorKind, TableUpdate};

    fn file(path: &str, rows: u64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(rows)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    async fn live_file_paths(table: &crate::table::Table) -> Vec<String> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut paths = Vec::new();
        for entry in manifest_list.entries() {
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            for e in manifest.entries() {
                if e.is_alive() {
                    paths.push(e.file_path().to_string());
                }
            }
        }
        paths.sort();
        paths
    }

    /// Seed three files in one manifest, then rewrite removing two and adding one.
    /// The surviving file must keep its original data sequence number, the removed
    /// files must be gone, and the snapshot operation must be `Replace`.
    #[tokio::test]
    async fn rewrite_files_partial_removal_preserves_survivor_sequence_number() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // One fast append → one manifest holding f1, f2, f3.
        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .add_data_files(vec![
                file("a.parquet", 1),
                file("b.parquet", 1),
                file("c.parquet", 1),
            ])
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        // Record the survivor (b)'s data sequence number before the rewrite.
        let base_snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = base_snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut survivor_seq_before = None;
        for entry in manifest_list.entries() {
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            for e in manifest.entries() {
                if e.file_path() == "b.parquet" {
                    survivor_seq_before = e.sequence_number();
                }
            }
        }
        assert!(survivor_seq_before.is_some());

        // Rewrite: remove a + c, add merged.parquet.
        let tx = Transaction::new(&table);
        let mut action_commit = Arc::new(
            tx.rewrite_files()
                .row_lineage_guaranteed()
                .add_data_files(vec![file("merged.parquet", 2)])
                .delete_files(vec![file("a.parquet", 1), file("c.parquet", 1)]),
        )
        .commit(&table)
        .await
        .unwrap();

        // The action emits a Replace snapshot.
        let updates = action_commit.take_updates();
        let new_snapshot = match &updates[0] {
            TableUpdate::AddSnapshot { snapshot } => snapshot,
            _ => unreachable!("first update must be AddSnapshot"),
        };
        assert_eq!(new_snapshot.summary().operation, Operation::Replace);
        assert_eq!(
            new_snapshot.parent_snapshot_id(),
            table.metadata().current_snapshot_id()
        );

        // Commit through the generic catalog and verify the live file set + survivor seq.
        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_files()
            .row_lineage_guaranteed()
            .add_data_files(vec![file("merged.parquet", 2)])
            .delete_files(vec![file("a.parquet", 1), file("c.parquet", 1)])
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        assert_eq!(live_file_paths(&table).await, vec![
            "b.parquet".to_string(),
            "merged.parquet".to_string()
        ]);

        let mut survivor_seq_after = None;
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        for entry in manifest_list.entries() {
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            for e in manifest.entries() {
                if e.file_path() == "b.parquet" {
                    survivor_seq_after = e.sequence_number();
                }
            }
        }
        assert_eq!(
            survivor_seq_before, survivor_seq_after,
            "survivor data sequence number must be preserved across the rewrite"
        );
    }

    async fn first_row_id_of(table: &crate::table::Table, path: &str) -> Option<i64> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        for entry in manifest_list.entries() {
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            for e in manifest.entries() {
                if e.file_path() == path {
                    return e.data_file().first_row_id();
                }
            }
        }
        None
    }

    /// Removing a file that precedes survivors in a V3 manifest shifts the positions
    /// their `first_row_id`s are inherited from. The rewrite must materialize each
    /// survivor's inherited id so its `_row_id`s stay pinned, and must not reassign
    /// the row lineage of the untouched files.
    #[tokio::test]
    async fn rewrite_files_preserves_v3_survivor_row_lineage() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // One manifest, base 0: a→0 (2 rows), b→2 (3 rows), c→5 (1 row). next=6.
        let table = {
            let tx = Transaction::new(&table);
            tx.fast_append()
                .add_data_files(vec![
                    file("a.parquet", 2),
                    file("b.parquet", 3),
                    file("c.parquet", 1),
                ])
                .apply(tx)
                .unwrap()
                .commit(&catalog)
                .await
                .unwrap()
        };
        assert_eq!(first_row_id_of(&table, "b.parquet").await, None); // null on disk
        let next_row_id_before = table.metadata().next_row_id();
        assert_eq!(next_row_id_before, 6);

        // Remove a (2 rows), replacing it with a new file of the same row count.
        let table = {
            let tx = Transaction::new(&table);
            tx.rewrite_files()
                .row_lineage_guaranteed()
                .add_data_files(vec![file("merged.parquet", 2)])
                .delete_files(vec![file("a.parquet", 2)])
                .apply(tx)
                .unwrap()
                .commit(&catalog)
                .await
                .unwrap()
        };

        assert_eq!(live_file_paths(&table).await, vec![
            "b.parquet".to_string(),
            "c.parquet".to_string(),
            "merged.parquet".to_string(),
        ]);
        // Survivors keep their original inherited ids, now materialized — despite
        // `a` (2 rows) being dropped from in front of them.
        assert_eq!(first_row_id_of(&table, "b.parquet").await, Some(2));
        assert_eq!(first_row_id_of(&table, "c.parquet").await, Some(5));
        // The survivors' rewritten manifest keeps its own base and reserves no new
        // range; only the added file's 2 rows advance next_row_id (6 → 8).
        assert_eq!(table.metadata().next_row_id(), 8);
    }

    /// Removing a file that is not live in the current snapshot must fail.
    #[tokio::test]
    async fn rewrite_files_rejects_unknown_removed_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .add_data_files(vec![file("a.parquet", 1)])
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        let result = Arc::new(
            tx_for(&table)
                .rewrite_files()
                .row_lineage_guaranteed()
                .add_data_files(vec![file("merged.parquet", 1)])
                .delete_files(vec![file("ghost.parquet", 1)]),
        )
        .commit(&table)
        .await;
        assert!(result.is_err(), "removing a non-live file must fail");
    }

    fn tx_for(table: &crate::table::Table) -> Transaction {
        Transaction::new(table)
    }

    /// A V3 rewrite is refused before any object is written unless the caller opts
    /// in that the added files preserve row lineage. The default (no opt-in)
    /// surfaces `FeatureUnsupported`, so a compaction writer that does not yet
    /// materialize `_row_id` / `_last_updated_sequence_number` in its outputs
    /// cannot silently reassign row identity through this action.
    #[tokio::test]
    async fn rewrite_files_rejects_v3_without_row_lineage_optin() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = {
            let tx = Transaction::new(&table);
            tx.fast_append()
                .add_data_files(vec![file("a.parquet", 1)])
                .apply(tx)
                .unwrap()
                .commit(&catalog)
                .await
                .unwrap()
        };

        let err = Arc::new(
            tx_for(&table)
                .rewrite_files()
                .add_data_files(vec![file("merged.parquet", 1)])
                .delete_files(vec![file("a.parquet", 1)]),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::FeatureUnsupported,
            "a V3 rewrite without the row-lineage opt-in must be refused: {err}"
        );
    }

    /// Even WITH the row-lineage opt-in, a rewrite that must drop files from a V3
    /// manifest carrying no assigned `first_row_id` (a pre-lineage V2→V3 carry-over)
    /// is rejected: re-emitting its survivors would inherit ids from a missing base.
    /// A file is appended under V2, the table is upgraded to V3 (leaving that
    /// manifest's `first_row_id` null), and a rewrite removing that file surfaces
    /// `FeatureUnsupported` — the symmetric counterpart of the guard in
    /// `RewriteManifestsAction`.
    #[tokio::test]
    async fn rewrite_files_rejects_v3_manifest_without_first_row_id() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        // A V2 manifest holding a.parquet: no first_row_id is assigned.
        let table = {
            let tx = Transaction::new(&table);
            tx.fast_append()
                .add_data_files(vec![file("a.parquet", 1)])
                .apply(tx)
                .unwrap()
                .commit(&catalog)
                .await
                .unwrap()
        };

        // Upgrade to V3: only bumps the version, so a.parquet's manifest stays
        // first_row_id = null and is now read under row lineage.
        let table = {
            let tx = Transaction::new(&table);
            tx.upgrade_table_version()
                .set_format_version(FormatVersion::V3)
                .apply(tx)
                .unwrap()
                .commit(&catalog)
                .await
                .unwrap()
        };

        // The opt-in clears the up-front V3 gate, so the commit reaches the
        // per-manifest guard: dropping a.parquet touches its null-base manifest.
        let err = Arc::new(
            tx_for(&table)
                .rewrite_files()
                .row_lineage_guaranteed()
                .add_data_files(vec![file("merged.parquet", 1)])
                .delete_files(vec![file("a.parquet", 1)]),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::FeatureUnsupported,
            "unexpected error: {err}"
        );
        assert!(
            err.message().contains("first_row_id"),
            "unexpected error: {}",
            err.message()
        );
    }
}
