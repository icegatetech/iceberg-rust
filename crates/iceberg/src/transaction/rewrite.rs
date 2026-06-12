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
use crate::spec::{DataFile, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
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
    /// stale value. An explicit [`set_snapshot_properties`] entry for the same
    /// key takes precedence over the inherited value.
    pub fn inherit_summary_property(mut self, key: impl Into<String>) -> Self {
        self.inherit_summary_keys.push(key.into());
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

        let mut kept_manifests: Vec<ManifestFile> = Vec::new();
        let mut removed_found: HashSet<String> = HashSet::new();

        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file
                .load_manifest(snapshot_produce.table.file_io())
                .await?;

            let touches_removed = manifest
                .entries()
                .iter()
                .any(|entry| entry.is_alive() && removed_file_paths.contains(entry.file_path()));

            if !touches_removed {
                // No removed file lives in this manifest: carry it forward unchanged.
                kept_manifests.push(manifest_file.clone());
                continue;
            }

            // Rewrite this manifest, re-emitting only the surviving live entries as
            // `Existing` so their original data sequence numbers are preserved.
            let mut writer = snapshot_produce.new_rewrite_manifest_writer()?;
            let mut survivors = 0usize;
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                if removed_file_paths.contains(entry.file_path()) {
                    removed_found.insert(entry.file_path().to_string());
                    continue;
                }
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
                writer.add_existing_file(
                    entry.data_file().clone(),
                    snapshot_id,
                    sequence_number,
                    entry.file_sequence_number,
                )?;
                survivors += 1;
            }

            if survivors > 0 {
                kept_manifests.push(writer.write_manifest_file().await?);
            }
            // A rewritten manifest with no survivors is dropped entirely.
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

    use crate::TableUpdate;
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Operation, Struct,
    };
    use crate::transaction::tests::make_v3_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction, TransactionAction};

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
}
