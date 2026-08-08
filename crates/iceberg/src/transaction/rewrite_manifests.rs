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

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{
    FormatVersion, Literal, ManifestContentType, ManifestEntry, ManifestFile, Operation, SchemaRef,
    Struct,
};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer, collect_manifests_entries,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// `RewriteManifestsAction` repacks a set of small DATA manifests of the current
/// snapshot into fewer, larger ones, producing a `replace` snapshot.
pub struct RewriteManifestsAction {
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
    input_manifest_paths: Vec<String>,
    target_manifest_size_bytes: u64,
    inherit_summary_keys: Vec<String>,
}

/// Default target output-manifest size (8 MiB), matching the maintenance
/// configuration default. The caller normally overrides it.
const DEFAULT_TARGET_MANIFEST_SIZE_BYTES: u64 = 8 * 1024 * 1024;

/// Repacking a single manifest cannot reduce the manifest count, so the caller
/// must name at least this many distinct inputs for the commit to be worth making.
const MIN_INPUT_MANIFESTS: usize = 2;

impl RewriteManifestsAction {
    pub(crate) fn new() -> Self {
        Self {
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
            input_manifest_paths: vec![],
            target_manifest_size_bytes: DEFAULT_TARGET_MANIFEST_SIZE_BYTES,
            inherit_summary_keys: vec![],
        }
    }

    /// Add manifest object paths selected for repacking. Every path must name a
    /// DATA manifest that is live in the table's current snapshot; the commit
    /// fails otherwise.
    pub fn add_input_manifests(mut self, paths: impl IntoIterator<Item = String>) -> Self {
        self.input_manifest_paths.extend(paths);
        self
    }

    /// Set the target size of each output manifest. Entries are packed into an
    /// output manifest until its estimated size reaches this value.
    pub fn target_manifest_size_bytes(mut self, target: u64) -> Self {
        self.target_manifest_size_bytes = target;
        self
    }

    /// Set commit UUID for the snapshot.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }


    /// Set snapshot summary properties. At least one property is required: a
    /// manifest rewrite adds no data files, and the snapshot producer rejects a
    /// commit that has neither added files nor properties, so callers pass a
    /// marker property to record that this `replace` is a manifest rewrite.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }

    /// Carry forward a snapshot-summary property from the snapshot this rewrite
    /// supersedes (its parent) onto the new `replace` snapshot, UNCHANGED. See
    /// [`RewriteFilesAction::inherit_summary_property`](crate::transaction::rewrite_files::RewriteFilesAction::inherit_summary_property);
    /// this is how a `replace` preserves a monotonic marker such as a
    /// write-ahead-log offset without the caller freezing a possibly stale value.
    pub fn inherit_summary_property(mut self, key: impl Into<String>) -> Self {
        self.inherit_summary_keys.push(key.into());
        self
    }
}

#[async_trait]
impl TransactionAction for RewriteManifestsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // Checked before the producer is constructed, so a plan that cannot reduce
        // anything never opens a transaction and never grows `metadata.json` with an
        // empty `replace`. Distinct paths, not raw length: naming one manifest twice
        // is a planner bug, and it should surface as such rather than as a skip.
        let distinct_inputs: HashSet<&str> = self
            .input_manifest_paths
            .iter()
            .map(String::as_str)
            .collect();
        if distinct_inputs.len() < MIN_INPUT_MANIFESTS {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "rewrite_manifests requires at least two input manifests",
            ));
        }
        if self.snapshot_properties.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "rewrite_manifests requires at least one snapshot property (a marker) to avoid an empty commit",
            ));
        }
        if self.target_manifest_size_bytes == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "rewrite_manifests requires a positive target_manifest_size_bytes",
            ));
        }
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.snapshot_properties.clone(),
            vec![],
        )
        .with_inherit_summary_keys(self.inherit_summary_keys.clone());

        snapshot_producer
            .commit(
                RewriteManifestsOperation {
                    input_manifest_paths: self.input_manifest_paths.clone(),
                    target_manifest_size_bytes: self.target_manifest_size_bytes,
                },
                DefaultManifestProcess,
            )
            .await
    }
}

/// Snapshot operation for [`RewriteManifestsAction`]: records `Operation::Replace`
/// (data is unchanged; only metadata is rewritten), removes no data files, and
/// repacks the selected manifests in [`Self::existing_manifest`].
struct RewriteManifestsOperation {
    input_manifest_paths: Vec<String>,
    target_manifest_size_bytes: u64,
}

impl SnapshotProduceOperation for RewriteManifestsOperation {
    fn operation(&self) -> Operation {
        Operation::Replace
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        // TODO(med): iteration needs to be optimized

        let metadata = snapshot_produce.table.metadata();
        let Some(snapshot) = metadata.current_snapshot() else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Cannot rewrite manifests on a table with no current snapshot",
            ));
        };

        let manifest_list = snapshot_produce
            .table
            .manifest_list_reader(snapshot)
            .load()
            .await?;

        // Pick the named inputs out of the manifest list. Iterating the LIST rather
        // than the caller's paths is what makes group formation — and, on V3, the
        // order carried manifests are assigned row-id bases in — independent of the
        // order the caller happened to name them in.
        let input_manifests: HashSet<&str> = self
            .input_manifest_paths
            .iter()
            .map(String::as_str)
            .collect();
        let selected_manifests: Vec<ManifestFile> = manifest_list
            .entries()
            .iter()
            .filter(|manifest| input_manifests.contains(manifest.manifest_path.as_str()))
            .cloned()
            .collect();

        // Every requested input must be live in the current snapshot.
        if selected_manifests.len() != input_manifests.len() {
            let found: HashSet<&str> = selected_manifests
                .iter()
                .map(|m| m.manifest_path.as_str())
                .collect();
            let mut missing: Vec<&str> = input_manifests
                .iter()
                .copied()
                .filter(|path| !found.contains(path))
                .collect();
            missing.sort_unstable();
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot rewrite manifests that are not in the current snapshot: {}",
                    missing.join(", ")
                ),
            ));
        }

        // Only DATA manifests are eligible: delete manifests carry different
        // semantics and are never repacked. The rewrite writer only ever builds a
        // DATA manifest, so a delete manifest reaching it would silently re-emit
        // delete-file entries as data-file entries.
        for manifest in &selected_manifests {
            if manifest.content != ManifestContentType::Data {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot rewrite a non-data manifest: {}",
                        manifest.manifest_path
                    ),
                ));
            }
        }

        // A single output manifest carries exactly one partition spec, so every
        // input in a group must share one. Cross-spec groups are a caller/planner
        // error and are rejected rather than silently rewritten under one spec.
        // This is also where the spec is DERIVED: the writer uses the inputs' own
        // spec, never the table's current default, so a historical spec stays
        // repackable after the partition key evolves.
        let spec_id = selected_manifests[0].partition_spec_id;
        if selected_manifests
            .iter()
            .any(|m| m.partition_spec_id != spec_id)
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Cannot rewrite manifests that span multiple partition specs in one commit",
            ));
        }
        let partition_spec = metadata
            .partition_spec_by_id(spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Partition spec id {spec_id} referenced by an input manifest is not present in table metadata"
                    ),
                )
            })?
            .as_ref()
            .clone();

        let row_lineage_required = metadata.format_version() == FormatVersion::V3;
        if row_lineage_required && selected_manifests.iter().any(|m| m.first_row_id.is_none()) {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Cannot rewrite a V3 data manifest without an assigned first_row_id: \
                 repacking it would reassign row lineage",
            ));
        }

        // Read the bodies of the selected manifests.
        let mut groups: Vec<RewriteGroup> = Vec::new();
        for manifest_list_entry in &selected_manifests {
            let manifest_file = manifest_list_entry
                .load_manifest(snapshot_produce.table.file_io())
                .await?;
            // Schema the manifest was written under: it contains every source column
            // `partition_spec` references, even if the current schema no longer does.
            let schema = manifest_file.metadata().schema().clone();
            // Validate the spec's source columns exist in this schema (the historical
            // -spec case). The type itself is not kept: the packer and writer recompute
            // it, so only the error return matters here.
            partition_spec.partition_type(&schema)?;

            // Every live entry is re-emitted; deleted entries are dropped but still
            // count toward the row-lineage offset (handled inside the helper).
            let mut manifest_entries: Vec<ManifestEntry> = Vec::new();
            collect_manifests_entries(
                manifest_list_entry,
                manifest_file.entries(),
                row_lineage_required,
                |entry| entry.is_alive(),
                &mut manifest_entries,
            )?;
            let member = ScannedManifest {
                manifest: manifest_list_entry.clone(),
                entries: manifest_entries,
            };
            match groups
                .iter_mut()
                .find(|g| g.schema.schema_id() == schema.schema_id())
            {
                Some(group) => group.manifests.push(member),
                None => groups.push(RewriteGroup {
                    schema,
                    manifests: vec![member],
                }),
            }
        }

        // Plan the output manifests WITHOUT writing anything, group by group: within
        // each group cluster entries by partition value (so output manifests keep
        // narrow partition summaries) and pack them into size-bounded buckets. Each
        // bucket is exactly one output manifest written below.
        let mut input_manifests: Vec<ManifestFile> = Vec::new();
        let mut planned: Vec<PlannedManifest> = Vec::new();
        for mut group in groups {
            let inputs = group.manifests.len();
            let input_bytes = group.input_bytes();
            let live_entries: usize = group.manifests.iter().map(|m| m.entries.len()).sum();

            let estimated_outputs = if live_entries == 0 {
                0
            } else {
                usize::try_from(input_bytes.div_ceil(self.target_manifest_size_bytes))
                    .unwrap_or(usize::MAX)
            };
            if estimated_outputs >= inputs {
                continue;
            }

            // Pack entries into output manifests targeting `target_manifest_size_bytes`
            // of MANIFEST bytes. Manifest byte size is proportional to entry count, not
            // to referenced data-file size, so packing is driven by an estimate of
            // per-entry manifest bytes — the mean byte cost per entry OF THIS GROUP,
            // measured over the same manifests the gate above just weighed.
            let avg_entry_bytes = if live_entries > 0 {
                (input_bytes / live_entries as u64).max(1)
            } else {
                1
            };
            let mut entries: Vec<ManifestEntry> = Vec::with_capacity(live_entries);
            for manifest in &mut group.manifests {
                entries.append(&mut manifest.entries);
            }
            entries.sort_by(|a, b| {
                compare_partition_structs(a.data_file().partition(), b.data_file().partition())
            });
            let group_planned = PlannedManifest::pack(
                entries,
                group.schema,
                avg_entry_bytes,
                self.target_manifest_size_bytes,
            );
            // The packer works off a mean, so a group whose entries are unevenly sized
            // can still plan more buckets than the byte estimate promised. Re-check
            // what was actually planned, so every accepted group strictly reduces.
            if group_planned.len() >= inputs {
                continue;
            }

            input_manifests.extend(group.manifests.into_iter().map(|m| m.manifest));
            planned.extend(group_planned);
        }

        if input_manifests.is_empty() {
            return Err(Error::new(
                ErrorKind::NoReduction,
                "rewrite_manifests produced no reduction",
            ));
        }

        let input_paths: HashSet<&str> = input_manifests
            .iter()
            .map(|m| m.manifest_path.as_str())
            .collect();
        let mut carried_manifests: Vec<ManifestFile> = manifest_list
            .entries()
            .iter()
            .filter(|m| !input_paths.contains(m.manifest_path.as_str()))
            .cloned()
            .collect();

        // Materialize the planned output manifests.
        let mut rewritten: Vec<ManifestFile> = Vec::with_capacity(planned.len());
        for output_manifest in planned {
            let mut writer = snapshot_produce
                .new_rewrite_manifest_writer(output_manifest.schema, partition_spec.clone())?;
            // Smallest materialized `first_row_id` packed into this manifest becomes
            // its assigned base on V3, so the manifest-list writer keeps it as
            // already-assigned and leaves `next_row_id` untouched; every packed file
            // already carries its own materialized id, so the base is only a valid
            // lower bound. Stays `None` on V1/V2.
            let mut min_row_id: Option<u64> = None;
            for entry in &output_manifest.entries {
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
                    // On the V3 path `collect_manifests_entries` already rejected
                    // negatives; convert via `try_from` rather than `as u64` so no
                    // invalid value can silently wrap into a bogus manifest base on any
                    // path that reaches here.
                    let first_row_id = u64::try_from(first_row_id).map_err(|_| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Negative first_row_id {first_row_id} on a packed data file during manifest rewrite"
                            ),
                        )
                    })?;
                    min_row_id = Some(min_row_id.map_or(first_row_id, |min| min.min(first_row_id)));
                }
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
            rewritten.push(writer.write_manifest_file().await?);
        }

        carried_manifests.extend(rewritten);
        Ok(carried_manifests)
    }
}

struct ScannedManifest {
    manifest: ManifestFile,
    entries: Vec<ManifestEntry>,
}

struct RewriteGroup {
    schema: SchemaRef,
    /// Non-empty, in manifest-list order (inherited from the selection order).
    manifests: Vec<ScannedManifest>,
}

impl RewriteGroup {
    /// Manifest bytes this group would rewrite.
    fn input_bytes(&self) -> u64 {
        self.manifests
            .iter()
            .map(|m| manifest_bytes(&m.manifest))
            .sum()
    }
}

/// Manifest object size in bytes. `manifest_length` is a signed Avro field; a
/// negative value is meaningless, so it floors at zero rather than wrapping.
fn manifest_bytes(manifest: &ManifestFile) -> u64 {
    manifest.manifest_length.max(0) as u64
}

/// One planned output manifest: the entries packed into it and the schema to write
/// them under. Built without any I/O, so the no-reduction gate can compare the exact
/// output count against the input count before the first object is written.
struct PlannedManifest {
    schema: SchemaRef,
    entries: Vec<ManifestEntry>,
}

impl PlannedManifest {
    /// Pack `entries` into size-bounded output manifests, all written under `schema`.
    fn pack(
        mut entries: Vec<ManifestEntry>,
        schema: SchemaRef,
        avg_entry_bytes: u64,
        target_bytes: u64,
    ) -> Vec<Self> {
        let entries_per_manifest = (target_bytes.div_ceil(avg_entry_bytes.max(1)) as usize).max(1);
        let mut planned: Vec<Self> =
            Vec::with_capacity(entries.len().div_ceil(entries_per_manifest));
        while entries.len() > entries_per_manifest {
            let rest = entries.split_off(entries_per_manifest);
            planned.push(Self {
                schema: schema.clone(),
                entries,
            });
            entries = rest;
        }
        if !entries.is_empty() {
            planned.push(Self { schema, entries });
        }
        planned
    }
}

fn compare_partition_structs(a: &Struct, b: &Struct) -> Ordering {
    for (left, right) in a.iter().zip(b.iter()) {
        let ordering = compare_optional_literals(left, right);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

fn compare_optional_literals(left: Option<&Literal>, right: Option<&Literal>) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => {
            match (left.as_primitive_literal(), right.as_primitive_literal()) {
                (Some(left), Some(right)) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
                _ => Ordering::Equal,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, Operation,
        Struct, Transform, UnboundPartitionSpec,
    };
    use crate::table::Table;
    use crate::transaction::tests::{
        make_v2_minimal_table_in_catalog, make_v3_minimal_table_in_catalog,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction, TransactionAction};
    use crate::{Catalog, ErrorKind, TableCommit, TableRequirement, TableUpdate};

    /// A large target that keeps every seeded manifest in a single output.
    const LARGE_TARGET: u64 = 100 * 1024 * 1024;

    fn data_file(path: &str, rows: u64, spec_id: i32, partition: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(rows)
            .partition(Struct::from_iter([Some(Literal::long(partition))]))
            .partition_spec_id(spec_id)
            .build()
            .unwrap()
    }

    fn file(path: &str) -> DataFile {
        data_file(path, 1, 0, 0)
    }

    fn tx_for(table: &Table) -> Transaction {
        Transaction::new(table)
    }

    fn marker() -> HashMap<String, String> {
        HashMap::from([("icegate.manifest-rewrite".to_string(), "true".to_string())])
    }

    async fn append_file<C: Catalog>(catalog: &C, table: Table, path: &str) -> Table {
        let tx = Transaction::new(&table);
        tx.fast_append()
            .add_data_files(vec![file(path)])
            .apply(tx)
            .unwrap()
            .commit(catalog)
            .await
            .unwrap()
    }

    async fn manifest_paths(table: &Table) -> Vec<String> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
        manifest_list
            .entries()
            .iter()
            .map(|m| m.manifest_path.clone())
            .collect()
    }

    async fn live_file_paths(table: &Table) -> Vec<String> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
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

    async fn data_sequence_number_of(table: &Table, path: &str) -> Option<i64> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
        for entry in manifest_list.entries() {
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            for e in manifest.entries() {
                if e.file_path() == path {
                    return e.sequence_number();
                }
            }
        }
        None
    }

    async fn append_files(catalog: &impl Catalog, table: Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(&table);
        tx.fast_append()
            .add_data_files(files)
            .apply(tx)
            .unwrap()
            .commit(catalog)
            .await
            .unwrap()
    }

    /// Effective (inherited or materialized) `first_row_id` of every live data
    /// file, keyed by path. Applies the V3 inheritance rule against each manifest's
    /// stored order, so the result is comparable across a metadata-only rewrite that
    /// reorders and repacks the entries.
    async fn effective_first_row_ids(table: &Table) -> HashMap<String, i64> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
        let mut out = HashMap::new();
        for entry in manifest_list.entries() {
            let base = entry.first_row_id.map(|v| v as i64);
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            let mut offset: i64 = 0;
            for e in manifest.entries() {
                let effective = match e.data_file().first_row_id() {
                    Some(v) => Some(v),
                    None => {
                        let v = base.map(|b| b + offset);
                        offset += e.data_file().record_count() as i64;
                        v
                    }
                };
                if e.is_alive() {
                    out.insert(e.file_path().to_string(), effective.unwrap());
                }
            }
        }
        out
    }

    /// Whether every live data file carries a non-null on-disk `first_row_id`.
    async fn all_live_first_row_ids_materialized(table: &Table) -> bool {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
        for entry in manifest_list.entries() {
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            for e in manifest.entries() {
                if e.is_alive() && e.data_file().first_row_id().is_none() {
                    return false;
                }
            }
        }
        true
    }

    /// A V3 metadata-only rewrite preserves the file → `_row_id` mapping. Entries
    /// span three manifests with multi-row files and are repacked into one, in a
    /// different order (partitions force a re-sort), yet each file keeps its
    /// inherited `first_row_id`; the table's `next_row_id` is unchanged and the
    /// `replace` snapshot adds zero rows.
    #[tokio::test]
    async fn rewrite_manifests_preserves_v3_row_lineage() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // M1 base 0: a→0 (3 rows), b→3 (2 rows). Distinct partitions so the repack
        // reorders the files, exercising position-independent materialization.
        let table = append_files(&catalog, table, vec![
            data_file("a.parquet", 3, 0, 5),
            data_file("b.parquet", 2, 0, 1),
        ])
        .await;
        // M2 base 5: c→5 (4 rows). M3 base 9: d→9 (1 row).
        let table = append_files(&catalog, table, vec![data_file("c.parquet", 4, 0, 3)]).await;
        let table = append_files(&catalog, table, vec![data_file("d.parquet", 1, 0, 2)]).await;

        let before = effective_first_row_ids(&table).await;
        assert_eq!(before.get("a.parquet"), Some(&0));
        assert_eq!(before.get("b.parquet"), Some(&3));
        assert_eq!(before.get("c.parquet"), Some(&5));
        assert_eq!(before.get("d.parquet"), Some(&9));
        let next_row_id_before = table.metadata().next_row_id();
        assert_eq!(next_row_id_before, 10);

        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 3);

        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(inputs)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        assert_eq!(manifest_paths(&table).await.len(), 1);
        // The file → row-id mapping is unchanged, now materialized on disk.
        assert_eq!(effective_first_row_ids(&table).await, before);
        assert!(
            all_live_first_row_ids_materialized(&table).await,
            "every re-emitted file must carry a materialized first_row_id"
        );
        // No row identity is reassigned: next_row_id is untouched and the replace
        // assigned zero rows.
        assert_eq!(table.metadata().next_row_id(), next_row_id_before);
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .added_rows_count(),
            Some(0)
        );
    }

    /// Three single-manifest appends repack into one manifest. The live data-file
    /// set and each survivor's data sequence number are preserved, and the new
    /// snapshot is a `Replace` whose parent is the pre-rewrite current snapshot.
    #[tokio::test]
    async fn rewrite_manifests_repacks_group_preserving_files_and_sequence_numbers() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;
        let table = append_file(&catalog, table, "c.parquet").await;

        let before_live = live_file_paths(&table).await;
        assert_eq!(before_live, vec![
            "a.parquet".to_string(),
            "b.parquet".to_string(),
            "c.parquet".to_string(),
        ]);
        let seq_b_before = data_sequence_number_of(&table, "b.parquet").await;
        assert!(seq_b_before.is_some());
        let parent_id = table.metadata().current_snapshot_id();

        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 3);

        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(inputs)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        assert_eq!(manifest_paths(&table).await.len(), 1);
        assert_eq!(live_file_paths(&table).await, before_live);
        assert_eq!(
            data_sequence_number_of(&table, "b.parquet").await,
            seq_b_before,
            "survivor data sequence number must be preserved across the rewrite"
        );

        let snapshot = table.metadata().current_snapshot().unwrap();
        assert_eq!(snapshot.summary().operation, Operation::Replace);
        assert_eq!(snapshot.parent_snapshot_id(), parent_id);
    }

    /// A DATA manifest the caller did not name is carried forward byte-for-byte (its
    /// object path is unchanged), leaving the live data-file set intact.
    #[tokio::test]
    async fn rewrite_manifests_carries_forward_unselected_manifest() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;
        let table = append_file(&catalog, table, "c.parquet").await;

        let before = manifest_paths(&table).await;
        assert_eq!(before.len(), 3);
        let unnamed = manifest_holding(&table, "c.parquet").await;

        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests([
                manifest_holding(&table, "a.parquet").await.manifest_path,
                manifest_holding(&table, "b.parquet").await.manifest_path,
            ])
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        let after = manifest_paths(&table).await;
        assert_eq!(after.len(), 2, "one carried + one repacked");
        let carried: Vec<&String> = after.iter().filter(|p| before.contains(p)).collect();
        assert_eq!(
            carried,
            vec![&unnamed.manifest_path],
            "exactly the manifest left out of the input set must survive \
             byte-for-byte, got: {after:?}"
        );
        assert_eq!(live_file_paths(&table).await, vec![
            "a.parquet".to_string(),
            "b.parquet".to_string(),
            "c.parquet".to_string(),
        ]);
    }

    /// A named input set that splits across schemas commits the groups that reduce
    /// and carries the rest forward, instead of failing as a whole.
    ///
    /// The caller names all three manifests, having no way to see from the manifest
    /// list that one of them was written under a different schema and so cannot share
    /// an output with the others. That lone manifest forms a one-member group, which
    /// cannot reduce; failing the commit over it would make every later run repeat the
    /// same skip while two perfectly mergeable manifests sat in the same set. Instead
    /// the two compatible manifests merge and the odd one is carried forward
    /// byte-for-byte.
    ///
    /// A plain non-partition column addition is enough to produce this shape — no
    /// partition-spec evolution required.
    #[tokio::test]
    async fn rewrite_manifests_drops_a_non_reducing_group_and_commits_the_rest() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Two schema-0 manifests of three entries each — the bigger candidates.
        let table = append_files(&catalog, table, vec![
            file("a1.parquet"),
            file("a2.parquet"),
            file("a3.parquet"),
        ])
        .await;
        let table = append_files(&catalog, table, vec![
            file("b1.parquet"),
            file("b2.parquet"),
            file("b3.parquet"),
        ])
        .await;

        // Add a non-partition column and make it current: the partition spec (and so
        // the partition type) is unchanged, only the schema id advances.
        let table = evolve_schema(&catalog, table).await;

        // One schema-1 manifest of a single entry — the SMALLEST candidate.
        let table = append_files(&catalog, table, vec![file("c1.parquet")]).await;

        let schema_0_first = manifest_holding(&table, "a1.parquet").await;
        let schema_0_second = manifest_holding(&table, "b1.parquet").await;
        let schema_1_only = manifest_holding(&table, "c1.parquet").await;
        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 3);

        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(inputs)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        let after = manifest_paths(&table).await;
        assert_eq!(after.len(), 2, "one repacked schema-0 output + one carried");
        assert!(
            after.contains(&schema_1_only.manifest_path),
            "the lone schema-1 manifest cannot reduce on its own and must be carried \
             forward, not fail the commit"
        );
        assert!(
            !after.contains(&schema_0_first.manifest_path)
                && !after.contains(&schema_0_second.manifest_path),
            "both schema-0 manifests must have been repacked, got: {after:?}"
        );
        assert_eq!(live_file_paths(&table).await, vec![
            "a1.parquet".to_string(),
            "a2.parquet".to_string(),
            "a3.parquet".to_string(),
            "b1.parquet".to_string(),
            "b2.parquet".to_string(),
            "b3.parquet".to_string(),
            "c1.parquet".to_string(),
        ]);
    }

    /// A group's verdict is decided over its OWN inputs, so an unrelated group in the
    /// same scan window cannot change it.
    ///
    /// Two groups with very different per-entry costs: a FAT one (four entries per
    /// manifest) and a THIN one (one). The target is the thin group's own byte total,
    /// so by the byte budget the thin group reduces (`ceil(thin/target) == 1 < 2`) and
    /// the fat group does not (`ceil(fat/target) >= 2`, since `fat > thin`). The fat
    /// group is committed against twice — alone, and beside the thin group — and must
    /// be rejected identically both times, while the thin group merges regardless of
    /// the fat entries sitting next to it. A shared per-entry average taken across all
    /// the named inputs would blend the two and flip at least one of those verdicts.
    #[tokio::test]
    async fn rewrite_manifests_group_verdict_ignores_other_groups_entry_sizes() {
        use super::manifest_bytes;

        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Two FAT schema-0 manifests, four entries each.
        let table = append_files(
            &catalog,
            table,
            (0..4).map(|i| file(&format!("f{i}.parquet"))).collect(),
        )
        .await;
        let table = append_files(
            &catalog,
            table,
            (4..8).map(|i| file(&format!("f{i}.parquet"))).collect(),
        )
        .await;
        let fat_first = manifest_holding(&table, "f0.parquet").await;
        let fat_second = manifest_holding(&table, "f4.parquet").await;
        // The same table as it stands before the thin group exists.
        let without_thin = table.clone();

        // Two THIN schema-1 manifests, one entry each.
        let table = evolve_schema(&catalog, table).await;
        let table = append_files(&catalog, table, vec![file("t0.parquet")]).await;
        let table = append_files(&catalog, table, vec![file("t1.parquet")]).await;
        let thin_first = manifest_holding(&table, "t0.parquet").await;
        let thin_second = manifest_holding(&table, "t1.parquet").await;

        let thin_bytes = manifest_bytes(&thin_first) + manifest_bytes(&thin_second);
        let fat_bytes = manifest_bytes(&fat_first) + manifest_bytes(&fat_second);
        assert!(
            fat_bytes > thin_bytes,
            "the four-entry manifests must outweigh the one-entry ones, or the byte \
             gate does not separate the two groups"
        );
        let target = thin_bytes;

        // (1) The fat group alone: rejected by the reduction gate, nothing written.
        let err = Arc::new(
            tx_for(&without_thin)
                .rewrite_manifests()
                .add_input_manifests([
                    fat_first.manifest_path.clone(),
                    fat_second.manifest_path.clone(),
                ])
                .target_manifest_size_bytes(target)
                .set_snapshot_properties(marker()),
        )
        .commit(&without_thin)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::NoReduction,
            "unexpected error: {err}"
        );
        assert!(
            err.message().contains("produced no reduction"),
            "the fat group must be turned down by the reduction gate: {}",
            err.message()
        );

        // (2) Beside the thin group: same verdict for the fat group, and the thin
        // group merges — neither decision moved because of the other.
        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(manifest_paths(&table).await)
            .target_manifest_size_bytes(target)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        let after = manifest_paths(&table).await;
        assert_eq!(
            after.len(),
            3,
            "one merged thin output + both fat manifests"
        );
        assert!(
            after.contains(&fat_first.manifest_path) && after.contains(&fat_second.manifest_path),
            "the fat group must be carried forward exactly as when it was planned \
             alone, got: {after:?}"
        );
        assert!(
            !after.contains(&thin_first.manifest_path)
                && !after.contains(&thin_second.manifest_path),
            "the thin group must merge on its own byte budget, got: {after:?}"
        );
        let mut expected: Vec<String> = (0..8).map(|i| format!("f{i}.parquet")).collect();
        expected.extend(["t0.parquet".to_string(), "t1.parquet".to_string()]);
        assert_eq!(live_file_paths(&table).await, expected);
    }

    /// Two manifests that fit the target merge into a single output (M < N).
    #[tokio::test]
    async fn rewrite_manifests_merges_two_into_one() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;

        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 2);

        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(inputs)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        assert_eq!(manifest_paths(&table).await.len(), 1);
        assert_eq!(live_file_paths(&table).await, vec![
            "a.parquet".to_string(),
            "b.parquet".to_string(),
        ]);
    }

    /// One output manifest carries exactly one partition spec, so the writer's spec is
    /// DERIVED from the named inputs rather than taken from the table's default: naming
    /// the spec-1 pair compacts spec 1, naming the spec-0 pair compacts the historical
    /// spec 0, and naming one of each is rejected.
    #[tokio::test]
    async fn rewrite_manifests_derives_partition_spec_from_inputs() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;

        // Evolve to a second partition spec (identity on `y`) and make it default.
        let unbound = UnboundPartitionSpec::builder()
            .add_partition_field(2, "y", Transform::Identity)
            .unwrap()
            .build();
        let commit = TableCommit::builder()
            .ident(table.identifier().to_owned())
            .updates(vec![
                TableUpdate::AddSpec { spec: unbound },
                TableUpdate::SetDefaultSpec { spec_id: 1 },
            ])
            .requirements(vec![TableRequirement::UuidMatch {
                uuid: table.metadata().uuid(),
            }])
            .build();
        let table = catalog.update_table(commit).await.unwrap();

        // Two manifests under spec 1 alongside the two spec-0 manifests.
        let table = append_files(&catalog, table, vec![data_file("c.parquet", 1, 1, 7)]).await;
        let table = append_files(&catalog, table, vec![data_file("d.parquet", 1, 1, 8)]).await;

        let spec_manifests = |spec_id: i32| {
            let table = table.clone();
            async move {
                manifest_files(&table)
                    .await
                    .iter()
                    .filter(|m| m.partition_spec_id == spec_id)
                    .map(|m| m.manifest_path.clone())
                    .collect::<Vec<String>>()
            }
        };
        let spec_0_manifests = spec_manifests(0).await;
        let spec_1_manifests = spec_manifests(1).await;
        assert_eq!(spec_0_manifests.len(), 2);
        assert_eq!(spec_1_manifests.len(), 2);
        assert_eq!(manifest_paths(&table).await.len(), 4);

        // Naming the spec-1 pair compacts spec 1, 4 → 3 manifests.
        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(spec_1_manifests)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        let after = manifest_files(&table).await;
        assert_eq!(after.len(), 3, "the two spec-1 manifests merge into one");
        assert_eq!(
            after
                .iter()
                .filter(|m| spec_0_manifests.contains(&m.manifest_path))
                .count(),
            2,
            "manifests of another spec must be carried forward untouched"
        );

        // Naming the historical spec-0 pair compacts spec 0 — under ITS OWN spec, not
        // the table's current default — 3 → 2.
        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(spec_0_manifests)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        let after = manifest_files(&table).await;
        assert_eq!(after.len(), 2);
        assert_eq!(after.iter().filter(|m| m.partition_spec_id == 0).count(), 1);
        assert_eq!(live_file_paths(&table).await, vec![
            "a.parquet".to_string(),
            "b.parquet".to_string(),
            "c.parquet".to_string(),
            "d.parquet".to_string(),
        ]);

        // One manifest of each spec cannot share an output manifest: rejected outright
        // rather than silently rewritten under one of the two specs.
        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(manifest_paths(&table).await)
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "unexpected error: {err}"
        );
        assert!(
            err.message().contains("multiple partition specs"),
            "unexpected error: {}",
            err.message()
        );
    }

    /// The marker property is written into the new snapshot summary, and an
    /// inherited key present on the parent is carried forward — while an inherited
    /// key ABSENT from the parent adds nothing (the no-op inheritance branch).
    #[tokio::test]
    async fn rewrite_manifests_records_marker_and_inherits_summary_property() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;

        // Seed the parent snapshot with a WAL-offset-like property.
        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .set_snapshot_properties(HashMap::from([(
                "wal.offset".to_string(),
                "42".to_string(),
            )]))
            .add_data_files(vec![file("b.parquet")])
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 2);

        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(inputs)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .inherit_summary_property("wal.offset")
            // Absent from the parent: nothing to carry forward, so it must NOT
            // appear in the new summary (the no-op inheritance branch).
            .inherit_summary_property("missing.key")
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        let summary = table.metadata().current_snapshot().unwrap().summary();
        assert_eq!(
            summary
                .additional_properties
                .get("icegate.manifest-rewrite"),
            Some(&"true".to_string()),
            "marker property must be present in the new snapshot summary"
        );
        assert_eq!(
            summary.additional_properties.get("wal.offset"),
            Some(&"42".to_string()),
            "inherited property must be carried forward from the parent"
        );
        assert!(
            !summary.additional_properties.contains_key("missing.key"),
            "an inherited key absent from the parent must not be synthesized"
        );
    }

    /// Repacking a single manifest cannot reduce anything, so naming fewer than two
    /// inputs is a planner error caught before a transaction is ever opened.
    #[tokio::test]
    async fn rewrite_manifests_requires_two_inputs() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;

        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(manifest_paths(&table).await)
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "unexpected error: {err}"
        );
        assert!(
            err.message().contains("at least two input manifests"),
            "unexpected error: {}",
            err.message()
        );
    }

    /// The same manifest named twice is one manifest, not two: the count gate sees
    /// through the duplication and reports a planner error rather than opening a
    /// transaction that could only end in a skip.
    #[tokio::test]
    async fn rewrite_manifests_rejects_duplicate_input_paths() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;

        let duplicated = manifest_holding(&table, "a.parquet").await.manifest_path;
        let snapshot_before = table.metadata().current_snapshot_id();

        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests([duplicated.clone(), duplicated])
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "unexpected error: {err}"
        );
        assert!(
            err.message().contains("at least two input manifests"),
            "unexpected error: {}",
            err.message()
        );
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            reloaded.metadata().current_snapshot_id(),
            snapshot_before,
            "a rejected plan must not have committed anything"
        );
    }

    /// A named input that is not live in the current snapshot is a planner error: the
    /// caller chose it from a manifest list that no longer describes the table.
    #[tokio::test]
    async fn rewrite_manifests_rejects_missing_input() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;

        let ghost = format!(
            "{}/metadata/ghost-manifest.avro",
            table.metadata().location()
        );
        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests([
                    manifest_holding(&table, "a.parquet").await.manifest_path,
                    ghost.clone(),
                ])
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "unexpected error: {err}"
        );
        assert!(
            err.message().contains("not in the current snapshot") && err.message().contains(&ghost),
            "unexpected error: {}",
            err.message()
        );
    }

    /// A table with no snapshot at all cannot hold the manifests the caller named.
    #[tokio::test]
    async fn rewrite_manifests_rejects_missing_current_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(["ghost-a.avro".to_string(), "ghost-b.avro".to_string()])
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "unexpected error: {err}"
        );
        assert!(
            err.message().contains("no current snapshot"),
            "unexpected error: {}",
            err.message()
        );
    }

    /// An empty snapshot-property map is rejected: a marker is required.
    #[tokio::test]
    async fn rewrite_manifests_requires_marker_property() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;

        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(manifest_paths(&table).await),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert!(
            err.message().contains("marker"),
            "unexpected error: {}",
            err.message()
        );
    }

    /// A zero target manifest size is a caller/config error (a maintenance job
    /// misconfigured with `0`): packing toward a zero budget is meaningless. It is
    /// caught up front as `DataInvalid`, before any snapshot work.
    #[tokio::test]
    async fn rewrite_manifests_rejects_zero_target() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;

        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 2);

        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(inputs)
                .target_manifest_size_bytes(0)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "unexpected error: {err}"
        );
        assert!(
            err.message()
                .contains("positive target_manifest_size_bytes"),
            "unexpected error: {}",
            err.message()
        );
    }

    /// A V3 data manifest without an assigned `first_row_id` — a pre-lineage V2→V3
    /// carry-over — cannot be repacked: it has no base to inherit from, so packing
    /// its files alongside materialized ones would let a reader re-derive colliding
    /// `_row_id`s. Two V2 manifests are appended, then the table is upgraded to V3
    /// (which only flips the version and resets `next_row_id`, leaving the existing
    /// manifests' `first_row_id` null). `first_row_id` is recorded in the manifest
    /// LIST, so a caller can see it and must not name such a manifest; doing so is
    /// refused before anything is written and row identity is never reassigned.
    #[tokio::test]
    async fn rewrite_manifests_rejects_v3_manifest_without_first_row_id() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;

        // Upgrade V2 → V3: only bumps the format version, so the two V2-written
        // manifests keep a null first_row_id and are now read under row lineage.
        let tx = Transaction::new(&table);
        let table = tx
            .upgrade_table_version()
            .set_format_version(crate::spec::FormatVersion::V3)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
        assert_eq!(
            table.metadata().format_version(),
            crate::spec::FormatVersion::V3
        );

        let before = manifest_paths(&table).await;
        assert_eq!(before.len(), 2);

        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(before.clone())
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
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
        assert_eq!(
            manifest_paths(&table).await,
            before,
            "the pre-lineage manifests must be left exactly as they were"
        );
    }

    /// A V1/V2 table has no row lineage: entries are repacked verbatim (the
    /// `!row_lineage` branch of `collect_manifests_entries`), `set_first_row_id` is
    /// never called, and two manifests still merge into one with the live file set
    /// and each survivor's data sequence number preserved. This is the only
    /// production branch of the materialize boundary the V3-only suite never runs.
    #[tokio::test]
    async fn rewrite_manifests_v2_repacks_verbatim() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;

        let before_live = live_file_paths(&table).await;
        assert_eq!(before_live, vec![
            "a.parquet".to_string(),
            "b.parquet".to_string(),
        ]);
        let seq_b_before = data_sequence_number_of(&table, "b.parquet").await;
        assert!(seq_b_before.is_some());
        let parent_id = table.metadata().current_snapshot_id();

        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 2);

        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(inputs)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        assert_eq!(manifest_paths(&table).await.len(), 1);
        assert_eq!(live_file_paths(&table).await, before_live);
        assert_eq!(
            data_sequence_number_of(&table, "b.parquet").await,
            seq_b_before,
            "survivor data sequence number must be preserved across the V2 rewrite"
        );

        let snapshot = table.metadata().current_snapshot().unwrap();
        assert_eq!(snapshot.summary().operation, Operation::Replace);
        assert_eq!(snapshot.parent_snapshot_id(), parent_id);
    }

    /// A group under an OLDER partition spec whose source column was dropped from
    /// the current schema still repacks. The rewrite writes each output manifest
    /// under the schema its entries were written with (which still contains the
    /// source column), not the current schema — otherwise serializing the partition
    /// tuples would fail to resolve the missing source column. The merged manifest
    /// keeps the old spec id and the old partition tuples, and the table's default
    /// spec and current schema are left untouched.
    #[tokio::test]
    async fn rewrite_manifests_repacks_historical_spec_after_source_column_dropped() {
        use crate::spec::{NestedField, PrimitiveType, Schema, Type};

        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // Two spec-0 manifests, partitioned on `x` (source id 1).
        let table = append_files(&catalog, table, vec![data_file("a.parquet", 1, 0, 5)]).await;
        let table = append_files(&catalog, table, vec![data_file("b.parquet", 1, 0, 7)]).await;

        // Captured BEFORE the evolution below: these are the historical spec-0
        // manifests the rewrite must repack under their own spec.
        let spec_0_inputs = manifest_paths(&table).await;
        assert_eq!(spec_0_inputs.len(), 2);

        // Evolve: make identity(`y`) the default spec and drop `x` from the current
        // schema. Spec 0 is now historical and references a column absent from the
        // current schema — the metadata builder keeps such specs on purpose.
        let unbound = UnboundPartitionSpec::builder()
            .add_partition_field(2, "y", Transform::Identity)
            .unwrap()
            .build();
        let schema_without_x = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(3, "z", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        let commit = TableCommit::builder()
            .ident(table.identifier().to_owned())
            .updates(vec![
                TableUpdate::AddSpec { spec: unbound },
                TableUpdate::SetDefaultSpec { spec_id: 1 },
                TableUpdate::AddSchema {
                    schema: schema_without_x,
                },
                TableUpdate::SetCurrentSchema { schema_id: 1 },
            ])
            .requirements(vec![TableRequirement::UuidMatch {
                uuid: table.metadata().uuid(),
            }])
            .build();
        let table = catalog.update_table(commit).await.unwrap();
        assert!(
            table.metadata().current_schema().field_by_id(1).is_none(),
            "the spec-0 source column must be gone from the current schema"
        );
        let default_spec_before = table.metadata().default_partition_spec_id();
        let current_schema_before = table.metadata().current_schema_id();
        assert_eq!(default_spec_before, 1);

        // Repack the two historical spec-0 manifests.
        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(spec_0_inputs)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        // One merged output manifest, still under the historical spec 0.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
        assert_eq!(manifest_list.entries().len(), 1);
        assert_eq!(manifest_list.entries()[0].partition_spec_id, 0);

        // The old partition tuples survive the repack unchanged.
        let manifest = manifest_list.entries()[0]
            .load_manifest(table.file_io())
            .await
            .unwrap();
        let partitions: Vec<Struct> = manifest
            .entries()
            .iter()
            .filter(|e| e.is_alive())
            .map(|e| e.data_file().partition().clone())
            .collect();
        assert!(partitions.contains(&Struct::from_iter([Some(Literal::long(5))])));
        assert!(partitions.contains(&Struct::from_iter([Some(Literal::long(7))])));

        // Live files are preserved and the table's default spec / current schema are
        // untouched by the metadata-only rewrite.
        assert_eq!(live_file_paths(&table).await, vec![
            "a.parquet".to_string(),
            "b.parquet".to_string(),
        ]);
        assert_eq!(
            table.metadata().default_partition_spec_id(),
            default_spec_before
        );
        assert_eq!(table.metadata().current_schema_id(), current_schema_before);
    }

    /// Lower bounds of every live data file, keyed by path, decoded through each
    /// output manifest's own stored schema — so a field id absent from the schema an
    /// entry is re-emitted under is dropped exactly as a reader would drop it.
    async fn live_lower_bounds(table: &Table) -> HashMap<String, HashMap<i32, Datum>> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
        let mut out = HashMap::new();
        for entry in manifest_list.entries() {
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            for e in manifest.entries() {
                if e.is_alive() {
                    out.insert(
                        e.file_path().to_string(),
                        e.data_file().lower_bounds().clone(),
                    );
                }
            }
        }
        out
    }

    /// Manifests can share a partition spec yet be written under different schemas —
    /// e.g. after a non-partition column is added — while the partition type is
    /// unchanged. Such manifests must NOT be repacked under a single schema: an
    /// output manifest stores one schema, and a reader decodes every entry's column
    /// bounds against it, so a field id absent from that schema is silently dropped.
    /// Grouping by schema keeps each output decodable, preserving the lower/upper
    /// bounds (and other metrics) of every re-emitted file. Without it, the four
    /// inputs would collapse into one manifest under the first schema and the
    /// schema-1-only column's bounds would vanish.
    #[tokio::test]
    async fn rewrite_manifests_preserves_bounds_across_mixed_schemas() {
        // A data file carrying identical lower and upper bounds for the given fields.
        fn bounded(path: &str, bounds: Vec<(i32, Datum)>) -> DataFile {
            DataFileBuilder::default()
                .content(DataContentType::Data)
                .file_path(path.to_string())
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(100)
                .record_count(1)
                .partition(Struct::from_iter([Some(Literal::long(0))]))
                .partition_spec_id(0)
                .lower_bounds(HashMap::from_iter(bounds.clone()))
                .upper_bounds(HashMap::from_iter(bounds))
                .build()
                .unwrap()
        }

        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Two manifests under schema 0 (x, y, z), each carrying a bound for the
        // non-partition column y (field 2).
        let table = append_files(&catalog, table, vec![bounded("a.parquet", vec![(
            2,
            Datum::long(10),
        )])])
        .await;
        let table = append_files(&catalog, table, vec![bounded("b.parquet", vec![(
            2,
            Datum::long(20),
        )])])
        .await;

        // Add a non-partition column w (field 4) and make it current, keeping the
        // default partition spec (0). The partition type is unchanged, but the
        // schema id advances — the exact shape that fooled partition-type grouping.
        let table = evolve_schema(&catalog, table).await;
        assert_eq!(table.metadata().current_schema_id(), 1);

        // Two more manifests under schema 1, carrying bounds for both y (field 2)
        // and the new column w (field 4).
        let table = append_files(&catalog, table, vec![bounded("c.parquet", vec![
            (2, Datum::long(30)),
            (4, Datum::long(300)),
        ])])
        .await;
        let table = append_files(&catalog, table, vec![bounded("d.parquet", vec![
            (2, Datum::long(40)),
            (4, Datum::long(400)),
        ])])
        .await;

        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 4);

        let table = {
            let tx = Transaction::new(&table);
            tx.rewrite_manifests()
                .add_input_manifests(inputs)
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker())
                .apply(tx)
                .unwrap()
                .commit(&catalog)
                .await
                .unwrap()
        };

        // The two schemas cannot share an output manifest, so the repack yields two
        // (one per schema) — still a reduction from the four inputs.
        assert_eq!(manifest_paths(&table).await.len(), 2);
        assert_eq!(live_file_paths(&table).await, vec![
            "a.parquet".to_string(),
            "b.parquet".to_string(),
            "c.parquet".to_string(),
            "d.parquet".to_string(),
        ]);

        // Bounds survive the repack, decoded through each output's own schema.
        let bounds = live_lower_bounds(&table).await;
        // Column y is preserved for every file.
        assert_eq!(
            bounds.get("a.parquet").and_then(|b| b.get(&2)),
            Some(&Datum::long(10))
        );
        assert_eq!(
            bounds.get("b.parquet").and_then(|b| b.get(&2)),
            Some(&Datum::long(20))
        );
        assert_eq!(
            bounds.get("c.parquet").and_then(|b| b.get(&2)),
            Some(&Datum::long(30))
        );
        assert_eq!(
            bounds.get("d.parquet").and_then(|b| b.get(&2)),
            Some(&Datum::long(40))
        );
        // The schema-1-only column w survives — it would be dropped if c/d were
        // re-emitted under schema 0, which has no field 4.
        assert_eq!(
            bounds.get("c.parquet").and_then(|b| b.get(&4)),
            Some(&Datum::long(300))
        );
        assert_eq!(
            bounds.get("d.parquet").and_then(|b| b.get(&4)),
            Some(&Datum::long(400))
        );
    }

    /// The packer's own per-entry manifest-byte estimate over ALL of the table's
    /// manifests: total manifest bytes divided by live entries. Passing it as the
    /// target makes the packer close a bucket after every entry, so a rewrite that
    /// selects every manifest plans exactly as many outputs as it has entries.
    async fn mean_entry_bytes(table: &Table) -> u64 {
        let total: u64 = manifest_files(table)
            .await
            .iter()
            .map(|m| m.manifest_length.max(0) as u64)
            .sum();
        let entries = live_file_paths(table).await.len() as u64;
        assert!(entries > 0, "the table must hold at least one live entry");
        total / entries
    }

    /// Add [`schema_with_extra_column`] through `catalog` and make it current.
    async fn evolve_schema(catalog: &impl Catalog, table: Table) -> Table {
        let commit = TableCommit::builder()
            .ident(table.identifier().to_owned())
            .updates(vec![
                TableUpdate::AddSchema {
                    schema: schema_with_extra_column(),
                },
                TableUpdate::SetCurrentSchema { schema_id: 1 },
            ])
            .requirements(vec![TableRequirement::UuidMatch {
                uuid: table.metadata().uuid(),
            }])
            .build();
        catalog.update_table(commit).await.unwrap()
    }

    /// The seeded table's schema plus a non-partition column `w` (field 4), as schema
    /// id 1. Adding it advances the schema id while leaving the partition spec — and
    /// so the partition type — untouched: the shape in which two manifests share a
    /// spec yet still cannot share an output manifest.
    fn schema_with_extra_column() -> crate::spec::Schema {
        use crate::spec::{NestedField, PrimitiveType, Schema, Type};

        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(3, "z", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(4, "w", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap()
    }

    /// The manifest-list entry of the manifest holding `file_path`, so a test can
    /// name a manifest by its content instead of by its generated object path.
    async fn manifest_holding(table: &Table, file_path: &str) -> ManifestFile {
        for entry in manifest_files(table).await {
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            if manifest
                .entries()
                .iter()
                .any(|e| e.file_path() == file_path)
            {
                return entry;
            }
        }
        panic!("no manifest holds {file_path}");
    }

    /// Every manifest file in the current snapshot, cloned from the manifest list.
    async fn manifest_files(table: &Table) -> Vec<ManifestFile> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
        manifest_list.entries().to_vec()
    }

    /// The decoded `(lower, upper)` Long partition-summary bounds of a manifest's
    /// single partition field — the narrow range a manifest-list pruner reads.
    fn summary_bounds_long(manifest: &ManifestFile) -> (i64, i64) {
        use crate::spec::{Datum, PrimitiveLiteral, PrimitiveType};
        let summary = &manifest.partitions.as_ref().unwrap()[0];
        let decode = |bytes: &serde_bytes::ByteBuf| -> i64 {
            match Datum::try_from_bytes(bytes.as_ref(), PrimitiveType::Long)
                .unwrap()
                .literal()
            {
                PrimitiveLiteral::Long(v) => *v,
                other => panic!("expected a Long partition bound, got {other:?}"),
            }
        };
        (
            decode(summary.lower_bound.as_ref().unwrap()),
            decode(summary.upper_bound.as_ref().unwrap()),
        )
    }

    /// Run manifest-list-level pruning (the real [`ManifestEvaluator`]) over
    /// `manifests` and return the number and total manifest bytes that survive —
    /// i.e. the manifests a scan with this partition filter would still open.
    fn pruned(
        manifests: &[ManifestFile],
        filter: &crate::expr::BoundPredicate,
    ) -> (usize, u64) {
        use crate::expr::visitors::manifest_evaluator::ManifestEvaluator;
        let evaluator = ManifestEvaluator::builder(filter.clone()).build();
        let mut count = 0usize;
        let mut bytes = 0u64;
        for manifest in manifests {
            if evaluator.eval(manifest).unwrap() {
                count += 1;
                bytes += manifest.manifest_length.max(0) as u64;
            }
        }
        (count, bytes)
    }

    /// Multi-output packing by size, and the pruning acceptance metric. A single
    /// spec/schema group spanning many partitions is packed into several
    /// size-bounded output manifests (exercising the bucket-close path), and because
    /// entries are clustered by partition value first, each output keeps a narrow,
    /// contiguous partition summary. So manifest-list pruning stays effective: a
    /// point predicate still opens exactly one output manifest, and a bucket-aligned
    /// range predicate opens no more manifest bytes than before the rewrite.
    #[tokio::test]
    async fn rewrite_manifests_packs_by_size_and_preserves_pruning() {
        use crate::expr::{Bind, Reference};
        use crate::spec::{NestedField, PrimitiveType, Schema, Type};

        let catalog = new_memory_catalog().await;
        let mut table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Six single-file appends, each a distinct partition value of `x` (spec 0 is
        // identity(x)). Six manifests, one schema and one spec, so the rewrite forms a
        // single group that can only be split by size. The partition values are seeded
        // in a SCATTERED order, so the manifest-list order is non-monotonic and only
        // the packer's clustering sort produces contiguous buckets — without it, the
        // scattered order would pack overlapping, wide summaries and fail (b)/(c).
        for x in [3i64, 0, 4, 1, 5, 2] {
            table = append_files(&catalog, table, vec![data_file(
                &format!("f{x}.parquet"),
                1,
                0,
                x,
            )])
            .await;
        }

        assert_eq!(manifest_paths(&table).await.len(), 6);

        // Reproduce the packer's own size arithmetic so the target is chosen relative
        // to the real per-entry manifest cost: avg = total_input_bytes / entries. A
        // target of 2*avg closes a bucket every second entry, yielding three outputs
        // of two partitions each — {0,1}, {2,3}, {4,5}.
        let inputs = manifest_files(&table).await;
        let total_input_bytes: u64 = inputs.iter().map(|m| m.manifest_length.max(0) as u64).sum();
        let avg_entry_bytes = total_input_bytes / 6;
        assert!(
            avg_entry_bytes > 0,
            "seeded manifests must have non-zero length"
        );
        let target = 2 * avg_entry_bytes;

        // A partition-schema binding `x` (field id 1000) at accessor position 0, the
        // position the manifest evaluator indexes into a manifest's partition summary.
        let partition_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1000, "x", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );
        let point = Reference::new("x")
            .equal_to(Datum::long(3))
            .bind(partition_schema.clone(), true)
            .unwrap();
        let range = Reference::new("x")
            .greater_than_or_equal_to(Datum::long(4))
            .bind(partition_schema.clone(), true)
            .unwrap();

        // Before the rewrite: how the two predicates prune the six single-partition
        // inputs. Point `x == 3` hits one input; range `x >= 4` hits two ({4}, {5}).
        let (before_point_count, _) = pruned(&inputs, &point);
        let (before_range_count, before_range_bytes) = pruned(&inputs, &range);
        assert_eq!(before_point_count, 1, "one input holds x=3");
        assert_eq!(before_range_count, 2, "two inputs hold x>=4");

        // All six manifests are named, so only the packer decides the output count.
        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(inputs.iter().map(|m| m.manifest_path.clone()))
            .target_manifest_size_bytes(target)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        // (a) Three size-bounded outputs from one group — the bucket-close path ran —
        // and a strict reduction from the six inputs.
        let outputs = manifest_files(&table).await;
        assert_eq!(
            outputs.len(),
            3,
            "six entries packed two-per-bucket must yield three output manifests"
        );
        assert!(outputs.len() < inputs.len());
        assert_eq!(
            live_file_paths(&table).await,
            (0..6).map(|x| format!("f{x}.parquet")).collect::<Vec<_>>()
        );

        // (b) Clustering kept each output's partition summary narrow, contiguous, and
        // non-overlapping — strictly inside the full [0, 5] range.
        let mut ranges: Vec<(i64, i64)> = outputs.iter().map(summary_bounds_long).collect();
        ranges.sort();
        assert_eq!(
            ranges,
            vec![(0, 1), (2, 3), (4, 5)],
            "outputs must cluster contiguous partitions, not scatter them"
        );
        for (lower, upper) in &ranges {
            assert!(
                upper - lower < 5,
                "each output summary must be narrower than the full [0, 5] range"
            );
        }

        // (c) Pruning acceptance metric — it must not regress. The point predicate
        // still opens exactly one manifest (clustering left x=3 in a bucket whose
        // contiguous range covers only {2, 3}), and the bucket-aligned range predicate
        // opens no more manifests, and no more manifest bytes, than before the rewrite.
        let (after_point_count, _) = pruned(&outputs, &point);
        assert_eq!(
            after_point_count, 1,
            "clustered packing keeps a point query on a single output manifest"
        );
        let (after_range_count, after_range_bytes) = pruned(&outputs, &range);
        assert!(
            after_range_count <= before_range_count,
            "a bucket-aligned range predicate must not open more manifests after the rewrite"
        );
        assert!(
            after_range_bytes <= before_range_bytes,
            "a bucket-aligned range predicate must not read more manifest bytes after the \
             rewrite: {after_range_bytes} > {before_range_bytes}"
        );
    }

    /// A no-reduction plan writes nothing and commits nothing. The gate is evaluated
    /// before the first output manifest is written — and the snapshot producer
    /// flushes the manifest list only after the `existing_manifest` hook returns — so
    /// a rejected rewrite leaves the committed snapshot and its manifest set exactly
    /// as they were: no new snapshot, no manifest, no manifest list.
    #[tokio::test]
    async fn rewrite_manifests_no_reduction_leaves_table_unchanged() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_file(&catalog, table, "a.parquet").await;
        let table = append_file(&catalog, table, "b.parquet").await;

        let before_snapshot = table.metadata().current_snapshot_id();
        let mut before_manifests = manifest_paths(&table).await;
        before_manifests.sort();
        assert_eq!(before_manifests.len(), 2);

        // A target of one entry's worth of manifest bytes packs one output manifest
        // per entry, so the two named inputs cannot be reduced and the gate rejects
        // the plan before any object is written.
        let target = mean_entry_bytes(&table).await;
        let tx = Transaction::new(&table);
        let err = tx
            .rewrite_manifests()
            .add_input_manifests(before_manifests.clone())
            .target_manifest_size_bytes(target)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .err()
            .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::NoReduction,
            "a non-reducing rewrite must surface as a typed NoReduction, not: {err}"
        );

        // Reload from the catalog: no new snapshot was committed and the manifest set
        // is exactly the one from before the attempted rewrite.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(reloaded.metadata().current_snapshot_id(), before_snapshot);
        let mut after_manifests = manifest_paths(&reloaded).await;
        after_manifests.sort();
        assert_eq!(after_manifests, before_manifests);
    }

    /// A data manifest entry with a negative on-disk `first_row_id` is corrupt (row
    /// ids are non-negative, and the parser does not enforce it). The rewrite must
    /// reject it as `DataInvalid` at the shared materialization boundary — before any
    /// output manifest is written — rather than sign-cast `-1` into `u64::MAX` and
    /// use it as a manifest base.
    #[tokio::test]
    async fn rewrite_manifests_rejects_negative_first_row_id() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Seed two manifests; the first file carries a corrupt negative first_row_id.
        let mut corrupt = data_file("a.parquet", 1, 0, 0);
        corrupt.first_row_id = Some(-1);
        let table = append_files(&catalog, table, vec![corrupt]).await;
        let table = append_files(&catalog, table, vec![data_file("b.parquet", 1, 0, 0)]).await;
        assert_eq!(first_row_id_on_disk(&table, "a.parquet").await, Some(-1));

        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 2);

        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(inputs)
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "unexpected error: {err}"
        );
    }

    /// The raw on-disk `first_row_id` of a data file, without applying inheritance —
    /// so a corrupt negative value seeded for a test is observable.
    async fn first_row_id_on_disk(table: &Table, path: &str) -> Option<i64> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
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

    // ----------------------------------------------------------------------------
    // Storage decorator that counts every PERSISTED object, used to prove a
    // rejected no-reduction rewrite writes nothing at all — not even an orphan
    // manifest or manifest list that never makes it into a catalog commit.
    //
    // Only actual persistence is counted: `Storage::write` (single-shot) and a
    // `FileWrite::close` (streaming), the two points at which an object enters the
    // store. Handle creation (`new_output`, `writer`) is NOT counted, so merely
    // constructing the manifest-list writer before the gate does not register.
    //
    // It also counts manifest READS — the object GETs a bounded planner must not
    // let run away with a table's manifest count. `ManifestFile::load_manifest`
    // fetches the whole object through `Storage::read`, and the manifest list is
    // fetched the same way, so the two are told apart by the manifest list's
    // `snap-` file-name prefix (`SnapshotProducer::generate_manifest_list_file_path`).
    // ----------------------------------------------------------------------------

    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream::BoxStream;
    use serde::{Deserialize, Serialize};

    use crate::Result;
    use crate::io::{
        FileIO, FileIOBuilder, FileMetadata, FileRead, FileWrite, InputFile, MemoryStorage,
        OutputFile, Storage, StorageConfig, StorageFactory,
    };

    /// Whether an object path is a manifest rather than a manifest list.
    fn is_manifest_path(path: &str) -> bool {
        path.ends_with(".avro")
            && !path
                .rsplit('/')
                .next()
                .is_some_and(|name| name.starts_with("snap-"))
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    struct CountingStorage {
        // Shares the underlying store across clones (its data is `Arc`-backed).
        #[serde(skip)]
        inner: MemoryStorage,
        // Number of objects persisted through this storage.
        #[serde(skip)]
        writes: Arc<AtomicUsize>,
        // Number of MANIFEST objects fetched through this storage.
        #[serde(skip)]
        manifest_reads: Arc<AtomicUsize>,
    }

    #[async_trait]
    #[typetag::serde]
    impl Storage for CountingStorage {
        async fn exists(&self, path: &str) -> Result<bool> {
            self.inner.exists(path).await
        }

        async fn metadata(&self, path: &str) -> Result<FileMetadata> {
            self.inner.metadata(path).await
        }

        async fn read(&self, path: &str) -> Result<Bytes> {
            if is_manifest_path(path) {
                self.manifest_reads.fetch_add(1, Ordering::SeqCst);
            }
            self.inner.read(path).await
        }

        async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
            self.inner.reader(path).await
        }

        async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            self.inner.write(path, bs).await
        }

        async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
            Ok(Box::new(CountingFileWrite {
                inner: self.inner.writer(path).await?,
                writes: self.writes.clone(),
            }))
        }

        async fn delete(&self, path: &str) -> Result<()> {
            self.inner.delete(path).await
        }

        async fn delete_prefix(&self, path: &str) -> Result<()> {
            self.inner.delete_prefix(path).await
        }

        async fn delete_stream(&self, paths: BoxStream<'static, String>) -> Result<()> {
            self.inner.delete_stream(paths).await
        }

        fn new_input(&self, path: &str) -> Result<InputFile> {
            // Like `new_output` below: route through THIS storage so the fetch is
            // observed. Handing back the inner storage's handle would read past the
            // decorator and leave every count at zero.
            Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
        }

        fn new_output(&self, path: &str) -> Result<OutputFile> {
            // The output file must route writes back through THIS counting storage,
            // not the inner one, so persistence is observed. Cloning shares both the
            // store and the counter.
            Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
        }
    }

    struct CountingFileWrite {
        inner: Box<dyn FileWrite>,
        writes: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl FileWrite for CountingFileWrite {
        async fn write(&mut self, bs: Bytes) -> Result<()> {
            // Buffered; not yet persisted, so not counted here.
            self.inner.write(bs).await
        }

        async fn close(&mut self) -> Result<()> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            self.inner.close().await
        }
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    struct CountingStorageFactory {
        #[serde(skip)]
        inner: MemoryStorage,
        #[serde(skip)]
        writes: Arc<AtomicUsize>,
        #[serde(skip)]
        manifest_reads: Arc<AtomicUsize>,
    }

    #[typetag::serde]
    impl StorageFactory for CountingStorageFactory {
        fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
            Ok(Arc::new(CountingStorage {
                inner: self.inner.clone(),
                writes: self.writes.clone(),
                manifest_reads: self.manifest_reads.clone(),
            }))
        }
    }

    /// Build an empty V3 table backed by `file_io`, with no catalog behind it:
    /// every commit below is folded into the metadata directly, and every object
    /// the commit writes lands on `file_io`.
    fn make_v3_table(file_io: FileIO) -> Table {
        use std::fs::File;
        use std::io::BufReader;

        use crate::TableIdent;
        use crate::spec::TableMetadata;

        let f = File::open(format!(
            "{}/testdata/table_metadata/TableMetadataV3ValidMinimal.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let metadata: TableMetadata = serde_json::from_reader(BufReader::new(f)).unwrap();
        Table::builder()
            .metadata(metadata)
            .metadata_location("s3://bucket/test/location/metadata/v3.json".to_string())
            .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
            .file_io(file_io)
            .build()
            .unwrap()
    }

    /// Build a V3 table backed by `file_io` and seed it with one data manifest per
    /// given file, via one fast append each, folding every append's updates into the
    /// metadata directly (no catalog needed). The manifests are persisted on
    /// `file_io`.
    async fn make_v3_table_with_manifests(file_io: FileIO, files: &[&str]) -> Table {
        let mut table = make_v3_table(file_io);

        for path in files {
            let action = Arc::new(
                Transaction::new(&table)
                    .fast_append()
                    .add_data_files(vec![file(path)]),
            );
            let mut action_commit = action.commit(&table).await.unwrap();
            let updates = action_commit.take_updates();
            table = Transaction::update_table_metadata(table, &updates).unwrap();
        }
        table
    }

    /// A rejected no-reduction rewrite writes ZERO objects. The existing catalog
    /// test proves the committed snapshot is unchanged, but an implementation that
    /// leaked an orphan manifest or manifest list without committing would also pass
    /// it. Here the storage counts every persisted object, so any stray write fails
    /// the test even though nothing is ever committed.
    #[tokio::test]
    async fn rewrite_manifests_no_reduction_writes_no_objects() {
        let writes = Arc::new(AtomicUsize::new(0));
        let factory = CountingStorageFactory {
            inner: MemoryStorage::new(),
            writes: writes.clone(),
            manifest_reads: Arc::new(AtomicUsize::new(0)),
        };
        let file_io = FileIOBuilder::new(Arc::new(factory)).build();

        let table = make_v3_table_with_manifests(file_io, &["a.parquet", "b.parquet"]).await;
        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 2);
        // One entry's worth of manifest bytes: the packer closes a bucket per entry,
        // so the two named inputs plan two outputs and the gate rejects them.
        let target = mean_entry_bytes(&table).await;

        // Seeding persisted manifests and manifest lists: this both sets up the
        // table AND proves the counter is actually wired (so the `== 0` check below
        // is not vacuous).
        assert!(
            writes.load(Ordering::SeqCst) > 0,
            "the counting storage must observe the seeding writes"
        );
        // Discount the seeding writes; only the rewrite attempt is under test.
        writes.store(0, Ordering::SeqCst);

        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(inputs)
                .target_manifest_size_bytes(target)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::NoReduction,
            "unexpected error: {err}"
        );
        assert_eq!(
            writes.load(Ordering::SeqCst),
            0,
            "a rejected no-reduction rewrite must not persist any object (no orphan \
             manifest or manifest list)"
        );
    }

    /// The action's I/O is exactly the plan it was handed: it fetches the bodies of
    /// the NAMED inputs and no others, and writes one output manifest plus one
    /// manifest list. The manifests left unnamed are never opened — that is what makes
    /// the caller's cap on the input set a real bound on the work a commit does, and
    /// it is why planning can afford to run off manifest-list entries alone.
    #[tokio::test]
    async fn rewrite_manifests_reads_only_the_named_inputs() {
        let writes = Arc::new(AtomicUsize::new(0));
        let manifest_reads = Arc::new(AtomicUsize::new(0));
        let file_io = FileIOBuilder::new(Arc::new(CountingStorageFactory {
            inner: MemoryStorage::new(),
            writes: writes.clone(),
            manifest_reads: manifest_reads.clone(),
        }))
        .build();

        let files = [
            "a.parquet",
            "b.parquet",
            "c.parquet",
            "d.parquet",
            "e.parquet",
        ];
        let table = make_v3_table_with_manifests(file_io, &files).await;
        let before = manifest_paths(&table).await;
        assert_eq!(before.len(), 5);
        let named = vec![
            manifest_holding(&table, "a.parquet").await.manifest_path,
            manifest_holding(&table, "b.parquet").await.manifest_path,
        ];

        // Discount the seeding I/O; only the rewrite attempt is under test.
        writes.store(0, Ordering::SeqCst);
        manifest_reads.store(0, Ordering::SeqCst);

        let mut action_commit = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(named.clone())
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .unwrap();

        assert_eq!(
            manifest_reads.load(Ordering::SeqCst),
            named.len(),
            "the action must fetch the bodies of the named inputs and nothing else"
        );
        assert_eq!(
            writes.load(Ordering::SeqCst),
            2,
            "the commit must write exactly one output manifest and one manifest list"
        );

        let updates = action_commit.take_updates();
        let table = Transaction::update_table_metadata(table, &updates).unwrap();

        let after = manifest_paths(&table).await;
        assert_eq!(after.len(), 4, "two of five inputs replaced by one output");
        assert!(
            named.iter().all(|p| !after.contains(p)),
            "both named inputs must have been replaced, got: {after:?}"
        );
        assert_eq!(
            after.iter().filter(|p| before.contains(p)).count(),
            3,
            "every unnamed manifest must survive byte-for-byte, got: {after:?}"
        );
        assert_eq!(
            live_file_paths(&table).await,
            files.map(str::to_string).to_vec()
        );
    }

    // ----------------------------------------------------------------------------
    // Seeding manifest lists directly.
    //
    // No action in this crate writes a DATA manifest whose entries are all deleted
    // (`rewrite_files` drops such a manifest instead of re-emitting it) nor a
    // DELETE manifest — yet a table written by another engine carries both, and
    // compacting the first away while leaving the second alone is precisely what
    // metadata compaction must do. Both are therefore hand-written here and spliced
    // into a snapshot through the snapshot producer.
    // ----------------------------------------------------------------------------

    use crate::spec::{ManifestContentType, ManifestFile, ManifestWriter, ManifestWriterBuilder};
    use crate::transaction::snapshot::{
        DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
    };

    /// Snapshot operation whose manifest list is exactly the manifests it is given.
    struct SeedManifestsOperation {
        manifests: Vec<ManifestFile>,
    }

    impl SnapshotProduceOperation for SeedManifestsOperation {
        fn operation(&self) -> Operation {
            Operation::Replace
        }

        async fn delete_entries(
            &self,
            _snapshot_produce: &SnapshotProducer<'_>,
        ) -> Result<Vec<crate::spec::ManifestEntry>> {
            Ok(vec![])
        }

        async fn existing_manifest(
            &self,
            _snapshot_produce: &SnapshotProducer<'_>,
        ) -> Result<Vec<ManifestFile>> {
            Ok(self.manifests.clone())
        }
    }

    /// Commit a snapshot whose manifest list is the table's current one plus
    /// `extra`, folding the updates into the metadata (no catalog needed).
    async fn seed_snapshot(table: Table, extra: Vec<ManifestFile>) -> Table {
        let mut manifests = if table.metadata().current_snapshot().is_some() {
            manifest_files(&table).await
        } else {
            vec![]
        };
        manifests.extend(extra);

        let mut action_commit =
            SnapshotProducer::new(&table, uuid::Uuid::now_v7(), marker(), vec![])
                .commit(SeedManifestsOperation { manifests }, DefaultManifestProcess)
                .await
                .unwrap();
        let updates = action_commit.take_updates();
        Transaction::update_table_metadata(table, &updates).unwrap()
    }

    fn manifest_writer(table: &Table, name: &str, content: ManifestContentType) -> ManifestWriter {
        let path = format!("{}/metadata/{name}.avro", table.metadata().location());
        let builder = ManifestWriterBuilder::new(
            table.file_io().new_output(path).unwrap(),
            Some(table.metadata().current_snapshot_id().unwrap_or(1)),
            table.metadata().current_schema().clone(),
            table.metadata().default_partition_spec().as_ref().clone(),
        );
        match content {
            ManifestContentType::Data => builder.build_v3_data(),
            ManifestContentType::Deletes => builder.build_v3_deletes(),
        }
    }

    /// Stamp the sequence numbers a committed manifest would already carry. A
    /// freshly written manifest leaves them unassigned for the manifest-list writer
    /// to fill in, which it only does for manifests of the snapshot being written —
    /// and these are seeded as if an earlier snapshot had produced them.
    fn as_committed(mut manifest: ManifestFile) -> ManifestFile {
        manifest.sequence_number = 1;
        manifest.min_sequence_number = 1;
        manifest
    }

    /// A DATA manifest holding nothing but deleted entries — the residue a delete
    /// leaves behind. It carries an assigned `first_row_id`, so it is a legitimate
    /// V3 rewrite candidate.
    async fn deleted_only_data_manifest(table: &Table, name: &str, files: &[&str]) -> ManifestFile {
        let mut writer = manifest_writer(table, name, ManifestContentType::Data);
        for path in files {
            writer.add_delete_file(file(path), 1, Some(1)).unwrap();
        }
        writer.set_first_row_id(Some(0));
        as_committed(writer.write_manifest_file().await.unwrap())
    }

    /// A DELETE manifest with one live entry: never a rewrite candidate.
    async fn delete_manifest(table: &Table, name: &str) -> ManifestFile {
        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(format!("{name}-positions.parquet"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .build()
            .unwrap();
        let mut writer = manifest_writer(table, name, ManifestContentType::Deletes);
        writer
            .add_existing_file(
                delete_file,
                table.metadata().current_snapshot_id().unwrap_or(1),
                1,
                Some(1),
            )
            .unwrap();
        as_committed(writer.write_manifest_file().await.unwrap())
    }

    /// Two DATA manifests holding nothing but deleted entries are replaced by ZERO
    /// output manifests. That is a strict reduction (`0 < 2`) and it changes nothing
    /// about the data: the live file set is untouched and the DELETE manifest is
    /// carried forward. The commit writes exactly ONE object — the new manifest list
    /// — never an empty manifest.
    #[tokio::test]
    async fn rewrite_manifests_drops_data_manifests_with_no_live_entries() {
        let writes = Arc::new(AtomicUsize::new(0));
        let file_io = FileIOBuilder::new(Arc::new(CountingStorageFactory {
            inner: MemoryStorage::new(),
            writes: writes.clone(),
            manifest_reads: Arc::new(AtomicUsize::new(0)),
        }))
        .build();

        let table = make_v3_table(file_io);
        let first = deleted_only_data_manifest(&table, "deleted-1", &["x.parquet"]).await;
        let second = deleted_only_data_manifest(&table, "deleted-2", &["y.parquet"]).await;
        let deletes = delete_manifest(&table, "deletes-1").await;
        let table =
            seed_snapshot(table, vec![first.clone(), second.clone(), deletes.clone()]).await;

        assert_eq!(manifest_paths(&table).await.len(), 3);
        let before_live = live_file_paths(&table).await;

        // Only the seeded objects are counted out; the rewrite starts from zero.
        writes.store(0, Ordering::SeqCst);

        let mut action_commit = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                // The DELETE manifest is left out of the input set — naming it would be
                // rejected outright.
                .add_input_manifests([first.manifest_path.clone(), second.manifest_path.clone()])
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .unwrap();
        let updates = action_commit.take_updates();
        let table = Transaction::update_table_metadata(table, &updates).unwrap();

        assert_eq!(
            manifest_paths(&table).await,
            vec![deletes.manifest_path.clone()],
            "both deleted-only DATA manifests must vanish, the DELETE manifest must stay"
        );
        assert_eq!(
            live_file_paths(&table).await,
            before_live,
            "dropping manifests without live entries cannot change the live file set"
        );
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace
        );
        assert_eq!(
            writes.load(Ordering::SeqCst),
            1,
            "the commit must write only the manifest list, not an empty manifest"
        );
    }

    /// A live DATA manifest and a deleted-only one are both candidates: they merge
    /// into a single output holding just the live entry, while the DELETE manifest
    /// is carried forward byte-for-byte.
    #[tokio::test]
    async fn rewrite_manifests_merges_live_and_deleted_only_data_manifests() {
        let file_io = FileIOBuilder::new(Arc::new(CountingStorageFactory::default())).build();

        let table = make_v3_table(file_io);
        // One live DATA manifest via a fast append.
        let mut action_commit = Arc::new(
            Transaction::new(&table)
                .fast_append()
                .add_data_files(vec![file("a.parquet")]),
        )
        .commit(&table)
        .await
        .unwrap();
        let updates = action_commit.take_updates();
        let table = Transaction::update_table_metadata(table, &updates).unwrap();

        let deleted_only = deleted_only_data_manifest(&table, "deleted-1", &["x.parquet"]).await;
        let deletes = delete_manifest(&table, "deletes-1").await;
        let table = seed_snapshot(table, vec![deleted_only.clone(), deletes.clone()]).await;
        assert_eq!(manifest_paths(&table).await.len(), 3);
        let before_live = live_file_paths(&table).await;

        let mut action_commit = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests([
                    manifest_holding(&table, "a.parquet").await.manifest_path,
                    deleted_only.manifest_path.clone(),
                ])
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .unwrap();
        let updates = action_commit.take_updates();
        let table = Transaction::update_table_metadata(table, &updates).unwrap();

        let after = manifest_paths(&table).await;
        assert_eq!(after.len(), 2, "two DATA manifests merged into one");
        assert!(
            after.contains(&deletes.manifest_path),
            "the DELETE manifest must be carried forward byte-for-byte"
        );
        assert_eq!(live_file_paths(&table).await, before_live);
    }

    /// Naming a DELETE manifest is refused outright, before anything is written.
    ///
    /// The rewrite writer only ever builds a DATA manifest, so a delete manifest that
    /// reached it would have its delete-file entries silently re-emitted as data-file
    /// entries — a corrupt table. Content type is recorded in the manifest LIST, so
    /// a caller can see it and this is squarely a planner error.
    #[tokio::test]
    async fn rewrite_manifests_rejects_delete_manifest_input() {
        let writes = Arc::new(AtomicUsize::new(0));
        let file_io = FileIOBuilder::new(Arc::new(CountingStorageFactory {
            inner: MemoryStorage::new(),
            writes: writes.clone(),
            manifest_reads: Arc::new(AtomicUsize::new(0)),
        }))
        .build();

        let table = make_v3_table(file_io);
        let mut action_commit = Arc::new(
            Transaction::new(&table)
                .fast_append()
                .add_data_files(vec![file("a.parquet")]),
        )
        .commit(&table)
        .await
        .unwrap();
        let updates = action_commit.take_updates();
        let table = Transaction::update_table_metadata(table, &updates).unwrap();

        let deletes = delete_manifest(&table, "deletes-1").await;
        let table = seed_snapshot(table, vec![deletes.clone()]).await;
        assert_eq!(manifest_paths(&table).await.len(), 2);

        writes.store(0, Ordering::SeqCst);

        let err = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests([
                    manifest_holding(&table, "a.parquet").await.manifest_path,
                    deletes.manifest_path.clone(),
                ])
                .target_manifest_size_bytes(LARGE_TARGET)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .err()
        .unwrap();
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "unexpected error: {err}"
        );
        assert!(
            err.message().contains("non-data manifest")
                && err.message().contains(&deletes.manifest_path),
            "unexpected error: {}",
            err.message()
        );
        assert_eq!(
            writes.load(Ordering::SeqCst),
            0,
            "a rejected input set must not persist any object"
        );
    }

    /// A deleted-only manifest's bytes must not be charged against another group.
    ///
    /// It is the worst shape for a shared per-entry average: its manifest bytes count,
    /// but the live entries behind them do not exist, so blending it into a mean taken
    /// across all the named inputs inflates the estimated cost of every entry and can
    /// turn a perfectly good merge into a permanent skip. Here it also sits in a group
    /// of its own (written under a later schema), so it can only affect the live group
    /// through such a shared statistic. The live pair is committed against twice —
    /// without and with the deleted-only manifest named — and must merge both times.
    #[tokio::test]
    async fn rewrite_manifests_group_verdict_ignores_deleted_only_manifest_bytes() {
        use super::manifest_bytes;

        let file_io = FileIOBuilder::new(Arc::new(CountingStorageFactory::default())).build();
        let table = make_v3_table_with_manifests(file_io, &["a.parquet", "b.parquet"]).await;
        let live_first = manifest_holding(&table, "a.parquet").await;
        let live_second = manifest_holding(&table, "b.parquet").await;
        // The live group's own byte total: ceil(target / target) = 1 < 2 inputs.
        let target = manifest_bytes(&live_first) + manifest_bytes(&live_second);
        let without_deleted = table.clone();

        // Advance the schema so the seeded manifest forms a group of its own, then
        // splice in a DATA manifest that is nothing but deleted entries.
        let table = Transaction::update_table_metadata(table, &[
            TableUpdate::AddSchema {
                schema: schema_with_extra_column(),
            },
            TableUpdate::SetCurrentSchema { schema_id: 1 },
        ])
        .unwrap();
        // Enough deleted entries that the manifest outweighs the live group several
        // times over. A count that merely edges past it makes the test flaky: every
        // entry stores a randomly generated `snapshot_id`, which Avro encodes as a
        // variable-length varint, so a manifest's size drifts by tens of bytes between
        // runs.
        let deleted_paths: Vec<String> = (0..200).map(|i| format!("x{i}.parquet")).collect();
        let deleted_only = deleted_only_data_manifest(
            &table,
            "deleted-1",
            &deleted_paths.iter().map(String::as_str).collect::<Vec<_>>(),
        )
        .await;
        assert!(
            manifest_bytes(&deleted_only) > target,
            "the deleted-only manifest must outweigh the whole live group: below that \
             its bytes cannot drag a per-entry mean taken across all inputs far enough \
             to change the live group's verdict, and the test would pass vacuously \
             (deleted-only {} bytes vs live group {target} bytes)",
            manifest_bytes(&deleted_only)
        );
        let table = seed_snapshot(table, vec![deleted_only.clone()]).await;
        assert_eq!(manifest_paths(&table).await.len(), 3);

        // (1) Naming only the live pair: they merge.
        let live_pair = [
            live_first.manifest_path.clone(),
            live_second.manifest_path.clone(),
        ];
        let mut action_commit = Arc::new(
            tx_for(&without_deleted)
                .rewrite_manifests()
                .add_input_manifests(live_pair.clone())
                .target_manifest_size_bytes(target)
                .set_snapshot_properties(marker()),
        )
        .commit(&without_deleted)
        .await
        .unwrap();
        let updates = action_commit.take_updates();
        let merged = Transaction::update_table_metadata(without_deleted, &updates).unwrap();
        assert_eq!(manifest_paths(&merged).await.len(), 1);

        // (2) Naming the deleted-only manifest alongside them, they still merge — and
        // the deleted-only manifest, whose group writes zero outputs, drops out on its
        // own account.
        let mut action_commit = Arc::new(
            tx_for(&table)
                .rewrite_manifests()
                .add_input_manifests(
                    live_pair
                        .into_iter()
                        .chain([deleted_only.manifest_path.clone()]),
                )
                .target_manifest_size_bytes(target)
                .set_snapshot_properties(marker()),
        )
        .commit(&table)
        .await
        .unwrap();
        let updates = action_commit.take_updates();
        let table = Transaction::update_table_metadata(table, &updates).unwrap();

        let after = manifest_paths(&table).await;
        assert_eq!(
            after.len(),
            1,
            "the live pair must merge exactly as it did alone, got: {after:?}"
        );
        assert!(
            !after.contains(&deleted_only.manifest_path),
            "a manifest with no live entries must not survive the rewrite"
        );
        assert_eq!(live_file_paths(&table).await, vec![
            "a.parquet".to_string(),
            "b.parquet".to_string(),
        ]);
    }

    // ----------------------------------------------------------------------------
    // Optimistic-concurrency retry.
    //
    // The plan a rewrite commits must be the plan of the snapshot it commits
    // AGAINST. A catalog decorator injects one interfering commit in the window
    // between the transaction reading the table and its `update_table` landing —
    // the exact race the `RefSnapshotIdMatch` requirement exists to catch — so the
    // first attempt always loses the CAS and the retry has to re-plan.
    // ----------------------------------------------------------------------------

    use tokio::sync::Mutex as AsyncMutex;

    use crate::{Namespace, NamespaceIdent, TableCreation, TableIdent};

    /// What a concurrent writer does to the table mid-commit.
    #[derive(Debug)]
    enum Interference {
        /// Appends another data file, adding a manifest the original plan never saw.
        Append {
            path: &'static str,
            wal_offset: &'static str,
        },
        /// Runs its own manifest rewrite over `inputs`, REPLACING manifests the
        /// original plan had named.
        RewriteManifests { inputs: Vec<String> },
    }

    #[derive(Debug)]
    struct RacingCatalog<C: Catalog> {
        inner: C,
        // Taken by the first `update_table`; later calls pass straight through, so
        // exactly one conflict is injected and the retry can converge.
        pending: AsyncMutex<Option<Interference>>,
        // Commit attempts that reached the inner catalog, so a test can prove the
        // race really happened instead of passing on a single lucky attempt.
        attempts: Arc<AtomicUsize>,
    }

    impl<C: Catalog> RacingCatalog<C> {
        fn new(inner: C, interference: Interference) -> Self {
            Self {
                inner,
                pending: AsyncMutex::new(Some(interference)),
                attempts: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl<C: Catalog> Catalog for RacingCatalog<C> {
        async fn list_namespaces(
            &self,
            parent: Option<&NamespaceIdent>,
        ) -> Result<Vec<NamespaceIdent>> {
            self.inner.list_namespaces(parent).await
        }

        async fn create_namespace(
            &self,
            namespace: &NamespaceIdent,
            properties: HashMap<String, String>,
        ) -> Result<Namespace> {
            self.inner.create_namespace(namespace, properties).await
        }

        async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<Namespace> {
            self.inner.get_namespace(namespace).await
        }

        async fn namespace_exists(&self, namespace: &NamespaceIdent) -> Result<bool> {
            self.inner.namespace_exists(namespace).await
        }

        async fn update_namespace(
            &self,
            namespace: &NamespaceIdent,
            properties: HashMap<String, String>,
        ) -> Result<()> {
            self.inner.update_namespace(namespace, properties).await
        }

        async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
            self.inner.drop_namespace(namespace).await
        }

        async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
            self.inner.list_tables(namespace).await
        }

        async fn create_table(
            &self,
            namespace: &NamespaceIdent,
            creation: TableCreation,
        ) -> Result<Table> {
            self.inner.create_table(namespace, creation).await
        }

        async fn load_table(&self, table: &TableIdent) -> Result<Table> {
            self.inner.load_table(table).await
        }

        async fn drop_table(&self, table: &TableIdent) -> Result<()> {
            self.inner.drop_table(table).await
        }

        async fn purge_table(&self, table: &TableIdent) -> Result<()> {
            self.inner.purge_table(table).await
        }

        async fn table_exists(&self, table: &TableIdent) -> Result<bool> {
            self.inner.table_exists(table).await
        }

        async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
            self.inner.rename_table(src, dest).await
        }

        async fn register_table(
            &self,
            table: &TableIdent,
            metadata_location: String,
        ) -> Result<Table> {
            self.inner.register_table(table, metadata_location).await
        }

        async fn update_table(&self, commit: TableCommit) -> Result<Table> {
            if let Some(interference) = self.pending.lock().await.take() {
                let table = self.inner.load_table(commit.identifier()).await?;
                let tx = Transaction::new(&table);
                let tx = match interference {
                    Interference::Append { path, wal_offset } => tx
                        .fast_append()
                        .set_snapshot_properties(HashMap::from([(
                            "wal.offset".to_string(),
                            wal_offset.to_string(),
                        )]))
                        .add_data_files(vec![file(path)])
                        .apply(tx)
                        .unwrap(),
                    Interference::RewriteManifests { inputs } => tx
                        .rewrite_manifests()
                        .add_input_manifests(inputs)
                        .target_manifest_size_bytes(LARGE_TARGET)
                        .set_snapshot_properties(marker())
                        .apply(tx)
                        .unwrap(),
                };
                tx.commit(&self.inner).await?;
            }
            self.attempts.fetch_add(1, Ordering::SeqCst);
            self.inner.update_table(commit).await
        }
    }

    /// A concurrent APPEND between planning and committing. `fast_append` carries
    /// every existing manifest forward untouched, so the named inputs are all still
    /// live on the retry: the first attempt loses the CAS, the retry re-runs cleanly
    /// against the refreshed base, and the WAL offset is inherited from THAT parent —
    /// not from the one the first attempt saw. The manifest the racing snapshot added
    /// was never named, so it is carried forward.
    #[tokio::test]
    async fn rewrite_manifests_replans_after_concurrent_append() {
        let inner = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&inner).await;
        let table = append_file(&inner, table, "a.parquet").await;
        let table = append_file(&inner, table, "b.parquet").await;
        // The parent the first attempt sees carries wal.offset=42.
        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .set_snapshot_properties(HashMap::from([(
                "wal.offset".to_string(),
                "42".to_string(),
            )]))
            .add_data_files(vec![file("c.parquet")])
            .apply(tx)
            .unwrap()
            .commit(&inner)
            .await
            .unwrap();
        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 3);

        let catalog = RacingCatalog::new(inner, Interference::Append {
            path: "d.parquet",
            wal_offset: "99",
        });
        let attempts = catalog.attempts.clone();

        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_manifests()
            .add_input_manifests(inputs)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .inherit_summary_property("wal.offset")
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "the first attempt must lose the CAS and exactly one retry must follow"
        );
        assert_eq!(
            manifest_paths(&table).await.len(),
            2,
            "the three named inputs merge into one; the racing snapshot's manifest was \
             never named and is carried forward"
        );
        assert_eq!(live_file_paths(&table).await, vec![
            "a.parquet".to_string(),
            "b.parquet".to_string(),
            "c.parquet".to_string(),
            "d.parquet".to_string(),
        ]);
        let summary = table.metadata().current_snapshot().unwrap().summary();
        assert_eq!(
            summary.additional_properties.get("wal.offset"),
            Some(&"99".to_string()),
            "the inherited property must come from the FRESH parent snapshot"
        );
    }

    /// A concurrent MANIFEST REWRITE between planning and committing destroys a
    /// manifest the caller named, and the retry FAILS LOUDLY rather than quietly
    /// repacking something else.
    ///
    /// Maintenance serializes manifest rewrites per table, so this race is excluded by
    /// construction — but if that invariant is ever broken, the plan the caller
    /// authorized no longer describes the table, and silently substituting a different
    /// one would hide the breakage. The caller learns its plan is stale and re-plans;
    /// nothing is committed in the meantime.
    #[tokio::test]
    async fn rewrite_manifests_rejects_input_destroyed_by_a_concurrent_rewrite() {
        let inner = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&inner).await;
        let table = append_file(&inner, table, "a.parquet").await;
        let table = append_file(&inner, table, "b.parquet").await;
        let table = append_file(&inner, table, "c.parquet").await;
        let table = append_file(&inner, table, "d.parquet").await;
        let before_live = live_file_paths(&table).await;
        let before_snapshot = table.metadata().current_snapshot_id();
        let inputs = manifest_paths(&table).await;
        assert_eq!(inputs.len(), 4);

        // The racing job merges two of the four manifests this plan names, so those
        // two paths are gone by the time the retry re-runs.
        let racing_inputs: Vec<String> = inputs.iter().take(2).cloned().collect();
        let catalog = RacingCatalog::new(inner, Interference::RewriteManifests {
            inputs: racing_inputs.clone(),
        });
        let attempts = catalog.attempts.clone();

        let tx = Transaction::new(&table);
        let err = tx
            .rewrite_manifests()
            .add_input_manifests(inputs)
            .target_manifest_size_bytes(LARGE_TARGET)
            .set_snapshot_properties(marker())
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .err()
            .unwrap();

        // Exactly one commit reached the catalog: the first attempt, which lost the
        // CAS. The retry never got that far — it re-planned against the refreshed base,
        // found its named inputs gone, and refused before touching the catalog.
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "the retry must fail on the stale plan, before reaching the catalog"
        );
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "unexpected error: {err}"
        );
        assert!(
            racing_inputs
                .iter()
                .all(|path| err.message().contains(path)),
            "the error must name the inputs the racing rewrite destroyed: {}",
            err.message()
        );

        // The racing rewrite's own commit stands; ours contributed nothing.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_ne!(reloaded.metadata().current_snapshot_id(), before_snapshot);
        assert_eq!(
            live_file_paths(&reloaded).await,
            before_live,
            "neither commit may change the live data-file set"
        );
    }

    /// `compare_optional_literals` orders present-vs-null and primitive values, and
    /// treats two nulls as equal. It backs the partition clustering sort; its
    /// null branches are never hit by the end-to-end tests (every seeded partition
    /// value is `Some(long)`), so they are pinned directly here.
    #[test]
    fn compare_optional_literals_orders_nulls_and_values() {
        use std::cmp::Ordering;

        use super::compare_optional_literals;

        let lit = Literal::long(0);
        let other = Literal::long(1);
        assert_eq!(compare_optional_literals(None, None), Ordering::Equal);
        assert_eq!(compare_optional_literals(None, Some(&lit)), Ordering::Less);
        assert_eq!(
            compare_optional_literals(Some(&lit), None),
            Ordering::Greater
        );
        assert_eq!(
            compare_optional_literals(Some(&lit), Some(&other)),
            Ordering::Less
        );
        assert_eq!(
            compare_optional_literals(Some(&other), Some(&other)),
            Ordering::Equal
        );
        assert_eq!(
            compare_optional_literals(Some(&other), Some(&lit)),
            Ordering::Greater
        );
    }

    /// `compare_partition_structs` composes the per-field comparison
    /// lexicographically: the first differing field decides, ties break on the
    /// next, and a shared null compares equal.
    #[test]
    fn compare_partition_structs_is_lexicographic() {
        use std::cmp::Ordering;

        use super::compare_partition_structs;

        let s = |a: Option<i64>, b: Option<i64>| {
            Struct::from_iter([a.map(Literal::long), b.map(Literal::long)])
        };

        // The first field decides, regardless of the second.
        assert_eq!(
            compare_partition_structs(&s(Some(1), Some(9)), &s(Some(2), Some(0))),
            Ordering::Less
        );
        // A tie on the first field breaks on the second.
        assert_eq!(
            compare_partition_structs(&s(Some(1), Some(0)), &s(Some(1), Some(5))),
            Ordering::Less
        );
        // Null sorts before any value in the first field.
        assert_eq!(
            compare_partition_structs(&s(None, Some(0)), &s(Some(0), Some(0))),
            Ordering::Less
        );
        // All fields equal (including a shared null) compares equal.
        assert_eq!(
            compare_partition_structs(&s(None, Some(7)), &s(None, Some(7))),
            Ordering::Equal
        );
    }
}
