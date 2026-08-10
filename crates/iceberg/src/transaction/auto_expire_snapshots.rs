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

//! Automatic snapshot expiration, carried by a commit.
//!
//! Expiration is not a job of its own: every commit resolves the table's retention
//! policy from its properties and, when one is configured, rides a `RemoveSnapshots`
//! update along with whatever the commit was already going to write. A table that
//! does not configure a policy is never touched.
//!
//! This module powers the automatic, commit-riding expiration this fork adds. It
//! is deliberately independent from the upstream `ExpireSnapshotsAction`
//! (`expire_snapshots.rs`), which stays untouched as the explicit,
//! operator-driven maintenance API: both read the same `history.expire.*`
//! window properties, but they share no code, so upstream merges never
//! conflict here.
//!
//! [`expiration_updates`] is the whole of it from the outside: hand it the table a
//! commit is about to write and it answers with the updates that commit must carry.
//! It reads the clock and hands the rest to [`expiration_updates_at`], behind which
//! everything the answer depends on — the retention window, the ancestor chain,
//! statistics keyed by an expiring snapshot — stays.
//!
//! Expiration rides on commits and only on commits, so a table nothing writes to does
//! not converge to its retention window at all, and neither does a path that reaches
//! the table without opening a [`Transaction`](crate::transaction::Transaction).
//! A history already past the window converges only as fast as commits
//! arrive, at most [`MAX_EXPIRED_SNAPSHOTS_PER_COMMIT`] snapshots per commit. An empty
//! transaction is no way around that: [`Transaction::commit`](crate::transaction::Transaction::commit)
//! returns before expiration is ever reached when the transaction carries no action.
//!
//! TODO: a one-off catch-up, driven by the operator rather than by writes, running the
//! plan in batches until a neglected history is back inside the window.
//!
//! TODO: a reference that appears between the plan and the catalog applying it is not
//! guarded against. Every reference the metadata holds has its target protected — see
//! [`AutoExpirationPlan::collect_protected`] — but the metadata is the one the plan
//! was computed against, and the `RemoveSnapshots` it produces carries no
//! [`TableRequirement`](crate::TableRequirement) of its own. What the commit does carry
//! comes from its actions, and only the snapshot-writing ones contribute a
//! `RefSnapshotIdMatch`, for the branch they write; a commit made of property updates
//! alone contributes nothing at all. So a tag pointed at an expiring snapshot after the
//! plan was computed is dropped along with that snapshot by
//! [`TableMetadataBuilder::remove_snapshots`](crate::spec::TableMetadataBuilder::remove_snapshots),
//! silently and without any requirement failing. Pinning every reference present at
//! plan time would close the half where an existing reference is rolled back onto a
//! snapshot the plan expires. The half where the reference is created after the plan
//! stays open either way: `RefSnapshotIdMatch` names one reference at a time, and that
//! name does not exist yet when the requirements are built.
//!
//! The plan is computed against the local projection of the commit in progress rather
//! than against the metadata the catalog handed out, so the snapshot that was current
//! before this commit is in it too, and past the window it is a candidate like any
//! other. A catalog that rejects `remove-snapshots` outright is therefore left alone
//! only while the window still covers that snapshot: `history.expire.min-snapshots-to-keep`
//! of 2 or more, or a previous snapshot younger than `history.expire.max-snapshot-age-ms`.
//! Outside that, a commit carrying `RemoveSnapshots` reaches such a catalog and fails —
//! and what it fails is the write it rode in on, not a maintenance job.

use std::collections::HashSet;

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{TableMetadata, TableProperties, parse_property, parse_property_bool};
use crate::table::Table;
use crate::{TableIdent, TableUpdate};

/// Upper bound on the number of snapshots one commit may expire.
///
/// A first run against a long-neglected table would otherwise produce a single
/// `RemoveSnapshots` holding six-figure many ids: a multi-megabyte commit body,
/// and the catalog-side builder walks every retained snapshot against that list.
/// Oversized histories converge over several commits instead.
const MAX_EXPIRED_SNAPSHOTS_PER_COMMIT: usize = 1000;

/// The updates a commit against `table` must carry so the table's history stays
/// inside its retention window.
///
/// `table` is the local projection of the commit in progress — every action already
/// applied — so the snapshot this commit adds is present and current: it counts
/// towards `min_snapshots_to_keep` and roots the ancestor chain, which is what the
/// retention window is defined against.
///
/// Empty when the table configures no policy, or when nothing is eligible: a commit
/// that expires nothing carries no update at all.
pub(crate) fn expiration_updates(table: &Table) -> Result<Vec<TableUpdate>> {
    expiration_updates_at(
        table.metadata(),
        table.identifier(),
        chrono::Utc::now().timestamp_millis(),
    )
}

/// [`expiration_updates`] with the clock handed in, taking the two things a plan is
/// made of rather than the whole table.
///
/// `now_ms` is a parameter rather than a call to the clock so that the age window is
/// exercisable without waiting real time out; [`expiration_updates`] is where the
/// clock is read, and it is the only caller outside tests. `metadata` is everything
/// the plan is computed from, and `table_ident` names the table in the log lines
/// below and nowhere else — taking them apart is what lets the tests build a history
/// and nothing more.
fn expiration_updates_at(
    metadata: &TableMetadata,
    table_ident: &TableIdent,
    now_ms: i64,
) -> Result<Vec<TableUpdate>> {
    let Some(policy) = AutoExpirationPolicy::resolve(metadata)? else {
        return Ok(Vec::new());
    };

    let plan = AutoExpirationPlan::new(metadata, &policy, now_ms);
    if plan.expired.is_empty() {
        return Ok(Vec::new());
    }

    // Reported only for a commit that expires something: a table sitting inside its
    // retention window is not losing anything to the absent carrier yet, and a policy
    // naming a property nothing carries is otherwise a per-commit line on the ingest
    // path, at ingest frequency.
    if plan.preserve_carrier_missing {
        tracing::warn!(
            table = %table_ident,
            "no snapshot on the ancestor chain carries the preserved summary property; \
             only the retention window protects this history"
        );
    }

    tracing::debug!(
        table = %table_ident,
        snapshots_total = plan.snapshots_total,
        expired = plan.expired.len(),
        min_snapshots_to_keep = policy.min_snapshots_to_keep,
        max_snapshot_age_ms = policy.max_snapshot_age_ms,
        batch_limit = MAX_EXPIRED_SNAPSHOTS_PER_COMMIT,
        "expiring snapshots"
    );

    let mut updates = Vec::with_capacity(plan.expired.len() + 1);

    // Statistics files are keyed by snapshot id; dropping the snapshot without them
    // would strand the Puffin file as unreachable-but-referenced garbage. Only ids
    // the table actually has statistics for produce an update, and a table that has
    // none pays for neither lookup.
    if !metadata.statistics.is_empty() || !metadata.partition_statistics.is_empty() {
        for snapshot_id in &plan.expired {
            if metadata.statistics_for_snapshot(*snapshot_id).is_some() {
                updates.push(TableUpdate::RemoveStatistics {
                    snapshot_id: *snapshot_id,
                });
            }
            if metadata
                .partition_statistics_for_snapshot(*snapshot_id)
                .is_some()
            {
                updates.push(TableUpdate::RemovePartitionStatistics {
                    snapshot_id: *snapshot_id,
                });
            }
        }
    }

    updates.push(TableUpdate::RemoveSnapshots {
        snapshot_ids: plan.expired,
    });

    Ok(updates)
}

/// Snapshots the plan must never expire.
struct ProtectedSnapshots {
    ids: HashSet<i64>,
    preserve_carrier_missing: bool,
}

fn invalid_policy(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataInvalid, message)
}

/// Resolved snapshot-retention policy of a table.
///
/// Borrows the table metadata it was resolved from: the policy lives no longer than
/// the plan built against that same metadata.
#[derive(Debug)]
struct AutoExpirationPolicy<'a> {
    /// Number of the newest ancestors of the current snapshot that always survive.
    min_snapshots_to_keep: usize,
    /// Snapshots younger than this many milliseconds always survive.
    max_snapshot_age_ms: i64,
    /// Snapshot-summary key whose most recent carrier is preserved together with
    /// the ancestor chain leading to it. `None` disables that protection.
    preserve_summary_property: Option<&'a str>,
}

impl<'a> AutoExpirationPolicy<'a> {
    /// Resolve the table's retention policy from its properties.
    ///
    /// Returns `Ok(None)` when `history.expire.enabled` is absent or `false` — the
    /// table opted out and no history is touched.
    ///
    /// A window bound the table does not set falls back to the Iceberg spec default,
    /// the way every other property read in this crate resolves. Both spec defaults
    /// are wider than any window a table would configure deliberately, so a bound
    /// left out by mistake can only retain more history than intended, never expire
    /// more.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::DataInvalid` when a property that *is* set cannot be
    /// parsed, or is outside its valid range. Expiration rides along on commits that
    /// exist for other reasons, so this error fails the whole commit — it must stay
    /// reserved for a value someone actually wrote.
    fn resolve(metadata: &'a TableMetadata) -> Result<Option<Self>> {
        let properties = metadata.properties();

        // The key's absence is the opt-out, which is what the `false` default says:
        // a table that never configured a policy is never touched.
        if !parse_property_bool(
            properties,
            TableProperties::PROPERTY_HISTORY_EXPIRE_ENABLED,
            false,
        )? {
            return Ok(None);
        }

        // `gc.enabled=false` is an explicit operator decision that nothing may be
        // cleaned up on this table; expiring history is exactly such a cleanup,
        // so the more conservative setting wins. Upstream's ExpireSnapshotsAction
        // and Java's RemoveSnapshots refuse with an error here; this path rides
        // on every commit, where an error would fail unrelated writes, so it
        // skips instead.
        if !parse_property_bool(
            properties,
            TableProperties::PROPERTY_GC_ENABLED,
            TableProperties::PROPERTY_GC_ENABLED_DEFAULT,
        )? {
            return Ok(None);
        }

        // TODO: per-branch retention. The Iceberg spec lets a branch override
        // `history.expire.min-snapshots-to-keep` / `history.expire.max-snapshot-age-ms` on
        // the reference itself, and computes the retained set per branch, unioning the
        // results. Two things are missing here. Only the table-level policy is resolved,
        // so a branch asking for a window of its own does not get one; and the ancestor
        // walk starts at the current snapshot alone, so any other reference has its
        // target protected and nothing behind it — the ancestors of a side branch's head
        // expire as soon as they fall outside the age window, leaving that branch's head
        // readable but its history gone.

        let min_snapshots_to_keep = parse_property::<usize>(
            properties,
            TableProperties::PROPERTY_MIN_SNAPSHOTS_TO_KEEP,
            TableProperties::PROPERTY_MIN_SNAPSHOTS_TO_KEEP_DEFAULT,
        )?;
        if min_snapshots_to_keep == 0 {
            return Err(invalid_policy(format!(
                "{} must be positive",
                TableProperties::PROPERTY_MIN_SNAPSHOTS_TO_KEEP
            )));
        }

        let max_snapshot_age_ms = parse_property::<i64>(
            properties,
            TableProperties::PROPERTY_MAX_SNAPSHOT_AGE_MS,
            TableProperties::PROPERTY_MAX_SNAPSHOT_AGE_MS_DEFAULT,
        )?;
        if max_snapshot_age_ms < 0 {
            return Err(invalid_policy(format!(
                "{} must not be negative",
                TableProperties::PROPERTY_MAX_SNAPSHOT_AGE_MS
            )));
        }

        let preserve_summary_property = properties
            .get(TableProperties::PROPERTY_HISTORY_EXPIRE_PRESERVE_SUMMARY_PROPERTY)
            .map(String::as_str)
            .filter(|value| !value.is_empty());

        Ok(Some(AutoExpirationPolicy {
            min_snapshots_to_keep,
            max_snapshot_age_ms,
            preserve_summary_property,
        }))
    }
}

/// The snapshots one commit may expire.
#[derive(Debug)]
struct AutoExpirationPlan {
    /// Ids to expire, ordered oldest-first and capped at
    /// [`MAX_EXPIRED_SNAPSHOTS_PER_COMMIT`]. Empty means the commit carries no
    /// expiration update at all.
    expired: Vec<i64>,
    /// Number of snapshots the table held when the plan was computed.
    snapshots_total: usize,
    /// The policy names a summary property, but no snapshot on the ancestor chain
    /// carries it, so nothing beyond the retention window was protected. See
    /// [`AutoExpirationPlan::collect_protected`].
    preserve_carrier_missing: bool,
}

impl AutoExpirationPlan {
    /// Plan the snapshots this commit may expire.
    ///
    /// Every history takes the same path, including one that already sits inside the
    /// window: the walk is bounded by the number of snapshots the table holds, which
    /// is what the window keeps small in the first place, and a second path
    /// shortcutting on `snapshots_total <= min_snapshots_to_keep` would miss
    /// candidates outside the ancestor chain.
    fn new(
        metadata: &TableMetadata,
        policy: &AutoExpirationPolicy<'_>,
        now_ms: i64,
    ) -> AutoExpirationPlan {
        let snapshots_total = metadata.snapshots().len();
        let protected = Self::collect_protected(metadata, policy, snapshots_total);
        let age_cutoff_ms = now_ms.saturating_sub(policy.max_snapshot_age_ms);

        let mut candidates: Vec<(i64, i64)> = Vec::new();
        for snapshot in metadata.snapshots() {
            // The timestamp comparison comes first: it is cheaper than the set lookup.
            if snapshot.timestamp_ms() >= age_cutoff_ms {
                continue;
            }
            if protected.ids.contains(&snapshot.snapshot_id()) {
                continue;
            }
            candidates.push((snapshot.timestamp_ms(), snapshot.snapshot_id()));
        }

        // Catching an oversized history up takes the oldest first. Sorting all of it
        // to keep a thousandth of it is work the batch never reads: a partial
        // selection yields the same set in linear time, and the `(timestamp_ms,
        // snapshot_id)` key is unique, so which ids that set holds is determined.
        if candidates.len() > MAX_EXPIRED_SNAPSHOTS_PER_COMMIT {
            candidates.select_nth_unstable(MAX_EXPIRED_SNAPSHOTS_PER_COMMIT - 1);
            candidates.truncate(MAX_EXPIRED_SNAPSHOTS_PER_COMMIT);
        }
        // `snapshots()` iterates a `HashMap`, so without an explicit order the same
        // history would yield a different batch on every attempt and the catch-up of
        // an oversized history would stop being predictable.
        candidates.sort_unstable();

        AutoExpirationPlan {
            expired: candidates
                .into_iter()
                .map(|(_, snapshot_id)| snapshot_id)
                .collect(),
            snapshots_total,
            preserve_carrier_missing: protected.preserve_carrier_missing,
        }
    }

    /// Collect the snapshots the plan must never expire:
    ///
    /// - the target of every reference. A reference whose snapshot is gone is dropped
    ///   by [`TableMetadataBuilder::remove_snapshots`] without a word, so an
    ///   unprotected tag would vanish together with the snapshot it names.
    /// - the current snapshot and the newest `min_snapshots_to_keep` ancestors.
    /// - the ancestor chain down to — and including — the most recent carrier of
    ///   `preserve_summary_property`. Readers walk that chain by `parent_snapshot_id`
    ///   to reach the carrier, so cutting any link between the current snapshot and
    ///   the carrier hides the carrier just as surely as expiring it.
    ///
    /// Both of the last two are a prefix of the same chain, so one walk answers both,
    /// and it stops as soon as the window is behind it and the carrier is found:
    /// deeper ancestors are protected by neither rule.
    ///
    /// When the policy names a property no snapshot on the chain carries, nothing
    /// beyond the window is protected and `preserve_carrier_missing` reports it: there
    /// is no carrier for the walk to reach either way, and holding the history back
    /// would block expiration for as long as the property stays unwritten — which is
    /// indefinitely, and by exactly the mechanism this whole module exists to prevent.
    ///
    /// [`TableMetadataBuilder::remove_snapshots`]: crate::spec::TableMetadataBuilder::remove_snapshots
    fn collect_protected(
        metadata: &TableMetadata,
        policy: &AutoExpirationPolicy<'_>,
        snapshots_total: usize,
    ) -> ProtectedSnapshots {
        let mut ids: HashSet<i64> = metadata
            .refs
            .values()
            .map(|reference| reference.snapshot_id)
            .collect();

        // Kept even though the walk below inserts it at position 0: metadata whose
        // current snapshot is not in `snapshots` would otherwise leave it unprotected.
        if let Some(current_snapshot_id) = metadata.current_snapshot_id() {
            ids.insert(current_snapshot_id);
        }

        // Ancestors past the window that are still waiting on a carrier: they become
        // protected only if one turns up deeper, and are dropped otherwise. In the
        // ordinary case — carrier inside the window, or no carrier key at all — this
        // never allocates.
        let mut pending: Vec<i64> = Vec::new();
        let mut carrier_found = policy.preserve_summary_property.is_none();

        let mut next = metadata.current_snapshot_id();
        let mut position = 0usize;

        while let Some(snapshot_id) = next {
            // A cycle in `parent_snapshot_id` cannot be longer than the number of
            // snapshots, so the counter terminates the walk and a visited set is
            // not needed.
            if position == snapshots_total {
                break;
            }
            let Some(snapshot) = metadata.snapshot_by_id(snapshot_id) else {
                break;
            };

            if position < policy.min_snapshots_to_keep {
                ids.insert(snapshot_id);
            } else if !carrier_found {
                pending.push(snapshot_id);
            }

            if !carrier_found
                && policy
                    .preserve_summary_property
                    .is_some_and(|key| snapshot.summary().additional_properties.contains_key(key))
            {
                carrier_found = true;
                ids.insert(snapshot_id);
                ids.extend(pending.drain(..));
            }

            position += 1;

            // The window is behind the walk and the carrier is reachable: nothing
            // deeper is protected by either rule.
            if carrier_found && position >= policy.min_snapshots_to_keep {
                break;
            }

            next = snapshot.parent_snapshot_id();
        }

        ProtectedSnapshots {
            ids,
            preserve_carrier_missing: !carrier_found,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::TableIdent;
    use crate::spec::{
        FormatVersion, MAIN_BRANCH, NestedField, Operation, PartitionStatisticsFile, PrimitiveType,
        Schema, Snapshot, SnapshotReference, SnapshotRetention, SortOrder, StatisticsFile, Summary,
        TableMetadataBuilder, Transform, Type, UnboundPartitionSpec,
    };

    const TEST_LOCATION: &str = "s3://bucket/test/location";
    const BASE_TIMESTAMP_MS: i64 = 1_700_000_000_000;
    const CARRIER_PROPERTY: &str = "example.offset";

    fn schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap()
    }

    fn partition_spec() -> UnboundPartitionSpec {
        UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "y", Transform::Identity)
            .unwrap()
            .build()
    }

    /// A snapshot whose parent is the previous one in the chain, timestamped one
    /// second per id so ordering assertions read directly off the id.
    fn snapshot(snapshot_id: i64, parent_snapshot_id: Option<i64>) -> Snapshot {
        snapshot_with_summary(snapshot_id, parent_snapshot_id, HashMap::new())
    }

    /// As [`snapshot`], with the summary the caller needs on it.
    fn snapshot_with_summary(
        snapshot_id: i64,
        parent_snapshot_id: Option<i64>,
        additional_properties: HashMap<String, String>,
    ) -> Snapshot {
        snapshot_at(
            snapshot_id,
            parent_snapshot_id,
            BASE_TIMESTAMP_MS + snapshot_id * 1_000,
            additional_properties,
        )
    }

    /// The one snapshot builder these fixtures have. Timestamp is a parameter
    /// because [`snapshot`] derives it from the id, which cannot put two snapshots
    /// in the same millisecond.
    fn snapshot_at(
        snapshot_id: i64,
        parent_snapshot_id: Option<i64>,
        timestamp_ms: i64,
        additional_properties: HashMap<String, String>,
    ) -> Snapshot {
        Snapshot::builder()
            .with_snapshot_id(snapshot_id)
            .with_parent_snapshot_id(parent_snapshot_id)
            .with_sequence_number(snapshot_id)
            .with_timestamp_ms(timestamp_ms)
            .with_schema_id(0)
            .with_manifest_list(format!("/snap-{snapshot_id}.avro"))
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties,
            })
            .build()
    }

    fn metadata_builder(properties: HashMap<String, String>) -> TableMetadataBuilder {
        TableMetadataBuilder::new(
            schema(),
            partition_spec(),
            SortOrder::unsorted_order(),
            TEST_LOCATION.to_string(),
            FormatVersion::V2,
            properties,
        )
        .unwrap()
    }

    /// Metadata holding a linear history `1..=count`, current snapshot `count`.
    fn metadata_with_chain(count: i64, properties: HashMap<String, String>) -> TableMetadata {
        metadata_with_carriers_at(count, properties, &[])
    }

    /// As [`metadata_with_chain`], but the snapshots with the given ids carry
    /// [`CARRIER_PROPERTY`] in their summaries.
    fn metadata_with_carriers_at(
        count: i64,
        properties: HashMap<String, String>,
        carrier_snapshot_ids: &[i64],
    ) -> TableMetadata {
        chain_builder(count, properties, carrier_snapshot_ids)
            .build()
            .unwrap()
            .metadata
    }

    /// The builder behind [`metadata_with_carriers_at`], for tests that need to put
    /// further references or statistics on the history before it is built.
    fn chain_builder(
        count: i64,
        properties: HashMap<String, String>,
        carrier_snapshot_ids: &[i64],
    ) -> TableMetadataBuilder {
        let mut builder = metadata_builder(properties);
        for snapshot_id in 1..=count {
            let parent_snapshot_id = (snapshot_id > 1).then_some(snapshot_id - 1);
            let snapshot = if carrier_snapshot_ids.contains(&snapshot_id) {
                snapshot_with_summary(
                    snapshot_id,
                    parent_snapshot_id,
                    HashMap::from([(CARRIER_PROPERTY.to_string(), "42".to_string())]),
                )
            } else {
                snapshot(snapshot_id, parent_snapshot_id)
            };
            builder = builder.set_branch_snapshot(snapshot, MAIN_BRANCH).unwrap();
        }
        builder
    }

    fn enabled_properties(
        min_snapshots_to_keep: u32,
        max_snapshot_age_ms: i64,
    ) -> HashMap<String, String> {
        HashMap::from([
            (
                TableProperties::PROPERTY_HISTORY_EXPIRE_ENABLED.to_string(),
                "true".to_string(),
            ),
            (
                TableProperties::PROPERTY_MIN_SNAPSHOTS_TO_KEEP.to_string(),
                min_snapshots_to_keep.to_string(),
            ),
            (
                TableProperties::PROPERTY_MAX_SNAPSHOT_AGE_MS.to_string(),
                max_snapshot_age_ms.to_string(),
            ),
        ])
    }

    /// [`enabled_properties`] plus the carrier key the policy is to preserve.
    fn enabled_properties_with_carrier(
        min_snapshots_to_keep: u32,
        max_snapshot_age_ms: i64,
        preserve_summary_property: &str,
    ) -> HashMap<String, String> {
        let mut properties = enabled_properties(min_snapshots_to_keep, max_snapshot_age_ms);
        properties.insert(
            TableProperties::PROPERTY_HISTORY_EXPIRE_PRESERVE_SUMMARY_PROPERTY.to_string(),
            preserve_summary_property.to_string(),
        );
        properties
    }

    /// A `now` far enough past the fixture history that age never protects anything,
    /// including the longest fixture (one snapshot past the batch limit).
    fn now_after_history() -> i64 {
        BASE_TIMESTAMP_MS + 10_000_000
    }

    /// The identifier the log lines carry; nothing the plan decides depends on it.
    fn table_ident() -> TableIdent {
        TableIdent::from_strs(["ns1", "test1"]).unwrap()
    }

    /// The only entry into the module these tests use: everything the retention
    /// policy and the plan decide is observable in the updates a commit would carry.
    fn updates_for(metadata: TableMetadata, now_ms: i64) -> Result<Vec<TableUpdate>> {
        expiration_updates_at(&metadata, &table_ident(), now_ms)
    }

    /// The ids of `RemoveSnapshots`; empty when the updates carry no such update.
    fn expired_ids(updates: &[TableUpdate]) -> Vec<i64> {
        updates
            .iter()
            .find_map(|update| match update {
                TableUpdate::RemoveSnapshots { snapshot_ids } => Some(snapshot_ids.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    fn statistics_file(snapshot_id: i64) -> StatisticsFile {
        StatisticsFile {
            snapshot_id,
            statistics_path: format!("test/stats-{snapshot_id}.puffin"),
            file_size_in_bytes: 100,
            file_footer_size_in_bytes: 10,
            key_metadata: None,
            blob_metadata: vec![],
        }
    }

    fn partition_statistics_file(snapshot_id: i64) -> PartitionStatisticsFile {
        PartitionStatisticsFile {
            snapshot_id,
            statistics_path: format!("test/partition-stats-{snapshot_id}.parquet"),
            file_size_in_bytes: 100,
        }
    }

    #[test]
    fn expiration_is_off_without_the_enabled_property() {
        let updates =
            updates_for(metadata_with_chain(5, HashMap::new()), now_after_history()).unwrap();
        assert!(updates.is_empty());
    }

    #[test]
    fn expiration_is_off_when_disabled() {
        let mut properties = enabled_properties(3, 0);
        properties.insert(
            TableProperties::PROPERTY_HISTORY_EXPIRE_ENABLED.to_string(),
            "false".to_string(),
        );
        let updates = updates_for(metadata_with_chain(5, properties), now_after_history()).unwrap();
        assert!(updates.is_empty());
    }

    #[test]
    fn gc_disabled_turns_expiration_off() {
        let mut properties = enabled_properties(3, 0);
        properties.insert(
            TableProperties::PROPERTY_GC_ENABLED.to_string(),
            "false".to_string(),
        );
        let updates = updates_for(metadata_with_chain(5, properties), now_after_history()).unwrap();
        assert!(
            updates.is_empty(),
            "gc.enabled=false must veto automatic expiration"
        );
    }

    #[test]
    fn unparsable_gc_enabled_fails_the_commit() {
        let mut properties = enabled_properties(3, 0);
        properties.insert(
            TableProperties::PROPERTY_GC_ENABLED.to_string(),
            "maybe".to_string(),
        );
        assert_eq!(
            updates_for(metadata_with_chain(5, properties), now_after_history())
                .unwrap_err()
                .kind(),
            ErrorKind::DataInvalid
        );
    }

    #[test]
    fn enabled_flag_is_parsed_case_insensitively() {
        let mut properties = enabled_properties(3, 0);
        properties.insert(
            TableProperties::PROPERTY_HISTORY_EXPIRE_ENABLED.to_string(),
            "True".to_string(),
        );
        let updates = updates_for(metadata_with_chain(5, properties), now_after_history()).unwrap();
        assert_eq!(
            expired_ids(&updates),
            vec![1, 2],
            "\"True\" must enable the policy the same way \"true\" does"
        );
    }

    #[test]
    fn unparsable_enabled_flag_fails_the_commit() {
        let mut properties = enabled_properties(3, 0);
        properties.insert(
            TableProperties::PROPERTY_HISTORY_EXPIRE_ENABLED.to_string(),
            "yes".to_string(),
        );
        assert_eq!(
            updates_for(metadata_with_chain(5, properties), now_after_history())
                .unwrap_err()
                .kind(),
            ErrorKind::DataInvalid
        );
    }

    #[test]
    fn missing_min_snapshots_to_keep_falls_back_to_the_spec_default() {
        let mut properties = enabled_properties(3, 0);
        properties.remove(TableProperties::PROPERTY_MIN_SNAPSHOTS_TO_KEEP);
        let updates = updates_for(metadata_with_chain(5, properties), now_after_history()).unwrap();
        assert_eq!(
            expired_ids(&updates),
            vec![1, 2, 3, 4],
            "the spec default keeps one ancestor, so everything below it goes"
        );
    }

    #[test]
    fn missing_max_snapshot_age_falls_back_to_the_spec_default() {
        let mut properties = enabled_properties(3, 0);
        properties.remove(TableProperties::PROPERTY_MAX_SNAPSHOT_AGE_MS);
        let updates = updates_for(
            metadata_with_chain(5, properties),
            BASE_TIMESTAMP_MS + 5_000,
        )
        .unwrap();
        assert!(
            updates.is_empty(),
            "the spec default age window is five days, so nothing in the fixture is old enough"
        );
    }

    #[test]
    fn zero_min_snapshots_to_keep_fails_the_commit() {
        assert_eq!(
            updates_for(
                metadata_with_chain(5, enabled_properties(0, 0)),
                now_after_history()
            )
            .unwrap_err()
            .kind(),
            ErrorKind::DataInvalid
        );
    }

    #[test]
    fn unparsable_min_snapshots_to_keep_fails_the_commit() {
        let mut properties = enabled_properties(3, 0);
        properties.insert(
            TableProperties::PROPERTY_MIN_SNAPSHOTS_TO_KEEP.to_string(),
            "abc".to_string(),
        );
        assert_eq!(
            updates_for(metadata_with_chain(5, properties), now_after_history())
                .unwrap_err()
                .kind(),
            ErrorKind::DataInvalid
        );
    }

    #[test]
    fn unparsable_max_snapshot_age_fails_the_commit() {
        // The same parser as the count above, instantiated for `i64`: a unit suffix
        // is the shape a hand-written value takes, and it is not a number.
        let mut properties = enabled_properties(3, 0);
        properties.insert(
            TableProperties::PROPERTY_MAX_SNAPSHOT_AGE_MS.to_string(),
            "10m".to_string(),
        );
        assert_eq!(
            updates_for(metadata_with_chain(5, properties), now_after_history())
                .unwrap_err()
                .kind(),
            ErrorKind::DataInvalid
        );
    }

    #[test]
    fn negative_max_snapshot_age_fails_the_commit() {
        assert_eq!(
            updates_for(
                metadata_with_chain(5, enabled_properties(3, -1)),
                now_after_history()
            )
            .unwrap_err()
            .kind(),
            ErrorKind::DataInvalid
        );
    }

    #[test]
    fn empty_preserve_summary_property_is_treated_as_absent() {
        let metadata =
            metadata_with_carriers_at(5, enabled_properties_with_carrier(2, 0, ""), &[2]);
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(
            expired_ids(&updates),
            vec![1, 2, 3],
            "an empty carrier key names no property and must not enable the protection"
        );
    }

    #[test]
    fn history_of_exactly_min_is_left_alone() {
        let metadata = metadata_with_chain(3, enabled_properties(3, 0));
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert!(updates.is_empty());
    }

    #[test]
    fn history_within_the_age_window_is_left_alone() {
        // Age cutoff sits before the oldest snapshot, so nothing is old enough.
        let metadata = metadata_with_chain(4, enabled_properties(3, 1_000_000));
        let updates = updates_for(metadata, BASE_TIMESTAMP_MS + 4_000).unwrap();
        assert!(updates.is_empty());
    }

    #[test]
    fn an_extreme_max_snapshot_age_protects_the_whole_history() {
        // The largest age the policy accepts: parsable, not negative, so `resolve`
        // passes it through, and the cutoff it produces underflows. Saturating there
        // puts the cutoff before every timestamp; subtracting would wrap it past
        // `now` and expire the history it was meant to protect.
        let metadata = metadata_with_chain(5, enabled_properties(1, i64::MAX));
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert!(
            updates.is_empty(),
            "an age window this wide covers every snapshot the table has"
        );
    }

    #[test]
    fn a_table_without_snapshots_produces_no_updates() {
        // A table created with the policy already in its properties: there is no
        // current snapshot to root the ancestor walk and nothing to expire.
        let metadata = metadata_builder(enabled_properties(2, 0))
            .build()
            .unwrap()
            .metadata;
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert!(updates.is_empty());
    }

    #[test]
    fn one_snapshot_past_both_bounds_is_expired() {
        let metadata = metadata_with_chain(4, enabled_properties(3, 0));
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(expired_ids(&updates), vec![1]);
    }

    #[test]
    fn carrier_deeper_than_the_min_window_protects_the_chain_to_it() {
        let metadata = metadata_with_carriers_at(
            6,
            enabled_properties_with_carrier(2, 0, CARRIER_PROPERTY),
            &[2],
        );
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(
            expired_ids(&updates),
            vec![1],
            "snapshots 2..=6 form the chain down to the carrier and must survive"
        );
    }

    #[test]
    fn carrier_on_the_current_snapshot_protects_only_the_min_window() {
        let metadata = metadata_with_carriers_at(
            5,
            enabled_properties_with_carrier(2, 0, CARRIER_PROPERTY),
            &[5],
        );
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(expired_ids(&updates), vec![1, 2, 3]);
    }

    #[test]
    fn only_the_most_recent_carrier_protects_the_chain() {
        let metadata = metadata_with_carriers_at(
            5,
            enabled_properties_with_carrier(2, 0, CARRIER_PROPERTY),
            &[2, 4],
        );
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(
            expired_ids(&updates),
            vec![1, 2, 3],
            "the walk stops at the carrier on 4; the older carrier on 2 protects nothing"
        );
    }

    #[test]
    fn missing_carrier_leaves_expiration_to_the_window() {
        let metadata =
            metadata_with_chain(5, enabled_properties_with_carrier(2, 0, CARRIER_PROPERTY));
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(
            expired_ids(&updates),
            vec![1, 2, 3],
            "with no carrier to keep reachable, only the window protects anything"
        );
    }

    #[test]
    fn a_tag_outside_the_window_is_protected() {
        let metadata = chain_builder(5, enabled_properties(2, 0), &[])
            .set_ref("stable", SnapshotReference {
                snapshot_id: 1,
                retention: SnapshotRetention::Tag {
                    max_ref_age_ms: None,
                },
            })
            .unwrap()
            .build()
            .unwrap()
            .metadata;

        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(
            expired_ids(&updates),
            vec![2, 3],
            "expiring snapshot 1 would take the tag pointing at it with it"
        );
    }

    #[test]
    fn a_second_branch_is_protected() {
        let metadata = chain_builder(5, enabled_properties(2, 0), &[])
            .set_ref("side", SnapshotReference {
                snapshot_id: 2,
                retention: SnapshotRetention::branch(None, None, None),
            })
            .unwrap()
            .build()
            .unwrap()
            .metadata;

        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(expired_ids(&updates), vec![1, 3]);
    }

    #[test]
    fn a_detached_snapshot_is_expired() {
        // Detached: the target of no reference and not on the ancestor chain of the
        // current snapshot. The side branch moved forward, which is how its previous
        // head became one. Candidates are collected over every snapshot the metadata
        // holds rather than by walking that chain, which is the only reason such a
        // snapshot is reachable by the plan at all.
        let metadata = chain_builder(5, enabled_properties(2, 0), &[])
            .set_branch_snapshot(snapshot(10, None), "side")
            .unwrap()
            .set_branch_snapshot(snapshot(11, Some(10)), "side")
            .unwrap()
            .build()
            .unwrap()
            .metadata;

        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(
            expired_ids(&updates),
            vec![1, 2, 3, 10],
            "10 is off the chain the window is measured along, and 11 — the head of \
             the side branch — is protected as a reference target"
        );
    }

    #[test]
    fn a_history_shorter_than_the_window_is_left_alone() {
        // The window is wider than the history, so the walk runs out of ancestors
        // before it fills — where `history_of_exactly_min_is_left_alone` ends on the
        // window boundary.
        let metadata = metadata_with_chain(2, enabled_properties(3, 0));
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert!(updates.is_empty());
    }

    #[test]
    fn chain_protection_is_off_without_a_carrier_property() {
        let metadata = metadata_with_carriers_at(5, enabled_properties(2, 0), &[2]);
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(expired_ids(&updates), vec![1, 2, 3]);
    }

    #[test]
    fn batch_is_capped_at_the_limit_and_takes_the_oldest() {
        let count = (MAX_EXPIRED_SNAPSHOTS_PER_COMMIT + 10) as i64;
        let metadata = metadata_with_chain(count, enabled_properties(2, 0));
        let updates = updates_for(metadata, now_after_history()).unwrap();

        let expired = expired_ids(&updates);
        assert_eq!(expired.len(), MAX_EXPIRED_SNAPSHOTS_PER_COMMIT);
        assert_eq!(expired.first(), Some(&1));
        assert_eq!(
            expired.last(),
            Some(&(MAX_EXPIRED_SNAPSHOTS_PER_COMMIT as i64))
        );
    }

    #[test]
    fn snapshots_sharing_a_timestamp_are_ordered_by_id() {
        // Commits land in the same millisecond under any real ingest rate. The
        // timestamp alone then leaves the batch to the HashMap iteration order; the
        // id resolves the tie and is what makes the batch reproducible.
        let mut builder = metadata_builder(enabled_properties(2, 0));
        for snapshot_id in 1..=6 {
            let parent_snapshot_id = (snapshot_id > 1).then_some(snapshot_id - 1);
            builder = builder
                .set_branch_snapshot(
                    snapshot_at(
                        snapshot_id,
                        parent_snapshot_id,
                        BASE_TIMESTAMP_MS,
                        HashMap::new(),
                    ),
                    MAIN_BRANCH,
                )
                .unwrap();
        }
        let metadata = builder.build().unwrap().metadata;

        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(expired_ids(&updates), vec![1, 2, 3, 4]);
    }

    #[test]
    fn plan_is_deterministic_across_runs() {
        // `snapshots()` iterates a HashMap; a fresh metadata instance gives a fresh
        // iteration order, so two independent builds must still agree. The history is
        // wider than the batch limit on purpose: below it every candidate ends up in
        // the batch and a full sort settles the order, while above it the batch is a
        // partial selection, and which candidates it holds is settled by the sort key
        // being unique — nothing else.
        let count = (MAX_EXPIRED_SNAPSHOTS_PER_COMMIT + 10) as i64;
        let first = updates_for(
            metadata_with_chain(count, enabled_properties(2, 0)),
            now_after_history(),
        )
        .unwrap();
        let second = updates_for(
            metadata_with_chain(count, enabled_properties(2, 0)),
            now_after_history(),
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn a_cyclic_chain_wider_than_the_window_terminates() {
        // A window narrower than the history would stop the walk on its own boundary,
        // cycle guard or not. Here the window is wider than the whole history, so only
        // the snapshot count ends the walk: a regression in that guard hangs this test
        // rather than failing it.
        let metadata = metadata_builder(enabled_properties(5, 0))
            .set_branch_snapshot(snapshot(1, Some(2)), MAIN_BRANCH)
            .unwrap()
            .set_branch_snapshot(snapshot(2, Some(1)), MAIN_BRANCH)
            .unwrap()
            .set_branch_snapshot(snapshot(3, Some(2)), MAIN_BRANCH)
            .unwrap()
            .build()
            .unwrap()
            .metadata;

        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert!(
            updates.is_empty(),
            "the walk reaches every snapshot before the window closes"
        );
    }

    #[test]
    fn absent_parent_truncates_the_chain() {
        let metadata = metadata_builder(enabled_properties_with_carrier(2, 0, CARRIER_PROPERTY))
            .set_branch_snapshot(snapshot(1, None), MAIN_BRANCH)
            .unwrap()
            .set_branch_snapshot(snapshot(2, Some(1)), MAIN_BRANCH)
            .unwrap()
            // Parent 7 was never added to this metadata.
            .set_branch_snapshot(snapshot(3, Some(7)), MAIN_BRANCH)
            .unwrap()
            .build()
            .unwrap()
            .metadata;

        // The chain is just the current snapshot: 1 and 2 are unreachable by the
        // parent walk, so no reader can reach a carrier on them either.
        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(expired_ids(&updates), vec![1, 2]);
    }

    #[test]
    fn statistics_of_an_expired_snapshot_are_removed_with_it() {
        let metadata = chain_builder(5, enabled_properties(2, 0), &[])
            .set_statistics(statistics_file(1))
            .build()
            .unwrap()
            .metadata;

        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(expired_ids(&updates), vec![1, 2, 3]);
        assert!(
            updates
                .iter()
                .any(|update| matches!(update, TableUpdate::RemoveStatistics { snapshot_id } if *snapshot_id == 1)),
            "statistics keyed by an expired snapshot would be unreachable garbage"
        );
    }

    #[test]
    fn partition_statistics_of_an_expired_snapshot_are_removed_with_it() {
        let metadata = chain_builder(5, enabled_properties(2, 0), &[])
            .set_partition_statistics(partition_statistics_file(1))
            .build()
            .unwrap()
            .metadata;

        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert!(
            updates.iter().any(|update| matches!(
                update,
                TableUpdate::RemovePartitionStatistics { snapshot_id } if *snapshot_id == 1
            )),
            "partition statistics keyed by an expired snapshot would be unreachable garbage"
        );
    }

    #[test]
    fn statistics_of_a_surviving_snapshot_are_kept() {
        let metadata = chain_builder(5, enabled_properties(2, 0), &[])
            .set_statistics(statistics_file(5))
            .build()
            .unwrap()
            .metadata;

        let updates = updates_for(metadata, now_after_history()).unwrap();
        assert_eq!(expired_ids(&updates), vec![1, 2, 3]);
        assert!(
            !updates
                .iter()
                .any(|update| matches!(update, TableUpdate::RemoveStatistics { .. })),
            "the only statistics file belongs to a snapshot that survives"
        );
    }
}

/// Expiration as a committing writer sees it: these drive real transactions
/// through a catalog and read the result off the committed metadata, where the
/// tests above stop at the updates a commit would carry, against a fixed clock.
/// This is the only place [`expiration_updates`] is exercised with the real one.
#[cfg(test)]
mod commit_tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use crate::catalog::MockCatalog;
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, StatisticsFile,
        Struct, TableProperties,
    };
    use crate::table::Table;
    use crate::transaction::tests::make_v2_minimal_table_in_catalog_with_properties;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, Error, ErrorKind, TableUpdate};

    const MIN_SNAPSHOTS_TO_KEEP: usize = 3;
    const CARRIER_PROPERTY: &str = "example.offset";

    /// How many appends the fixtures write before the commit under test: enough that
    /// the retention window leaves part of that history eligible.
    const HISTORY_LENGTH: usize = 6;

    /// Retention that keeps the newest [`MIN_SNAPSHOTS_TO_KEEP`] snapshots and
    /// nothing else: a zero age window makes every older snapshot eligible at once,
    /// so no test has to wait out real time to observe expiration.
    fn retention_properties() -> HashMap<String, String> {
        HashMap::from([
            (
                TableProperties::PROPERTY_HISTORY_EXPIRE_ENABLED.to_string(),
                "true".to_string(),
            ),
            (
                TableProperties::PROPERTY_MIN_SNAPSHOTS_TO_KEEP.to_string(),
                MIN_SNAPSHOTS_TO_KEEP.to_string(),
            ),
            (
                TableProperties::PROPERTY_MAX_SNAPSHOT_AGE_MS.to_string(),
                "0".to_string(),
            ),
        ])
    }

    /// [`retention_properties`] plus [`CARRIER_PROPERTY`] as the key to preserve.
    /// Whether any snapshot ends up carrying it is up to the writer each test drives.
    fn retention_properties_with_a_carrier() -> HashMap<String, String> {
        let mut properties = retention_properties();
        properties.insert(
            TableProperties::PROPERTY_HISTORY_EXPIRE_PRESERVE_SUMMARY_PROPERTY.to_string(),
            CARRIER_PROPERTY.to_string(),
        );
        properties
    }

    /// [`retention_properties`] plus a retry budget a test can wait out: the commit
    /// retry defaults back off for minutes.
    fn retention_properties_with_fast_retries() -> HashMap<String, String> {
        let mut properties = retention_properties();
        properties.extend([
            (
                TableProperties::PROPERTY_COMMIT_MIN_RETRY_WAIT_MS.to_string(),
                "10".to_string(),
            ),
            (
                TableProperties::PROPERTY_COMMIT_MAX_RETRY_WAIT_MS.to_string(),
                "100".to_string(),
            ),
            (
                TableProperties::PROPERTY_COMMIT_TOTAL_RETRY_TIME_MS.to_string(),
                "1000".to_string(),
            ),
            (
                TableProperties::PROPERTY_COMMIT_NUM_RETRIES.to_string(),
                "3".to_string(),
            ),
        ]);
        properties
    }

    fn data_file(name: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("test/{name}.parquet"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    async fn append(table: &Table, catalog: &impl Catalog, name: &str) -> Table {
        let tx = Transaction::new(table);
        let action = tx.fast_append().add_data_files(vec![data_file(name)]);
        action.apply(tx).unwrap().commit(catalog).await.unwrap()
    }

    /// [`append`] whose snapshot carries [`CARRIER_PROPERTY`] in its summary, written
    /// the only way a caller of this crate can write it: through the action.
    async fn append_carrying_the_carrier(
        table: &Table,
        catalog: &impl Catalog,
        name: &str,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx
            .fast_append()
            .set_snapshot_properties(HashMap::from([(
                CARRIER_PROPERTY.to_string(),
                "42".to_string(),
            )]))
            .add_data_files(vec![data_file(name)]);
        action.apply(tx).unwrap().commit(catalog).await.unwrap()
    }

    fn statistics_file(snapshot_id: i64) -> StatisticsFile {
        StatisticsFile {
            snapshot_id,
            statistics_path: format!("test/stats-{snapshot_id}.puffin"),
            file_size_in_bytes: 100,
            file_footer_size_in_bytes: 10,
            key_metadata: None,
            blob_metadata: vec![],
        }
    }

    /// `table` carrying `properties`, applied to its metadata rather than committed.
    ///
    /// A history long enough to expire has to exist before the policy that expires
    /// it: appended under a live policy it would be cut down as it was written, and
    /// what each commit found is not what the test is about. So the history is
    /// written with no policy in place and the policy is put on the base the commit
    /// under test is produced against — which is exactly what the catalog hands a
    /// writer whose table was configured while it was not looking.
    fn with_properties(table: &Table, properties: HashMap<String, String>) -> Table {
        let metadata = table
            .metadata()
            .clone()
            .into_builder(None)
            .set_properties(properties)
            .unwrap()
            .build()
            .unwrap()
            .metadata;
        table.clone().with_metadata(Arc::new(metadata))
    }

    /// The snapshot ids of `table` oldest first, as its snapshot log records them.
    fn snapshot_ids_oldest_first(table: &Table) -> Vec<i64> {
        table
            .metadata()
            .history()
            .iter()
            .map(|entry| entry.snapshot_id)
            .collect()
    }

    /// The ids of `RemoveSnapshots`, sorted; empty when the updates carry no such
    /// update. Sorted because a commit-level fixture times its own snapshots: two
    /// landing in the same millisecond are ordered by id, which is random here, so
    /// only the set of ids is the test's business.
    fn expired_ids(updates: &[TableUpdate]) -> Vec<i64> {
        let mut ids = updates
            .iter()
            .find_map(|update| match update {
                TableUpdate::RemoveSnapshots { snapshot_ids } => Some(snapshot_ids.clone()),
                _ => None,
            })
            .unwrap_or_default();
        ids.sort_unstable();
        ids
    }

    /// The ids of `table` the retention window leaves eligible: its whole history
    /// but the newest [`MIN_SNAPSHOTS_TO_KEEP`], which is what a commit against a
    /// history built with a zero age window is expected to expire.
    fn ids_past_the_window(table: &Table) -> Vec<i64> {
        let history = snapshot_ids_oldest_first(table);
        let mut expired: Vec<i64> = history
            .into_iter()
            .rev()
            .skip(MIN_SNAPSHOTS_TO_KEEP)
            .collect();
        expired.sort_unstable();
        expired
    }

    /// Commit `tx` through a catalog that hands out `base` and captures the updates
    /// the commit produced instead of applying them.
    ///
    /// What a commit does not emit leaves no trace in the resulting metadata, and
    /// neither does the order it emitted things in; both are only observable here.
    async fn updates_of_a_commit_against(base: &Table, tx: Transaction) -> Vec<TableUpdate> {
        let captured_updates = Arc::new(Mutex::new(Vec::new()));
        let mut mock_catalog = MockCatalog::new();

        let loaded_table = base.clone();
        mock_catalog.expect_load_table().returning_st(move |_| {
            let loaded_table = loaded_table.clone();
            Box::pin(async move { Ok(loaded_table) })
        });

        let sink = Arc::clone(&captured_updates);
        let committed_table = base.clone();
        mock_catalog
            .expect_update_table()
            .times(1)
            .returning_st(move |mut commit| {
                sink.lock().unwrap().extend(commit.take_updates());
                let committed_table = committed_table.clone();
                Box::pin(async move { Ok(committed_table) })
            });

        tx.commit(&mock_catalog).await.unwrap();

        let captured_updates = captured_updates.lock().unwrap();
        captured_updates.clone()
    }

    /// A commit that writes no snapshot: it exists to carry a property change, and
    /// what the tests below read off it is the expiration that rode along.
    fn set_properties_transaction(
        base: &Table,
        properties: HashMap<String, String>,
    ) -> Transaction {
        let tx = Transaction::new(base);
        let mut action = tx.update_table_properties();
        for (key, value) in properties {
            action = action.set(key, value);
        }
        action.apply(tx).unwrap()
    }

    /// A zero age window puts the cutoff at the clock the commit reads, and a
    /// snapshot written in that same millisecond is not older than it. Fixtures that
    /// expect their whole history to be eligible wait that millisecond out.
    async fn let_the_history_age() {
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    #[tokio::test]
    async fn repeated_appends_converge_to_the_retention_window() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, retention_properties())
                .await;

        for i in 0..8 {
            table = append(&table, &catalog, &format!("f{i}")).await;
            // Inside the loop: convergence is the next commit expiring what this one
            // wrote, so every snapshot has to age past the commit that follows it.
            let_the_history_age().await;

            let metadata = table.metadata();
            let snapshot_count = metadata.snapshots().len();
            assert!(
                snapshot_count <= MIN_SNAPSHOTS_TO_KEEP + 1,
                "history must stay inside the window, saw {snapshot_count} snapshots"
            );
            let current_snapshot_id = metadata
                .current_snapshot_id()
                .expect("the commit just made a snapshot current");
            assert!(
                metadata.snapshot_by_id(current_snapshot_id).is_some(),
                "the current snapshot must never be expired"
            );
            assert_eq!(
                metadata.history().last().map(|entry| entry.snapshot_id),
                Some(current_snapshot_id),
                "the snapshot log must still end on the current snapshot"
            );
        }
    }

    /// The carrier as a reader reaches it: from the current snapshot down the
    /// `parent_snapshot_id` chain. Panics naming the snapshot the walk broke on, since
    /// a cut chain and an expired carrier hide the carrier alike.
    fn walk_the_chain_to_the_carrier(table: &Table, carrier_snapshot_id: i64) {
        let metadata = table.metadata();
        let mut snapshot_id = metadata
            .current_snapshot_id()
            .expect("the last append made a snapshot current");

        loop {
            let snapshot = metadata.snapshot_by_id(snapshot_id).unwrap_or_else(|| {
                panic!("snapshot {snapshot_id} on the chain to the carrier was expired")
            });
            if snapshot_id == carrier_snapshot_id {
                assert!(
                    snapshot
                        .summary()
                        .additional_properties
                        .contains_key(CARRIER_PROPERTY),
                    "the action wrote the carrier property somewhere the plan does not read"
                );
                return;
            }
            snapshot_id = snapshot
                .parent_snapshot_id()
                .expect("the chain ran out of ancestors before reaching the carrier");
        }
    }

    #[tokio::test]
    async fn a_carrier_written_by_an_action_keeps_the_chain_to_it_reachable() {
        let catalog = new_memory_catalog().await;
        let mut table = make_v2_minimal_table_in_catalog_with_properties(
            &catalog,
            retention_properties_with_a_carrier(),
        )
        .await;

        // Below the carrier, so nothing protects it once the window moves past.
        table = append(&table, &catalog, "f0").await;
        let doomed_snapshot_id = table
            .metadata()
            .current_snapshot_id()
            .expect("the append made a snapshot current");

        table = append_carrying_the_carrier(&table, &catalog, "f1").await;
        let carrier_snapshot_id = table
            .metadata()
            .current_snapshot_id()
            .expect("the append made a snapshot current");

        // Enough appends to push the carrier out of the retention window: past this
        // point only the chain protection keeps it.
        for i in 2..MIN_SNAPSHOTS_TO_KEEP + 3 {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }

        assert!(
            table
                .metadata()
                .snapshot_by_id(doomed_snapshot_id)
                .is_none(),
            "the snapshot below the carrier is protected by neither rule and must be gone"
        );
        walk_the_chain_to_the_carrier(&table, carrier_snapshot_id);
    }

    #[tokio::test]
    async fn a_commit_that_adds_no_snapshot_still_expires_history() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, HashMap::new()).await;
        for i in 0..HISTORY_LENGTH {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }
        let base = with_properties(&table, retention_properties());
        let_the_history_age().await;

        let tx = set_properties_transaction(
            &base,
            HashMap::from([("example.key".to_string(), "value".to_string())]),
        );
        let updates = updates_of_a_commit_against(&base, tx).await;

        assert!(
            !updates
                .iter()
                .any(|update| matches!(update, TableUpdate::AddSnapshot { .. })),
            "the point of this commit is that it writes no snapshot of its own"
        );
        assert_eq!(
            expired_ids(&updates),
            ids_past_the_window(&base),
            "expiration rides on every commit, not only on the ones that append"
        );
    }

    #[tokio::test]
    async fn enabling_retention_expires_history_in_the_same_commit() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, HashMap::new()).await;
        for i in 0..HISTORY_LENGTH {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }
        let_the_history_age().await;

        // The policy is resolved from the local projection of the commit — the base
        // with this commit's own `SetProperties` already applied — so the commit that
        // switches expiration on is the one that expires, not the one after it.
        let tx = set_properties_transaction(&table, retention_properties());
        let updates = updates_of_a_commit_against(&table, tx).await;

        assert_eq!(expired_ids(&updates), ids_past_the_window(&table));
    }

    #[tokio::test]
    async fn disabling_retention_stops_expiration_in_the_same_commit() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, HashMap::new()).await;
        for i in 0..HISTORY_LENGTH {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }
        let base = with_properties(&table, retention_properties());
        let_the_history_age().await;

        // The mirror of the test above: the switch takes effect on the commit that
        // flips it, so a table being taken off expiration loses nothing on the way out.
        let tx = set_properties_transaction(
            &base,
            HashMap::from([(
                TableProperties::PROPERTY_HISTORY_EXPIRE_ENABLED.to_string(),
                "false".to_string(),
            )]),
        );
        let updates = updates_of_a_commit_against(&base, tx).await;

        assert!(
            expired_ids(&updates).is_empty(),
            "the history is past the window, and only the disabling flag keeps it"
        );
    }

    #[tokio::test]
    async fn an_invalid_policy_fails_the_commit_without_reaching_the_catalog() {
        let catalog = new_memory_catalog().await;
        let table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, HashMap::new()).await;
        let mut properties = retention_properties();
        properties.insert(
            TableProperties::PROPERTY_MIN_SNAPSHOTS_TO_KEEP.to_string(),
            "0".to_string(),
        );
        let base = with_properties(&table, properties);

        let mut mock_catalog = MockCatalog::new();
        let loaded_table = base.clone();
        mock_catalog.expect_load_table().returning_st(move |_| {
            let loaded_table = loaded_table.clone();
            Box::pin(async move { Ok(loaded_table) })
        });
        mock_catalog.expect_update_table().times(0);

        let tx = set_properties_transaction(
            &base,
            HashMap::from([("example.key".to_string(), "value".to_string())]),
        );
        let error = tx.commit(&mock_catalog).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains(TableProperties::PROPERTY_MIN_SNAPSHOTS_TO_KEEP),
            "the commit must fail on the policy, not on something the action did: {error}"
        );
        assert!(
            !error.retryable(),
            "a policy the table cannot parse is not going to parse on the next attempt"
        );
    }

    #[tokio::test]
    async fn a_retry_replans_against_the_refreshed_table() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, HashMap::new()).await;

        // Two histories of the same table: one inside the retention window, one past
        // it. The catalog hands out the short one first and the long one on the retry,
        // standing in for a writer that raced ahead in between.
        for i in 0..MIN_SNAPSHOTS_TO_KEEP {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }
        let short_history = with_properties(&table, retention_properties_with_fast_retries());
        for i in MIN_SNAPSHOTS_TO_KEEP..HISTORY_LENGTH {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }
        let long_history = with_properties(&table, retention_properties_with_fast_retries());
        let_the_history_age().await;

        let captured_updates: Arc<Mutex<Vec<Vec<TableUpdate>>>> = Arc::new(Mutex::new(Vec::new()));
        let mut mock_catalog = MockCatalog::new();

        let loads = AtomicUsize::new(0);
        let (first_base, second_base) = (short_history.clone(), long_history.clone());
        mock_catalog.expect_load_table().returning_st(move |_| {
            let base = if loads.fetch_add(1, Ordering::SeqCst) == 0 {
                first_base.clone()
            } else {
                second_base.clone()
            };
            Box::pin(async move { Ok(base) })
        });

        let sink = Arc::clone(&captured_updates);
        let attempts = AtomicUsize::new(0);
        let committed_table = long_history.clone();
        mock_catalog
            .expect_update_table()
            .times(2)
            .returning_st(move |mut commit| {
                sink.lock().unwrap().push(commit.take_updates());
                if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Box::pin(async move {
                        Err(
                            Error::new(ErrorKind::CatalogCommitConflicts, "Commit conflict")
                                .with_retryable(true),
                        )
                    });
                }
                let committed_table = committed_table.clone();
                Box::pin(async move { Ok(committed_table) })
            });

        let tx = set_properties_transaction(
            &short_history,
            HashMap::from([("example.key".to_string(), "value".to_string())]),
        );
        tx.commit(&mock_catalog).await.unwrap();

        let attempts = captured_updates.lock().unwrap();
        assert_eq!(attempts.len(), 2);
        assert!(
            expired_ids(&attempts[0]).is_empty(),
            "the base of the first attempt is inside the window, so nothing was eligible"
        );
        assert_eq!(
            expired_ids(&attempts[1]),
            ids_past_the_window(&long_history),
            "the retry must plan against the table it reloaded, not against the stale base"
        );
    }

    #[tokio::test]
    async fn expiration_follows_a_carrier_inherited_by_compaction() {
        let catalog = new_memory_catalog().await;
        let mut table = make_v2_minimal_table_in_catalog_with_properties(
            &catalog,
            retention_properties_with_a_carrier(),
        )
        .await;

        table = append(&table, &catalog, "f0").await;
        let doomed_snapshot_id = table.metadata().current_snapshot_id().unwrap();

        table = append_carrying_the_carrier(&table, &catalog, "f1").await;
        let superseded_carrier_snapshot_id = table.metadata().current_snapshot_id().unwrap();

        // Compaction is the only way the carrier moves forward: a `replace` snapshot
        // drops summary properties, and the action carries this one over from the
        // snapshot it supersedes — the current snapshot of its base.
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![data_file("compacted")])
            .delete_files(vec![data_file("f1")])
            .inherit_summary_property(CARRIER_PROPERTY);
        table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();
        let carrier_snapshot_id = table.metadata().current_snapshot_id().unwrap();

        // Past this point the compaction snapshot is out of the retention window, so
        // only its being the most recent carrier keeps it and the chain to it.
        for i in 0..MIN_SNAPSHOTS_TO_KEEP {
            table = append(&table, &catalog, &format!("g{i}")).await;
        }

        assert!(
            table
                .metadata()
                .snapshot_by_id(doomed_snapshot_id)
                .is_none(),
            "the snapshot below the carrier is protected by neither rule and must be gone"
        );
        assert!(
            table
                .metadata()
                .snapshot_by_id(superseded_carrier_snapshot_id)
                .is_none(),
            "protection follows the newest carrier: the one compaction superseded \
             holds nothing back"
        );
        walk_the_chain_to_the_carrier(&table, carrier_snapshot_id);
    }

    #[tokio::test]
    async fn an_empty_transaction_expires_nothing() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, HashMap::new()).await;
        for i in 0..HISTORY_LENGTH {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }
        let base = with_properties(&table, retention_properties());

        // Expiration rides on commits and only on commits: a transaction with no
        // action returns before it reaches the catalog at all, so a history past its
        // window cannot be brought back inside it this way.
        let mut mock_catalog = MockCatalog::new();
        mock_catalog.expect_load_table().times(0);
        mock_catalog.expect_update_table().times(0);

        let committed = Transaction::new(&base).commit(&mock_catalog).await.unwrap();
        assert_eq!(
            committed.metadata().snapshots().len(),
            HISTORY_LENGTH,
            "the table is returned as it was, history included"
        );
    }

    #[tokio::test]
    async fn a_table_without_the_policy_keeps_every_snapshot() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, HashMap::new()).await;

        for i in 0..HISTORY_LENGTH {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }

        assert_eq!(table.metadata().snapshots().len(), HISTORY_LENGTH);
    }

    #[tokio::test]
    async fn statistics_of_an_expired_snapshot_are_removed_with_it() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, retention_properties())
                .await;

        table = append(&table, &catalog, "f0").await;
        let doomed_snapshot_id = table.metadata().current_snapshot_id().unwrap();

        let tx = Transaction::new(&table);
        table = tx
            .update_statistics()
            .set_statistics(statistics_file(doomed_snapshot_id))
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
        assert!(
            table
                .metadata()
                .statistics_for_snapshot(doomed_snapshot_id)
                .is_some(),
            "the statistics file must be in place before the snapshot expires"
        );

        for i in 1..6 {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }

        assert!(
            table
                .metadata()
                .snapshot_by_id(doomed_snapshot_id)
                .is_none(),
            "the snapshot should have left the window by now"
        );
        assert!(
            table
                .metadata()
                .statistics_for_snapshot(doomed_snapshot_id)
                .is_none(),
            "statistics keyed by an expired snapshot would be unreachable garbage"
        );
    }

    #[tokio::test]
    async fn expiration_without_statistics_emits_no_statistics_updates() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, retention_properties())
                .await;
        for i in 0..MIN_SNAPSHOTS_TO_KEEP + 2 {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }
        let_the_history_age().await;

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![data_file("last")]);
        let updates = updates_of_a_commit_against(&table, action.apply(tx).unwrap()).await;

        assert!(
            updates
                .iter()
                .any(|update| matches!(update, TableUpdate::RemoveSnapshots { .. })),
            "the history is past the window, so this commit must expire something"
        );
        assert!(
            !updates.iter().any(|update| matches!(
                update,
                TableUpdate::RemoveStatistics { .. }
                    | TableUpdate::RemovePartitionStatistics { .. }
            )),
            "the table has no statistics, so no statistics updates may be emitted"
        );
    }

    #[tokio::test]
    async fn remove_snapshots_is_the_last_update_of_the_commit() {
        let catalog = new_memory_catalog().await;
        let mut table =
            make_v2_minimal_table_in_catalog_with_properties(&catalog, retention_properties())
                .await;
        for i in 0..MIN_SNAPSHOTS_TO_KEEP + 2 {
            table = append(&table, &catalog, &format!("f{i}")).await;
        }
        let_the_history_age().await;

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![data_file("last")]);
        let updates = updates_of_a_commit_against(&table, action.apply(tx).unwrap()).await;

        // A contract for the catalog rather than a guard against a live bug: every
        // reference target is protected here, so `SetSnapshotRef` could not name an
        // expired snapshot even if the updates were reordered. A catalog applying
        // them in sequence is free to validate more strictly than that, and this is
        // the order it is promised.
        let position_of = |wanted: fn(&TableUpdate) -> bool| {
            updates
                .iter()
                .position(wanted)
                .expect("the commit appends a snapshot to the branch and expires history")
        };
        let expiration =
            position_of(|update| matches!(update, TableUpdate::RemoveSnapshots { .. }));
        assert_eq!(
            expiration,
            updates.len() - 1,
            "expiration is what the commit carries last"
        );
        assert!(
            position_of(|update| matches!(update, TableUpdate::AddSnapshot { .. })) < expiration
                && position_of(|update| matches!(update, TableUpdate::SetSnapshotRef { .. }))
                    < expiration
        );
    }
}
