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

//! Parallel prefetch of the split block bloom filters (SBBF) read by the
//! bloom filter phase of the `ArrowReader` pipeline.
//!
//! `FileScanTaskReader::filter_row_groups_by_bloom_filter` reads one SBBF at a
//! time, one round trip per relevant column per row group (apache/iceberg-rust#3191).
//! This module fetches the SBBFs of all candidate row groups up front, in
//! parallel, and serves that function's reads from memory.
//!
//! Invariants:
//! - `filter_row_groups_by_bloom_filter` is not changed. It still issues its own
//!   reads; a read that exactly matches a prefetched SBBF range is answered from
//!   memory, and any other read goes to the file as before.
//! - An SBBF that could not be fetched or parsed is answered with an error, which
//!   `filter_row_groups_by_bloom_filter` treats as a missing filter, so that column
//!   cannot prune its row group. Such SBBFs are counted in [`BloomFilterMetrics::read_errors`] and
//!   reported with one `tracing::warn!` per data file.
//! - Only byte-adjacent SBBFs are merged into one request, so no row group data
//!   is fetched along with them.
//!
//! Limitation: a column chunk without `bloom_filter_length` is not prefetched.
//! `filter_row_groups_by_bloom_filter` reads it serially, and its read errors are
//! only visible as that function's `debug!` logs.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bytes::Bytes;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::bloom_filter::Sbbf;
use parquet::file::metadata::ParquetMetaData;

use super::{ArrowFileReader, ParquetReadOptions};
use crate::arrow::scan_metrics::BloomFilterMetrics;
use crate::expr::BoundPredicate;
use crate::expr::visitors::bloom_filter_evaluator::collect_bloom_filter_field_ids;
use crate::io::{FileMetadata, FileRead};
use crate::{Error, ErrorKind, Result};

/// Prefetches the SBBFs of one data file for the bloom filter phase.
///
/// Does nothing when bloom filter pruning is disabled.
pub(super) struct BloomFilterPrefetcher {
    /// `None` when bloom filter pruning is disabled; every call is then a no-op.
    enabled: Option<EnabledBloomFilterPrefetcher>,
}

impl BloomFilterPrefetcher {
    /// Wraps the reader of a data file so that prefetched SBBFs are served from
    /// memory. Returns `reader` unchanged when `bloom_filter_enabled` is false.
    pub(super) fn attach_to_reader(
        reader: ArrowFileReader,
        bloom_filter_enabled: bool,
        data_file_path: &str,
        metrics: &BloomFilterMetrics,
    ) -> (ArrowFileReader, Self) {
        if !bloom_filter_enabled {
            return (reader, Self { enabled: None });
        }

        let (file_metadata, parquet_read_options, file_read) = reader.into_parts();
        let file_read: Arc<dyn FileRead> = Arc::from(file_read);
        let cache = Arc::new(BloomFilterCache::default());

        let reader = ArrowFileReader::new(
            FileMetadata {
                size: file_metadata.size,
            },
            Box::new(BloomFilterCachingRead {
                inner: Arc::clone(&file_read),
                cache: Arc::clone(&cache),
            }),
        )
        .with_parquet_read_options(parquet_read_options);

        let enabled = EnabledBloomFilterPrefetcher {
            fetcher: SbbfFetcher {
                file_read,
                file_size: file_metadata.size,
                parquet_read_options,
            },
            cache,
            data_file_path: data_file_path.to_string(),
            metrics: metrics.clone(),
        };

        (reader, Self {
            enabled: Some(enabled),
        })
    }

    /// Fetches, in parallel, the SBBFs that the bloom filter phase will read for
    /// `candidate_row_groups`, and keeps them in memory until the returned
    /// [`BloomFilterPrefetchGuard`] is dropped.
    ///
    /// Columns are selected exactly as `filter_row_groups_by_bloom_filter` selects
    /// them. When that selection fails, nothing is prefetched: the filter reports
    /// the same error itself.
    pub(super) async fn prefetch(
        &self,
        predicate: &BoundPredicate,
        metadata: &ParquetMetaData,
        candidate_row_groups: &[usize],
        field_id_map: &HashMap<i32, usize>,
    ) -> BloomFilterPrefetchGuard {
        let Some(enabled) = &self.enabled else {
            return BloomFilterPrefetchGuard { cache: None };
        };

        let Ok(sbbf_ranges) =
            Self::plan_sbbf_ranges(predicate, metadata, candidate_row_groups, field_id_map)
        else {
            return BloomFilterPrefetchGuard { cache: None };
        };
        if sbbf_ranges.is_empty() {
            return BloomFilterPrefetchGuard { cache: None };
        }

        // TODO(med): keeps the SBBFs of every candidate row group of a data file in memory at once, for up to concurrency_limit_data_files files; bound it by fetching in windows
        let fetched = enabled.fetcher.fetch(&sbbf_ranges).await;
        enabled.report_unreadable(&fetched);
        enabled.cache.insert_all(fetched.entries);

        BloomFilterPrefetchGuard {
            cache: Some(Arc::clone(&enabled.cache)),
        }
    }

    /// Byte ranges of the SBBFs that `filter_row_groups_by_bloom_filter` reads for
    /// `candidate_row_groups`, in row group order.
    ///
    /// A column chunk without `bloom_filter_offset` has no SBBF and is skipped, as the
    /// filter skips it. A column chunk without `bloom_filter_length` is also skipped:
    /// the filter reads it on its own.
    fn plan_sbbf_ranges(
        predicate: &BoundPredicate,
        metadata: &ParquetMetaData,
        candidate_row_groups: &[usize],
        field_id_map: &HashMap<i32, usize>,
    ) -> Result<Vec<Range<u64>>> {
        let column_indices: Vec<usize> = collect_bloom_filter_field_ids(predicate)?
            .into_iter()
            .filter_map(|field_id| field_id_map.get(&field_id).copied())
            .collect();

        let mut sbbf_ranges = Vec::with_capacity(candidate_row_groups.len() * column_indices.len());
        for &row_group_idx in candidate_row_groups {
            let row_group = metadata.row_group(row_group_idx);
            for &column_idx in &column_indices {
                let column = row_group.column(column_idx);
                let (Some(offset), Some(length)) =
                    (column.bloom_filter_offset(), column.bloom_filter_length())
                else {
                    continue;
                };
                let (Ok(offset), Ok(length)) = (u64::try_from(offset), u64::try_from(length))
                else {
                    continue;
                };
                sbbf_ranges.push(offset..offset + length);
            }
        }

        Ok(sbbf_ranges)
    }
}

/// State of a [`BloomFilterPrefetcher`] with bloom filter pruning enabled.
struct EnabledBloomFilterPrefetcher {
    fetcher: SbbfFetcher,
    cache: Arc<BloomFilterCache>,
    data_file_path: String,
    metrics: BloomFilterMetrics,
}

impl EnabledBloomFilterPrefetcher {
    /// Counts the unreadable SBBFs of this file and reports them with one warning.
    fn report_unreadable(&self, fetched: &FetchedBloomFilters) {
        let unreadable = fetched.unreadable_count();
        if unreadable == 0 {
            return;
        }

        self.metrics.add_read_errors(unreadable as u64);
        tracing::warn!(
            data_file_path = %self.data_file_path,
            unreadable_bloom_filters = unreadable,
            first_error = fetched.first_error.as_deref().unwrap_or_default(),
            "Bloom filters could not be read; their columns cannot prune row groups"
        );
    }
}

/// Fetches SBBFs of one data file.
struct SbbfFetcher {
    /// The data file reader shared with the [`BloomFilterCachingRead`] of the scan.
    file_read: Arc<dyn FileRead>,
    file_size: u64,
    parquet_read_options: ParquetReadOptions,
}

impl SbbfFetcher {
    /// Fetches `sbbf_ranges`, merging only byte-adjacent ranges, and checks that
    /// every range holds a complete SBBF.
    async fn fetch(&self, sbbf_ranges: &[Range<u64>]) -> FetchedBloomFilters {
        let mut reader = ArrowFileReader::new(
            FileMetadata {
                size: self.file_size,
            },
            Box::new(Arc::clone(&self.file_read)),
        )
        .with_parquet_read_options(ParquetReadOptions {
            range_coalesce_bytes: 0,
            ..self.parquet_read_options
        });

        match reader.get_byte_ranges(sbbf_ranges.to_vec()).await {
            Ok(fetched) => {
                let mut first_error = None;
                let entries = sbbf_ranges
                    .iter()
                    .cloned()
                    .zip(fetched)
                    .map(|(range, bytes)| match Sbbf::from_bytes(&bytes) {
                        Ok(_) => (range, CachedBloomFilter::Fetched(bytes)),
                        Err(err) => {
                            first_error.get_or_insert_with(|| err.to_string());
                            (range, CachedBloomFilter::Unreadable)
                        }
                    })
                    .collect();
                FetchedBloomFilters {
                    entries,
                    first_error,
                }
            }
            // The first failed request cancels the others, so none of the ranges
            // of this file can be trusted.
            Err(err) => FetchedBloomFilters {
                entries: sbbf_ranges
                    .iter()
                    .map(|range| (range.clone(), CachedBloomFilter::Unreadable))
                    .collect(),
                first_error: Some(err.to_string()),
            },
        }
    }
}

/// Result of fetching the SBBFs of one data file.
struct FetchedBloomFilters {
    entries: Vec<(Range<u64>, CachedBloomFilter)>,
    /// The first fetch or parse error, reported in the warning.
    first_error: Option<String>,
}

impl FetchedBloomFilters {
    /// Number of fetched ranges that do not hold a complete SBBF.
    fn unreadable_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|(_, entry)| matches!(entry, CachedBloomFilter::Unreadable))
            .count()
    }
}

/// A prefetched SBBF range.
enum CachedBloomFilter {
    /// The bytes hold a complete SBBF.
    Fetched(Bytes),
    /// The range could not be fetched, or its bytes are not a complete SBBF.
    Unreadable,
}

/// Prefetched SBBFs of one data file, keyed by their exact byte range.
#[derive(Default)]
struct BloomFilterCache {
    entries: Mutex<HashMap<Range<u64>, CachedBloomFilter>>,
}

impl BloomFilterCache {
    fn insert_all(&self, entries: Vec<(Range<u64>, CachedBloomFilter)>) {
        self.lock().extend(entries);
    }

    /// Removes and returns the entry for exactly `range`. The bloom filter phase
    /// reads every SBBF once, so an entry is never needed twice.
    fn take(&self, range: &Range<u64>) -> Option<CachedBloomFilter> {
        self.lock().remove(range)
    }

    fn clear(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<Range<u64>, CachedBloomFilter>> {
        // The map stays consistent even if a holder panicked: every operation
        // above is a single map call.
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Serves reads of prefetched SBBF ranges from memory and passes every other
/// read to the data file.
struct BloomFilterCachingRead {
    inner: Arc<dyn FileRead>,
    cache: Arc<BloomFilterCache>,
}

#[async_trait::async_trait]
impl FileRead for BloomFilterCachingRead {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        match self.cache.take(&range) {
            Some(CachedBloomFilter::Fetched(bytes)) => Ok(bytes),
            Some(CachedBloomFilter::Unreadable) => Err(Error::new(
                ErrorKind::Unexpected,
                format!("Bloom filter at bytes {range:?} could not be read during prefetch"),
            )),
            None => self.inner.read(range).await,
        }
    }
}

/// Keeps prefetched SBBFs in memory while alive. Dropping it releases the SBBFs
/// that the bloom filter phase did not read.
#[must_use = "dropping the guard releases the prefetched bloom filters before the bloom filter phase reads them"]
pub(super) struct BloomFilterPrefetchGuard {
    cache: Option<Arc<BloomFilterCache>>,
}

impl Drop for BloomFilterPrefetchGuard {
    fn drop(&mut self) {
        if let Some(cache) = &self.cache {
            cache.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream::BoxStream;
    use futures::{StreamExt, TryStreamExt};
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::basic::Compression;
    use parquet::file::metadata::ParquetMetaDataReader;
    use parquet::file::properties::WriterProperties;
    use parquet::schema::types::ColumnPath;
    use serde::{Deserialize, Serialize};
    use tokio::sync::{Barrier, Notify};
    use tokio::time::timeout;
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Level, Metadata, Subscriber};

    use crate::arrow::{ArrowReaderBuilder, BloomFilterMetrics};
    use crate::expr::{Bind, Predicate, Reference};
    use crate::io::{
        FileIO, FileIOBuilder, FileMetadata, FileRead, FileWrite, InputFile, MemoryStorage,
        OutputFile, Storage, StorageConfig, StorageFactory,
    };
    use crate::scan::{FileScanTask, FileScanTaskStream};
    use crate::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Schema, Type};
    use crate::{Error, ErrorKind, Result, Runtime};

    /// Upper bound for every wait; a sequential or leaked read hangs until it fires.
    const TIMEOUT: Duration = Duration::from_secs(10);
    const DATA_FILE_PATH: &str = "memory:///bloom/data.parquet";
    const ROW_GROUPS: i32 = 3;
    const ROWS_PER_GROUP: i32 = 100;
    /// Offset between the values of columns `a` and `b` in the same row.
    const B_OFFSET: i32 = 10_000;

    // ------------------------------------------------------------------------
    // Storage that observes and alters reads of SBBF bytes
    // ------------------------------------------------------------------------

    /// What a read that overlaps an SBBF does.
    #[derive(Clone, Debug, Default)]
    enum SbbfReadBehavior {
        #[default]
        PassThrough,
        Fail,
        /// Fails reads that overlap the given range; other SBBF reads pass through.
        FailOverlapping(Range<u64>),
        /// Returns zeroes of the requested length.
        Zeroes,
        /// Waits until the barrier's party count of such reads is in flight.
        Barrier(Arc<Barrier>),
        /// Never completes; flags when the read future is dropped.
        Hang(Arc<HangSignal>),
    }

    #[derive(Debug, Default)]
    struct HangSignal {
        started: Notify,
        /// Number of hanging reads started so far.
        started_count: AtomicUsize,
        dropped: AtomicBool,
    }

    struct DropFlag(Arc<HangSignal>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.dropped.store(true, Ordering::SeqCst);
        }
    }

    /// Shared state of [`BloomFilterProbeStorage`]: every read of the data file and
    /// the behavior of reads that overlap an SBBF.
    #[derive(Debug, Default)]
    struct ReadProbe {
        file_size: AtomicU64,
        sbbf_ranges: Mutex<Vec<Range<u64>>>,
        behavior: Mutex<SbbfReadBehavior>,
        reads: Mutex<Vec<Range<u64>>>,
    }

    impl ReadProbe {
        fn set_behavior(&self, behavior: SbbfReadBehavior) {
            *self.behavior.lock().unwrap() = behavior;
        }

        /// Whether `range` overlaps an SBBF. Footer reads (ending at the end of the
        /// file) are excluded: a small file's footer prefetch covers every byte.
        fn overlaps_sbbf(&self, range: &Range<u64>) -> bool {
            range.end != self.file_size.load(Ordering::SeqCst)
                && self
                    .sbbf_ranges
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|sbbf| range.start < sbbf.end && sbbf.start < range.end)
        }

        /// Takes every read recorded so far, sorted by start offset.
        fn take_reads(&self) -> Vec<Range<u64>> {
            let mut reads = std::mem::take(&mut *self.reads.lock().unwrap());
            reads.sort_by_key(|range| (range.start, range.end));
            reads
        }

        /// Takes the recorded reads that overlap an SBBF, sorted by start offset.
        fn take_sbbf_reads(&self) -> Vec<Range<u64>> {
            self.take_reads()
                .into_iter()
                .filter(|range| self.overlaps_sbbf(range))
                .collect()
        }
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    struct BloomFilterProbeStorage {
        #[serde(skip)]
        inner: MemoryStorage,
        #[serde(skip)]
        probe: Arc<ReadProbe>,
    }

    #[async_trait]
    #[typetag::serde]
    impl Storage for BloomFilterProbeStorage {
        async fn exists(&self, path: &str) -> Result<bool> {
            self.inner.exists(path).await
        }

        async fn metadata(&self, path: &str) -> Result<FileMetadata> {
            self.inner.metadata(path).await
        }

        async fn read(&self, path: &str) -> Result<Bytes> {
            self.inner.read(path).await
        }

        async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
            Ok(Box::new(ProbeFileRead {
                inner: self.inner.reader(path).await?,
                probe: Arc::clone(&self.probe),
            }))
        }

        async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
            self.inner.write(path, bs).await
        }

        async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
            self.inner.writer(path).await
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
            // Route reads through THIS storage so the probe observes them.
            Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
        }

        fn new_output(&self, path: &str) -> Result<OutputFile> {
            Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
        }
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    struct BloomFilterProbeStorageFactory {
        #[serde(skip)]
        storage: BloomFilterProbeStorage,
    }

    #[typetag::serde]
    impl StorageFactory for BloomFilterProbeStorageFactory {
        fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
            Ok(Arc::new(self.storage.clone()))
        }
    }

    struct ProbeFileRead {
        inner: Box<dyn FileRead>,
        probe: Arc<ReadProbe>,
    }

    #[async_trait]
    impl FileRead for ProbeFileRead {
        async fn read(&self, range: Range<u64>) -> Result<Bytes> {
            self.probe.reads.lock().unwrap().push(range.clone());
            if !self.probe.overlaps_sbbf(&range) {
                return self.inner.read(range).await;
            }

            let behavior = self.probe.behavior.lock().unwrap().clone();
            match behavior {
                SbbfReadBehavior::PassThrough => self.inner.read(range).await,
                SbbfReadBehavior::Fail => Err(Error::new(
                    ErrorKind::Unexpected,
                    "injected bloom filter read failure",
                )),
                SbbfReadBehavior::FailOverlapping(failing) => {
                    if range.start < failing.end && failing.start < range.end {
                        Err(Error::new(
                            ErrorKind::Unexpected,
                            "injected bloom filter read failure",
                        ))
                    } else {
                        self.inner.read(range).await
                    }
                }
                SbbfReadBehavior::Zeroes => {
                    Ok(Bytes::from(vec![0; (range.end - range.start) as usize]))
                }
                SbbfReadBehavior::Barrier(barrier) => {
                    barrier.wait().await;
                    self.inner.read(range).await
                }
                SbbfReadBehavior::Hang(signal) => {
                    let _flag = DropFlag(Arc::clone(&signal));
                    signal.started_count.fetch_add(1, Ordering::SeqCst);
                    signal.started.notify_one();
                    std::future::pending::<()>().await;
                    unreachable!("a pending future never completes")
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Data file fixture
    // ------------------------------------------------------------------------

    /// How values are spread across row groups.
    #[derive(Clone, Copy)]
    enum Layout {
        /// Row group `g` holds `a = 4 * i + g`, so every row group's min/max covers
        /// the whole domain, and values `≡ 3 (mod 4)` are in no row group.
        Interleaved,
        /// Row group `g` holds `a = 1000 * g + i`, so min/max ranges do not overlap.
        Disjoint,
    }

    impl Layout {
        fn a(self, group: i32, i: i32) -> i32 {
            match self {
                Layout::Interleaved => 4 * i + group,
                Layout::Disjoint => 1000 * group + i,
            }
        }
    }

    /// SBBF byte ranges of the bloom filter columns `a` and `b` in one row group.
    struct RowGroupSbbfRanges {
        a: Range<u64>,
        b: Range<u64>,
    }

    /// A Parquet data file with 3 row groups and columns `a` (id 1) and `b` (id 2)
    /// with bloom filters and `c` (id 3) without, behind a [`ReadProbe`].
    struct Fixture {
        file_io: FileIO,
        probe: Arc<ReadProbe>,
        schema: Arc<Schema>,
        file_size: u64,
        /// Computed from the footer, independently of the prefetch code.
        sbbf_ranges: Vec<RowGroupSbbfRanges>,
    }

    fn field_with_id(name: &str, id: i32) -> Field {
        Field::new(name, DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    }

    impl Fixture {
        async fn new(layout: Layout) -> Self {
            let storage = BloomFilterProbeStorage::default();
            let probe = Arc::clone(&storage.probe);
            let file_io =
                FileIOBuilder::new(Arc::new(BloomFilterProbeStorageFactory { storage })).build();

            let schema = Arc::new(
                Schema::builder()
                    .with_schema_id(1)
                    .with_fields(vec![
                        NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::required(2, "b", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::required(3, "c", Type::Primitive(PrimitiveType::Int)).into(),
                    ])
                    .build()
                    .unwrap(),
            );
            let arrow_schema = Arc::new(ArrowSchema::new(vec![
                field_with_id("a", 1),
                field_with_id("b", 2),
                field_with_id("c", 3),
            ]));

            let mut props = WriterProperties::builder().set_compression(Compression::UNCOMPRESSED);
            for column in ["a", "b"] {
                props = props
                    .set_column_bloom_filter_ndv(ColumnPath::from(column), ROWS_PER_GROUP as u64)
                    .set_column_bloom_filter_fpp(ColumnPath::from(column), 0.001);
            }

            let mut buffer = Vec::new();
            let mut writer =
                ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), Some(props.build()))
                    .unwrap();
            for group in 0..ROW_GROUPS {
                let a: Vec<i32> = (0..ROWS_PER_GROUP).map(|i| layout.a(group, i)).collect();
                let b: Vec<i32> = a.iter().map(|value| value + B_OFFSET).collect();
                let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
                    Arc::new(Int32Array::from(a.clone())),
                    Arc::new(Int32Array::from(b)),
                    Arc::new(Int32Array::from(a)),
                ])
                .unwrap();
                writer.write(&batch).unwrap();
                // Force a row group boundary per batch.
                writer.flush().unwrap();
            }
            writer.close().unwrap();

            let bytes = Bytes::from(buffer);
            let metadata = ParquetMetaDataReader::new()
                .parse_and_finish(&bytes)
                .unwrap();
            assert_eq!(metadata.num_row_groups(), ROW_GROUPS as usize);
            let sbbf_range = |group: usize, column: usize| {
                let chunk = metadata.row_group(group).column(column);
                let offset = chunk.bloom_filter_offset().unwrap() as u64;
                offset..offset + chunk.bloom_filter_length().unwrap() as u64
            };
            let sbbf_ranges: Vec<RowGroupSbbfRanges> = (0..metadata.num_row_groups())
                .map(|group| {
                    assert!(
                        metadata
                            .row_group(group)
                            .column(2)
                            .bloom_filter_offset()
                            .is_none(),
                        "column c must have no bloom filter"
                    );
                    RowGroupSbbfRanges {
                        a: sbbf_range(group, 0),
                        b: sbbf_range(group, 1),
                    }
                })
                .collect();

            let file_size = bytes.len() as u64;
            probe.file_size.store(file_size, Ordering::SeqCst);
            *probe.sbbf_ranges.lock().unwrap() = sbbf_ranges
                .iter()
                .flat_map(|ranges| [ranges.a.clone(), ranges.b.clone()])
                .collect();
            file_io
                .new_output(DATA_FILE_PATH)
                .unwrap()
                .write(bytes)
                .await
                .unwrap();

            Self {
                file_io,
                probe,
                schema,
                file_size,
                sbbf_ranges,
            }
        }

        /// A task that reads the file as a table with `schema`.
        fn task_with_schema(&self, schema: Arc<Schema>, predicate: &Predicate) -> FileScanTask {
            FileScanTask::builder()
                .with_file_size_in_bytes(self.file_size)
                .with_start(0)
                .with_length(0)
                .with_data_file_path(DATA_FILE_PATH.to_string())
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2, 3])
                .with_predicate(Some(predicate.clone().bind(schema, true).unwrap()))
                .with_case_sensitive(false)
                .build()
        }

        fn task(&self, predicate: &Predicate) -> FileScanTask {
            self.task_with_schema(self.schema.clone(), predicate)
        }

        fn tasks(&self, predicate: &Predicate) -> FileScanTaskStream {
            Box::pin(futures::stream::iter(vec![Ok(self.task(predicate))]))
        }

        /// Reads the file and returns its rows as one batch (`None` for no rows)
        /// and the bloom filter counters of the scan.
        async fn read(
            &self,
            predicate: &Predicate,
            bloom_filter_enabled: bool,
        ) -> (Option<RecordBatch>, BloomFilterMetrics) {
            self.read_tasks(self.tasks(predicate), bloom_filter_enabled)
                .await
        }

        /// Reads `tasks` in one scan, like [`Self::read`].
        async fn read_tasks(
            &self,
            tasks: FileScanTaskStream,
            bloom_filter_enabled: bool,
        ) -> (Option<RecordBatch>, BloomFilterMetrics) {
            let result = ArrowReaderBuilder::new(self.file_io.clone(), Runtime::current())
                .with_bloom_filter_enabled(bloom_filter_enabled)
                .build()
                .read(tasks)
                .unwrap();
            let metrics = result.metrics().bloom_filter().clone();
            let batches: Vec<RecordBatch> = timeout(TIMEOUT, result.stream().try_collect())
                .await
                .expect("scan timed out")
                .unwrap();
            (collapse(&batches), metrics)
        }

        /// Reads with bloom filter pruning on and off, asserts the rows are
        /// identical, and returns the counters of the pruned read.
        async fn read_and_compare(&self, predicate: &Predicate) -> BloomFilterMetrics {
            let (off, _) = self.read(predicate, false).await;
            self.probe.take_reads();
            let (on, metrics) = self.read(predicate, true).await;
            assert_eq!(on, off, "bloom filter pruning changed the rows returned");
            metrics
        }
    }

    fn collapse(batches: &[RecordBatch]) -> Option<RecordBatch> {
        let first = batches.first()?;
        Some(arrow_select::concat::concat_batches(&first.schema(), batches).unwrap())
    }

    fn a_equals(value: i32) -> Predicate {
        Reference::new("a").equal_to(Datum::int(value))
    }

    /// `a = value AND b = value + B_OFFSET`: both bloom filter columns of the rows
    /// with `a = value`.
    fn a_and_b_equal(value: i32) -> Predicate {
        a_equals(value).and(Reference::new("b").equal_to(Datum::int(value + B_OFFSET)))
    }

    /// Counts `WARN` events emitted on the current thread while installed.
    #[derive(Clone, Default)]
    struct WarnCounter(Arc<AtomicUsize>);

    impl Subscriber for WarnCounter {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }

        fn record(&self, _: &Id, _: &Record<'_>) {}

        fn record_follows_from(&self, _: &Id, _: &Id) {}

        fn event(&self, event: &Event<'_>) {
            if *event.metadata().level() == Level::WARN {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        fn enter(&self, _: &Id) {}

        fn exit(&self, _: &Id) {}
    }

    // ------------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------------

    #[tokio::test]
    async fn test_bloom_filter_prefetch_reads_candidates_in_parallel() {
        let fixture = Fixture::new(Layout::Interleaved).await;
        // Each row group's SBBFs are one request; a serial reader would wait on the
        // barrier forever with a single request in flight.
        fixture
            .probe
            .set_behavior(SbbfReadBehavior::Barrier(Arc::new(Barrier::new(
                ROW_GROUPS as usize,
            ))));

        let metrics = fixture.read_and_compare(&a_and_b_equal(41)).await;

        assert_eq!(metrics.row_groups_pruned(), 2);
        assert_eq!(metrics.row_groups_matched(), 1);
    }

    #[tokio::test]
    async fn test_bloom_filter_prefetch_respects_range_fetch_concurrency() {
        const RANGE_FETCH_CONCURRENCY: usize = 2;
        let fixture = Fixture::new(Layout::Interleaved).await;
        // One request per row group, and no request ever completes.
        assert!(fixture.sbbf_ranges.len() > RANGE_FETCH_CONCURRENCY);
        let signal = Arc::new(HangSignal::default());
        fixture
            .probe
            .set_behavior(SbbfReadBehavior::Hang(Arc::clone(&signal)));

        let mut stream = ArrowReaderBuilder::new(fixture.file_io.clone(), Runtime::current())
            .with_bloom_filter_enabled(true)
            .with_range_fetch_concurrency(RANGE_FETCH_CONCURRENCY)
            .build()
            .read(fixture.tasks(&a_and_b_equal(41)))
            .unwrap()
            .stream();

        let next = stream.next();
        let started = signal.started.notified();
        futures::pin_mut!(next, started);
        match timeout(TIMEOUT, futures::future::select(next, started)).await {
            Ok(futures::future::Either::Right(_)) => {}
            Ok(futures::future::Either::Left(_)) => {
                panic!("the scan yielded before its bloom filter read started")
            }
            Err(_) => panic!("the bloom filter read did not start"),
        }

        assert_eq!(
            signal.started_count.load(Ordering::SeqCst),
            RANGE_FETCH_CONCURRENCY
        );
    }

    #[tokio::test]
    async fn test_bloom_filter_prefetch_reads_only_sbbf_bytes() {
        let fixture = Fixture::new(Layout::Interleaved).await;
        // Row groups are far smaller than the default 1 MiB coalescing gap, so
        // default coalescing would merge SBBFs with the data between them.
        assert!(fixture.file_size < 1024 * 1024);

        fixture.read_and_compare(&a_and_b_equal(41)).await;

        let expected: Vec<Range<u64>> = fixture
            .sbbf_ranges
            .iter()
            .map(|ranges| {
                assert_eq!(
                    ranges.a.end, ranges.b.start,
                    "SBBFs of a row group are adjacent"
                );
                ranges.a.start..ranges.b.end
            })
            .collect();
        let sbbf_reads = fixture.probe.take_sbbf_reads();
        assert_eq!(sbbf_reads, expected);

        let read_bytes: u64 = sbbf_reads.iter().map(|range| range.end - range.start).sum();
        let sbbf_bytes: u64 = fixture
            .sbbf_ranges
            .iter()
            .map(|ranges| (ranges.a.end - ranges.a.start) + (ranges.b.end - ranges.b.start))
            .sum();
        assert_eq!(read_bytes, sbbf_bytes);
    }

    #[tokio::test]
    async fn test_bloom_filter_prefetch_skips_row_groups_pruned_by_min_max() {
        let fixture = Fixture::new(Layout::Disjoint).await;

        let metrics = fixture.read_and_compare(&a_and_b_equal(1005)).await;

        let row_group_1 = &fixture.sbbf_ranges[1];
        assert_eq!(fixture.probe.take_sbbf_reads(), vec![
            row_group_1.a.start..row_group_1.b.end
        ]);
        assert_eq!(metrics.row_groups_pruned(), 0);
        assert_eq!(metrics.row_groups_matched(), 1);
    }

    #[tokio::test]
    async fn test_bloom_filter_prefetch_skips_columns_without_bloom_filter() {
        let fixture = Fixture::new(Layout::Interleaved).await;
        let predicate = Reference::new("c").equal_to(Datum::int(41));

        let (off, _) = fixture.read(&predicate, false).await;
        let reads_off = fixture.probe.take_reads();
        let (on, metrics) = fixture.read(&predicate, true).await;
        let reads_on = fixture.probe.take_reads();

        assert_eq!(on, off);
        assert_eq!(reads_on, reads_off);
        assert_eq!(metrics.read_errors(), 0);
    }

    #[tokio::test]
    async fn test_bloom_filter_prefetch_skips_non_literal_predicates() {
        let fixture = Fixture::new(Layout::Interleaved).await;
        // `a` has an SBBF, but a range predicate cannot probe it.
        let predicate = Reference::new("a").less_than(Datum::int(5));

        let metrics = fixture.read_and_compare(&predicate).await;

        assert_eq!(fixture.probe.take_sbbf_reads(), Vec::<Range<u64>>::new());
        assert_eq!(
            (
                metrics.row_groups_pruned(),
                metrics.row_groups_matched(),
                metrics.read_errors()
            ),
            (0, ROW_GROUPS as u64, 0)
        );
    }

    async fn assert_unreadable_bloom_filters_keep_row_groups(behavior: SbbfReadBehavior) {
        // A failed request stops the others, so only complete fetches read every
        // merged range.
        let every_request_completes = matches!(behavior, SbbfReadBehavior::Zeroes);
        let fixture = Fixture::new(Layout::Interleaved).await;
        fixture.probe.set_behavior(behavior);
        let warnings = WarnCounter::default();
        let _subscriber = tracing::subscriber::set_default(warnings.clone());

        let metrics = fixture.read_and_compare(&a_and_b_equal(41)).await;

        assert_eq!(metrics.read_errors(), (ROW_GROUPS * 2) as u64);
        assert_eq!(metrics.row_groups_pruned(), 0);
        assert_eq!(metrics.row_groups_matched(), ROW_GROUPS as u64);
        assert_eq!(warnings.0.load(Ordering::SeqCst), 1, "one warning per file");

        // Unreadable SBBFs are not read again one by one.
        let merged: Vec<Range<u64>> = fixture
            .sbbf_ranges
            .iter()
            .map(|ranges| ranges.a.start..ranges.b.end)
            .collect();
        let sbbf_reads = fixture.probe.take_sbbf_reads();
        assert!(!sbbf_reads.is_empty());
        assert!(
            sbbf_reads.iter().all(|read| merged.contains(read)),
            "unreadable bloom filters were read again: {sbbf_reads:?}"
        );
        if every_request_completes {
            assert_eq!(sbbf_reads, merged);
        }
    }

    #[tokio::test]
    async fn test_bloom_filter_read_failure_keeps_row_groups() {
        assert_unreadable_bloom_filters_keep_row_groups(SbbfReadBehavior::Fail).await;
    }

    #[tokio::test]
    async fn test_corrupt_bloom_filter_keeps_row_groups() {
        assert_unreadable_bloom_filters_keep_row_groups(SbbfReadBehavior::Zeroes).await;
    }

    #[tokio::test]
    async fn test_single_failed_prefetch_request_disables_pruning_for_file() {
        let fixture = Fixture::new(Layout::Interleaved).await;
        // Only the request for row group 1 fails; row groups 0 and 2 could still
        // be pruned by their own SBBFs.
        let row_group_1 = &fixture.sbbf_ranges[1];
        fixture
            .probe
            .set_behavior(SbbfReadBehavior::FailOverlapping(
                row_group_1.a.start..row_group_1.b.end,
            ));
        let warnings = WarnCounter::default();
        let _subscriber = tracing::subscriber::set_default(warnings.clone());

        let metrics = fixture.read_and_compare(&a_and_b_equal(41)).await;

        assert_eq!(
            (
                metrics.row_groups_pruned(),
                metrics.row_groups_matched(),
                metrics.read_errors()
            ),
            (0, ROW_GROUPS as u64, (ROW_GROUPS * 2) as u64)
        );
        assert_eq!(warnings.0.load(Ordering::SeqCst), 1, "one warning per file");
    }

    #[tokio::test]
    async fn test_bloom_filter_counters() {
        let fixture = Fixture::new(Layout::Interleaved).await;

        let present = fixture.read_and_compare(&a_equals(41)).await;
        assert_eq!(
            (
                present.row_groups_pruned(),
                present.row_groups_matched(),
                present.read_errors()
            ),
            (2, 1, 0)
        );

        let absent = fixture.read_and_compare(&a_equals(43)).await;
        assert_eq!(
            (
                absent.row_groups_pruned(),
                absent.row_groups_matched(),
                absent.read_errors()
            ),
            (3, 0, 0)
        );

        let (_, disabled) = fixture.read(&a_equals(41), false).await;
        assert_eq!(
            (
                disabled.row_groups_pruned(),
                disabled.row_groups_matched(),
                disabled.read_errors()
            ),
            (0, 0, 0)
        );
    }

    #[tokio::test]
    async fn test_bloom_filter_disabled_by_default() {
        let fixture = Fixture::new(Layout::Interleaved).await;

        let result = ArrowReaderBuilder::new(fixture.file_io.clone(), Runtime::current())
            .build()
            .read(fixture.tasks(&a_and_b_equal(41)))
            .unwrap();
        let metrics = result.metrics().bloom_filter().clone();
        let _: Vec<RecordBatch> = timeout(TIMEOUT, result.stream().try_collect())
            .await
            .expect("scan timed out")
            .unwrap();

        assert_eq!(fixture.probe.take_sbbf_reads(), Vec::<Range<u64>>::new());
        assert_eq!(
            (
                metrics.row_groups_pruned(),
                metrics.row_groups_matched(),
                metrics.read_errors()
            ),
            (0, 0, 0)
        );
    }

    #[tokio::test]
    async fn test_bloom_filter_metrics_accumulate_across_data_files() {
        let fixture = Fixture::new(Layout::Interleaved).await;
        // The same data file twice: each task is processed as its own file.
        let read_two_files_and_compare = async |predicate: &Predicate| {
            let two_files = || -> FileScanTaskStream {
                let task = fixture.task(predicate);
                Box::pin(futures::stream::iter(vec![Ok(task.clone()), Ok(task)]))
            };
            let (off, _) = fixture.read_tasks(two_files(), false).await;
            let (on, metrics) = fixture.read_tasks(two_files(), true).await;
            assert_eq!(on, off, "bloom filter pruning changed the rows returned");
            metrics
        };

        let metrics = read_two_files_and_compare(&a_equals(41)).await;
        assert_eq!(
            (
                metrics.row_groups_pruned(),
                metrics.row_groups_matched(),
                metrics.read_errors()
            ),
            (4, 2, 0)
        );

        fixture.probe.set_behavior(SbbfReadBehavior::Fail);
        let warnings = WarnCounter::default();
        let _subscriber = tracing::subscriber::set_default(warnings.clone());

        let metrics = read_two_files_and_compare(&a_and_b_equal(41)).await;
        assert_eq!(
            (
                metrics.row_groups_pruned(),
                metrics.row_groups_matched(),
                metrics.read_errors()
            ),
            (0, 6, 12)
        );
        assert_eq!(warnings.0.load(Ordering::SeqCst), 2, "one warning per file");
    }

    #[tokio::test]
    async fn test_bloom_filter_phase_skips_columns_absent_from_data_file() {
        let fixture = Fixture::new(Layout::Interleaved).await;
        // The table gained column `d` after the data file was written.
        let mut fields = fixture.schema.as_struct().fields().to_vec();
        fields.push(NestedField::optional(4, "d", Type::Primitive(PrimitiveType::Int)).into());
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(2)
                .with_fields(fields)
                .build()
                .unwrap(),
        );
        let predicate = a_equals(41).or(Reference::new("d").equal_to(Datum::int(5)));
        let tasks = || -> FileScanTaskStream {
            Box::pin(futures::stream::iter(vec![Ok(
                fixture.task_with_schema(schema.clone(), &predicate)
            )]))
        };

        let (off, _) = fixture.read_tasks(tasks(), false).await;
        fixture.probe.take_reads();
        let (on, metrics) = fixture.read_tasks(tasks(), true).await;

        assert_eq!(on, off, "bloom filter pruning changed the rows returned");
        // `d = 5` is false for a column the file does not have.
        assert_eq!(on.map(|batch| batch.num_rows()), Some(1));
        // Only `a` has an SBBF to read: `b` is not in the predicate, `d` not in the file.
        let a_ranges: Vec<Range<u64>> = fixture
            .sbbf_ranges
            .iter()
            .map(|ranges| ranges.a.clone())
            .collect();
        assert_eq!(fixture.probe.take_sbbf_reads(), a_ranges);
        assert_eq!(
            (
                metrics.row_groups_pruned(),
                metrics.row_groups_matched(),
                metrics.read_errors()
            ),
            (0, ROW_GROUPS as u64, 0)
        );
    }

    #[tokio::test]
    async fn test_dropping_scan_stream_cancels_bloom_filter_reads() {
        let fixture = Fixture::new(Layout::Interleaved).await;
        let signal = Arc::new(HangSignal::default());
        fixture
            .probe
            .set_behavior(SbbfReadBehavior::Hang(Arc::clone(&signal)));

        let mut stream = ArrowReaderBuilder::new(fixture.file_io.clone(), Runtime::current())
            .with_bloom_filter_enabled(true)
            .build()
            .read(fixture.tasks(&a_equals(41)))
            .unwrap()
            .stream();

        {
            let next = stream.next();
            let started = signal.started.notified();
            futures::pin_mut!(next, started);
            match timeout(TIMEOUT, futures::future::select(next, started)).await {
                Ok(futures::future::Either::Right(_)) => {}
                Ok(futures::future::Either::Left(_)) => {
                    panic!("the scan yielded before its bloom filter read started")
                }
                Err(_) => panic!("the bloom filter read did not start"),
            }
        }
        assert!(!signal.dropped.load(Ordering::SeqCst));

        drop(stream);

        assert!(
            signal.dropped.load(Ordering::SeqCst),
            "dropping the scan stream must drop its bloom filter read"
        );
    }
}
