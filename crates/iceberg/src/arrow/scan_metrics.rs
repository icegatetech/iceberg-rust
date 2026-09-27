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

//! Scan metrics and I/O counting for Parquet data file reads.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;

use crate::error::Result;
use crate::io::FileRead;
use crate::scan::ArrowRecordBatchStream;

/// Wraps a [`FileRead`] to count bytes read via a shared atomic counter.
pub(crate) struct CountingFileRead<F: FileRead> {
    inner: F,
    bytes_read: Arc<AtomicU64>,
}

impl<F: FileRead> CountingFileRead<F> {
    pub(crate) fn new(inner: F, bytes_read: Arc<AtomicU64>) -> Self {
        Self { inner, bytes_read }
    }
}

#[async_trait::async_trait]
impl<F: FileRead> FileRead for CountingFileRead<F> {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        debug_assert!(range.end >= range.start);
        self.bytes_read
            .fetch_add(range.end - range.start, Ordering::Relaxed);
        self.inner.read(range).await
    }
}

/// Metrics collected during an Iceberg scan.
#[derive(Clone, Debug)]
pub struct ScanMetrics {
    bytes_read: Arc<AtomicU64>,
    bloom_filter: BloomFilterMetrics,
}

impl ScanMetrics {
    pub(crate) fn new() -> Self {
        Self {
            bytes_read: Arc::new(AtomicU64::new(0)),
            bloom_filter: BloomFilterMetrics::default(),
        }
    }

    pub(crate) fn bytes_read_counter(&self) -> &Arc<AtomicU64> {
        &self.bytes_read
    }

    /// Total bytes read from storage during this scan, including data files and delete files.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Row group counters of the bloom filter phase.
    pub fn bloom_filter(&self) -> &BloomFilterMetrics {
        &self.bloom_filter
    }
}

/// Result of [`ArrowReader::read`](super::ArrowReader::read), containing the
/// record batch stream and metrics collected during the scan.
pub struct ScanResult {
    stream: ArrowRecordBatchStream,
    metrics: ScanMetrics,
}

impl ScanResult {
    pub(crate) fn new(stream: ArrowRecordBatchStream, metrics: ScanMetrics) -> Self {
        Self { stream, metrics }
    }

    /// Consumes the result, returning only the record batch stream.
    pub fn stream(self) -> ArrowRecordBatchStream {
        self.stream
    }

    /// Returns a reference to the scan metrics.
    pub fn metrics(&self) -> &ScanMetrics {
        &self.metrics
    }
}

/// Row group counters of the bloom filter phase of one scan.
///
/// Every row group that reaches the bloom filter phase is counted exactly
/// once, as pruned or as matched. Clones share the counters.
#[derive(Clone, Debug, Default)]
pub struct BloomFilterMetrics {
    row_groups_pruned: Arc<AtomicU64>,
    row_groups_matched: Arc<AtomicU64>,
    read_errors: Arc<AtomicU64>,
}

impl BloomFilterMetrics {
    /// Row groups the bloom filter phase proved to have no matching rows.
    pub fn row_groups_pruned(&self) -> u64 {
        self.row_groups_pruned.load(Ordering::Relaxed)
    }

    /// Row groups that entered the bloom filter phase and were kept, including
    /// row groups for which no bloom filter was read.
    pub fn row_groups_matched(&self) -> u64 {
        self.row_groups_matched.load(Ordering::Relaxed)
    }

    /// Bloom filters that could not be fetched or parsed. Their columns are
    /// treated as might-match. Counted for bloom filters whose column chunk
    /// records `bloom_filter_length`; a failure to read any other bloom filter
    /// is only logged at debug level.
    pub fn read_errors(&self) -> u64 {
        self.read_errors.load(Ordering::Relaxed)
    }

    /// Records the outcome of the bloom filter phase for one data file:
    /// `candidates` row groups entered the phase and `survivors` of them were kept.
    pub(crate) fn record_row_groups(&self, candidates: usize, survivors: usize) {
        debug_assert!(survivors <= candidates);
        let pruned = candidates.saturating_sub(survivors);
        self.row_groups_pruned
            .fetch_add(pruned as u64, Ordering::Relaxed);
        self.row_groups_matched
            .fetch_add(survivors as u64, Ordering::Relaxed);
    }

    /// Records `count` bloom filters that could not be fetched or parsed.
    pub(crate) fn add_read_errors(&self, count: u64) {
        self.read_errors.fetch_add(count, Ordering::Relaxed);
    }
}
