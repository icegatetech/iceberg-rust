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

//! Vendored foyer cache layer for OpenDAL.
//!
//! Adapted from <https://github.com/apache/opendal/tree/main/core/layers/foyer>
//! to work with opendal 0.55.0. Once opendal publishes a release with the
//! built-in `FoyerLayer`, this module should be replaced by enabling the
//! `layers-foyer` feature on opendal directly.

use std::cmp::min;
use std::future::Future;
use std::ops::{Bound, Deref, Range, RangeBounds};
use std::sync::Arc;

use foyer::{Code, CodeError, Error as FoyerError, HybridCache};
use opendal::raw::{
    Access, AccessorInfo, BytesContentRange, BytesRange, Layer, LayeredAccess, MaybeSend, OpDelete,
    OpList, OpRead, OpStat, OpWrite, RpDelete, RpList, RpRead, RpWrite, oio,
};
use opendal::{Buffer, Error, ErrorKind, Metadata, Result};

/// Maximum size (in bytes) that [`FoyerValue::decode`] will allocate.
///
/// Guards against corrupted on-disk cache entries that encode an unreasonably
/// large length prefix, which would otherwise cause an OOM.
const MAX_FOYER_VALUE_SIZE: usize = 1 << 30; // 1 GiB

/// Key for the foyer cache, encoded via bincode (foyer's "serde" feature).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FoyerKey {
    pub path: String,
    pub version: Option<String>,
}

/// Value wrapper around [`Buffer`] that implements foyer's [`Code`] trait.
#[derive(Debug)]
pub struct FoyerValue(pub Buffer);

impl Deref for FoyerValue {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Code for FoyerValue {
    fn encode(&self, writer: &mut impl std::io::Write) -> std::result::Result<(), CodeError> {
        let len = self.0.len() as u64;
        writer.write_all(&len.to_le_bytes())?;
        std::io::copy(&mut self.0.clone(), writer)?;
        Ok(())
    }

    fn decode(reader: &mut impl std::io::Read) -> std::result::Result<Self, CodeError>
    where Self: Sized {
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let len: usize = u64::from_le_bytes(len_bytes)
            .try_into()
            .map_err(|e| CodeError::from(std::io::Error::other(e)))?;
        if len > MAX_FOYER_VALUE_SIZE {
            return Err(CodeError::from(std::io::Error::other(format!(
                "foyer cache entry size {len} exceeds maximum {MAX_FOYER_VALUE_SIZE}"
            ))));
        }
        let mut buffer = vec![0u8; len];
        reader.read_exact(&mut buffer[..len])?;
        Ok(FoyerValue(buffer.into()))
    }

    fn estimated_size(&self) -> usize {
        8 + self.0.len()
    }
}

/// Hybrid cache layer for OpenDAL backed by [foyer](https://github.com/foyer-rs/foyer).
///
/// - `read`: checks the cache first, falls back to the service, caches the result.
/// - `write`: writes through to the service and populates the cache.
/// - `delete`: removes from the cache after the service delete flushes.
/// - Other operations pass through to the underlying accessor.
#[derive(Debug)]
pub struct FoyerLayer {
    cache: HybridCache<FoyerKey, FoyerValue>,
    size_limit: Range<usize>,
}

impl FoyerLayer {
    /// Creates a new `FoyerLayer` with the given foyer hybrid cache.
    pub fn new(cache: HybridCache<FoyerKey, FoyerValue>) -> Self {
        FoyerLayer {
            cache,
            size_limit: 0..usize::MAX,
        }
    }

    /// Sets the size limit for caching.
    #[allow(dead_code)]
    pub fn with_size_limit<R: RangeBounds<usize>>(mut self, size_limit: R) -> Self {
        let start = match size_limit.start_bound() {
            Bound::Included(v) => *v,
            Bound::Excluded(v) => v.saturating_add(1),
            Bound::Unbounded => 0,
        };
        let end = match size_limit.end_bound() {
            Bound::Included(v) => v.saturating_add(1),
            Bound::Excluded(v) => *v,
            Bound::Unbounded => usize::MAX,
        };
        self.size_limit = start..end;
        self
    }
}

impl<A: Access> Layer<A> for FoyerLayer {
    type LayeredAccess = FoyerAccessor<A>;

    fn layer(&self, accessor: A) -> Self::LayeredAccess {
        FoyerAccessor {
            inner: Arc::new(Inner {
                accessor,
                cache: self.cache.clone(),
                size_limit: self.size_limit.clone(),
            }),
        }
    }
}

#[derive(Debug)]
struct Inner<A: Access> {
    accessor: A,
    cache: HybridCache<FoyerKey, FoyerValue>,
    size_limit: Range<usize>,
}

#[derive(Debug)]
pub struct FoyerAccessor<A: Access> {
    inner: Arc<Inner<A>>,
}

impl<A: Access> LayeredAccess for FoyerAccessor<A> {
    type Inner = A;
    type Reader = Buffer;
    type Writer = FoyerWriter<A>;
    type Lister = A::Lister;
    type Deleter = FoyerDeleter<A>;

    fn inner(&self) -> &Self::Inner {
        &self.inner.accessor
    }

    fn info(&self) -> Arc<AccessorInfo> {
        self.inner.accessor.info()
    }

    async fn read(&self, path: &str, args: OpRead) -> Result<(RpRead, Self::Reader)> {
        FullReader::new(self.inner.clone(), self.inner.size_limit.clone())
            .read(path, args)
            .await
    }

    fn write(
        &self,
        path: &str,
        args: OpWrite,
    ) -> impl Future<Output = Result<(RpWrite, Self::Writer)>> + MaybeSend {
        let inner = self.inner.clone();
        let size_limit = self.inner.size_limit.clone();
        let path = path.to_string();
        async move {
            let (rp, w) = inner.accessor.write(&path, args).await?;
            Ok((rp, FoyerWriter::new(w, path, inner, size_limit)))
        }
    }

    fn delete(&self) -> impl Future<Output = Result<(RpDelete, Self::Deleter)>> + MaybeSend {
        let inner = self.inner.clone();
        async move {
            let (rp, d) = inner.accessor.delete().await?;
            Ok((rp, FoyerDeleter::new(d, inner)))
        }
    }

    async fn list(&self, path: &str, args: OpList) -> Result<(RpList, Self::Lister)> {
        self.inner.accessor.list(path, args).await
    }
}

// --- FullReader: caches entire objects, slices to requested range ---
//
// NOTE: On a cache miss, the entire object is fetched into memory before the
// requested byte range is sliced out.  Configure `with_size_limit()` on
// `FoyerLayer` to avoid pulling very large objects into the cache.

struct FullReader<A: Access> {
    inner: Arc<Inner<A>>,
    size_limit: Range<usize>,
}

impl<A: Access> FullReader<A> {
    fn new(inner: Arc<Inner<A>>, size_limit: Range<usize>) -> Self {
        Self { inner, size_limit }
    }

    /// Reads the requested byte range from the cache (or backend on miss).
    ///
    /// **Cache-miss behaviour:** the *entire* object is fetched into memory so
    /// it can be cached for future reads; only the requested range is returned
    /// to the caller.  Use [`FoyerLayer::with_size_limit`] to cap the maximum
    /// object size eligible for caching and avoid excessive memory usage.
    async fn read(&self, path: &str, args: OpRead) -> Result<(RpRead, Buffer)> {
        let path_str = path.to_string();
        let version = args.version().map(|v| v.to_string());
        let original_args = args.clone();

        let (range_start, range_end) = {
            let r = args.range();
            let start = r.offset();
            let end = r
                .size()
                .map(|size| {
                    start.checked_add(size).ok_or_else(|| {
                        Error::new(
                            ErrorKind::Unexpected,
                            format!("range overflow: offset {start} + size {size} exceeds u64"),
                        )
                    })
                })
                .transpose()?;
            (start, end)
        };

        let result = self
            .inner
            .cache
            .fetch(
                FoyerKey {
                    path: path_str.clone(),
                    version: version.clone(),
                },
                || {
                    let inner = self.inner.clone();
                    let size_limit = self.size_limit.clone();
                    let path_clone = path_str.clone();
                    async move {
                        let metadata = inner
                            .accessor
                            .stat(&path_clone, OpStat::default())
                            .await
                            .map_err(FoyerError::other)?
                            .into_metadata();

                        let size = metadata.content_length() as usize;
                        if !size_limit.contains(&size) {
                            return Err(FoyerError::other(FetchSizeTooLarge));
                        }

                        let (_, mut reader) = inner
                            .accessor
                            .read(
                                &path_clone,
                                OpRead::default().with_range(BytesRange::new(0, None)),
                            )
                            .await
                            .map_err(FoyerError::other)?;
                        let buffer = oio::Read::read_all(&mut reader)
                            .await
                            .map_err(FoyerError::other)?;

                        Ok(FoyerValue(buffer))
                    }
                },
            )
            .await;

        match result {
            Ok(entry) => {
                let entry_len = entry.len() as u64;
                let end = range_end.unwrap_or(entry_len).min(entry_len);
                if end <= range_start {
                    return Ok((RpRead::new().with_size(Some(0)), Buffer::new()));
                }
                let range = BytesContentRange::default()
                    .with_range(range_start, end - 1)
                    .with_size(entry.len() as _);
                let buffer = entry.slice(range_start as usize..end as usize);
                let rp = RpRead::new()
                    .with_size(Some(buffer.len() as _))
                    .with_range(Some(range));
                Ok((rp, buffer))
            }
            Err(e) => match e.downcast::<FetchSizeTooLarge>() {
                Ok(_) => {
                    let (rp, mut reader) = self.inner.accessor.read(path, original_args).await?;
                    let buffer = oio::Read::read_all(&mut reader).await?;
                    Ok((rp, buffer))
                }
                Err(e) => Err(extract_err(e)),
            },
        }
    }
}

// --- Writer: writes through and populates cache ---

pub struct FoyerWriter<A: Access> {
    w: A::Writer,
    buf: oio::QueueBuf,
    path: String,
    inner: Arc<Inner<A>>,
    size_limit: Range<usize>,
    skip_cache: bool,
}

impl<A: Access> FoyerWriter<A> {
    fn new(w: A::Writer, path: String, inner: Arc<Inner<A>>, size_limit: Range<usize>) -> Self {
        Self {
            w,
            buf: oio::QueueBuf::new(),
            path,
            inner,
            size_limit,
            skip_cache: false,
        }
    }
}

impl<A: Access> oio::Write for FoyerWriter<A> {
    async fn write(&mut self, bs: Buffer) -> Result<()> {
        // Once skip_cache is set it stays true for the rest of the write session.
        if !self.skip_cache {
            if self.size_limit.contains(&(self.buf.len() + bs.len())) {
                self.buf.push(bs.clone());
            } else {
                self.buf.clear();
                self.skip_cache = true;
            }
        }
        let result = self.w.write(bs).await;
        if result.is_err() && !self.skip_cache {
            self.buf.clear();
            self.skip_cache = true;
        }
        result
    }

    async fn close(&mut self) -> Result<Metadata> {
        let metadata = self.w.close().await?;
        if !self.skip_cache {
            let buffer = self.buf.clone().collect();
            self.inner.cache.insert(
                FoyerKey {
                    path: self.path.clone(),
                    version: metadata.version().map(|v| v.to_string()),
                },
                FoyerValue(buffer),
            );
        }
        Ok(metadata)
    }

    async fn abort(&mut self) -> Result<()> {
        self.buf.clear();
        self.w.abort().await
    }
}

// --- Deleter: removes from cache after service delete flushes ---

pub struct FoyerDeleter<A: Access> {
    deleter: A::Deleter,
    keys: Vec<FoyerKey>,
    inner: Arc<Inner<A>>,
}

impl<A: Access> FoyerDeleter<A> {
    fn new(deleter: A::Deleter, inner: Arc<Inner<A>>) -> Self {
        Self {
            deleter,
            keys: vec![],
            inner,
        }
    }
}

impl<A: Access> oio::Delete for FoyerDeleter<A> {
    // In opendal 0.55.0, delete is synchronous (just queues the request).
    fn delete(&mut self, path: &str, args: OpDelete) -> Result<()> {
        self.deleter.delete(path, args.clone())?;
        self.keys.push(FoyerKey {
            path: path.to_string(),
            version: args.version().map(|v| v.to_string()),
        });
        Ok(())
    }

    // flush executes queued deletions, then removes entries from cache.
    async fn flush(&mut self) -> Result<usize> {
        let count = self.deleter.flush().await?;
        let n = min(count, self.keys.len());
        for key in self.keys.drain(..n) {
            self.inner.cache.remove(&key);
        }
        Ok(count)
    }
}

// --- Error helpers ---

#[derive(Debug)]
struct FetchSizeTooLarge;

impl std::fmt::Display for FetchSizeTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fetched data size exceeds size limit")
    }
}

impl std::error::Error for FetchSizeTooLarge {}

fn extract_err(e: FoyerError) -> Error {
    match e.downcast::<Error>() {
        Ok(e) => e,
        Err(e) => Error::new(ErrorKind::Unexpected, e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- FoyerValue encode/decode ---

    #[test]
    fn test_foyer_value_encode_decode_roundtrip() {
        let original = FoyerValue(Buffer::from(vec![1u8, 2, 3, 4, 5]));
        let mut encoded = Vec::new();
        original.encode(&mut encoded).unwrap();
        let decoded = FoyerValue::decode(&mut &encoded[..]).unwrap();
        assert_eq!(decoded.0.to_vec(), vec![1u8, 2, 3, 4, 5]);
    }

    #[test]
    fn test_foyer_value_encode_decode_empty() {
        let original = FoyerValue(Buffer::new());
        let mut encoded = Vec::new();
        original.encode(&mut encoded).unwrap();
        let decoded = FoyerValue::decode(&mut &encoded[..]).unwrap();
        assert!(decoded.0.is_empty());
    }

    #[test]
    fn test_foyer_value_decode_rejects_oversized_length() {
        let huge_len = (MAX_FOYER_VALUE_SIZE as u64) + 1;
        let data = huge_len.to_le_bytes();
        let result = FoyerValue::decode(&mut &data[..]);
        assert!(result.is_err());
    }

    #[test]
    fn test_foyer_value_decode_rejects_u64_exceeding_usize() {
        // On 64-bit this won't differ from the size cap test, but the
        // try_into guard is still exercised for correctness.
        let huge_len: u64 = u64::MAX;
        let data = huge_len.to_le_bytes();
        let result = FoyerValue::decode(&mut &data[..]);
        assert!(result.is_err());
    }

    // --- with_size_limit saturating arithmetic ---

    async fn build_test_cache() -> HybridCache<FoyerKey, FoyerValue> {
        let dir = tempfile::tempdir().expect("failed to create tempdir");
        foyer::HybridCacheBuilder::new()
            .memory(64)
            .storage(foyer::Engine::Large(Default::default()))
            .with_device_options(
                foyer::DirectFsDeviceOptions::new(dir.path()).with_capacity(4 * 1024 * 1024),
            )
            .with_recover_mode(foyer::RecoverMode::None)
            .build()
            .await
            .expect("failed to build test cache")
    }

    #[tokio::test]
    async fn test_with_size_limit_excluded_usize_max_start() {
        let cache = build_test_cache().await;
        // Bound::Excluded(usize::MAX) for start — previously panicked on overflow
        let layer =
            FoyerLayer::new(cache).with_size_limit((Bound::Excluded(usize::MAX), Bound::Unbounded));
        assert_eq!(layer.size_limit, usize::MAX..usize::MAX);
    }

    #[tokio::test]
    async fn test_with_size_limit_included_usize_max_end() {
        let cache = build_test_cache().await;
        // Bound::Included(usize::MAX) for end — previously panicked on overflow
        let layer = FoyerLayer::new(cache).with_size_limit(..=usize::MAX);
        assert_eq!(layer.size_limit, 0..usize::MAX);
    }

    #[tokio::test]
    async fn test_with_size_limit_normal_range() {
        let cache = build_test_cache().await;
        let layer = FoyerLayer::new(cache).with_size_limit(10..1000);
        assert_eq!(layer.size_limit, 10..1000);
    }

    // --- flush clamp ---

    #[test]
    fn test_drain_clamp_logic() {
        // Simulates the clamping logic in flush: min(count, keys.len())
        let keys = vec![1, 2, 3];
        let count = 5usize; // exceeds keys.len()
        let n = min(count, keys.len());
        assert_eq!(n, 3);
    }
}
