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

use std::future::Future;
use std::ops::{Bound, Deref, Range, RangeBounds};
use std::sync::Arc;

use foyer::{Code, CodeError, Error as FoyerError, HybridCache};
use opendal::raw::oio;
use opendal::raw::*;
use opendal::*;

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
    where
        Self: Sized,
    {
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;
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
            Bound::Excluded(v) => *v + 1,
            Bound::Unbounded => 0,
        };
        let end = match size_limit.end_bound() {
            Bound::Included(v) => *v + 1,
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

struct FullReader<A: Access> {
    inner: Arc<Inner<A>>,
    size_limit: Range<usize>,
}

impl<A: Access> FullReader<A> {
    fn new(inner: Arc<Inner<A>>, size_limit: Range<usize>) -> Self {
        Self { inner, size_limit }
    }

    async fn read(&self, path: &str, args: OpRead) -> Result<(RpRead, Buffer)> {
        let path_str = path.to_string();
        let version = args.version().map(|v| v.to_string());
        let original_args = args.clone();

        let (range_start, range_end) = {
            let r = args.range();
            let start = r.offset();
            let end = r.size().map(|size| start + size);
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
                        let buffer =
                            oio::Read::read_all(&mut reader).await.map_err(FoyerError::other)?;

                        Ok(FoyerValue(buffer))
                    }
                },
            )
            .await;

        match result {
            Ok(entry) => {
                let end = range_end.unwrap_or(entry.len() as u64);
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
        if self.size_limit.contains(&(self.buf.len() + bs.len())) {
            self.buf.push(bs.clone());
            self.skip_cache = false;
        } else {
            self.buf.clear();
            self.skip_cache = true;
        }
        self.w.write(bs).await
    }

    async fn close(&mut self) -> Result<Metadata> {
        let buffer = self.buf.clone().collect();
        let metadata = self.w.close().await?;
        if !self.skip_cache {
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
        for key in self.keys.drain(..) {
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
