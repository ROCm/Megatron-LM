# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" FS Reader with metadata cached support. """

import os
from typing import Dict, Union

from torch.distributed.checkpoint import FileSystemReader, Metadata


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Metadata is shared across all reader instances that use the same checkpoint
    directory (same path), since the loaded metadata is identical.

    Attributes:
        _metadata_cache (Dict[str, Metadata]): Class-level cache keyed by checkpoint path.
    """

    _metadata_cache: Dict[str, Metadata] = {}

    def __init__(self, path: Union[str, os.PathLike], cache_metadata: bool = True) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
        """
        super().__init__(path=path)
        self._cache_metadata = cache_metadata
        self._abs_path = os.path.abspath(os.fspath(path))

    def _metadata_signature(self) -> tuple:
        """Build a cache key that is invalidated when the checkpoint is overwritten.

        Keying solely on the path is unsafe: a new checkpoint with a different structure
        can be written to the same path (e.g. reused directories), in which case a stale
        cached metadata would be returned and cause load failures. We additionally
        incorporate the metadata file's modification time and size so that rewriting the
        checkpoint at the same path invalidates the cache entry.
        """
        metadata_path = self._get_metadata_path()
        try:
            stat = os.stat(metadata_path)
            return (self._abs_path, stat.st_mtime_ns, stat.st_size)
        except OSError:
            # Fall back to path-only key if the metadata file can't be stat-ed.
            return (self._abs_path,)

    def read_metadata(self) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.
        Shared across instances when the checkpoint directory is the same and the
        on-disk metadata file is unchanged.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if not self._cache_metadata:
            return super().read_metadata()

        cache_key = self._metadata_signature()
        if cache_key not in CachedMetadataFileSystemReader._metadata_cache:
            CachedMetadataFileSystemReader._metadata_cache[cache_key] = super().read_metadata()
        return CachedMetadataFileSystemReader._metadata_cache[cache_key]

    @classmethod
    def clear_metadata_cache(cls):
        """
        Clear the metadata cache.
        """
        cls._metadata_cache.clear()
