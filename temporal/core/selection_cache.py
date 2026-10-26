"""Content-addressed, order-sensitive frozen-selector indices (no learned features)."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def update_array(digest, value):
    array = np.ascontiguousarray(value)
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())


class SelectionCache:
    def __init__(self, directory, mode='readwrite'):
        if mode not in ('read', 'readwrite'):
            raise ValueError('selection cache mode must be read or readwrite')
        self.root, self.mode = Path(directory), mode
        if mode == 'read' and not self.root.is_dir():
            raise FileNotFoundError(f'Build selection cache first: {self.root}')
        if mode == 'readwrite':
            self.root.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def key(self, namespace, features, xyz, frames, times, points):
        digest = hashlib.sha256(namespace.encode())
        # Preserve exact P1 order: no assumptions about CPU/GPU tie sorting.
        for value in (features, xyz, frames, times, points):
            if value is None:
                digest.update(b'None')
            else:
                update_array(digest, value)
        return digest.hexdigest()

    def path(self, key):
        return self.root/key[:2]/f'{key}.npz'

    def load(self, key, shape, history_size):
        path = self.path(key)
        if not path.exists():
            self.misses += 1
            if self.mode == 'read':
                raise FileNotFoundError(f'Selection cache miss: {path}; regenerate with the same '
                                        'checkpoint, device, input order and configuration')
            return None
        with np.load(path, allow_pickle=False) as data:
            if str(data['key']) != key:
                raise ValueError(f'Selection cache identity mismatch: {path}')
            indices = data['indices'].astype(np.int64)
            valid = np.unpackbits(data['valid'], count=int(np.prod(shape))).reshape(shape).astype(bool)
        if indices.shape != shape or np.any(indices < 0) or np.any(indices >= history_size):
            raise ValueError(f'Invalid selection cache indices: {path}')
        self.hits += 1
        return indices, valid

    def save(self, key, indices, valid, metadata):
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if indices.size and indices.max() > np.iinfo(np.int32).max:
            raise ValueError('Selection indices exceed int32 storage')
        fd, temporary = tempfile.mkstemp(prefix='.selection-', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                # Uncompressed ZIP: avoid repeating CPU decompression each epoch.
                np.savez(stream, key=np.asarray(key), indices=indices.astype(np.int32),
                         valid=np.packbits(valid), metadata=np.asarray(json.dumps(metadata)))
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
