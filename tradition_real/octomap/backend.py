"""Execution backend only. Mapping configuration and cache semantics stay unchanged."""
import importlib
import math
import os

from .octree import OcTree as PythonOcTree

_native = None


def backend_name():
    name = os.environ.get('TRADITION_REAL_OCTOMAP_BACKEND', 'cpp')
    if name not in ('cpp', 'python'):
        raise ValueError('TRADITION_REAL_OCTOMAP_BACKEND must be cpp or python')
    return name


def native_module():
    global _native
    if _native is None:
        try:
            module = importlib.import_module('tradition_real.octomap._native')
        except ImportError as error:
            raise RuntimeError(
                'OctoMap C++ backend is not built for this Python environment. '
                'Run: python -m pip install pybind11 && '
                'bash run_tradition_real.sh build-octomap. '
                'For the slower reference implementation explicitly set '
                'TRADITION_REAL_OCTOMAP_BACKEND=python') from error
        from .build import source_fingerprint
        if module.source_fingerprint != source_fingerprint():
            raise RuntimeError('OctoMap C++ binary is stale; run bash run_tradition_real.sh build-octomap')
        _native = module
    return _native


def execution_info():
    """Non-semantic provenance; deliberately separate from the cache signature."""
    name = backend_name()
    result = {'backend': name, 'device': 'cpu'}
    if name == 'cpp':
        import hashlib
        from pathlib import Path
        module = native_module()
        result.update(source_fingerprint=module.source_fingerprint,
                      binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                      upstream_commit=module.upstream_commit)
    return result


class OcTree:
    """Facade keeping the existing adapter and its method calls untouched.

    Native query results are read-only value snapshots, never dangling pointers
    into nodes that a subsequent upstream update may prune.
    """
    tree_depth = 16
    tree_max_val = 32768

    def __init__(self, resolution):
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError('Resolution must be finite and positive')
        self.backend = backend_name()
        cls = native_module().OcTree if self.backend == 'cpp' else PythonOcTree
        self._impl = cls(resolution)

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def insertPointCloud(self, scan, sensor_origin, maxrange=-1.0, lazy_eval=False, discretize=False):
        return self._impl.insertPointCloud(scan, sensor_origin, maxrange, lazy_eval, discretize)
