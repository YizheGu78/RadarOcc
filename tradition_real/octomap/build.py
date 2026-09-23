"""Build an in-tree pybind11 module against the unmodified pinned OctoMap.

Run with the same Python environment used for preprocessing. No ROS, CUDA,
system OctoMap, root installation, setup.py or changes to the parent project.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = ('AbstractOcTree.cpp', 'AbstractOccupancyOcTree.cpp', 'OcTree.cpp',
           'OcTreeNode.cpp', 'Pointcloud.cpp', 'ScanGraph.cpp',
           'math/Pose6D.cpp', 'math/Quaternion.cpp', 'math/Vector3.cpp')


def source_fingerprint():
    manifest = json.loads((ROOT / 'OCTOMAP_SOURCE_MANIFEST.json').read_text())
    paths = ['OCTOMAP_SOURCE_MANIFEST.json', 'octomap/bindings.cpp',
             'octomap/build.py', 'octomap/backend.py', 'octomap/__init__.py']
    for entry in manifest['files']:
        path = ROOT / entry['local_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Pinned OctoMap source was modified: {path}')
        paths.append(entry['local_path'])
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode() + b'\0' + (ROOT / name).read_bytes() + b'\0')
    return digest.hexdigest()


def main():
    if sys.platform not in ('linux', 'darwin'):
        raise SystemExit('This build entry supports Linux/macOS; on Windows use WSL.')
    try:
        import pybind11
    except ImportError as error:
        raise SystemExit('Install build dependency: python -m pip install pybind11') from error
    fingerprint = source_fingerprint()
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if not suffix:
        raise SystemExit('Python did not report an extension suffix')
    target = HERE / ('_native' + suffix)
    vendor = ROOT / 'vendor/octomap'
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler:
        raise SystemExit('CXX must name a C++ compiler')
    # Intentionally no -ffast-math, -march=native, OpenMP or algorithm changes.
    flags = ['-O3', '-std=c++11', '-shared', '-fPIC', '-ffp-contract=off',
             '-fvisibility=hidden', f'-DOCTOMAP_BINDING_FINGERPRINT="{fingerprint}"']
    if sys.platform == 'darwin':
        flags += ['-undefined', 'dynamic_lookup']
    includes = [pybind11.get_include(), sysconfig.get_path('include'),
                sysconfig.get_path('platinclude'), str(vendor / 'include')]
    # Compile into an adjacent temporary directory, then atomically replace only
    # this generated binary. A failed compile leaves any existing binary intact.
    with tempfile.TemporaryDirectory(prefix='.native-build-', dir=HERE) as temporary:
        binary = Path(temporary) / target.name
        command = compiler + flags + [f'-I{p}' for p in dict.fromkeys(includes) if p]
        command += [str(HERE / 'bindings.cpp'), *[str(vendor / 'src' / p) for p in SOURCES],
                    '-o', str(binary)]
        print('Building pinned OctoMap for', sys.executable, flush=True)
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as error:
            raise SystemExit('C++ compiler missing; install g++/build-essential or set CXX.') from error
        # Import the temporary module in a fresh process before publishing it.
        subprocess.run([sys.executable, '-c',
                        'import importlib.util,sys; '
                        's=importlib.util.spec_from_file_location("_native",sys.argv[1]); '
                        'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                        'assert m.source_fingerprint == sys.argv[2]; '
                        'assert m.OcTree(.4).size() == 0', str(binary), fingerprint], check=True)
        os.replace(binary, target)
    print('Built:', target)
    print('Original OctoMap C++ backend is now the default (CPU).')


if __name__ == '__main__':
    main()
