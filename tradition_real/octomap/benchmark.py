"""Reproducible Python/C++ whole-map-frame timing, with exact output checks.

Synthetic workload only; includes unchanged grid export, DBSCAN and 42D.
Excludes dataset IO, compression, GT, RF and compilation. No data is written.
"""
import argparse
import json
import os
import time

import numpy as np

from ..config import Config
from ..pipeline import Pipeline
from .backend import native_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', type=int, default=1000)
    parser.add_argument('--frames', type=int, default=6)
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()
    if args.points < 1 or args.frames < 1:
        parser.error('points and frames must be positive')
    native_module()  # check/load before starting the clocks
    cfg = Config()
    rng = np.random.default_rng(args.seed)
    clouds = []
    for frame in range(args.frames):
        rpc = np.zeros((args.points, 11), np.float64)
        rpc[:, :3] = rng.uniform([2., -24., -1.5], [50., 24., 2.5], (args.points, 3))
        rpc[:, 3] = rng.uniform(1., 20., args.points)
        rpc[:, 4] = rng.uniform(-3., 3., args.points)
        rpc[:, 5] = np.linalg.norm(rpc[:, :3], axis=1)
        rpc[:, 6] = np.arctan2(rpc[:, 1], rpc[:, 0])
        rpc[:, 7] = np.arcsin(rpc[:, 2] / rpc[:, 5])
        clouds.append(rpc)
    durations, reference = {}, []
    old = os.environ.get('TRADITION_REAL_OCTOMAP_BACKEND')
    try:
        for name in ('python', 'cpp'):
            os.environ['TRADITION_REAL_OCTOMAP_BACKEND'] = name
            pipe = Pipeline(cfg)
            times = []
            for ordinal, rpc in enumerate(clouds):
                pose = np.eye(4); pose[0, 3] = ordinal * .1
                start = time.perf_counter()
                result = pipe.map_frame(rpc, pose, np.array([2.54, -.3, -.7]), '3', ordinal)
                times.append(time.perf_counter() - start)
                if name == 'python':
                    reference.append(result)
                else:
                    other = reference[ordinal]
                    for field in ('probability', 'observed', 'occupied', 'free', 'bev_probability'):
                        np.testing.assert_array_equal(getattr(result, field), getattr(other, field))
                    assert len(result.candidates) == len(other.candidates)
                    for a, b in zip(result.candidates, other.candidates):
                        for field in ('features', 'voxels'):
                            np.testing.assert_array_equal(a[field], b[field])
                print(f'{name} frame {ordinal + 1}/{args.frames}: {times[-1]:.3f}s', flush=True)
            durations[name] = {'seconds': sum(times), 'per_frame_seconds': times}
    finally:
        if old is None:
            os.environ.pop('TRADITION_REAL_OCTOMAP_BACKEND', None)
        else:
            os.environ['TRADITION_REAL_OCTOMAP_BACKEND'] = old
    print(json.dumps({'synthetic': True, 'points_per_frame': args.points, 'frames': args.frames,
                      'temporal_window': cfg.temporal_window, 'exact_output_match': True,
                      'timings': durations,
                      'speedup': durations['python']['seconds'] / durations['cpp']['seconds']}, indent=2))


if __name__ == '__main__':
    main()
