#!/usr/bin/env python3
"""Independent regression and real-frame geometry evidence for corrected recipes."""
import argparse
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/geometry_v2/checks.json'))
    args = p.parse_args()
    suite = unittest.TestLoader().discover(str(ROOT/'tests'), pattern='test_geometry_v2.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # Real inputs: validate scalar K-Radar conversion and produce GT counts,
    # rather than asserting a fabricated expected overlap percentage.
    from temporal.core.ego_alignment import RadarBinGeometry, KRADAR_COORDINATES, radar_to_lidar_default
    from tools.temporal_vis.vis_alignment import read_infos
    from tools.temporal_vis.compare_coordinate_neighbor_selection import frame_data
    from tools.temporal.train_top8_ranker import ranker_memberships
    from tools.temporal_vis.audit_dynamic_objects import transform
    import numpy as np
    geom = RadarBinGeometry.from_file(ROOT/'data/K-Radar-official-meta/resources/info_arr.mat',
                                     coordinate_convention=KRADAR_COORDINATES)
    rows = read_infos(ROOT/'data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl')
    frames = []
    for scene in ['11', '27', '8', '18', '20']:
        options = [r for r in rows if str(r.get('scene_id', r.get('scene_token'))) == scene]
        if not options:
            continue
        info = options[len(options)//2]
        xyz, features, _, _ = frame_data(info, geom)
        directory = ROOT/'data/K-RadarOcc/train'/scene/'bbox'
        index = info['occ_frame_idx']
        boxes = np.load(directory/f'bbox{index}.npy', allow_pickle=False).reshape(-1, 7)
        ids = np.load(directory/f'bbox_token{index}.npy', allow_pickle=False).reshape(-1).astype(str)
        labels = ranker_memberships(transform(xyz, radar_to_lidar_default()), boxes, ids)
        frames.append(dict(scene=scene, order=int(info['order_in_scene']), points=len(xyz),
                           unique_GT_points=int((labels >= 0).sum())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
        skipped=len(result.skipped), actual_GT_frames=frames,
        baseline_note='Small projection was corrected in source; this check covers temporal geometry/protocol. Run V12 for projection contracts and V13 for real CUDA whole-model checks.'), indent=2)+'\n')
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
