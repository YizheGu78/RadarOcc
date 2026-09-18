import unittest
from dataclasses import replace

import numpy as np

from tradition.core.config import MotionConfig, UnwrappingConfig
from tradition.core.geometry import wrapped_velocity_residual
from tradition.core.types import DopplerEvidence, EgoMotion, MotionLabel, TemporalDetectionFrame
from tradition.motion.range_kalman import RangeKalmanUnwrapper
from tradition.motion.temporal_consistency import PoseAlignedTemporalClassifier
from tradition.tests.test_pose_temporal_pipeline import _detection, _pose


def observation(frame, target_speed=19.6, ego_speed=23., sign=1., dt=.1, noise=0.):
    time = frame * dt
    pose = _pose(x=ego_speed * time)
    ego = EgoMotion(pose, np.array([ego_speed, 0., 0.]), np.array([ego_speed, 0., 0.]), 0.)
    x = 60 + (target_speed - ego_speed) * time + noise
    detections = []
    for y in (-.05, .05):
        los_x = x / np.hypot(x, y)
        doppler = wrapped_velocity_residual(sign * (ego_speed - target_speed) * los_x, 3.84)
        detections.append(_detection((x, y, 0.), doppler))
    return detections, ego


class RangeKalmanTests(unittest.TestCase):
    def run_track(self, target_speed=19.6, sign=1., frames=range(30), dt=.1, noise=False):
        tracker = RangeKalmanUnwrapper(motion_cfg=MotionConfig(stationary_velocity_sign=sign),
                                      config=UnwrappingConfig(frame_dt_s=dt))
        for i in frames:
            detections, ego = observation(i, target_speed, sign=sign, dt=dt,
                                         noise=.03*np.sin(i) if noise else 0.)
            residuals, evidence = tracker.update(detections, ego, f'3_{i:05d}')
        return tracker, residuals, evidence

    def test_moving_truck_both_doppler_signs(self):
        for sign in (1., -1.):
            tracker, residuals, evidence = self.run_track(sign=sign)
            np.testing.assert_allclose(residuals, -sign*19.6, atol=.01)
            self.assertEqual(evidence.tolist(), [-1, -1])
            self.assertEqual(tracker.last_diagnostics['ambiguity_k'], [int(-sign*5)]*2)

    def test_static_target_with_fast_ego(self):
        # Keep target ahead of the ego during the test.
        tracker, residuals, evidence = self.run_track(target_speed=0., frames=range(20))
        np.testing.assert_allclose(residuals, 0., atol=.001)
        self.assertEqual(evidence.tolist(), [1, 1])

    def test_noise_and_skipped_frame_indices(self):
        _, residuals, evidence = self.run_track(frames=range(0, 30, 2), noise=True)
        np.testing.assert_allclose(residuals, -19.6, atol=.01)
        self.assertEqual(evidence.tolist(), [-1, -1])

    def test_nondefault_time_interval(self):
        _, residuals, _ = self.run_track(dt=.2, frames=range(20))
        np.testing.assert_allclose(residuals, -19.6, atol=.01)

    def test_warmup_empty_scene_reset_and_gap(self):
        tracker, _, _ = self.run_track()
        detections, ego = observation(30)
        residuals, evidence = tracker.update(detections, ego, '4_00030')
        self.assertTrue(np.isnan(residuals).all())
        self.assertEqual(evidence.tolist(), [0, 0])
        residuals, evidence = tracker.update([], ego, '4_00031')
        self.assertEqual(len(residuals), 0)
        residuals, evidence = tracker.update(detections, ego, '4_00040')
        self.assertTrue(np.isnan(residuals).all())
        self.assertEqual(len(tracker.tracks), 1)
        tracker.reset()
        self.assertFalse(tracker.tracks)

    def test_repeated_or_reversed_time_resets(self):
        for index in (29, 28):
            tracker, _, _ = self.run_track()
            detections, ego = observation(index)
            values, _ = tracker.update(detections, ego, f'3_{index:05d}')
            self.assertTrue(np.isnan(values).all())

    def test_centroid_jump_restarts(self):
        tracker, _, _ = self.run_track()
        detections, ego = observation(30, noise=5.)
        values, evidence = tracker.update(detections, ego, '3_00030')
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(evidence.tolist(), [0, 0])

    def test_uncertain_and_dynamic_not_overridden_by_static_neighbours(self):
        temporal = PoseAlignedTemporalClassifier()
        detections = [_detection((10., 0., 0.)), _detection((10., .1, 0.)), _detection((10., .2, 0.))]
        frame = TemporalDetectionFrame('3_00000', _pose(), detections,
                                      np.array([0., np.nan, -19.6]), np.array([1, 0, -1]))
        result = temporal.update(frame)
        self.assertEqual(result.current_indices.tolist(), [0, 2])
        self.assertEqual(result.current_motion_labels, [MotionLabel.STATIC, MotionLabel.DYNAMIC])

    def test_high_uncertainty_does_not_force_an_alias(self):
        tracker = RangeKalmanUnwrapper(config=UnwrappingConfig(velocity_floor_std_mps=2.))
        for i in range(20):
            detections, ego = observation(i)
            values, evidence = tracker.update(detections, ego, f'3_{i:05d}')
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(evidence.tolist(), [0, 0])

    def test_ambiguous_spatial_association_starts_new_tracks(self):
        tracker = RangeKalmanUnwrapper()
        detections, ego = observation(0)
        tracker.update(detections, ego, '3_00000')
        detections, ego = observation(1)
        second = [replace(d, xyz_lidar_m=d.xyz_lidar_m + [0., 3., 0.],
                          xyz_radar_m=d.xyz_radar_m + [0., 3., 0.]) for d in detections]
        values, _ = tracker.update(detections + second, ego, '3_00001')
        self.assertTrue(np.isnan(values).all())
        self.assertNotIn(0, tracker.last_diagnostics['track_ids'])

    def test_rotated_pose_and_lateral_ego_velocity(self):
        tracker = RangeKalmanUnwrapper()
        world_target = np.array([70., 15., 0.])
        velocity_world = np.array([10., 3., 0.])
        for i in range(30):
            pose = _pose(x=i, y=.3*i, yaw=.3)
            velocity = pose[:3, :3].T @ velocity_world
            point = pose[:3, :3].T @ (world_target - pose[:3, 3])
            detections = []
            for offset in (-.02, .02):
                xyz = point + np.array([0., offset, 0.])
                doppler = wrapped_velocity_residual(xyz @ velocity / np.linalg.norm(xyz), 3.84)
                detections.append(_detection(tuple(xyz), doppler))
            ego = EgoMotion(pose, velocity, velocity, 0.)
            values, evidence = tracker.update(detections, ego, f'3_{i:05d}')
        np.testing.assert_allclose(values, 0., atol=.001)
        self.assertEqual(evidence.tolist(), [1, 1])

    def test_pipeline_integration_and_metadata(self):
        from tradition.pipeline.traditional_radar_pipeline import build_rpc_pipeline
        pipeline = build_rpc_pipeline(unwrapping_cfg=UnwrappingConfig())
        class Identity:
            def read(self, value): return value
            def detect(self, value): return value
            def filter(self, value): return value
        pipeline.reader = pipeline.detector = pipeline.reliability_filter = Identity()
        for i in range(30):
            detections, ego = observation(i)
            result = pipeline.predict_temporal_file(detections, f'3_{i:05d}', ego)
        self.assertEqual(result.metadata['velocity_unwrapping']['ambiguity_k'], [-5, -5])
        self.assertEqual(result.motion_labels, [MotionLabel.DYNAMIC]*2)
        pipeline.reset_sequence()
        self.assertFalse(pipeline.velocity_unwrapper.tracks)


if __name__ == '__main__':
    unittest.main()
