"""Causal range-only CA tracking and Doppler ambiguity resolution.

KF state is *sensor range*, range rate and range acceleration. Association is
in world coordinates. Doppler never updates the KF (no alias feedback loop).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from tradition.core.config import KRadarConfig, MotionConfig, UnwrappingConfig
from tradition.core.types import DopplerEvidence, EgoMotion, RadarDetection
from tradition.io.pose_reader import RadarOccPoseReader
from tradition.motion.doppler_classifier import EgoCompensatedDopplerClassifier


@dataclass
class _Track:
    identifier: int
    history: list = field(default_factory=list)
    state: np.ndarray | None = None
    covariance: np.ndarray | None = None


class RangeKalmanUnwrapper:
    def __init__(self, radar_cfg=None, motion_cfg=None, config=None):
        self.radar_cfg = radar_cfg or KRadarConfig()
        self.motion_cfg = motion_cfg or MotionConfig()
        self.config = config or UnwrappingConfig()
        if not np.isfinite(self.radar_cfg.doppler_period_mps) or self.radar_cfg.doppler_period_mps <= 0:
            raise ValueError("Doppler period must be finite and positive.")
        self.doppler = EgoCompensatedDopplerClassifier(self.radar_cfg, self.motion_cfg)
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 0
        self.last_time = None
        self.scene = None
        self.last_diagnostics = {}

    def _clusters(self, detections, excluded_mask=None):
        # Spatial connected components for the non-persistent complement.
        # Persistent occupancy is established before this velocity stage and
        # must not bias cluster centres or create stationary KF tracks.
        xyz = np.asarray([d.xyz_lidar_m for d in detections]).reshape(-1, 3)
        excluded = (
            np.zeros(len(detections), dtype=bool)
            if excluded_mask is None
            else np.asarray(excluded_mask, dtype=bool)
        )
        if excluded.shape != (len(detections),):
            raise ValueError("excluded_mask must match detections.")
        valid = np.flatnonzero(np.all(np.isfinite(xyz), axis=1) & ~excluded)
        if not len(valid):
            return []
        tree = cKDTree(xyz[valid])
        unseen = set(range(len(valid)))
        clusters = []
        while unseen:
            seed = min(unseen)
            unseen.remove(seed)
            pending, component = [seed], []
            while pending:
                i = pending.pop()
                component.append(int(valid[i]))
                for j in tree.query_ball_point(xyz[valid[i]], self.config.cluster_radius_m):
                    if j in unseen:
                        unseen.remove(j)
                        pending.append(j)
            if len(component) >= self.config.min_cluster_points:
                clusters.append(np.asarray(sorted(component)))
        return clusters

    def _observe(self, track, time, center, distance):
        cfg = self.config
        if track.state is not None:
            dt = time - track.history[-1][0]
            f = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1.]])
            # Continuous white jerk process noise.
            q = cfg.jerk_variance * np.array([
                [dt**5/20, dt**4/8, dt**3/6],
                [dt**4/8, dt**3/3, dt**2/2],
                [dt**3/6, dt**2/2, dt]])
            state = f @ track.state
            covariance = f @ track.covariance @ f.T + q
            innovation = distance - state[0]
            variance = covariance[0, 0] + cfg.range_std_m**2
            if abs(innovation) > cfg.innovation_sigma * np.sqrt(variance):
                # Changed scatterer/association: discard the old velocity prior.
                track.history.clear()
                track.state = track.covariance = None
            else:
                gain = covariance[:, 0] / variance
                track.state = state + gain * innovation
                a = np.eye(3) - np.outer(gain, [1., 0., 0.])
                track.covariance = a @ covariance @ a.T + np.outer(gain, gain) * cfg.range_std_m**2
        track.history.append((time, center, distance))
        track.history = track.history[-cfg.history_size:]
        if track.state is None and len(track.history) >= cfg.init_frames:
            times = np.array([h[0] - time for h in track.history])
            design = np.column_stack([np.ones(len(times)), times])
            fit, _, _, _ = np.linalg.lstsq(design, [h[2] for h in track.history], rcond=None)
            track.state = np.array([fit[0], fit[1], 0.])
            track.covariance = np.zeros((3, 3))
            track.covariance[:2, :2] = cfg.range_std_m**2 * np.linalg.inv(design.T @ design)
            track.covariance[2, 2] = cfg.initial_acceleration_std_mps2**2

    def update(
        self,
        detections: list[RadarDetection],
        ego: EgoMotion,
        token: str,
        excluded_mask=None,
    ):
        cfg = self.config
        time = RadarOccPoseReader.frame_index(token) * cfg.frame_dt_s
        scene = str(token).rsplit('_', 1)[0]
        if self.scene != scene or (self.last_time is not None and time <= self.last_time):
            self.reset()
        self.scene, self.last_time = scene, time
        self.tracks = [t for t in self.tracks if time - t.history[-1][0] <= cfg.max_gap_s]
        wrapped = self.doppler.residuals_with_velocity(detections, ego.linear_velocity_radar_mps)
        output = np.full(len(detections), np.nan)
        evidence = np.full(len(detections), int(DopplerEvidence.UNCERTAIN), dtype=np.int8)
        ids = np.full(len(detections), -1, dtype=int)
        ambiguity = [None] * len(detections)
        kf_prediction = [None] * len(detections)
        excluded = (
            np.zeros(len(detections), dtype=bool)
            if excluded_mask is None
            else np.asarray(excluded_mask, dtype=bool)
        )
        if excluded.shape != (len(detections),):
            raise ValueError("excluded_mask must match detections.")
        output[excluded] = 0.0
        evidence[excluded] = int(DopplerEvidence.STATIC)
        clusters = self._clusters(detections, excluded)
        observations = []
        for indices in clusters:
            center_lidar = np.median([detections[i].xyz_lidar_m for i in indices], axis=0)
            center = ego.pose_lidar_to_world[:3, :3] @ center_lidar + ego.pose_lidar_to_world[:3, 3]
            # Match the range measurement to the robust radar centroid.
            center_radar = np.median([detections[i].xyz_radar_m for i in indices], axis=0)
            observations.append((center, center_radar, float(np.linalg.norm(center_radar))))
        # Conservative mutual-unique association. If multiple candidates are
        # plausible, restart rather than transfer a velocity/alias to an object.
        edges = []
        for ti, track in enumerate(self.tracks):
            history = track.history
            dt = time - history[-1][0]
            predicted = history[-1][1].copy()
            if len(history) >= 2:
                velocity = (history[-1][1] - history[0][1]) / (history[-1][0] - history[0][0])
                predicted += velocity * dt
                gate = cfg.association_radius_m
            else:
                gate = cfg.association_radius_m + cfg.max_speed_mps * dt
            for ci, (center, _, _) in enumerate(observations):
                if np.linalg.norm(center - predicted) <= gate:
                    edges.append((ti, ci))
        track_counts = Counter(t for t, c in edges)
        cluster_counts = Counter(c for t, c in edges)
        assignments = {}
        for ti, ci in edges:
            if track_counts[ti] == 1 and cluster_counts[ci] == 1:
                assignments[ci] = self.tracks[ti]
        # Retire ambiguous tracks so stale states cannot steal later matches.
        ambiguous_tracks = {ti for ti, ci in edges if ci not in assignments}
        self.tracks = [t for ti, t in enumerate(self.tracks) if ti not in ambiguous_tracks]
        for ci, (indices, (center, center_radar, distance)) in enumerate(zip(clusters, observations)):
            if not np.isfinite(distance) or distance <= 1e-6:
                continue
            track = assignments.get(ci)
            if track is None:
                track = _Track(self.next_id)
                self.next_id += 1
                self.tracks.append(track)
            self._observe(track, time, center, distance)
            ids[indices] = track.identifier
            if track.state is None:
                continue
            los = center_radar / distance
            # With s=+1: Doppler = -range_rate; compensated residual =
            # -s * (range_rate + dot(ego_velocity, LOS)). NOT +target_speed.
            predicted = -self.motion_cfg.stationary_velocity_sign * (
                track.state[1] + los @ ego.linear_velocity_radar_mps)
            sigma = np.sqrt(max(0., track.covariance[1, 1]) + cfg.velocity_floor_std_mps**2)
            # Range differential is a consistency check, not independent data
            # in another weighted likelihood (it shares observations with KF).
            h = track.history
            coarse = (h[-1][2] - h[0][2]) / (h[-1][0] - h[0][0])
            coarse += track.state[2] * (h[-1][0] - h[0][0]) / 2
            if abs(coarse - track.state[1]) > cfg.range_check_tolerance_mps:
                continue
            period = self.radar_cfg.doppler_period_mps
            for index in indices:
                kf_prediction[index] = float(predicted)
                value = wrapped[index]
                if not np.isfinite(value):
                    continue
                k = int(np.rint((predicted - value) / period))
                candidate = value + k * period
                error = abs(candidate - predicted)
                # Require the uncertainty interval to stay within one alias
                # cell, and reject measurements inconsistent with the prior.
                if (abs(candidate) > cfg.max_speed_mps or
                    error > cfg.max_doppler_error_mps or
                    error + cfg.confidence_sigma * sigma >= period / 2):
                    continue
                output[index] = candidate
                ambiguity[index] = k
                if abs(candidate) <= self.motion_cfg.static_residual_threshold_mps:
                    evidence[index] = int(DopplerEvidence.STATIC)
                elif abs(candidate) >= self.motion_cfg.dynamic_residual_threshold_mps:
                    evidence[index] = int(DopplerEvidence.DYNAMIC)
        self.last_diagnostics = {
            'method': 'range_ca_kalman', 'token': str(token),
            'velocity_convention': 'ego_compensated_doppler_residual',
            'point_indices': list(range(len(detections))),
            'track_ids': ids.tolist(), 'ambiguity_k': ambiguity,
            'wrapped_residual_mps': [float(x) if np.isfinite(x) else None for x in wrapped],
            'kf_residual_mps': kf_prediction,
            'unwrapped_residual_mps': [float(x) if np.isfinite(x) else None for x in output],
            'persistent_excluded_count': int(excluded.sum()),
            'resolved_motion_count': int(np.isfinite(output[~excluded]).sum()),
            'unresolved_motion_count': int((~np.isfinite(output[~excluded])).sum()),
            'resolved_count': int(np.isfinite(output).sum()),
            'unresolved_count': int((~np.isfinite(output)).sum()),
        }
        return output, evidence
