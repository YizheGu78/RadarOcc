"""Analytical checks of Stage B physics, abstention, pairing and GT boundary."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.io import savemat

from tools.temporal_vis import audit_neighborhood_motion as stage
from tools.temporal_vis.directional_box_matching import BoxSearch
from tools.temporal_vis.neighborhood_motion import (
    VotingConfig, compensate_radial, ego_velocity, load_doppler_table,
    peak_statistics, predict_directions,
)


def features(bins, power=1.):
    bins = np.asarray(bins)
    return np.column_stack([np.full((len(bins),3),power),np.repeat(bins[:,None],3,axis=1),np.ones((len(bins),2))])


class MotionTests(unittest.TestCase):
    def test_physical_table_and_unsorted_large_power_peaks(self):
        table = np.array([-2.,-1.,0,1,2])
        f = np.array([[1e34,3e34,2e34,4,2,3,0,0]])
        radial,_,valid,std = peak_statistics(f,table,2.)
        self.assertAlmostEqual(radial[0],4/6)
        self.assertTrue(valid[0]); self.assertTrue(np.isfinite(std[0]))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'arr_doppler.mat'
            savemat(path,dict(arr_doppler=table[None]))
            np.testing.assert_array_equal(load_doppler_table(path),table)
            savemat(path,dict(arrRange=table[None]))
            with self.assertRaisesRegex(ValueError,'Doppler table'):
                load_doppler_table(path)

    def test_invalid_zero_power_and_separated_peaks_abstain(self):
        f = features([0,1,2,3,4])
        f[0,:3] = 0
        f[1,3] = 1.5
        f[2,3] = 99
        f[3,0] = np.nan
        f[4,3:6] = [0,2,4]
        _,_,valid,_ = peak_statistics(f,np.arange(-2,3),.5)
        self.assertFalse(valid.any())

    def test_los_projection_sign_and_no_modulo_after_compensation(self):
        xyz = np.array([[10.,0,0],[6.,8,0]])
        result,cosine,projection = compensate_radial(xyz,[-15,-9],[15,0,0],1)
        np.testing.assert_allclose(result,0,atol=1e-12)
        np.testing.assert_allclose(projection,[15,9])
        np.testing.assert_allclose(cosine,[1,.6])
        np.testing.assert_allclose(compensate_radial(xyz,[15,9],[15,0,0],-1)[0],0,atol=1e-12)
        # A direct measured + ego value above the bin-table interval stays so.
        self.assertEqual(compensate_radial(xyz[:1],[1],[15,0,0],1)[0][0],16)

    def test_radar_origin_velocity_rotation_and_lever_arm(self):
        current,previous,ext = np.eye(4),np.eye(4),np.eye(4)
        current[0,3] = 1.5
        np.testing.assert_allclose(ego_velocity(current,previous,ext,.1),[15,0,0])
        current = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1.]])
        ext[0,3] = 2
        # Radar origin moves from (2,0) to (0,2); current frame is rotated.
        np.testing.assert_allclose(ego_velocity(current,previous,ext,1),[2,2,0])
        with self.assertRaises(ValueError):
            ego_velocity(current,previous,ext,0)

    def test_neighbors_repair_outlier_without_self_or_gt(self):
        xyz = np.array([[10.,0,0],[10.1,0,0],[10.2,0,0],[10.3,0,0],[10.4,0,0]])
        f = features([0,4,4,4,4])
        prediction = predict_directions(xyz,f,np.arange(-2,3),[0,0,0],1)
        self.assertEqual(prediction['single'][0],1)
        self.assertEqual(prediction['neighborhood'][0],-1)
        self.assertEqual(prediction['neighbor_count'][0],4)
        # Arrays are unchanged: the other sign hypothesis is independent.
        again = predict_directions(xyz,f,np.arange(-2,3),[0,0,0],-1)
        self.assertEqual(again['neighborhood'][0],1)

    def test_stationary_sparse_and_offaxis_fall_back(self):
        xyz = np.array([[10.,0,0],[10.1,0,0],[10.2,0,0],[10.3,0,0]])
        table = np.arange(-2,3)
        stationary = predict_directions(xyz,features([2]*4),table,[0,0,0],1)
        self.assertFalse(stationary['neighborhood'].any())
        self.assertEqual(stationary['fallback_reason'][0],'near_zero')
        one = predict_directions(xyz[:1],features([4]),table,[0,0,0],1)
        self.assertEqual(one['fallback_reason'][0],'too_few_neighbors')
        xyz[:,1] = 20
        offaxis = predict_directions(xyz,features([4]*4),table,[0,0,0],1)
        self.assertFalse(offaxis['single'].any())
        self.assertFalse(offaxis['neighborhood'].any())
        empty = predict_directions(np.empty((0,3)),np.empty((0,8)),table,[0,0,0],1)
        self.assertEqual(len(empty['neighborhood']),0)

    def test_mixed_background_does_not_become_false_unanimity(self):
        xyz = np.tile([10.,0,0],(7,1))
        # For query 0 the other six tokens split evenly static/moving.
        result = predict_directions(xyz,features([4,4,4,4,2,2,2]),np.arange(-2,3),[0,0,0],1)
        self.assertEqual(result['neighborhood'][0],0)
        self.assertEqual(result['fallback_reason'][0],'inconsistent_neighbors')

    def test_variable_direction_search_matches_individual_stage_a(self):
        rng = np.random.default_rng(24)
        history = rng.uniform(-4,4,(300,3))
        labels = rng.integers(-1,2,300)
        queries = rng.uniform(-2,2,(12,3))
        directions = np.tile([-1,0,1],4)
        search = BoxSearch(history,labels,4,.5,1.5,1)
        result = stage.adaptive_counts(search,queries,0,directions,8)
        expected = {key:0 for key in stage.box.COUNT_KEYS}
        for query,direction in zip(queries,directions):
            row = search.counts(query[None],0,'symmetric' if direction == 0 else 'correct_direction',int(direction) or 1,8)
            for key in expected:
                if key != 'n_history_same': expected[key] += row[key]
        expected['n_history_same'] = int((labels == 0).sum())
        self.assertEqual(result,expected)
        fallback = stage.adaptive_counts(search,queries,0,np.zeros(12),8)
        self.assertEqual(fallback,search.counts(queries,0,'symmetric',1,8))
        self.assertEqual(stage.adaptive_counts(search,queries,99,directions,8)['n_no_history'],12)

    def test_causal_velocity_uses_previous_split_entry_never_future(self):
        data = stage.MotionDataset.__new__(stage.MotionDataset)
        data.positions = {2:0,7:1}; data.orders = [2,7]
        data.pose = Mock(side_effect=lambda order: np.eye(4))
        data.time = Mock(side_effect=lambda order:order*.1)
        data.ext = np.eye(4)
        self.assertEqual(data.causal_velocity(2),(None,None,None))
        _,previous,dt = data.causal_velocity(7)
        self.assertEqual(previous,2);self.assertAlmostEqual(dt,.5)
        self.assertEqual({c.args[0] for c in data.pose.call_args_list},{2,7})

    def test_motion_loader_preserves_p1_order_and_never_reads_gt(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'frame.npz'; count = 175*250
            raw = features(np.arange(count)%5).T
            np.savez(path,power_val=raw,elevation_ind=np.zeros(count),azimuth_ind=np.zeros(count))
            data = stage.MotionDataset.__new__(stage.MotionDataset)
            data.by_order = {0:dict(radar_path=str(path))}
            data.args = SimpleNamespace(ego_mask_radius=0)
            data.annotation = Mock(side_effect=AssertionError('GT must not be read'))
            data.geometry = stage.audit.va.RadarBinGeometry(np.arange(175),[0],[0])
            xyz,f,ids = data.motion_p1(0)
            np.testing.assert_array_equal(ids,np.arange(250,count))
            np.testing.assert_array_equal(f,raw[:,250:].T)
            self.assertEqual(len(xyz),count-250)
            data.annotation.assert_not_called()

    def test_full_paired_audit_outputs_and_missing_pose_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d); table = np.arange(-2.,3.)
            np.savez(output/'doppler.npz',doppler_mps=table)
            args = stage.parser().parse_args(['--output-dir',str(output),
                '--doppler-bins',str(output/'doppler.npz'),'--gaps','1','--box-lengths','4','--save-token-details'])
            def points(order):
                return np.array([[10.+order+.1*i,0,0] for i in range(4)])
            data = SimpleNamespace(ext=np.eye(4),positions={0:0,1:1,2:2},selected=lambda:[1,2],
                motion_p1=lambda order:(points(order),features([4]*4),np.arange(4)),
                causal_velocity=lambda order:(None,None,None) if order == 1 else (np.zeros(3),1,.1),
                p1=lambda order:(points(order),np.zeros(4,dtype=int)),
                annotation=lambda order:(np.array([[10.+order,0,-1,2,2,2,0]]),np.array(['A']),np.array([2])),
                pose=lambda order:np.eye(4),time=lambda order:order*.1,histories=lambda order:[(1,order-1)])
            result = stage.run(args,data,table,VotingConfig())
            self.assertEqual(len(result),8)
            with (output/'motion_per_object_history.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            fallback = [r for r in rows if r['order']=='1' and r['method']=='neighborhood']
            self.assertTrue(all(int(r['n_direction_uncertain'])==4 for r in fallback))
            decided = [r for r in rows if r['order']=='2' and r['method']=='neighborhood']
            self.assertEqual([int(r['n_direction_correct']) for r in decided],[4,0])
            self.assertEqual([int(r['n_direction_reversed']) for r in decided],[0,4])
            for sign in [1,-1]:
                self.assertEqual({r['n_eligible'] for r in rows if int(r['doppler_sign'])==sign},{'4'})
            protocol = json.loads((output/'motion_protocol.json').read_text())
            self.assertTrue(protocol['gt_free_prediction'])
            self.assertEqual(protocol['selected_orders'],[1,2])
            self.assertTrue((output/'motion_token_details.csv').is_file())
            # Empty common cohort still generates usable headers and a report.
            args.min_direction_dx = 100
            self.assertEqual(stage.run(args,data,table,VotingConfig()),[])


if __name__ == '__main__':
    unittest.main()
