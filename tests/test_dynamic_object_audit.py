"""Analytical checks for matching taxonomy and paired trail evaluation."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tools.temporal_vis import audit_dynamic_objects as audit


class AuditTests(unittest.TestCase):
    def test_matching_failure_partition(self):
        query = np.array([[0.,0,0],[10,0,0],[20,0,0]])
        history = np.array([[.1,0,0],[10.1,0,0],[10.2,0,0]])
        labels = np.array([0,-1,0])
        r = audit.matching_counts(query, history, labels, 0, 2., 1)
        self.assertEqual([r[k] for k in ['n_success','n_topk_miss','n_outside_radius']], [1,1,1])
        r = audit.matching_counts(query, history, labels, 99, 2., 1)
        self.assertEqual(r['n_no_history'], 3)
        self.assertEqual(r['n_eligible'], 0)
        self.assertIsNone(audit.ratio(0,0))

    def test_rotated_bottom_center_box_and_overlap(self):
        box = np.array([0,0,2,4,2,2,np.pi/2])
        points = np.array([[0,1.5,3],[1.5,0,3],[0,0,1],[0,0,4.1]])
        np.testing.assert_array_equal(audit.inside(points,box), [True,False,False,False])
        self.assertEqual(audit.memberships(points,[box,box])[0], -2)

    def test_sparse_axis_conventions_and_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'grid.npy'
            np.save(p, np.array([[3,1,0,2]]))
            self.assertEqual(audit.dense(p,[2,3,4])[0,1,3],2)
            self.assertEqual(audit.dense(p,[4,3,2],'xyz')[3,1,0],2)
            with self.assertRaises(ValueError):
                audit.dense(p,[2,2,2])

    def test_history_matches_split_slots(self):
        data = audit.Dataset.__new__(audit.Dataset)
        data.orders=[1,3,8]; data.positions={1:0,3:1,8:2}; data.by_order=dict.fromkeys(data.orders)
        data.args=SimpleNamespace(gaps=[1,2],history_mode='dataset')
        self.assertEqual(data.histories(8),[(5,3),(7,1)])
        data.args.history_mode='scene-order'
        self.assertEqual(data.histories(8),[])

    def test_world_speed_removes_ego_motion(self):
        data = audit.Dataset.__new__(audit.Dataset)
        data.args=SimpleNamespace(speed_window=2,max_speed_rmse=.01)
        data.by_order=dict.fromkeys(range(5))
        data.time=lambda j: .1*j
        def pose(j):
            p=np.eye(4);p[0,3]=j;return p
        data.pose=pose
        # Target moves 0.2 m/frame in world while ego moves 1 m/frame.
        data.annotation=lambda j: (np.array([[10-.8*j,0,0,1,1,2,0]]),np.array(['A']),np.array([2]))
        speed,rmse=data.speed(2,'A')
        self.assertAlmostEqual(speed,2.)
        self.assertLess(rmse,1e-10)

    def test_paired_occupancy_and_matching_join(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/'out';out.mkdir()
            info={'scene_id':3,'token':'3_1','occ_frame_idx':1,'occ_path':str(root/'gt.npy')}
            gt=np.zeros((4,1,1),int);gt[2]=2;np.save(info['occ_path'],gt)
            for model in ['single','knn']:
                p=root/model/'3'/'3_1';p.mkdir(parents=True)
                pred=gt.copy()
                if model=='knn':pred[0]=1
                np.save(p/'pred_c.npy',pred)
            match=root/'matching.csv'
            audit.write_csv(match,[dict(scene=3,order=1,object_id='A',frame_gap=1,n_eligible=100,n_success=25)])
            args=SimpleNamespace(matching_csv=match,grid_shape=[4,1,1],pc_range=[0,0,0,4,1,1],
                grid_frame='lidar',gt_order='xyz',fov_deg=360,single_root=root/'single',knn_root=root/'knn',
                skip_missing=False,valid_mask_template=None,scene='3',max_plots=0,output_dir=out)
            box=lambda x: np.array([[x,.5,0,1,1,1,0]])
            data=SimpleNamespace(ext=np.eye(4),by_order={1:info},selected=lambda:[1],
                annotation=lambda j:(box(2.5 if j==1 else .5),np.array(['A']),np.array([2])),
                pose=lambda j:np.eye(4),time=lambda j:float(j),speed=lambda j,t:(2.,0.),histories=lambda j:[(1,0)])
            audit.occupancy(args,data)
            with (out/'occupancy_per_object.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2)
            union=rows[1]
            self.assertEqual(float(union['same_object_recall']),.25)
            self.assertEqual(float(union['single_tfpr']),0.)
            self.assertEqual(float(union['knn_tfpr']),1.)
            self.assertEqual(float(union['knn_target_recall']),1.)
            # Strict evaluation mask removes unobserved trail, denominator becomes undefined.
            mask=np.ones((4,1,1),bool);mask[0]=False;np.save(root/'mask.npy',mask)
            args.valid_mask_template=str(root/'mask.npy');audit.occupancy(args,data)
            with (out/'occupancy_per_object.csv').open() as f: row=next(csv.DictReader(f))
            self.assertEqual(int(row['n_trail_free']),0)
            self.assertEqual(row['knn_tfpr'],'')


if __name__=='__main__':
    unittest.main()
