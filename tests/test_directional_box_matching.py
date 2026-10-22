"""Check paired box geometry, GT ego alignment, coverage and fixed-K ranking."""
import unittest
import numpy as np
from tools.temporal_vis.directional_box_matching import (
    METHODS, BoxSearch, box_limits, oracle_delta,
)


class DirectionalBoxTests(unittest.TestCase):
    def test_equal_volumes_and_direction_mirror(self):
        for length in [4,6,8]:
            for sign in [-1,1]:
                volumes=[]
                for method in METHODS:
                    low,high=box_limits(method,sign,length,.5,1.5,1.)
                    volumes.append(np.prod(high-low))
                np.testing.assert_allclose(volumes, [(length+.5)*6]*3)
                a,b=box_limits('correct_direction',sign,length,.5,1.5,1.)
                c,d=box_limits('wrong_direction',sign,length,.5,1.5,1.)
                self.assertAlmostEqual(a[0],-d[0]);self.assertAlmostEqual(b[0],-c[0])

    def test_oracle_removes_ego_motion_and_uses_current_radar_axes(self):
        current_pose=np.eye(4);current_pose[0,3]=10
        ext=np.eye(4);ext[:3,3]=[2,-.3,-.7]
        current_box=np.array([5,0,0,2,2,2,0])
        hist_box=np.array([15,0,0,2,2,2,0])
        delta,center=oracle_delta(current_box,hist_box,current_pose,np.eye(4),ext)
        np.testing.assert_allclose(delta,[0,0,0],atol=1e-12)
        np.testing.assert_allclose(center,[3,.3,1.7])
        hist_box[0]-=3
        delta,_=oracle_delta(current_box,hist_box,current_pose,np.eye(4),ext)
        np.testing.assert_allclose(delta,[-3,0,0],atol=1e-12)
        # A rotated current frame changes the direction components.
        rot=np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1.]])
        delta,_=oracle_delta(np.array([5,0,0,2,2,2,0]),
                             np.array([0,8,0,2,2,2,0]),rot,np.eye(4),np.eye(4))
        np.testing.assert_allclose(delta,[3,0,0],atol=1e-12)

    def test_direction_can_remove_closer_background_at_k8(self):
        history=np.zeros((9,3));history[:8,0]=-np.linspace(.6,1.3,8);history[8,0]=1.5
        labels=np.array([-1]*8+[0]);search=BoxSearch(history,labels,4,.5,1.5,1.)
        outcomes={m: search.counts(np.zeros((1,3)),0,m,1,8) for m in METHODS}
        self.assertEqual(outcomes['symmetric']['n_covered'],1)
        self.assertEqual(outcomes['symmetric']['n_topk_miss'],1)
        self.assertEqual(outcomes['correct_direction']['n_success'],1)
        self.assertEqual(outcomes['wrong_direction']['n_outside_box'],1)

    def test_coverage_does_not_imply_recall_and_query_distance_is_unchanged(self):
        # Distance to shifted box centre would incorrectly favor the same-object token.
        history=np.zeros((9,3));history[:8,0]=np.linspace(.1,.8,8);history[8,0]=3.
        search=BoxSearch(history,np.array([-1]*8+[0]),4,.5,1.5,1.)
        r=search.counts(np.zeros((1,3)),0,'correct_direction',1,8)
        self.assertEqual(r['n_covered'],1);self.assertEqual(r['n_success'],0)
        self.assertEqual(r['n_topk_miss'],1);self.assertEqual(r['n_selected'],8)
        r=search.counts(np.zeros((1,3)),99,'correct_direction',1,8)
        self.assertEqual(r['n_no_history'],1);self.assertEqual(r['n_eligible'],0)

    def test_tree_matches_brute_force_boxes_and_nearest_eight(self):
        rng=np.random.RandomState(35)
        history=rng.uniform(-5,5,(400,3));labels=rng.randint(-1,3,400)
        queries=rng.uniform(-3,3,(20,3));search=BoxSearch(history,labels,6,.5,1.5,1.)
        for method in METHODS:
            for sign in [-1,1]:
                actual=search.counts(queries,0,method,sign,8)
                covered=success=selected=same_selected=candidates=0
                low,high=box_limits(method,sign,6,.5,1.5,1.)
                for q in queries:
                    relative=history-q
                    ids=np.flatnonzero(np.all((relative>=low-1e-9)&(relative<=high+1e-9),axis=1))
                    covered+=int(np.any(labels[ids]==0));candidates+=len(ids)
                    ids=ids[np.lexsort((ids,np.sum(relative[ids]**2,axis=1)))[:8]]
                    selected+=len(ids);same_selected+=int((labels[ids]==0).sum())
                    success+=int(np.any(labels[ids]==0))
                self.assertEqual([actual[k] for k in ['n_covered','n_success','n_selected','n_selected_same','n_candidates']],
                                 [covered,success,selected,same_selected,candidates])

    def test_inclusive_edges_and_original_index_ties(self):
        history=np.array([[4.,1.5,1.],[-.5,-1.5,-1.],[4.001,0,0],[.2,0,0],[-.2,0,0]])
        search=BoxSearch(history,np.array([0,0,0,-1,0]),4,.5,1.5,1.)
        r=search.counts(np.zeros((1,3)),0,'correct_direction',1,1)
        self.assertEqual(r['n_candidates'],4)
        self.assertEqual(r['n_success'],0)  # exact distance tie resolves to original index 3


if __name__=='__main__':
    unittest.main()
