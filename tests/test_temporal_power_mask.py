import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from temporal.temporal_plugin import TemporalPlugin
from temporal.fusion.local_attention import local_neighbors


class TemporalPowerMaskTest(unittest.TestCase):
    def test_filtered_tree_ids_refer_to_original_history(self):
        xyz=np.zeros((1,2,4,3));xyz[...,0]=[0,3,4,10]
        points=np.array([[[False,False,True,True],[False,False,True,True]]])
        ids,valid=local_neighbors(xyz,np.ones((1,2),bool),k=2,radius_m=.5,point_valid_mask=points)
        self.assertFalse(valid[0,:2].any())
        self.assertEqual(ids[0,2,0],2)
        self.assertEqual(ids[0,3,0],3)
        self.assertFalse(valid[0,:,1].any())
        _,empty=local_neighbors(xyz,np.ones((1,2),bool),point_valid_mask=np.zeros((1,2,4),bool))
        self.assertFalse(empty.any())

    def test_optional_zeroing_boundaries_gradients_and_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bins.npz'
            np.savez(path,range_m=[0,3.7,3.8,10],azimuth_rad=[0],elevation_rad=[0])
            cfg=dict(mode='ego_attention',alignment=dict(bins_path=str(path)))
            original=TemporalPlugin(**cfg)
            masked=TemporalPlugin(**cfg,power_mask=dict(enabled=True,radius_m=3.7))
            disabled=TemporalPlugin(**cfg,power_mask=dict(enabled=False,radius_m=3.7))
            masked.load_state_dict(original.state_dict(),strict=True)
            disabled.load_state_dict(original.state_dict(),strict=True)
            features=(torch.rand(1,2,4,8)+1).requires_grad_()
            saved=features.detach().clone()
            coords=torch.zeros(1,2,4,3);coords[...,0]=torch.arange(4)
            poses=torch.eye(4).repeat(1,2,1,1);poses[:,1,0,3]=2
            kwargs=dict(coords=coords,poses=poses,frame_offsets=torch.tensor([[-1,0]]),
                        valid_mask=torch.ones(1,2,dtype=torch.bool))
            a,_=original(features,**kwargs);b,_=disabled(features,**kwargs)
            self.assertTrue(torch.equal(a,b))
            out,debug=masked(features,**kwargs)
            self.assertEqual(debug['power_masked_per_frame'],[[2,2]])
            self.assertTrue(torch.equal(out[...,:2,3:6],features[:,-1,:2,3:6]))
            for c in [0,1,2,6,7]:self.assertTrue((out[0,:2,c]==0).all())
            self.assertTrue(torch.equal(out[0,2:],features[0,-1,2:]))
            self.assertTrue(torch.equal(features.detach(),saved))
            # Nonzero correction cannot resurrect excluded current power.
            with torch.no_grad():masked.fusion.output.bias.fill_(.2)
            out,_=masked(features,**kwargs)
            self.assertTrue((out[0,:2,2]==0).all())
            out.sum().backward()
            self.assertTrue(torch.isfinite(features.grad).all())


if __name__=='__main__':unittest.main()
