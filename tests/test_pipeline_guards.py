import unittest
import numpy as np
from grasppose.edge_adapters import _classify_vgn_outputs,_resolve_K
from grasppose.geometry import depth_to_cloud,depth_range_str
from grasppose.tsdf import ProjectiveTSDFBuilder
from grasppose.vgn import vgn_to_graspgroup

class GeometryTests(unittest.TestCase):
    def test_depth_to_cloud_uses_mask_and_K(self):
        d=np.ones((3,3),np.float32); K=np.array([[2.,0,1.],[0,2.,1.],[0,0,1.]]); m=np.zeros((3,3),bool); m[1,1]=True; p=depth_to_cloud(d,K,m)
        self.assertEqual(p.shape,(1,3)); self.assertTrue(np.allclose(p[0],[0,0,1]))
    def test_depth_range_empty_is_safe(self): self.assertIn('khong do duoc',depth_range_str(np.zeros((2,2),np.float32)))
    def test_intrinsics_required(self):
        with self.assertRaises(ValueError): _resolve_K(None,None,640,480)
        self.assertEqual(_resolve_K(None,60.,640,480).shape,(3,3))
class TSDFTests(unittest.TestCase):
    def test_projective_grid_shape_and_encoding(self):
        h,w=48,64; d=np.full((h,w),.6,np.float32); K=np.array([[60.,0,w/2],[0,60.,h/2],[0,0,1.]]); m=np.ones((h,w),bool); cloud=depth_to_cloud(d,K,m); out=ProjectiveTSDFBuilder().build(d,K,m,cloud)
        self.assertEqual(out['grid'].shape,(1,40,40,40)); self.assertGreater(out['observed_voxels'],0); self.assertGreaterEqual(float(out['grid'].min()),0.); self.assertLessEqual(float(out['grid'].max()),1.)
class VGNTests(unittest.TestCase):
    def test_output_classification_by_names(self):
        out={'quality':np.zeros((1,1,40,40,40)),'rotation':np.zeros((1,4,40,40,40)),'width':np.zeros((1,1,40,40,40))}; q,r,w=_classify_vgn_outputs(out); self.assertEqual(q.shape,(40,40,40)); self.assertEqual(r.shape,(4,40,40,40)); self.assertEqual(w.shape,(40,40,40))
    def test_vgn_conversion(self):
        tsdf=np.ones((1,40,40,40),np.float32); qual=np.zeros((40,40,40),np.float32); qual[20,20,20]=1.; rot=np.zeros((4,40,40,40),np.float32); rot[3,...]=1.; width=np.full((40,40,40),5.,np.float32); gg=vgn_to_graspgroup(tsdf,qual,rot,width,.0075,np.eye(4),threshold=.01)
        self.assertEqual(gg.shape[1],17); self.assertGreaterEqual(len(gg),1); self.assertAlmostEqual(gg[0,1],.0375,places=5)
if __name__=='__main__': unittest.main()
