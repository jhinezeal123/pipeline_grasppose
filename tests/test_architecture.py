import unittest
import numpy as np
from grasppose.orchestrator import GraspPipeline

class Vision:
    def __init__(self,c): self.c=c
    def prepare(self,image,prompt): self.c.append('vision.prepare'); self.shape=image.shape[:2]; return self
    def inference(self):
        self.c.append('vision.inference'); h,w=self.shape; m=np.zeros((h,w),bool); m[1:-1,1:-1]=True
        return {'det':{'boxes':np.array([[1,1,w-1,h-1]],np.float32),'scores':np.array([.9],np.float32),'labels':['object'],'reason':None},'seg':{'mask':m,'iou':np.array([.9],np.float32),'best':0,'n_pred':1,'reason':None}}
    def release(self): self.c.append('vision.release')
class Depth:
    def __init__(self,c): self.c=c
    def prepare(self,image,camera_K=None,fov_x=None): self.c.append('depth.prepare'); self.shape=image.shape[:2]; self.K=camera_K; return self
    def inference(self): self.c.append('depth.inference'); h,w=self.shape; return {'depth':np.full((h,w),.6,np.float32),'intrinsics':np.asarray(self.K,np.float32),'fov_x_deg':60.,'scale':1.,'reason':None}
    def release(self): self.c.append('depth.release')
class TSDF:
    def __init__(self,c): self.c=c
    def build(self,*a,**k): self.c.append('tsdf.build'); return {'grid':np.full((1,40,40,40),.5,np.float32),'voxel_size':.0075,'T_cam_volume':np.eye(4,dtype=np.float32),'observed_voxels':10}
class Grasper:
    def __init__(self,c): self.c=c
    def prepare(self,*a): self.c.append('grasp.prepare'); return self
    def inference(self): self.c.append('grasp.inference'); return {'graspgroup':np.zeros((0,17),np.float64),'reason':None}
    def release(self): self.c.append('grasp.release')

class ArchitectureTests(unittest.TestCase):
    def test_edge_order_and_release(self):
        c=[]; p=GraspPipeline(lambda:Vision(c),lambda:Depth(c),lambda:TSDF(c),lambda:Grasper(c)); K=np.array([[100.,0,2],[0,100.,2],[0,0,1]])
        out=p.run(np.zeros((4,4,3),np.uint8),'object',camera_K=K)
        self.assertEqual(out['tsdf']['grid'].shape,(1,40,40,40)); self.assertGreater(len(out['cloud']),0)
        self.assertLess(c.index('vision.release'),c.index('depth.prepare')); self.assertLess(c.index('depth.release'),c.index('tsdf.build')); self.assertLess(c.index('tsdf.build'),c.index('grasp.prepare')); self.assertIn('grasp.release',c)
    def test_empty_mask_skips_depth_and_vgn(self):
        c=[]
        class EmptyVision(Vision):
            def inference(self): r=super().inference(); r['seg']['mask'][:]=False; r['seg']['reason']='none'; return r
        p=GraspPipeline(lambda:EmptyVision(c),lambda:Depth(c),lambda:TSDF(c),lambda:Grasper(c)); out=p.run(np.zeros((4,4,3),np.uint8),'x',camera_K=np.eye(3))
        self.assertEqual(len(out['cloud']),0); self.assertNotIn('depth.prepare',c); self.assertNotIn('grasp.prepare',c)

if __name__=='__main__': unittest.main()
