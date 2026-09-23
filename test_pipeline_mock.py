#!/usr/bin/env python3
import numpy as np
import pipeline as P
class Vision:
    def prepare(self,image,prompt): self.image=image; return self
    def inference(self):
        h,w=self.image.shape[:2]; m=np.zeros((h,w),bool); m[h//4:3*h//4,w//4:3*w//4]=True
        return {'det':{'boxes':np.array([[w/4,h/4,3*w/4,3*h/4]],np.float32),'scores':np.array([.9],np.float32),'labels':['object'],'reason':None},'seg':{'mask':m,'iou':np.array([.9],np.float32),'best':0,'n_pred':1,'reason':None}}
    def release(self): pass
class Depth:
    def prepare(self,image,camera_K=None,fov_x=None): self.image=image; self.K=camera_K; return self
    def inference(self): h,w=self.image.shape[:2]; return {'depth':np.full((h,w),.6,np.float32),'intrinsics':self.K.astype(np.float32),'fov_x_deg':60.,'scale':1.,'reason':None}
    def release(self): pass
class TSDF:
    def build(self,*a,**k): return {'grid':np.ones((1,40,40,40),np.float32),'voxel_size':.0075,'T_cam_volume':np.eye(4),'observed_voxels':100}
class Grasp:
    def prepare(self,*a): return self
    def inference(self):
        g=np.zeros((1,17),np.float64); g[0,0]=.9; g[0,1]=.05; g[0,4:13]=np.eye(3).reshape(-1); g[0,13:16]=[0,0,.6]; return {'graspgroup':g,'reason':None}
    def release(self): pass
def main():
    img=np.zeros((120,160,3),np.uint8); K=np.array([[120.,0,80.],[0,120.,60.],[0,0,1.]]); old=P.run_phases
    P.run_phases=lambda image,prompt,**kw:P.DEFAULT_PIPELINE.run(image,prompt,camera_K=K,vision=Vision(),depther=Depth(),tsdf_builder=TSDF(),grasper=Grasp())
    try: r=P.pipeline(img,'object',camera_K=K)
    finally: P.run_phases=old
    assert set(r)=={'box','mask','depthmap','grasp','depth_m'}; assert all(r[k].shape==img.shape for k in ('box','mask','depthmap','grasp')); assert abs(r['depth_m']-.6)<1e-5
    print('TAT CA MUC DEU PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
