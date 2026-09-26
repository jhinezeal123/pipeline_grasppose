#!/usr/bin/env python3
"""Export the original ETH VGN ConvNet checkpoint to ONNX.

Then build model/vgn.engine on the target Jetson with trtexec --fp16.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(a,b,k): return nn.Conv3d(a,b,k,padding=k//2)
def conv_stride(a,b,k): return nn.Conv3d(a,b,k,stride=2,padding=k//2)


class Encoder(nn.Module):
    def __init__(self,in_channels,filters,kernels):
        super().__init__(); self.conv1=conv_stride(in_channels,filters[0],kernels[0]); self.conv2=conv_stride(filters[0],filters[1],kernels[1]); self.conv3=conv_stride(filters[1],filters[2],kernels[2])
    def forward(self,x): return F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))


class Decoder(nn.Module):
    def __init__(self,in_channels,filters,kernels):
        super().__init__(); self.conv1=conv(in_channels,filters[0],kernels[0]); self.conv2=conv(filters[0],filters[1],kernels[1]); self.conv3=conv(filters[1],filters[2],kernels[2])
    def forward(self,x):
        x=F.relu(self.conv1(x)); x=F.interpolate(x,10); x=F.relu(self.conv2(x)); x=F.interpolate(x,20); x=F.relu(self.conv3(x)); return F.interpolate(x,40)


class ConvNet(nn.Module):
    """Exact module names/shapes from ethz-asl/vgn so upstream state_dict fits."""
    def __init__(self):
        super().__init__(); self.encoder=Encoder(1,[16,32,64],[5,3,3]); self.decoder=Decoder(64,[64,32,16],[3,3,5]); self.conv_qual=conv(16,1,5); self.conv_rot=conv(16,4,5); self.conv_width=conv(16,1,5)
    def forward(self,x):
        x=self.decoder(self.encoder(x)); return torch.sigmoid(self.conv_qual(x)),F.normalize(self.conv_rot(x),dim=1),self.conv_width(x)


def _load_checkpoint(path):
    try: state=torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: state=torch.load(path,map_location="cpu")
    if isinstance(state,dict) and "model_state_dict" in state: state=state["model_state_dict"]
    return state


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--out",default="model/vgn.onnx"); args=ap.parse_args()
    net=ConvNet().eval(); net.load_state_dict(_load_checkpoint(args.checkpoint),strict=True); dummy=torch.zeros(1,1,40,40,40)
    torch.onnx.export(net,dummy,args.out,opset_version=13,input_names=["tsdf"],output_names=["quality","rotation","width"],dynamic_axes=None); print(args.out)


if __name__=="__main__": main()
