"""Rectangular letterbox and inverse coordinates shared by PC and board."""
import cv2
import numpy as np

def letterbox(image, shape_hw):
    h,w=image.shape[:2];oh,ow=map(int,shape_hw)
    gain=min(oh/h,ow/w)
    nw,nh=round(w*gain),round(h*gain)
    small=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR) if (nw,nh)!=(w,h) else image
    left=round((ow-nw)/2-.1);top=round((oh-nh)/2-.1)
    padded=cv2.copyMakeBorder(small,top,oh-nh-top,left,ow-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    if padded.shape[:2]!=(oh,ow):raise RuntimeError('Letterbox shape mismatch')
    return np.ascontiguousarray(padded),gain,(left,top)

def preprocess(image,shape_hw):
    padded,gain,pad=letterbox(image,shape_hw)
    x=np.ascontiguousarray(padded[...,::-1].transpose(2,0,1)[None],dtype=np.float32)/np.float32(255)
    return x,gain,pad

def decode(raw,image_shape,shape_hw,score=.001):
    h,w=image_shape[:2];oh,ow=shape_hw
    gain=min(oh/h,ow/w);nw,nh=round(w*gain),round(h*gain)
    left=round((ow-nw)/2-.1);top=round((oh-nh)/2-.1)
    b=raw[0,raw[0,:,4]>=score].copy()
    b[:,[0,2]]=(b[:,[0,2]]-left)/gain
    b[:,[1,3]]=(b[:,[1,3]]-top)/gain
    b[:,[0,2]]=np.clip(b[:,[0,2]],0,w)
    b[:,[1,3]]=np.clip(b[:,[1,3]],0,h)
    return b
