"""Frozen official weights with an explicit rectangular FP32 input."""
import os,time
from pathlib import Path
import cv2,numpy as np

class FP32Detector:
    def __init__(self,weights,shape_hw=(736,1280),threads=2,config_dir=None,long_side=None):
        os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_OFFLINE']='true'
        if config_dir:os.environ['YOLO_CONFIG_DIR']=str(config_dir)
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(threads);cv2.setNumThreads(threads)
        self.hw=tuple(shape_hw);self.long_side=long_side;self.model=YOLO(str(weights),task='detect');self.shapes=[]
        self.hook=self.model.model.register_forward_pre_hook(lambda m,args:self.shapes.append((list(args[0].shape),str(args[0].dtype))))
    def infer(self,image,score=.001):
        start=time.perf_counter()
        if self.long_side:
            h,w=image.shape[:2];gain=min(self.long_side/h,self.long_side/w);nh,nw=round(h*gain),round(w*gain)
            self.hw=(nh+(self.long_side-nh)%32,nw+(self.long_side-nw)%32)
        r=self.model.predict(image,imgsz=self.long_side or self.hw,rect=bool(self.long_side),conf=score,max_det=300,device='cpu',verbose=False,augment=False)[0]
        actual,dtype=self.shapes[-1]
        if actual!=[1,3,*self.hw] or dtype!='torch.float32':raise RuntimeError((actual,dtype))
        b=r.boxes.data.cpu().numpy();b=b[b[:,5]==0].copy()
        if not np.isfinite(b).all():raise RuntimeError('Invalid FP32 predictions')
        self.shapes.clear()
        return {'predictions':b,'analysis_seconds':time.perf_counter()-start}
    def close(self):self.hook.remove()
