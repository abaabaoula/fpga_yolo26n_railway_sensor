"""Rectangular O8 execution over an explicit newly compiled graph."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import cv2,numpy as np
from .geometry import letterbox,decode

class RectDetector:
    def __init__(self,base,candidate,device='pc',threads=2,options=4,plan=True,library=None,fast_golden=False,shadow_golden=False):
        base=Path(base).resolve();candidate=Path(candidate).resolve()
        info=json.loads((candidate/'preparation.json').read_text())
        self.hw=tuple(info['shape_hw']);self.device=device
        cv2.setNumThreads(threads)
        if device=='pc':
            import torch
            torch.set_num_threads(threads)
        spec=importlib.util.spec_from_file_location('_rect_native_graph',base/'native_graph.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        prog=candidate/'compiled'/('program_r'+str(self.hw[1]))
        if hashlib.sha256((prog/'graph.bin').read_bytes()).hexdigest()!=info['program_sha256']:
            raise RuntimeError('Candidate program SHA differs from preparation record')
        self.graph=mod.NativeGraph(self.hw[1],pc=device=='pc',program=str(prog),model_dir=str(candidate/'compiled/model'),plan=str(candidate/'compiled/resident_plan/resident_plan.bin') if plan else None,options=options,library=library,firmware_manifest=str(base/'hardware/accepted/firmware_manifest.json'))
        if device=='pc' and fast_golden:
            from .pc_integer import BoundedIntegerBackend
            self.graph.backend=BoundedIntegerBackend(shadow=shadow_golden)
        actual=self.graph.meta['shapes'][self.graph.meta['names'][self.graph.meta['input_id']]][0]
        if actual!=[1,3,*self.hw]:raise RuntimeError(('Wrong graph input',actual,self.hw))

    def infer(self,image,score=.001,return_raw=False):
        start=time.perf_counter()
        padded,_,_=letterbox(image,self.hw)
        g=self.graph
        rc=g.lib.o7_run_bgr(g.handle,padded.ctypes.data,g.raw.ctypes.data)
        if rc:raise RuntimeError(g.lib.o7_error().decode(),g.failure)
        g.lib.o7_timers(g.handle,g.timers.ctypes.data)
        raw=g.raw.copy()
        all_boxes=decode(raw,image.shape,self.hw,score)
        b=all_boxes[all_boxes[:,5]==0].copy()
        if not np.isfinite(raw).all() or raw.shape!=(1,300,6):raise RuntimeError('Invalid detector output')
        result={'predictions':b,'analysis_seconds':time.perf_counter()-start}
        if return_raw:result['raw']=raw
        return result

    def close(self):
        if getattr(self,'graph',None):self.graph.close();self.graph=None
