"""Deployable detector orchestration. No prediction cache or ground truth."""
import time
import numpy as np
from railway_person_optimization.application import nms
from .policy import Scheduler,geometry_hint
def analyze(global_engine,fine_engine,image,scheduler=None,score=.25,early_filter=True):
    start=time.perf_counter();scheduler=scheduler or Scheduler()
    g=global_engine.infer(image,score=.001)
    decision_start=time.perf_counter()
    hint=geometry_hint(image) if scheduler.config.geometry else None
    decision=scheduler.decide(image.shape,g['predictions'],hint)
    decision_seconds=time.perf_counter()-decision_start
    boxes=[g['predictions']];actions=[{'kind':'global','shape_hw':list(global_engine.hw),'seconds':g['analysis_seconds']}]
    for x0,y0,x1,y1 in decision['crops']:
        r=fine_engine.infer(np.ascontiguousarray(image[y0:y1,x0:x1]),score=.001)
        b=r['predictions'].copy();b[:,[0,2]]+=x0;b[:,[1,3]]+=y0;boxes.append(b)
        actions.append({'kind':'fine','crop_xyxy':[x0,y0,x1,y1],'shape_hw':list(fine_engine.hw),'seconds':r['analysis_seconds']})
    fusion_start=time.perf_counter();combined=np.concatenate(boxes)
    # In descending-score greedy NMS, a box below the requested output score
    # cannot suppress an above-threshold box. Keep all coarse candidates for
    # routing, but omit irrelevant low-score boxes from final NMS only.
    if early_filter:combined=combined[combined[:,4]>=score]
    b=nms(combined)
    return {'predictions':b[b[:,4]>=score].copy(),'analysis_seconds':time.perf_counter()-start,'decision':decision,'actions':actions,'early_filter':early_filter,'python_components':{'decision_seconds':decision_seconds,'fusion_seconds':time.perf_counter()-fusion_start}}
