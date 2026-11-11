"""Budgeted rectangular global pass and optional panoramic refinement."""
import time
import numpy as np

def nms(boxes,threshold=.5,limit=300):
    b=np.asarray(boxes,dtype=np.float32).reshape(-1,6)
    if not len(b):return b.copy()
    order=np.argsort(-b[:,4],kind='stable');keep=[]
    while len(order) and len(keep)<limit:
        i=int(order[0]);keep.append(i);order=order[1:]
        if not len(order):break
        low=np.maximum(b[i,:2],b[order,:2]);high=np.minimum(b[i,2:4],b[order,2:4]);intersection=np.maximum(high-low,0).prod(axis=1)
        area=np.maximum(b[i,2:4]-b[i,:2],0).prod();other=np.maximum(b[order,2:4]-b[order,:2],0).prod(axis=1)
        overlap=intersection/np.maximum(area+other-intersection,np.finfo(np.float32).tiny)
        order=order[overlap<=threshold]
    return b[keep].copy()

def choose_strip(image_shape,predictions,policy='top',height=384):
    h,w=image_shape[:2];height=min(height,h)
    if policy=='top':return (0,0,w,height)
    if policy!='adaptive':raise ValueError(policy)
    p=np.asarray(predictions).reshape(-1,6)
    eligible=p[(p[:,4]>=.01)&((p[:,3]-p[:,1])<192)]
    if not len(eligible):return (0,0,w,height)
    centers=(eligible[:,1]+eligible[:,3])/2
    weights=eligible[:,4]*(1-eligible[:,4])*np.minimum(1,64/np.maximum(eligible[:,3]-eligible[:,1],1))
    positions=sorted(set([0,h-height]+[int(np.clip(round(y-height/2),0,h-height)) for y in centers]))
    score=lambda y:float(weights[(centers>=y)&(centers<=y+height)].sum())
    y=max(positions,key=lambda y:(score(y),-y))
    return (0,y,w,y+height)

def analyze(global_engine,image,fine_engine=None,policy='top',score=.001):
    start=time.perf_counter();global_result=global_engine.infer(image,score=.001)
    coarse=global_result['predictions'];actions=[{'kind':'global','shape_hw':list(global_engine.hw),'seconds':global_result['analysis_seconds']}]
    if fine_engine is None:b=coarse
    else:
        x0,y0,x1,y1=choose_strip(image.shape,coarse,policy)
        fine=fine_engine.infer(np.ascontiguousarray(image[y0:y1,x0:x1]),score=.001)
        r=fine['predictions'].copy();r[:,[0,2]]+=x0;r[:,[1,3]]+=y0
        b=nms(np.concatenate([coarse,r]))
        actions.append({'kind':'refinement','crop_xyxy':[x0,y0,x1,y1],'shape_hw':list(fine_engine.hw),'seconds':fine['analysis_seconds']})
    return {'predictions':b[b[:,4]>=score].copy(),'analysis_seconds':time.perf_counter()-start,'actions':actions}
