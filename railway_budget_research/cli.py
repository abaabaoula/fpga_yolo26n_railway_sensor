"""Explicit normal entry. Board uses an existing O8 overlay with download=False."""
import argparse,json,time,sys,os,hashlib
from pathlib import Path
import cv2
from railway_person_optimization.detector import RectDetector
from .application import analyze
from .strategies import make_scheduler
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['pc','board'],default='pc');p.add_argument('--images',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--models-root',type=Path);p.add_argument('--o8-root',type=Path);p.add_argument('--method',default='selected');p.add_argument('--score',type=float,default=.25);p.add_argument('--warmups',type=int,default=0);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);root=a.root.resolve();board=a.device=='board';report_dir=root/'reports' if board else root/'reports/railway_budget_research';protocol=json.loads((report_dir/'protocol_v1.json').read_text());method=a.method
    if method=='selected':method=json.loads((report_dir/'selected_method.json').read_text())['selected']
    base=a.o8_root or (root.parent/'yolo26_optimization_8' if board else root/'yolo26_optimization_8');models=a.models_root or (root.parent/'railway_person_optimization/candidates' if board else root/'output/railway_person_optimization')
    library=root.parent/'railway_person_optimization/native_sources_v2/native_rail.so' if board else root/'railway_person_optimization/runtime/native_rail.dll'
    digest=hashlib.sha256()
    with library.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):digest.update(block)
    expected='e3184f05cd21e09813229189da9c5db3051757ee37dbbe4c7ad2180da5cf6859' if board else '8091dc1215ff8d4a12c5615eb2c80f9b9afb528f56a653a918846e55a3896a3c'
    if digest.hexdigest()!=expected:raise RuntimeError('Frozen native library SHA changed')
    lock=None;engines=[];result={'method':method,'device':a.device,'score':a.score,'programming':False,'cached_inference':False,'images':[],'timing_scope':'decoded BGR -> all inference/selection/fusion; init/cold/warmups separate; read/draw/save excluded'}
    try:
        if board:
            import fcntl
            lock=(base.parent/'yolo26_stage_d/reports/accelerator_job.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.perf_counter();g=RectDetector(base,models/'int8_384x640_railptq_v1',device=a.device,fast_golden=not board,library=str(library));engines.append(g)
        f=RectDetector(base,models/'int8_256x1280_railptq_top384_v1',device=a.device,fast_golden=not board,library=str(library));engines.append(f);result['init_seconds']=time.perf_counter()-start
        result['engine_memory']={'global':g.graph.memory(),'fine':f.graph.memory()}
        if not board:
            import psutil
            result['RSS_after_init_bytes']=psutil.Process().memory_info().rss
        model=json.loads((report_dir/'learned_router.json').read_text()) if method=='learned_router' else None
        scheduler=make_scheduler(method,protocol,model)
        image=cv2.imread(a.images[0]);assert image is not None
        for i in range(a.warmups):analyze(g,f,image,make_scheduler(method,protocol,model),a.score)
        for i,path in enumerate(a.images):
            image=cv2.imread(path)
            if image is None:raise RuntimeError('Cannot decode '+path)
            r=analyze(g,f,image,scheduler,a.score);result['images'].append({'source':str(Path(path).resolve()),'predictions':r['predictions'].tolist(),'analysis_seconds':r['analysis_seconds'],'decision':r['decision'],'actions':r['actions'],'python_components':r['python_components'],'first_unwarmed':i==0 and a.warmups==0})
            for b in r['predictions']:
                x0,y0,x1,y1=map(round,b[:4]);cv2.rectangle(image,(x0,y0),(x1,y1),(80,240,170),2)
            cv2.imwrite(str(a.output/(str(i).zfill(4)+'.jpg')),image)
        result['status']='PASS'
    finally:
        for engine in reversed(engines):engine.close()
        if lock:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        (a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'status':result['status'],'device':a.device,'method':method,'images':len(result['images']),'init_seconds':result['init_seconds']}))
if __name__=='__main__':main()
