"""New-image entry point; PC FP32 rectangles or matching loaded O8 board."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--profile',choices=['pc_fast','pc_quality','compact','balanced'],default='balanced')
    p.add_argument('--device',choices=['pc','board'],default='pc')
    p.add_argument('--images',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--score',type=float);p.add_argument('--warmups',type=int,default=0)
    p.add_argument('--o8-root',type=Path);p.add_argument('--models-root',type=Path);p.add_argument('--selection',type=Path)
    a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root));a.output.mkdir(parents=True,exist_ok=False)
    import cv2,numpy as np
    from railway_person_optimization.fp32 import FP32Detector
    from railway_person_optimization.detector import RectDetector
    from railway_person_optimization.application import analyze
    base=a.o8_root or ((root.parent if a.device=='board' else root)/'yolo26_optimization_8')
    models=a.models_root or (root/'candidates' if a.device=='board' else root/'output/railway_person_optimization')
    selection=a.selection or (root/'config/selected_profiles.json' if (root/'config/selected_profiles.json').is_file() else root/'reports/railway_person_optimization/selected_profiles.json')
    selected=json.loads(selection.read_text())
    pc_profile=a.profile.startswith('pc_');coarse=fine=None;lock=None;st=time.perf_counter()
    try:
        if pc_profile:
            if a.device!='pc':raise RuntimeError('FP32 profiles currently support the PC CPU only; PYNQ ARM FP32 has not been deployed.')
            side=1280 if a.profile=='pc_fast' else 1536
            coarse=FP32Detector(base/'reproduction_inputs/yolo26n.pt',long_side=side,config_dir=a.output/'ultralytics')
            score=.25 if a.score is None else a.score
            identity={'backend':'official frozen FP32 PC CPU','long_side':side,'padding':'minimum stride32 rectangle per input aspect ratio','weights_sha256':digest(base/'reproduction_inputs/yolo26n.pt')}
        else:
            key='compact' if a.profile=='compact' else selected['balanced_selected'];spec=selected['selected_variants'][key]
            if a.device=='board':
                import fcntl
                lock=Path('/home/xilinx/jupyter_notebooks/yolo26_stage_d/reports/accelerator_job.lock').open('a+')
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            library=root/'railway_person_optimization/runtime'/('native_rail.dll' if a.device=='pc' else 'native_rail.so')
            if not library.is_file():raise RuntimeError('Run the documented runtime preparation or use the completed portable release: '+str(library))
            manifest=root/'railway_person_optimization/runtime/manifest.json'
            info=json.loads(manifest.read_text());expected=info['PC' if a.device=='pc' else 'ARM']['sha256']
            if digest(library)!=expected:raise RuntimeError('Runtime SHA mismatch')
            fine=RectDetector(base,models/spec['fine'],device=a.device,library=str(library),fast_golden=a.device=='pc')
            coarse=RectDetector(base,models/spec['global'],device=a.device,library=str(library),fast_golden=a.device=='pc')
            score=selected['scores'][key] if a.score is None else a.score
            identity={'backend':'actual O8 FPGA/ARM' if a.device=='board' else 'PC exact integer/native reference; not board latency','variant':key,**spec,'library_sha256':expected,'programming':False}
        initialized=time.perf_counter()-st;records=[]
        for i,path in enumerate(a.images):
            image=cv2.imread(str(path))
            if image is None:raise RuntimeError('Cannot decode '+str(path))
            warm=[]
            for _ in range(a.warmups):warm.append(coarse.infer(image)['analysis_seconds'] if pc_profile else analyze(coarse,image,fine,spec['policy'],score)['analysis_seconds'])
            r=coarse.infer(image,score=.001) if pc_profile else analyze(coarse,image,fine,spec['policy'],score=.001)
            boxes=r['predictions'];visible=boxes[boxes[:,4]>=score]
            canvas=image.copy()
            for x0,y0,x1,y1,confidence,cls in visible:
                cv2.rectangle(canvas,(round(x0),round(y0)),(round(x1),round(y1)),(0,220,255),2)
                cv2.putText(canvas,'person %.2f'%confidence,(max(0,round(x0)),max(18,round(y0)-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,255),2)
            cv2.imwrite(str(a.output/f'{i:03d}_detections.jpg'),canvas)
            row={'source':str(path.resolve()),'source_sha256':digest(path),'initialization_seconds':initialized if i==0 else None,'warmups_seconds':warm,'analysis_seconds':r['analysis_seconds'],'score':score,'actual_global_shape_hw':coarse.hw,'predictions_at_floor_0001':boxes.tolist(),'display_predictions':visible.tolist(),'actions':r.get('actions'),'identity':identity}
            (a.output/f'{i:03d}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8');records.append(row)
            print(json.dumps({'image':str(path),'profile':a.profile,'device':a.device,'people':len(visible),'analysis_ms':r['analysis_seconds']*1000,'shape_hw':coarse.hw},ensure_ascii=False),flush=True)
        (a.output/'summary.json').write_text(json.dumps({'profile':a.profile,'device':a.device,'images':len(records),'identity':identity},indent=2))
    finally:
        if coarse:coarse.close()
        if fine:fine.close()
        if lock:lock.close()

if __name__=='__main__':main()
