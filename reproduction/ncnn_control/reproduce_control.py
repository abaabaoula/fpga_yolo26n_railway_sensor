"""Run all5000 COCO person images in an isolated venv and verify old per-image predictions."""
import argparse,contextlib,hashlib,importlib.metadata,io,json,sys,time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--payload',type=Path,required=True)
parser.add_argument('--coco-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();payload=args.payload.resolve();out=args.output.resolve()
out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(payload/'src'))
import cv2,numpy as np,ncnn,pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ncnn_detector import NCNNDetector

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((payload/'manifest.json').read_text())
for name,record in manifest['files'].items():assert sha(payload/name)==record['sha256']
cfg=(Path(sys.prefix)/'pyvenv.cfg').read_text()
assert 'include-system-site-packages = false' in cfg
modules={m.__name__:str(Path(m.__file__).resolve()) for m in (np,cv2,ncnn,pycocotools)}
assert all(Path(p).is_relative_to(Path(sys.prefix).resolve()) for p in modules.values())
assert not any('yolo26_stage_b' in p or 'deps_pc' in p for p in sys.path)
annotation=args.coco_root/'annotations/instances_val2017.json'
previous=json.loads((payload/'reference_metrics.json').read_text())
references=json.loads((payload/'reference_hashes.json').read_text())
api=COCO(str(annotation));ids=sorted(api.getImgIds());assert len(ids)==5000
environment={'python':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'modules':modules,
    'versions':{n:importlib.metadata.version(n) for n in ('numpy','opencv-python','ncnn','pycocotools')},
    'system_site_packages':False,'status':'RUNNING','scope':manifest['scope']}
(out/'environment.json').write_text(json.dumps(environment,indent=2))
predictions=[];mismatches=[];engine=NCNNDetector(payload/'models/384x640')
cv2.setNumThreads(2);started=time.perf_counter()
try:
    with (out/'predictions.jsonl').open('x',buffering=1) as stream:
        for index,iid in enumerate(ids):
            path=args.coco_root/'val2017'/api.imgs[iid]['file_name']
            expected=references[str(iid)];assert sha(path)==expected['image_sha256']
            image=cv2.imread(str(path));assert image is not None
            boxes=engine.infer(image,.001)['predictions']
            values=[{'image_id':iid,'category_id':1,'bbox':[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])],'score':float(b[4])} for b in boxes]
            digest=hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if digest!=expected['predictions_sha256']:mismatches.append(iid)
            stream.write(json.dumps({'image_id':iid,'image_sha256':expected['image_sha256'],'predictions':values})+'\n')
            predictions.extend(values)
            if (index+1)%100==0:
                (out/'progress.json').write_text(json.dumps({'status':'RUNNING','images':index+1,'prediction_hash_differences':len(mismatches)}))
                print('CLEAN_CONTROL',index+1,len(mismatches),flush=True)
finally:engine.close()
capture=io.StringIO()
with contextlib.redirect_stdout(capture):
    result=api.loadRes(predictions);ev=COCOeval(api,result,'bbox');ev.params.catIds=[1];ev.params.imgIds=ids
    ev.evaluate();ev.accumulate();ev.summarize()
metrics={'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),'AP75':float(ev.stats[2]),'AP_small_area':float(ev.stats[3]),'AR100':float(ev.stats[8])}
differences={k:metrics[k]-previous[k] for k in metrics}
record={'status':'PASS' if not mismatches and all(abs(d)<1e-12 for d in differences.values()) else 'DIFFERENCES_REQUIRE_REVIEW',
    'images':len(ids),'prediction_hash_difference_images':mismatches,'metrics':metrics,'difference_from_previous':differences,
    'annotations_sha256':sha(annotation),'wall_s':time.perf_counter()-started,'PC_latency_claim':False,
    'stdout':capture.getvalue(),'scope':manifest['scope']}
(out/'result.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
assert record['status']=='PASS'
