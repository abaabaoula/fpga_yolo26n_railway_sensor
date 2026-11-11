"""Joint-source accuracy and mature-backend historical-test comparison."""
import argparse
import collections
import importlib.util
import os
from common import *
os.environ["OMP_NUM_THREADS"]="2"
import cv2
import numpy as np
from railway_person_optimization.detector import RectDetector
from railway_budget_research.application import analyze
from railway_sensors_validation_1.strategies import scheduler
from railway_sensors_validation_1.ncnn_detector import NCNNDetector
sys.path.insert(0,str(ROOT/"railway_person_research"))
spec=importlib.util.spec_from_file_location("_matrix_metrics",ROOT/"scripts/railway_person_research/evaluate_matrix.py")
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)

parser=argparse.ArgumentParser()
parser.add_argument("--split",choices=["joint","historical","external"],default="joint")
args=parser.parse_args()
if args.split=="joint":
    manifest=REPORT/"joint_validation_rows.jsonl";dataset=DATA
    methods=["global","fixed_top","old_adaptive","fixed_focus","roi_score","grid_sahi","ncnn_adaptive","ncnn_whole1280"]
elif args.split=="historical":
    manifest=PROTOCOL/"test.jsonl";dataset=DATA;methods=["ncnn_adaptive","ncnn_whole1280"]
else:
    manifest=REPORT/"external_new_available.jsonl";dataset=OUTPUT/"datasets/OSDaR23_new"
    methods=["global","fixed_top","old_adaptive","fixed_focus","roi_score","grid_sahi","ncnn_adaptive","ncnn_whole1280"]
selected=[json.loads(line) for line in manifest.read_text().splitlines()]
out=OUTPUT/("accuracy_"+args.split+"_v1");out.mkdir(exist_ok=True)
protocol=json.loads((ROOT/"reports/railway_budget_research/protocol_v1.json").read_text())
contract={"manifest_sha256":sha(manifest),"image_ids":[r["image_id"] for r in selected],"methods":methods,
          "primary_score":.25,"inference_floor":.001,"selection":"frozen methods; no selection on this matrix",
          "role":"previously viewed historical test" if args.split=="historical" else args.split,
          "timing":"PC accuracy computation and accuracy cache only; not measured board latency"}
cp=out/"contract.json"
if cp.exists():assert json.loads(cp.read_text())==contract
else:dump(cp,contract)
g=f=ng=nf=large=None
try:
    if any(not n.startswith("ncnn") for n in methods):
        common={"device":"pc","fast_golden":True,"library":str(ROOT/"output/railway_system_optimization_1/native_pc_v3/native_rs1.dll"),"options":6276}
        g=RectDetector(ROOT/"yolo26_optimization_8",ROOT/"output/railway_person_optimization/int8_384x640_railptq_v1",**common)
        f=RectDetector(ROOT/"yolo26_optimization_8",ROOT/"output/railway_person_optimization/int8_256x1280_railptq_top384_v1",**common)
    ng=NCNNDetector(OUTPUT/"ncnn_models/384x640",pc_dependencies=OUTPUT/"deps_pc")
    nf=NCNNDetector(OUTPUT/"ncnn_models/256x1280",pc_dependencies=OUTPUT/"deps_pc")
    large=NCNNDetector(OUTPUT/"ncnn_models/736x1280",pc_dependencies=OUTPUT/"deps_pc")
    for index,row in enumerate(selected):
        path=out/(key(row["image_id"])+".json")
        if path.exists():
            assert json.loads(path.read_text())["contract_sha256"]==sha(cp)
            continue
        source=dataset/row["image_id"];assert sha(source)==row["image_sha256"]
        image=cv2.imread(str(source));variants={}
        for name in methods:
            if name=="ncnn_adaptive":
                result=analyze(ng,nf,image,scheduler("old_adaptive",protocol),.001)
            elif name=="ncnn_whole1280":
                result=analyze(large,None,image,scheduler("global",protocol),.001)
            else:
                result=analyze(g,g if name=="grid_sahi" else f,image,scheduler(name,protocol),.001)
            variants[name]={"predictions":result["predictions"].tolist(),"actions":result["actions"],
                            "calls":len(result["actions"])}
        dump(path,{"image_id":row["image_id"],"source_sha256":row["image_sha256"],"variants":variants,
                   "contract_sha256":sha(cp)})
        if (index+1)%10==0:print("RAIL_MATRIX",args.split,index+1,len(selected),flush=True)
finally:
    for engine in (large,nf,ng,f,g):
        if engine:engine.close()
summary={}
for name in methods:
    per=[];predictions={}
    for row in selected:
        record=json.loads((out/(key(row["image_id"])+".json")).read_text())
        boxes=np.asarray(record["variants"][name]["predictions"],np.float32).reshape(-1,6)
        predictions[row["image_id"]]=boxes
        per.append({"image_id":row["image_id"],"group":row["capture_group_id"],**metrics.counts(boxes,row,.25)})
    total={k:sum(p[k] for p in per) for k in per[0] if k not in ("image_id","group")}
    result={"primary":metrics.rates(total,len(per)),"per_group":{}}
    for group in sorted({p["group"] for p in per}):
        subset=[p for p in per if p["group"]==group]
        result["per_group"][group]=metrics.rates({k:sum(p[k] for p in subset) for k in total},len(subset))
    if total["gt"]:result["annotation_relative_AP"]=metrics.coco_metrics(ROOT,selected,predictions)
    summary[name]=result
    dump(REPORT/("accuracy_"+args.split+"_"+name+"_per_image.json"),per)
dump(REPORT/("accuracy_"+args.split+"_summary.json"),{"status":"PASS","images":len(selected),
     "groups":len({r["capture_group_id"] for r in selected}),"contract_sha256":sha(cp),"methods":summary,
     "partial_labels":True,"UPI_is_certified_false_positive_rate":False,"role":contract["role"]})
print(json.dumps({n:{"recall":r["primary"]["recall"],"small":r["primary"]["small_recall"]} for n,r in summary.items()}),flush=True)
