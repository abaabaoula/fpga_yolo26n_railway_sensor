"""Run the preserved ROI policy with the independently selected native runtime."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from railway_person_optimization.detector import RectDetector
from railway_budget_research.application import analyze
from railway_budget_research.strategies import make_scheduler
from .cpu_detector import CPUDetector

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=["pc","board","cpu"],default="pc")
    p.add_argument("--profile",choices=["candidate","accepted"],default="candidate")
    p.add_argument("--images",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--score",type=float,default=.25)
    p.add_argument("--warmups",type=int,default=0)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    pc_layout=(root/"yolo26_optimization_8").is_dir()
    base=(root if pc_layout else root.parent)/"yolo26_optimization_8"
    prior=(root if pc_layout else root.parent)/"railway_person_optimization"
    budget=(root if pc_layout else root.parent)/"railway_budget_research"
    models=root/"output/railway_person_optimization" if pc_layout else prior/"candidates"
    if pc_layout:
        library=(root/"output/railway_system_optimization_1/native_pc_v3/native_rs1.dll"
                 if a.profile=="candidate" else prior/"runtime/native_rail.dll")
        protocol=root/"reports/railway_budget_research/protocol_v1.json"
    else:
        library=(root/"builds/native_arm_v3/native_rs1.so"
                 if a.profile=="candidate" else prior/"native_sources_v2/native_rail.so")
        protocol=budget/"reports/protocol_v1.json"
    expected=("cefbb097ea028e0947ff784b11269092f2aacbb5d5b5a022097be99c98a483b8" if pc_layout else
              "3ec0523f64a00f19f14330af562e840d2e890b7890c858e1f74943ec6d1a0d43")
    if a.profile=="candidate":
        assert hashlib.sha256(library.read_bytes()).hexdigest()==expected,"Candidate library differs from tested build"
    assert a.device!="board" or not pc_layout,"Board mode requires the PYNQ layout"
    a.output.mkdir(parents=True,exist_ok=False)
    lock=None
    if a.device=="board":
        import fcntl
        lock=(base.parent/"yolo26_stage_d/reports/accelerator_job.lock").open("a+")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    engines=[]
    record={"device":a.device,"profile":a.profile,"score":a.score,"programming":False,
            "ROI":"preserved old_adaptive","library_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),
            "predictions_cached":False,"records":[]}
    try:
        start=time.perf_counter()
        options=6276 if a.profile=="candidate" else 4
        for name in ["int8_384x640_railptq_v1","int8_256x1280_railptq_top384_v1"]:
            if a.device=="cpu":
                assert a.profile=="candidate","CPU baseline uses the new native kernel"
                e=CPUDetector(models/name,library,options)
            else:
                e=RectDetector(base,models/name,device=a.device,library=str(library),
                               options=options,fast_golden=a.device=="pc")
            engines.append(e)
        record["init_seconds"]=time.perf_counter()-start
        config=json.loads(protocol.read_text())
        first=cv2.imread(str(a.images[0]))
        assert first is not None
        record["warmups_ms"]=[]
        for i in range(a.warmups):
            r=analyze(*engines,first,make_scheduler("old_adaptive",config),a.score)
            record["warmups_ms"].append(r["analysis_seconds"]*1000)
        for i,path in enumerate(a.images):
            read=time.perf_counter();image=cv2.imread(str(path));assert image is not None,path
            read_seconds=time.perf_counter()-read
            r=analyze(*engines,image,make_scheduler("old_adaptive",config),a.score)
            before=time.perf_counter();draw=image.copy()
            for box in r["predictions"]:
                x0,y0,x1,y1=map(lambda x:int(round(float(x))),box[:4])
                cv2.rectangle(draw,(x0,y0),(x1,y1),(100,240,150),2)
                cv2.putText(draw,"person %.2f"%box[4],(x0,max(12,y0-4)),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(100,240,150),1)
            target=a.output/("%04d.jpg"%i);assert cv2.imwrite(str(target),draw)
            record["records"].append({"input":str(Path(path).resolve()),"image_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "predictions":r["predictions"].tolist(),"analysis_ms":r["analysis_seconds"]*1000,
                "read_seconds":read_seconds,"draw_save_seconds":time.perf_counter()-before,
                "actions":r["actions"],"decision":r["decision"],
                "cold_unwarmed":i==0 and a.warmups==0,"output":str(target.resolve())})
        record["status"]="PASS"
    finally:
        for e in reversed(engines):e.close()
        if lock:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        (a.output/"result.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"status":record["status"],"records":len(record["records"]),
                      "analysis_ms":[x["analysis_ms"] for x in record["records"]]}))

if __name__=="__main__":
    main()
