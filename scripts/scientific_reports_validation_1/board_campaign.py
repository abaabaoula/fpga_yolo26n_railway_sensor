"""Three actual-board paired-source rounds, controlled runtime/ROI factors."""
import fcntl
import hashlib
import json
import resource
import sys
import time
import traceback
from pathlib import Path
import cv2
import numpy as np

root=Path("/home/xilinx/jupyter_notebooks/scientific_reports_validation_1")
sv=root.parent/"railway_sensors_validation_1"
prior=root.parent/"railway_person_optimization"
rs1=root.parent/"railway_system_optimization_1"
base=root.parent/"yolo26_optimization_8"
budget=root.parent/"railway_budget_research"
sys.path[:0]=[str(sv),str(prior),str(budget)]
from railway_person_optimization.detector import RectDetector
from railway_budget_research.application import analyze
from railway_sensors_validation_1.strategies import scheduler
from railway_sensors_validation_1.ncnn_detector import NCNNDetector

out=root/"reports/campaign_v1"
out.mkdir(parents=True,exist_ok=True)
assert not (out/"complete.json").exists()
trial=json.loads((sv/"reports/ncnn_trial_v2/result.json").read_text())
assert trial["status"]=="PASS"
int8_trial=json.loads((sv/"reports/ncnn_int8_trial_v2/result.json").read_text())
assert int8_trial["status"]=="PASS"
protocol=json.loads((root/"protocol_v1.json").read_text())
pack=root/"board_pack_v1"
manifest=json.loads((pack/"manifest.json").read_text())
sources=manifest["images"]
assert len(sources)==240 and len({s["capture_group_id"] for s in sources})==6
for source in sources:
    image=cv2.imread(str(pack/source["file"]))
    assert hashlib.sha256(image.tobytes()).hexdigest()==source["decoded_BGR_sha256"],("JPEG decode differs; restage PNG",source["file"])
library=rs1/"builds/native_arm_v3/native_rs1.so"
assert hashlib.sha256(library.read_bytes()).hexdigest()=="3ec0523f64a00f19f14330af562e840d2e890b7890c858e1f74943ec6d1a0d43"
configs=[{'name': 'global_opt', 'strategy': 'global', 'options': 6276}, {'name': 'adaptive_opt', 'strategy': 'old_adaptive', 'options': 6276}, {'name': 'ncnn_global640', 'strategy': 'global', 'backend': 'ncnn'}, {'name': 'ncnn_adaptive', 'strategy': 'old_adaptive', 'backend': 'ncnn'}, {'name': 'ncnn_int8_adaptive', 'strategy': 'old_adaptive', 'backend': 'ncnn_int8'}, {'name': 'ncnn_whole1280', 'strategy': 'global', 'backend': 'ncnn'}]
contract={"configs":configs,"source_manifest_sha256":hashlib.sha256((pack/"manifest.json").read_bytes()).hexdigest(),
    "rounds":3,"warmups":"one cold + two further calls, all separately recorded",
    "timing":"Decoded BGR to final person detections at score .25. Read/init/warmups/status/logging excluded and separately recorded.",
    "order":"round0 forward, round1 reverse, round2 rotated; sequential contexts due CMA capacity",
    "backend_scope":"NCNN FP32 and official KL128 INT8, original one-to-one head; custom asymmetric INT8 arithmetic differs. No cross-backend bit-exact claim.",
    "source_selection":manifest["selection"],"programming":False,"runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
contract_path=out/"contract.json"
if contract_path.exists():assert json.loads(contract_path.read_text())==contract
else:contract_path.write_text(json.dumps(contract,indent=2))
lock=(base.parent/"yolo26_stage_d/reports/accelerator_job.lock").open("a+")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
memstat=Path("/sys/class/drm/card0/device/memstat_raw")
assert int(memstat.read_text().split()[0])==0
engines=[]
state={"status":"RUNNING","complete_phases":[],"programming":False}
def progress(phase,**extra):
    state.update(phase=phase,**extra)
    (out/"progress.json").write_text(json.dumps(state,indent=2))
def telemetry():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    status={line.split(":")[0]:line.split(":",1)[1].strip() for line in Path("/proc/self/status").read_text().splitlines()
            if line.split(":")[0] in ("VmRSS","VmHWM","VmSwap")}
    sensors={}
    for path in [Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),*Path("/sys/class/thermal").glob("thermal_zone*/temp")]:
        try:sensors[str(path)]=path.read_text().strip()
        except OSError:pass
    return {"process":status,"major_faults":usage.ru_majflt,"minor_faults":usage.ru_minflt,
            "memstat":memstat.read_text(),"sensors":sensors}
def create(config):
    if str(config.get("backend","")).startswith("ncnn"):
        profile="whole1280" if "whole1280" in config["name"] else "global640" if "global640" in config["name"] else "adaptive"
        chosen_trial=int8_trial if config["backend"]=="ncnn_int8" else trial
        threads=chosen_trial["selected_threads"].get(profile,chosen_trial["selected_threads"]["adaptive"])
        models="ncnn_int8_models" if config["backend"]=="ncnn_int8" else "ncnn_models"
        bridge=sv/"build_ncnn_v1/libsv_ncnn.so"
        g=NCNNDetector(sv/models/("736x1280" if profile=="whole1280" else "384x640"),threads,bridge=bridge)
        engines.append(g)
        f=None
        if profile=="adaptive":
            f=NCNNDetector(sv/models/"256x1280",threads,bridge=bridge);engines.append(f)
        return g,f
    kwargs={"device":"board","library":str(library),"options":config["options"],"plan":config.get("resident",True)}
    g=RectDetector(base,prior/"candidates/int8_384x640_railptq_v1",**kwargs);engines.append(g)
    f=None
    if config["strategy"]=="grid_sahi":f=g
    elif config["strategy"]!="global":
        f=RectDetector(base,prior/"candidates/int8_256x1280_railptq_top384_v1",**kwargs);engines.append(f)
    return g,f
def close():
    for engine in reversed(engines):engine.close()
    engines.clear()
    assert int(memstat.read_text().split()[0])==0,"BOs not released"
    # Release free allocator arenas uniformly between sequential contexts.
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)
def check_pc(config,source,predictions):
    if str(config.get("backend","")).startswith("ncnn"):
        return {"actual_ARM_output_used_for_accuracy":True,"PC_accuracy_transfer":False}
    reference=json.loads((root/"pc_reference_v1"/(hashlib.sha256(source["image_id"].encode()).hexdigest()[:24]+".json")).read_text())
    assert reference["source_sha256"]==source["source_sha256"]
    expected=np.asarray(reference["variants"][config["strategy"]]["predictions"],np.float32).reshape(-1,6)
    assert np.array_equal(expected,predictions),(config["name"],source["image_id"],expected.shape,predictions.shape)
    return {"PC_exact":True,"predictions":len(expected)}

try:
    cv2.setNumThreads(2)
    for round_index in range(3):
        order=configs if round_index==0 else list(reversed(configs)) if round_index==1 else configs[2:]+configs[:2]
        for config in order:
            phase_name=f"round{round_index}_{config['name']}"
            phase=out/phase_name
            if (phase/"result.json").exists() and json.loads((phase/"result.json").read_text())["status"]=="PASS":
                state["complete_phases"].append(phase_name);continue
            phase.mkdir(exist_ok=False)
            record={"status":"RUNNING","config":config,"round":round_index,"samples":[],"warmups":[]}
            stream=(phase/"samples.jsonl").open("a",buffering=1)
            try:
                start=time.perf_counter();g,f=create(config);record["init_seconds"]=time.perf_counter()-start
                image=cv2.imread(str(pack/sources[0]["file"]))
                for warmup in range(3):
                    result=analyze(g,f,image,scheduler(config["strategy"],protocol),.25)
                    record["warmups"].append({"cold":warmup==0,"ms":result["analysis_seconds"]*1000})
                    progress(phase_name+":warmup",warmup=warmup)
                policy=scheduler(config["strategy"],protocol)
                indices=list(range(len(sources)))
                if round_index==1:indices.reverse()
                elif round_index==2:indices=indices[80:]+indices[:80]
                for index in indices:
                    source=sources[index];image=cv2.imread(str(pack/source["file"]))
                    assert hashlib.sha256(image.tobytes()).hexdigest()==source["decoded_BGR_sha256"]
                    before=telemetry();result=analyze(g,f,image,policy,.25);after=telemetry()
                    check=check_pc(config,source,result["predictions"])
                    hardware=g.graph.status() if hasattr(g,"graph") else None
                    if hardware:assert int(hardware["error"],0)==0
                    item={"source":index,"image_id":source["image_id"],"group":source["capture_group_id"],
                          "ms":result["analysis_seconds"]*1000,"predictions":result["predictions"].tolist(),
                          "actions":result["actions"],"calls":len(result["actions"]),"python_components":result.get("python_components"),"before":before,"after":after,
                          "hardware":hardware,"check":check}
                    record["samples"].append(item);stream.write(json.dumps(item)+"\n")
                    progress(phase_name,source=index,samples_in_phase=len(record["samples"]))
                record["memory"]=[engine.graph.memory() for engine in engines if hasattr(engine,"graph")]
                record["status"]="PASS"
            except BaseException as error:
                record.update(status="FAIL",error=repr(error),traceback=traceback.format_exc());raise
            finally:
                close();record["closed_memstat"]=memstat.read_text()
                (phase/"result.json").write_text(json.dumps(record,indent=2));stream.close()
            state["complete_phases"].append(phase_name);progress(phase_name+":complete")
    state["status"]="PASS";progress("COMPLETE")
    (out/"complete.json").write_text(json.dumps(state,indent=2))
except BaseException as error:
    state.update(status="FAIL",error=repr(error),traceback=traceback.format_exc());progress("FAIL");raise
finally:
    if engines:close()
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    print(json.dumps(state),flush=True)
