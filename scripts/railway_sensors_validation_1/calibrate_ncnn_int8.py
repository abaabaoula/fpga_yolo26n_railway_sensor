"""Official NCNN KL PTQ baseline; frozen 128 development calibration IDs only."""
import os
import shutil
import subprocess
from common import *
import cv2
import numpy as np
from railway_person_optimization.geometry import preprocess
from railway_sensors_validation_1.ncnn_detector import NCNNDetector

tools=OUTPUT/"ncnn_tools_pc"
extension={"scope":"additional mature CPU INT8 baseline only; deployed O8/RS1 mathematics unchanged",
    "tool_version":"NCNN20261020","method":"official ncnn2table KL + ncnn2int8",
    "calibration":"reuse frozen 128 development image IDs; no new test/validation data and no labels",
    "training":False,"selection":"fixed official KL method, no accuracy-based calibration-method search"}
dump(REPORT/"baseline_int8_extension_v1.json",extension)
for shape in ("384x640","256x1280","736x1280"):
    out=OUTPUT/"ncnn_int8_models"/shape
    if (out/"verified.json").exists():
        print("REUSE_VERIFIED_INT8",shape,flush=True);continue
    out.mkdir(parents=True,exist_ok=True)
    hw=tuple(map(int,shape.split("x")))
    name="int8_256x1280_railptq_top384_v1" if shape=="256x1280" else "int8_384x640_railptq_v1"
    ids_path=ROOT/"output/railway_person_optimization"/name/"calibration_ids.json"
    ids=json.loads(ids_path.read_text());assert len(ids)==128
    heldout={r["image_sha256"] for split in ("validation","test") for r in rows(split)}
    assert not heldout&{r["image_sha256"] for r in ids}
    calib=OUTPUT/"ncnn_calibration"/shape;calib.mkdir(parents=True,exist_ok=True)
    paths=[]
    for index,row in enumerate(ids):
        target=calib/f"{index:03}.npy"
        if not target.exists():
            source=DATA/row["image_id"];assert sha(source)==row["image_sha256"]
            image=cv2.imread(str(source))
            if shape=="256x1280":image=np.ascontiguousarray(image[:384])
            tensor=preprocess(image,hw)[0][0]
            np.save(target,tensor)
        paths.append(target.resolve().as_posix())
    listing=out/"calibration_list.txt";listing.write_text("\n".join(paths)+"\n",encoding="utf8")
    source=OUTPUT/"ncnn_models"/shape
    table=out/"calibration_kl.table"
    commands=[]
    if not table.exists():
        command=[str(tools/"ncnn2table.exe"),str(source/"model.ncnn.param"),str(source/"model.ncnn.bin"),
                 str(listing),str(table),f"shape=[{hw[1]},{hw[0]},3]","type=1","method=kl","thread=2"]
        commands.append(command)
        print("CALIBRATE_KL",shape,flush=True)
        with (out/"calibration.log").open("w") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=out)
    command=[str(tools/"ncnn2int8.exe"),str(source/"model.ncnn.param"),str(source/"model.ncnn.bin"),
             str(out/"model.ncnn.param"),str(out/"model.ncnn.bin"),str(table)]
    commands.append(command)
    with (out/"quantize.log").open("w") as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=out)
    layers=[]
    for line in (out/"model.ncnn.param").read_text().splitlines()[2:]:
        fields=line.split()
        if fields and fields[0] in ("Convolution","ConvolutionDepthWise","Gemm","InnerProduct"):
            layers.append({"type":fields[0],"name":fields[1],"quantization_params":[x for x in fields if x.startswith("8=") or x.startswith("9=")]})
    metadata=json.loads((source/"verified.json").read_text())
    metadata.update(FP32=False,precision="NCNN official KL INT8 baseline",calibration_ids_sha256=sha(ids_path),
        calibration_count=128,calibration_table_sha256=sha(table),source_FP32_param_sha256=sha(source/"model.ncnn.param"),
        source_FP32_bin_sha256=sha(source/"model.ncnn.bin"),commands=commands,quantized_layer_metadata=layers,
        files={p.name:sha(p) for p in out.glob("model.ncnn.*")})
    dump(out/"verified.json",metadata)
    engine=NCNNDetector(out,pc_dependencies=OUTPUT/"deps_pc")
    reference=np.load(source/"reference.npz")
    raw=engine.forward(np.ascontiguousarray(reference["input"]))
    assert np.isfinite(raw).all() and raw.shape==reference["torch_raw"].shape
    np.savez(out/"reference.npz",input=reference["input"],ncnn_raw=raw,FP32_raw=reference["torch_raw"])
    engine.close();reference.close()
    dump(out/"smoke.json",{"status":"PASS","elements":raw.size,"finite":True,
        "scope":"model conversion smoke; accuracy separately evaluated, not FP32 equivalence"})
    print("INT8_READY",shape,len(layers),flush=True)
