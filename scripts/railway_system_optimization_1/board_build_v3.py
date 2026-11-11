"""Build only in the independent workspace; no device/configuration writes."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path("/home/xilinx/jupyter_notebooks/railway_system_optimization_1")
src = root / "native_sources/candidate_v3"
out = root / "builds/native_arm_v3"
out.mkdir(parents=True, exist_ok=False)
manifest = json.loads((src / "source_manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((src / name).read_bytes()).hexdigest() == digest, name
cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fopenmp", "-ffp-contract=off", "-fPIC",
       "-mcpu=cortex-a9", "-mfpu=neon", "-mfloat-abi=hard", str(src / "queue_runtime.cpp"),
       "-o", str(out / "native_rs1.so"), "-ldl"]
with (out / "build.log").open("w") as f:
    r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=180)
record = {"command": cmd, "exit": r.returncode, "programming": False,
          "sources": manifest["files"]}
if r.returncode == 0:
    record["sha256"] = hashlib.sha256((out / "native_rs1.so").read_bytes()).hexdigest()
(out / "build.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
assert r.returncode == 0
