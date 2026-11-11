import os
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / "railway_system_optimization_1/native_sources/candidate_v3"
out = root / "output/railway_system_optimization_1/native_pc_v3"
out.mkdir(parents=True, exist_ok=False)
compiler = os.environ.get("RAILWAY_CXX", "g++")
cmd = [compiler, "-O3", "-std=c++17", "-shared", "-fopenmp", "-ffp-contract=off",
       "-static-libgcc", "-static-libstdc++", str(source / "queue_runtime.cpp"),
       "-Wl,-Bstatic", "-lgomp", "-Wl,-Bdynamic", "-o", str(out / "native_rs1.dll")]
with (out / "build.log").open("w") as f:
    r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=180)
record = {"command": cmd, "exit": r.returncode}
if r.returncode == 0:
    record["sha256"] = hashlib.sha256((out / "native_rs1.dll").read_bytes()).hexdigest()
(out / "build.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record))
assert r.returncode == 0
