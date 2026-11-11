import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
NAME = "railway_sensors_validation_1"
REPORT = ROOT / "reports" / NAME
OUTPUT = ROOT / "output" / NAME
DATA = ROOT / "output/railway_person_research/datasets/RailGoerl24_raw"
PROTOCOL = ROOT / "reports/railway_person_research/w02_protocol_v1"
sys.path.insert(0, str(ROOT))
def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
def dump(path, value):
    path = Path(path).resolve()
    assert any(path.is_relative_to(p.resolve()) for p in (REPORT, OUTPUT))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")
    temporary.replace(path)
def key(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]
def rows(split):
    return [json.loads(line) for line in (PROTOCOL / (split + ".jsonl")).read_text(encoding="utf8").splitlines()]
