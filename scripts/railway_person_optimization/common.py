"""Paths and strict evidence helpers for the isolated railway optimization."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/railway_person_optimization'
OUTPUT = ROOT / 'output/railway_person_optimization'
APP = ROOT / 'railway_person_optimization'
BASE = ROOT / 'yolo26_optimization_8'
DATA = ROOT / 'output/railway_person_research/datasets/RailGoerl24_raw'
PROTOCOL = ROOT / 'reports/railway_person_research/w02_protocol_v1'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(path)

def rows(split, ids=None):
    data = [json.loads(s) for s in (PROTOCOL / (split+'.jsonl')).read_text(encoding='utf8').splitlines()]
    if ids is not None:
        index = {r['image_id']: r for r in data}
        return [index[i] for i in ids]
    return data
