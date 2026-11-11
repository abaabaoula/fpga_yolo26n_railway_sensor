"""Freeze supplemental baselines and final imported external sources before inference."""
from common import *
audit=json.loads((REPORT/"external_local_audit.json").read_text())
assert audit["status"]=="ALL_LOCKED_GROUPS_IMPORTED"
scripts=["railway_matrix_pc.py","railway_matrix_int8_pc.py"]
record={"date":"2026-11-02","manifest_sha256":sha(REPORT/"external_new_available.jsonl"),
        "images":audit["images"],"groups":audit["groups"],"primary_policy_sha256":sha(REPORT/"methods_frozen_before_external_v1.json"),
        "supplemental_baselines":["grid_sahi","ncnn_adaptive","ncnn_whole1280","ncnn_int8_adaptive","ncnn_int8_whole1280","ncnn_global640","ncnn_int8_global640"],
        "supplement_scope":"Standard baseline additions after initial group7 coverage audit (0 supplied persons); fixed before any new external predictions. Primary methods and thresholds were frozen earlier.",
        "score":.25,"AP_floor":.001,"match_IoU":.5,"no_threshold_or_method_selection_on_external":True,
        "annotation_handling":"Original metadata-only uniform selection retained. Selected frame without official URI annotation is unknown and excluded, never a negative or replaced.",
        "source_hashes":{n:sha(ROOT/"scripts/railway_sensors_validation_1"/n) for n in scripts},
        "models":{str(p.relative_to(OUTPUT)):sha(p) for folder in ("ncnn_models","ncnn_int8_models")
                  for p in sorted((OUTPUT/folder).glob("*/*")) if p.name in ("model.ncnn.param","model.ncnn.bin","verified.json")}}
p=REPORT/"external_final_freeze_v1.json"
if p.exists():assert json.loads(p.read_text())==record
else:dump(p,record)
print(json.dumps({"external_frozen":True,"images":audit["images"],"boxes":audit["boxes"],"groups":audit["groups"]}))
