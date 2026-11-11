"""Verify mature FP32 one-to-one output against frozen pre-existing ONNX."""
from common import *
import cv2
import numpy as np
import onnxruntime as ort
from railway_sensors_validation_1.ncnn_detector import NCNNDetector, official_topk
from railway_person_optimization.geometry import preprocess

lookup = {r["image_id"]: r for r in rows("development")}
ids = json.loads((ROOT / "reports/railway_person_optimization/development_smoke_64.json").read_text())[:5]
checks = []
for text, old_name in (("384x640", "int8_384x640_railptq_v1"),
                       ("256x1280", "int8_256x1280_railptq_top384_v1")):
    engine = NCNNDetector(OUTPUT / "ncnn_models" / text, pc_dependencies=OUTPUT / "deps_pc")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    frozen = ROOT / "output/railway_person_optimization" / old_name / "official_source.onnx"
    session = ort.InferenceSession(str(frozen), options, providers=["CPUExecutionProvider"])
    for image_id in ids:
        row = lookup[image_id]
        assert sha(DATA / image_id) == row["image_sha256"]
        image = cv2.imread(str(DATA / image_id))
        if text == "256x1280":
            image = image[:384]
        tensor = preprocess(image, engine.hw)[0]
        expected = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        raw = engine.forward(tensor)
        actual = official_topk(raw)
        mask = expected[0, :, 4] >= .01
        # Low-score ties may reorder. Compare above-floor detections as a set.
        a = actual[0]
        errors = []
        for box in expected[0, mask]:
            candidates = a[a[:, 5] == box[5]]
            assert len(candidates)
            distances = np.max(np.abs(candidates[:, :4] - box[:4]), axis=1)
            j = int(np.argmin(distances))
            errors.append((float(distances[j]), float(abs(candidates[j, 4] - box[4]))))
        assert max((x[0] for x in errors), default=0) < .2, errors
        assert max((x[1] for x in errors), default=0) < .003, errors
        checks.append({"shape": text, "image_id": image_id, "ONNX_sha256": sha(frozen),
                       "above_001": len(errors), "max_coordinate_error": max((x[0] for x in errors), default=0),
                       "max_score_error": max((x[1] for x in errors), default=0)})
    engine.close()
dump(REPORT / "ncnn_frozen_onnx_checks.json", {"status": "PASS", "checks": checks,
     "scope": "FP32 numerical closeness, not bit-exact. Frozen one-to-one ONNX; no automatic one-to-many head switch."})
print(json.dumps({"status": "PASS", "checks": len(checks), "detections": sum(c["above_001"] for c in checks)}))
