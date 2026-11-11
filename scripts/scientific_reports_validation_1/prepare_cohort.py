"""Freeze 240 metadata-selected sources before inspecting their prediction/label outcomes."""
from common import *
import collections
import shutil
import cv2

manifest_path = REPORT / "joint240_rows.jsonl"
assert not manifest_path.exists(), "Selection already frozen; inspect existing files"
excluded = set()
for name in ("joint_validation_rows.jsonl", "joint_validation_v2_rows.jsonl"):
    excluded.update(json.loads(s)["image_id"] for s in (PRIOR_REPORT / name).read_text(encoding="utf8").splitlines())
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for row in rows("validation"):
    if row["image_id"] not in excluded:
        groups[row["capture_group_id"]][row["sequence_id"]].append(row)
assert len(groups) == 6
selected, allocation = [], []
for group, sequences in sorted(groups.items()):
    seqs = sorted(sequences)
    quotas = {s: 0 for s in seqs}
    for _ in range(40):
        eligible = [s for s in seqs if quotas[s] < len(sequences[s])]
        chosen = min(eligible, key=lambda s: (quotas[s], s))
        quotas[chosen] += 1
    for seq in seqs:
        members = sorted(sequences[seq], key=lambda r: (r["frame_index"], r["image_id"]))
        n = quotas[seq]
        indices = [round((len(members)-1)*(i+.5)/n) for i in range(n)]
        assert len(set(indices)) == n
        selected.extend(members[i] for i in indices)
        allocation.append({"group": group, "sequence": seq, "available": len(members), "selected": n})
selected.sort(key=lambda r: (r["capture_group_id"], r["sequence_id"], r["frame_index"], r["image_id"]))
assert len(selected) == len({r["image_id"] for r in selected}) == len({r["image_sha256"] for r in selected}) == 240
assert not excluded.intersection(r["image_id"] for r in selected)
# Commit exact rows first: no model or annotation-derived selection criteria above.
manifest_path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in selected), encoding="utf8")
out = OUTPUT / "board_pack_v1"
out.mkdir(exist_ok=False)
manifest = {"selection": "40 per capture group; sequence-balanced, midpoint quantiles, metadata only",
    "data_role": "previously viewed validation; group balanced, not prevalence or unseen evaluation",
    "source_rows_sha256": sha(manifest_path), "images": []}
for index, row in enumerate(selected):
    source = DATA / row["image_id"]
    assert sha(source) == row["image_sha256"]
    target = out / f"image_{index:03}.jpg"
    shutil.copyfile(source, target)
    image = cv2.imread(str(source))
    assert image is not None and image.shape[:2] == (row["height"], row["width"])
    manifest["images"].append({"file": target.name, "image_id": row["image_id"], "source_sha256": sha(target),
        "decoded_BGR_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "capture_group_id": row["capture_group_id"],
        "sequence_id": row["sequence_id"], "width": row["width"], "height": row["height"]})
dump(out / "manifest.json", manifest)
persons = [o for r in selected for o in r["objects"] if o["class_name_raw"] == "person"]
dump(REPORT / "cohort_selection_audit.json", {"status": "PASS", "images": len(selected), "groups": len(groups),
    "sequences": len(allocation), "allocation": allocation, "excluded_previous_sources": len(excluded),
    "selection_did_not_use_labels_or_predictions": True, "GT_counted_after_selection_commit": True,
    "supplied_person_boxes": len(persons), "h_lt64": sum(o["bbox_xyxy_raw"][3]-o["bbox_xyxy_raw"][1] < 64 for o in persons),
    "rows_sha256": sha(manifest_path), "board_manifest_sha256": sha(out / "manifest.json"),
    "encoded_bytes": sum((out / s["file"]).stat().st_size for s in manifest["images"])})
print(json.dumps(json.loads((REPORT / "cohort_selection_audit.json").read_text())))
