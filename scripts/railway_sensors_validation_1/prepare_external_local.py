"""Import only pre-locked RGB/label members from user-downloaded official ZIPs."""
import argparse
import collections
import zipfile
import zlib
from common import *
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    args = parser.parse_args()
    plan_path = ROOT / "reports/railway_system_optimization_1/external_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    policy = REPORT / "methods_frozen_before_external_v1.json"
    frozen = {"contract_sha256": sha(REPORT / "contract_v1.json"),
              "plan_sha256": sha(plan_path), "primary": "old_adaptive + RS1 options6276",
              "methods": ["global", "fixed_top", "old_adaptive", "fixed_focus", "roi_score"],
              "score": .25, "AP_score_floor": .001, "IoU": .5,
              "selection": "No new external labels or predictions used to choose any method or threshold",
              "prior_protocol_sha256": sha(ROOT / "reports/railway_budget_research/protocol_v1.json")}
    if policy.exists():
        assert json.loads(policy.read_text()) == frozen
    else:
        dump(policy, frozen)
    base = OUTPUT / "datasets/OSDaR23_new"
    records, archives, missing = [], [], []
    for spec in plan["resources"]:
        archive = args.downloads / (spec["name"] + ".zip")
        if not archive.exists():
            missing.append(spec)
            continue
        with zipfile.ZipFile(archive) as z:
            members = sorted(i.filename for i in z.infolist()
                             if i.filename.startswith("rgb_center/") and i.filename.endswith(".png"))
            assert members
            n = min(plan["frames_per_sequence"], len(members))
            selected = [members[i] for i in sorted(set(round(j * (len(members)-1) / max(1, n-1)) for j in range(n)))]
            labels = [i.filename for i in z.infolist() if i.filename.endswith("_labels.json")]
            assert len(labels) == 1
            selection_path = base / (spec["name"] + "_selection.json")
            selection = {"members": selected, "labels": labels[0], "available_RGB": len(members),
                         "metadata_only_selection": True, "plan_sha256": sha(plan_path)}
            if selection_path.exists():
                assert json.loads(selection_path.read_text()) == selection
            else:
                dump(selection_path, selection)
            checks = []
            for member in selected + labels:
                assert not Path(member).is_absolute() and ".." not in Path(member).parts
                info = z.getinfo(member)
                target = base / spec["name"] / member
                blob = z.read(member)  # ZIP validates its own CRC too.
                assert len(blob) == info.file_size and zlib.crc32(blob) == info.CRC
                if target.exists():
                    assert target.read_bytes() == blob
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(blob)
                checks.append({"member": member, "bytes": len(blob), "CRC": info.CRC, "sha256": sha(target)})
            label_file = base / spec["name"] / labels[0]
            label = json.loads(label_file.read_text())["openlabel"]
            uri_to_frame = {v["frame_properties"]["streams"]["rgb_center"]["uri"].lstrip("/"): k
                            for k, v in label["frames"].items()
                            if "rgb_center" in v.get("frame_properties", {}).get("streams", {})}
            missing_annotation_members = [m for m in selected if m not in uri_to_frame]
            for member in selected:
                if member not in uri_to_frame:
                    continue  # Missing annotation is unknown, never a negative example.
                path = base / spec["name"] / member
                image = cv2.imread(str(path))
                assert image is not None
                h, w = image.shape[:2]
                index = uri_to_frame[member]
                frame = label["frames"][index]
                streams = frame["frame_properties"]["streams"]
                assert streams["rgb_center"]["uri"].lstrip("/") == member
                objects, polygons = [], []
                for oid, obj in frame.get("objects", {}).items():
                    kind = label["objects"][oid]["type"]
                    data = obj.get("object_data", {})
                    if kind == "person":
                        for box in data.get("bbox", []):
                            if box.get("coordinate_system") != "rgb_center":
                                continue
                            x, y, bw, bh = map(float, box["val"])
                            assert bw > 0 and bh > 0
                            objects.append({"class_name_raw": "person", "bbox_xyxy_raw": [x-bw/2,y-bh/2,x+bw/2,y+bh/2],
                                            "object_id": oid, "attributes": box.get("attributes", {})})
                    if kind == "crowd":
                        polygons += [p["val"] for p in data.get("poly2d", []) if p.get("coordinate_system") == "rgb_center"]
                records.append({"image_id": spec["name"] + "/" + member, "image_sha256": sha(path),
                    "decoded_BGR_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "width": w, "height": h,
                    "sequence_id": spec["name"], "capture_group_id": "OSDaR_" + spec["name"].split("_")[0],
                    "frame_index": int(index), "timestamp_seconds": float(Path(member).stem.split("_", 1)[1]),
                    "objects": objects, "crowd_ignore_polygons": polygons,
                    "everythingAnnotated": "not_certified_complete", "annotation_sha256": sha(label_file)})
            archives.append({"name": spec["name"], "path": str(archive), "bytes": archive.stat().st_size,
                             "sha256": sha(archive), "verified_members": checks,
                             "excluded_missing_annotation": missing_annotation_members,
                             "exclusion_rule": "Metadata URI join only; no object/prediction filtering or replacement"})
    old = [r for split in ("development", "validation", "test") for r in rows(split)]
    for path in (ROOT / "reports/railway_budget_research/external_new.jsonl",
                 ROOT / "reports/railway_person_research/w05_w07/external.jsonl"):
        old += [json.loads(x) for x in path.read_text().splitlines()]
    old_hashes = {r["image_sha256"] for r in old}
    overlaps = [r["image_id"] for r in records if r["image_sha256"] in old_hashes]
    assert not overlaps and len({r["image_sha256"] for r in records}) == len(records)
    manifest = REPORT / "external_new_available.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf8")
    bins = collections.Counter()
    for row in records:
        for obj in row["objects"]:
            height = obj["bbox_xyxy_raw"][3] - obj["bbox_xyxy_raw"][1]
            bins["<16" if height < 16 else "16-32" if height < 32 else "32-64" if height < 64 else ">=64"] += 1
    audit = {"images": len(records), "groups": len({r["capture_group_id"] for r in records}),
             "boxes": sum(len(r["objects"]) for r in records), "height_counts": dict(bins),
             "overlap_with_previous_images": overlaps, "archives": archives,
             "missing_resources": missing, "manifest_sha256": sha(manifest),
             "annotation_scope": "supplied instances; no newly invented labels, no certified false-positive rate",
             "status": "ALL_LOCKED_GROUPS_IMPORTED" if not missing else "PARTIAL_LOCKED_GROUPS_IMPORTED"}
    dump(REPORT / "external_local_audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("status", "images", "groups", "boxes", "height_counts")}), flush=True)

if __name__ == "__main__":
    main()
