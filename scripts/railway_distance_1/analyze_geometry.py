"""Reproduce OSDaR23 person/cuboid associations and frozen detection by range.

No annotation, threshold, detector, or accepted report is modified.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from railway_person_research.evaluation import match_predictions, iou_matrix

REPORT = ROOT / 'reports/railway_distance_1'
DATA = ROOT / 'output/railway_sensors_validation_1/datasets/OSDaR23_new'
MANIFEST = ROOT / 'reports/railway_sensors_validation_1/external_new_available.jsonl'
CACHE = ROOT / 'output/railway_sensors_validation_1/accuracy_external_v1'
AXES = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
BINS = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 200), (200, float('inf'))]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rotation(quaternion):
    """Active column-vector rotation, quaternion ordered x,y,z,w."""
    x, y, z, w = np.asarray(quaternion, dtype=float)
    s = 2 / (x*x + y*y + z*z + w*w)
    return np.array([[1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
                     [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
                     [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]])


def optical(points, quaternion, translation):
    # Coordinate transformation into the camera frame (passive rotation).
    # Dataset's second quaternion (.5,-.5,.5,-.5) is likewise passive,
    # giving [-y,-z,x]. This is NOT R*p+t for an active point transform.
    return (AXES @ rotation(quaternion).T @
            (np.asarray(points).reshape(-1, 3) - translation).T).T


def project(points, K, distortion):
    assert np.all(np.asarray(points)[:, 2] > 0)
    return cv2.projectPoints(np.asarray(points), np.zeros(3), np.zeros(3),
                             K, distortion)[0].reshape(-1, 2)


def write_csv(name, rows):
    assert rows
    with (REPORT / name).open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    # Verify the passive optical-axis convention independently of actual labels.
    q_axes = rotation([.5, -.5, .5, -.5])
    np.testing.assert_allclose(q_axes.T, AXES, atol=1e-14)
    np.testing.assert_allclose(optical([[10, 2, 3]], [0, 0, 0, 1], [0, 0, 0]),
                               [[-2, -3, 10]], atol=1e-14)
    manifest = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    labels, label_hashes = {}, {}
    associations, linked, diagnostics, detections, image_records = [], [], [], [], []
    methods = None
    independent = {}
    with (ROOT / 'reports/scientific_reports_validation_1/tables/external_failures_second_evaluator.csv').open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            independent[row['method'], row['image_id'], row['object_id']] = row['matched'] == 'True'
    for record in manifest:
        seq, image_id = record['sequence_id'], record['image_id']
        if seq not in labels:
            path = DATA / seq / f'{seq}_labels.json'
            labels[seq] = json.loads(path.read_text(encoding='utf8'))['openlabel']
            label_hashes[seq] = sha(path)
        assert label_hashes[seq] == record['annotation_sha256']
        label = labels[seq]
        frame = label['frames'][str(record['frame_index'])]
        assert frame['frame_properties']['streams']['rgb_center']['uri'].lstrip('/') == image_id.split('/', 1)[1]
        pose = label['coordinate_systems']['rgb_center']['pose_wrt_parent']
        assert label['coordinate_systems']['rgb_center']['parent'] == 'base'
        lidar = label['coordinate_systems']['lidar']
        assert lidar['parent'] == 'base'
        assert lidar['pose_wrt_parent']['translation'] == [0, 0, 0]
        np.testing.assert_allclose(rotation(lidar['pose_wrt_parent']['quaternion']), np.eye(3), atol=1e-12)
        intr = label['streams']['rgb_center']['stream_properties']['intrinsics_pinhole']
        K = np.asarray(intr['camera_matrix']).reshape(3, 4)[:, :3]
        distortion = np.asarray(intr['distortion_coeffs'])
        quaternion, translation = pose['quaternion'], np.asarray(pose['translation'])
        gt = np.asarray([o['bbox_xyxy_raw'] for o in record['objects']], dtype=float).reshape(-1, 4)
        cache_path = CACHE / (hashlib.sha256(image_id.encode()).hexdigest()[:24] + '.json')
        cache = json.loads(cache_path.read_text())
        assert cache['image_id'] == image_id
        assert cache['source_sha256'] == record['image_sha256']
        if methods is None:
            methods = list(cache['variants'])
        assert list(cache['variants']) == methods
        matched = {}
        for method in methods:
            predictions = cache['variants'][method]['predictions']
            result = match_predictions(predictions, gt, score=.25, iou=.5)
            matched[method] = {g: p for p, g in result['assignments']}
            for index, obj in enumerate(record['objects']):
                assert (index in matched[method]) == independent[method, image_id, obj['object_id']]
        num_linked = 0
        for index, obj in enumerate(record['objects']):
            oid = obj['object_id']
            assert label['objects'][oid]['type'] == 'person'
            raw_object = frame['objects'][oid]['object_data']
            boxes = [b for b in raw_object.get('bbox', []) if b['coordinate_system'] == 'rgb_center']
            assert len(boxes) == 1
            cx, cy, width, height = boxes[0]['val']
            expected = [cx-width/2, cy-height/2, cx+width/2, cy+height/2]
            np.testing.assert_allclose(expected, obj['bbox_xyxy_raw'], atol=1e-8, rtol=0)
            cuboids = [c for c in raw_object.get('cuboid', []) if c['coordinate_system'] == 'lidar']
            assert len(cuboids) <= 1
            attributes = {a['name']: a['val'] for a in boxes[0].get('attributes', {}).get('text', [])}
            base = dict(image_id=image_id, sequence_id=seq, capture_group_id=record['capture_group_id'],
                        frame_index=record['frame_index'], object_id=oid, track_id=f'{seq}/{oid}',
                        x0=expected[0], y0=expected[1], x1=expected[2], y1=expected[3],
                        width_px=width, height_px=height, pose=attributes.get('pose', 'unknown'),
                        occlusion=attributes.get('occlusion', 'unknown'),
                        age=attributes.get('age', 'unknown'), has_cuboid=bool(cuboids))
            associations.append(base.copy())
            geo = None
            if cuboids:
                num_linked += 1
                val = np.asarray(cuboids[0]['val'], dtype=float)
                assert len(val) == 10 and np.all(val[7:] > 0)
                center = val[:3]
                camera_center = optical([center], quaternion, translation)[0]
                distance = float(np.linalg.norm(camera_center))
                np.testing.assert_allclose(distance, np.linalg.norm(center-translation), atol=1e-10)
                # Nonzero camera pose and rotation are numerically invertible.
                recovered = rotation(quaternion) @ AXES.T @ camera_center + translation
                np.testing.assert_allclose(recovered, center, atol=1e-10)
                uv = project(camera_center[None], K, distortion)[0]
                uv_no_dist = project(camera_center[None], K, np.zeros_like(distortion))[0]
                offsets = np.array(list(itertools.product([-.5, .5], repeat=3))) * val[7:]
                corners = (rotation(val[3:7]) @ offsets.T).T + center
                corners_camera = optical(corners, quaternion, translation)
                corner_uv = project(corners_camera, K, distortion)
                projected_box = [*corner_uv.min(axis=0), *corner_uv.max(axis=0)]
                inside = expected[0] <= uv[0] <= expected[2] and expected[1] <= uv[1] <= expected[3]
                geo = dict(**base, lidar_x_m=center[0], lidar_y_m=center[1], lidar_z_m=center[2],
                           camera_x_m=camera_center[0], camera_y_m=camera_center[1],
                           optical_z_m=camera_center[2], range_m=distance,
                           cuboid_dx_m=val[7], cuboid_dy_m=val[8], cuboid_dz_m=val[9],
                           projected_u=uv[0], projected_v=uv[1],
                           center_in_2d_box=bool(inside),
                           normalized_center_error=float(np.linalg.norm((uv-[cx, cy])/[width, height])),
                           projected_box_iou=float(iou_matrix([projected_box], [expected])[0, 0]),
                           distortion_displacement_px=float(np.linalg.norm(uv-uv_no_dist)))
                linked.append(geo)
                R = rotation(quaternion)
                hypotheses = {'documented_passive_pose': camera_center,
                              'active_direct': AXES @ (R @ center + translation),
                              'subtract_then_active': AXES @ (R @ (center-translation)),
                              'passive_plus_translation': AXES @ (R.T @ center + translation)}
                for name, point in hypotheses.items():
                    pix = project(point[None], K, distortion)[0]
                    diagnostics.append(dict(image_id=image_id, object_id=oid, convention=name,
                                            projected_u=pix[0], projected_v=pix[1],
                                            center_in_2d_box=bool(expected[0] <= pix[0] <= expected[2] and expected[1] <= pix[1] <= expected[3]),
                                            normalized_center_error=float(np.linalg.norm((pix-[cx, cy])/[width, height]))))
            for method in methods:
                prediction_index = matched[method].get(index)
                score = '' if prediction_index is None else cache['variants'][method]['predictions'][prediction_index][4]
                detections.append(dict(**base, method=method, matched=prediction_index is not None,
                                       matched_confidence=score, range_m='' if geo is None else geo['range_m'],
                                       optical_z_m='' if geo is None else geo['optical_z_m']))
        if num_linked:
            image_records.append(dict(image_id=image_id, image_sha256=record['image_sha256'],
                                      decoded_BGR_sha256=record['decoded_BGR_sha256'],
                                      width=record['width'], height=record['height'],
                                      K=K.tolist(), distortion=distortion.tolist(), pose=pose,
                                      linked_instances=num_linked))
    assert len(manifest) == 86 and len(associations) == 271
    assert len(linked) == 98 and len(image_records) == 20
    assert len({r['track_id'] for r in linked}) == 11
    strata = []
    for lo, hi in BINS:
        cohort = [r for r in linked if lo <= r['range_m'] < hi]
        for method in methods:
            selected = [r for r in detections if r['method'] == method and r['has_cuboid'] and lo <= r['range_m'] < hi]
            count = sum(r['matched'] for r in selected)
            strata.append(dict(method=method, distance_bin=f'[{lo},{hi})', instances=len(cohort),
                               images=len({r['image_id'] for r in cohort}),
                               tracks=len({r['track_id'] for r in cohort}), matched=count,
                               recall='' if not cohort else count/len(cohort)))
    overview = {}
    for method in methods:
        all_method = [r for r in detections if r['method'] == method]
        subset = [r for r in all_method if r['has_cuboid']]
        hits = [r for r in subset if r['matched']]
        overview[method] = dict(all_instances=271, all_matches=sum(r['matched'] for r in all_method),
                                linked_instances=98, linked_matches=len(hits),
                                farthest_matched_range_m=max((r['range_m'] for r in hits), default=None),
                                farthest_matched_instances=[r for r in hits if r['range_m'] == max(s['range_m'] for s in hits)] if hits else [])
    conventions = {}
    for name in sorted({r['convention'] for r in diagnostics}):
        rows = [r for r in diagnostics if r['convention'] == name]
        conventions[name] = dict(center_in_box=sum(r['center_in_2d_box'] for r in rows),
                                 median_normalized_center_error=float(np.median([r['normalized_center_error'] for r in rows])))
    summary = dict(images=86, person_instances=271, linked_instances=98, unavailable_instances=173,
                   linked_images=20, linked_tracks=11,
                   reference='camera-to-3D-cuboid-centre Euclidean range; optical Z stored separately',
                   transform='p_optical = A R(q)^T (p_lidar - t); normalized xyzw quaternion',
                   reference_range_min_m=min(r['range_m'] for r in linked),
                   reference_range_max_m=max(r['range_m'] for r in linked),
                   optical_z_min_m=min(r['optical_z_m'] for r in linked),
                   optical_z_max_m=max(r['optical_z_m'] for r in linked),
                   reprojection=conventions, methods=overview,
                   manifest_sha256=sha(MANIFEST), annotation_sha256=label_hashes,
                   independent_second_evaluator='all 2168 object/method outcomes agree',
                   warning='Previously viewed exploratory cohort; repeated tracks and nonrandom 3D availability. No operational maximum range.')
    for name, rows in [('associations_all271.csv', associations), ('reference_linked98.csv', linked),
                       ('detection_instances.csv', detections), ('detection_by_range.csv', strata),
                       ('reprojection_convention_diagnostics.csv', diagnostics)]:
        write_csv(name, rows)
    (REPORT / 'linked_images.json').write_text(json.dumps(image_records, indent=2) + '\n')
    (REPORT / 'geometry_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('methods', 'annotation_sha256')}, indent=2))
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'farthest_matched_instances'} for k, v in overview.items()}, indent=2))


if __name__ == '__main__':
    main()
