"""Summarize actual distance references and predeclared monocular controls."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/railway_distance_1'
OUT = ROOT / 'output/railway_distance_1'
sys.path.insert(0, str(Path(__file__).parent))
from analyze_geometry import BINS, sha, write_csv


def load_csv(name):
    with (REPORT / name).open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def metrics(rows):
    if not rows:
        return dict(instances=0, images=0, tracks=0)
    ref = np.array([r['reference_z_m'] for r in rows])
    pred = np.array([r['estimated_z_m'] for r in rows])
    ae = np.abs(pred-ref)
    return dict(instances=len(rows), images=len({r['image_id'] for r in rows}),
                tracks=len({r['track_id'] for r in rows}),
                mean_absrel=float(np.mean(ae/ref)), median_absrel=float(np.median(ae/ref)),
                median_absolute_error_m=float(np.median(ae)), mean_absolute_error_m=float(np.mean(ae)),
                rmse_m=float(np.sqrt(np.mean((pred-ref)**2))),
                mean_bias_m=float(np.mean(pred-ref)), within20_percent=float(np.mean(ae/ref <= .2)),
                estimate_min_m=float(pred.min()), estimate_max_m=float(pred.max()))


def main():
    contract = json.loads((REPORT / 'metric3d_contract.json').read_text())
    refs = load_csv('reference_linked98.csv')
    references = {(r['image_id'], r['object_id']): r for r in refs}
    images = json.loads((REPORT / 'linked_images.json').read_text())
    calibration = {r['image_id']: r for r in images}
    predictions = []
    for path in sorted((OUT / 'metric3d_predictions').glob('*.json')):
        record = json.loads(path.read_text())
        assert record['contract_digest'] == contract['digest']
        assert record['map_nonfinite_pixels'] == 0 and record['map_nonpositive_pixels'] == 0
        assert sha(path.with_suffix('.npz')) == record['depth_map_sha256']
        predictions.append(record)
    assert len(predictions) == 20 and sum(len(r['objects']) for r in predictions) == 98
    measurements, clamp = [], []
    for prediction in predictions:
        for obj in prediction['objects']:
            row = references[prediction['image_id'], obj['object_id']]
            assert obj['metric_z_m'] is not None and obj['invalid_pixels'] == 0
            assert abs(float(row['optical_z_m'])-obj['optical_z_reference_m']) < 1e-10
            base = dict(image_id=prediction['image_id'], object_id=obj['object_id'], track_id=row['track_id'],
                        sequence_id=row['sequence_id'], reference_range_m=float(row['range_m']),
                        reference_z_m=float(row['optical_z_m']), height_px=float(row['height_px']),
                        center_in_2d_box=row['center_in_2d_box'] == 'True',
                        pose=row['pose'], occlusion=row['occlusion'], age=row['age'])
            for method, estimate in [('Metric3Dv2_ViT_small', obj['metric_z_m']),
                                     ('Metric3Dv2_clamp300', obj['official_clamped_z_m'])]:
                r = dict(**base, method=method, estimated_z_m=estimate,
                         absolute_error_m=abs(estimate-base['reference_z_m']),
                         relative_error=abs(estimate-base['reference_z_m'])/base['reference_z_m'])
                (measurements if method == 'Metric3Dv2_ViT_small' else clamp).append(r)
            if row['pose'] == 'upright' and row['occlusion'] == '0-25 %' and row['age'] == 'adult':
                for height in [1.5, 1.7, 1.9]:
                    estimate = calibration[prediction['image_id']]['K'][1][1] * height / float(row['height_px'])
                    measurements.append(dict(**base, method=f'height_prior_{height:.1f}m', estimated_z_m=estimate,
                                             absolute_error_m=abs(estimate-base['reference_z_m']),
                                             relative_error=abs(estimate-base['reference_z_m'])/base['reference_z_m']))
    by_method = {}
    strata, tracks = [], []
    for method in sorted({r['method'] for r in measurements}):
        selected = [r for r in measurements if r['method'] == method]
        by_method[method] = metrics(selected)
        for lo, hi in BINS:
            subset = [r for r in selected if lo <= r['reference_range_m'] < hi]
            m = metrics(subset)
            strata.append(dict(method=method, distance_bin=f'[{lo},{hi})',
                               **{key: m.get(key, '') for key in ['instances', 'images', 'tracks', 'mean_absrel', 'median_absolute_error_m', 'mean_bias_m']}))
        for track in sorted({r['track_id'] for r in selected}):
            subset = [r for r in selected if r['track_id'] == track]
            tracks.append(dict(method=method, track_id=track, **metrics(subset)))
    depth = [r for r in measurements if r['method'] == 'Metric3Dv2_ViT_small']
    upright_ids = {(r['image_id'], r['object_id']) for r in measurements if r['method'] == 'height_prior_1.7m'}
    sensitivity = dict(
        projected_center_inside=metrics([r for r in depth if r['center_in_2d_box']]),
        projected_center_outside=metrics([r for r in depth if not r['center_in_2d_box']]),
        Metric3D_same59_as_height_control=metrics([r for r in depth if (r['image_id'], r['object_id']) in upright_ids]),
        official_clamp300=metrics(clamp),
        clamped_target_count=sum(a['estimated_z_m'] != b['estimated_z_m'] for a, b in zip(depth, clamp)))
    track_errors = [r['mean_absrel'] for r in tracks if r['method'] == 'Metric3Dv2_ViT_small']
    # Independent re-read: compute range directly in the original LiDAR axes,
    # without using optical conversion or the geometry script's quaternion code.
    raw_labels, maximum_difference = {}, 0.
    for row in refs:
        seq = row['sequence_id']
        if seq not in raw_labels:
            path = ROOT / 'output/railway_sensors_validation_1/datasets/OSDaR23_new' / seq / (seq + '_labels.json')
            raw_labels[seq] = json.loads(path.read_text())['openlabel']
        label = raw_labels[seq]
        obj = label['frames'][row['frame_index']]['objects'][row['object_id']]['object_data']
        cub = next(v for v in obj['cuboid'] if v['coordinate_system'] == 'lidar')
        t = label['coordinate_systems']['rgb_center']['pose_wrt_parent']['translation']
        direct = sum((float(a)-float(b))**2 for a, b in zip(cub['val'][:3], t))**.5
        maximum_difference = max(maximum_difference, abs(direct-float(row['range_m'])))
    assert maximum_difference < 1e-10
    summary = dict(methods=by_method, sensitivity=sensitivity,
                   metric3d_track_macro_mean_absrel=float(np.mean(track_errors)),
                   all20_inference_seconds=[p['inference_seconds'] for p in predictions],
                   inference_mean_seconds=float(np.mean([p['inference_seconds'] for p in predictions])),
                   independent_direct_range_max_difference_m=maximum_difference,
                   source_contract_digest=contract['digest'], nonfinite_target_estimates=0,
                   decision='Do not propagate monocular metre estimates to the main RailGoerl24 cohort. '
                            'This distant, repeated-track cohort does not validate transferable metric range or a reliable detection horizon.',
                   timing_scope='Offline CPU depth-model forward only; not FPGA, live application or sensor-to-alarm timing.')
    (REPORT / 'depth_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_csv('depth_instances.csv', measurements)
    write_csv('depth_by_range.csv', strata)
    write_csv('depth_by_track.csv', tracks)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
