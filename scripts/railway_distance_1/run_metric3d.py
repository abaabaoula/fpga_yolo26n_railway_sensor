"""Frozen, untrained, offline full-frame Metric3Dv2-ViT-S validation on CPU."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import faulthandler
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/railway_distance_1'
REPORT = ROOT / 'reports/railway_distance_1'
sys.path.insert(0, str(OUT / 'runtime_deps'))
sys.path.insert(0, str(OUT / 'vendor_cpu'))
sys.path.insert(0, str(ROOT / 'scripts/railway_distance_1'))
from analyze_geometry import sha

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def main():
    temporary = OUT / 'tmp'
    temporary.mkdir(exist_ok=True)
    tempfile.tempdir = str(temporary)
    trace = (REPORT / 'metric3d_trace.log').open('w')
    faulthandler.enable(file=trace)
    faulthandler.dump_traceback_later(45, repeat=True, file=trace)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261104)
    cv2.setNumThreads(1)
    from mmengine import Config
    from mono.model.monodepth_model import get_configured_monodepth_model
    contract = dict(protocol_sha256=sha(REPORT / 'protocol.md'),
                    checkpoint_sha256=sha(OUT / 'metric3d_vit_small.pth'),
                    image_manifest_sha256=sha(REPORT / 'linked_images.json'),
                    cpu_patch_sha256=sha(REPORT / 'cpu_portability.patch'),
                    input_hw=[616, 1064], target_rule='central 40% width x 60% height, median',
                    focal_rule='canonical_depth * (fx * resize_scale) / 1000',
                    threads=8, training=False, scale_fitting=False,
                    script_sha256=sha(__file__))
    contract_digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    prediction_dir = OUT / 'metric3d_predictions'
    prediction_dir.mkdir(exist_ok=True)
    images = json.loads((REPORT / 'linked_images.json').read_text())
    import csv
    with (REPORT / 'reference_linked98.csv').open(newline='') as stream:
        reference = list(csv.DictReader(stream))
    environment = dict(python=sys.version, torch=torch.__version__, opencv=cv2.__version__,
                       numpy=np.__version__, device='CPU', platform=platform.platform(),
                       logical_cpus=os.cpu_count(),
                       packages={name: importlib.metadata.version(name) for name in
                                 ['mmengine', 'timm', 'huggingface_hub', 'safetensors']})
    (REPORT / 'metric3d_contract.json').write_text(json.dumps(dict(contract=contract, digest=contract_digest,
                                                                 environment=environment), indent=2) + '\n')
    start = time.perf_counter()
    print('Reading frozen configuration', flush=True)
    config = Config.fromfile(str(OUT / 'vendor_cpu/mono/configs/HourglassDecoder/vit.raft5.small.py'))
    print('Constructing frozen model', flush=True)
    model = get_configured_monodepth_model(config)
    checkpoint = torch.load(OUT / 'metric3d_vit_small.pth', map_location='cpu', weights_only=True)
    status = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    (REPORT / 'metric3d_weight_load.json').write_text(json.dumps(dict(missing_keys=status.missing_keys,
        unexpected_keys=status.unexpected_keys, parameters=sum(p.numel() for p in model.parameters()),
        initialization_seconds=time.perf_counter()-start), indent=2) + '\n')
    # The public checkpoint omits DINO's masked-training token. Full-image
    # inference passes masks=None, so this parameter is never read. Keep the
    # exact missing-key record, and reject every other missing/unexpected key.
    assert status.missing_keys == ['depth_model.encoder.mask_token'], status
    assert not status.unexpected_keys, status
    faulthandler.cancel_dump_traceback_later()
    model.eval()
    print(f'Loaded frozen Metric3Dv2 ViT-small on CPU in {time.perf_counter()-start:.2f}s', flush=True)
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    for index, record in enumerate(images):
        image_id = record['image_id']
        key = hashlib.sha256(image_id.encode()).hexdigest()[:24]
        result_file = prediction_dir / (key + '.json')
        if result_file.exists():
            previous = json.loads(result_file.read_text())
            assert previous['contract_digest'] == contract_digest
            assert sha(prediction_dir / (key + '.npz')) == previous['depth_map_sha256']
            print(f'{index+1}/{len(images)} reused {image_id}', flush=True)
            continue
        image_path = ROOT / 'output/railway_sensors_validation_1/datasets/OSDaR23_new' / image_id
        assert sha(image_path) == record['image_sha256']
        bgr = cv2.imread(str(image_path))
        assert bgr is not None and hashlib.sha256(bgr.tobytes()).hexdigest() == record['decoded_BGR_sha256']
        rgb = bgr[:, :, ::-1]
        height, width = rgb.shape[:2]
        assert (height, width) == (record['height'], record['width'])
        scale = min(616/height, 1064/width)
        resized = cv2.resize(rgb, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_LINEAR)
        h, w = resized.shape[:2]
        pad_h, pad_w = 616-h, 1064-w
        pad = [pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2]
        padded = cv2.copyMakeBorder(resized, *pad, cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
        tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float()
        tensor = ((tensor-mean)/std).unsqueeze(0)
        begin = time.perf_counter()
        with torch.inference_mode():
            depth, confidence, extra = model.inference({'input': tensor})
            seconds = time.perf_counter()-begin
            depth = depth.squeeze()
            depth = depth[pad[0]:depth.shape[0]-pad[1], pad[2]:depth.shape[1]-pad[3]]
            depth = F.interpolate(depth[None, None], (height, width), mode='bilinear').squeeze()
            metric = (depth * (record['K'][0][0]*scale) / 1000).cpu().numpy().astype(np.float32)
        objects = []
        for row in reference:
            if row['image_id'] != image_id:
                continue
            x0, y0, x1, y1 = [float(row[k]) for k in ['x0', 'y0', 'x1', 'y1']]
            cx, cy = (x0+x1)/2, (y0+y1)/2
            bw, bh = x1-x0, y1-y0
            xx0, xx1 = max(0, math.floor(cx-.2*bw)), min(width, math.ceil(cx+.2*bw))
            yy0, yy1 = max(0, math.floor(cy-.3*bh)), min(height, math.ceil(cy+.3*bh))
            roi = metric[yy0:yy1, xx0:xx1]
            valid = np.isfinite(roi) & (roi > 0)
            estimate = float(np.median(roi)) if valid.all() and roi.size else None
            objects.append(dict(object_id=row['object_id'], track_id=row['track_id'],
                                optical_z_reference_m=float(row['optical_z_m']),
                                range_reference_m=float(row['range_m']),
                                roi_xyxy=[xx0, yy0, xx1, yy1], roi_pixels=int(roi.size),
                                invalid_pixels=int((~valid).sum()), metric_z_m=estimate,
                                official_clamped_z_m=None if estimate is None else min(300., max(0., estimate)),
                                p10_m=float(np.percentile(roi[valid], 10)) if valid.any() else None,
                                p90_m=float(np.percentile(roi[valid], 90)) if valid.any() else None,
                                pixels_over_300m=int((roi > 300).sum())))
        np.savez_compressed(prediction_dir / (key + '.npz'), optical_depth_m=metric)
        result = dict(image_id=image_id, contract_digest=contract_digest,
                      source_sha256=record['image_sha256'], focal_px=record['K'][0][0], scale=scale,
                      pad_tblr=pad, resized_hw=[h, w], inference_seconds=seconds,
                      map_nonfinite_pixels=int((~np.isfinite(metric)).sum()),
                      map_nonpositive_pixels=int((metric <= 0).sum()),
                      depth_map_sha256=sha(prediction_dir / (key + '.npz')), objects=objects)
        result_file.write_text(json.dumps(result, indent=2) + '\n')
        print(f'{index+1}/{len(images)} inference {seconds:.2f}s, {len(objects)} target regions: {image_id}', flush=True)
        del depth, confidence, extra, metric
    print('All 20 full-frame inferences and 98 frozen target-region estimates completed.', flush=True)


if __name__ == '__main__':
    class Tee:
        def __init__(self, stream, log):
            self.stream, self.log = stream, log
        def write(self, text):
            self.stream.write(text)
            self.log.write(text)
            self.log.flush()
        def flush(self):
            self.stream.flush()
            self.log.flush()
    log = (REPORT / 'metric3d_completed_run.log').open('a', encoding='utf8')
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    main()
