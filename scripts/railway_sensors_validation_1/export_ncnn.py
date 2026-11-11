"""Keep official one-to-one head; move only unsupported TopK outside NCNN."""
import argparse
import os
import types
from common import *
os.environ.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", YOLO_CONFIG_DIR=str(OUTPUT / "ultralytics_config"))
sys.path.insert(0, str(OUTPUT / "deps_pc"))
import numpy as np
import torch
import pnnx
import ncnn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from railway_person_optimization.geometry import preprocess

def raw_one2one(self, x):
    return self._inference(self.forward_head(x, **self.one2one))

def topk(raw):
    """Official two-stage class-aware selection; stable ties explicitly defined."""
    a = raw[0].T
    scores = a[:, 4:]
    k = min(300, len(a))
    anchors = np.argsort(-scores.max(axis=1), kind="stable")[:k]
    selected = scores[anchors].reshape(-1)
    winners = np.argsort(-selected, kind="stable")[:k]
    return np.concatenate((a[anchors[winners // 80], :4], selected[winners, None],
                           (winners % 80).astype(np.float32)[:, None]), axis=1)[None]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="384x640,256x1280,736x1280")
    args = parser.parse_args()
    torch.set_num_threads(2)
    weights = ROOT / "yolo26_optimization_8/reproduction_inputs/yolo26n.pt"
    for text in args.shapes.split(","):
        hw = tuple(map(int, text.split("x")))
        out = OUTPUT / "ncnn_models" / text
        if (out / "verified.json").exists():
            print("REUSE_VERIFIED", text, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(weights), task="detect").model.float().eval()
        # Old checkpoints predate the current property-backed _end2end flag.
        # Explicitly select the frozen one-to-one branch before fusion.
        model.end2end = True
        model.fuse(verbose=False)
        head = model.model[-1]
        assert head.end2end and head.nc == 80
        image_row = rows("development")[0]
        import cv2
        image = cv2.imread(str(DATA / image_row["image_id"]))
        tensor = torch.from_numpy(preprocess(image, hw)[0])
        with torch.no_grad():
            official = model(tensor)[0].numpy()
        for layer in model.modules():
            if isinstance(layer, Detect):
                layer.export = True
                layer.format = "ncnn"
                layer.dynamic = False
        head.forward = types.MethodType(raw_one2one, head)
        with torch.no_grad():
            expected = model(tensor).numpy()
        selected = topk(expected)
        assert expected.shape[1] == 84 and official.shape == selected.shape == (1, 300, 6)
        # Check non-tied, above-floor detections against unmodified official inference.
        mask = official[0, :, 4] >= .001
        assert np.allclose(official[0, mask], selected[0, mask], atol=1e-4, rtol=1e-5)
        print("EXPORT", text, expected.shape, flush=True)
        previous = Path.cwd()
        os.chdir(out)
        try:
            pnnx.export(model, inputs=tensor, ptpath="model.pt", pnnxparam="model.pnnx.param",
                        pnnxbin="model.pnnx.bin", pnnxpy="model_pnnx.py", pnnxonnx="model.pnnx.onnx",
                        ncnnparam="model.ncnn.param", ncnnbin="model.ncnn.bin", ncnnpy="model_ncnn.py",
                        fp16=False, device="cpu")
        finally:
            os.chdir(previous)
        net = ncnn.Net()
        net.opt.num_threads = 2
        net.opt.use_vulkan_compute = False
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        assert net.load_param(str(out / "model.ncnn.param")) == 0
        assert net.load_model(str(out / "model.ncnn.bin")) == 0
        with net.create_extractor() as extractor:
            assert extractor.input("in0", ncnn.Mat(tensor.numpy()[0]).clone()) == 0
            code, output = extractor.extract("out0")
            assert code == 0
            actual = np.array(output)[None]
        assert actual.shape == expected.shape
        np.savez(out / "reference.npz", input=tensor.numpy(), torch_raw=expected, ncnn_raw=actual,
                 official_topk=official)
        error = np.abs(actual - expected)
        assert np.allclose(actual, expected, atol=.08, rtol=.002), float(error.max())
        dump(out / "verified.json", {"shape_hw": hw, "weights_sha256": sha(weights),
             "head": "original one-to-one weights; same 80 classes; decoded XYXY+scores before TopK",
             "topk": "CPU official two-stage selection, deterministic stable tie policy",
             "FP32": True, "ncnn": ncnn.__version__, "pnnx": pnnx.__version__,
             "raw_shape": actual.shape, "max_absolute_error": float(error.max()),
             "mean_absolute_error": float(error.mean()), "source_image": image_row["image_id"],
             "files": {p.name: sha(p) for p in out.glob("model.ncnn.*")}})
        print("VERIFIED", text, float(error.max()), flush=True)

if __name__ == "__main__":
    main()
