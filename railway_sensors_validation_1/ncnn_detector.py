"""Independent mature FP32 backend, original one-to-one head and external TopK."""
import ctypes
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from railway_person_optimization.geometry import preprocess, decode

def official_topk(raw):
    values = raw[0].T
    scores = values[:, 4:]
    k = min(300, len(values))
    anchors = np.argsort(-scores.max(axis=1), kind="stable")[:k]
    selected = scores[anchors].reshape(-1)
    winners = np.argsort(-selected, kind="stable")[:k]
    return np.concatenate((values[anchors[winners // 80], :4], selected[winners, None],
                           (winners % 80).astype(np.float32)[:, None]), axis=1)[None]

class NCNNDetector:
    def __init__(self, model, threads=2, bridge=None, pc_dependencies=None):
        self.model = Path(model)
        metadata = json.loads((self.model / "verified.json").read_text())
        for filename, digest in metadata["files"].items():
            assert hashlib.sha256((self.model / filename).read_bytes()).hexdigest() == digest, filename
        self.hw = tuple(metadata["shape_hw"])
        self.last_raw = None
        self.bridge = None
        if bridge:
            self.bridge = ctypes.CDLL(str(bridge))
            self.bridge.sv_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            self.bridge.sv_create.restype = ctypes.c_void_p
            self.bridge.sv_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            self.bridge.sv_destroy.argtypes = [ctypes.c_void_p]
            self.handle = self.bridge.sv_create(str(self.model / "model.ncnn.param").encode(),
                           str(self.model / "model.ncnn.bin").encode(), threads)
            assert self.handle
            self.buffer = np.empty((1, 84, sum((self.hw[0]//s)*(self.hw[1]//s) for s in (8,16,32))), np.float32)
        else:
            if pc_dependencies:
                sys.path.insert(0, str(pc_dependencies))
            import ncnn
            self.ncnn = ncnn
            self.net = ncnn.Net()
            self.net.opt.num_threads = threads
            self.net.opt.use_vulkan_compute = False
            self.net.opt.use_fp16_packed = False
            self.net.opt.use_fp16_storage = False
            self.net.opt.use_fp16_arithmetic = False
            assert self.net.load_param(str(self.model / "model.ncnn.param")) == 0
            assert self.net.load_model(str(self.model / "model.ncnn.bin")) == 0

    def forward(self, tensor):
        if self.bridge:
            shape = np.zeros(4, np.int32)
            count = self.bridge.sv_run(self.handle, tensor.ctypes.data, self.hw[1], self.hw[0],
                                       self.buffer.ctypes.data, self.buffer.size, shape.ctypes.data)
            assert count == self.buffer.size, (count, self.buffer.size, shape.tolist())
            assert list(shape[:3]) == [2, self.buffer.shape[2], 84], shape.tolist()
            return self.buffer.copy()
        with self.net.create_extractor() as extractor:
            assert extractor.input("in0", self.ncnn.Mat(tensor[0]).clone()) == 0
            code, output = extractor.extract("out0")
            assert code == 0
            return np.array(output)[None]

    def infer(self, image, score=.001, return_raw=False):
        start = time.perf_counter()
        tensor = preprocess(image, self.hw)[0]
        raw = official_topk(self.forward(tensor))
        self.last_raw = raw
        boxes = decode(raw, image.shape, self.hw, score)
        boxes = boxes[boxes[:, 5] == 0].copy()
        result = {"predictions": boxes, "analysis_seconds": time.perf_counter()-start}
        if return_raw:
            result["raw"] = raw.copy()
        return result

    def close(self):
        if self.bridge and self.handle:
            self.bridge.sv_destroy(self.handle)
            self.handle = None
        elif getattr(self, "net", None) is not None:
            self.net.clear()
            self.net = None
