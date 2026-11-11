"""Native integer ARM-only execution. No PYNQ import, BO or device access."""
import ctypes
import hashlib
import json
import struct
import time
from pathlib import Path
import cv2
import numpy as np
from railway_person_optimization.geometry import letterbox, decode

class CPUDetector:
    def __init__(self, candidate, library, options=132):
        candidate = Path(candidate)
        prep = json.loads((candidate / "preparation.json").read_text())
        self.hw = tuple(prep["shape_hw"])
        prog = candidate / "compiled" / ("program_r" + str(self.hw[1]))
        assert hashlib.sha256((prog / "graph.bin").read_bytes()).hexdigest() == prep["program_sha256"]
        self.meta = json.loads((prog / "program.json").read_text())
        # Prove the INT32 accumulation bound for every actual dense layer.
        # The accepted board payload intentionally omits PC-only constants.npz.
        # Read the very same constants directly from the SHA-checked graph.
        blob = (prog / "graph.bin").read_bytes()
        magic, nt, nn, input_id, output_id = struct.unpack_from("<5i", blob, 0)
        assert magic == 0x3747504f and nt == len(self.meta["names"])
        wanted = {name for node in self.meta["nodes"] if node["op"] == "QuantConv"
                  for name in node["inputs"][1:3]}
        constants = {}
        offset = 20
        for index in range(nt):
            kind, rank, *rest = struct.unpack_from("<9i", blob, offset)
            dims, count = rest[:6], rest[6]
            offset += 36
            name = self.meta["names"][index]
            if name in wanted:
                assert count > 0 and kind in [3, 6]
                dtype = np.dtype("int8" if kind == 3 else "<i4")
                shape = tuple(dims[:rank])
                assert count == int(np.prod(shape))*dtype.itemsize
                constants[name] = np.frombuffer(blob, dtype=dtype,
                                                count=int(np.prod(shape)), offset=offset).reshape(shape)
            offset += count
        bounds = []
        for node in self.meta["nodes"]:
            if node["op"] != "QuantConv":
                continue
            attrs = node["attrs"]
            weight = constants[node["inputs"][1]].astype(np.int64)
            bias = constants[node["inputs"][2]].astype(np.int64)
            maximum = int((max(abs(-128-attrs[6]), abs(127-attrs[6]))
                           * np.abs(weight-attrs[7]).reshape(weight.shape[0], -1).sum(1)
                           + np.abs(bias)).max())
            assert maximum < 2**31, (node["name"], maximum)
            bounds.append(maximum)
        del constants, blob
        self.max_accumulator_bound = max(bounds)
        self.lib = ctypes.CDLL(str(library))
        ptr = ctypes.c_void_p
        self.lib.o8_create.argtypes = [ctypes.c_char_p, ptr, ptr, ptr, ctypes.c_char_p, ctypes.c_int]
        self.lib.o8_create.restype = ptr
        self.lib.o7_error.restype = ctypes.c_char_p
        self.lib.o8_set_options.argtypes = [ptr, ctypes.c_int]
        self.lib.o7_run_bgr.argtypes = [ptr, ptr, ptr]
        self.lib.o7_destroy.argtypes = [ptr]
        self.lib.o7_timers.argtypes = [ptr, ptr]
        self.lib.o7_memory.argtypes = [ptr, ptr]
        self.handle = self.lib.o8_create(str(prog / "graph.bin").encode(), None, None, None,
                                        str(candidate / "compiled/resident_plan/resident_plan.bin").encode(), 0)
        if not self.handle:
            raise RuntimeError(self.lib.o7_error().decode())
        assert self.lib.o8_set_options(self.handle, options) == 0
        self.raw = np.empty((1, 300, 6), np.float32)
        self.timers = np.empty(16, np.float64)
        cv2.setNumThreads(2)

    def infer(self, image, score=.001, return_raw=False):
        start = time.perf_counter()
        padded, _, _ = letterbox(image, self.hw)
        rc = self.lib.o7_run_bgr(self.handle, padded.ctypes.data, self.raw.ctypes.data)
        if rc:
            raise RuntimeError(self.lib.o7_error().decode())
        self.lib.o7_timers(self.handle, self.timers.ctypes.data)
        raw = self.raw.copy()
        boxes = decode(raw, image.shape, self.hw, score)
        boxes = boxes[boxes[:, 5] == 0].copy()
        assert np.isfinite(raw).all()
        r = {"predictions": boxes, "analysis_seconds": time.perf_counter()-start}
        if return_raw:
            r["raw"] = raw
        return r

    def close(self):
        if self.handle:
            self.lib.o7_destroy(self.handle)
            self.handle = None
