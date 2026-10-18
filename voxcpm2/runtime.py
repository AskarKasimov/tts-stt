"""Small, model-independent runtime helpers. Call feature operations under the model lock."""

import copy
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np


def reference_cache_size():
    size = int(os.getenv("VOXCPM_REFERENCE_CACHE_SIZE", "4"))
    if not 0 <= size <= 16:
        raise ValueError("VOXCPM_REFERENCE_CACHE_SIZE must be 0–16")
    return size


def isolated(value):
    if hasattr(value, "detach"):
        value = value.detach()
        if not value.isfinite().all().item():
            raise RuntimeError("Reference features contain nonfinite values")
        return value.clone()
    if isinstance(value, np.ndarray) and not np.isfinite(value).all():
        raise RuntimeError("Reference features contain nonfinite values")
    if isinstance(value, dict):
        return {key: isolated(item) for key, item in value.items()}
    if isinstance(value, list):
        return [isolated(item) for item in value]
    if isinstance(value, tuple):
        return tuple(isolated(item) for item in value)
    return copy.deepcopy(value)


class ReferenceFeatures:
    def __init__(self, size=4):
        self.size = size
        self.entries = OrderedDict()
        self.model = None

    def __len__(self):
        return len(self.entries)

    def get(self, model, path, revision):
        if self.model is not model:
            self.entries.clear()
            self.model = model
        tts = model.tts_model
        config = getattr(tts, "config", None)
        key = (revision, hashlib.sha256(Path(path).read_bytes()).digest(),
               str(getattr(tts, "device", "unknown")), str(getattr(config, "dtype", "unknown")),
               "reference-only-v1")
        if self.size and key in self.entries:
            self.entries.move_to_end(key)
            return isolated(self.entries[key]), True
        prepared = tts.build_prompt_cache(reference_wav_path=path)
        if hashlib.sha256(Path(path).read_bytes()).digest() != key[1]:
            raise RuntimeError("Reference file changed while preparing features")
        if not isinstance(prepared, dict) or not prepared:
            raise RuntimeError("Reference feature preparation returned an invalid cache")
        prepared = isolated(prepared)
        if self.size:
            self.entries[key] = prepared
            while len(self.entries) > self.size:
                self.entries.popitem(last=False)
        return isolated(prepared), False
