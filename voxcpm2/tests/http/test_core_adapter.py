"""Exercise the public core method without importing torch or loading weights."""
import ast
import re
import threading
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def core_generate():
    source = Path("src/voxcpm/core.py").read_text()
    tree = ast.parse(source)
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "VoxCPM")
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "_generate")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    scope = {"re": re, "os": __import__("os"), "VoxCPM2Model": TTS, "next_and_close": lambda it: next(it)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "core.py", "exec"), scope)
    return scope["_generate"]


class Tensor:
    def squeeze(self, axis):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return np.zeros(4)


class TTS:
    def _generate_with_prompt_cache(self, **kwargs):
        self.options = kwargs
        yield Tensor(), None, None
    def build_prompt_cache(self, **kwargs):
        raise AssertionError("Supplied features must not be encoded again")


def test_supplied_features_keep_public_normalization_retry_seed_and_waveform():
    tts = TTS()
    model = SimpleNamespace(tts_model=tts, denoiser=None, text_normalizer=SimpleNamespace(normalize=lambda text: text + "!"))
    cache = {"reference": "features"}
    output = list(core_generate()(model, " Раз  два\nтри", prompt_cache=cache, normalize=True, seed=510))
    assert len(output) == 1 and output[0].shape == (4,)
    assert tts.options["target_text"] == " Раз два три!"
    assert tts.options["prompt_cache"] is cache
    assert tts.options["retry_badcase"] is True
    assert tts.options["seed"] == 510


def test_supplied_features_reject_conflicting_paths():
    with pytest.raises(ValueError, match="prompt_cache"):
        list(core_generate()(SimpleNamespace(tts_model=TTS()), "Ответ", prompt_cache={"x": 1}, reference_wav_path="not-present.wav"))


def test_public_core_forwards_stream_cancellation_and_disables_only_stream_retries():
    tts = TTS()
    model = SimpleNamespace(tts_model=tts, denoiser=None)
    cancel = threading.Event()
    list(core_generate()(model, "Ответ", streaming=True, retry_badcase=False, cancel_event=cancel))
    assert tts.options["cancel_event"] is cancel
    assert tts.options["retry_badcase"] is False


def test_diffusion_cancellation_is_checked_before_each_estimator_step():
    source = Path("src/voxcpm/modules/locdit/unified_cfm.py").read_text()
    tree = ast.parse(source)
    method = next(item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == "solve_euler")
    method.decorator_list = []
    for arg in method.args.args:
        arg.annotation = None
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "unified_cfm.py", "exec"), scope)
    event = threading.Event()
    event.set()
    with pytest.raises(RuntimeError, match="cancel"):
        scope["solve_euler"](None, None, [1, 0.5, 0], None, None, cancel_event=event)


@pytest.mark.parametrize("failure", [False, True])
def test_upstream_optimizer_reports_swallowed_errors(monkeypatch, failure):
    tree = ast.parse(Path("src/voxcpm/model/voxcpm2.py").read_text())
    method = next(item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == "optimize")
    for arg in method.args.args:
        arg.annotation = None
    def compile_model(fn, **kwargs):
        if failure:
            raise RuntimeError("compile unavailable")
        return fn
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    scope = {"torch": SimpleNamespace(compile=compile_model), "sys": sys, "time": __import__("time")}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "voxcpm2.py", "exec"), scope)
    model = SimpleNamespace(device="cuda:0", base_lm=SimpleNamespace(forward_step=lambda: None), residual_lm=SimpleNamespace(forward_step=lambda: None), feat_encoder=lambda: None, feat_decoder=SimpleNamespace(estimator=lambda: None))
    assert scope["optimize"](model) is model
    assert model.optimization_enabled is (not failure)
    assert (model.optimization_error is not None) is failure


def test_closing_public_stream_closes_inner_generator():
    closed = threading.Event()
    class ClosingTTS(TTS):
        def _generate_with_prompt_cache(self, **kwargs):
            try:
                yield Tensor(), None, None
                yield Tensor(), None, None
            finally:
                closed.set()
    model = SimpleNamespace(tts_model=ClosingTTS(), denoiser=None)
    generator = core_generate()(model, "Ответ", streaming=True, retry_badcase=False)
    next(generator)
    generator.close()
    assert closed.is_set()
