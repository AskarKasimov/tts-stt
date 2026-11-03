import pytest

import api
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm2 import VoxCPM2Model


def test_prompt_cache_reaches_model_tokenizer_with_separator():
    tokenized = []

    class StopAfterTokenization(Exception):
        pass

    class TokenizerModel:
        def _generate_with_prompt_cache(self, **kwargs):
            return VoxCPM2Model._generate_with_prompt_cache(self, **kwargs)

        def text_tokenizer(self, text):
            tokenized.append(text)
            raise StopAfterTokenization

    pipeline = object.__new__(VoxCPM)
    pipeline.tts_model = TokenizerModel()
    pipeline.denoiser = None
    cache = {"mode": "ref_continuation", "prompt_text": api.REFERENCE_TEXT + " "}
    with pytest.raises(StopAfterTokenization):
        pipeline.generate(text="ответ", prompt_cache=cache)
    assert tokenized == [api.REFERENCE_TEXT + " ответ"]


def test_prebuilt_cache_rejects_conflicting_inputs():
    pipeline = object.__new__(VoxCPM)
    pipeline.tts_model = object()
    with pytest.raises(ValueError, match="prompt_cache cannot be combined"):
        pipeline.generate(text="ответ", prompt_cache={}, reference_wav_path="unused.wav")
