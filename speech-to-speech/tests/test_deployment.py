"""CPU checks for failure modes; these do not run the real model."""

import base64
import copy
import importlib.util
import io
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load("runtime")
client = load("client")
dependencies = load("check_dependencies")


def wav_bytes(*, silence=False):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<h", 0 if silence else 1200) * 1600)
    return output.getvalue()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.config = {
            "enable_audio_output": True,
            "quantization_config": {
                "quant_method": "auto-round", "bits": 4, "group_size": 128,
                "packing_format": "auto_round:auto_gptq", "block_name_to_quantize": "thinker.model.layers",
            },
        }
        (self.path / "config.json").write_text(json.dumps(self.config))
        names = ["thinker.model.layers.0.qweight", "talker.weight", "code2wav.weight"]
        header = {name: {"dtype": "I32", "shape": [1], "data_offsets": [i * 4, i * 4 + 4]}
                  for i, name in enumerate(names)}
        data = json.dumps(header).encode()
        self.shard = self.path / "model.safetensors"
        self.shard.write_bytes(struct.pack("<Q", len(data)) + data + bytes(12))
        self.index = {"weight_map": {name: self.shard.name for name in names}}
        (self.path / "model.safetensors.index.json").write_text(json.dumps(self.index))
        versions = copy.deepcopy(runtime.VERSIONS)
        versions["model_tensor_bytes"] = 12
        self.context = patch.object(runtime, "VERSIONS", versions)
        self.context.start()
        self.addCleanup(self.context.stop)

    def test_complete_checkpoint(self):
        self.assertEqual(runtime.validate_checkpoint(self.path)["tensor_bytes"], 12)

    def test_truncated_download_rejected(self):
        self.shard.write_bytes(self.shard.read_bytes()[:-1])
        with self.assertRaisesRegex(RuntimeError, "не полностью"):
            runtime.validate_checkpoint(self.path)

    def test_missing_voice_stage_rejected(self):
        del self.index["weight_map"]["talker.weight"]
        (self.path / "model.safetensors.index.json").write_text(json.dumps(self.index))
        with self.assertRaisesRegex(RuntimeError, "трёх стадий"):
            runtime.validate_checkpoint(self.path)

    def test_wrong_quantization_rejected(self):
        self.config["quantization_config"]["bits"] = 8
        (self.path / "config.json").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(RuntimeError, "формат квантования"):
            runtime.validate_checkpoint(self.path)

    def test_unprepared_server_does_not_start(self):
        with patch.object(runtime, "gpu_check"), patch.object(runtime, "PREPARED", self.path / "missing.json"):
            with self.assertRaisesRegex(RuntimeError, "Сначала"):
                runtime.serve()


class ClientTests(unittest.TestCase):
    def test_silence_rejected(self):
        with self.assertRaisesRegex(ValueError, "тишину"):
            client.validate_wav(wav_bytes(silence=True))

    def test_truncated_wav_rejected(self):
        with self.assertRaisesRegex(ValueError, "обрезано"):
            client.validate_wav(wav_bytes()[:-2])

    def test_duration_limit(self):
        with self.assertRaisesRegex(ValueError, "длиннее"):
            client.validate_wav(wav_bytes(), max_seconds=0.05)

    def test_audio_input_and_separate_audio_choice(self):
        audio = wav_bytes()
        result = {"choices": [
            {"message": {"content": "Четыре."}},
            {"message": {"audio": {"data": base64.b64encode(audio).decode()}}},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "question.wav"
            target = Path(directory) / "answer.wav"
            source.write_bytes(audio)
            with patch.object(client.urllib.request, "urlopen") as mocked:
                mocked.return_value.__enter__.return_value = io.BytesIO(json.dumps(result).encode())
                client.speech_reply("http://localhost:8091", source, target)
                request = mocked.call_args.args[0]
                payload = json.loads(request.data)
                url = payload["messages"][1]["content"][0]["audio_url"]["url"]
                self.assertEqual(base64.b64decode(url.split(",", 1)[1]), audio)
                self.assertEqual(payload["modalities"], ["text", "audio"])
            self.assertEqual(target.read_bytes(), audio)
            self.assertEqual(target.with_suffix(".txt").read_text(), "Четыре.")

    def test_text_only_response_is_failure(self):
        result = {"choices": [{"message": {"content": "Ответ без звука"}}]}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "answer.wav"
            with patch.object(client.urllib.request, "urlopen") as mocked:
                mocked.return_value.__enter__.return_value = io.BytesIO(json.dumps(result).encode())
                with self.assertRaisesRegex(RuntimeError, "одновременно текст и аудио"):
                    client.request_reply("http://localhost:8091", "Тест", target)
            self.assertFalse(target.exists())


class DependencyTests(unittest.TestCase):
    @staticmethod
    def distribution(name, version, requires=()):
        return SimpleNamespace(metadata={"Name": name}, version=version, requires=requires)

    def run_check(self, distributions):
        with patch.object(dependencies.importlib.metadata, "distributions", return_value=distributions):
            with patch("sys.stdout", new=io.StringIO()):
                dependencies.main()

    def test_lmcache_telemetry_conflict_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "lmcache"):
            self.run_check([
                self.distribution("lmcache", "1.0", ["opentelemetry-api<=1.40.0"]),
                self.distribution("opentelemetry-api", "1.45.1"),
            ])

    def test_shadowed_apt_metadata_does_not_override_imported_package(self):
        self.run_check([
            self.distribution("application", "1.0", ["cryptography>=47"]),
            self.distribution("cryptography", "47.0.0"),
            self.distribution("cryptography", "41.0.7"),
        ])

    def test_only_exact_documented_nccl_override_is_allowed(self):
        torch = self.distribution("torch", "2.13.0", ["nvidia-nccl-cu13==2.29.7"])
        self.run_check([torch, self.distribution("nvidia-nccl-cu13", "2.30.7")])
        with self.assertRaisesRegex(SystemExit, "nvidia-nccl"):
            self.run_check([torch, self.distribution("nvidia-nccl-cu13", "2.31.0")])


if __name__ == "__main__":
    unittest.main()
