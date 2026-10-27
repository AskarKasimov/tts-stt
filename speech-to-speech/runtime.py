"""Container entrypoint: validate GPU, download a pinned model, serve offline."""

import argparse
import importlib.metadata
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

from gpu_policy import validate_gpu, validate_stage_budgets

ROOT = Path(__file__).resolve().parent
VERSIONS = json.loads((ROOT / "versions.json").read_text())
PREPARED = Path("/models/prepared.json")


def gpu_check():
    import torch

    if importlib.metadata.version("vllm") != VERSIONS["vllm_version"]:
        raise RuntimeError("Версия vLLM не совпадает с versions.json; пересоберите образ.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Контейнер должен видеть ровно одну CUDA GPU. Проверьте NVIDIA Container Toolkit и GPU_ID.")
    properties = torch.cuda.get_device_properties(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU/драйвер не поддерживает BF16.")
    free, total = torch.cuda.mem_get_info()
    budget = validate_gpu(total, free, torch.cuda.get_device_capability())
    x = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
    if (x @ x).float().mean().item() != 64:
        raise RuntimeError("Проверка CUDA BF16 matmul не прошла.")
    torch.cuda.synchronize()
    from vllm_omni.config.stage_config import load_deploy_config

    deploy = load_deploy_config(ROOT / "deploy.yaml")
    validate_stage_budgets(deploy.stages)
    report = {
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "total_gib": round(total / 1024**3, 2),
        "free_gib": round(free / 1024**3, 2),
        **budget,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "vllm": importlib.metadata.version("vllm"),
        "versions": VERSIONS,
    }
    Path("/output").mkdir(exist_ok=True)
    Path("/output/gpu-check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def validate_checkpoint(path):
    """Check quantization scope, stage presence and complete safetensors payloads."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    quant = config.get("quantization_config", {})
    expected = {"quant_method": "auto-round", "bits": 4, "group_size": 128,
                "packing_format": "auto_round:auto_gptq", "block_name_to_quantize": "thinker.model.layers"}
    if any(quant.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Checkpoint имеет неожиданный формат квантования.")
    if config.get("enable_audio_output") is not True:
        raise RuntimeError("Checkpoint не поддерживает аудиовыход.")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = index["weight_map"]
    if not all(any(k.startswith(prefix + ".") for k in weights) for prefix in ("thinker", "talker", "code2wav")):
        raise RuntimeError("В checkpoint отсутствует одна из трёх стадий.")
    if not any(k.startswith("thinker.") and k.endswith(".qweight") for k in weights):
        raise RuntimeError("Не найдены INT4-веса Thinker.")
    if any(k.startswith(("talker.", "code2wav.")) and k.endswith(".qweight") for k in weights):
        raise RuntimeError("Голосовые стадии неожиданно квантованы.")
    total_bytes = 0
    seen = set()
    for name in sorted(set(weights.values())):
        shard = path / name
        if Path(name).name != name:
            raise RuntimeError("Недопустимое имя shard в индексе.")
        with shard.open("rb") as source:
            length_bytes = source.read(8)
            if len(length_bytes) != 8:
                raise RuntimeError(f"Повреждён заголовок {name}.")
            header_size = struct.unpack("<Q", length_bytes)[0]
            if not 0 < header_size < 64 * 1024**2:
                raise RuntimeError(f"Некорректный размер заголовка {name}.")
            header = json.loads(source.read(header_size))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        expected_keys = {k for k, filename in weights.items() if filename == name}
        if set(tensors) != expected_keys:
            raise RuntimeError(f"Индекс не совпадает с содержимым {name}.")
        end = max(tensor["data_offsets"][1] for tensor in tensors.values())
        if shard.stat().st_size != 8 + header_size + end:
            raise RuntimeError(f"Файл {name} скачан не полностью.")
        total_bytes += sum(t["data_offsets"][1] - t["data_offsets"][0] for t in tensors.values())
        seen.update(tensors)
    if total_bytes != VERSIONS["model_tensor_bytes"] or seen != set(weights):
        raise RuntimeError("Размер или состав весов не совпадает с закреплённой моделью.")
    return {"path": str(path), "tensor_bytes": total_bytes, "shards": len(set(weights.values()))}


def prepare():
    from huggingface_hub import snapshot_download

    PREPARED.unlink(missing_ok=True)
    path = snapshot_download(
        repo_id=VERSIONS["model_id"], revision=VERSIONS["model_revision"], max_workers=4,
    )
    checked = validate_checkpoint(path)
    manifest = {**checked, "model_id": VERSIONS["model_id"], "revision": VERSIONS["model_revision"]}
    PREPARED.write_text(json.dumps(manifest, indent=2))
    print("Веса скачаны и проверены. Можно запускать сервис.", flush=True)


def serve():
    gpu_check()
    if not PREPARED.exists():
        raise RuntimeError("Сначала выполните ./run.sh prepare или ./run.sh install.")
    manifest = json.loads(PREPARED.read_text())
    if (manifest.get("model_id"), manifest.get("revision")) != (VERSIONS["model_id"], VERSIONS["model_revision"]):
        raise RuntimeError("Кеш от другой версии модели. Выполните ./run.sh prepare.")
    checked = validate_checkpoint(manifest["path"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    args = [
        "vllm", "serve", checked["path"], "--omni",
        "--served-model-name", VERSIONS["served_model_name"],
        "--host", "0.0.0.0", "--port", "8091",
        "--deploy-config", str(ROOT / "deploy.yaml"),
    ]
    print("Запуск Thinker INT4 + Talker BF16 + Code2Wav BF16 на GPU 0.", flush=True)
    os.execvp(args[0], args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "prepare", "serve", "health"])
    action = parser.parse_args().action
    if action == "health":
        with urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("Сервис ещё не готов.")
    else:
        {"check": gpu_check, "prepare": prepare, "serve": serve}[action]()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ОШИБКА: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
