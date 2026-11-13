"""Real model HTTP handlers with fake inference for the Go/Python contract smoke."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import uvicorn
from speech_receiver.config import Config
from speech_receiver.http import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["stt", "tts"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--brokers", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / ("gigaam" if args.operation == "stt" else "voxcpm2") / "api.py"
    spec = importlib.util.spec_from_file_location("contract_model_api", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if args.operation == "stt":

        class ASR:
            def transcribe(self, *_args, **_kwargs):
                return SimpleNamespace(text="контракт", words=[])

        app = module.create_app(model_loader=lambda _: ASR())
        validator = module.resolve_model_name
    else:

        class TTS:
            tts_model = SimpleNamespace(sample_rate=16000)

            def generate(self, **_kwargs):
                return np.zeros(160, dtype=np.float32)

        module._load_model = lambda: TTS()
        app = module.app
        validator = module.SynthesisRequest.model_validate
    cfg = Config(
        operation=args.operation,
        data_dir=args.data_dir,
        brokers=args.brokers.split(","),
        topic=args.topic,
        clients={
            "dev": {
                "token": "local-development-client-token-0001",
                "priority": "normal",
            }
        },
        instances=[],
    )
    app = install(app, cfg, validator)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
