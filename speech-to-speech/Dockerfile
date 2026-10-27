# Linux amd64 manifest of the official v0.32.0 image; CUDA 13.0.2 / PyTorch 2.13.
FROM vllm/vllm-openai:v0.32.0@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/s2s
COPY versions.json /opt/s2s/versions.json
COPY check_dependencies.py /opt/s2s/check_dependencies.py
COPY constraints.lock /opt/s2s/constraints.lock
# Preserve the image's CUDA binary stack when resolving Omni dependencies.
RUN python3 -c 'import importlib.metadata as m; from pathlib import Path; Path("/tmp/base-constraints.txt").write_text("\n".join(f"{n}=={m.version(n)}" for n in ("vllm", "torch", "torchaudio", "torchvision")))' \
    && git init /opt/vllm-omni \
    && cd /opt/vllm-omni \
    && git remote add origin https://github.com/vllm-project/vllm-omni.git \
    && git fetch --depth 1 origin 9842ebc17cf1d9c6b440ecb6a79039162f68a1dd \
    && git checkout --detach FETCH_HEAD \
    && VLLM_OMNI_VERSION_OVERRIDE=0.32.0+g9842ebc VLLM_OMNI_TARGET_DEVICE=cuda uv pip install --python "$(python3 -c 'import sys; print(sys.executable)')" --no-cache -c /tmp/base-constraints.txt -c /opt/s2s/constraints.lock . 'opentelemetry-sdk==1.40.0' 'opentelemetry-exporter-otlp==1.40.0' \
    && uv pip uninstall --python "$(python3 -c 'import sys; print(sys.executable)')" opentelemetry-exporter-otlp-common opentelemetry-exporter-http-transport \
    && python3 /opt/s2s/check_dependencies.py \
    && uv pip freeze --python "$(python3 -c 'import sys; print(sys.executable)')" > /opt/s2s/installed-packages.txt

COPY deploy.yaml gpu_policy.py gpu_policy.json runtime.py client.py api.py verify_api.py /opt/s2s/
ENTRYPOINT ["python3", "/opt/s2s/runtime.py"]
CMD ["serve"]
