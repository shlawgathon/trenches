"""Deploy OpenAI-compatible vLLM endpoints for the six Trenches entity models.

Usage:
    modal deploy backend/serve_modal.py
"""

from __future__ import annotations

import subprocess
import os
from pathlib import Path

import modal

APP_NAME = "trenches-inference"
PYTHON_VERSION = "3.12"
PORT = 8000
VLLM_VERSION = "0.12.0"
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = "0.92"
CONTAINER_TIMEOUT_SECONDS = 60 * 60 * 24
STARTUP_TIMEOUT_SECONDS = 60 * 20

MODEL_REPOS: dict[str, str] = {
    "us": "AlazarM/trenches-us-qwen3-8b-real",
    "israel": "AlazarM/trenches-israel-qwen3-8b-real",
    "iran": "AlazarM/trenches-iran-qwen3-8b-real",
    "hezbollah": "AlazarM/trenches-hezbollah-qwen3-8b-real",
    "gulf": "AlazarM/trenches-gulf-qwen3-8b-real",
    "oversight": "AlazarM/trenches-oversight-qwen3-8b-real",
}

HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"


def _resolve_hf_token() -> str:
    if token := os.environ.get("HF_TOKEN", "").strip():
        return token
    if token := os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip():
        return token
    if HF_TOKEN_PATH.exists():
        return HF_TOKEN_PATH.read_text(encoding="utf-8").strip()
    return ""


HF_TOKEN = _resolve_hf_token()

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]>=0.36.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        }
    )
)

hf_cache = modal.Volume.from_name("trenches-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict(
    {
        "HF_TOKEN": HF_TOKEN,
        "HUGGINGFACE_HUB_TOKEN": HF_TOKEN,
    }
)


def _launch_vllm_server(model_id: str) -> None:
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--model",
        model_id,
        "--served-model-name",
        model_id,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--enforce-eager",
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    subprocess.Popen(command)


COMMON_FUNCTION_KWARGS = dict(
    image=image,
    gpu="L40S",
    cpu=8.0,
    memory=65536,
    timeout=CONTAINER_TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    scaledown_window=15 * 60,
    min_containers=0,
    max_containers=1,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
)


@app.function(**COMMON_FUNCTION_KWARGS)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS, label="trenches-us")
def serve_us() -> None:
    _launch_vllm_server(MODEL_REPOS["us"])


@app.function(**COMMON_FUNCTION_KWARGS)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS, label="trenches-israel")
def serve_israel() -> None:
    _launch_vllm_server(MODEL_REPOS["israel"])


@app.function(**COMMON_FUNCTION_KWARGS)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS, label="trenches-iran")
def serve_iran() -> None:
    _launch_vllm_server(MODEL_REPOS["iran"])


@app.function(**COMMON_FUNCTION_KWARGS)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS, label="trenches-hezbollah")
def serve_hezbollah() -> None:
    _launch_vllm_server(MODEL_REPOS["hezbollah"])


@app.function(**COMMON_FUNCTION_KWARGS)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS, label="trenches-gulf")
def serve_gulf() -> None:
    _launch_vllm_server(MODEL_REPOS["gulf"])


@app.function(**COMMON_FUNCTION_KWARGS)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS, label="trenches-oversight")
def serve_oversight() -> None:
    _launch_vllm_server(MODEL_REPOS["oversight"])
