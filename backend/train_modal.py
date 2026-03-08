"""Trenches GRPO training on Modal using the local backend and replay JSON files.

Usage:
    modal run backend/train_modal.py::smoke --entity us
    modal run --detach backend/train_modal.py::train --entity us
    modal run --detach backend/train_modal.py::train_all
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import modal

app = modal.App("trenches-grpo-training")

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_BACKEND_DIR = REPO_ROOT / "backend"
LOCAL_BACKEND_SRC_DIR = LOCAL_BACKEND_DIR / "src"
LOCAL_BACKEND_PACKAGE_DIR = LOCAL_BACKEND_SRC_DIR / "trenches_env"
LOCAL_ENTITIES_DIR = REPO_ROOT / "entities"
LOCAL_REPLAY_DIR = LOCAL_BACKEND_DIR / "src" / "trenches_env" / "historical_replays"
REMOTE_REPO_ROOT = Path("/opt/trenches")
REMOTE_BACKEND_DIR = REMOTE_REPO_ROOT / "backend"
REMOTE_BACKEND_SRC_DIR = REMOTE_BACKEND_DIR / "src"
CHECKPOINTS_DIR = Path("/checkpoints")
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_ENTITY_ORDER = ("us", "israel", "iran", "hezbollah", "gulf", "oversight")
DEFAULT_REPLAY_SUFFIX = "_2025_events"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_VLLM_SERVER_PORT = 8001
MODAL_GPU_FALLBACKS = ["B200:2", "H200:2", "H100:2", "A100-80GB:2", "L40S:2"]


def _discover_local_replays() -> dict[str, str]:
    replay_ids_by_agent: dict[str, str] = {}
    if not LOCAL_REPLAY_DIR.exists():
        return replay_ids_by_agent

    for replay_file in sorted(LOCAL_REPLAY_DIR.glob("*.json")):
        try:
            payload = json.loads(replay_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        replay_id = payload.get("replay_id")
        training_agent = payload.get("training_agent")
        if not isinstance(replay_id, str) or not replay_id:
            continue
        if not isinstance(training_agent, str) or not training_agent:
            continue

        current = replay_ids_by_agent.get(training_agent)
        preferred = f"{training_agent}{DEFAULT_REPLAY_SUFFIX}"
        if current is None or replay_id == preferred:
            replay_ids_by_agent[training_agent] = replay_id

    return replay_ids_by_agent


LOCAL_REPLAYS = _discover_local_replays()
ENTITY_REPLAYS: tuple[tuple[str, str], ...] = tuple(
    (entity, LOCAL_REPLAYS.get(entity, f"{entity}{DEFAULT_REPLAY_SUFFIX}"))
    for entity in DEFAULT_ENTITY_ORDER
)

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .env({"PYTHONPATH": str(REMOTE_BACKEND_SRC_DIR)})
    .pip_install_from_pyproject(str(LOCAL_BACKEND_DIR / "pyproject.toml"), optional_dependencies=["train"])
    .pip_install("vllm==0.12.0")
    .add_local_dir(
        str(LOCAL_BACKEND_PACKAGE_DIR),
        remote_path=str(REMOTE_BACKEND_SRC_DIR / "trenches_env"),
        ignore=["**/__pycache__/**", "**/*.pyc", ".venv/**"],
    )
    .add_local_dir(
        str(LOCAL_ENTITIES_DIR),
        remote_path=str(REMOTE_REPO_ROOT / "entities"),
        ignore=["**/__pycache__/**", "**/*.pyc", ".DS_Store"],
    )
)

checkpoints_volume = modal.Volume.from_name(
    "trenches-checkpoints",
    create_if_missing=True,
)


def _resolve_replay_id(entity: str, replay_id: str | None) -> str:
    if replay_id:
        return replay_id

    known = dict(ENTITY_REPLAYS)
    if entity in known:
        return known[entity]
    return f"{entity}{DEFAULT_REPLAY_SUFFIX}"


def _build_training_command(
    *,
    entity: str,
    replay_id: str,
    model_id: str,
    max_steps: int,
    train_size: int,
    num_generations: int,
    max_prompt_length: int,
    max_completion_length: int,
) -> list[str]:
    output_dir = CHECKPOINTS_DIR / f"{entity}-qwen3-8b"
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        sys.executable,
        "-m",
        "trenches_env.training_cli",
        "--model-id",
        model_id,
        "--generation-backend",
        "vllm",
        "--vllm-mode",
        "server",
        "--port",
        str(DEFAULT_BACKEND_PORT),
        "--vllm-server-port",
        str(DEFAULT_VLLM_SERVER_PORT),
        "--training-agent",
        entity,
        "--training-stage",
        "stage_1_dense",
        "--replay-id",
        replay_id,
        "--train-size",
        str(train_size),
        "--max-steps",
        str(max_steps),
        "--num-generations",
        str(num_generations),
        "--per-device-train-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "8",
        "--learning-rate",
        "5e-6",
        "--beta",
        "0.001",
        "--warmup-steps",
        "10",
        "--temperature",
        "0.8",
        "--top-k",
        "10",
        "--top-p",
        "0.95",
        "--optim",
        "auto",
        "--max-prompt-length",
        str(max_prompt_length),
        "--max-completion-length",
        str(max_completion_length),
        "--save-strategy",
        "steps",
        "--save-steps",
        "25",
        "--output-dir",
        str(output_dir),
        "--preview-samples",
        "3",
    ]


def _wait_for_vllm_server(port: int, *, timeout_seconds: float = 240.0) -> None:
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    last_error = "unknown startup failure"

    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2.0) as response:
                status = getattr(response, "status", response.getcode())
            if status == 200:
                return
            last_error = f"HTTP {status}"
        except URLError as exc:
            last_error = str(exc)
        time.sleep(2.0)

    raise RuntimeError(f"Timed out waiting for vLLM server at {url}: {last_error}")


def _start_vllm_server(*, model_id: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    command = [
        "trl",
        "vllm-serve",
        "--model",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        str(DEFAULT_VLLM_SERVER_PORT),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(REMOTE_REPO_ROOT),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )
    try:
        _wait_for_vllm_server(DEFAULT_VLLM_SERVER_PORT)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    return process


def _run_training(
    *,
    entity: str,
    replay_id: str | None,
    model_id: str = DEFAULT_MODEL_ID,
    max_steps: int = 100,
    train_size: int = 256,
    num_generations: int = 16,
    max_prompt_length: int = 1536,
    max_completion_length: int = 256,
) -> None:
    resolved_replay_id = _resolve_replay_id(entity, replay_id)
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["TRENCHES_DISABLE_RSS"] = "1"

    vllm_process = _start_vllm_server(model_id=model_id)
    train_cmd = _build_training_command(
        entity=entity,
        replay_id=resolved_replay_id,
        model_id=model_id,
        max_steps=max_steps,
        train_size=train_size,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )

    print(f"Starting GRPO training for {entity} with replay {resolved_replay_id}")
    training_env = os.environ.copy()
    training_env["CUDA_VISIBLE_DEVICES"] = "1"
    try:
        result = subprocess.run(
            train_cmd,
            cwd=str(REMOTE_REPO_ROOT),
            env=training_env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
    finally:
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_process.kill()

    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for {entity} on replay {resolved_replay_id} with exit code {result.returncode}"
        )

    print(f"Training complete for {entity}. Replay: {resolved_replay_id}")
    checkpoints_volume.commit()


@app.function(
    image=training_image,
    gpu=MODAL_GPU_FALLBACKS,
    timeout=60 * 60 * 4,
    volumes={str(CHECKPOINTS_DIR): checkpoints_volume},
)
def train(
    entity: str,
    replay_id: str | None = None,
    max_steps: int = 100,
    train_size: int = 256,
    num_generations: int = 16,
) -> None:
    _run_training(
        entity=entity,
        replay_id=replay_id,
        max_steps=max_steps,
        train_size=train_size,
        num_generations=num_generations,
    )


@app.function(
    image=training_image,
    gpu=MODAL_GPU_FALLBACKS,
    timeout=60 * 60 * 2,
    volumes={str(CHECKPOINTS_DIR): checkpoints_volume},
)
def smoke(entity: str = "us", replay_id: str | None = None) -> None:
    _run_training(
        entity=entity,
        replay_id=replay_id,
        model_id=DEFAULT_MODEL_ID,
        max_steps=1,
        train_size=4,
        num_generations=2,
        max_prompt_length=512,
        max_completion_length=64,
    )


@app.local_entrypoint()
def train_all() -> None:
    print(f"Launching {len(ENTITY_REPLAYS)} training jobs in parallel")
    list(train.starmap(ENTITY_REPLAYS))
    print("All training jobs completed.")
    print("Download checkpoints with:")
    print("  modal volume get trenches-checkpoints .")
