"""Trenches GRPO training on Modal with vLLM colocate mode.

Uses a single H200 (141GB VRAM) per entity — enough for policy + ref + vLLM + optimizer.

Usage:
    # Smoke test (single entity, 1 step):
    modal run train_modal.py::smoke --entity us

    # Full training (single entity, 100 steps):
    modal run --detach train_modal.py::train --entity us --replay-id us_synthetic_seed_2025_2026

    # Full training (all 6 entities in parallel):
    modal run --detach train_modal.py::train_all
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("trenches-grpo-training")

# ---------------------------------------------------------------------------
# Image: install trl, vllm, transformers, and the trenches backend
# ---------------------------------------------------------------------------
GIT_REPO_URL = "https://github.com/shlawgathon/trenches.git"
GIT_REF = "main"

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .env({"IMAGE_VERSION": "4"})  # bump to force rebuild when main changes
    .run_commands(
        f"git clone --depth 1 --branch {GIT_REF} --single-branch {GIT_REPO_URL} /opt/trenches",
        "uv pip install --system -e '/opt/trenches/backend[train]'",
    )
    .uv_pip_install(
        "trl==0.29.0",
        "vllm==0.12.0",
        "transformers>=4.57",
        "huggingface_hub",
    )
)

# ---------------------------------------------------------------------------
# Volume: persistent checkpoint storage
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = Path("/checkpoints")
checkpoints_volume = modal.Volume.from_name(
    "trenches-checkpoints", create_if_missing=True
)

# ---------------------------------------------------------------------------
# Default hyperparameters (from POST_TRAINING_PLAN.md)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"

ENTITIES = [
    ("us", "us_synthetic_seed_2025_2026"),
    ("israel", "israel_synthetic_seed_2025_2026"),
    ("iran", "iran_synthetic_seed_2025_2026"),
    ("hezbollah", "hezbollah_synthetic_seed_2025_2026"),
    ("gulf", "gulf_synthetic_seed_2025_2026"),
    ("oversight", "oversight_2025_events"),
]


def _run_training(
    entity: str,
    replay_id: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_steps: int = 100,
    train_size: int = 256,
    num_generations: int = 16,
    max_prompt_length: int = 1536,
    max_completion_length: int = 256,
) -> None:
    """Run GRPO training with vLLM colocate mode on a single GPU."""
    output_dir = CHECKPOINTS_DIR / f"{entity}-qwen3-8b"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Required env vars for single-process colocate mode
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["TRENCHES_DISABLE_RSS"] = "1"

    train_cmd = [
        sys.executable, "-m", "trenches_env.training_cli",
        "--model-id", model_id,
        "--generation-backend", "vllm",
        "--training-agent", entity,
        "--training-stage", "stage_1_dense",
        "--replay-id", replay_id,
        "--train-size", str(train_size),
        "--max-steps", str(max_steps),
        "--num-generations", str(num_generations),
        "--per-device-train-batch-size", "1",
        "--gradient-accumulation-steps", "8",
        "--learning-rate", "5e-6",
        "--beta", "0.001",
        "--warmup-steps", "10",
        "--temperature", "0.8",
        "--top-k", "10",
        "--top-p", "0.95",
        "--optim", "adamw_bnb_8bit",
        "--max-prompt-length", str(max_prompt_length),
        "--max-completion-length", str(max_completion_length),
        "--save-strategy", "steps",
        "--save-steps", "25",
        "--output-dir", str(output_dir),
        "--preview-samples", "3",
        "--no-vllm-enable-sleep-mode",  # H200 has 141GB — all 3 models fit without sleep mode
    ]

    print(f"Starting GRPO training for {entity} (colocate mode, single GPU)")
    result = subprocess.run(
        train_cmd,
        cwd="/opt/trenches",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {entity} with exit code {result.returncode}")

    print(f"Training complete for {entity}. Checkpoint at {output_dir}")
    checkpoints_volume.commit()


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------
@app.function(
    image=training_image,
    gpu="H200",
    timeout=60 * 60 * 4,  # 4 hours max
    volumes={str(CHECKPOINTS_DIR): checkpoints_volume},
)
def train(
    entity: str,
    replay_id: str,
    max_steps: int = 100,
    train_size: int = 256,
    num_generations: int = 16,
) -> None:
    """Train a single entity with vLLM colocate mode on H200 (141GB)."""
    _run_training(
        entity=entity,
        replay_id=replay_id,
        max_steps=max_steps,
        train_size=train_size,
        num_generations=num_generations,
    )


@app.function(
    image=training_image,
    gpu="H200",
    timeout=60 * 60 * 4,
    volumes={str(CHECKPOINTS_DIR): checkpoints_volume},
)
def smoke(entity: str = "us") -> None:
    """Quick smoke test: 1 step, 4 samples, 2 generations."""
    replay_id = dict(ENTITIES).get(entity, f"{entity}_synthetic_seed_2025_2026")
    _run_training(
        entity=entity,
        replay_id=replay_id,
        max_steps=1,
        train_size=4,
        num_generations=2,
        max_prompt_length=512,
        max_completion_length=64,
    )


@app.local_entrypoint()
def train_all() -> None:
    """Launch all 6 entities in parallel."""
    print(f"Launching {len(ENTITIES)} training jobs in parallel")
    results = list(train.starmap(ENTITIES))
    print(f"All {len(results)} training jobs completed!")
    print("Download checkpoints with:")
    print("  modal volume get trenches-checkpoints .")
