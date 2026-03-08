#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware


USER = "AlazarM"
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "hf_spaces" / "training-space-template"
GIT_REF = "main"
TOKEN = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
API = HfApi(token=TOKEN)

SPACES = [
    ("us", "us_2025_events"),
    ("israel", "israel_2025_events"),
    ("iran", "iran_2025_events"),
    ("hezbollah", "hezbollah_2025_events"),
    ("gulf", "gulf_2025_events"),
    ("oversight", "oversight_2025_events"),
]

COMMON_VARS = {
    "GIT_REPO_URL": "https://github.com/shlawgathon/trenches.git",
    "GIT_REF": GIT_REF,
    "MODEL_ID": "Qwen/Qwen3-8B",
    "GENERATION_BACKEND": "vllm",
    "VLLM_VERSION": "vllm==0.12.0",
    "TRAINING_STAGE": "stage_1_dense",
    "TRAIN_SIZE": "4",
    "MAX_STEPS": "1",
    "NUM_GENERATIONS": "2",
    "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
    "GRADIENT_ACCUMULATION_STEPS": "1",
    "LEARNING_RATE": "5e-6",
    "BETA": "0.001",
    "WARMUP_STEPS": "0",
    "TEMPERATURE": "0.8",
    "TOP_K": "10",
    "TOP_P": "0.95",
    "OPTIM": "adamw_bnb_8bit",
    "MAX_PROMPT_LENGTH": "512",
    "MAX_COMPLETION_LENGTH": "64",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.18",
    "VLLM_ENABLE_SLEEP_MODE": "true",
    "SAVE_STRATEGY": "no",
    "UPLOAD_MESSAGE": "Upload Trenches vLLM pilot checkpoint",
}


def upsert_var(repo_id: str, key: str, value: str) -> None:
    API.add_space_variable(repo_id, key, value)


def main() -> None:
    for entity, replay_id in SPACES:
        repo_id = f"{USER}/trenches-train-{entity}"
        model_repo = f"{USER}/trenches-{entity}-qwen3-8b-real"

        API.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        API.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(TEMPLATE_DIR),
            commit_message=f"Deploy training space template for {entity}",
        )

        API.add_space_secret(repo_id, "HF_TOKEN", TOKEN)
        upsert_var(repo_id, "ENTITY", entity)
        upsert_var(repo_id, "REPLAY_ID", replay_id)
        upsert_var(repo_id, "MODEL_REPO", model_repo)
        for key, value in COMMON_VARS.items():
            upsert_var(repo_id, key, value)

        API.request_space_hardware(repo_id, SpaceHardware.A100_LARGE, sleep_time=172800)
        API.restart_space(repo_id, factory_reboot=True)
        print(repo_id)


if __name__ == "__main__":
    main()
