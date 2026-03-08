#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware


SPACE_REPO = "AlazarM/trenches-us-qwen3-8b-chat"
MODEL_REPO = "AlazarM/trenches-us-qwen3-8b-real"
SPACE_DIR = Path(__file__).resolve().parents[1] / "hf_spaces" / "us-inference-space"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be set.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=SPACE_REPO,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message="Deploy US Qwen3-8B chat Space",
    )
    api.add_space_secret(SPACE_REPO, "HF_TOKEN", token)
    api.add_space_variable(SPACE_REPO, "MODEL_REPO", MODEL_REPO)
    api.request_space_hardware(SPACE_REPO, SpaceHardware.T4_SMALL, sleep_time=3600)
    api.restart_space(SPACE_REPO, factory_reboot=True)

    for _ in range(12):
        runtime = api.get_space_runtime(SPACE_REPO)
        stage = getattr(runtime, "stage", None)
        hardware = getattr(runtime, "hardware", None)
        requested = getattr(runtime, "requested_hardware", None)
        print(f"stage={stage} hardware={hardware} requested={requested}", flush=True)
        time.sleep(10)

    print(f"https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
