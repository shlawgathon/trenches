#!/usr/bin/env python3
from __future__ import annotations

import os

import modal


USER = "AlazarM"
VOLUME_NAME = "trenches-checkpoints"
ENTITIES = ("us", "israel", "iran", "hezbollah", "gulf", "oversight")
TOKEN = os.environ.get("HF_TOKEN", "")

app = modal.App("trenches-upload-checkpoints")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.31.0")
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=image,
    timeout=60 * 60 * 4,
    cpu=4.0,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": TOKEN})],
)
def upload_entity(entity: str, repo_id: str) -> dict[str, object]:
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    folder = Path("/checkpoints") / f"{entity}-qwen3-8b"
    if not folder.exists():
        raise FileNotFoundError(f"Missing checkpoint folder: {folder}")

    print(f"[{entity}] entries: {sorted(p.name for p in folder.iterdir())}")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        allow_patterns=[
            "README.md",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "vocab.json",
            "training_args.bin",
            "model.safetensors.index.json",
            "model-*.safetensors",
        ],
        ignore_patterns=["checkpoint-*", "checkpoint-*/*"],
        commit_message=f"Upload {entity} Trenches Qwen3-8B checkpoint from Modal volume",
    )
    info = api.model_info(repo_id)
    return {
        "entity": entity,
        "repo_id": repo_id,
        "commit_oid": getattr(commit_info, "oid", None),
        "commit_url": getattr(commit_info, "commit_url", None),
        "siblings": len(info.siblings),
    }


@app.local_entrypoint()
def main() -> None:
    if not TOKEN:
        raise RuntimeError("HF_TOKEN must be set before running this script.")

    for entity in ENTITIES:
        repo_id = f"{USER}/trenches-{entity}-qwen3-8b-real"
        print(f"START {entity} -> {repo_id}", flush=True)
        result = upload_entity.remote(entity, repo_id)
        print(f"DONE {result}", flush=True)
