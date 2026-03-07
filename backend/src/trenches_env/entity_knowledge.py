from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_ENTITIES_ROOT = Path(__file__).resolve().parents[3] / "entities"


@lru_cache(maxsize=1)
def resolve_entities_root() -> Path:
    configured_root = os.getenv("TRENCHES_ENTITIES_ROOT")
    if configured_root:
        candidate = Path(configured_root).expanduser().resolve()
        if candidate.exists():
            return candidate

    fallback_candidates = (
        DEFAULT_ENTITIES_ROOT,
        Path.cwd() / "entities",
        Path.cwd().parent / "entities",
    )
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate

    return DEFAULT_ENTITIES_ROOT


@lru_cache(maxsize=None)
def load_entity_pack(agent_id: str) -> dict[str, Any]:
    entity_dir = resolve_entities_root() / agent_id
    profile_path = entity_dir / "profile.json"
    assets_path = entity_dir / "assets.json"

    if not profile_path.exists() or not assets_path.exists():
        return {"profile": {}, "assets": {}}

    with profile_path.open("r", encoding="utf-8") as profile_file:
        profile = json.load(profile_file)

    with assets_path.open("r", encoding="utf-8") as assets_file:
        assets = json.load(assets_file)

    return {
        "profile": profile,
        "assets": assets,
    }
