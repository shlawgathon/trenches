from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
ENTITIES_ROOT = REPO_ROOT / "entities"


@lru_cache(maxsize=None)
def load_entity_pack(agent_id: str) -> dict[str, Any]:
    entity_dir = ENTITIES_ROOT / agent_id
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
