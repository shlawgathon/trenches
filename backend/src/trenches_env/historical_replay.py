from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

from pydantic import BaseModel, Field

from trenches_env.models import EventSeverity, HistoricalEvent


class HistoricalReplayDefinition(BaseModel):
    replay_id: str
    name: str
    description: str
    training_agent: str = "us"
    events: list[HistoricalEvent] = Field(default_factory=list)


SEVERITY_SCORES: dict[EventSeverity, float] = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}

SEVERITY_ORDER: tuple[EventSeverity, ...] = ("low", "medium", "high", "critical")


@lru_cache(maxsize=1)
def _load_replays() -> dict[str, HistoricalReplayDefinition]:
    # Scan both dirs: historical_replays/ (curated real data) and
    # synthetic_historical_replays/ (synthetic seed data for smoke-testing).
    replay_dirs = [
        files("trenches_env").joinpath("historical_replays"),
        files("trenches_env").joinpath("synthetic_historical_replays"),
    ]
    replays: dict[str, HistoricalReplayDefinition] = {}
    for replay_dir in replay_dirs:
        try:
            children = list(replay_dir.iterdir())
        except (FileNotFoundError, TypeError):
            continue
        for child in children:
            if not str(child).endswith(".json"):
                continue
            payload = json.loads(child.read_text(encoding="utf-8"))
            replay = HistoricalReplayDefinition.model_validate(payload)
            replays[replay.replay_id] = replay
    return replays


def list_historical_replays() -> list[HistoricalReplayDefinition]:
    return [replay.model_copy(deep=True) for replay in _load_replays().values()]


def get_historical_replay(replay_id: str) -> HistoricalReplayDefinition:
    replay = _load_replays().get(replay_id)
    if replay is None:
        raise KeyError(replay_id)
    return replay.model_copy(deep=True)


def default_replay_id_for_agent(agent_id: str) -> str | None:
    for replay in _load_replays().values():
        if replay.training_agent == agent_id:
            return replay.replay_id
    return None


def severity_score(severity: EventSeverity) -> float:
    return SEVERITY_SCORES[severity]


def severity_distance(expected: EventSeverity, actual: EventSeverity) -> int:
    return abs(SEVERITY_ORDER.index(expected) - SEVERITY_ORDER.index(actual))
