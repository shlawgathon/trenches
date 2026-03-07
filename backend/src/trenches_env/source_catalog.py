from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Annotated, Literal

from pydantic import BaseModel, Field

SourceDelivery = Literal["training_core", "live_demo"]
SourceKind = Literal["rss", "api", "scrape", "telegram", "structured", "video"]


class UrlEndpoint(BaseModel):
    kind: Literal["url"]
    url: str
    method: str | None = None


class WorldMonitorEndpoint(BaseModel):
    kind: Literal["worldmonitor"]
    rpc: str
    selector: str | None = None


class TelegramEndpoint(BaseModel):
    kind: Literal["telegram"]
    handle: str


class VideoEndpoint(BaseModel):
    kind: Literal["video"]
    channel: str


SourceEndpoint = Annotated[
    UrlEndpoint | WorldMonitorEndpoint | TelegramEndpoint | VideoEndpoint,
    Field(discriminator="kind"),
]


class SourceSpec(BaseModel):
    id: str
    agentId: str
    delivery: SourceDelivery
    name: str
    kind: SourceKind
    endpoint: SourceEndpoint
    auth: str
    allowlistStatus: str
    tags: list[str] = Field(default_factory=list)
    rationale: str
    notes: str | None = None


class SourceManifest(BaseModel):
    generatedAt: str
    sourceCount: int
    sources: list[SourceSpec]


@lru_cache(maxsize=1)
def load_source_manifest() -> SourceManifest:
    raw = files("trenches_env").joinpath("source_manifest.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    manifest = SourceManifest.model_validate(payload)
    if manifest.sourceCount != len(manifest.sources):
        raise ValueError("source manifest count does not match actual source length")
    return manifest


def get_all_sources() -> list[SourceSpec]:
    return list(load_source_manifest().sources)


def get_sources_for_agent(agent_id: str, delivery: SourceDelivery | None = None) -> list[SourceSpec]:
    sources = [source for source in load_source_manifest().sources if source.agentId == agent_id]
    if delivery is not None:
        sources = [source for source in sources if source.delivery == delivery]
    return sources


def get_source_by_id(source_id: str) -> SourceSpec:
    for source in load_source_manifest().sources:
        if source.id == source_id:
            return source
    raise KeyError(source_id)
