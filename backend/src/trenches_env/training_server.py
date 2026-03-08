from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI

from trenches_env.agents import AGENT_IDS
from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import SourcePacket
from trenches_env.openenv_adapter import (
    TrenchesOpenEnvEnvironment,
    create_openenv_fastapi_app,
)
from trenches_env.source_catalog import get_sources_for_agent


class NoOpSourceHarvester:
    """Offline source adapter used for replay-only post-training runs."""

    def stop(self) -> None:
        return None

    def last_sync_at(self) -> datetime | None:
        return None

    def refresh_agents(
        self,
        agent_ids: list[str] | None = None,
        *,
        include_live: bool = False,
        force: bool = False,
    ) -> dict[str, int]:
        targets = agent_ids or list(AGENT_IDS)
        return {agent_id: 0 for agent_id in targets}

    def warm_start_agents(
        self,
        agent_ids: list[str] | None = None,
        *,
        include_live: bool = False,
        max_training_sources: int = 2,
        max_live_sources: int = 1,
        force: bool = False,
    ) -> dict[str, int]:
        targets = agent_ids or list(AGENT_IDS)
        return {agent_id: 0 for agent_id in targets}

    def refresh_due_batch(self, *, include_live: bool = True) -> int:
        return 0

    def get_packets_for_agent(
        self,
        agent_id: str,
        *,
        include_live: bool = False,
    ) -> tuple[list[SourcePacket], list[SourcePacket]]:
        training_packets = [
            self._pending_packet(source.id, source.name, source.delivery, source.kind, source.endpoint.kind)
            for source in get_sources_for_agent(agent_id, "training_core")
        ]
        live_packets = [
            self._pending_packet(source.id, source.name, source.delivery, source.kind, source.endpoint.kind)
            for source in get_sources_for_agent(agent_id, "live_demo")
        ]
        return training_packets, live_packets if include_live else []

    @staticmethod
    def _pending_packet(
        source_id: str,
        source_name: str,
        delivery: str,
        kind: str,
        endpoint_kind: str,
    ) -> SourcePacket:
        return SourcePacket(
            source_id=source_id,
            source_name=source_name,
            delivery=delivery,
            kind=kind,
            endpoint_kind=endpoint_kind,
            status="pending",
            summary="Disabled during replay-only post-training.",
        )


def create_training_app() -> FastAPI:
    app = FastAPI(title="Trenches Replay Training Backend", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    openenv_app = create_openenv_fastapi_app(
        lambda: TrenchesOpenEnvEnvironment(
            env=FogOfWarDiplomacyEnv(source_harvester=NoOpSourceHarvester())
        )
    )
    if openenv_app is not None:
        app.mount("/openenv", openenv_app)

    return app
