from __future__ import annotations

import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Protocol

import httpx

from trenches_env.agents import AGENT_IDS
from trenches_env.models import SourcePacket, utc_now
from trenches_env.source_catalog import SourceSpec, get_all_sources, get_sources_for_agent

_USER_AGENT = "trenches-source-harvester/0.1 (+https://github.com/koala73/worldmonitor)"

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_HTML_META_RE = re.compile(
    r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
_HTML_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_TELEGRAM_MESSAGE_RE = re.compile(
    r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)

_WORLDMONITOR_PROBE_URLS: dict[str, str] = {
    "aviation/v1/list-airport-delays": "https://www.aviationstack.com/",
    "climate/v1/list-climate-anomalies": "https://open-meteo.com/",
    "conflict/v1/get-humanitarian-summary": "https://data.humdata.org/",
    "conflict/v1/list-acled-events": "https://acleddata.com/",
    "conflict/v1/list-iran-events": "https://liveuamap.com/",
    "conflict/v1/list-ucdp-events": "https://ucdp.uu.se/",
    "cyber/v1/list-cyber-threats": "https://www.abuse.ch/",
    "displacement/v1/get-displacement-summary": "https://data.humdata.org/",
    "displacement/v1/get-population-exposure": "https://data.humdata.org/",
    "economic/v1/list-gulf-fdi": "https://www.worldbank.org/en/topic/financialsector/brief/foreign-direct-investment",
    "infrastructure/v1/get-cable-health": "https://www.submarinecablemap.com/",
    "infrastructure/v1/list-internet-outages": "https://radar.cloudflare.com/outage-center",
    "intelligence/v1/get-country-intel-brief": "https://www.gdeltproject.org/",
    "intelligence/v1/get-pizzint-status": "https://www.gdeltproject.org/",
    "intelligence/v1/get-risk-scores": "https://www.gdeltproject.org/",
    "intelligence/v1/search-gdelt-documents": "https://www.gdeltproject.org/",
    "maritime/v1/list-navigational-warnings": "https://msi.nga.mil/",
    "market/v1/get-country-stock-index": "https://finance.yahoo.com/",
    "market/v1/list-commodity-quotes": "https://finance.yahoo.com/commodities/",
    "market/v1/list-gulf-quotes": "https://finance.yahoo.com/",
    "military/v1/get-theater-posture": "https://news.usni.org/",
    "military/v1/list-military-bases": "https://www.globalsecurity.org/",
    "military/v1/list-military-flights": "https://opensky-network.org/",
    "military/v1/list-oref-alerts": "https://www.oref.org.il/eng",
    "natural/v1/list-natural-events": "https://eonet.gsfc.nasa.gov/",
    "news/v1/list-feed-digest": "https://news.google.com/",
    "prediction/v1/list-prediction-markets": "https://polymarket.com/",
    "seismology/v1/list-earthquakes": "https://earthquake.usgs.gov/",
    "supply-chain/v1/get-chokepoint-status": "https://www.marinetraffic.com/",
    "supply-chain/v1/get-critical-minerals": "https://www.bgs.ac.uk/mineralsuk/",
    "supply-chain/v1/get-shipping-rates": "https://www.freightos.com/",
    "trade/v1/get-tariff-trends": "https://www.wto.org/",
    "trade/v1/get-trade-restrictions": "https://www.wto.org/",
    "unrest/v1/list-unrest-events": "https://acleddata.com/",
    "wildfire/v1/list-fire-detections": "https://firms.modaps.eosdis.nasa.gov/map/",
}

_SOURCE_ID_PROBE_URLS: dict[str, list[str]] = {
    "israel-oref": ["https://www.idf.il/en/mini-sites/home-front-command/"],
    "israel-opensky-flights": ["https://openskynetwork.github.io/opensky-api/"],
    "israel-wingbits-enrichment": ["https://docs.wingbits.com/"],
    "israel-tel-aviv-webcam": ["https://www.youtube.com/watch?v=gmtlJ_m2r5A"],
    "iran-tehran-webcam": ["https://www.youtube.com/watch?v=-zGuR1qVKrU"],
    "hezbollah-humanitarian-summary": ["https://www.unhcr.org/"],
    "hezbollah-rudaw-live": ["https://svs.itworkscdn.net/rudawlive/rudawlive.smil/playlist.m3u8"],
    "gulf-aljazeera-arabic-live": ["https://www.youtube.com/watch?v=bNyUyrR0PHo"],
    "gulf-middle-east-webcam": ["https://www.youtube.com/watch?v=4E-iFtUM2kk"],
    "oversight-hapi-displacement": ["https://www.unhcr.org/"],
    "oversight-worldpop-exposure": ["https://hub.worldpop.org/"],
}


class SourceFetchError(RuntimeError):
    pass


class SourceFetcher(Protocol):
    def fetch(self, url: str) -> tuple[str, str]:
        ...


class HttpSourceFetcher:
    def __init__(self, timeout_seconds: float = 8.0) -> None:
        self._client = httpx.Client(
            follow_redirects=True,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "*/*",
            },
            timeout=timeout_seconds,
        )

    def fetch(self, url: str) -> tuple[str, str]:
        response = self._client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "text/plain")
        return response.text[:200_000], content_type

    def close(self) -> None:
        self._client.close()


class SourceProbeResolver:
    def resolve_candidates(self, source: SourceSpec) -> list[str]:
        if source.id in _SOURCE_ID_PROBE_URLS:
            return _SOURCE_ID_PROBE_URLS[source.id]
        endpoint = source.endpoint
        if endpoint.kind == "url":
            return [endpoint.url]
        if endpoint.kind == "telegram":
            handle = endpoint.handle.lstrip("@")
            return [f"https://t.me/s/{handle}"]
        if endpoint.kind == "video":
            channel = endpoint.channel if endpoint.channel.startswith("@") else f"@{endpoint.channel}"
            return [f"https://www.youtube.com/{channel}/live", f"https://www.youtube.com/{channel}"]
        if endpoint.kind == "worldmonitor":
            probe_url = _WORLDMONITOR_PROBE_URLS.get(endpoint.rpc)
            return [probe_url] if probe_url else []
        return []


class SourceHarvester:
    def __init__(
        self,
        fetcher: SourceFetcher | None = None,
        *,
        auto_start: bool = False,
        poll_interval_seconds: float = 20.0,
        batch_size: int = 8,
    ) -> None:
        self.fetcher = fetcher or HttpSourceFetcher()
        self.probe_resolver = SourceProbeResolver()
        self.poll_interval_seconds = poll_interval_seconds
        self.batch_size = batch_size
        self._cache: dict[str, SourcePacket] = {}
        self._cursor = 0
        self._last_sync_at: datetime | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        if auto_start:
            self.start()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_poll_loop, name="trenches-source-harvester", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if hasattr(self.fetcher, "close"):
            self.fetcher.close()  # type: ignore[call-arg]

    def last_sync_at(self) -> datetime | None:
        with self._lock:
            return self._last_sync_at

    def refresh_agents(
        self,
        agent_ids: list[str] | None = None,
        *,
        include_live: bool = False,
        force: bool = False,
    ) -> dict[str, int]:
        targets = agent_ids or list(AGENT_IDS)
        refreshed_by_agent = {agent_id: 0 for agent_id in targets}
        for agent_id in targets:
            for source in self._iter_agent_sources(agent_id, include_live=include_live):
                packet = self._cache.get(source.id)
                if not force and packet is not None and not self._is_due(packet, source):
                    continue
                self._store_packet(self._collect_source(source))
                refreshed_by_agent[agent_id] += 1
        return refreshed_by_agent

    def probe_source(self, source: SourceSpec) -> SourcePacket:
        packet = self._collect_source(source)
        self._store_packet(packet)
        return packet

    def refresh_due_batch(self, *, include_live: bool = True) -> int:
        sources = [
            source
            for source in get_all_sources()
            if include_live or source.delivery == "training_core"
        ]
        refreshed = 0
        if not sources:
            return refreshed

        for _ in range(len(sources)):
            source = sources[self._cursor % len(sources)]
            self._cursor += 1
            packet = self._cache.get(source.id)
            if packet is not None and not self._is_due(packet, source):
                continue
            self._store_packet(self._collect_source(source))
            refreshed += 1
            if refreshed >= self.batch_size:
                break
        return refreshed

    def get_packets_for_agent(
        self,
        agent_id: str,
        *,
        include_live: bool = False,
    ) -> tuple[list[SourcePacket], list[SourcePacket]]:
        training_packets = [
            self._cache.get(source.id) or self._pending_packet(source)
            for source in get_sources_for_agent(agent_id, "training_core")
        ]
        live_packets = [
            self._cache.get(source.id) or self._pending_packet(source)
            for source in get_sources_for_agent(agent_id, "live_demo")
        ]
        return training_packets, live_packets if include_live else []

    def all_sources_have_probe_targets(self) -> bool:
        return all(bool(self.probe_resolver.resolve_candidates(source)) for source in get_all_sources())

    def _run_poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.refresh_due_batch(include_live=True)
            except Exception:
                pass
            self._stop_event.wait(self.poll_interval_seconds)

    def _iter_agent_sources(self, agent_id: str, *, include_live: bool) -> list[SourceSpec]:
        sources = get_sources_for_agent(agent_id, "training_core")
        if include_live:
            sources += get_sources_for_agent(agent_id, "live_demo")
        return sources

    def _store_packet(self, packet: SourcePacket) -> None:
        with self._lock:
            self._cache[packet.source_id] = packet
            if packet.fetched_at is not None:
                self._last_sync_at = packet.fetched_at

    def _collect_source(self, source: SourceSpec) -> SourcePacket:
        probe_urls = self.probe_resolver.resolve_candidates(source)
        if not probe_urls:
            return SourcePacket(
                source_id=source.id,
                source_name=source.name,
                delivery=source.delivery,
                kind=source.kind,
                endpoint_kind=source.endpoint.kind,
                status="error",
                error=f"No probe target configured for {source.endpoint.kind}:{getattr(source.endpoint, 'kind', 'unknown')}",
            )

        last_error: str | None = None
        for probe_url in probe_urls:
            try:
                raw_text, content_type = self.fetcher.fetch(probe_url)
                summary, sample_items = self._extract_summary(source, raw_text, content_type)
                return SourcePacket(
                    source_id=source.id,
                    source_name=source.name,
                    delivery=source.delivery,
                    kind=source.kind,
                    endpoint_kind=source.endpoint.kind,
                    status="ok",
                    fetched_at=utc_now(),
                    probe_url=probe_url,
                    summary=summary,
                    sample_items=sample_items,
                )
            except Exception as exc:
                last_error = str(exc)

        return SourcePacket(
            source_id=source.id,
            source_name=source.name,
            delivery=source.delivery,
            kind=source.kind,
            endpoint_kind=source.endpoint.kind,
            status="error",
            fetched_at=utc_now(),
            probe_url=probe_urls[-1],
            error=last_error or "probe failed",
        )

    def _pending_packet(self, source: SourceSpec) -> SourcePacket:
        probe_urls = self.probe_resolver.resolve_candidates(source)
        return SourcePacket(
            source_id=source.id,
            source_name=source.name,
            delivery=source.delivery,
            kind=source.kind,
            endpoint_kind=source.endpoint.kind,
            status="pending",
            probe_url=probe_urls[0] if probe_urls else None,
            summary=f"{source.name} is wired and waiting for the harvester to collect the latest snapshot.",
        )

    @staticmethod
    def _is_due(packet: SourcePacket, source: SourceSpec) -> bool:
        if packet.status == "pending" or packet.fetched_at is None:
            return True
        now = datetime.now(timezone.utc)
        return now - packet.fetched_at >= timedelta(seconds=_ttl_seconds(source))

    def _extract_summary(self, source: SourceSpec, raw_text: str, content_type: str) -> tuple[str, list[str]]:
        stripped = raw_text.lstrip()
        if source.endpoint.kind == "telegram":
            summary, sample_items = _extract_telegram_summary(raw_text)
        elif "xml" in content_type or stripped.startswith("<?xml") or stripped.startswith("<rss") or stripped.startswith("<feed"):
            summary, sample_items = _extract_xml_summary(raw_text)
        elif "json" in content_type or stripped.startswith("{") or stripped.startswith("["):
            summary, sample_items = _extract_json_summary(raw_text)
        else:
            summary, sample_items = _extract_html_summary(raw_text)

        if not summary:
            summary = source.rationale
        if not sample_items:
            sample_items = [summary]
        return summary[:320], sample_items[:3]


def _ttl_seconds(source: SourceSpec) -> int:
    if source.kind == "telegram":
        return 180
    if source.kind == "video":
        return 600
    if source.kind in {"api", "structured"}:
        return 900
    return 1_200


def _clean_text(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw)
    text = unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _extract_telegram_summary(raw_text: str) -> tuple[str, list[str]]:
    matches = [_clean_text(match) for match in _TELEGRAM_MESSAGE_RE.findall(raw_text)]
    messages = [match for match in matches if match]
    if not messages:
        return _extract_html_summary(raw_text)
    summary = messages[0]
    return summary, messages[:3]


def _extract_xml_summary(raw_text: str) -> tuple[str, list[str]]:
    try:
        root = ET.fromstring(raw_text)
    except ET.ParseError:
        titles = [_clean_text(match) for match in re.findall(r"<title[^>]*>(.*?)</title>", raw_text, re.DOTALL | re.IGNORECASE)]
        titles = [title for title in titles if title]
        return (titles[0], titles[:3]) if titles else ("", [])

    titles: list[str] = []
    for element in root.iter():
        if element.tag.split("}")[-1] != "title":
            continue
        title = _clean_text(element.text or "")
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= 4:
            break

    if not titles:
        return "", []
    return titles[0], titles[:3]


def _extract_json_summary(raw_text: str) -> tuple[str, list[str]]:
    payload = json.loads(raw_text)
    items: list[str] = []
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:3]:
            if isinstance(value, (dict, list)):
                items.append(f"{key}: {type(value).__name__}")
            else:
                items.append(f"{key}: {value}")
    elif isinstance(payload, list):
        for value in payload[:3]:
            items.append(str(value))

    if not items:
        return "", []
    return items[0], items


def _extract_html_summary(raw_text: str) -> tuple[str, list[str]]:
    title_match = _HTML_TITLE_RE.search(raw_text)
    meta_match = _HTML_META_RE.search(raw_text)
    h1_match = _HTML_H1_RE.search(raw_text)

    items = [
        _clean_text(match)
        for match in (
            title_match.group(1) if title_match else "",
            h1_match.group(1) if h1_match else "",
            meta_match.group(1) if meta_match else "",
        )
        if match
    ]
    items = [item for item in items if item]
    if not items:
        return "", []
    return items[0], items[:3]
