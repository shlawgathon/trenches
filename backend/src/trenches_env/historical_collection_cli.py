from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from trenches_env.agents import AGENT_IDS
from trenches_env.historical_collection import (
    CollectedHistoricalArticle,
    HistoricalCollectionWindow,
    build_gdelt_query,
    build_replay_definition,
    build_source_profiles_for_agent,
    build_article_id,
    dump_raw_articles,
    dump_replay_definition,
    format_gdelt_datetime,
    iter_month_windows,
    parse_gdelt_datetime,
    resolve_window,
)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_INTERVAL_SECONDS = 5.2
GDELT_MAX_ATTEMPTS = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect historical replay candidates into Trenches replay JSON format.")
    parser.add_argument("--training-agent", choices=[*AGENT_IDS, "all"], default="us")
    parser.add_argument("--window", action="append", choices=["2025", "2026"], default=["2025"])
    parser.add_argument(
        "--output-dir",
        default="backend/src/trenches_env/historical_replays",
        help="Directory for replay JSON files.",
    )
    parser.add_argument(
        "--raw-dir",
        default="backend/tmp-historical-raw",
        help="Directory for raw collected article JSONL files.",
    )
    parser.add_argument("--max-records-per-query", type=int, default=50)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--max-sources-per-agent", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def _decode_gdelt_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        message = response.text.strip()
        raise RuntimeError(f"GDELT returned a non-JSON response: {message[:280]}") from exc


def _request_gdelt(
    client: httpx.Client,
    *,
    params: dict[str, Any],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, GDELT_MAX_ATTEMPTS + 1):
        response = client.get(GDELT_DOC_API, params=params)
        if response.status_code == 429:
            last_error = RuntimeError("GDELT rate limited the request.")
            time.sleep(GDELT_MIN_INTERVAL_SECONDS * (attempt + 1))
            continue
        response.raise_for_status()
        try:
            payload = _decode_gdelt_payload(response)
            time.sleep(GDELT_MIN_INTERVAL_SECONDS)
            return payload
        except RuntimeError as exc:
            last_error = exc
            message = str(exc)
            if "Please limit requests to one every 5 seconds" not in message:
                raise
            time.sleep(GDELT_MIN_INTERVAL_SECONDS * attempt)
    raise RuntimeError(f"GDELT request failed after {GDELT_MAX_ATTEMPTS} attempts: {last_error}")


def _fetch_gdelt_articles(
    client: httpx.Client,
    *,
    agent_id: str,
    window: HistoricalCollectionWindow,
    max_records_per_query: int,
    max_sources_per_agent: int,
) -> list[CollectedHistoricalArticle]:
    articles: list[CollectedHistoricalArticle] = []
    for profile in build_source_profiles_for_agent(agent_id)[:max_sources_per_agent]:
        query = build_gdelt_query(profile)
        if not query:
            continue
        for month_window in iter_month_windows(window):
            params = {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": max_records_per_query,
                "startdatetime": format_gdelt_datetime(month_window.start_date),
                "enddatetime": format_gdelt_datetime(month_window.end_date - timedelta(days=1), end_of_day=True),
                "sort": "datedesc",
            }
            try:
                payload = _request_gdelt(client, params=params)
            except Exception as exc:
                print(
                    f"[historical-collector] skip agent={agent_id} source={profile.source_id} "
                    f"window={month_window.window_id}: {exc}",
                    file=sys.stderr,
                )
                continue
            for item in payload.get("articles", []):
                url = str(item.get("url") or "").strip()
                title = str(item.get("title") or "").strip()
                seendate = str(item.get("seendate") or "").strip()
                domain = str(item.get("domain") or "").strip()
                if not url or not title or not seendate:
                    continue
                timestamp = parse_gdelt_datetime(seendate)
                if timestamp.date() < window.start_date or timestamp.date() >= window.end_date:
                    continue
                articles.append(
                    CollectedHistoricalArticle(
                        article_id=build_article_id(url, timestamp),
                        agent_id=agent_id,
                        source_id=profile.source_id,
                        source_name=profile.source_name,
                        title=title,
                        url=url,
                        domain=domain or url.split("/")[2],
                        timestamp=timestamp,
                        query=query,
                        window_id=window.window_id,
                        tags=sorted(set([*profile.tags, *profile.query_terms[:3]])),
                        language=item.get("language"),
                        source_country=item.get("sourcecountry"),
                    )
                )
    return articles


def _collect_for_agent(
    client: httpx.Client,
    *,
    agent_id: str,
    windows: list[str],
    output_dir: Path,
    raw_dir: Path,
    max_records_per_query: int,
    max_events: int,
    max_sources_per_agent: int,
) -> list[Path]:
    written: list[Path] = []
    for window_id in windows:
        resolved_window = resolve_window(window_id, now=datetime.now(UTC))
        articles = _fetch_gdelt_articles(
            client,
            agent_id=agent_id,
            window=resolved_window,
            max_records_per_query=max_records_per_query,
            max_sources_per_agent=max_sources_per_agent,
        )
        replay = build_replay_definition(
            training_agent=agent_id,
            window=resolved_window,
            articles=articles,
            max_events=max_events,
        )
        replay_path = output_dir / f"{replay.replay_id}.json"
        raw_path = raw_dir / f"{replay.replay_id}.articles.jsonl"
        dump_replay_definition(replay_path, replay)
        dump_raw_articles(raw_path, articles)
        written.append(replay_path)
    return written


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    agent_ids = list(AGENT_IDS) if args.training_agent == "all" else [args.training_agent]

    with httpx.Client(timeout=args.timeout_seconds, headers={"User-Agent": "trenches-historical-collector/0.1"}) as client:
        written: list[Path] = []
        for agent_id in agent_ids:
            written.extend(
                _collect_for_agent(
                    client,
                    agent_id=agent_id,
                    windows=args.window,
                    output_dir=output_dir,
                    raw_dir=raw_dir,
                    max_records_per_query=args.max_records_per_query,
                    max_events=args.max_events,
                    max_sources_per_agent=args.max_sources_per_agent,
                )
            )

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
