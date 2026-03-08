from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field

from trenches_env.agents import AGENT_IDS
from trenches_env.historical_replay import HistoricalReplayDefinition
from trenches_env.models import EventSeverity, HistoricalEvent, HistoricalEventImpact
from trenches_env.source_catalog import get_sources_for_agent
from trenches_env.source_catalog import SourceSpec, UrlEndpoint

_SITE_PATTERN = re.compile(r"site:([A-Za-z0-9.-]+)")
_NON_WORD_PATTERN = re.compile(r"[^a-z0-9]+")

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "shipping": ("shipping", "tanker", "hormuz", "maritime", "vessel", "escort", "transit", "port"),
    "commodities": ("gold", "silver", "copper", "lithium", "lng", "commodity", "mineral", "rare earth"),
    "border": ("border", "rocket", "missile", "drone", "swarm", "launch", "incursion", "front"),
    "corridor": ("corridor", "logistics", "syria", "bekaa", "interdiction", "proxy", "sustainment"),
    "domestic": ("sanction", "protest", "unrest", "inflation", "reserve", "political", "domestic"),
    "cyber": ("cyber", "outage", "malware", "network", "infrastructure", "blackout"),
    "market": ("market", "investor", "bond", "stocks", "premium", "insurance", "trade"),
    "humanitarian": ("humanitarian", "aid", "displacement", "civilian", "refugee", "shelter"),
    "diplomacy": ("ceasefire", "talks", "summit", "mediat", "backchannel", "framework", "deconfliction"),
}

NEGATIVE_MARKERS = (
    "attack",
    "strike",
    "threat",
    "harassment",
    "swarm",
    "sanction",
    "disruption",
    "outage",
    "volley",
    "incursion",
    "retaliat",
    "unrest",
)

POSITIVE_MARKERS = (
    "ceasefire",
    "stabil",
    "assurance",
    "resupply",
    "escort",
    "framework",
    "deconfliction",
    "reopen",
    "relief",
    "backchannel",
    "reprieve",
)

AGENT_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "us": ("Hormuz", "shipping", "CENTCOM", "sanctions", "Gulf", "Iran", "Israel", "Hezbollah"),
    "israel": ("Israel", "IDF", "Hezbollah", "Lebanon", "Iran", "Syria", "rocket", "drone"),
    "iran": ("Iran", "IRGC", "proxy", "Hormuz", "sanctions", "Israel", "United States"),
    "hezbollah": ("Hezbollah", "Lebanon", "Israel", "rocket", "drone", "border", "south Lebanon"),
    "gulf": ("Gulf", "Hormuz", "shipping", "energy", "LNG", "oil", "Saudi", "UAE", "Qatar"),
    "oversight": ("regional escalation", "cyber", "shipping", "humanitarian", "ceasefire", "attribution"),
}

PREFERRED_SOURCE_IDS: dict[str, tuple[str, ...]] = {
    "us": ("us-reuters-us", "us-usni-news", "us-politico"),
    "israel": ("israel-times-of-israel", "israel-haaretz"),
    "iran": ("iran-iran-international", "iran-fars-news", "iran-al-arabiya"),
    "hezbollah": (),
    "gulf": ("gulf-reuters-business", "gulf-arab-news", "gulf-the-national-gcc"),
    "oversight": (),
}

FALLBACK_COLLECTION_PROFILES: dict[str, tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...]] = {
    "hezbollah": (
        ("hezbollah-reuters", "Reuters Middle East", ("reuters.com",), ("hezbollah", "Lebanon", "Israel", "rocket", "drone")),
        ("hezbollah-aljazeera", "Al Jazeera", ("aljazeera.com",), ("hezbollah", "Lebanon", "Israel", "border")),
    ),
    "oversight": (
        ("oversight-reuters", "Reuters World", ("reuters.com",), ("regional escalation", "shipping", "cyber", "humanitarian")),
        ("oversight-un-news", "UN News", ("news.un.org",), ("regional escalation", "humanitarian", "ceasefire", "displacement")),
    ),
}

TOPIC_IMPACT_FACTORS: dict[str, tuple[float, float, float]] = {
    "shipping": (1.0, 1.2, 1.5),
    "commodities": (0.5, 1.2, 1.0),
    "border": (1.25, 0.4, 0.1),
    "corridor": (1.0, 0.3, 0.2),
    "domestic": (0.7, 0.5, 0.2),
    "cyber": (0.8, 0.9, 0.4),
    "market": (0.4, 1.1, 0.5),
    "humanitarian": (0.6, 0.2, 0.1),
    "diplomacy": (-0.9, -0.8, -0.6),
}

AGENT_TOPIC_METRICS: dict[str, dict[str, tuple[str, ...]]] = {
    "us": {
        "shipping": ("shipping_security", "regional_access"),
        "diplomacy": ("regional_access", "shipping_security"),
        "domestic": ("domestic_support",),
        "market": ("domestic_support", "force_posture"),
    },
    "israel": {
        "border": ("homeland_security", "northern_deterrence", "reserve_endurance"),
        "corridor": ("northern_deterrence",),
        "diplomacy": ("us_resupply_confidence", "reserve_endurance"),
        "domestic": ("reserve_endurance", "us_resupply_confidence"),
    },
    "iran": {
        "shipping": ("hormuz_leverage",),
        "corridor": ("proxy_corridor", "deterrence_credibility"),
        "domestic": ("regime_stability",),
        "diplomacy": ("deterrence_credibility",),
    },
    "hezbollah": {
        "border": ("resistance_credibility", "launch_survivability"),
        "corridor": ("logistics_depth",),
        "domestic": ("political_cover",),
        "diplomacy": ("political_cover",),
    },
    "gulf": {
        "shipping": ("shipping_continuity", "investor_confidence"),
        "commodities": ("investor_confidence", "diplomatic_flexibility"),
        "cyber": ("infrastructure_security", "investor_confidence"),
        "diplomacy": ("diplomatic_flexibility", "shipping_continuity"),
        "market": ("investor_confidence",),
    },
    "oversight": {
        "cyber": ("trace_clarity",),
        "shipping": ("trace_clarity", "autonomy_balance"),
        "humanitarian": ("intervention_legitimacy",),
        "diplomacy": ("intervention_legitimacy", "autonomy_balance"),
    },
}

SEVERITY_BASE: dict[EventSeverity, float] = {
    "low": 1.5,
    "medium": 3.5,
    "high": 6.0,
    "critical": 8.5,
}

WINDOW_PRESETS: dict[str, tuple[date, date]] = {
    "2025": (date(2025, 1, 1), date(2026, 1, 1)),
    "2026": (date(2026, 1, 1), date(2027, 1, 1)),
}


class HistoricalCollectionWindow(BaseModel):
    window_id: str
    start_date: date
    end_date: date


class HistoricalSourceProfile(BaseModel):
    agent_id: str
    source_id: str
    source_name: str
    rationale: str
    domains: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    query_terms: list[str] = Field(default_factory=list)
    priority: int = 0


class CollectedHistoricalArticle(BaseModel):
    article_id: str
    agent_id: str
    source_id: str
    source_name: str
    title: str
    url: str
    domain: str
    timestamp: datetime
    query: str
    window_id: str
    tags: list[str] = Field(default_factory=list)
    language: str | None = None
    source_country: str | None = None


def resolve_window(window_id: str, *, now: datetime | None = None) -> HistoricalCollectionWindow:
    if window_id not in WINDOW_PRESETS:
        known = ", ".join(sorted(WINDOW_PRESETS))
        raise ValueError(f"Unknown collection window {window_id}. Known windows: {known}")
    start_date, end_date = WINDOW_PRESETS[window_id]
    current = (now or datetime.now(UTC)).date()
    if end_date > current + timedelta(days=1):
        end_date = current + timedelta(days=1)
    return HistoricalCollectionWindow(window_id=window_id, start_date=start_date, end_date=end_date)


def iter_month_windows(window: HistoricalCollectionWindow) -> list[HistoricalCollectionWindow]:
    current = window.start_date
    windows: list[HistoricalCollectionWindow] = []
    while current < window.end_date:
        next_month = date(current.year + (1 if current.month == 12 else 0), 1 if current.month == 12 else current.month + 1, 1)
        windows.append(
            HistoricalCollectionWindow(
                window_id=f"{window.window_id}-{current.strftime('%Y-%m')}",
                start_date=current,
                end_date=min(next_month, window.end_date),
            )
        )
        current = next_month
    return windows


def _priority_for_source(source: SourceSpec) -> int:
    score = 0
    tags = set(source.tags)
    if "official" in tags:
        score += 3
    if "wire" in tags:
        score += 2
    if source.allowlistStatus == "allowed":
        score += 1
    return score


def _extract_domains_from_source(source: SourceSpec) -> list[str]:
    endpoint = source.endpoint
    if not isinstance(endpoint, UrlEndpoint):
        return []
    parsed = urlparse(endpoint.url)
    domains: set[str] = set()
    hostname = parsed.hostname or ""
    if hostname and hostname != "news.google.com":
        domains.add(hostname.removeprefix("www."))
    query_values = parse_qs(parsed.query).get("q", [])
    for query_value in query_values:
        for match in _SITE_PATTERN.findall(query_value):
            domains.add(match.removeprefix("www."))
    return sorted(domains)


def build_source_profiles_for_agent(agent_id: str) -> list[HistoricalSourceProfile]:
    profiles: list[HistoricalSourceProfile] = []
    preferred_order = {source_id: index for index, source_id in enumerate(PREFERRED_SOURCE_IDS.get(agent_id, ()))}
    for source in get_sources_for_agent(agent_id, delivery="training_core"):
        if source.kind not in {"rss", "api", "scrape"}:
            continue
        domains = _extract_domains_from_source(source)
        if not domains:
            continue
        profiles.append(
            HistoricalSourceProfile(
                agent_id=agent_id,
                source_id=source.id,
                source_name=source.name,
                rationale=source.rationale,
                domains=domains,
                tags=list(source.tags),
                query_terms=list(AGENT_QUERY_TERMS.get(agent_id, ())),
                priority=_priority_for_source(source) + (20 - preferred_order[source.id] if source.id in preferred_order else 0),
            )
        )
    for source_id, source_name, domains, query_terms in FALLBACK_COLLECTION_PROFILES.get(agent_id, ()):
        profiles.append(
            HistoricalSourceProfile(
                agent_id=agent_id,
                source_id=source_id,
                source_name=source_name,
                rationale="Fallback historical collection profile for GDELT-backed replay building.",
                domains=list(domains),
                tags=[agent_id, "historical-fallback"],
                query_terms=list(query_terms),
                priority=5,
            )
        )
    profiles.sort(key=lambda item: (-item.priority, item.source_name))
    return profiles


def build_gdelt_query(profile: HistoricalSourceProfile) -> str:
    domains = profile.domains[:4]
    if len(domains) == 1:
        domain_clause = f"domainis:{domains[0]}"
    else:
        domain_clause = " OR ".join(f"domainis:{domain}" for domain in domains)
        if domain_clause:
            domain_clause = f"({domain_clause})"
    valid_terms = []
    for term in profile.query_terms[:8]:
        normalized = re.sub(r"[^A-Za-z0-9]+", "", term)
        if len(normalized) < 5:
            continue
        valid_terms.append(term)
    terms = " OR ".join(json.dumps(term) for term in valid_terms)
    if terms:
        terms = f"({terms})"
    if domain_clause and terms:
        return f"{domain_clause} AND {terms}"
    if terms:
        return terms
    return domain_clause


def parse_gdelt_datetime(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z") and "T" in value:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    if len(value) == 14 and value.isdigit():
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    if len(value) == 15 and value.endswith("Z") and value[:-1].isdigit():
        return datetime.strptime(value, "%Y%m%d%H%M%SZ").replace(tzinfo=UTC)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def build_article_id(url: str, timestamp: datetime) -> str:
    digest = hashlib.sha1(f"{url}|{timestamp.isoformat()}".encode("utf-8")).hexdigest()
    return digest[:16]


def dedupe_articles(articles: list[CollectedHistoricalArticle]) -> list[CollectedHistoricalArticle]:
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[CollectedHistoricalArticle] = []
    for article in sorted(articles, key=lambda item: item.timestamp):
        normalized_url = article.url.rstrip("/")
        normalized_title = _NON_WORD_PATTERN.sub(" ", article.title.lower()).strip()
        title_key = f"{article.timestamp.date().isoformat()}::{normalized_title}"
        if normalized_url in seen_urls or title_key in seen_titles:
            continue
        seen_urls.add(normalized_url)
        seen_titles.add(title_key)
        deduped.append(article)
    return deduped


def infer_topic(title: str) -> str:
    lowered = title.lower()
    scored: list[tuple[int, str]] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score:
            scored.append((score, topic))
    if not scored:
        return "diplomacy"
    scored.sort(reverse=True)
    return scored[0][1]


def infer_severity(title: str, topic: str) -> EventSeverity:
    lowered = title.lower()
    if any(marker in lowered for marker in ("critical", "massive", "major", "swarm", "ground operation")):
        return "critical"
    if any(marker in lowered for marker in ("strike", "attack", "retaliat", "incursion", "disruption", "outage")):
        return "high"
    if topic in {"shipping", "cyber", "commodities", "domestic", "corridor"}:
        return "medium"
    return "low"


def infer_polarity(title: str, topic: str) -> int:
    lowered = title.lower()
    if any(marker in lowered for marker in POSITIVE_MARKERS):
        return 1
    if any(marker in lowered for marker in NEGATIVE_MARKERS):
        return -1
    if topic == "diplomacy":
        return 1
    if topic in {"shipping", "border", "corridor", "cyber", "humanitarian", "commodities"}:
        return -1
    return 0


def infer_actors_and_targets(title: str, agent_id: str) -> tuple[list[str], list[str]]:
    lowered = title.lower()
    actors: list[str] = []
    targets: list[str] = []
    for candidate in AGENT_IDS:
        if candidate == "us":
            tokens = ("united states", "u.s.", "washington", "centcom", "pentagon", "us ")
        elif candidate == "israel":
            tokens = ("israel", "idf")
        elif candidate == "iran":
            tokens = ("iran", "irgc", "tehran")
        elif candidate == "hezbollah":
            tokens = ("hezbollah",)
        elif candidate == "gulf":
            tokens = ("gulf", "saudi", "uae", "qatar", "oman", "bahrain")
        else:
            tokens = ("oversight", "monitor", "trace")
        if any(token in lowered for token in tokens):
            actors.append(candidate)
    if not actors:
        actors = [agent_id]
    topic = infer_topic(title)
    if topic == "shipping":
        targets = ["shipping_lanes"]
    elif topic == "border":
        targets = ["northern_front" if agent_id in {"israel", "hezbollah"} else "border_zone"]
    elif topic == "corridor":
        targets = ["proxy_corridor"]
    elif topic == "cyber":
        targets = ["energy_networks"]
    elif topic == "commodities":
        targets = ["commodity_markets"]
    else:
        targets = [agent_id]
    return sorted(set(actors)), targets


def infer_impact(agent_id: str, topic: str, severity: EventSeverity, polarity: int) -> HistoricalEventImpact:
    base = SEVERITY_BASE[severity]
    tension_factor, market_factor, oil_factor = TOPIC_IMPACT_FACTORS.get(topic, (0.5, 0.3, 0.2))
    sign = 1 if polarity >= 0 else -1
    if polarity == 0:
        sign = 1 if topic not in {"diplomacy"} else -1

    tension_delta = round(base * tension_factor * sign, 2)
    market_delta = round(base * market_factor * sign, 2)
    oil_delta = round(base * oil_factor * sign, 2)

    metric_scale = max(1.5, base * 0.7)
    actor_metric_deltas: dict[str, dict[str, float]] = {}
    for target_agent, metric_map in AGENT_TOPIC_METRICS.items():
        metrics = metric_map.get(topic, ())
        if not metrics:
            continue
        direction = sign
        if target_agent == agent_id and topic == "diplomacy":
            direction = 1
        elif target_agent == agent_id and topic in {"shipping", "border", "corridor", "cyber", "humanitarian", "commodities"}:
            direction = -1 if sign > 0 else 1
        elif target_agent in {"iran", "hezbollah"} and topic in {"shipping", "border", "corridor"} and sign > 0:
            direction = 1
        elif topic == "diplomacy":
            direction = 1
        actor_metric_deltas[target_agent] = {
            metric: round(metric_scale * direction, 2) for metric in metrics
        }

    return HistoricalEventImpact(
        tension_delta=tension_delta,
        market_stress_delta=market_delta,
        oil_pressure_delta=oil_delta,
        actor_metric_deltas=actor_metric_deltas,
    )


def article_to_historical_event(article: CollectedHistoricalArticle, *, training_agent: str) -> HistoricalEvent:
    topic = infer_topic(article.title)
    severity = infer_severity(article.title, topic)
    polarity = infer_polarity(article.title, topic)
    actors, targets = infer_actors_and_targets(article.title, training_agent)
    return HistoricalEvent(
        event_id=f"{training_agent}-{article.timestamp.strftime('%Y%m%d%H%M%S')}-{article.article_id[:8]}",
        timestamp=article.timestamp,
        topic=topic,
        region=training_agent if training_agent != "oversight" else "global",
        actors=actors,
        targets=targets,
        severity=severity,
        summary=article.title,
        public_summary=article.title,
        source_type="gdelt_historical_collection",
        confirmed=True,
        tags=sorted(set([*article.tags, topic, article.domain])),
        impact=infer_impact(training_agent, topic, severity, polarity),
    )


def build_replay_definition(
    *,
    training_agent: str,
    window: HistoricalCollectionWindow,
    articles: list[CollectedHistoricalArticle],
    max_events: int = 128,
) -> HistoricalReplayDefinition:
    events = [article_to_historical_event(article, training_agent=training_agent) for article in dedupe_articles(articles)]
    events.sort(key=lambda item: item.timestamp)
    events = events[:max_events]
    return HistoricalReplayDefinition(
        replay_id=f"{training_agent}_historical_{window.window_id}",
        name=f"{training_agent.upper()} historical replay {window.start_date.isoformat()} to {window.end_date.isoformat()}",
        description=(
            "Historically collected replay built from allowlisted source domains via the GDELT DOC API. "
            "Titles and impacts are heuristic and should be curator-reviewed before production post-training."
        ),
        training_agent=training_agent,
        events=events,
    )


def dump_raw_articles(path: Path, articles: list[CollectedHistoricalArticle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for article in sorted(articles, key=lambda item: item.timestamp):
            handle.write(article.model_dump_json())
            handle.write("\n")


def dump_replay_definition(path: Path, replay: HistoricalReplayDefinition) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")


def format_gdelt_datetime(day: date, *, end_of_day: bool = False) -> str:
    dt = datetime.combine(day, time.max if end_of_day else time.min, tzinfo=UTC)
    return dt.strftime("%Y%m%d%H%M%S")
