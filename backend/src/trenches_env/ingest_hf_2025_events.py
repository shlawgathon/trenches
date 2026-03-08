"""
Ingest Reubencf/2025_events from HuggingFace and convert to per-entity
historical replay JSONs compatible with the training CLI.

Usage:
    python -m trenches_env.ingest_hf_2025_events \
        --output-dir backend/src/trenches_env/historical_replays

Produces one replay JSON per entity (us, israel, iran, hezbollah, gulf, oversight).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ENTITIES = ["us", "israel", "iran", "hezbollah", "gulf", "oversight"]

# ---------------------------------------------------------------------------
# Entity-detection keyword maps
# ---------------------------------------------------------------------------
ENTITY_KEYWORDS: dict[str, list[str]] = {
    "us": [
        "united states", "u.s.", "us ", "american", "pentagon", "white house",
        "trump", "biden", "congress", "cia", "state department", "centcom",
        "nato", "washington", "new york", "california", "texas", "florida",
        "fbi", "homeland security", "secret service", "marine", "us military",
        "us navy", "us army", "air force", "coast guard",
    ],
    "israel": [
        "israel", "israeli", "idf", "netanyahu", "knesset", "mossad",
        "tel aviv", "jerusalem", "west bank", "gaza", "hamas", "shin bet",
        "iron dome", "kibbutz", "settler", "zionist",
    ],
    "iran": [
        "iran", "iranian", "irgc", "tehran", "khamenei", "rouhani",
        "persian gulf", "nuclear", "enrichment", "centrifuge", "natanz",
        "fordow", "arak", "jcpoa", "basij",
    ],
    "hezbollah": [
        "hezbollah", "hezb", "nasrallah", "lebanon", "lebanese", "beirut",
        "litani", "south lebanon", "bekaa",
    ],
    "gulf": [
        "saudi", "uae", "emirates", "qatar", "bahrain", "oman", "kuwait",
        "opec", "aramco", "riyadh", "doha", "abu dhabi", "dubai",
        "oil price", "oil market", "brent crude", "wti crude", "hormuz",
        "strait of hormuz", "tanker", "shipping lane", "red sea",
        "houthi", "yemen", "aden",
    ],
    "oversight": [
        "united nations", "un ", "iaea", "icj", "icc ", "amnesty",
        "human rights", "war crime", "investigation", "tribunal",
        "ceasefire", "peacekeep", "humanitarian", "international law",
    ],
}

# ---------------------------------------------------------------------------
# Section → topic mapping
# ---------------------------------------------------------------------------
SECTION_TOPIC_MAP: dict[str, str] = {
    "Armed conflicts and attacks": "conflict",
    "Attacks and armed conflicts": "conflict",
    "International relations": "diplomacy",
    "Politics and elections": "domestic",
    "Business and economy": "economy",
    "Law and crime": "security",
    "Law and Crime": "security",
    "Disasters and accidents": "disaster",
    "Disaster and accidents": "disaster",
    "Disasters and incidents": "disaster",
    "Health and environment": "health",
    "Science and technology": "technology",
    "Science and Technology": "technology",
    "Arts and culture": "culture",
    "Arts and Culture": "culture",
    "Sports": "sports",
}

# ---------------------------------------------------------------------------
# Severity heuristics based on content keywords
# ---------------------------------------------------------------------------
HIGH_SEVERITY_KW = [
    "kill", "dead", "death", "massacre", "bomb", "airstrike", "air strike",
    "missile", "rocket", "explosion", "war crime", "genocide", "invasion",
    "nuclear", "crisis", "emergency", "collapse",
]
CRITICAL_SEVERITY_KW = [
    "mass casualt", "hundreds killed", "thousands", "nuclear weapon",
    "declaration of war", "martial law", "coup",
]


# ---------------------------------------------------------------------------
# Region heuristics
# ---------------------------------------------------------------------------
REGION_KEYWORDS: dict[str, list[str]] = {
    "gulf": ["gulf", "hormuz", "tanker", "shipping", "saudi", "uae", "qatar",
             "bahrain", "oman", "kuwait", "opec", "yemen", "houthi", "red sea", "aden"],
    "levant": ["lebanon", "hezbollah", "beirut", "litani", "syria", "syrian", "damascus"],
    "iran": ["iran", "iranian", "tehran", "irgc"],
    "israel": ["israel", "israeli", "gaza", "west bank", "jerusalem", "tel aviv"],
    "us": ["united states", "u.s.", "washington", "pentagon", "congress"],
    "iraq": ["iraq", "iraqi", "baghdad", "kurdistan", "mosul"],
    "africa": ["africa", "sudan", "somalia", "libya", "egypt", "nigeria"],
    "europe": ["europe", "ukraine", "russia", "nato", "eu ", "britain", "france", "germany"],
    "asia": ["china", "india", "japan", "korea", "taiwan", "pacific", "asean"],
    "global": [],
}


def detect_entities(text: str) -> list[str]:
    """Return list of entity IDs mentioned in text."""
    text_lower = text.lower()
    matched = []
    for entity, keywords in ENTITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                matched.append(entity)
                break
    return matched or ["oversight"]  # default to oversight if no entity detected


def detect_region(text: str) -> str:
    text_lower = text.lower()
    for region, keywords in REGION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return region
    return "global"


def detect_severity(text: str) -> str:
    text_lower = text.lower()
    for kw in CRITICAL_SEVERITY_KW:
        if kw in text_lower:
            return "critical"
    for kw in HIGH_SEVERITY_KW:
        if kw in text_lower:
            return "high"
    return "medium"


def detect_topic(section: str, content: str) -> str:
    """Map section to topic, with content-based refinement."""
    topic = SECTION_TOPIC_MAP.get(section, "other")
    content_lower = content.lower()

    # Refine based on content
    if any(kw in content_lower for kw in ["sanction", "tariff", "trade war"]):
        topic = "sanctions"
    elif any(kw in content_lower for kw in ["oil", "energy", "opec", "crude", "gas price"]):
        topic = "energy"
    elif any(kw in content_lower for kw in ["shipping", "tanker", "hormuz", "maritime"]):
        topic = "shipping"
    elif any(kw in content_lower for kw in ["nuclear", "enrichment", "iaea", "centrifuge"]):
        topic = "nuclear"
    elif any(kw in content_lower for kw in ["cyber", "hack", "malware"]):
        topic = "cyber"
    elif any(kw in content_lower for kw in ["ceasefire", "peace", "negotiat", "diplomac"]):
        topic = "diplomacy"
    elif any(kw in content_lower for kw in ["drone", "missile", "airstrike", "strike"]):
        topic = "military"
    elif any(kw in content_lower for kw in ["border", "frontier", "incursion"]):
        topic = "border"

    return topic


def detect_targets(text: str) -> list[str]:
    """Extract target keywords."""
    text_lower = text.lower()
    targets = []
    target_map = {
        "shipping_lanes": ["shipping", "tanker", "maritime", "vessel"],
        "energy_networks": ["oil", "pipeline", "refinery", "energy", "power grid"],
        "military_base": ["military base", "airbase", "naval base", "base"],
        "civilians": ["civilian", "resident", "hospital", "school", "refugee"],
        "infrastructure": ["infrastructure", "bridge", "road", "airport", "port"],
        "government": ["government", "parliament", "ministry", "embassy"],
        "proxy_corridor": ["proxy", "militia", "corridor", "supply line"],
        "northern_front": ["northern", "border", "frontier"],
        "nuclear_facility": ["nuclear", "enrichment", "reactor", "centrifuge"],
    }
    for target, keywords in target_map.items():
        for kw in keywords:
            if kw in text_lower:
                targets.append(target)
                break
    return targets[:3] if targets else ["general"]


def make_event_id(month: str, day: int, content: str) -> str:
    """Generate a deterministic event ID."""
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"evt-2025-{month[:3].lower()}-{day:02d}-{h}"


MONTH_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def make_timestamp(month: str, day: int) -> str:
    m = MONTH_NUM.get(month, 1)
    # Clamp day to valid range
    import calendar
    max_day = calendar.monthrange(2025, m)[1]
    d = min(day, max_day)
    dt = datetime(2025, m, d, 12, 0, 0, tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def compute_impact(severity: str, topic: str, entities: list[str]) -> dict:
    """Heuristic impact deltas based on severity and topic."""
    severity_mult = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.5}
    mult = severity_mult.get(severity, 1.0)

    # Base deltas by topic
    topic_deltas = {
        "conflict":   {"tension": 5.0, "market": 2.0, "oil": 1.0},
        "military":   {"tension": 6.0, "market": 2.5, "oil": 1.5},
        "shipping":   {"tension": 3.0, "market": 4.0, "oil": 6.0},
        "energy":     {"tension": 2.0, "market": 5.0, "oil": 7.0},
        "nuclear":    {"tension": 7.0, "market": 3.0, "oil": 2.0},
        "diplomacy":  {"tension": -3.0, "market": -2.0, "oil": -1.0},
        "sanctions":  {"tension": 3.0, "market": 3.0, "oil": 2.0},
        "border":     {"tension": 5.0, "market": 1.0, "oil": 0.5},
        "cyber":      {"tension": 3.0, "market": 3.5, "oil": 1.5},
        "security":   {"tension": 2.0, "market": 1.0, "oil": 0.5},
        "domestic":   {"tension": 1.0, "market": 1.5, "oil": 0.5},
        "disaster":   {"tension": 1.0, "market": 2.0, "oil": 1.0},
        "economy":    {"tension": 0.5, "market": 4.0, "oil": 2.0},
        "health":     {"tension": 0.5, "market": 1.0, "oil": 0.0},
        "technology": {"tension": 0.0, "market": 1.0, "oil": 0.0},
        "culture":    {"tension": 0.0, "market": 0.0, "oil": 0.0},
        "sports":     {"tension": 0.0, "market": 0.0, "oil": 0.0},
        "other":      {"tension": 1.0, "market": 1.0, "oil": 0.5},
    }

    base = topic_deltas.get(topic, topic_deltas["other"])
    t = round(base["tension"] * mult, 1)
    m = round(base["market"] * mult, 1)
    o = round(base["oil"] * mult, 1)

    return {
        "tension_delta": t,
        "market_stress_delta": m,
        "oil_pressure_delta": o,
        "actor_metric_deltas": {},
    }


def truncate_content(content: str, max_chars: int = 300) -> str:
    """Get a clean summary from the content field."""
    # Take the first meaningful paragraph
    lines = content.strip().split("\n")
    # Skip title lines (often short headers)
    summary_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip very short header-like lines
        if len(line) < 30 and not line.endswith("."):
            continue
        summary_lines.append(line)
        if len(" ".join(summary_lines)) >= max_chars:
            break
    summary = " ".join(summary_lines)
    # Remove (Source) citations
    summary = re.sub(r"\s*\([^)]*(?:Reuters|AP|BBC|AFP|CNN|Al Jazeera)[^)]*\)\s*", " ", summary)
    return summary[:max_chars].strip()


def build_replays(output_dir: Path) -> dict[str, int]:
    """Main conversion logic."""
    from datasets import load_dataset

    print("Loading Reubencf/2025_events from HuggingFace...")
    ds = load_dataset("Reubencf/2025_events", split="train")
    print(f"Loaded {len(ds)} rows")

    # Collect events per entity
    entity_events: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    total_assigned = 0

    for row in ds:
        content = row["content"]
        section = row["section"]
        month = row["month"]
        day = row["day"]

        text = f"{content} {section}"
        entities = detect_entities(text)

        if not entities:
            skipped += 1
            continue

        event_id = make_event_id(month, day, content)
        timestamp = make_timestamp(month, day)
        topic = detect_topic(section, content)
        region = detect_region(text)
        severity = detect_severity(content)
        targets = detect_targets(content)
        summary = truncate_content(content)
        impact = compute_impact(severity, topic, entities)

        # Build the event
        event = {
            "event_id": event_id,
            "timestamp": timestamp,
            "topic": topic,
            "region": region,
            "actors": entities,
            "targets": targets,
            "severity": severity,
            "summary": summary,
            "public_summary": summary[:200],
            "source_type": "hf_2025_events",
            "confirmed": True,
            "tags": [topic, section.lower().replace(" ", "_"), region],
            "impact": impact,
        }

        # Assign to each relevant entity
        for entity in entities:
            entity_events[entity].append(event)
            total_assigned += 1

    print(f"Matched events per entity:")
    for entity in ENTITIES:
        evts = entity_events.get(entity, [])
        print(f"  {entity}: {len(evts)}")

    # Sort each entity's events by timestamp and deduplicate by event_id
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for entity in ENTITIES:
        events = entity_events.get(entity, [])
        if not events:
            print(f"  WARNING: No events for {entity}, skipping")
            continue

        # Deduplicate by event_id
        seen = set()
        deduped = []
        for e in events:
            if e["event_id"] not in seen:
                seen.add(e["event_id"])
                deduped.append(e)

        # Sort by timestamp
        deduped.sort(key=lambda e: e["timestamp"])

        replay = {
            "replay_id": f"{entity}_2025_events",
            "name": f"{entity.upper()} Real Events 2025",
            "description": (
                f"Real historical events from 2025 relevant to the {entity} entity. "
                f"Source: Reubencf/2025_events HuggingFace dataset ({len(deduped)} events). "
                f"Impact values are heuristic — curator review recommended before production training."
            ),
            "training_agent": entity,
            "events": deduped,
        }

        out_path = output_dir / f"{entity}_2025_events.json"
        with open(out_path, "w") as f:
            json.dump(replay, f, indent=2)

        stats[entity] = len(deduped)
        print(f"  Wrote {out_path.name}: {len(deduped)} events")

    print(f"\nTotal: {total_assigned} entity-event assignments, {skipped} unmatched rows")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Ingest Reubencf/2025_events into replay JSONs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/trenches_env/historical_replays"),
        help="Directory to write per-entity replay JSONs",
    )
    args = parser.parse_args()
    build_replays(args.output_dir)


if __name__ == "__main__":
    main()
