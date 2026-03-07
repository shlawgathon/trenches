from __future__ import annotations

from dataclasses import dataclass, field

from trenches_env.models import ExternalSignal

DEFAULT_SCENARIO_ID = "baseline_alert"


@dataclass(frozen=True)
class ScenarioSignal:
    source: str
    headline: str
    region: str | None = None
    tags: tuple[str, ...] = ()
    severity: float = 0.5


@dataclass(frozen=True)
class ScenarioMetricShift:
    agent_id: str
    metric: str
    delta: float


@dataclass(frozen=True)
class ScenarioAssetImpact:
    owner: str
    intensity: float
    reason: str
    section_bias: tuple[str, ...]
    max_assets: int = 1
    max_status: str | None = None
    mode: str = "damage"


@dataclass(frozen=True)
class ScenarioLatentEvent:
    topic: str
    summary: str
    source: str = "scenario"
    severity: float = 0.5
    visibility: str = "mixed"
    reliability: float = 0.65
    affected_agents: tuple[str, ...] = ()
    public_summary: str | None = None
    private_summary: str | None = None
    decay_rate: float = 0.06


@dataclass(frozen=True)
class ScenarioDefinition:
    id: str
    name: str
    description: str
    tags: tuple[str, ...] = ()
    benchmark_turns: int = 6
    benchmark_enabled: bool = True
    world_overrides: dict[str, float] = field(default_factory=dict)
    coalition_overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)
    hidden_intent_overrides: dict[str, str] = field(default_factory=dict)
    metric_shifts: tuple[ScenarioMetricShift, ...] = ()
    asset_impacts: tuple[ScenarioAssetImpact, ...] = ()
    latent_events: tuple[ScenarioLatentEvent, ...] = ()
    public_events: tuple[ScenarioSignal, ...] = ()
    benchmark_signal_turns: dict[int, tuple[ScenarioSignal, ...]] = field(default_factory=dict)


def _signal(
    source: str,
    headline: str,
    *,
    region: str | None = None,
    tags: tuple[str, ...] = (),
    severity: float = 0.5,
) -> ScenarioSignal:
    return ScenarioSignal(source=source, headline=headline, region=region, tags=tags, severity=severity)


SCENARIOS: dict[str, ScenarioDefinition] = {
    DEFAULT_SCENARIO_ID: ScenarioDefinition(
        id=DEFAULT_SCENARIO_ID,
        name="Baseline Alert Posture",
        description="Default elevated regional posture with no extra scripted shock beyond the standing crisis baseline.",
        tags=("baseline", "regional", "general"),
        benchmark_enabled=False,
    ),
    "shipping_crisis": ScenarioDefinition(
        id="shipping_crisis",
        name="Shipping Crisis",
        description="A Gulf shipping shock with tanker risk, insurance stress, and chokepoint pressure forcing maritime and market decisions.",
        tags=("shipping", "gulf", "oil", "maritime"),
        benchmark_turns=7,
        world_overrides={"tension_level": 64.0, "market_stress": 58.0, "oil_pressure": 78.0},
        metric_shifts=(
            ScenarioMetricShift("us", "shipping_security", -10.0),
            ScenarioMetricShift("us", "domestic_support", -4.0),
            ScenarioMetricShift("us", "force_posture", 3.0),
            ScenarioMetricShift("gulf", "shipping_continuity", -14.0),
            ScenarioMetricShift("gulf", "infrastructure_security", -8.0),
            ScenarioMetricShift("gulf", "investor_confidence", -12.0),
            ScenarioMetricShift("iran", "hormuz_leverage", 8.0),
            ScenarioMetricShift("iran", "deterrence_credibility", 4.0),
        ),
        asset_impacts=(
            ScenarioAssetImpact(
                owner="gulf",
                intensity=28.0,
                reason="scenario shipping disruption across export terminals",
                section_bias=("port", "energy", "energy-port", "chokepoint"),
                max_assets=2,
                max_status="malfunctioning",
            ),
            ScenarioAssetImpact(
                owner="us",
                intensity=16.0,
                reason="scenario escort posture strain on maritime assets",
                section_bias=("naval", "port", "maritime", "chokepoint"),
                max_assets=1,
                max_status="malfunctioning",
            ),
        ),
        latent_events=(
            ScenarioLatentEvent(
                topic="shipping",
                summary="Insurers and maritime operators privately assess that drone scouting near Hormuz is coordinated rather than isolated.",
                severity=0.66,
                visibility="mixed",
                reliability=0.71,
                affected_agents=("us", "iran", "gulf"),
                public_summary="Private maritime risk assessments suggest the Hormuz disruption pattern may be coordinated.",
                private_summary="Privately, operators believe the disruption pattern is coordinated and likely to persist without force protection.",
                decay_rate=0.04,
            ),
        ),
        public_events=(
            _signal(
                "scenario",
                "Multiple commercial tankers report evasive maneuvers near the Strait of Hormuz after repeated drone sightings.",
                region="gulf",
                tags=("shipping", "oil", "hormuz"),
                severity=0.72,
            ),
            _signal(
                "scenario",
                "Marine insurers widen war-risk premiums as Gulf export confidence deteriorates.",
                region="gulf",
                tags=("market", "shipping", "oil"),
                severity=0.58,
            ),
        ),
        benchmark_signal_turns={
            1: (
                _signal(
                    "benchmark-maritime-watch",
                    "Fresh AIS-linked reporting shows a tanker queue building outside the Strait of Hormuz after a suspected drone intercept.",
                    region="gulf",
                    tags=("shipping", "oil", "hormuz"),
                    severity=0.72,
                ),
            ),
            2: (
                _signal(
                    "benchmark-insurance-wire",
                    "War-risk underwriters temporarily suspend standard cover for vessels transiting the central Gulf lanes.",
                    region="gulf",
                    tags=("market", "shipping", "oil"),
                    severity=0.63,
                ),
            ),
            4: (
                _signal(
                    "benchmark-diplomatic-cable",
                    "Regional partners open emergency talks on escorted shipping corridors and deconfliction hotlines.",
                    region="gulf",
                    tags=("diplomacy", "shipping"),
                    severity=0.44,
                ),
            ),
        },
    ),
    "border_flareup": ScenarioDefinition(
        id="border_flareup",
        name="Northern Border Flare-Up",
        description="A sudden Israel-Hezbollah border escalation with salvo pressure, civil defense strain, and retaliatory signaling.",
        tags=("border", "levan", "salvo", "northern-front"),
        benchmark_turns=7,
        world_overrides={"tension_level": 72.0, "market_stress": 42.0, "oil_pressure": 46.0},
        metric_shifts=(
            ScenarioMetricShift("israel", "homeland_security", -12.0),
            ScenarioMetricShift("israel", "northern_deterrence", -6.0),
            ScenarioMetricShift("israel", "reserve_endurance", -4.0),
            ScenarioMetricShift("hezbollah", "resistance_credibility", 6.0),
            ScenarioMetricShift("hezbollah", "launch_survivability", -4.0),
        ),
        asset_impacts=(
            ScenarioAssetImpact(
                owner="israel",
                intensity=30.0,
                reason="scenario salvo pressure on northern defensive assets",
                section_bias=("front", "civil", "air-defense", "infrastructure"),
                max_assets=2,
                max_status="malfunctioning",
            ),
            ScenarioAssetImpact(
                owner="hezbollah",
                intensity=18.0,
                reason="scenario counterbattery and ISR exposure",
                section_bias=("launch", "front", "logistics"),
                max_assets=1,
                max_status="malfunctioning",
            ),
        ),
        latent_events=(
            ScenarioLatentEvent(
                topic="border",
                summary="Hidden launch-cell replenishment is sustaining the border flare-up despite public claims of containment.",
                severity=0.7,
                visibility="mixed",
                reliability=0.69,
                affected_agents=("israel", "hezbollah", "iran"),
                public_summary="Private reporting suggests the northern flare-up is more durable than public statements imply.",
                private_summary="Private reporting indicates replenishment activity is sustaining launch capacity despite public claims of containment.",
                decay_rate=0.05,
            ),
        ),
        public_events=(
            _signal(
                "scenario",
                "Cross-border rocket and drone volleys trigger alerts across the Galilee and immediate reserve call-ups.",
                region="israel",
                tags=("attack", "border", "drone"),
                severity=0.78,
            ),
            _signal(
                "scenario",
                "Northern communities report civil-defense strain as repeated intercept windows compress response times.",
                region="israel",
                tags=("humanitarian", "attack", "civilian"),
                severity=0.57,
            ),
        ),
        benchmark_signal_turns={
            1: (
                _signal(
                    "benchmark-front-watch",
                    "New launch detections indicate additional rocket preparation sites south of the Litani.",
                    region="israel",
                    tags=("attack", "border", "launch"),
                    severity=0.74,
                ),
            ),
            3: (
                _signal(
                    "benchmark-civil-defense",
                    "Interception leakage forces temporary closures around Haifa approaches and northern transport links.",
                    region="israel",
                    tags=("attack", "humanitarian", "infrastructure"),
                    severity=0.62,
                ),
            ),
            5: (
                _signal(
                    "benchmark-mediator",
                    "Backchannel mediators test a short pause tied to reduced launch tempo and repositioned artillery.",
                    region="israel",
                    tags=("diplomacy", "humanitarian"),
                    severity=0.46,
                ),
            ),
        },
    ),
    "corridor_interdiction": ScenarioDefinition(
        id="corridor_interdiction",
        name="Corridor Interdiction",
        description="Repeated interdiction pressure on the Iran-Hezbollah sustainment corridor creates strategic ambiguity and logistics stress.",
        tags=("corridor", "proxy", "interdiction", "logistics"),
        benchmark_turns=6,
        world_overrides={"tension_level": 66.0, "market_stress": 46.0, "oil_pressure": 54.0},
        metric_shifts=(
            ScenarioMetricShift("iran", "proxy_corridor", -12.0),
            ScenarioMetricShift("iran", "deterrence_credibility", -6.0),
            ScenarioMetricShift("hezbollah", "logistics_depth", -10.0),
            ScenarioMetricShift("israel", "northern_deterrence", 6.0),
            ScenarioMetricShift("us", "regional_access", 2.0),
        ),
        asset_impacts=(
            ScenarioAssetImpact(
                owner="iran",
                intensity=26.0,
                reason="scenario interdiction damage along the western sustainment chain",
                section_bias=("corridor", "logistics", "front"),
                max_assets=2,
                max_status="malfunctioning",
            ),
            ScenarioAssetImpact(
                owner="hezbollah",
                intensity=20.0,
                reason="scenario transit shortages and reserve burn",
                section_bias=("logistics", "corridor", "reserve"),
                max_assets=1,
                max_status="malfunctioning",
            ),
        ),
        latent_events=(
            ScenarioLatentEvent(
                topic="domestic",
                summary="Private financial channels indicate sanctions pressure is biting harder than public statements admit.",
                severity=0.62,
                visibility="mixed",
                reliability=0.67,
                affected_agents=("iran", "us", "gulf"),
                public_summary="Private market reporting suggests sanctions pressure is worsening beneath official messaging.",
                private_summary="Private channel reporting indicates domestic financial stress is compounding faster than public statements admit.",
                decay_rate=0.05,
            ),
        ),
        public_events=(
            _signal(
                "scenario",
                "Repeated strikes and seizures along the Iraq-Syria transit chain slow movement into the Levant theater.",
                region="iran",
                tags=("attack", "proxy", "logistics"),
                severity=0.7,
            ),
        ),
        benchmark_signal_turns={
            2: (
                _signal(
                    "benchmark-satellite-brief",
                    "New imagery suggests a damaged transit node is forcing reroutes deeper into the western corridor.",
                    region="iran",
                    tags=("attack", "proxy", "logistics"),
                    severity=0.66,
                ),
            ),
            4: (
                _signal(
                    "benchmark-regional-wire",
                    "Air-defense repositioning in Syria points to fears of additional corridor interdiction this week.",
                    region="iran",
                    tags=("attack", "intel", "proxy"),
                    severity=0.55,
                ),
            ),
        },
    ),
    "domestic_unrest": ScenarioDefinition(
        id="domestic_unrest",
        name="Domestic Unrest Shock",
        description="Political backlash, protest cycles, and cyber-linked disruptions reduce internal resilience and complicate escalation choices.",
        tags=("domestic", "unrest", "cyber", "politics"),
        benchmark_turns=6,
        world_overrides={"tension_level": 58.0, "market_stress": 54.0, "oil_pressure": 44.0},
        metric_shifts=(
            ScenarioMetricShift("us", "domestic_support", -12.0),
            ScenarioMetricShift("iran", "regime_stability", -10.0),
            ScenarioMetricShift("gulf", "investor_confidence", -6.0),
            ScenarioMetricShift("oversight", "runaway_risk", 8.0),
        ),
        asset_impacts=(
            ScenarioAssetImpact(
                owner="us",
                intensity=14.0,
                reason="scenario domestic continuity strain on command and civic systems",
                section_bias=("command", "capital", "civil"),
                max_assets=1,
                max_status="malfunctioning",
            ),
            ScenarioAssetImpact(
                owner="iran",
                intensity=16.0,
                reason="scenario protest and outage pressure on internal continuity assets",
                section_bias=("command", "capital", "civil"),
                max_assets=1,
                max_status="malfunctioning",
            ),
        ),
        public_events=(
            _signal(
                "scenario",
                "Protests, intermittent network outages, and political infighting add pressure to domestic decision cycles across the region.",
                tags=("unrest", "cyber", "politics"),
                severity=0.62,
            ),
        ),
        benchmark_signal_turns={
            1: (
                _signal(
                    "benchmark-domestic-wire",
                    "New footage shows protests widening after overnight communication outages and emergency curfews.",
                    tags=("unrest", "cyber", "humanitarian"),
                    severity=0.61,
                ),
            ),
            3: (
                _signal(
                    "benchmark-market-desk",
                    "Risk assets weaken as domestic instability narratives dominate regional coverage.",
                    tags=("market", "unrest"),
                    severity=0.49,
                ),
            ),
        },
    ),
    "coalition_fracture": ScenarioDefinition(
        id="coalition_fracture",
        name="Coalition Fracture",
        description="An allied split over escalation and resupply creates a brittle diplomatic environment and coordination loss.",
        tags=("coalition", "diplomacy", "resupply"),
        benchmark_turns=6,
        world_overrides={"tension_level": 56.0, "market_stress": 49.0, "oil_pressure": 52.0},
        coalition_overrides={
            "us": (),
            "israel": (),
            "iran": ("hezbollah",),
            "hezbollah": ("iran",),
            "gulf": (),
            "oversight": (),
        },
        metric_shifts=(
            ScenarioMetricShift("us", "regional_access", -10.0),
            ScenarioMetricShift("us", "domestic_support", -6.0),
            ScenarioMetricShift("israel", "us_resupply_confidence", -12.0),
            ScenarioMetricShift("gulf", "diplomatic_flexibility", -10.0),
            ScenarioMetricShift("gulf", "investor_confidence", -5.0),
            ScenarioMetricShift("oversight", "intervention_legitimacy", -6.0),
        ),
        public_events=(
            _signal(
                "scenario",
                "Allied partners publicly diverge on resupply timing, escalation thresholds, and maritime burden-sharing.",
                tags=("diplomacy", "market", "coalition"),
                severity=0.55,
            ),
        ),
        benchmark_signal_turns={
            2: (
                _signal(
                    "benchmark-diplomatic-wire",
                    "Emergency talks fail to fully restore allied messaging discipline after another public dispute over regional commitments.",
                    tags=("diplomacy", "coalition", "market"),
                    severity=0.57,
                ),
            ),
            4: (
                _signal(
                    "benchmark-backchannel",
                    "Quiet envoys propose a limited burden-sharing package tied to de-escalation benchmarks and restored shipping coordination.",
                    tags=("diplomacy", "shipping"),
                    severity=0.42,
                ),
            ),
        },
    ),
}


def list_scenario_definitions() -> list[ScenarioDefinition]:
    return [SCENARIOS[scenario_id] for scenario_id in SCENARIOS]


def benchmark_scenario_ids() -> list[str]:
    return [scenario.id for scenario in list_scenario_definitions() if scenario.benchmark_enabled]


def get_scenario_definition(scenario_id: str | None = None) -> ScenarioDefinition:
    resolved_scenario_id = scenario_id or DEFAULT_SCENARIO_ID
    try:
        return SCENARIOS[resolved_scenario_id]
    except KeyError as exc:
        known = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario_id: {resolved_scenario_id}. Known scenarios: {known}") from exc


def scenario_signals_for_turn(scenario_id: str | None, turn: int) -> list[ExternalSignal]:
    scenario = get_scenario_definition(scenario_id)
    return [
        ExternalSignal(
            source=signal.source,
            headline=signal.headline,
            region=signal.region,
            tags=list(signal.tags),
            severity=signal.severity,
        )
        for signal in scenario.benchmark_signal_turns.get(turn, ())
    ]
