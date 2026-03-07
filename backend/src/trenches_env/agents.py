from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgentId = Literal["us", "israel", "iran", "hezbollah", "gulf", "oversight"]


@dataclass(frozen=True)
class AgentProfile:
    display_name: str
    role: str
    intelligence_focus: tuple[str, ...]
    baseline_private_intel: tuple[str, ...]


AGENT_IDS: tuple[AgentId, ...] = (
    "us",
    "israel",
    "iran",
    "hezbollah",
    "gulf",
    "oversight",
)


AGENT_PROFILES: dict[AgentId, AgentProfile] = {
    "us": AgentProfile(
        display_name="US / CENTCOM",
        role="Alliance management, sanctions, domestic stability",
        intelligence_focus=("polls", "markets", "alliances", "shipping"),
        baseline_private_intel=(
            "Domestic approval is sensitive to prolonged escalation.",
            "Forward naval posture can deter but also spike market stress.",
        ),
    ),
    "israel": AgentProfile(
        display_name="Israel / IDF",
        role="Border defense, strike planning, proxy disruption",
        intelligence_focus=("northern front", "sirens", "proxy movement", "air defense"),
        baseline_private_intel=(
            "Border warning posture remains elevated in the north.",
            "Fast retaliation can secure deterrence but raises coalition risk.",
        ),
    ),
    "iran": AgentProfile(
        display_name="Iran / IRGC",
        role="Asymmetric retaliation, proxy coordination, survival",
        intelligence_focus=("proxy network", "oil chokepoints", "internal losses", "deception"),
        baseline_private_intel=(
            "Proxy coordination is most effective when attribution stays ambiguous.",
            "Energy chokepoints remain the strongest leverage point.",
        ),
    ),
    "hezbollah": AgentProfile(
        display_name="Hezbollah",
        role="Asymmetric swarming, opportunistic escalation",
        intelligence_focus=("border gaps", "morale", "small-unit pressure", "drone windows"),
        baseline_private_intel=(
            "Small, frequent attacks are harder to pre-empt than large waves.",
            "Alignment with Tehran matters more than independent visibility.",
        ),
    ),
    "gulf": AgentProfile(
        display_name="Gulf Coalition",
        role="Market hedging, shipping security, selective alignment",
        intelligence_focus=("oil", "shipping", "capital flows", "neutrality"),
        baseline_private_intel=(
            "Energy shock containment matters more than direct battlefield gains.",
            "Neutral positioning creates leverage only while trade routes remain open.",
        ),
    ),
    "oversight": AgentProfile(
        display_name="Fleet Oversight",
        role="Risk scoring, intervention, trace auditing",
        intelligence_focus=("global risk", "misalignment", "cascades", "de-escalation"),
        baseline_private_intel=(
            "Misread incentives are the strongest predictor of runaway escalation.",
            "Interventions should reduce risk without collapsing agent autonomy.",
        ),
    ),
}
