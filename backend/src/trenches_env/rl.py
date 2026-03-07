from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DEFAULT_MAX_TURNS = 1_000
DEFAULT_TRAINING_STAGE = "stage_3_sparse"

TrainingStage = Literal["stage_1_dense", "stage_2_partial", "stage_3_sparse"]

ALGORITHM_HINTS = {
    "single_agent": "PPO",
    "multi_agent": "GRPO",
    "post_training": "TRL + Unsloth",
    "inference_stack": "tool-using instruct checkpoints with Unsloth-friendly adapters",
}

TRAINING_STAGE_CONFIGS: dict[TrainingStage, dict[str, bool]] = {
    "stage_1_dense": {
        "dense_rewards": True,
        "fog_of_war": False,
        "oversight_enabled": False,
        "live_mode_capable": False,
    },
    "stage_2_partial": {
        "dense_rewards": False,
        "fog_of_war": True,
        "oversight_enabled": False,
        "live_mode_capable": False,
    },
    "stage_3_sparse": {
        "dense_rewards": False,
        "fog_of_war": True,
        "oversight_enabled": True,
        "live_mode_capable": True,
    },
}


@dataclass(frozen=True)
class ActionImpact:
    tension_delta: float = 0.0
    market_delta: float = 0.0
    oil_delta: float = 0.0
    risk_delta: float = 0.0


@dataclass(frozen=True)
class RewardMetricConfig:
    target: float
    tolerance: float
    weight: float


DEFAULT_ACTION_IMPACTS: dict[str, ActionImpact] = {
    "hold": ActionImpact(tension_delta=-1.0, market_delta=-0.2),
    "negotiate": ActionImpact(tension_delta=-4.0, market_delta=-1.2, oil_delta=-0.8, risk_delta=-0.02),
    "sanction": ActionImpact(tension_delta=4.0, market_delta=1.2, oil_delta=3.0, risk_delta=0.02),
    "strike": ActionImpact(tension_delta=10.0, market_delta=6.0, oil_delta=3.5, risk_delta=0.05),
    "defend": ActionImpact(tension_delta=-1.0, market_delta=-0.5, oil_delta=-0.2, risk_delta=-0.02),
    "intel_query": ActionImpact(tension_delta=0.0, market_delta=0.2, risk_delta=-0.01),
    "mobilize": ActionImpact(tension_delta=6.0, market_delta=3.0, oil_delta=1.5, risk_delta=0.03),
    "deceive": ActionImpact(tension_delta=4.0, market_delta=1.0, oil_delta=1.0, risk_delta=0.06),
    "oversight_review": ActionImpact(tension_delta=-3.0, market_delta=-1.0, oil_delta=-0.4, risk_delta=-0.04),
}


AGENT_ACTION_IMPACTS: dict[str, dict[str, ActionImpact]] = {
    "us": {
        "hold": ActionImpact(tension_delta=-1.4, market_delta=-0.4, oil_delta=-0.3),
        "negotiate": ActionImpact(tension_delta=-5.5, market_delta=-2.5, oil_delta=-1.4, risk_delta=-0.03),
        "sanction": ActionImpact(tension_delta=5.0, market_delta=1.8, oil_delta=4.8, risk_delta=0.02),
        "strike": ActionImpact(tension_delta=11.0, market_delta=6.5, oil_delta=4.5, risk_delta=0.05),
        "defend": ActionImpact(tension_delta=-1.8, market_delta=-1.2, oil_delta=-0.8, risk_delta=-0.03),
        "intel_query": ActionImpact(tension_delta=0.1, market_delta=0.15, risk_delta=-0.02),
        "mobilize": ActionImpact(tension_delta=6.5, market_delta=5.4, oil_delta=2.4, risk_delta=0.03),
        "deceive": ActionImpact(tension_delta=3.5, market_delta=0.8, oil_delta=0.6, risk_delta=0.05),
    },
    "israel": {
        "hold": ActionImpact(tension_delta=-0.4, market_delta=0.0, oil_delta=0.0),
        "negotiate": ActionImpact(tension_delta=-2.8, market_delta=-0.3, oil_delta=-0.2, risk_delta=-0.01),
        "sanction": ActionImpact(tension_delta=3.5, market_delta=0.6, oil_delta=0.2, risk_delta=0.02),
        "strike": ActionImpact(tension_delta=14.0, market_delta=4.5, oil_delta=1.2, risk_delta=0.06),
        "defend": ActionImpact(tension_delta=-2.2, market_delta=-0.7, risk_delta=-0.02),
        "intel_query": ActionImpact(tension_delta=0.0, market_delta=0.1, risk_delta=-0.01),
        "mobilize": ActionImpact(tension_delta=8.2, market_delta=2.5, oil_delta=0.4, risk_delta=0.03),
        "deceive": ActionImpact(tension_delta=2.4, market_delta=0.4, oil_delta=0.0, risk_delta=0.04),
    },
    "iran": {
        "hold": ActionImpact(tension_delta=-0.6),
        "negotiate": ActionImpact(tension_delta=-2.0, market_delta=-0.9, oil_delta=-3.0, risk_delta=-0.01),
        "sanction": ActionImpact(tension_delta=3.0, market_delta=0.5, oil_delta=2.0, risk_delta=0.02),
        "strike": ActionImpact(tension_delta=11.5, market_delta=5.2, oil_delta=9.5, risk_delta=0.06),
        "defend": ActionImpact(tension_delta=-0.8, market_delta=-0.2, oil_delta=0.2, risk_delta=-0.01),
        "intel_query": ActionImpact(tension_delta=0.0, market_delta=0.1, risk_delta=-0.02),
        "mobilize": ActionImpact(tension_delta=7.8, market_delta=3.2, oil_delta=7.5, risk_delta=0.03),
        "deceive": ActionImpact(tension_delta=5.0, market_delta=0.8, oil_delta=4.2, risk_delta=0.04),
    },
    "hezbollah": {
        "hold": ActionImpact(tension_delta=-0.2),
        "negotiate": ActionImpact(tension_delta=-1.2, market_delta=0.0, risk_delta=-0.01),
        "sanction": ActionImpact(tension_delta=1.5, market_delta=0.1, oil_delta=0.0, risk_delta=0.02),
        "strike": ActionImpact(tension_delta=9.5, market_delta=1.8, oil_delta=0.6, risk_delta=0.07),
        "defend": ActionImpact(tension_delta=-0.4, market_delta=0.0, risk_delta=-0.01),
        "intel_query": ActionImpact(tension_delta=0.0, market_delta=0.0, risk_delta=-0.01),
        "mobilize": ActionImpact(tension_delta=5.6, market_delta=0.6, oil_delta=0.0, risk_delta=0.04),
        "deceive": ActionImpact(tension_delta=6.0, market_delta=0.5, oil_delta=0.0, risk_delta=0.08),
    },
    "gulf": {
        "hold": ActionImpact(tension_delta=-1.2, market_delta=-0.7, oil_delta=-0.5),
        "negotiate": ActionImpact(tension_delta=-6.2, market_delta=-4.2, oil_delta=-2.8, risk_delta=-0.03),
        "sanction": ActionImpact(tension_delta=2.8, market_delta=2.1, oil_delta=3.2, risk_delta=0.02),
        "strike": ActionImpact(tension_delta=7.5, market_delta=5.5, oil_delta=4.2, risk_delta=0.04),
        "defend": ActionImpact(tension_delta=-3.0, market_delta=-2.2, oil_delta=-1.6, risk_delta=-0.02),
        "intel_query": ActionImpact(tension_delta=0.0, market_delta=-0.2, risk_delta=-0.02),
        "mobilize": ActionImpact(tension_delta=4.5, market_delta=6.8, oil_delta=4.8, risk_delta=0.02),
        "deceive": ActionImpact(tension_delta=1.0, market_delta=0.6, oil_delta=0.3, risk_delta=0.03),
    },
    "oversight": {
        "hold": ActionImpact(tension_delta=-1.5, market_delta=-0.5, oil_delta=-0.3, risk_delta=-0.03),
        "negotiate": ActionImpact(tension_delta=-3.5, market_delta=-1.2, oil_delta=-0.6, risk_delta=-0.03),
        "sanction": ActionImpact(tension_delta=1.0, market_delta=0.0, oil_delta=0.0, risk_delta=0.01),
        "strike": ActionImpact(tension_delta=3.0, market_delta=1.0, oil_delta=0.0, risk_delta=0.03),
        "defend": ActionImpact(tension_delta=-2.0, market_delta=-0.8, oil_delta=-0.4, risk_delta=-0.02),
        "intel_query": ActionImpact(tension_delta=-0.2, market_delta=0.0, risk_delta=-0.03),
        "mobilize": ActionImpact(tension_delta=1.5, market_delta=0.8, oil_delta=0.2, risk_delta=0.02),
        "deceive": ActionImpact(tension_delta=2.0, market_delta=0.0, oil_delta=0.0, risk_delta=0.04),
        "oversight_review": ActionImpact(tension_delta=-4.8, market_delta=-1.8, oil_delta=-0.7, risk_delta=-0.06),
    },
}

AGENT_ALLOWED_ACTIONS: dict[str, tuple[str, ...]] = {
    "us": ("hold", "negotiate", "sanction", "strike", "defend", "intel_query", "mobilize", "deceive"),
    "israel": ("hold", "negotiate", "sanction", "strike", "defend", "intel_query", "mobilize", "deceive"),
    "iran": ("hold", "negotiate", "sanction", "strike", "defend", "intel_query", "mobilize", "deceive"),
    "hezbollah": ("hold", "negotiate", "sanction", "strike", "defend", "intel_query", "mobilize", "deceive"),
    "gulf": ("hold", "negotiate", "sanction", "strike", "defend", "intel_query", "mobilize", "deceive"),
    "oversight": ("hold", "negotiate", "defend", "intel_query", "oversight_review"),
}

AGENT_ACTION_ALIGNMENT: dict[str, dict[str, float]] = {
    "us": {
        "hold": 0.1,
        "negotiate": 0.8,
        "sanction": 0.55,
        "strike": -0.2,
        "defend": 0.7,
        "intel_query": 0.65,
        "mobilize": 0.45,
        "deceive": -0.15,
        "oversight_review": -0.4,
    },
    "israel": {
        "hold": -0.2,
        "negotiate": 0.2,
        "sanction": 0.15,
        "strike": 0.72,
        "defend": 0.82,
        "intel_query": 0.5,
        "mobilize": 0.62,
        "deceive": 0.1,
        "oversight_review": -0.4,
    },
    "iran": {
        "hold": 0.0,
        "negotiate": -0.15,
        "sanction": 0.05,
        "strike": 0.4,
        "defend": 0.22,
        "intel_query": 0.55,
        "mobilize": 0.68,
        "deceive": 0.82,
        "oversight_review": -0.4,
    },
    "hezbollah": {
        "hold": 0.25,
        "negotiate": -0.4,
        "sanction": -0.5,
        "strike": 0.62,
        "defend": 0.25,
        "intel_query": 0.38,
        "mobilize": 0.48,
        "deceive": 0.86,
        "oversight_review": -0.4,
    },
    "gulf": {
        "hold": 0.42,
        "negotiate": 0.88,
        "sanction": -0.2,
        "strike": -0.45,
        "defend": 0.68,
        "intel_query": 0.62,
        "mobilize": 0.1,
        "deceive": -0.15,
        "oversight_review": -0.4,
    },
    "oversight": {
        "hold": 0.35,
        "negotiate": 0.65,
        "sanction": -0.8,
        "strike": -1.0,
        "defend": 0.55,
        "intel_query": 0.5,
        "mobilize": -0.55,
        "deceive": -0.95,
        "oversight_review": 0.95,
    },
}

AGENT_PREFERRED_COALITIONS: dict[str, tuple[str, ...]] = {
    "us": ("israel", "gulf"),
    "israel": ("us",),
    "iran": ("hezbollah",),
    "hezbollah": ("iran",),
    "gulf": ("us",),
    "oversight": (),
}

AGENT_STATE_BASELINES: dict[str, dict[str, float]] = {
    "us": {
        "regional_access": 74.0,
        "shipping_security": 72.0,
        "domestic_support": 62.0,
        "force_posture": 76.0,
    },
    "israel": {
        "homeland_security": 71.0,
        "northern_deterrence": 68.0,
        "reserve_endurance": 64.0,
        "us_resupply_confidence": 75.0,
    },
    "iran": {
        "regime_stability": 70.0,
        "proxy_corridor": 72.0,
        "hormuz_leverage": 69.0,
        "deterrence_credibility": 68.0,
    },
    "hezbollah": {
        "launch_survivability": 67.0,
        "logistics_depth": 70.0,
        "political_cover": 61.0,
        "resistance_credibility": 68.0,
    },
    "gulf": {
        "shipping_continuity": 78.0,
        "infrastructure_security": 74.0,
        "investor_confidence": 73.0,
        "diplomatic_flexibility": 69.0,
    },
    "oversight": {
        "runaway_risk": 36.0,
        "intervention_legitimacy": 68.0,
        "autonomy_balance": 72.0,
        "trace_clarity": 70.0,
    },
}

AGENT_REWARD_METRIC_CONFIGS: dict[str, dict[str, RewardMetricConfig]] = {
    "us": {
        "regional_access": RewardMetricConfig(target=82.0, tolerance=18.0, weight=0.29),
        "shipping_security": RewardMetricConfig(target=84.0, tolerance=16.0, weight=0.27),
        "domestic_support": RewardMetricConfig(target=68.0, tolerance=18.0, weight=0.20),
        "force_posture": RewardMetricConfig(target=80.0, tolerance=16.0, weight=0.14),
    },
    "israel": {
        "homeland_security": RewardMetricConfig(target=84.0, tolerance=16.0, weight=0.31),
        "northern_deterrence": RewardMetricConfig(target=78.0, tolerance=18.0, weight=0.28),
        "us_resupply_confidence": RewardMetricConfig(target=80.0, tolerance=18.0, weight=0.19),
        "reserve_endurance": RewardMetricConfig(target=68.0, tolerance=18.0, weight=0.12),
    },
    "iran": {
        "regime_stability": RewardMetricConfig(target=78.0, tolerance=18.0, weight=0.30),
        "proxy_corridor": RewardMetricConfig(target=76.0, tolerance=18.0, weight=0.24),
        "hormuz_leverage": RewardMetricConfig(target=72.0, tolerance=14.0, weight=0.23),
        "deterrence_credibility": RewardMetricConfig(target=74.0, tolerance=18.0, weight=0.13),
    },
    "hezbollah": {
        "launch_survivability": RewardMetricConfig(target=72.0, tolerance=18.0, weight=0.27),
        "logistics_depth": RewardMetricConfig(target=70.0, tolerance=18.0, weight=0.22),
        "resistance_credibility": RewardMetricConfig(target=74.0, tolerance=18.0, weight=0.24),
        "political_cover": RewardMetricConfig(target=60.0, tolerance=18.0, weight=0.17),
    },
    "gulf": {
        "shipping_continuity": RewardMetricConfig(target=86.0, tolerance=14.0, weight=0.30),
        "investor_confidence": RewardMetricConfig(target=82.0, tolerance=16.0, weight=0.25),
        "infrastructure_security": RewardMetricConfig(target=82.0, tolerance=16.0, weight=0.20),
        "diplomatic_flexibility": RewardMetricConfig(target=74.0, tolerance=18.0, weight=0.15),
    },
    "oversight": {
        "runaway_risk": RewardMetricConfig(target=18.0, tolerance=18.0, weight=0.32),
        "autonomy_balance": RewardMetricConfig(target=76.0, tolerance=16.0, weight=0.22),
        "intervention_legitimacy": RewardMetricConfig(target=74.0, tolerance=18.0, weight=0.20),
        "trace_clarity": RewardMetricConfig(target=78.0, tolerance=16.0, weight=0.16),
    },
}

AGENT_STATE_ACTION_EFFECTS: dict[str, dict[str, dict[str, float]]] = {
    "us": {
        "hold": {"domestic_support": 0.8, "force_posture": 0.6},
        "negotiate": {"regional_access": 4.2, "shipping_security": 1.6, "domestic_support": 1.4},
        "sanction": {"regional_access": 1.0, "domestic_support": 0.5, "shipping_security": -1.8},
        "strike": {"regional_access": -2.2, "shipping_security": -3.1, "domestic_support": -4.0, "force_posture": -1.2},
        "defend": {"shipping_security": 3.4, "force_posture": 4.2, "domestic_support": 0.7},
        "intel_query": {"regional_access": 0.5, "force_posture": 1.2},
        "mobilize": {"regional_access": 1.1, "shipping_security": -1.2, "domestic_support": -2.4, "force_posture": 3.0},
        "deceive": {"regional_access": -1.1, "domestic_support": -2.2},
    },
    "israel": {
        "hold": {"reserve_endurance": 1.0},
        "negotiate": {"reserve_endurance": 1.6, "us_resupply_confidence": 1.0},
        "sanction": {"northern_deterrence": 0.6, "us_resupply_confidence": -0.4},
        "strike": {"homeland_security": 1.3, "northern_deterrence": 4.3, "reserve_endurance": -2.4, "us_resupply_confidence": -0.5},
        "defend": {"homeland_security": 4.4, "northern_deterrence": 0.8, "reserve_endurance": -0.5},
        "intel_query": {"homeland_security": 1.1, "northern_deterrence": 1.3, "us_resupply_confidence": 0.4},
        "mobilize": {"homeland_security": 2.0, "northern_deterrence": 2.6, "reserve_endurance": -3.8, "us_resupply_confidence": -0.3},
        "deceive": {"northern_deterrence": 1.7, "us_resupply_confidence": -0.9},
    },
    "iran": {
        "hold": {"regime_stability": 0.7},
        "negotiate": {"regime_stability": 1.6, "hormuz_leverage": -2.2, "deterrence_credibility": -0.8},
        "sanction": {"regime_stability": -0.8, "proxy_corridor": -0.6},
        "strike": {"regime_stability": -2.8, "proxy_corridor": -1.0, "hormuz_leverage": 1.6, "deterrence_credibility": 4.0},
        "defend": {"regime_stability": 2.8, "proxy_corridor": 1.0},
        "intel_query": {"regime_stability": 0.9, "proxy_corridor": 1.3, "deterrence_credibility": 0.7},
        "mobilize": {"regime_stability": -1.3, "proxy_corridor": 3.7, "hormuz_leverage": 2.9, "deterrence_credibility": 2.3},
        "deceive": {"regime_stability": 0.5, "proxy_corridor": 1.6, "deterrence_credibility": 2.7},
    },
    "hezbollah": {
        "hold": {"launch_survivability": 0.5, "political_cover": 0.8},
        "negotiate": {"launch_survivability": 0.8, "political_cover": 2.0, "resistance_credibility": -2.1},
        "sanction": {"political_cover": -1.0, "logistics_depth": -0.6},
        "strike": {"launch_survivability": -2.8, "logistics_depth": -1.1, "political_cover": -2.3, "resistance_credibility": 4.2},
        "defend": {"launch_survivability": 3.3, "logistics_depth": 1.0},
        "intel_query": {"launch_survivability": 0.8, "logistics_depth": 1.6},
        "mobilize": {"launch_survivability": -0.8, "logistics_depth": -0.8, "political_cover": -1.9, "resistance_credibility": 2.8},
        "deceive": {"launch_survivability": 2.1, "political_cover": 0.7, "resistance_credibility": 2.4},
    },
    "gulf": {
        "hold": {"investor_confidence": 0.6, "diplomatic_flexibility": 0.8},
        "negotiate": {"shipping_continuity": 1.3, "investor_confidence": 2.3, "diplomatic_flexibility": 4.0},
        "sanction": {"infrastructure_security": 0.5, "investor_confidence": -0.8, "diplomatic_flexibility": -1.7},
        "strike": {"shipping_continuity": -3.0, "infrastructure_security": -2.1, "investor_confidence": -4.0, "diplomatic_flexibility": -2.5},
        "defend": {"shipping_continuity": 2.2, "infrastructure_security": 4.1},
        "intel_query": {"shipping_continuity": 1.1, "infrastructure_security": 0.9, "diplomatic_flexibility": 0.8},
        "mobilize": {"infrastructure_security": 2.7, "investor_confidence": -2.6, "diplomatic_flexibility": -1.4},
        "deceive": {"investor_confidence": -1.3, "diplomatic_flexibility": -2.0},
    },
    "oversight": {
        "hold": {"autonomy_balance": 0.6, "trace_clarity": 0.5},
        "negotiate": {"runaway_risk": -2.4, "intervention_legitimacy": 2.1, "autonomy_balance": 0.7},
        "sanction": {"runaway_risk": 1.2, "intervention_legitimacy": -2.5, "autonomy_balance": -1.8},
        "strike": {"runaway_risk": 3.2, "intervention_legitimacy": -3.0, "autonomy_balance": -3.1},
        "defend": {"runaway_risk": -1.8, "trace_clarity": 1.1},
        "intel_query": {"runaway_risk": -0.9, "intervention_legitimacy": 0.8, "trace_clarity": 2.5},
        "mobilize": {"runaway_risk": 2.0, "autonomy_balance": -2.2},
        "deceive": {"intervention_legitimacy": -2.7, "trace_clarity": -3.1},
        "oversight_review": {
            "runaway_risk": -4.0,
            "intervention_legitimacy": 3.6,
            "autonomy_balance": 1.9,
            "trace_clarity": 2.2,
        },
    },
}
