from __future__ import annotations

from typing import Literal

DEFAULT_MAX_TURNS = 1_000
DEFAULT_TRAINING_STAGE = "stage_3_sparse"

TrainingStage = Literal["stage_1_dense", "stage_2_partial", "stage_3_sparse"]

ALGORITHM_HINTS = {
    "single_agent": "PPO",
    "multi_agent": "GRPO",
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

AGENT_REWARD_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    "us": (0.25, 0.25, 0.35, 0.15),
    "israel": (0.30, 0.35, 0.20, 0.15),
    "iran": (0.20, 0.40, 0.25, 0.15),
    "hezbollah": (0.25, 0.30, 0.20, 0.25),
    "gulf": (0.25, 0.20, 0.40, 0.15),
    "oversight": (0.10, 0.50, 0.10, 0.30),
}
