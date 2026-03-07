# RL.md: Reinforcement Learning Mechanics in Fog of War Diplomacy Simulator

This document describes the RL design for the Fog of War Diplomacy Simulator and, critically, the boundary between what OpenEnv supports directly and what this project must implement on top of OpenEnv.

The short version:

- OpenEnv is the environment packaging and execution layer.
- The crisis simulator, reward model, six-agent orchestration, and oversight logic are project code.
- CTDE, MAPPO, PPO, GRPO, TRL, and RLlib are training-stack choices outside OpenEnv itself.

The current repository now includes a native OpenEnv-facing adapter layer around the simulator, while still retaining the older session-oriented FastAPI API used by the dashboard.

---

## OpenEnv Support Boundary

### What OpenEnv directly supports

OpenEnv gives us the environment contract and runtime surface:

- An async environment interface built around `reset()` and `step(action)`.
- Environment-side `Action`, `Observation`, and `State` models.
- Client-side `StepResult` objects that expose:
  - `observation`
  - scalar `reward`
  - `done`
- Client access through `EnvClient`, typically from a Docker image or a deployed URL.
- Containerized environment packaging.
- Optional custom UI support.

In other words, OpenEnv is well-suited to hosting the simulator and stepping it remotely, but it is not a multi-agent RL trainer and it does not natively provide Gymnasium-style multi-agent dict rewards or observations.

### What this project layers on top

The following are project-level features, not native OpenEnv guarantees:

- Six-agent world state with partial observability.
- Joint action parsing and per-agent observation projection.
- Per-agent reward vectors and reward breakdowns.
- Oversight risk scoring and corrective interventions.
- Curriculum design, CTDE, MAPPO, PPO, GRPO, TRL, and RLlib integration.
- Live data ingestion plans, RSS/Telegram/video source routing, and demo-only live mode.
- The current FastAPI session API in this repo.

### Important design consequence

OpenEnv expects one action in and one scalar reward out per `step()`. For this simulator, that means one of two patterns:

1. Encode the full six-agent joint move as a single structured OpenEnv action, then expose per-agent details through custom observation fields and `state()`.
2. Build a higher-level coordinator outside OpenEnv that manages multiple policies against one shared world state.

For the MVP and the current backend shape, pattern `1` is the cleaner fit.

---

## OpenEnv-Aligned Environment Contract

The OpenEnv adapter for this project should look conceptually like this:

```python
from openenv.core import Action, Environment, Observation
from openenv.core.env_server.types import State


class JointAction(Action):
    actions: dict[str, dict]


class DiplomacyObservation(Observation):
    training_agent: str
    reward_breakdown: dict


class DiplomacyState(State):
    training_agent: str
    world: dict


class DiplomacyEnvironment(Environment):
    @property
    def state(self) -> DiplomacyState:
        return DiplomacyState(
            episode_id=self.session_id,
            step_count=self.turn,
            training_agent=self.training_agent,
            world=self._serialize_world(self.world),
        )

    async def reset(self):
        self.world = self._initial_world()
        return self._build_observation(self.world)

    async def step(self, action):
        # `action` is a structured joint action for all six agents.
        joint_actions = self._decode_joint_action(action)

        self._apply_actions(self.world, joint_actions)
        oversight = self._compute_oversight(self.world, joint_actions)
        self._apply_oversight(self.world, oversight)

        per_agent_rewards = self._compute_rewards(self.world, joint_actions)

        # OpenEnv needs a scalar reward. The trainer/coordinator chooses
        # which policy is being optimized for this rollout.
        scalar_reward = per_agent_rewards[self.training_agent]

        return DiplomacyObservation(
            reward=scalar_reward,
            done=self._is_done(self.world),
            training_agent=self.training_agent,
            reward_breakdown=per_agent_rewards[self.training_agent],
        )
```

This is the key correction relative to earlier drafts: the simulator may compute rich multi-agent state internally, but the OpenEnv-facing `step()` still emits a single `Observation`, and the client sees a scalar-reward `StepResult`.

---

## Relationship to the Current Repo

The current backend environment is not yet a direct OpenEnv environment. Today it is a custom simulator plus FastAPI session layer that exposes:

- session creation/reset
- live-mode toggles
- turn stepping
- structured per-agent observations and reward breakdowns

That is still useful. It means the simulator logic is already mostly in place, and the OpenEnv integration work becomes an adapter task rather than a full rewrite.

The practical implication is:

- `backend/src/trenches_env/env.py` is the world simulator.
- OpenEnv should wrap or call into that simulator.
- `RL.md` must not describe current behavior as if the repo is already using OpenEnv natively.

---

## Reward System: Project Logic on Top of OpenEnv

Rewards remain a project design choice. OpenEnv does not impose the reward formula; it only transports the scalar training reward through `Observation.reward` and exposes richer environment state through `state()`.

### Design constraints

- Reward computation happens after action processing and state updates.
- Each component should be normalized before weighting.
- Oversight should modify environment state, not rescale rewards.
- Hidden incentives should not be used as a direct reward signal.
- Full per-agent reward breakdowns can live in simulator state or custom observation fields even though the outward training reward is scalar.

### Core Reward Formula

Per timestep $t$, for each agent $i$:

$$r_t^i = w_1 \cdot \hat{C}_t^i + w_2 \cdot \hat{E}_t^i + w_3 \cdot \hat{M}_t^i + w_4 \cdot \hat{B}_t^i$$

where each component is normalized independently to `[-1, 1]` before aggregation.

### Components

- $C_t$: Coalition stability.
- $E_t$: Escalation penalty using an EMA of tension level.
- $M_t$: Market/economic gain from observable stress reduction.
- $B_t$: Behavioral consistency from observable action and rationale traces rather than hidden oracle state.

### OpenEnv-facing reward rule

Internally we may compute:

```python
per_agent_rewards = {
    "us": ...,
    "israel": ...,
    "iran": ...,
    "hezbollah": ...,
    "gulf": ...,
    "oversight": ...,
}
```

But the OpenEnv adapter should emit:

```python
DiplomacyObservation(
    reward=per_agent_rewards[active_training_agent],
    done=done,
    reward_breakdown=per_agent_rewards[active_training_agent],
)
```

That keeps the simulator expressive without claiming unsupported native multi-agent reward output.

---

## Oversight: OpenEnv-Compatible Placement

The previous Gymnasium-wrapper framing was too specific. OpenEnv does not give us a native `gym.Wrapper` abstraction, so oversight should be implemented as part of the simulator transition or as a thin project-side interceptor around the environment.

An OpenEnv-compatible pattern is:

```python
async def step(self, action):
    joint_actions = self._decode_joint_action(action)
    self._apply_actions(self.world, joint_actions)

    oversight = self._compute_oversight(self.world, joint_actions)
    if oversight["triggered"]:
        self._apply_oversight(self.world, oversight)

    per_agent_rewards = self._compute_rewards(self.world, joint_actions)

    return DiplomacyObservation(
        reward=per_agent_rewards[self.training_agent],
        done=self._is_done(self.world),
        reward_breakdown=per_agent_rewards[self.training_agent],
    )
```

This preserves the intended semantics:

- oversight changes the transition
- reward is not rescaled
- intervention details remain inspectable through `state()` or custom observation fields

---

## Multi-Agent Training Architecture

### What OpenEnv does not do for us

OpenEnv is not a built-in multi-agent trainer. It does not natively provide:

- CTDE
- MAPPO
- GRPO
- centralized critics
- per-agent replay buffers
- RLlib or TRL integration

Those belong in the training harness.

### Recommended architecture

Use OpenEnv as the rollout environment, then place the multi-agent trainer above it:

1. OpenEnv hosts one simulator instance.
2. Each rollout step carries a structured joint action for all six agents.
3. The simulator computes the full per-agent reward vector.
4. The OpenEnv adapter returns the scalar reward for the currently optimized policy and exposes richer diagnostics through `state()` and custom observation fields.
5. The trainer reconstructs per-agent trajectories from state snapshots and session traces.

This works for:

- independent PPO baselines
- CTDE with a centralized critic
- MAPPO-style actor-critic training
- GRPO-style grouped rollout training

But again: these are external training choices, not native OpenEnv features.

---

## Algorithm Choice

### CTDE

CTDE remains a sound design choice for this simulator because agents interact in a shared partially observable world. The centralized critic is trainer-side logic and does not require native OpenEnv support.

### GRPO vs PPO

GRPO is still a plausible fit for sparse long-horizon signals, but the doc should treat it as an external training-stack choice. OpenEnv will not provide `GRPOTrainer`; it only supplies environment rollouts.

PPO remains a valid baseline, especially for shorter or denser curriculum stages.

### Practical recommendation

Phrase the implementation plan as:

- OpenEnv for rollout generation
- custom trainer or external framework for policy updates
- state snapshots and session traces for reconstructing per-agent returns

not as:

- OpenEnv natively handles GRPO or multi-agent PPO

---

## Training Flow

1. Package the simulator as an OpenEnv environment or connect to it through `EnvClient.from_url(...)` once deployed.
2. Reset the environment to get the initial joint observation.
3. Query all six policies to produce one joint action.
4. Step the environment and capture:
   - next observation
   - scalar reward for the active policy
   - done flag
   - environment state with per-agent rewards, oversight data, and world trace
5. Reconstruct trainer-side trajectories for CTDE, MAPPO, PPO, or GRPO.
6. Keep all training episodic and reproducible.

### Live mode rule

Live mode is inference/demo only.

Training should use:

- episodic rollouts
- fixed seeds where needed
- replayed or sampled event bundles

Training should not depend on live RSS/Telegram/video streams if reproducibility matters.

That means the earlier idea of "Stage 3 training with live RSS injection" should be replaced by "Stage 3 training with replayed sampled event bundles and oversight enabled."

---

## Evaluation

These evaluation targets are still reasonable project metrics:

| Metric | Definition | Target |
|---|---|---|
| Avg reward/episode | Mean $\sum_t r_t$ over recent episodes | Upward trend |
| De-escalation rate | % of episodes ending with tension < 30 | > 60% |
| Oversight intervention rate | Interventions per episode | Decreasing over training |
| Behavioral consistency | Mean $B_t$ across agents | > 0.7 |
| Coalition durability | Avg turns before first betrayal | > 200 |

But the trainer must compute them from rollout traces. OpenEnv will not provide these metrics automatically.

---

## Known Challenges

- Multi-agent credit assignment is trainer complexity, not environment complexity.
- If training data comes from changing live sources, reproducibility degrades fast.
- Reward hacking remains a real risk.
- OpenEnv scalar reward output means the adapter boundary must be explicit and carefully documented.

---

## Summary of Corrections

| Earlier claim | Corrected statement |
|---|---|
| OpenEnv extends Gymnasium with dict observations/actions and done-truncated-info tuples | OpenEnv uses its own async `reset()` / `step()` contract with typed `Action`/`Observation`/`State`; the client exposes `StepResult` with scalar reward |
| OpenEnv natively handles six-agent dict rewards and observations | Multi-agent orchestration is project logic layered on top of OpenEnv |
| OpenEnv envs are raw FastAPI `/reset` and `/step` servers | This repo uses FastAPI today, but OpenEnv itself exposes an environment contract plus `EnvClient` transport |
| OpenEnv directly supports CTDE, MAPPO, GRPO, TRL, or RLlib | Those are trainer-side integrations outside OpenEnv |
| Oversight should be a Gym wrapper | In this project it should be implemented inside the simulator transition or a thin project-side interceptor |
| Stage 3 training can use live RSS injection | Live mode is demo-only; training should remain episodic and reproducible |
| The current repo is already a native OpenEnv environment | The current repo now includes a native OpenEnv-facing adapter while still keeping the session-oriented dashboard API |

This version is the correct mental model: OpenEnv is the execution shell for the simulator, while nearly all of the interesting multi-agent RL behavior is our own design sitting above that shell.
