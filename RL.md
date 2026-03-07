# RL.md: Reinforcement Learning Mechanics in Fog of War Diplomacy Simulator

This document provides a full breakdown of the reinforcement learning (RL) aspects in the Fog of War Diplomacy Simulator, an OpenEnv-based multi-agent environment simulating the 2026 US-Israel-Iran geopolitical crisis. It covers how OpenEnv functions as the RL framework, the reward system (including the general formula, per-entity customizations, and oversight modulation), training workflows, and evaluation. The design emphasizes sparse, delayed rewards to encourage long-horizon planning, emergent behaviors like coalition formation and deception, and scalable oversight aligned with the Fleet AI sub-theme.

RL here post-trains LLM agents to optimize entity-specific goals in a partially observable, multi-agent setting using a Centralized Training, Decentralized Execution (CTDE) framework. Rewards are the key signal, guiding agents toward de-escalation without collapsing into endless conflict.

---

## Overview of OpenEnv and RL Integration

### What is OpenEnv?

OpenEnv is an open-source RL library (built by Meta-PyTorch, integrated with Hugging Face ecosystems) designed for agentic environments, particularly those involving LLMs. It extends Gymnasium's core API (`reset()`, `step()`, observation/action spaces) with features for containerization, reproducibility, and distributed training. Key aspects relevant to our simulator:

**Gymnasium-Compatible Interfaces**:
- `reset(seed=None)`: Initializes the environment state (e.g., tension at 50%, random hidden incentives) and returns initial observations (dict per agent) and info.
- `step(actions)`: Takes a dict of actions (text-based from LLMs), updates the world state stochastically, computes rewards (dict per agent), and returns new observations, rewards, done/truncated flags, and info.
- Spaces: Multi-agent dicts (e.g., `observation_space = gym.spaces.Dict({...})`) for partial observability — each agent receives a personalized, noisy view of the state.

**Containerization and Server Model**:
- Environments run as isolated Docker containers with a FastAPI server exposing HTTP endpoints (`/reset`, `/step`).
- A client (`openenv.Client('http://localhost:8000')`) interacts with the containerized env, enabling distributed training across machines or cloud (e.g., Kubernetes).
- Ensures reproducibility for hackathon judging — upload to Hugging Face Hub as a Docker Space, where judges can `from_hub()` and run episodes.

**Multi-Agent Support**:
- Handles 6 agents natively: actions, observations, and rewards as dicts keyed by agent ID (e.g., `{"US": action_text}`).
- Partial observability: agents see only their slice of the state, forcing inference about others' behaviors from observable actions.

**RL Workflow Integration**:
- Compatible with TRL (Transformer Reinforcement Learning), RLlib for multi-agent algorithms, or GRPO for group-relative preference optimization.
- Training Loop: Client resets env → Agents (LLMs) generate actions from observations → Step env → Collect rewards/trajectories → Update policies.
- Episodes: Long-horizon (up to 1000+ turns) with sparse rewards; terminal on high tension (≥ 100) or max turns.
- **Important**: "Live Mode" (infinite sessions, real-time RSS injection) is an *inference/demo mode only* and does not participate in the training loop. All training operates over episodic rollouts with well-defined boundaries.

---

## Reward System: General Principles

Rewards are **sparse and delayed** to push beyond shallow reasoning — agents receive signals only on meaningful milestones (e.g., coalition stability over 50 turns), encouraging long-horizon planning.

**Key design constraints**:
- **Computation Timing**: In `step()`, after action processing and state updates.
- **Component Normalization**: Each reward component is independently normalized to `[-1, 1]` *before* weighting, ensuring the weighted sum is also bounded without relying on post-hoc clipping.
- **Shared vs. Individual**: Coalition bonuses split among allies to foster cooperation.
- **Oversight as Environment Transition**: Oversight intervenes by modifying the *next state* (e.g., forcing a re-action or injecting a corrective event), not by rescaling reward signals. Rescaling rewards teaches agents to hide risky behavior from oversight rather than avoid it.
- **No Oracle Information at Reward Time**: Reward signals are computed only from observable quantities available at execution time. Hidden incentives are not used as a direct reward signal (see $B_t$ correction below).

### Core Reward Formula

Per timestep $t$, for each agent $i$:

$$r_t^i = w_1 \cdot \hat{C}_t^i + w_2 \cdot \hat{E}_t^i + w_3 \cdot \hat{M}_t^i + w_4 \cdot \hat{B}_t^i$$

where each $\hat{\cdot}$ denotes a component normalized to $[-1, 1]$ independently before aggregation, and weights $w_i \geq 0$, $\sum w_i = 1$.

**Components**:

- $C_t$: **Coalition Stability** — $\frac{\#\text{allied agents} - \#\text{betrayals}}{\#\text{total agents}}$ — encourages durable alliances; delayed signal emitted every 10 turns, zero otherwise.

- $E_t$: **Escalation Penalty** — $-\text{EMA}_\alpha(\text{tension}_t)$, where EMA is an exponential moving average with decay $\alpha = 0.05$. This replaces the original sigmoid-on-delta formulation, which produced near-zero signal during gradual tension build-up (the most dangerous trajectory). The EMA penalizes *sustained high tension*, not just sudden spikes.

  ```python
  self.ema_tension = alpha * self.tension_level + (1 - alpha) * self.ema_tension
  e_t = -self.ema_tension / 100.0  # normalize to [-1, 0]
  ```

- $M_t$: **Market/Economic Gain** — $\frac{\Delta\text{oil stability} + \Delta\text{sanctions relief}}{2}$ — tied to observable news events; positive for averting economic shocks.

- $B_t$: **Behavioral Consistency Bonus** — replaces the original "Belief Alignment Bonus," which required oracle access to hidden incentives at reward time, creating an epistemically invalid training signal. Instead, $B_t$ measures consistency between an agent's *stated reasoning* (from its LLM output) and its *observed actions*, detectable from the environment trace:

  $$B_t = 1 - \frac{\|\text{stated\_intent}_t - \text{action\_vector}_t\|_1}{D_{\max}}$$

  where `action_vector` is a learned embedding of the taken action and `stated_intent` is an embedding of the agent's reasoning trace. This rewards coherent agency without oracle access. Theory-of-mind training is handled separately via self-play curriculum (see Training Flow).

**Implementation**:

```python
def _normalize(self, value: float, lo: float, hi: float) -> float:
    """Map value in [lo, hi] to [-1, 1]."""
    return 2.0 * (value - lo) / (hi - lo) - 1.0

def _compute_rewards(self) -> dict[str, float]:
    rewards = {}
    for agent in self.agents:
        # Coalition stability: normalized to [-1, 1] by construction
        c_t = (len(self.coalitions[agent]) - self.betrayals[agent]) / len(self.agents)

        # Escalation: EMA-based, normalized to [-1, 0]
        alpha = 0.05
        self.ema_tension[agent] = (
            alpha * self.tension_level + (1 - alpha) * self.ema_tension[agent]
        )
        e_t = -self.ema_tension[agent] / 100.0

        # Market gain: clamp to [-1, 1]
        m_t = np.clip((self.oil_delta + self.sanctions_delta) / 2, -1, 1)

        # Behavioral consistency: from observable trace only
        b_t = self._behavioral_consistency(agent)

        components = np.array([c_t, e_t, m_t, b_t])
        # Each component already in [-1, 1]; weighted sum also in [-1, 1]
        r_t = float(np.dot(WEIGHTS[agent], components))
        rewards[agent] = r_t

    return rewards
```

---

### Oversight: Environment-Level Intervention

Oversight operates as a **Gymnasium wrapper** that intercepts `step()` *before* rewards are returned, and may modify the transition rather than the reward. This is the architecturally correct location for oversight in an RL system: it affects the environment dynamics the agent must learn to navigate, not the reward signal (which would incentivize concealment).

```python
class OversightWrapper(gym.Wrapper):
    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        risk = self._calculate_risk(obs, actions, info)
        info["oversight_risk"] = risk
        if risk > 0.5:
            # Intervene: inject corrective diplomatic event into next state
            obs = self._inject_corrective_event(obs)
            info["oversight_intervened"] = True
            # Do NOT modify rewards — intervention changes the world, not the signal
        return obs, rewards, terminated, truncated, info
```

**Risk Calculation**:

The risk score is computed via a Bayesian belief update over a learned transition model:

$$R(a_t) = \sum_{s'} B'(s') \cdot U(s', a_t) \cdot P(\text{escalate} \mid s', a_t)$$

where:
- $B'(s') = \eta \sum_s P(o_t \mid a_t, s') \cdot T(s, a_t, s') \cdot B(s)$ — Bayesian belief update.
- $T(s, a_t, s')$ is estimated from a *learned world model* trained on rollout data (not assumed known), updated via maximum likelihood over observed transitions.
- $P(\text{escalate} \mid s', a_t)$ is a sigmoid on the tension delta predicted by the world model.
- $U(s', a_t) = -10$ for states projecting prolonged conflict (tension > 70 for > 20 turns).

This grounds the oversight formula in learnable quantities rather than assuming access to the true transition dynamics.

---

## Rewards Breakdown Per Entity

Weights reflect entity identities. These values are **initializations for hyperparameter search** (e.g., via Optuna), not hand-tuned finals. All ablations should compare against a uniform-weight baseline.

| Agent | $w_1$ (Coalition) | $w_2$ (Escalation) | $w_3$ (Market) | $w_4$ (Behavioral) | Primary motivation |
|---|---|---|---|---|---|
| US | 0.25 | 0.25 | 0.35 | 0.15 | Economic/poll dominance |
| Israel | 0.30 | 0.35 | 0.20 | 0.15 | Security, threat elimination |
| Iran | 0.20 | 0.40 | 0.25 | 0.15 | Survival, attrition |
| Hezbollah | 0.25 | 0.30 | 0.20 | 0.25 | Asymmetric opportunism |
| Gulf Coalition | 0.25 | 0.20 | 0.40 | 0.15 | Neutral economic leverage |
| Oversight | 0.10 | 0.50 | 0.10 | 0.30 | Global de-escalation |

Oversight's reward is tied to the *aggregate* escalation reduction across all agents, giving it a global rather than entity-local objective.

---

## Training and Evaluation

### Multi-Agent Credit Assignment (CTDE)

Standard independent PPO cannot disentangle individual agent contributions to joint outcomes in a cooperative-competitive setting. This simulator uses **Centralized Training, Decentralized Execution (CTDE)**:

- **Centralized Critic**: During training, a shared critic $V_\phi(s_{\text{global}}, \{a_i\})$ takes the full global state and all agents' actions as input. This resolves the non-stationarity problem (each agent's policy appears to change from another agent's perspective during training).
- **Decentralized Execution**: At inference, each agent acts using only its local observation $o_i$. The centralized critic is discarded.
- **Implementation**: Use MAPPO (Multi-Agent PPO) from the `epymarl` library, or implement manually with a shared critic network updated alongside per-agent actor networks.

### Algorithm Choice: GRPO over PPO for Sparse Rewards

Standard PPO with GAE (Generalized Advantage Estimation) has high variance on sparse, long-horizon rewards — the credit assignment signal degrades over 100+ turns. This simulator uses **GRPO (Group Relative Policy Optimization)**:

- Sample a *group* of $G$ rollouts per agent per update step.
- Advantage is computed relative to the group's mean return: $A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}$.
- This is variance-reduced and doesn't require a value function baseline, which is difficult to learn under sparsity.
- Compatible with TRL's `GRPOTrainer`.

PPO remains acceptable for dense-reward ablations or short-horizon curriculum stages (< 100 turns).

### Training Flow

1. **Initialization**: Load LLM agents with entity-specific system prompts encoding identities; initialize env via `openenv.Client`.
2. **Curriculum**:
    - Stage 1 (turns 0–100): Dense rewards enabled (reduce sparsity), no fog of war. Agents learn basic action-reward associations.
    - Stage 2 (turns 0–500): Sparse rewards, partial observability. Fog of war enabled.
    - Stage 3 (turns 0–1000+): Full sparsity, live RSS event injection, oversight wrapper active.
3. **Episode Loop**: `reset()` → generate actions (LLM conditioned on observation) → `step()` → collect `(obs, action, reward, next_obs, done)` tuples into replay buffer.
4. **Policy Update**: Every $N$ episodes, update actors via GRPO; update centralized critic via MSE on bootstrapped returns.
5. **Theory-of-Mind Curriculum**: Periodically freeze all agents except one; the unfrozen agent is trained against fixed opponents, then roles rotate. This produces agents that can model fixed policies before tackling co-adaptive opponents.

### Evaluation

| Metric | Definition | Target |
|---|---|---|
| Avg reward/episode | Mean $\sum_t r_t$ over last 100 episodes | Monotonically increasing |
| De-escalation rate | % of episodes ending with tension < 30 | > 60% |
| Oversight intervention rate | Interventions per episode | Decreasing over training |
| Behavioral consistency | Mean $B_t$ across agents | > 0.7 |
| Coalition durability | Avg turns before first betrayal | > 200 |

**Known Challenges**:
- Non-stationarity from live RSS data: mitigate with prioritized experience replay and periodic policy snapshots.
- Reward hacking: monitor for degenerate strategies (e.g., agents achieving high coalition score by never taking actions). Add entropy bonuses to actor losses if action diversity collapses.

---

## Summary of Design Corrections

| Original Issue | Correction |
|---|---|
| $B_t$ used hidden oracle state | Replaced with behavioral consistency from observable traces |
| Escalation penalty blind to slow drift | Replaced sigmoid-on-delta with EMA of tension level |
| Weights summing to 1 didn't guarantee bounded rewards | Each component independently normalized before weighting |
| Oversight rescaled rewards (incentivizes concealment) | Oversight now injects corrective state transitions via wrapper |
| `T(s, a, s')` assumed known | Estimated from a learned world model updated on rollout data |
| PPO on 1000-turn sparse rewards | Replaced with GRPO; PPO retained only for dense curriculum stages |
| Multi-agent credit assignment unaddressed | CTDE with MAPPO centralized critic |
| Live mode conflated with training | Live mode is inference/demo only; training is strictly episodic |
| Python bug: loop variable mutation | Fixed — dict values updated correctly |
| Weights presented as final | Reframed as initialization for Optuna hyperparameter search |