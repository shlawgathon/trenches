# How Post-Training Works: Steps, Data, and Rewards

## The Loop in One Sentence

Each GRPO step resets the environment at a random replay position, generates 16 completions, scores them against the real timeline, and updates the model to favor better responses.

## Steps vs Data

You have **10 replay events** and **100 GRPO steps**. They don't map 1:1.

```
Step 1:  reset() → random position in 10-event timeline
         → generate 16 completions
         → score all 16 against the next revealed event
         → GRPO update (reinforce good, suppress bad)

Step 2:  reset() → different random position
         → generate 16 completions → score → update

...

Step 100: same process
```

Across 100 steps × 16 generations = **1,600 rollouts** through those 10 events.
Each event seen ~160 times from different angles.

## What The Model Sees (Input)

Built by `_render_observation_prompt()` from the replay timeline:

```
You are training the us policy in the Trenches OpenEnv historical replay
environment. Return strict JSON only.

Training agent: us
Turn: 3

Historical brief:
- Commercial shipping insurers flag elevated Gulf transit risk near Hormuz.
- Washington reinforces maritime protection with Gulf partners.
- A renewed cross-border volley drives northern-front alerting.

Public brief:
- Gulf transit risk elevated near Hormuz.
- Coalition deconfliction messaging underway.

Private brief:
- Domestic approval is sensitive to prolonged escalation.
- Forward naval posture can deter but also spike market stress.

Strategic state:
- regional_access: 74.5
- shipping_security: 72.0
- domestic_support: 63.9
- force_posture: 76.0

Allowed actions: hold, negotiate, sanction, strike, defend, intel_query, mobilize, deceive
```

## What The Model Returns (Output)

```json
{
  "action": {
    "type": "sanction",
    "target": "iran",
    "summary": "Target proxy logistics channels to degrade corridor sustainment."
  },
  "prediction": {
    "topic": "domestic",
    "predicted_actor": "us",
    "predicted_target": "iran",
    "time_horizon_turns": 1,
    "expected_severity": "medium",
    "confidence": 0.7,
    "summary": "Washington will announce a sanctions package aimed at proxy sustainment.",
    "rationale": "Escalating Hormuz pressure creates political pressure for economic action."
  }
}
```

## Ground Truth (Revealed Event)

The environment reveals the next event from `us_synthetic_seed_2025_2026.json`:

```json
{
  "event_id": "evt-2025-04-us-sanctions-package",
  "timestamp": "2025-04-22T12:00:00Z",
  "topic": "domestic",
  "actors": ["us"],
  "targets": ["iran"],
  "severity": "medium",
  "summary": "Washington rolls out a coordinated sanctions package aimed at procurement and logistics channels linked to proxy sustainment."
}
```

## Scoring

```
action_reward:   +0.55  (sanction aligns with us policy at 0.55 per rl.py)
forecast_reward: +0.82  (topic ✅ actor ✅ target ✅ severity ✅ confidence ✅)
─────────────────────────
total_reward:    +1.37  → fed back to GRPO
```

## Where Each Piece Comes From

| Data              | Source File                 | What It Provides                                                          |
| ----------------- | --------------------------- | ------------------------------------------------------------------------- |
| Replay events     | `historical_replays/*.json` | 10 historical events (timestamp, topic, actors, severity, impact)         |
| Intel briefings   | `source_manifest.json`      | Public + private brief items                                              |
| Agent identity    | `agents.py`                 | Role, intel focus, private intel baseline                                 |
| Reward config     | `rl.py`                     | Allowed actions, action alignment scores, state baselines, metric targets |
| Environment logic | `env.py`                    | Builds observation, applies actions, scores predictions, computes rewards |
| Training loop     | `training_cli.py`           | Connects model ↔ environment via GRPO rollouts                            |
| OpenEnv boundary  | `openenv_adapter.py`        | reset/step interface between TRL and the simulator                        |

## Key Numbers

| Metric                        | Value | Formula                                |
| ----------------------------- | ----- | -------------------------------------- |
| Total rollouts per entity     | 1,600 | 100 steps × 16 generations             |
| Times each event is seen      | ~160  | 1,600 ÷ 10 events                      |
| Effective batch size          | 8     | batch_size(1) × grad_accum(8)          |
| Completions compared per step | 16    | GRPO ranks them relative to each other |
