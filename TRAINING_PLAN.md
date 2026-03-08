# Trenches Training Plan

This document is the working plan for the historical prediction training setup.

## Goal

Train six separate entity models in the same OpenEnv-backed simulator so they do two things at each turn:

1. choose an action
2. predict what will happen next

The core idea is:

- the environment replays a real historical event window
- each model only sees information available up to that point in time
- each model generates a predicted future timeline
- the environment later reveals what actually happened
- reward is based partly on whether the model predicted correctly

Target training window:

- 2025
- 2026

## Intended Training Shape

Two timelines exist at once:

1. `ground_truth_timeline`
   The real historical sequence of events.

2. `predicted_timeline`
   What the entity believed would happen next, based only on available information at that turn.

The environment reward should compare the second timeline against the first.

## Why OpenEnv Is The Right Boundary

OpenEnv is the environment interface, not the trainer itself.

That is exactly what we need:

- `reset()` starts a historical replay episode at a chosen point
- `step()` accepts an entity output
- the env advances time
- the env computes reward from action quality and prediction quality

Training should happen outside the backend with something like Hugging Face TRL.

## What Exists Already

The current backend already has:

- an OpenEnv environment boundary
- session and step logic
- per-entity observations
- per-entity rewards
- latent state
- latent events
- belief state
- source projection
- scenario and benchmark support
- a structured `Prediction` schema
- prediction storage and scoring in session state
- replay mode driven by historical event timestamps
- a bundled seed replay dataset for a first `us` proof run
- a replay-aware TRL/OpenEnv CLI training loop

## What Is Missing

The backend does not yet have:

- a six-agent training runner
- a larger curated truth dataset beyond the bundled seed replay
- a proper evaluation report for prediction quality
- baselines and train/eval split reporting

## Planned Implementation Order

### Phase 1: Historical Replay Foundation

1. Define a normalized historical event schema.
2. Build a replay dataset for selected 2025-2026 events.
3. Add historical replay mode to the backend environment.
4. Ensure agents only see information available before each replay timestamp.

### Phase 2: Prediction Contract

1. Add a structured `Prediction` object for each agent.
2. Extend agent outputs so a turn can include:
   - `action`
   - `prediction`
3. Store prediction history in session state.

### Phase 3: Reward Logic

1. Add reward terms for:
   - correct topic
   - correct actor
   - correct target
   - correct timing window
   - correct severity band
   - confidence calibration
2. Penalize:
   - confident false predictions
   - vague predictions
   - repeated contradiction with real history
3. Exclude fake/manual events from training reward.

### Phase 4: Training Loop

1. Train one entity first.
2. Use OpenEnv + HF TRL.
3. Prove a working historical replay training loop.
4. Scale to six entity-specific models.

### Phase 5: Evaluation

1. Build evaluation metrics for forecast quality.
2. Compare against simple baselines.
3. Separate train and eval windows.
4. Report before/after performance.

## Recommended Minimal Event Schema

Each historical event should have:

- `event_id`
- `timestamp`
- `topic`
- `region`
- `actors`
- `targets`
- `severity`
- `summary`
- `source_type`
- `confirmed`
- `tags`

## Recommended Prediction Schema

Each prediction should have:

- `prediction_id`
- `agent_id`
- `turn`
- `timestamp`
- `topic`
- `predicted_actor`
- `predicted_target`
- `time_horizon_turns`
- `expected_severity`
- `confidence`
- `summary`
- `rationale`

## Critical Design Rules

1. No leakage.
   The model must never see future information.

2. Real events and fake events must be separated.
   Manual events can drive behavior but must not drive training reward.

3. Action and prediction should remain separate outputs.
   Mixing them into one blob will make both training and debugging worse.

4. Train one entity first before scaling to six.
   Prove the loop on one actor before multiplying complexity.

5. Evaluate against baselines.
   Otherwise there is no evidence the training helped.

## Suggested First Entity

Start with:

- `us`

Why:

- broad observation surface
- strong strategic tradeoffs
- likely easiest to benchmark against known 2025-2026 developments

## Known Future Work

After the first working replay-training loop:

- train all six entities
- compare model families
- add branch evaluation for counterfactual timelines
- add replay UI for predicted vs actual timeline alignment

## Working Status

Current status:

- first working historical replay loop implemented for `us`
- OpenEnv step accepts separate `action` and `prediction`
- forecast reward is blended into entity reward on replay steps
- TRL CLI training path is implemented and smoke-tested end to end
- multi-entity scaling and evaluation still pending

This file should be updated as the forecasting/replay training system is built.
