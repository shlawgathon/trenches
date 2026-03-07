# Improvements

This document captures the three highest-leverage improvements for the simulator after the MVP. Each one is large enough to materially change training quality, realism, and operator usefulness.

## 1. Hidden-World Engine

### Objective
Replace the current mostly surface-level world update loop with a canonical latent world state that each entity only perceives through partial, noisy, delayed, and bias-shaped observations.

### Why This Matters
Right now the simulator has actor-specific rewards and source bundles, but the underlying world can still be too direct and too legible. That makes the task easier than the real problem and increases the risk of reward gaming. A hidden-world engine forces the policies to reason under uncertainty instead of reacting to a clean omniscient state.

### What To Build
- A canonical state graph for logistics, infrastructure integrity, domestic resilience, proxy health, coalition confidence, chokepoint access, and military readiness.
- Observation projection layers that transform latent state into actor-specific intel packets with source lag, missingness, confidence, and bias.
- Source reliability and deception mechanics so models must reason about contradictory or manipulated evidence.
- Damage persistence so strikes and mobilization affect later turns instead of only the immediate step.
- Cross-front coupling so a Gulf shock can change Israeli decision quality, US domestic resilience, or Iranian leverage.

### Implementation Shape
- Keep one hidden state store per session.
- Each step applies actions to the hidden state first.
- Each source/tool reads from that hidden state through a projection function.
- Each actor receives only its projection, not the full state.
- The viewer dashboard can still render a privileged map and replay, but that view remains unavailable to the models.

### Success Criteria
- Agents can no longer trivially infer the full world from public state alone.
- Different source bundles produce meaningfully different beliefs for the same event.
- Training runs become less brittle and less prone to one-step exploitation.

## 2. Evaluation Harness And Curriculum

### Objective
Turn training from open-ended sandbox play into measurable policy development with regression protection, seeded scenarios, and staged learning.

### Why This Matters
The project will improve much faster once performance is measured against doctrine-specific benchmarks rather than by whether the simulator runs. Without evaluation, reward shaping tends to drift and policies often learn artifacts instead of strategy.

### What To Build
- Seeded scenario packs for shipping crises, border flare-ups, corridor interdiction, domestic unrest, and coalition fracture.
- Policy scorecards per entity with doctrine-aligned metrics.
- Reward-gaming checks that detect obviously degenerate policies.
- Self-play and adversarial evaluation between versions.
- Curriculum stages that move from narrow tactical cases to full multi-front regional crises.
- Unsloth-based per-entity post-training loops so each actor can be adapted efficiently without retraining the full stack.

### Implementation Shape
- Add fixed seeds and replayable scenario fixtures.
- Run benchmark suites after policy changes.
- Store reward decomposition and trace outputs for each benchmark run.
- Train smaller doctrine-specific adapters first, then graduate them into the full environment.

### Success Criteria
- Every entity has a stable benchmark suite.
- Policy regressions are visible in CI or scheduled evaluation runs.
- New reward changes can be justified with measurable gains, not intuition alone.

## 3. Command Dashboard And Replay Observability

### Objective
Promote the frontend from a session viewer to a real command-and-control observability layer for simulation, training, and debugging.

### Why This Matters
If a run collapses, you need to know why immediately. A polished dashboard is not just presentation; it is the main debugging surface for understanding model behavior, source health, intervention timing, and reward dynamics.

### What To Build
- A unified operational map showing entities, geolocated assets, fronts, chokepoints, and coalition links.
- A per-agent monitoring deck for model status, source health, reward decomposition, and recent actions.
- Step-by-step replay with diff views between timesteps.
- Source-ingestion health views so failed feeds are visible.
- Oversight visibility showing when intervention risk crossed thresholds and what triggered it.
- Run comparison views so two policies or seeds can be compared side by side.

### Implementation Shape
- Keep the map viewer privileged for the human operator only.
- Feed the dashboard from structured session snapshots, not hand-built UI-only state.
- Surface both raw metrics and human-readable summaries.
- Preserve replay history so failures can be audited after the run ends.

### Success Criteria
- A user can explain a bad decision by tracing source inputs, action choice, and reward terms.
- Replay is fast enough to inspect long runs without digging through logs.
- The dashboard is useful for both live demos and offline training analysis.
