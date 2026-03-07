# Backend Summary

This is the backend handoff for the frontend team.

## Plain-English State Model

There are four different layers of state:

1. `world.latent_state`
   Backend truth. Rewards and simulation logic use this.

2. `world.latent_events`
   Canonical hidden event chain. News, actions, asset damage, and oversight now create or update these events.

3. `world.actor_state`
   Lagged/public summary of the world.

4. `observations[agent_id]`
   What each entity actually sees. This can be partial, delayed, contradictory, and low-confidence.

The frontend should not treat those layers as interchangeable.

## Real Model Behavior

Each entity has a `model_bindings[agent_id]` object.

That tells you:

- which provider is configured
- which model is configured
- whether the binding is ready for inference
- which tools/actions the entity is allowed to use
- whether the entity is currently on real provider execution or heuristic fallback

Current behavior:

- if a provider binding is ready, the backend tries real provider inference first
- if that fails or returns an invalid action, the backend falls back explicitly to heuristic policy
- action metadata records whether the action came from `provider_inference` or `heuristic_fallback`

## Main Endpoints

Server file:

- [backend/src/trenches_env/server.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/server.py)

### Health And Capabilities

- `GET /healthz`
  Returns `{ "status": "ok" }`

- `GET /capabilities`
  Returns:
  - session/OpenEnv capability flags
  - CORS settings
  - per-entity `model_bindings`

Use this once at app startup.

### Session Lifecycle

- `POST /sessions`
  Creates a session.

- `POST /sessions/{session_id}/reset`
  Resets an existing session.

- `GET /sessions/{session_id}`
  Returns the latest `SessionState`.

- `POST /sessions/{session_id}/step`
  Advances one turn.

Request body:
- `actions: Record<agentId, AgentAction>`
- `external_signals: ExternalSignal[]`

Response:
- `StepSessionResponse`
  - `session`
  - `oversight`
  - `done`

### Live News And Reaction Timeline

- `POST /sessions/{session_id}/news`
  Injects public/news signals, lets the backend resolve entity reactions, steps the world, and returns the structured reaction entry for that news event.

Request body:
- `signals: ExternalSignal[]`
- `agent_ids?: string[]`

Notes:
- if `agent_ids` is omitted, all entities react
- if `agent_ids` is provided, only those entities are auto-resolved for that news event
- this still goes through the same env step path, so it stays aligned with OpenEnv behavior

Response:
- `IngestNewsResponse`
  - `session`
  - `oversight`
  - `reaction`
  - `done`

- `GET /sessions/{session_id}/reactions`
  Returns the rolling `reaction_log`.

Use these two endpoints for:

- incoming-news timeline
- “who reacted to what” UI
- live world-monitoring panels

### Provider Diagnostics

- `GET /sessions/{session_id}/providers/diagnostics`
  Returns per-entity provider runtime health and recent inference telemetry.

Important fields per entity:

- `status`
- `request_count`
- `success_count`
- `error_count`
- `consecutive_failures`
- `last_latency_ms`
- `avg_latency_ms`
- `last_success_at`
- `last_error_at`
- `last_error`

Use this for:

- provider health badges
- fallback warnings
- “model is unhealthy” operator panels
- debugging why an entity is on heuristic fallback

### Live Source Controls

- `POST /sessions/{session_id}/live`
  Enables or disables live mode.

- `POST /sessions/{session_id}/sources/refresh`
  Forces source refresh and rebuilds observations.

- `GET /sessions/{session_id}/sources/monitor`
  Returns source-health and delivery status per entity.

### Scenarios And Benchmarks

- `GET /scenarios`
  Returns seeded scenarios.

- `POST /benchmarks/run`
  Runs scenario benchmarks and returns scorecards.

### OpenEnv

Legacy tuple-style endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`

If `openenv-core` is installed, native OpenEnv is mounted at:

- `/openenv`

OpenEnv file:

- [backend/src/trenches_env/openenv_adapter.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/openenv_adapter.py)

## Main Schemas

Schema file:

- [backend/src/trenches_env/models.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/models.py)

### SessionState

Main top-level object for the frontend.

Important fields:

- `session_id`
- `world`
- `observations`
- `rewards`
- `model_bindings`
- `recent_traces`
- `action_log`
- `reaction_log`
- `live`
- `episode`

### WorldState

Important fields:

- `latent_state`
- `latent_events`
- `actor_state`
- `active_events`
- `asset_state`
- `coalition_graph`
- `risk_scores`
- `last_actions`

Important distinction:

- `latent_events` are canonical hidden events
- `active_events` are the public-facing projection of those latent events

### AgentObservation

Main entity-facing view.

Important fields:

- `decision_prompt`
- `available_actions`
- `available_data_sources`
- `strategic_state`
- `strategic_assets`
- `asset_alerts`
- `source_packets`
- `training_source_packets`
- `live_source_packets`
- `projection`

### ObservationProjection

This explains how messy the entity’s current view is.

Important fields:

- `mode`
- `worldview_reliability`
- `delayed_source_count`
- `contested_source_count`
- `contradiction_packet_count`
- `contradiction_topics`
- `obscured_metric_count`
- `notes`

Frontend rule:

Show this clearly. Do not present entity observations as perfect truth.

### EntityModelBinding

Per-entity provider/runtime config.

Important fields:

- `provider`
- `model_name`
- `configured`
- `ready_for_inference`
- `decision_mode`
- `supports_tool_calls`
- `supports_structured_output`
- `action_tools`
- `observation_tools`
- `notes`

### ProviderAgentDiagnostics

Per-entity runtime telemetry for provider-backed execution.

Important fields:

- `agent_id`
- `provider`
- `model_name`
- `configured`
- `ready_for_inference`
- `decision_mode`
- `status`
- `request_count`
- `success_count`
- `error_count`
- `consecutive_failures`
- `last_latency_ms`
- `avg_latency_ms`
- `last_success_at`
- `last_error_at`
- `last_error`

### ActionLogEntry

Per-action activity feed row.

Important fields:

- `turn`
- `actor`
- `action_type`
- `summary`
- `target`
- `reward_total`
- `metadata`

Use this for the entity activity log.

### ReactionLogEntry

Structured “public release -> entity reaction” object.

Important fields:

- `event_id`
- `turn`
- `source`
- `latent_event_ids`
- `signals`
- `actor_outcomes`
- `oversight_triggered`
- `tension_before`
- `tension_after`
- `market_stress_after`
- `oil_pressure_after`

This is the easiest object for a live news feed.

### Latent Events

The backend now treats event flow as first-class, not just metric movement.

Main schema:

- `LatentEvent`

Key fields:

- `event_id`
- `topic`
- `status`
- `severity`
- `visibility`
- `reliability`
- `origin`
- `affected_agents`
- `affected_assets`
- `started_at_turn`
- `last_updated_turn`
- `decay_rate`
- `linked_event_ids`
- `narratives`

What this means:

- scenarios can seed hidden events
- incoming news creates or updates hidden events
- entity actions create hidden events
- linked spillover events can be spawned
- public event feeds are projected from latent events
- source contradictions now key off latent events, not only metric heuristics

### ReactionActorOutcome

One entity’s response to one news event.

Important fields:

- `agent_id`
- `action`
- `reward_total`
- `decision_mode`

## What Is Good To Go

Backend pieces that are ready for frontend integration:

- session lifecycle
- live source monitoring
- latent truth vs public state split
- latent event engine and event-driven public projection
- contradiction-aware observation projection
- per-entity rewards
- per-entity action logging
- structured reaction logging for public/news events
- seeded scenarios
- benchmark runs
- provider bindings
- real provider execution with explicit fallback
- provider runtime diagnostics
- OpenEnv-compatible environment flow

## What Is Still Left

### Backend

1. Persist replay history.
   `recent_traces`, `action_log`, `reaction_log`, and latent event evolution are still rolling in-memory state, not durable history.

2. Deepen the latent event graph.
   The event engine now exists, but it can still be improved with stronger causal chains, event merging, event resolution rules, and richer cross-front propagation.

3. Add event-delta summaries.
   A compact backend-generated turn delta would make replay/debug views much easier to build.

4. Keep hardening provider execution.
   Retries and diagnostics now exist. The next step is richer classification for rate limits, timeout classes, and provider-specific retry traces.

5. Add a durable event archive or export path.
   There is still no persistent event timeline outside in-memory session state.

### Frontend

1. Build the app shell around:
   - `/capabilities`
   - `/scenarios`
   - `/sessions`
   - `/sessions/{id}`
   - `/sessions/{id}/step`
   - `/sessions/{id}/news`
   - `/sessions/{id}/reactions`
   - `/sessions/{id}/providers/diagnostics`
   - `/sessions/{id}/live`
   - `/sessions/{id}/sources/monitor`

2. Add entity cards that show:
   - projected state
   - reward total
   - provider readiness
   - provider health/latency
   - latest action
   - uncertainty/projection info

3. Add a live news/reaction timeline.
   Use `/sessions/{id}/news` for ingestion and `reaction_log` or `/sessions/{id}/reactions` for history.

4. Add latent event visibility to operator surfaces.
   Show:
   - key latent event topics
   - event severity
   - event visibility
   - linked spillovers

5. Add a source-health panel.
   Use `/sessions/{id}/sources/monitor`.

6. Add replay panels.
   Use `recent_traces`, `action_log`, `reaction_log`, and `world.latent_events`.

7. Make uncertainty visible.
   Show reliability, contradiction topics, delayed sources, and contested-source counts.

## Rule Of Thumb For Frontend

If the UI means:

- “what the entity believes” -> use `session.observations[agent_id]`
- “what the operator/debugger sees” -> use `session.world`
- “what hidden developments are driving the sim” -> use `session.world.latent_events`
- “what the backend can execute” -> use `session.model_bindings`
- “what just happened on a turn” -> use `session.action_log` and `session.recent_traces`
- “what public news triggered reactions” -> use `session.reaction_log`
