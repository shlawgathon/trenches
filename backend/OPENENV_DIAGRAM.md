# OpenEnv Input / Output Flow

```mermaid
graph TD
    subgraph "GRPO Trainer — TRL"
        TRAINER["GRPOTrainer"]
    end

    subgraph "OpenEnv Client — openenv_client.py"
        CLIENT["TrenchesEnvClient"]
        STEP_PAYLOAD["_step_payload()"]
        PARSE_RESULT["_parse_result()"]
        PARSE_STATE["_parse_state()"]
    end

    subgraph "OpenEnv Server — openenv_adapter.py"
        ENV["TrenchesOpenEnvEnvironment"]
        RESET["reset()"]
        STEP["step()"]
        STATE["state"]
    end

    subgraph "Simulator"
        SIM["FogOfWarDiplomacyEnv"]
    end

    TRAINER -->|"calls reset / step"| CLIENT

    %% --- reset flow ---
    CLIENT -->|"reset(training_agent, seed, ...)"| RESET
    RESET -->|"create_session()"| SIM

    %% --- step flow ---
    CLIENT -->|"TrenchesOpenEnvAction"| STEP_PAYLOAD
    STEP_PAYLOAD -->|"JSON payload"| STEP
    STEP -->|"step_session()"| SIM
    SIM -->|"SessionState"| ENV
    ENV -->|"JSON response"| PARSE_RESULT
    PARSE_RESULT -->|"StepResult"| CLIENT
    CLIENT -->|"StepResult(obs, reward, done)"| TRAINER

    %% --- state flow ---
    CLIENT -->|"get state"| PARSE_STATE
    STATE --> PARSE_STATE
```

## Input: `TrenchesOpenEnvAction`

| Field                          | Type                     | Description                                                  |
| ------------------------------ | ------------------------ | ------------------------------------------------------------ |
| `action`                       | `AgentAction`            | Single-agent action (convenience for single-policy training) |
| `actions`                      | `dict[str, AgentAction]` | Joint actions keyed by agent ID                              |
| `prediction`                   | `Prediction`             | Single-agent prediction                                      |
| `predictions`                  | `dict[str, Prediction]`  | Joint predictions keyed by agent ID                          |
| `external_signals`             | `list[ExternalSignal]`   | Live external data injected into the turn                    |
| `autofill_missing_with_policy` | `bool`                   | Auto-fill missing agents via policy inference                |
| `autofill_missing_with_hold`   | `bool`                   | Auto-fill missing agents with "hold" action                  |

## Output: `TrenchesOpenEnvObservation`

| Field                    | Type                              | Description                                                                 |
| ------------------------ | --------------------------------- | --------------------------------------------------------------------------- |
| `reward`                 | `float`                           | Scalar training reward for the focused agent                                |
| `done`                   | `bool`                            | Whether the episode has ended                                               |
| `session_id`             | `str`                             | Current session identifier                                                  |
| `training_agent`         | `str`                             | Which agent is being trained                                                |
| `turn`                   | `int`                             | Current turn number                                                         |
| `agent_observation`      | `AgentObservation`                | The training agent's observation                                            |
| `joint_observations`     | `dict[str, AgentObservation]`     | All agents' observations (if requested)                                     |
| `reward_breakdown`       | `RewardBreakdown`                 | Detailed reward components                                                  |
| `oversight`              | `OversightIntervention`           | Any oversight system interventions                                          |
| `historical_replay`      | `HistoricalReplayState`           | Replay state (ground truth hidden)                                          |
| `revealed_event`         | `HistoricalEvent`                 | The historical event revealed this turn                                     |
| `prediction_assessments` | `dict[str, PredictionAssessment]` | Accuracy of past predictions                                                |
| `done_reason`            | `str`                             | Why the episode ended (`tension_threshold`, `max_turns`, `replay_complete`) |

## State: `TrenchesOpenEnvState`

| Field               | Type                         | Description                       |
| ------------------- | ---------------------------- | --------------------------------- |
| `session_id`        | `str`                        | Session identifier                |
| `training_agent`    | `str`                        | Focused agent                     |
| `training_stage`    | `TrainingStage`              | Current training stage            |
| `max_turns`         | `int`                        | Episode length limit              |
| `live_enabled`      | `bool`                       | Whether live data ingestion is on |
| `reward_breakdowns` | `dict[str, RewardBreakdown]` | All agents' reward details        |
| `last_oversight`    | `OversightIntervention`      | Most recent oversight action      |
| `session`           | `SessionState`               | Full session snapshot             |
