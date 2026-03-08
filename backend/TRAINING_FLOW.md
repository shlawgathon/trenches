# Trenches OpenEnv Training Flow

## End-to-End Training Pipeline

```mermaid
flowchart TD
    subgraph CLI["training_cli.py — CLI Entry Point"]
        A["python -m trenches_env.training_cli<br/>--model-id · --training-agent · --replay-id<br/>--output-dir · --generation-backend"]
    end

    A -->|"Loads base model<br/>from HuggingFace Hub"| B["🤗 HuggingFace Model<br/>(e.g. Qwen/Qwen3-8B<br/>or sshleifer/tiny-gpt2)"]
    A -->|"Starts in-process"| C["FastAPI Backend<br/>server.py → uvicorn<br/>localhost:8000"]

    B --> D["GRPOTrainer<br/>(HF TRL)"]

    subgraph GRPO["GRPO Training Loop (per step)"]
        D -->|"1. Build prompts<br/>from base_prompt × train_size"| E["Prompt Dataset"]
        E -->|"2. rollout_func()"| F["OpenEnv Client<br/>POST /openenv/reset"]
        F -->|"Returns observation"| G["Render Grounded Prompt<br/>agent obs + historical brief<br/>+ strategic state + allowed actions"]
        G -->|"3. Generate completions"| H{Generation Backend?}
        H -->|transformers| I["transformers .generate()<br/>(CPU / Apple Silicon)"]
        H -->|vllm| J["vLLM inference<br/>(Linux CUDA GPU)"]
        I --> K["Parse JSON Output<br/>→ action + prediction"]
        J --> K
        K -->|"4. POST /openenv/step"| L["OpenEnv Environment<br/>openenv_adapter.py"]
    end

    subgraph ENV["OpenEnv Environment Boundary"]
        L --> M["FogOfWarDiplomacyEnv<br/>env.py"]
        M -->|"Load replay"| N["Replay Data<br/>synthetic_historical_replays/<br/>us_synthetic_seed_2025_2026.json"]
        M -->|"Apply action in sim"| O["Advance World State"]
        M -->|"Reveal next event"| P["Compare prediction<br/>vs actual event"]
        P --> Q["Compute Blended Reward<br/>action_reward + forecast_reward"]
    end

    Q -->|"5. Return env_reward<br/>+ forecast_reward"| D
    D -->|"6. GRPO policy update<br/>(gradient step)"| D

    D -->|"After max_steps"| R["trainer.save_model()"]
    R -->|"Writes checkpoint"| S["📁 output-dir/<br/>(--output-dir flag)"]

    R -->|"Optional"| T["Preview Rollouts<br/>--preview-samples N"]

    style CLI fill:#1a1a2e,stroke:#e94560,color:#fff
    style GRPO fill:#16213e,stroke:#0f3460,color:#fff
    style ENV fill:#0f3460,stroke:#533483,color:#fff
    style S fill:#e94560,stroke:#fff,color:#fff
```

## Model Storage Locations

| What                              | Where                                                    | Notes                                                                                                                                                                  |
| --------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Base model (source)**           | HuggingFace Hub or a local checkpoint directory          | Loaded at training start via `AutoTokenizer.from_pretrained(model_id)` + `GRPOTrainer(model=model_id)`                                                                 |
| **HF cache (downloaded weights)** | `~/.cache/huggingface/hub/`                              | Automatic HF cache, reused across runs                                                                                                                                 |
| **Trained checkpoint (output)**   | `--output-dir` flag                                      | Default: `trl-openenv-historical-replay/`. Examples: `backend/tmp-training-run/`, `backend/us-qwen-replay-run/`, `backend/us-vllm-replay-run/`                         |
| **Replay dataset**                | `backend/src/trenches_env/synthetic_historical_replays/` | Bundled JSON files (e.g. `us_synthetic_seed_2025_2026.json`). ⚠️ **All 6 replays are currently synthetic seed data** — replace with curated truth sets for production. |

## Per-Entity Model Pattern

```mermaid
flowchart LR
    subgraph Entities["6 Entity Models (1 per agent)"]
        US["us model<br/>📁 backend/us-run/"]
        ISR["israel model<br/>📁 backend/israel-run/"]
        IRN["iran model<br/>📁 backend/iran-run/"]
        HEZ["hezbollah model<br/>📁 backend/hezbollah-run/"]
        GULF["gulf model<br/>📁 backend/gulf-run/"]
        OVR["oversight model<br/>📁 backend/oversight-run/"]
    end

    subgraph Replays["Replay Datasets"]
        R1["us_synthetic_seed_2025_2026.json ✅"]
        R2["israel_synthetic_seed_2025_2026.json ✅"]
        R3["iran_synthetic_seed_2025_2026.json ✅"]
        R4["hezbollah_synthetic_seed_2025_2026.json ✅"]
        R5["gulf_synthetic_seed_2025_2026.json ✅"]
        R6["oversight_synthetic_seed_2025_2026.json ✅"]
    end

    R1 --> US
    R2 --> ISR
    R3 --> IRN
    R4 --> HEZ
    R5 --> GULF
    R6 --> OVR

    BASE["🤗 Base Model<br/>(shared starting point)"] --> US
    BASE --> ISR
    BASE --> IRN
    BASE --> HEZ
    BASE --> GULF
    BASE --> OVR

    style Entities fill:#16213e,stroke:#e94560,color:#fff
    style Replays fill:#0f3460,stroke:#533483,color:#fff
    style BASE fill:#e94560,stroke:#fff,color:#fff
```

> ✅ = implemented (all 6 replays are **synthetic seed data** for smoke-testing — replace with curated truth sets for production)

The first collection step for replacing those seeds is now:

```bash
python -m trenches_env.historical_collection_cli --training-agent us --window 2025 --window 2026
```

That collector writes replay JSON in the same schema as the bundled seed files plus raw article audit JSONL for review.

Saved output directories are reusable as future `--model-id` inputs and can be served with standard Hugging Face-compatible deployment tooling.

## Data Sources During Post-Training

All data is bundled in the repo. No external API calls during post-training.

```mermaid
flowchart LR
    subgraph Bundled["All in backend/src/trenches_env/"]
        REPLAY["synthetic_historical_replays/*.json<br/>10 synthetic events per entity<br/>(timestamps, topics, actors, severity, impacts)"]
        MANIFEST["source_manifest.json<br/>63KB intel briefings<br/>(public + private)"]
        AGENTS["agents.py<br/>6 agent profiles<br/>(role, intel focus, private intel)"]
        RL["rl.py<br/>Reward configs, allowed actions,<br/>strategic state baselines"]
    end

    REPLAY -->|"Replay timeline"| ENV["env.py builds<br/>observation"]
    MANIFEST -->|"Intel briefings"| ENV
    AGENTS -->|"Agent identity"| ENV
    RL -->|"Reward + actions"| ENV

    ENV --> PROMPT["Model sees:<br/>• decision prompt<br/>• historical brief<br/>• public/private brief<br/>• strategic state<br/>• allowed actions"]

    PROMPT --> MODEL["Model outputs JSON<br/>{action, prediction}"]
    MODEL --> SCORE["env.py scores:<br/>action reward + forecast reward<br/>→ GRPO update"]

    style Bundled fill:#0f3460,stroke:#533483,color:#fff
    style SCORE fill:#e94560,stroke:#fff,color:#fff
```

## Dual-Output Per Step

Each training step requires the model to produce **two outputs**:

```mermaid
flowchart LR
    MODEL["Entity Model"] --> ACTION["action<br/>{type, target, summary}"]
    MODEL --> PRED["prediction<br/>{topic, actor, target,<br/>severity, confidence,<br/>time_horizon, summary}"]

    ACTION -->|"Applied in simulator"| SIM["World State Update"]
    PRED -->|"Compared against<br/>revealed event"| SCORE["Forecast Reward"]

    SIM --> BLEND["Blended Reward<br/>= action_reward + forecast_reward"]
    SCORE --> BLEND

    style MODEL fill:#e94560,stroke:#fff,color:#fff
    style BLEND fill:#533483,stroke:#fff,color:#fff
```
