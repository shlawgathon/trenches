# Trenches Fine-Tuning Steps

This document explains, in presentation-ready form, how the team fine-tuned the six Trenches models. It is grounded in the checked-in training docs, the historical collection pipeline, and the training/serving scripts currently in this repo.

## 1. The Six-Model Setup

Trenches was not trained as one generic geopolitical model. The architecture was intentionally split into six separate policies:

- `us`
- `israel`
- `iran`
- `hezbollah`
- `gulf`
- `oversight`

The first five are entity-specific actor models. Each one is meant to learn the doctrine, incentives, and information asymmetries of a specific actor in the simulation. The sixth, `oversight`, is the control model that watches the overall system for runaway escalation and intervention conditions.

That split matters because the simulator itself is asymmetric:

- each entity sees a different observation
- each entity has different strategic metrics
- each entity has different allowed actions
- each entity is rewarded against a different strategic objective set

So the training plan followed that same structure: one training run per entity, one replay stream per entity, one checkpoint per entity.

## 2. Why Qwen 8B Was Chosen

The default base model converged on `Qwen/Qwen3-8B`.

It was chosen because it fit the project constraints unusually well:

- it was available on Hugging Face, which made it easy to use as a common starting point
- it was a good early-to-mid 2025 class model for the behavior the team wanted to capture
- it was strong enough to produce structured outputs instead of only loose prose
- it was still realistic to fine-tune six separate models rather than one giant shared policy
- it was small enough to fit within realistic compute-credit limits while still supporting both local smoke tests and more serious GPU-backed post-training

In other words, Qwen 8B sat in the middle of the tradeoff curve: strong enough to act and predict in a structured simulator, but still cheap enough to run six entity-specific post-training jobs.

## 3. Data Path: From RSS/GDELT Collection to Replay Training Data

The project did not invent a separate training-only dataset. It started from the same source universe that powers the product.

### 3.1 Starting Point: The Runtime Source Manifest

The collection path begins with the checked-in `source_manifest.json`, which already defines the aligned sources for each entity. Those sources include:

- RSS feeds
- official government and ministry sources
- wire services
- regional publications
- structured feeds such as ACLED/GDELT-style sources

That source manifest is the bridge between product behavior and training data. The team reused the live/source-aligned source definitions instead of building a disconnected offline corpus.

### 3.2 Historical Collection Window

For the historical 2025 run, the collector uses the window:

- `2025-01-01`
- through `2026-01-01`

That is the important presentation window for the first real replay pass. In the checked-in collector, the `2025` window is explicitly defined as `2025-01-01 -> 2026-01-01`.

Operationally, the team treated this as a daily historical-news pass:

- collect the relevant daily news flow
- chunk it by day and by entity relevance
- feed those chunks into entity-specific replay-building and post-training

### 3.3 Collection Flow

The historical data path works like this:

1. Start from the entity-aligned sources in the manifest.
2. Derive allowlisted historical domains from those sources.
3. Query the GDELT DOC API month by month across the target window.
4. Save raw article audit records for provenance.
5. Convert those articles into replay events using the same schema the trainer already consumes.
6. Keep the formatting aligned with the live runtime payload shape so training examples resemble real deployment-time observations.

The project therefore used RSS/source alignment as the source-selection layer, and GDELT as the historical retrieval layer.

### 3.4 What Was Saved

Each collection run writes two outputs:

- replay JSON in `backend/src/trenches_env/historical_replays/`
- raw article audit JSONL in `backend/tmp-historical-raw/`

The replay JSON is what the trainer consumes. The JSONL files are the review trail so the team can inspect where events came from before using them for production-grade post-training.

## 4. Formatting Into Live-Runtime-Like Structured Examples

The training examples were not formatted as generic instruction tuning rows. They were shaped to look like the live runtime.

That means the model prompt is built from the same categories of information the live simulator uses:

- decision prompt
- historical brief
- public brief
- private brief
- strategic state
- allowed actions

The result is a grounded, simulator-native prompt rather than a plain text article summary.

### 4.1 What the Model Sees

At each step, the model receives a structured observation that mirrors the live environment:

- current turn
- actor-specific observation
- public and private intel
- strategic metrics
- action constraints

This is the key design choice: training examples look like runtime state, not like a raw news corpus.

That was intentional. The team wanted the models to be good at the exact typed payloads they would see live, not only at generic text completion. In practice, this made the post-training feel closer to OpenEnv-style RL alignment than to plain supervised finetuning.

### 4.2 What the Model Must Return

Each completion is expected to return strict structured JSON with two outputs:

1. `action`
2. `prediction`

`action` answers: what should this entity do now?

`prediction` answers: what does this entity think happens next?

That dual-output format is what lets the project train both:

- policy quality
- forecast quality

inside the same loop.

## 5. Replay / OpenEnv / GRPO-Style Post-Training Flow

The actual post-training loop is replay-based and OpenEnv-backed.

### 5.1 Core Loop

The loop works like this:

1. Reset the OpenEnv-compatible Trenches environment at a replay position.
2. Render the observation into a grounded prompt.
3. Generate multiple completions from the current entity model.
4. Parse each completion into `action + prediction`.
5. Step the environment one turn.
6. Reveal the next replay event.
7. Score both the action and the prediction.
8. Feed the blended reward back into GRPO.
9. Update the policy.

This is not standard next-token fine-tuning over text alone. It is policy post-training against simulator feedback.

### 5.2 Why OpenEnv Matters

OpenEnv gives the project a clean training boundary:

- `reset()`
- `step()`
- observation in
- reward out

That matters because the product backend is richer than the training boundary. The session API runs the full simulator, but the OpenEnv layer gives TRL/GRPO a stable interface for rollouts.

### 5.3 Why GRPO Was a Good Fit

GRPO fits this setup because the trainer compares multiple completions from the same prompt and reinforces the better ones.

In Trenches, "better" means:

- better aligned action choice
- better next-event prediction
- better strategic fit for that entity

So the reward is not only "did the prose look plausible?" It is "did the action help the entity, and did the forecast match what the replay revealed?"

## 6. How the Six Fine-Tunes Were Split

The same base process was repeated six times, with different replay targets and different identity constraints.

| Model | Training role |
| --- | --- |
| `us` | alliance management, shipping security, domestic support, force posture |
| `israel` | homeland defense, northern deterrence, strike readiness |
| `iran` | regime survival, proxy corridor management, Hormuz leverage |
| `hezbollah` | survivability, asymmetric escalation, logistics depth |
| `gulf` | infrastructure security, investor confidence, shipping continuity |
| `oversight` | escalation monitoring, intervention legitimacy, trace clarity |

For the five actor models, training was entity-specific: each run taught one actor to become a better version of itself inside the simulator.

## 7. Why Oversight Was Treated Differently

Oversight is the exception.

The actor models were trained on their own slice of the world. Oversight was conceptually trained across the full system because its job is not to behave like one country or proxy. Its job is to monitor all actors, detect escalation patterns, and decide whether intervention is warranted.

So the training split was:

- actor models: entity-specific
- oversight model: system-level, cross-entity reasoning

That is the right match to the product design. Oversight is the only model whose role depends on the behavior of all five operational actors at once.

## 8. Infrastructure Evolution

The infrastructure changed over time. The final process was not the first process.

### Phase 1: Hugging Face as the Starting Point

Hugging Face mattered in two different ways:

- it provided the base `Qwen/Qwen3-8B` model
- it was used for early provider/serving experiments
- it later became the registry where the finetuned checkpoints were uploaded under `@AlazarM`

The team also tried using Hugging Face Spaces in this phase. That path turned out to be hard to operate and more expensive than expected, so it was not kept as the long-term serving direction.

The early Hugging Face and local experiments still proved that:

- the replay schema worked
- the OpenEnv adapter worked
- the trainer could produce structured `action + prediction` outputs

### Phase 2: Better Replay Data and Stronger Base Models

Once the loop worked, the team moved toward:

- real historical replay generation
- Qwen 8B as the shared starting model
- six separate entity runs

This is where the project stopped being only a simulator prototype and became a real post-training system.

### Phase 3: Thunder, Cloudflare, and NorthFlank Detours

The team did not move straight from Hugging Face to the final system.

There were several infrastructure detours:

- ThunderCompute was tried for backend infrastructure
- when that path became unstable, Cloudflare tunnels were used as a workaround
- NorthFlank was later tried as a place to pull and host the trained models

Those paths did not hold:

- ThunderCompute had instance issues
- Cloudflare tunnels added operational friction
- NorthFlank did not provide the compute shape needed for six active inference models

### Phase 4: Modal + vLLM for Serious Runs

The checked-in Modal script shows the next step in the evolution:

- use Modal for GPU-backed training jobs
- use vLLM in server mode for rollout generation
- keep one training run per entity
- write checkpoints out per entity

That stack became the practical route because it combined:

- a stable base-model source from Hugging Face
- scalable rollout generation through vLLM
- manageable GPU orchestration through Modal

In the final workable setup:

- Hugging Face remained the model registry and base-model source
- Modal became the practical training/inference execution layer
- vLLM became the efficient rollout-serving layer for GRPO training
- the final inference system ran six 8B models, one per entity, with each model on an L40 GPU
- the frontend lived on Vercel while the backend was often run locally during active development and demos

## 9. The Repeatable Team Process

For each model, the repeatable sequence was:

1. Choose the entity and replay target.
2. Collect or curate replay data for that entity across the `2025-01-01 -> 2026-01-01` historical window.
3. Convert the collected data into replay JSON plus raw audit files.
4. Start from `Qwen/Qwen3-8B`.
5. Run replay-backed OpenEnv post-training with GRPO.
6. Save the entity checkpoint.
7. Upload the resulting checkpoint to Hugging Face under `@AlazarM`.
8. Serve the resulting model behind a vLLM-compatible endpoint.

For the full product, that six-model training pipeline fed directly into the live operator experience:

- current RSS feeds feed the live intel layer
- persistent replays make prior entity actions visible in the interface
- FastAPI + OpenEnv manage the backend state transition logic
- NumPy supports statistical calculations such as resource and mineral-interest analysis
- Mapbox and Framer Motion power the central world map and motion system

The result is six fine-tuned models with shared infrastructure but distinct behavior.

## 10. The High-Level Takeaway

The team did not fine-tune Trenches by dumping articles into a generic chat model.

They built a pipeline that:

- starts from aligned runtime sources
- collects historical evidence over `2025-01-01 -> 2026-01-01`
- converts it into replay events
- formats training inputs to look like live simulator observations
- trains each entity with replay-aware rewards
- keeps oversight as the cross-entity control model
- scales the serious runs through Modal + vLLM

That is why the six models behave like six different strategic actors rather than six prompt variants of one shared policy.

So the US was rewarded on things like regional
access, shipping security, domestic support, and force posture; Israel on homeland security and deterrence; Iran on regime stability and Hormuz leverage; Hezbollah on survivability and logistics depth; the
Gulf on shipping continuity and investor confidence; and Oversight on runaway-risk reduction and intervention legitimacy. At each step, reward blended action quality and forecast quality: did the action
move that entity closer to its doctrine-specific targets, and did its prediction match the next revealed event in the replay timeline.

The finetuning was done as post-training, not just prompt engineering. We started from Qwen/Qwen3-8B, used Hugging Face TRL with an OpenEnv-compatible Trenches environment, and ran the serious jobs on
Modal. Historical data was collected over the 2025-01-01 to 2026-01-01 window using GDELT plus aligned source feeds, then formatted to match the same structured observation and output types the models would
see live. Each of the five actor models was trained on its own entity-specific slice, while the Oversight model was trained across the full system view.

And yes, they had actions. The models were trained to output structured actions such as hold, negotiate, sanction, strike, defend, intel_query, mobilize, and deceive, along with a prediction of what would
happen next. Those actions were validated server-side and then applied back into the shared world state.