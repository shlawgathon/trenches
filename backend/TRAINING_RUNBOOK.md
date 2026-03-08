# Trenches OpenEnv Training Runbook

This runbook shows how to run the current CLI training loop for the Trenches entity models.

The important architecture rule is simple:

- each entity is its own model
- each run trains one entity to become a better version of itself
- training happens through the native OpenEnv environment boundary
- the environment scores both action quality and forecast quality

The first implemented proof path is the `us` entity.

## What This Training Loop Does

On each replay step the model must return two separate outputs:

1. an `action`
2. a `prediction`

The backend then:

1. applies the action in the simulator
2. reveals the next historical event in the replay timeline
3. scores the prediction against that revealed event
4. blends forecast reward into the entity reward

This means the `us` model is not learning to be a generic strategist. It is learning to be a better `us` policy inside this simulator.

## Current Scope

Implemented now:

- native OpenEnv replay-aware training loop
- 6 **synthetic** seed replay datasets (us, israel, iran, hezbollah, gulf, oversight) — replace with curated truth sets for production
- CLI trainer using Hugging Face TRL
- portable local generation path with `transformers`
- GPU-oriented generation path with `vllm`

Not implemented yet:

- evaluation/baseline reporting across all entities
- UI training controls
- production (non-synthetic) replay datasets

## Requirements

Use Python `3.12`.

From the repo root:

```bash
cd /Users/xiao/trenches
```

Create a virtualenv:

```bash
uv venv backend/.venv --python 3.12
```

Install the backend plus training dependencies:

```bash
uv pip install --python backend/.venv/bin/python -e 'backend[train]' 'openenv-core[core]>=0.2.1,<0.3.0' 'torch>=2.10.0'
```

## Tokens And Env Vars

No `.env` file is required for the default public smoke test.

You only need a token if you use a gated or private Hugging Face model.

If needed:

```bash
export HF_TOKEN=your_huggingface_token
```

You do not need OpenAI, Anthropic, or other provider keys for the local replay smoke run.

Optional noise reduction:

```bash
export TRL_EXPERIMENTAL_SILENCE=1
```

## Local Smoke Run

This is the fastest way to prove the loop works on a laptop or Mac.

It uses:

- `sshleifer/tiny-gpt2`
- `transformers` generation backend
- `us` replay
- one tiny GRPO run

Run:

```bash
backend/.venv/bin/python -m trenches_env.training_cli \
  --model-id sshleifer/tiny-gpt2 \
  --generation-backend transformers \
  --training-agent us \
  --training-stage stage_1_dense \
  --replay-id us_synthetic_seed_2025_2026 \
  --train-size 4 \
  --max-steps 1 \
  --num-generations 2 \
  --max-prompt-length 512 \
  --max-completion-length 48 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --output-dir backend/tmp-training-run \
  --preview-samples 1
```

What to expect:

- the trainer starts a local backend
- the trainer talks to `/openenv`
- one short GRPO pass runs
- model artifacts are written to `backend/tmp-training-run`
- the preview step prints a rollout sample after training

This exact path has already been smoke-tested in this repo.

## Better Local Run

Once the smoke test works, switch to a stronger public instruct model.

Example:

```bash
backend/.venv/bin/python -m trenches_env.training_cli \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --generation-backend transformers \
  --training-agent us \
  --training-stage stage_1_dense \
  --replay-id us_synthetic_seed_2025_2026 \
  --train-size 32 \
  --max-steps 8 \
  --num-generations 4 \
  --max-prompt-length 1024 \
  --max-completion-length 220 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --output-dir backend/us-qwen-replay-run \
  --preview-samples 3
```

On CPU or Apple Silicon this will still be slow. That is expected.

## GPU Run With vLLM

Use this on a Linux CUDA machine when you want the documented OpenEnv + TRL path.

First install `vllm` in the same environment.

Then run:

```bash
backend/.venv/bin/python -m trenches_env.training_cli \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --generation-backend vllm \
  --training-agent us \
  --training-stage stage_1_dense \
  --replay-id us_synthetic_seed_2025_2026 \
  --train-size 64 \
  --max-steps 16 \
  --num-generations 4 \
  --max-prompt-length 1024 \
  --max-completion-length 220 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --output-dir backend/us-vllm-replay-run \
  --preview-samples 3
```

Notes:

- `vllm` is not the default because many local machines do not support it cleanly
- the CLI auto-detects a usable backend when `--generation-backend auto` is used
- `transformers` is the safer fallback for local proof runs

## Running Another Entity Later

The trainer already supports `--training-agent`, but only the bundled `us` replay is packaged right now.

The future pattern for the other five entities is:

1. create a replay file for that entity
2. point the trainer at that replay id
3. write the checkpoint to a separate output directory

Example shape:

```bash
backend/.venv/bin/python -m trenches_env.training_cli \
  --training-agent israel \
  --replay-id israel_synthetic_seed_2025_2026 \
  --output-dir backend/israel-run
```

That command shape is correct, but the `israel` replay file does not exist yet.

## How To Verify The Environment Signal

Run the focused tests:

```bash
cd /Users/xiao/trenches/backend
pytest -q tests/test_openenv_adapter.py tests/test_server.py
```

These tests cover:

- replay reset/step behavior
- prediction storage
- forecast reward scoring
- OpenEnv adapter behavior
- server wiring

## What Files Matter

Core training files:

- `backend/src/trenches_env/training_cli.py`
- `backend/src/trenches_env/openenv_adapter.py`
- `backend/src/trenches_env/env.py`
- `backend/src/trenches_env/models.py`
- `backend/src/trenches_env/historical_replay.py`
- `backend/src/trenches_env/synthetic_historical_replays/us_synthetic_seed_2025_2026.json`

## Troubleshooting

If you see `No module named 'trl'` or `No module named 'openenv'`:

- reinstall into `backend/.venv`
- make sure you are using `backend/.venv/bin/python`

If TRL complains that `generation_batch_size` is not divisible by `num_generations`:

- keep `--num-generations` small
- use the current CLI defaults

If `vllm` fails locally:

- switch to `--generation-backend transformers`

If a model is gated:

- export `HF_TOKEN`

If the run finishes with flat rewards on a tiny smoke model:

- that does not mean the environment is broken
- it usually means the toy model generated poor outputs
- use a better instruct model and a longer run

## Short Version

If you only want the shortest possible proof:

```bash
cd /Users/xiao/trenches
uv venv backend/.venv --python 3.12
uv pip install --python backend/.venv/bin/python -e 'backend[train]' 'openenv-core[core]>=0.2.1,<0.3.0' 'torch>=2.10.0'
backend/.venv/bin/python -m trenches_env.training_cli \
  --model-id sshleifer/tiny-gpt2 \
  --generation-backend transformers \
  --training-agent us \
  --replay-id us_synthetic_seed_2025_2026 \
  --train-size 4 \
  --max-steps 1 \
  --num-generations 2 \
  --output-dir backend/tmp-training-run
```

That is the current hackathon-safe path.
