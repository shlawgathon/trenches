# Post-Training Plan: 6 Entities × 1 Hour Parallel

## Overview

6 Modal A100-80GB×2 containers running in parallel. Total wall time: **~1 hour**. Total cost: **~$33**. Base model: **Qwen/Qwen3-8B** (full precision, vLLM server mode).

GRPO post-training on OpenEnv. Qwen3-8B already knows how to reason — we're aligning it to each entity's policy behavior through the environment reward signal.

## Infrastructure

**Modal** with vLLM **server mode** (2 GPUs per entity):

- **GPU 0**: vLLM server (dedicated inference GPU — full 80GB for model + KV cache)
- **GPU 1**: GRPO training (dedicated training GPU — full 80GB for policy + ref + optimizer)

No memory contention. No sleep mode hacks. Full vLLM speed.

## Cost

| Item          | Rate      | Quantity        | Cost     |
| ------------- | --------- | --------------- | -------- |
| A100-80GB × 2 | ~$5.56/hr | 6 entities × 1h | **~$33** |

## Optimal Hyperparameters

Researched from TRL docs, DeepSeek-R1 paper, Open-R1 recipe, and TRL OpenEnv examples.

```yaml
# Model
model_id: Qwen/Qwen3-8B
# No quantization — full precision on A100 80GB.

# GRPO Core (from DeepSeek-R1 + Open-R1 recipes)
algorithm: GRPO
loss_type: grpo
beta: 0.001 # KL coefficient (DeepSeek-R1 uses 0.001)
num_generations: 16 # DeepSeek-R1: "sample 16 outputs per prompt"
max_steps: 100 # 1 hour on A100 with vLLM server mode
warmup_steps: 10 # Stabilize early training

# Learning Rate
learning_rate: 5e-6 # Open-R1 + OpenEnv Sudoku example both use 5e-6

# Batching
per_device_train_batch_size: 1
gradient_accumulation_steps: 8 # Effective batch = 8

# Context
max_prompt_length: 1536
max_completion_length: 256

# Generation
generation_backend: vllm
vllm_mode: server # Separate GPU for vLLM — no memory contention
temperature: 0.8
top_k: 10

# Saving
save_strategy: steps
save_steps: 25 # 4 checkpoints per run

# Preview
preview_samples: 3
training_stage: stage_1_dense
```

### Why These Settings

| Setting                    | Value                  | Source/Reasoning                                      |
| -------------------------- | ---------------------- | ----------------------------------------------------- |
| `num_generations: 16`      | DeepSeek-R1            | More rollouts = better advantage estimation           |
| `beta: 0.001`              | DeepSeek-R1            | Low KL penalty allows exploration                     |
| `learning_rate: 5e-6`      | Open-R1 + TRL examples | Post-training converges with higher LR                |
| `gradient_accumulation: 8` | TRL OpenEnv Sudoku     | Effective batch of 8 stabilizes updates               |
| `temperature: 0.8`         | TRL OpenEnv Sudoku     | Encourages diverse completions                        |
| `vllm_mode: server`        | Modal 2-GPU            | Eliminates GPU memory contention                      |
| `No quantization`          | A100 80GB              | Full precision avoids noise, simplifies checkpointing |

## Commands

```bash
# Smoke test (single entity, 1 step)
modal run backend/train_modal.py::smoke --entity us

# Full training (single entity)
modal run --detach backend/train_modal.py::train --entity us --replay-id us_synthetic_seed_2025_2026

# Full training (all 6 entities in parallel)
modal run --detach backend/train_modal.py::train_all

# Download checkpoints
modal volume get trenches-checkpoints .
```

## HuggingFace Hub Output

```
shlawgathon/trenches-us-qwen3-8b
shlawgathon/trenches-israel-qwen3-8b
shlawgathon/trenches-iran-qwen3-8b
shlawgathon/trenches-hezbollah-qwen3-8b
shlawgathon/trenches-gulf-qwen3-8b
shlawgathon/trenches-oversight-qwen3-8b
```

Each checkpoint contains: `config.json`, `model.safetensors`, `tokenizer.json`, `generation_config.json`, `training_args.bin`

## Build Steps

1. ~~Create 5 replay datasets (israel, iran, hezbollah, gulf, oversight)~~ ✅ done
2. ~~Add `--quantize-4bit` to `training_cli.py`~~ ✅ done
3. ~~Add `beta`, `warmup_steps`, `temperature`, `top_k`, `save_strategy` CLI args~~ ✅ done
4. ~~Add `bitsandbytes>=0.43.0` to `pyproject.toml`~~ ✅ done
5. ~~Smoke test locally with tiny-gpt2~~ ✅ done (US + Israel pass)
6. ~~Smoke test on HF T4 GPU~~ ✅ done
7. ~~Create Modal training script (`train_modal.py`)~~ ✅ done
8. Smoke test on Modal: `modal run backend/train_modal.py::smoke --entity us`
9. Spin up 6 Modal containers → 1 hour → done
