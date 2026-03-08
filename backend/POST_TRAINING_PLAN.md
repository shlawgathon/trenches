# Post-Training Plan: 6 Entities × 1 Hour Parallel

## Overview

6 HF A100 Spaces running in parallel. Total wall time: **1 hour**. Total cost: **$15**.

GRPO post-training on OpenEnv. Qwen3.5-9B already knows how to reason — we're aligning it to each entity's policy behavior through the environment reward signal.

## Cost

| Item      | Rate     | Quantity      | Cost    |
| --------- | -------- | ------------- | ------- |
| A100 80GB | $2.50/hr | 6 Spaces × 1h | **$15** |

## Optimal Hyperparameters

Researched from TRL docs, DeepSeek-R1 paper, Open-R1 recipe, and TRL OpenEnv examples.

```yaml
# Model
model_id: Qwen/Qwen3.5-9B
quantization: 4-bit NF4 (bitsandbytes, bnb_4bit_compute_dtype=bfloat16)
# Research: QLoRA + NF4 outperforms raw 4-bit for RL post-training.
# Quantization noise actually aids exploration (QeRL paper).

# GRPO Core (from DeepSeek-R1 + Open-R1 recipes)
algorithm: GRPO
loss_type: grpo
beta: 0.001 # KL coefficient (DeepSeek-R1 uses 0.001)
num_generations:
  16 # DeepSeek-R1: "sample 16 outputs per prompt"
  # More generations = better group-relative advantage signal
max_steps: 100 # 1 hour on A100 with these settings
warmup_steps: 10 # Stabilize early training

# Learning Rate
learning_rate:
  5e-6 # Open-R1 + OpenEnv Sudoku example both use 5e-6
  # Higher than our earlier 5e-7; research shows
  # post-training converges faster with this range

# Batching
per_device_train_batch_size: 1 # Memory-safe for 9B 4-bit
gradient_accumulation_steps: 8 # Effective batch = 8 (from TRL Sudoku OpenEnv example)

# Context
max_prompt_length: 1536
max_completion_length: 256

# Generation Sampling (from TRL OpenEnv Sudoku)
temperature: 0.8 # Balanced exploration vs exploitation
top_k: 10 # Focused sampling

# Saving
save_strategy: steps
save_steps: 25 # Checkpoint every 25 steps (4 saves per run)

# Inference
generation_backend: transformers # vllm if CUDA available
# If vllm: use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3

# Preview
preview_samples: 3
training_stage: stage_1_dense
```

### Why These Settings

| Setting                    | Value                  | Source/Reasoning                                                                               |
| -------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| `num_generations: 16`      | DeepSeek-R1            | More rollouts = better advantage estimation. 16 is the standard for GRPO                       |
| `beta: 0.001`              | DeepSeek-R1            | Low KL penalty allows the model to explore further from base policy                            |
| `learning_rate: 5e-6`      | Open-R1 + TRL examples | 10x higher than our earlier setting; post-training on instruct models converges with higher LR |
| `gradient_accumulation: 8` | TRL OpenEnv Sudoku     | Effective batch of 8 stabilizes updates without excessive VRAM                                 |
| `temperature: 0.8`         | TRL OpenEnv Sudoku     | Encourages diverse completions during rollout                                                  |
| `NF4 quantization`         | QeRL paper (2026)      | Quantization noise enhances RL exploration; NF4 outperforms raw 4-bit                          |

## Per-Space Command

Replace `ENTITY` with: `us`, `israel`, `iran`, `hezbollah`, `gulf`, `oversight`

```bash
python -m trenches_env.training_cli \
  --model-id Qwen/Qwen3.5-9B \
  --quantize-4bit \
  --training-agent ENTITY \
  --replay-id ENTITY_forecast_seed_2025_2026 \
  --output-dir checkpoints/ENTITY-qwen3.5-9b-4bit \
  --generation-backend transformers \
  --training-stage stage_1_dense \
  --max-steps 100 \
  --train-size 256 \
  --num-generations 16 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --max-prompt-length 1536 \
  --max-completion-length 256 \
  --preview-samples 3
```

## HuggingFace Hub Output

```
shlawgathon/trenches-us-qwen3.5-9b-4bit
shlawgathon/trenches-israel-qwen3.5-9b-4bit
shlawgathon/trenches-iran-qwen3.5-9b-4bit
shlawgathon/trenches-hezbollah-qwen3.5-9b-4bit
shlawgathon/trenches-gulf-qwen3.5-9b-4bit
shlawgathon/trenches-oversight-qwen3.5-9b-4bit
```

Each checkpoint contains: `config.json`, `model.safetensors`, `tokenizer.json`, `generation_config.json`, `training_args.bin`

## Build Steps

1. Create 5 replay datasets (israel, iran, hezbollah, gulf, oversight)
2. Add `--quantize-4bit` to `training_cli.py` (NF4 via bitsandbytes)
3. Add `beta`, `warmup_steps`, `temperature`, `top_k`, `save_strategy` CLI args
4. Add `bitsandbytes>=0.43.0` to `pyproject.toml`
5. Smoke test locally with tiny-gpt2
6. Spin up 6 HF A100 Spaces → 1 hour → done
