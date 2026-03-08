# Post-Training Plan: 6 Entities, 2 Hours, Done

Single A100 session. GRPO post-training on OpenEnv. **$5 total**.

Qwen3.5-9B is already a strong instruct model. This is **post-training** — we're aligning it to each entity's policy behavior through the OpenEnv reward signal, not training from scratch.

## Schedule

| Entity    | Time    | Steps         |
| --------- | ------- | ------------- |
| us        | ~20 min | 50 GRPO steps |
| israel    | ~20 min | 50            |
| iran      | ~20 min | 50            |
| hezbollah | ~20 min | 50            |
| gulf      | ~20 min | 50            |
| oversight | ~20 min | 50            |
| **Total** | **~2h** | **$5**        |

## Config

| Parameter                 | Value                                  |
| ------------------------- | -------------------------------------- |
| Model                     | `Qwen/Qwen3.5-9B` (4-bit bitsandbytes) |
| GPU                       | HuggingFace A100 80GB ($2.50/hr)       |
| Algorithm                 | GRPO (TRL + OpenEnv)                   |
| `--max-steps`             | 50                                     |
| `--train-size`            | 128                                    |
| `--num-generations`       | 8                                      |
| `--batch-size`            | 2                                      |
| `--gradient-accumulation` | 4                                      |
| `--learning-rate`         | 5e-7                                   |
| `--max-prompt-length`     | 1536                                   |
| `--max-completion-length` | 256                                    |

## Build Steps

1. **Create 5 replay datasets** — `historical_replays/` JSONs for israel, iran, hezbollah, gulf, oversight
2. **Add `--quantize-4bit`** to `training_cli.py` + `bitsandbytes` dep
3. **Create `post_train_all.sh`** runner script
4. **Smoke test** locally with tiny-gpt2
5. **Run on HF A100** — kick off, let it run ~2h

## Runner Script

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3.5-9B"
COMMON="--quantize-4bit --generation-backend transformers \
  --training-stage stage_1_dense --max-steps 50 --train-size 128 \
  --num-generations 8 --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 --learning-rate 5e-7 \
  --max-prompt-length 1536 --max-completion-length 256 \
  --preview-samples 3"

for AGENT in us israel iran hezbollah gulf oversight; do
  echo "===== Post-training $AGENT ($(date)) ====="
  python -m trenches_env.training_cli \
    --model-id "$MODEL" --training-agent "$AGENT" \
    --replay-id "${AGENT}_forecast_seed_2025_2026" \
    --output-dir "checkpoints/${AGENT}-qwen3.5-9b-4bit" \
    $COMMON
done
```

## Output

```
backend/checkpoints/
├── us-qwen3.5-9b-4bit/
├── israel-qwen3.5-9b-4bit/
├── iran-qwen3.5-9b-4bit/
├── hezbollah-qwen3.5-9b-4bit/
├── gulf-qwen3.5-9b-4bit/
└── oversight-qwen3.5-9b-4bit/
```
