# Trenches — TODO

## Reward System

- [ ] **Event-prediction RL rewards** — when a real-world event occurs and an agent's prior prediction/action aligns with it, grant a positive reward signal. This closes the loop between live data ingestion and agent learning.
  - Track agent predictions per turn (e.g., "Iran will retaliate within 2 turns")
  - Compare predictions against actual events that fire from RSS/OSINT feeds
  - Reward = f(prediction accuracy, lead time, specificity)
  - Only **real events** (from live feeds or env-generated stochastic events) impact the reward signal

- [ ] **Chat-injected fake events** — allow manual event injection via the chat panel that influences agent behavior but does **not** affect reward calculations.
  - Tag chat-injected events with `source: "manual"` vs real events with `source: "live"` or `source: "env"`
  - Agents still react to fake events (observe and act), but the reward function filters them out
  - Useful for demos, testing edge cases, and probing agent behavior without polluting the training signal

## UI / Frontend

- [x] **Event timeline with time control** — scrubber bar (like a video editor) for navigating, rewinding, and branching the simulation
  - **Scrubber bar** at the bottom: drag to jump to any turn/timestamp, play/pause, rewind, fast-forward
  - Two event types on the timeline: **predictions** (agent forecasts) and **actuals** (confirmed real events)
  - Predictions that matched actual outcomes are visually linked; incorrect ones shown faded
  - **Branching**: when a fake scenario is injected via chat, the timeline forks — you can scrub back to before the injection and see the "what if" branch vs the real timeline
  - Playback controls: step-by-step (turn by turn), continuous playback at adjustable speed
  - Markers on the scrubber for key events (escalations, interventions, injected scenarios)
  - Filterable by agent, event type, and time range
  - Feeds into the reward system — correct predictions on the timeline = positive RL signal

- [x] **Theater map with pulse nodes** — geographic canvas with per-agent activity pulse intensity and click-through context/action history.
- [x] **Inter-agent tension matrix** — N×N heatmap from cool blue to red for bilateral hostility at a glance.
- [x] **Simulation version tree (git-style)** — branch/fork views for what-if exploration with rewind-to-turn controls.
- [x] Merge tension/stats pills into top bar
- [x] Disable text selection on floating panels
- [x] Remove Mapbox logo
- [x] Clean up README

## Infrastructure

- [x] Push to HF Space (`AlazarM/trenches`)
- [ ] Add `NEXT_PUBLIC_MAPBOX_TOKEN` as HF Space secret

## Post-Training

- [x] 6 synthetic seed replay datasets (in `synthetic_historical_replays/`)
- [x] Training CLI with GRPO, hyperparameter args, checkpointing
- [x] Local smoke test (tiny-gpt2, US + Israel)
- [x] HF GPU smoke test on T4 ([trenches-training-smoke](https://huggingface.co/spaces/AlazarM/trenches-training-smoke))
- [x] All 6 entity models → `Qwen/Qwen3-8B` (no quantization)
- [x] Historical data collection pipeline (GDELT → replay JSON)
- [ ] Run historical collector for all 6 entities
- [ ] Curator-review collected replay data
- [ ] Spin up 6 HF A100 Spaces for production training
- [ ] Evaluation/baseline reporting
