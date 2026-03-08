# Data Handoff

## Chosen Base Model

Use:

- `Qwen/Qwen3-8B`

Why this is the best default for the `2025-01 -> 2026-01` post-training window:

- it was released inside the required time frame
- it is available on Hugging Face
- it is strong enough for structured action + prediction output
- it is still realistic to run six separate entity post-training jobs on it

This is the recommended first real base model for all six entities.

## What I Added For Data

The repo already had:

- synthetic seed replay JSON files under [backend/src/trenches_env/historical_replays](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/historical_replays)
- an OpenEnv replay training path
- a training CLI that consumes replay JSON with the `HistoricalReplayDefinition -> HistoricalEvent` schema

What I added is the first path from real historical sources into that same replay schema.

### New Files

- [backend/src/trenches_env/historical_collection.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/historical_collection.py)
  - builds historical source profiles from the existing source manifest
  - derives historical domains from allowlisted agent sources
  - defines the `2025` and `2026` collection windows
  - dedupes collected articles
  - converts collected articles into the exact replay event schema used by training

- [backend/src/trenches_env/historical_collection_cli.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/historical_collection_cli.py)
  - CLI collector
  - queries the GDELT DOC API month by month
  - writes raw article audit files
  - writes replay JSON files in the same schema as the existing synthetic seeds

- [backend/tests/test_historical_collection.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/tests/test_historical_collection.py)
  - validates source-profile extraction
  - validates article -> replay-event conversion
  - validates replay JSON compatibility with the existing historical replay loader

## What Source Data It Uses

The collector starts from the existing [backend/src/trenches_env/source_manifest.json](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/source_manifest.json).

That means it does not invent a separate source universe. It reuses the current project’s aligned sources, then extracts historical domains from them. In practice this means it leans on the project’s existing training-core sources such as:

- Reuters and wire-style reporting
- official government / ministry sources
- regional English-language outlets already assigned to the entities
- market / shipping / sanctions / diplomacy sources already present in the manifest

For historical collection, it converts those sources into domain-filtered GDELT queries and collects article candidates month by month.

## Output Files

The collector writes two outputs per run.

### 1. Replay JSON

Path example:

- `backend/src/trenches_env/historical_replays/us_historical_2025.json`

This matches the same structure as the existing synthetic seed files:

- `replay_id`
- `name`
- `description`
- `training_agent`
- `events[]`

Each event matches the current training schema:

- `event_id`
- `timestamp`
- `topic`
- `region`
- `actors`
- `targets`
- `severity`
- `summary`
- `public_summary`
- `source_type`
- `confirmed`
- `tags`
- `impact`

### 2. Raw Audit JSONL

Path example:

- `backend/tmp-historical-raw/us_historical_2025.articles.jsonl`

Each line contains:

- `article_id`
- `agent_id`
- `source_id`
- `source_name`
- `title`
- `url`
- `domain`
- `timestamp`
- `query`
- `window_id`

This is the provenance trail for curator review.

## Date Windows

The collector currently supports:

- `2025` -> `2025-01-01` through `2026-01-01`
- `2026` -> `2026-01-01` through the current day at collection time

Important note:

As of March 7, 2026, `2026` cannot honestly mean `2026-01-01 -> 2027-01-01` yet. The collector clamps future end dates to the current day so it does not pretend future historical data exists.

## What Is Real vs Heuristic

Real:

- source alignment from the project’s own source manifest
- historical article collection via GDELT
- raw audit/provenance files
- replay JSON output in the exact schema the training system already consumes

Heuristic:

- topic classification from article titles
- severity classification from article titles
- dedupe logic
- actor/target inference
- event `impact` generation

That heuristic layer is intentional. It gives you a bootstrap pipeline from real historical articles into replay training data, but the resulting replay should still be curator-reviewed before production post-training.

## Commands

From repo root:

```bash
backend/.venv/bin/python -m trenches_env.historical_collection_cli \
  --training-agent us \
  --window 2025 \
  --window 2026 \
  --max-records-per-query 50 \
  --max-events 128 \
  --output-dir backend/src/trenches_env/historical_replays \
  --raw-dir backend/tmp-historical-raw
```

All entities:

```bash
backend/.venv/bin/python -m trenches_env.historical_collection_cli \
  --training-agent all \
  --window 2025 \
  --window 2026 \
  --max-records-per-query 50 \
  --max-events 128 \
  --output-dir backend/src/trenches_env/historical_replays \
  --raw-dir backend/tmp-historical-raw
```

## Docs Updated

I also updated:

- [backend/TRAINING_RUNBOOK.md](/Users/alazarmanakelew/IdeaProjects/trenches/backend/TRAINING_RUNBOOK.md)
- [backend/TRAINING_FLOW.md](/Users/alazarmanakelew/IdeaProjects/trenches/backend/TRAINING_FLOW.md)
- [backend/POST_TRAINING_PLAN.md](/Users/alazarmanakelew/IdeaProjects/trenches/backend/POST_TRAINING_PLAN.md)
- [backend/pyproject.toml](/Users/alazarmanakelew/IdeaProjects/trenches/backend/pyproject.toml)

So the collection path is now documented and exposed as a real CLI entry point.

## Verification

The added data-collection path was verified locally with:

```bash
PYTHONPYCACHEPREFIX=/tmp/trenches-pyc python -m py_compile \
  backend/src/trenches_env/historical_collection.py \
  backend/src/trenches_env/historical_collection_cli.py
```

```bash
cd backend
uv run --extra dev python -m pytest \
  tests/test_historical_collection.py \
  tests/test_openenv_adapter.py \
  tests/test_server.py -q
```

Result:

- `20 passed in 8.78s`

## Handoff

What is ready now:

- a chosen base model: `Qwen/Qwen3-8B`
- a collector path from real historical sources into the existing replay schema
- raw provenance output
- replay JSON output compatible with the current OpenEnv training flow

What still needs to happen next:

1. Run the collector for each entity.
2. Curator-review the raw article audit files and the generated replay JSON.
3. Replace the current synthetic seed replays with reviewed historical replays.
4. Update the actual training runs to use `Qwen/Qwen3-8B` as the base model.
5. Keep the old synthetic seeds only for smoke tests.

One important truth:

The collector is the first real data path, but it does not magically make the replay production-grade by itself. The training-ready replay still needs human review because event impact shaping is currently heuristic.
