# Trenches Mermaid

This file keeps the architecture in one place, but breaks the Mermaid into smaller sections so it stays readable and renderable.

## 1. Historical Data + Replay Building

```mermaid
flowchart TB
    MANIFEST["Source Manifest<br/>entity-aligned RSS / official / wire / OSINT sources"]
    LIVEWS["Live world news<br/>RSS + current feeds"]
    GDELT["GDELT historical retrieval<br/>2025-01-01 -> 2026-01-01"]
    CHUNK["Daily chunking + entity routing"]
    FORMAT["Runtime-shaped formatting<br/>observation-like structured examples"]
    REPLAY["Replay JSON + raw audit JSONL<br/>per entity"]

    MANIFEST --> GDELT
    LIVEWS --> CHUNK
    GDELT --> CHUNK
    CHUNK --> FORMAT
    FORMAT --> REPLAY
```

- `source_manifest.json` defines the entity-aligned source universe.
- Historical retrieval and live feeds are both routed into runtime-shaped replay examples.
- The replay output is the training input used downstream.

## 2. Post-Training on Modal

```mermaid
flowchart TB
    BASE["Hugging Face base model<br/>Qwen/Qwen3-8B"]
    CLI["training_cli.py"]
    TRL["HF TRL<br/>GRPOTrainer"]
    OPENENV["OpenEnv boundary<br/>reset / step"]
    ENV["Trenches env<br/>fog of war + latent state + beliefs + rewards"]
    VLLM["vLLM generation engine"]
    CKPT["Entity checkpoints"]

    BASE --> CLI
    REPLAY --> CLI
    CLI --> TRL
    TRL --> OPENENV
    OPENENV --> ENV
    ENV --> OPENENV
    TRL --> VLLM
    VLLM --> TRL
    TRL --> CKPT
```

- Replays feed `training_cli.py`.
- `GRPOTrainer` uses the environment loop through the OpenEnv boundary.
- Training produces entity-specific checkpoints.

## 3. Model Registry + Inference Hosting

```mermaid
flowchart TB
    CKPT["Entity checkpoints"]
    US["US model"]
    ISR["Israel model"]
    IRN["Iran model"]
    HEZ["Hezbollah model"]
    GULF["Gulf model"]
    OVS["Oversight model<br/>trained across all data"]
    HFHUB["Hugging Face Hub<br/>@AlazarM checkpoints"]
    MODAL["Modal inference endpoints<br/>1 GPU container per entity"]

    CKPT --> US
    CKPT --> ISR
    CKPT --> IRN
    CKPT --> HEZ
    CKPT --> GULF
    CKPT --> OVS

    CKPT --> HFHUB
    HFHUB --> MODAL

    US --> MODAL
    ISR --> MODAL
    IRN --> MODAL
    HEZ --> MODAL
    GULF --> MODAL
    OVS --> MODAL
```

- Checkpoints are published to Hugging Face Hub.
- Modal exposes one inference endpoint per entity model.
- Oversight is separate from the five actor models.

## 4. Backend Runtime

```mermaid
flowchart TB
    FASTAPI["FastAPI backend"]
    CORE["OpenEnv Core + session manager"]
    HARVEST["Source harvester<br/>RSS / source packets / monitor"]
    PROVIDER["Provider runtime<br/>vLLM routing + fallback"]
    OVERSIGHT["Oversight logic"]
    WORLD["Shared world state<br/>latent events + actor state + belief state"]
    NUMPY["NumPy analytics<br/>resource / mineral / pressure calculations"]
    MODAL["Modal inference endpoints"]
    LIVEWS["Live world news"]

    FASTAPI --> CORE
    CORE --> HARVEST
    CORE --> PROVIDER
    CORE --> OVERSIGHT
    CORE --> WORLD
    WORLD --> NUMPY
    HARVEST --> WORLD
    OVERSIGHT --> WORLD
    PROVIDER --> WORLD
    MODAL --> PROVIDER
    LIVEWS --> HARVEST
```

- Session state lives in the backend runtime, not the frontend.
- Source harvesting and provider inference both feed the same world model.
- Oversight modifies or evaluates actions before the world advances.

## 5. Frontend Command Center

```mermaid
flowchart TB
    VERCEL["Vercel-hosted Next.js frontend"]
    GLOBE["Mapbox globe"]
    FEEDS["Live Intel Feed"]
    ACTIVITY["Entity Activity"]
    CHAT["Oversight chat"]
    REVERSE["Reverse feed / replay strip"]
    MOTION["Framer Motion UI layer"]
    FASTAPI["FastAPI backend"]
    WORLD["Backend session state"]

    VERCEL --> GLOBE
    VERCEL --> FEEDS
    VERCEL --> ACTIVITY
    VERCEL --> CHAT
    VERCEL --> REVERSE
    VERCEL --> MOTION

    FASTAPI --> VERCEL
    VERCEL --> FASTAPI
    WORLD --> VERCEL
```

- The frontend is an operator surface over backend session state.
- Timeline, feeds, map, and chat are all different views over the same simulation.

## 6. User-Facing Outputs + Legacy Infra

```mermaid
flowchart TB
    WORLD["Shared world state"]
    HARVEST["Source harvester"]
    PROVIDER["Provider runtime"]
    VERCEL["Frontend"]

    SNAP["Session snapshot<br/>truth + observation + belief projections"]
    TRACE["Persistent entity actions<br/>predicted vs actual replay"]
    NEWS["Current entity-relevant news"]

    SPACES["Hugging Face Spaces"]
    THUNDER["ThunderCompute"]
    CF["Cloudflare tunnels"]
    NFLANK["NorthFlank"]
    BASE["Base model experiments"]
    HFHUB["Hub model pulls"]
    FASTAPI["Early backend path"]

    WORLD --> SNAP
    HARVEST --> NEWS
    PROVIDER --> TRACE

    SNAP --> VERCEL
    TRACE --> VERCEL
    NEWS --> VERCEL

    BASE -. earlier experiments .-> SPACES
    FASTAPI -. earlier backend path .-> THUNDER
    THUNDER -. workaround .-> CF
    HFHUB -. model pull attempt .-> NFLANK
```

- This section isolates the user-visible outputs from the historical infrastructure experiments.
- The dotted lines are explicitly non-final paths.

## Reading Order

- Start with historical data and replay building.
- Then read post-training and model serving.
- Then read backend runtime and frontend command center.
- Finish with user-facing outputs and the legacy infra notes.
