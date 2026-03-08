# Trenches Master Mermaid

This file contains one master Mermaid visual that explains the full Trenches technical design across:

- historical data collection
- replay building
- post-training
- model registry and serving
- backend orchestration
- frontend operator experience

```mermaid
flowchart TB
    subgraph HIST["Historical Data + Replay Building"]
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
    end

    subgraph TRAIN["Post-Training on Modal"]
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
    end

    subgraph MODELS["Six Finetuned Models"]
        US["US model"]
        ISR["Israel model"]
        IRN["Iran model"]
        HEZ["Hezbollah model"]
        GULF["Gulf model"]
        OVS["Oversight model<br/>trained across all data"]
        CKPT --> US
        CKPT --> ISR
        CKPT --> IRN
        CKPT --> HEZ
        CKPT --> GULF
        CKPT --> OVS
    end

    subgraph REGISTRY["Registry + Inference Hosting"]
        HFHUB["Hugging Face Hub<br/>@AlazarM checkpoints"]
        MODAL["Modal inference endpoints<br/>1 L40 GPU per entity"]
        CKPT --> HFHUB
        HFHUB --> MODAL
    end

    subgraph RUNTIME["Backend Runtime"]
        FASTAPI["FastAPI backend"]
        CORE["OpenEnv Core + session manager"]
        HARVEST["Source harvester<br/>RSS / source packets / monitor"]
        PROVIDER["Provider runtime<br/>vLLM routing + fallback"]
        OVERSIGHT["Oversight logic"]
        WORLD["Shared world state<br/>latent events + actor state + belief state"]
        NUMPY["NumPy analytics<br/>resource / mineral / pressure calculations"]
        FASTAPI --> CORE
        CORE --> HARVEST
        CORE --> PROVIDER
        CORE --> OVERSIGHT
        CORE --> WORLD
        WORLD --> NUMPY
        HARVEST --> WORLD
        OVERSIGHT --> WORLD
        PROVIDER --> WORLD
    end

    subgraph FE["Frontend Command Center"]
        VERCEL["Vercel-hosted Next.js frontend"]
        GLOBE["Mapbox globe"]
        FEEDS["Live Intel Feed"]
        ACTIVITY["Entity Activity"]
        CHAT["Oversight chat"]
        REVERSE["Reverse feed / replay strip"]
        MOTION["Framer Motion UI layer"]
        VERCEL --> GLOBE
        VERCEL --> FEEDS
        VERCEL --> ACTIVITY
        VERCEL --> CHAT
        VERCEL --> REVERSE
        VERCEL --> MOTION
    end

    MODAL --> PROVIDER
    US --> MODAL
    ISR --> MODAL
    IRN --> MODAL
    HEZ --> MODAL
    GULF --> MODAL
    OVS --> MODAL

    LIVEWS --> HARVEST
    FASTAPI --> VERCEL
    VERCEL --> FASTAPI
    WORLD --> VERCEL

    subgraph UX["What the User Sees"]
        SNAP["Session snapshot<br/>truth + observation + belief projections"]
        TRACE["Persistent entity actions<br/>predicted vs actual replay"]
        NEWS["Current entity-relevant news"]
    end

    WORLD --> SNAP
    HARVEST --> NEWS
    PROVIDER --> TRACE
    SNAP --> VERCEL
    TRACE --> VERCEL
    NEWS --> VERCEL

    subgraph LEGACY["Historical Infra Attempts"]
        SPACES["Hugging Face Spaces"]
        THUNDER["ThunderCompute"]
        CF["Cloudflare tunnels"]
        NFLANK["NorthFlank"]
    end

    BASE -. earlier experiments .-> SPACES
    FASTAPI -. earlier backend path .-> THUNDER
    THUNDER -. workaround .-> CF
    HFHUB -. model pull attempt .-> NFLANK
```

## Reading Guide

- Left side: where training data came from and how it was shaped.
- Center: how `HF TRL + OpenEnv + vLLM` produced six separate finetuned models on Modal.
- Right side: how those models are served back into the live FastAPI simulation runtime.
- Bottom: how the Vercel frontend turns backend state into the globe, feeds, entity activity, and replay surfaces.
- Dotted lines: infrastructure paths that were tried earlier but were not the final operating choice.
