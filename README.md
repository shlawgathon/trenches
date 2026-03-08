# Trenches

A multi-agent crisis simulator built on [OpenEnv](https://github.com/openenv-ai/openenv). LLM agents navigate a fog-of-war geopolitical scenario — negotiating coalitions, managing deception, and responding to live global events — while a dedicated oversight agent monitors for dangerous escalation.

## Overview

Trenches drops six LLM-powered actors into a volatile 2026 Middle East crisis. Each agent operates under partial observability with role-specific intelligence, tools, and incentives. A scalable oversight mechanism intervenes when escalation risk crosses critical thresholds.

| Agent             | Role                                            | Model             |
| ----------------- | ----------------------------------------------- | ----------------- |
| 🇺🇸 United States  | Hawkish superpower balancing polls & projection | DeepSeek-V3.2     |
| 🇮🇱 Israel         | Regional actor with strike autonomy             | Qwen3.5-397B-A17B |
| 🇮🇷 Iran           | Adversary leveraging proxies & deception        | GLM-4.7           |
| 🪖 Hezbollah      | Non-state militia with asymmetric tactics       | Kimi-K2.5         |
| 🛢️ Gulf Coalition | Economic bloc protecting oil & stability        | MiniMax-M2.5      |
| 🔍 Oversight      | Monitors all actors, intervenes on escalation   | Ministral 14B     |

## Key Features

- **Fog of War** — agents see only their role-filtered intel, never the full world state
- **Live News Injection** — real-time RSS/OSINT feeds drive stochastic in-sim events
- **Scalable Oversight** — Bayesian risk scoring triggers interventions before runaway escalation
- **Tool Use** — agents call `query_intel`, `propose_negotiation`, `impose_sanctions`, etc.
- **Multi-component Rewards** — coalition stability, escalation penalty, market impact, belief alignment

## Stack

| Layer    | Tech                                                 |
| -------- | ---------------------------------------------------- |
| Frontend | Next.js 16 · Tailwind v4 · Mapbox GL · Framer Motion |
| Backend  | FastAPI · OpenEnv Core · NumPy                       |
| Infra    | Docker · Bun · uv                                    |

## Quick Start

```bash
# Frontend
bun install
bun run dev          # → http://localhost:3000

# Backend
cd backend
uv sync
source .venv/bin/activate
uvicorn trenches_env.server:app --reload --port 8000
```

Set your environment variables in `.env.local`:

```
NEXT_PUBLIC_MAPBOX_TOKEN=...
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Project Structure

```
trenches/
├── app/              # Next.js app router + API routes
├── src/
│   ├── components/   # Globe, NewsFeed, ActivityLog, ChatPanel
│   ├── hooks/        # React hooks
│   └── lib/          # Types, utils, bootstrap
├── backend/
│   ├── src/          # FastAPI server, OpenEnv environment
│   └── tests/        # Backend test suite
├── entities/         # Agent identity configs (US, Israel, Iran, etc.)
└── scripts/          # Utility scripts
```

## License

MIT
