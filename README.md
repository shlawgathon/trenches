---
title: Trenches
emoji: 🌍
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
---

<img width="512" height="512" alt="Image Background Remover (1)" src="https://github.com/user-attachments/assets/a1ab0df2-435f-444b-b8a1-36b1a64b55e8" />

# Trenches

A multi-agent crisis simulator built on [OpenEnv](https://github.com/openenv-ai/openenv). LLM agents navigate a fog-of-war geopolitical scenario — negotiating coalitions, managing deception, and responding to live global events — while a dedicated oversight agent monitors for dangerous escalation.

## Overview

Trenches drops six LLM-powered actors into a volatile 2026 Middle East crisis. Each agent operates under partial observability with role-specific intelligence, tools, and incentives. A scalable oversight mechanism intervenes when escalation risk crosses critical thresholds.

| Agent             | Role                                            | Model    |
| ----------------- | ----------------------------------------------- | -------- |
| 🇺🇸 United States  | Hawkish superpower balancing polls & projection | Qwen3-8B |
| 🇮🇱 Israel         | Regional actor with strike autonomy             | Qwen3-8B |
| 🇮🇷 Iran           | Adversary leveraging proxies & deception        | Qwen3-8B |
| 🪖 Hezbollah      | Non-state militia with asymmetric tactics       | Qwen3-8B |
| 🛢️ Gulf Coalition | Economic bloc protecting oil & stability        | Qwen3-8B |
| 🔍 Oversight      | Monitors all actors, intervenes on escalation   | Qwen3-8B |

## Key Features

- **Fog of War** — agents see only their role-filtered intel, never the full world state
- **Live News Injection** — real-time RSS/OSINT feeds drive stochastic in-sim events
- **Scalable Oversight** — Bayesian risk scoring triggers interventions before runaway escalation
- **Oversight Predictions** — oversight agent predicts what the highest-risk nation will do next turn, shown as red dots in the timeline PRED lane
- **Globe Interaction Arcs** — great-circle arc lines between nations on the Mapbox globe, color-coded by action type (🔴 strike, 🟠 intel, 🔵 defend, 🟢 negotiate, 🟡 sanction), with hover tooltips showing actor → target and action details
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
