# Trenches — Deployment README
> For the frontend engineer. Everything you need to connect Vercel to the live backend.

---

## Architecture Overview

```
Browser
  └── Vercel (Next.js frontend)
        └── /backend-api/* proxy
              └── Cloudflare Tunnel (HTTPS)
                    └── Instance 6 — H100 — trenches_env.server (port 8000)
                          ├── https://random-elephant-ranch-beverage.trycloudflare.com  → Instance 0 — US model
                          ├── https://fool-conducted-occurs-occurring.trycloudflare.com → Instance 1 — Hezbollah model
                          ├── https://months-flash-functional-overhead.trycloudflare.com → Instance 2 — Iran model
                          ├── https://responsibility-cowboy-collar-does.trycloudflare.com → Instance 3 — Gulf model
                          ├── https://upc-postcards-earnings-suppose.trycloudflare.com  → Instance 4 — Oversight model
                          └── https://cdna-dancing-discussion-claimed.trycloudflare.com → Instance 5 — Israel model
```

---

## Infrastructure

| Instance | GPU | Role | Model |
|---|---|---|---|
| 0 | A100 80GB | US agent | `AlazarM/trenches-us-qwen3-8b-real` |
| 1 | A100 80GB | Hezbollah agent | `AlazarM/trenches-hezbollah-qwen3-8b-real` |
| 2 | A100 80GB | Iran agent | `AlazarM/trenches-iran-qwen3-8b-real` |
| 3 | A100 80GB | Gulf agent | `AlazarM/trenches-gulf-qwen3-8b-real` |
| 4 | A100 80GB | Oversight agent | `AlazarM/trenches-oversight-qwen3-8b-real` |
| 5 | A100 80GB | Israel agent | `AlazarM/trenches-israel-qwen3-8b-real` |
| 6 | H100 | Backend server | `trenches_env.server` on port 8000 |

All instances on Thunder Compute. Instance 6 public IP: `185.216.21.128`

---

## Backend Public URL

The backend is exposed via Cloudflare Tunnel on Instance 6.

> **⚠️ The tunnel URL changes every restart.** When the tunnel is restarted, update `BACKEND_PROXY_TARGET` in Vercel and redeploy.

Current tunnel URL (update this when it changes):
```
https://<instance-6-tunnel-url>.trycloudflare.com
```

Verify it's live:
```bash
curl https://<tunnel-url>/healthz
# Expected: {"status":"ok"}

curl https://<tunnel-url>/capabilities
# Expected: all 6 agents showing "ready_for_inference": true
```

---

## Vercel Environment Variables

Set these in your Vercel project under **Settings → Environment Variables**:

```
BACKEND_PROXY_TARGET=https://<instance-6-tunnel-url>.trycloudflare.com
NEXT_PUBLIC_API_BASE_URL=/backend-api
NEXT_PUBLIC_VERCEL_API_BASE=/api
NEXT_PUBLIC_ENABLE_SOURCE_LOGIC=true
```

After setting env vars, **redeploy** the Vercel project.

### How the proxy works

The repo already rewrites `/backend-api/*` → `BACKEND_PROXY_TARGET`. So:

```
Browser → GET /backend-api/healthz
Vercel  → proxies to https://<tunnel>/healthz
Tunnel  → hits Instance 6 port 8000
```

The browser never talks directly to the backend IP.

---

## Verify End-to-End from Your Machine

```bash
# 1. Backend health
curl https://YOUR_VERCEL_APP/backend-api/healthz

# 2. All 6 agents ready
curl https://YOUR_VERCEL_APP/backend-api/capabilities

# 3. Create a session
curl -X POST https://YOUR_VERCEL_APP/backend-api/sessions \
  -H 'Content-Type: application/json' \
  -d '{}'
```

---

## Model Tunnel URLs (for backend config reference)

| Agent | Tunnel URL |
|---|---|
| US | `https://random-elephant-ranch-beverage.trycloudflare.com` |
| Hezbollah | `https://fool-conducted-occurs-occurring.trycloudflare.com` |
| Iran | `https://months-flash-functional-overhead.trycloudflare.com` |
| Gulf | `https://responsibility-cowboy-collar-does.trycloudflare.com` |
| Oversight | `https://upc-postcards-earnings-suppose.trycloudflare.com` |
| Israel | `https://cdna-dancing-discussion-claimed.trycloudflare.com` |

> These are also quick Cloudflare tunnels and will change if the model instances restart. If a model goes down, the backend `/capabilities` will show `ready_for_inference: false` for that agent.

---

## Failure Checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| `/healthz` returns non-200 | Backend tunnel down | Restart cloudflared on Instance 6, update Vercel env var |
| `/capabilities` shows agent not ready | Model tunnel down | Restart cloudflared on that model instance |
| Vercel returns 502/504 | Wrong `BACKEND_PROXY_TARGET` or no redeploy | Update env var and redeploy |
| Browser works but API calls fail | CORS config | Backend only allows localhost origins by default — check CORS settings |
| All agents show `fallback` | Backend lost env vars on restart | Re-export env vars and restart `trenches_env.server` on Instance 6 |

---

## Important Notes

- **Do not use raw IP addresses** in the frontend — always go through the Cloudflare tunnel
- **No API key** is required to hit the vLLM model servers (no `--api-key` flag was used)
- **Quick tunnels have no uptime guarantee** — for production, set up named Cloudflare tunnels with a real account
- All models run `vllm==0.12.0` with `--enforce-eager` (no CUDA graphs), `bfloat16`, `max-model-len 2048`
- Backend repo: `https://github.com/shlawgathon/trenches`