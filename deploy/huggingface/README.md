# Hugging Face Space Wrapper

These files are the final single-port wrapper for the intended production frontend stack:

- Next.js 16+
- React
- Tailwind CSS
- Bun
- FastAPI backend
- Nginx reverse proxy on port `7860`

Expected internal ports:

- frontend: `3000`
- backend: `8000`
- public Space port: `7860`

Expected frontend scripts in `package.json`:

```json
{
  "scripts": {
    "build": "bun --bun next build",
    "start": "bun --bun next start"
  }
}
```

Expected Hugging Face README front matter:

```yaml
---
title: Trenches
sdk: docker
app_port: 7860
---
```

Why this shape:

- Hugging Face Docker Spaces expose one public port, but allow multiple internal ports.
- Nginx fronts the app on `7860` and dispatches `/` to Next and `/api` or `/openenv` to the backend.

Before using this wrapper, the frontend migration away from the current Vite app must be complete.
