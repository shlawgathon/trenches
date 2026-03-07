The frontend now presents the RL environment as a black intelligence-style operator console instead of a generic dashboard. The main map in src/components/CommandMap.tsx:44 was restyled toward the
WorldMonitor look: darker basemap treatment, suppressed civilian labels, stronger fog/space treatment, and scanline/grid atmosphere. The app shell in src/App.tsx:161 now builds a monitoring snapshot per
agent from live session state and renders the monitoring deck beside the map, so the user can watch reward pressure, source health, recent actions, and model posture in one place. IMPROVEMENTS.md is fully
written at IMPROVEMENTS.md:1.

On the backend, reward shaping is no longer mostly shared. backend/src/trenches_env/rl.py:203 now defines doctrine-specific strategic state baselines and per-actor action effects, and backend/src/
trenches_env/env.py:222 now carries persistent actor_state, applies signal pressure and action pressure into that state, exposes it in observations, flattens geolocated assets for model/viewer use, and
computes unique reward functions for each entity at backend/src/trenches_env/env.py:892. Type surfaces were aligned in src/lib/types.ts:45, reward coverage was extended in backend/tests/
test_reward_differentiation.py:5, and the source manifest was regenerated in backend/src/trenches_env/source_manifest.json so the Israel/Hezbollah source specialization matches runtime.

What still needs to be done: the monitoring deck supports tool inventory, but entity tools.json packs are not wired into observations yet, so that part of the UI will stay empty until the tool layer is
integrated. The model labels in the monitoring view are still product-style placeholders, not final checkpoint selections. The map requires VITE_MAPBOX_TOKEN for full rendering, and the frontend still has a
large-bundle warning on build, so code-splitting is still worth doing before the UI grows further. The bigger roadmap items in IMPROVEMENTS.md:5 are still future work: hidden-world engine, benchmark/
curriculum harness, and deeper replay/comparison observability.

Verification: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev python -m pytest passed with 14 passed, 1 skipped. npm run typecheck passed. npm run build passed, with the existing Vite chunk-size warning
only.