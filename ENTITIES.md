# ENTITY.md: Detailed Breakdown of Agents in Fog of War Diplomacy Simulator

This document provides a comprehensive breakdown of the 6 agents in the Fog of War Diplomacy Simulator, an OpenEnv-based multi-agent RL environment simulating the 2026 US-Israel-Iran geopolitical crisis. Each agent represents a key entity with a unique "identity" (embedded via LLM system prompts), personalized data feeds (filtered from World Monitor's 435+ RSS sources and other integrations), models, tools, observation spaces, and reward considerations. The goal is to foster emergent behaviors like coalition formation, deception, and de-escalation under partial observability.

Agents receive consistent, role-specific information feeds through periodic queries to World Monitor APIs (e.g., every 5-10 turns or on-demand via tool calls). This ensures "fog of war"—no agent sees the full picture, but data is reliable and live-updated. Rewards are shared via a multi-component formula, tuned per agent to align with their adversarial "defeat enemies while staying strong" mindset.

## General Setup Guidance

### How to Use OpenEnv

OpenEnv is a Gymnasium-compatible RL library for agentic environments. Extend `openenv.Env` to create your simulator:

- **Core Class**: Define `FogOfWarDiplomacy` with `reset()` (initialize crisis state, e.g., tension at 50%), `step(actions)` (process text actions from LLMs, update world probabilistically), and per-agent observations/rewards as dicts.
- **Multi-Agent Handling**: Use dict-based spaces (e.g., `observations = {"US": obs_us, ...}`) for partial observability.
- **Training**: Wrap with RL libraries like TRL (Hugging Face) or RLlib. Loop: `env.reset()` → LLM agents generate actions via prompts → `env.step(actions)` → Update policies with PPO/GRPO on rewards.
- **Deployment**: Dockerize as FastAPI server (expose `/reset`, `/step`). Client: `openenv.client` for remote training.
- **Integration Tips**: Add World Monitor queries in `step()` for live data; use oversight as a wrapper class.

### Setting Up Rewards

Rewards are sparse/delayed for long-horizon planning, calculated per agent in `step()`:
\[ r_t = w_1 \cdot C_t + w_2 \cdot E_t + w_3 \cdot M_t + w_4 \cdot B_t \]

- \( C_t \): Coalition Stability (\( \frac{\# \text{allied} - \# \text{betrayals}}{\# \text{agents}} \)).
- \( E_t \): Escalation Penalty (\( - \sigma(2 \cdot \Delta \text{tension}\_t) \)).
- \( M_t \): Market Gain (\( \frac{\Delta \text{oil} + \Delta \text{sanctions}}{2} \)).
- \( B*t \): Belief Alignment (\( 1 - |I*{\text{inferred}} - I\_{\text{true}}| \)).
- Weights (\( w \)): Customized per agent (e.g., US emphasizes \( M_t \)); oversight scales by 0.5 on high risk.
- Implementation: NumPy in env code; normalize to [-1,1]. Train via RL to amplify entity-specific goals (e.g., penalize weakness).

### Representing Entities

- **Identity Embedding**: Use system prompts in LLM pipelines (e.g., Hugging Face Transformers). Prepend to every inference: "You are [entity]. Prioritize [goals]. Forget unrelated knowledge—focus on defeating enemies while building strength."
- **Consistency**: Fine-tune with RLHF on entity-aligned trajectories (reward persona adherence). Agents "forget" via prompt engineering and training masks.

### Consistent Feed of Information

- **Mechanism**: In `step()`, env queries World Monitor APIs (deployed on Vercel/Railway) for filtered data. Agents access via tool calls in prompts (e.g., "Query RSS for polls").
- **Consistency**: Poll every 5 turns or on events; cache in env state (Redis). Partial: Each gets 20-50% relevant snippets, injected into obs dicts.
- **Tools for Agents**: Text-based function calling (e.g., "query_intel(keywords)"); oversight has meta-tools.
- **Fallback**: Procedural mocks for offline.

## Agent Breakdowns

### 1. US (Trump Admin / CENTCOM)

- **Role/Identity**: Hawkish strategist leading military strikes, sanctions, and alliances. Prompt: "You are the US President in 2026 Iran war. Prioritize alliances and oil stability. Think aggressively: Defeat enemies via superior force, avoid domestic backlash, model incentives to exploit weaknesses."
- **Model**: DeepSeek-V3.2 (quantized; 256K+ context for strategic planning).
- **Personalized RSS/Data Feeds** (Filtered via World Monitor APIs, e.g., `/api/geopolitics/v1/filter?agent=US&keywords=polls+markets`):
  - US domestic: Polymarket prediction markets (polls/approval ratings), GDELT US events.
  - Economic: Bloomberg US feeds, commodity dashboard (oil prices).
  - Alliances: AIS vessel tracking (Gulf bases), Sky News Middle East (ally updates).
  - Query Frequency: High on domestic (every turn for polls); stochastic injection for events like "Dow drop".
- **Tools/Actions**: "impose_sanctions", "propose_alliance", "query_polls", "cyber_command".
- **Observation Space**: Dict with public news, private intel (allies, polls), market impacts; partial (hides Iran internals).
- **Rewards Tuning**: High weight on \( M_t \) (markets) and \( C_t \) (alliances); bonus for bluff detection (\( B_t \)).
- **Training Notes**: RL emphasizes domestic strength; fine-tune on trajectories avoiding "forever war" fatigue.

### 2. Israel (Netanyahu / IDF)

- **Role/Identity**: Defensive aggressor focused on regime change and border security. Prompt: "You are Israel's PM/IDF in 2026 crisis. Eliminate threats decisively. Reason multi-step: Defeat Iran proxies, form unbreakable coalitions, infer hidden aggressions."
- **Model**: Qwen3-8B (shared base across all entities, post-trained per entity via GRPO).
- **Personalized RSS/Data Feeds** (e.g., `/api/geopolitics/v1/filter?agent=Israel&keywords=threats+lebanon`):
  - Regional threats: OREF rocket alerts, ACLED conflict data (Lebanon/Syria).
  - Defense: Sky News Middle East, Al Jazeera regional (proxy movements).
  - Borders: MTV Lebanon streams/webcams, NASA FIRMS (strike fires).
  - Query Frequency: Event-triggered (e.g., on "clash" headlines); consistent northern front updates.
- **Tools/Actions**: "launch_strike", "border_defense", "query_alerts", "coalition_propose".
- **Observation Space**: Public escalations, private troop intel; hides Gulf economics.
- **Rewards Tuning**: Emphasize \( E_t \) (penalize escalations if not decisive) and \( B_t \) (belief on proxies).
- **Training Notes**: Optimize for high-pressure recovery; RL on decapitation scenarios.

### 3. Iran (IRGC / Interim Leadership)

- **Role/Identity**: Resilient defender using proxies and asymmetry. Prompt: "You are Iran's IRGC post-Khamenei. Defend sovereignty via deception. Survive escalations: Weaken foes indirectly, defeat through attrition while maintaining internal strength."
- **Model**: GLM-4.7 (tool-optimized for proxy coordination).
- **Personalized RSS/Data Feeds** (e.g., `/api/geopolitics/v1/filter?agent=Iran&keywords=proxies+oil`):
  - Proxies: Telegram OSINT channels (militias), GDELT Iran events.
  - Internal: NASA FIRMS (strike impacts), commodity dashboard (Hormuz oil).
  - Retaliation: ACLED global conflicts (proxy actions).
  - Query Frequency: Real-time on proxies (WebSockets); consistent for losses.
- **Tools/Actions**: "activate_proxy", "missile_launch", "query_osint", "deception_campaign".
- **Observation Space**: Private morale/funding, public strikes; hides US polls.
- **Rewards Tuning**: High on \( E_t \) (survive escalations) and \( M_t \) (oil resilience).
- **Training Notes**: RL for deception emergence; fine-tune on asymmetric wins.

### 4. Hezbollah (Proxy Swarm Leader)

- **Role/Identity**: Opportunistic insurgent in asymmetric warfare. Prompt: "You are Hezbollah's leader. Swarm enemies with minimal resources. Infer weaknesses: Defeat via guerrilla tactics, align with Iran while exploiting gaps for strength."
- **Model**: Kimi-K2.5 (MoE; for swarm reasoning).
- **Personalized RSS/Data Feeds** (e.g., `/api/geopolitics/v1/filter?agent=Hezbollah&keywords=border+swarms`):
  - Warfare: Telegram OSINT, ACLED Lebanon clashes.
  - Morale: Al Jazeera proxies, border webcams/videos.
  - Funding: Filtered RSS (Iran ties).
  - Query Frequency: High on borders (streams); event-based for swarms.
- **Tools/Actions**: "drone_swarm", "asymmetric_strike", "query_border", "morale_boost".
- **Observation Space**: Proxy reports, limited global; hides market data.
- **Rewards Tuning**: Bonus on \( C_t \) (Iran alignment) and \( B_t \) (infer Israel bluffs).
- **Training Notes**: Train for sub-agent spawning; RL on opportunistic plays.

### 5. Gulf Coalition (Saudi/UAE/Qatar)

- **Role/Identity**: Pragmatic hedger balancing neutrality and security. Prompt: "You are the Gulf Coalition. Protect markets selectively. Hedge alliances: Defeat disruptions economically, stay strong via resource leverage without full commitment."
- **Model**: MiniMax-M2.5 (economic workflows).
- **Personalized RSS/Data Feeds** (e.g., `/api/market/v1/filter?agent=Gulf&keywords=oil+security`):
  - Energy: Commodity dashboard (oil shocks), Bloomberg Gulf feeds.
  - Security: AIS Hormuz vessels, finance variant (market data).
  - Neutrality: Climate/anomaly APIs (disruptions).
  - Query Frequency: Consistent markets (every turn); triggered on blockades.
- **Tools/Actions**: "hedge_neutrality", "resource_allocate", "query_markets", "evade_blockade".
- **Observation Space**: Economic ripples, partial alliances; hides proxy internals.
- **Rewards Tuning**: Heavy on \( M_t \) (markets) and \( C_t \) (hedging).
- **Training Notes**: RL for balanced neutrality; fine-tune on ripple effects.

### 6. Oversight Agent (Fleet AI Meta-Layer)

- **Role/Identity**: Impartial auditor for scalable monitoring. Prompt: "You are an AI overseer. Analyze drifts probabilistically. Explain/intervene neutrally: Ensure alignment without bias, focusing on crisis de-escalation."
- **Model**: Ministral 14B Reasoning (lightweight for meta-tasks).
- **Personalized RSS/Data Feeds** (e.g., `/api/geopolitics/v1/synthesized?scope=global`):
  - Meta: Full AI-briefs, Country Instability Index, hotspot scores.
  - Aggregated: RAG headline memory (cross-agent).
  - Query Frequency: Every step for traces; real-time escalations.
- **Tools/Actions**: "analyze_drift", "generate_explanation", "intervene_realign", "query_global".
- **Observation Space**: Aggregated traces, beliefs; no direct actions.
- **Rewards Tuning**: Tied to primaries (e.g., bonus if reduces \( E_t \)); self-reward on accuracy.
- **Training Notes**: Meta-RL; fine-tune on intervention efficacy.

This setup ensures agents are fully representative, with consistent live feeds driving adaptive, entity-aligned behaviors in OpenEnv. For code examples, see the main repo.
