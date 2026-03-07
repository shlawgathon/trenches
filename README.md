# trench
A OpenEnv environment that trains agents to handle complex, adversarial multi-actor global crisis 


# PLAN.md: Comprehensive Development Plan for Fog of War Diplomacy Simulator

## Introduction

The Fog of War Diplomacy Simulator is a cutting-edge multi-agent reinforcement learning (RL) environment built for the OpenEnv Hackathon, targeting **Statement 1: Multi-Agent Interactions** with a strong emphasis on the **Fleet AI Sub-Theme: Scalable Oversight**. This project simulates the dynamic and volatile 2026 US-Israel-Iran geopolitical crisis, incorporating real-time elements such as airstrikes on Tehran, missile retaliations across the Gulf, threats to the Strait of Hormuz, leadership assassinations (e.g., fallout from Khamenei's death), US naval engagements (e.g., submarine sinkings of Iranian vessels), domestic political turmoil (e.g., Trump's oustings of figures like Noem and Mullin), market volatility (e.g., Dow drops of 800+ points), and shifting public opinion (e.g., 59% disapproval polls).

The simulator creates a partially observable "fog-of-war" world where LLM agents negotiate coalitions, manage deceptions, and respond to stochastic events drawn from live global news feeds. A dedicated oversight agent monitors behaviors scalably, intervening to prevent escalations. By blending live data from a forked World Monitor repository (aggregating 435+ RSS feeds, Telegram OSINT, video streams, and structured APIs), the environment pushes LLM training boundaries in adversarial, long-horizon settings.

This PLAN.md serves as a detailed roadmap to pull together all discussions from our chats. It outlines the scope, step-by-step implementation, rationale, challenges, and milestones. The plan is divided into phases: (1) Core OpenEnv Setup with Agents and Identities, (2) Tools and RSS Feed Integration, (3) Rewards Logic, (4) Training and Live Sessions, and (5) Supporting Infrastructure (Deployment, Dashboard, Testing). The goal is a hackathon-ready prototype by March 7-8, 2026, with potential for $10,000 USD bonus prizes under Fleet AI.

### Project Goals and Non-Goals
- **Goals**:
    - Build a reproducible, Dockerized OpenEnv environment that trains LLMs for emergent multi-agent behaviors in crises.
    - Integrate live news for adaptive simulations, solving the untried challenge of real-time geopolitical oversight.
    - Demonstrate scalable oversight with explainable interventions.
    - Achieve high hackathon impact: Difficult, novel, and value-adding for LLM training in multi-actor settings.
- **Non-Goals**:
    - Full-scale production deployment (focus on demo/training loops).
    - Ethical AI enforcement beyond oversight mechanics.
    - Integration with proprietary tools (stick to open-source LLMs and APIs).

### High-Level Timeline
- **Week 1 (Pre-Hackathon Prep)**: Fork repos, set up local dev, implement core env and agents.
- **Hackathon Days (March 7-8)**: Integrate tools/RSS, rewards, training loops; build dashboard; test live sessions.
- **Post-Hackathon**: Upload to HF Hub, refine for judging.

## Phase 1: Core OpenEnv Setup with Agents and Identities

### Overview
Start by creating the foundational OpenEnv environment. OpenEnv's Gymnasium-compatible API (e.g., `reset()`, `step()`) will simulate the crisis world, with multi-agent dict-based observations/actions for partial observability. Agents will be represented by specific LLMs, each embedded with a unique "identity" via system prompts to ensure role-aligned behaviors (e.g., adversarial focus on "defeating enemies while staying strong").

### Step-by-Step Implementation
1. **Install Dependencies**:
    - `pip install openenv-core gymnasium fastapi uvicorn requests numpy torch transformers trl unsloth` (for RL, inference, training).
    - Fork and clone World Monitor (https://github.com/koala73/worldmonitor) for data sidecar.

2. **Define Environment Class**:
    - Create `fog_of_war_diplomacy.py` extending `openenv.Env`.
    - Global State: Dict with `tension_level` (0-100), `coalitions` (graph of alliances), `events_queue` (stochastic black-swans from news).
    - `reset()`: Initialize to baseline crisis (e.g., tension=50, random hidden incentives).
    - `step(actions)`: Process dict of agent actions (text/tool calls), update state probabilistically (e.g., NumPy rand for outcomes), compute per-agent rewards/obs.

3. **Define 6 Agents**:
    - Agents: US, Israel, Iran, Hezbollah, Gulf Coalition, Oversight (as detailed in ENTITY.md).
    - Models: Load via Hugging Face (quantized for efficiency, e.g., AWQ/INT4).
        - US: DeepSeek-V3.2
        - Israel: Qwen3.5-397B-A17B
        - Iran: GLM-4.7
        - Hezbollah: Kimi-K2.5
        - Gulf: MiniMax-M2.5
        - Oversight: Ministral 14B Reasoning
    - Inference Setup: In training loops, use pipelines like `pipeline("text-generation", model=...)` for action generation.

4. **Embed Identities**:
    - System Prompts: Prepend to every LLM call (e.g., "You are [entity]. Focus on defeating enemies while building strength. Forget unrelated knowledge.").
    - Consistency: During RL, penalize deviations via reward shaping; use few-shot examples in prompts for initial alignment.
    - Rationale: Ensures agents act as "full representatives" (e.g., US hawkish on polls; Iran deceptive via proxies).

5. **Partial Observability**:
    - Obs Dict: Per-agent (e.g., {"public_news": "...", "private_intel": hidden incentives, "tool_results": []}).
    - Fog Mechanics: Hide others' states (e.g., US sees polls but not Iran's morale); inject noise via probabilistic filters.

### Rationale and Challenges
- **Rationale**: This phase establishes the multi-agent core, aligning with Statement 1's focus on cooperation/competition in partially observable settings. Identities drive emergent behaviors without hardcoding.
- **Challenges**: Balancing state space explosion (6 agents → high dims); mitigate with sparse rewards and oversight. Potential instability in long-horizon episodes (1000+ turns)—use curriculum learning.
- **Milestone**: Functional env with dummy agents stepping through 10 episodes.

## Phase 2: Tools and RSS Feed Integration

### Overview
Agents will use tools (as in TOOLS.md) for actions/intel gathering, with RSS feeds providing consistent, live information. Tools enable function-calling, while RSS integration via World Monitor ensures role-specific data without overwhelming the env.

### Step-by-Step Implementation
1. **Tool Framework**:
    - Parse Function Calls: In `step()`, use JSON/regex to extract calls from LLM outputs (e.g., {"tool": "query_intel", "params": {...}}).
    - Common Tools: Implement `query_intel`, `analyze_belief`, `propose_negotiation` (simulate outcomes, update coalitions).
    - Agent-Specific: Add exclusives (e.g., US: `impose_sanctions` → +tension for target).
    - Tool Results: Inject into next obs; add costs (e.g., -0.05 reward for queries).

2. **RSS/Data Integration**:
    - Deploy World Monitor: Fork repo, deploy to Vercel (edge functions for RSS polling) + Railway (relays for Telegram/streams).
    - Aggregation: World Monitor polls 435+ RSS (e.g., Bloomberg, Al Jazeera) every 5-10 mins, caches in Redis, exposes APIs (e.g., `/api/geopolitics/v1/filter`).
    - Per-Agent Filtering: In `query_intel`, call API with `?agent=US&keywords=polls` → returns tailored snippets (e.g., US: Polymarket polls; Iran: Telegram proxies).
    - Consistency: Poll on `reset()`/every 5 steps; cache in env state. Agents "decide" queries via prompts (e.g., "If tension >50, query enemy movements").
    - Fallbacks: Procedural mocks (e.g., random headlines) for offline.

3. **Live News Injection**:
    - Stochastic Events: Parse RSS results → trigger events (e.g., "strike" headline → +20 tension with p=0.7).
    - Rationale: Ties simulation to real 2026 trends, making agents adaptive.

### Rationale and Challenges
- **Rationale**: Tools + RSS create a realistic, data-driven world, solving partial observability by giving agents "what they want" (role-specific intel). This is untried at scale, adding hackathon novelty.
- **Challenges**: API latency (mitigate with caching); data overload (limit to 5-10 snippets per query). Ensure no full feeds—maintain fog.
- **Milestone**: Agents successfully query tools/RSS in a simulated episode, updating states.

## Phase 3: Rewards Logic

### Overview
Rewards drive learning, using a multi-component formula tailored per agent. Sparse/delayed to encourage long-horizon planning, with oversight modulation for stability.

### Step-by-Step Implementation
1. **Core Formula**:
    - Per-agent \( r_t = 0.3 C_t + 0.4 E_t + 0.2 M_t + 0.1 B_t \) (normalized [-1,1]).
    - Components: Coalition Stability (\( C_t \): allied vs. betrayals), Escalation Penalty (\( E_t \): sigmoid on tension delta), Market Gain (\( M_t \): oil/sanctions delta), Belief Alignment (\( B_t \): inferred vs. true incentives).
    - Custom Weights: E.g., US high on \( M_t \); Iran on \( E_t \).

2. **Oversight Integration**:
    - Risk Formula: Belief update \( B'(s') = \eta \sum P(o|a,s') T(s,a,s') B(s) \); Risk \( R = \sum B' U(s',a) \sigma(\beta (I_self - I_other)) \).
    - If R > 0.5, scale r_t by 0.5 or intervene (force re-action).

3. **Computation**:
    - In `step()`: NumPy for calcs; return rewards dict.
    - Shared Elements: Partial coalition rewards for cooperation.

### Rationale and Challenges
- **Rationale**: Aligns with Fleet AI—rewards scalable oversight while preventing reward hacking. Sparse nature fits super-long horizons.
- **Challenges**: Credit assignment in multi-agent (use centralized critic); tuning weights (hyperparam search via Optuna).
- **Milestone**: Rewards computed correctly in episodes, with oversight interventions.

## Phase 4: Training and Live Sessions

### Overview
Proceed to RL post-training and live demos, where agents act on incoming news. Use TRL/Unsloth for efficiency, with sessions handling real-time data.

### Step-by-Step Implementation
1. **Training Loop**:
    - Client Script: `client = openenv.Client('http://localhost:8000')`; Loop reset/step with LLM actions.
    - Methods: PPO for basics, GRPO for multi-agent; curriculum (start short episodes, scale to 1000+).
    - Data-Driven: Inject live RSS during steps → agents adapt (e.g., poll shifts US actions).

2. **Live Sessions**:
    - Mode: Run infinite-horizon "live" mode (no reset, continuous steps).
    - News Handling: WebSocket from World Monitor for real-time pushes → trigger events mid-session.
    - Demo: Agents negotiate based on fresh headlines (e.g., "missile strike" → Iran retaliates).

3. **Evaluation**:
    - Metrics: De-escalation rate, coalition stability, oversight interventions.
    - Logs: Trace actions/rewards for analysis.

### Rationale and Challenges
- **Rationale**: Trains for real crises; live sessions showcase adaptability, tying to hackathon's multi-actor focus.
- **Challenges**: Compute for large models (use Unsloth quantization); non-stationarity from live data (stabilize with buffers).
- **Milestone**: Trained agents in 5 episodes; live demo with simulated news.

## Phase 5: Supporting Infrastructure

### Deployment and Docker
- Full Dockerization: Env in FastAPI container; inference via vLLM Docker; Compose for orchestration.
- HF Hub: Upload env as Docker Space.

### Dashboard
- React + Vite: Military theme, tabs per agent, Plotly maps, live feeds.
- Deployment: Vercel for frontend, querying env API.

### Testing and Evaluation
- Unit: Test env steps, tool parsing.
- Integration: End-to-end episodes with mocks.
- Edge Cases: High tension collapses, oversight failures.

### Potential Extensions
- Add procedural proxies.
- Multimodal (view_x_video for streams).
- Ethical audits.

This plan scopes a robust, innovative project—let's iterate as you build!