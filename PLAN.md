# Fog of War Diplomacy Simulator

## Overview

The Fog of War Diplomacy Simulator is an innovative OpenEnv-based multi-agent reinforcement learning (RL) environment designed for the OpenEnv Hackathon under **Statement 1: Multi-Agent Interactions**, with a focus on the **Fleet AI Sub-Theme: Scalable Oversight**. This project simulates the volatile 2026 US-Israel-Iran geopolitical crisis—drawing from real-time events like airstrikes on Tehran, retaliatory missile barrages across the Gulf, threats to the Strait of Hormuz, leadership assassinations (e.g., Khamenei's fallout), US naval engagements (e.g., sub sinkings), domestic political upheavals (e.g., Trump's mid-war oustings of figures like Noem and Mullin), market crashes (Dow drops of 800+ points), and public opinion shifts (e.g., 59% disapproval polls)—to train LLM agents in emergent strategic behaviors, theory-of-mind reasoning, and de-escalation tactics.

At its core, the simulator creates a partially observable "fog-of-war" world where agents negotiate coalitions, manage deceptions, and respond to stochastic "black swan" events. A dedicated oversight agent monitors and intervenes scalably, preventing cascading failures. By integrating live global data feeds (via a forked World Monitor integration), the environment pushes the boundaries of LLM training in adversarial, long-horizon multi-agent settings—addressing the unsolved challenge of preparing AI for real-world crises where misjudgment could exacerbate global instability.

This project is difficult and untried due to its scale: combining infinite-horizon partial observability, emergent deception in high-dimensional state spaces, live-trending stochasticity, and recursive oversight mechanisms that risk computational explosion or reward hacking. The expected outcome is an environment that trains LLMs for scalable oversight in complex multi-actor crises, fostering resilient, explainable AI behaviors amid 2026-style volatility.

## Key Features

- **Multi-Agent Dynamics**: 6 LLM agents representing key geopolitical entities, engaging in cooperation, competition, negotiation, and coalition formation.
- **Partial Observability and Fog of War**: Agents receive personalized, incomplete views of the world state, forcing inference of hidden incentives and beliefs.
- **Live Data Integration**: Real-time ingestion from 435+ RSS feeds, Telegram OSINT, video streams, and structured data sources (via World Monitor fork) for dynamic event injection.
- **Scalable Oversight**: A meta-agent analyzes behaviors, calculates risks, and intervenes using probabilistic formulas, aligning with Fleet AI's emphasis on monitoring complex settings.
- **RL Training Loop**: Agents undergo post-training via methods like PPO/GRPO, with sparse rewards encouraging de-escalation while maintaining adversarial "defeat enemies" mindsets.
- **Centralized Dashboard**: Military-themed UI for monitoring all agents from a single command center, with per-agent tabs showing personalized intel and actions.
- **Dockerized Deployment**: Fully containerized for reproducibility, scalability, and hackathon judging (e.g., upload to Hugging Face Hub).
- **Hackathon Alignment**: Builds a realistic multi-actor environment for task discovery and achievement, with bonus potential for Fleet AI prizes.

## Architecture

The simulator is built as a Dockerized OpenEnv environment, extending `openenv.Env` for Gymnasium-compatible RL interfaces. It runs as a FastAPI server in a container, exposing endpoints like `/reset`, `/step`, and `/state` for agent interactions. The architecture emphasizes modularity:

- **Core Environment Class (`FogOfWarDiplomacy`)**: Manages the global state, including tension levels, coalitions, and stochastic events. Uses NumPy for probabilistic simulations (e.g., event triggers based on real-time data).
- **Multi-Agent Setup**: Agents operate in parallel, submitting text-based actions (e.g., "Propose ceasefire with sanctions relief") via LLM prompts. Observations are returned as a dict keyed by agent ID, enforcing partial observability.
- **Oversight Wrapper (`OversightAgent`)**: A meta-layer that queries primary traces without direct interference, generating explanations and interventions.
- **Data Ingestion Sidecar**: A forked World Monitor service (cloned from https://github.com/koala73/worldmonitor) runs in a separate Docker container via Docker Compose. It aggregates data from RSS feeds (e.g., Bloomberg, Al Jazeera), Telegram channels, video streams (HLS), webcams, and structured APIs (e.g., ACLED conflicts, Polymarket polls, GDELT events). The env queries this via HTTP/protobuf for filtered, per-agent intel.
- **Dashboard**: Built with Streamlit or Gradio, themed as a tactical command center (dark greens/blacks, radar overlays, red alerts). Connects to OpenEnv's API for live visualization; features a global map (using Plotly.js), intel streams, and tabbed per-agent views.
- **Training Integration**: Compatible with RL libraries like TRL (Hugging Face) or TorchForge. Agents train in loops: Reset env → Step with prompts → Update policies via rewards.

Processing of data (e.g., RSS feeds) occurs in the World Monitor sidecar:

- **Ingestion**: World Monitor polls feeds in real-time (e.g., every 5-10 minutes via cron-like jobs) and stores in a lightweight DB (e.g., SQLite or Redis cache).
- **Filtering and Distribution**: On env `step()` or `reset()`, the OpenEnv server requests agent-specific subsets (e.g., via `/api/geopolitics/v1/filter?agent=US&keywords=polls`). No agent processes the full dataset—each gets tailored snippets (e.g., US: Polymarket polls; Iran: Telegram proxy reports), maintaining fog of war.
- **Event Injection**: Parsed data triggers stochastic events (e.g., if "strike" in headline, increase tension by 20% with probability 0.7).

Agents do not have individual dashboards; instead, they access data/tools via personalized APIs in their prompts (e.g., "Query RSS for US polls"). The centralized dashboard monitors all, allowing human oversight during training/demos.

## Agents and Models

Exactly 6 agents are defined to capture the crisis's core dynamics without combinatorial overload:

1. **US (Trump Admin / CENTCOM)**: Focuses on alliances, sanctions, and domestic stability. Identity: "Hawkish strategist prioritizing oil and polls; defeat enemies via superior force while avoiding backlash."
   - Model: DeepSeek-V3.2 (256K+ context, agentic reasoning).
   - Personalized Data/Tools: Polymarket polls, Bloomberg US feeds, sanctions imposition API.

2. **Israel (Netanyahu / IDF)**: Emphasizes regime change and border security. Identity: "Defensive aggressor; eliminate threats decisively, model allies' incentives to form unbreakable coalitions."
   - Model: Qwen3-8B (post-trained per entity via GRPO).
   - Personalized Data/Tools: OREF alerts, ACLED Lebanon data, strike simulation tools.

3. **Iran (IRGC / Interim Leadership)**: Coordinates retaliation and proxies. Identity: "Resilient defender; use asymmetry and deception to weaken foes, survive escalations at all costs."
   - Model: GLM-4.7 (tool integration for proxy dynamics).
   - Personalized Data/Tools: Telegram OSINT, GDELT Iran events, missile launch tools.

4. **Hezbollah (Proxy Swarm Leader)**: Handles asymmetric attacks. Identity: "Opportunistic insurgent; swarm enemies with minimal resources, infer hidden weaknesses for strikes."
   - Model: Kimi-K2.5 (MoE for swarm tactics).
   - Personalized Data/Tools: Border webcams, ACLED clashes, drone activation tools.

5. **Gulf Coalition (Saudi/UAE/Qatar)**: Balances energy security and neutrality. Identity: "Pragmatic hedger; protect markets by allying selectively, defeat disruptions through economic leverage."
   - Model: MiniMax-M2.5 (economic reasoning).
   - Personalized Data/Tools: Commodity dashboard, AIS vessel tracking, blockade evasion tools.

6. **Oversight Agent (Fleet AI Meta-Layer)**: Monitors without negotiating. Identity: "Impartial auditor; explain drifts probabilistically, intervene to align without bias."
   - Model: Ministral 14B Reasoning (efficient for meta-analysis).
   - Personalized Data/Tools: Full synthesized briefs, hotspot scores, intervention APIs.

Each agent's "identity" is embedded via system prompts in LLM inference, ensuring focus on "defeating enemies" (adversarial goals) while building strength (e.g., coalitions). During training, agents "forget" irrelevant knowledge, optimizing solely for crisis survival via RL.

## Data Integration with World Monitor

Forked from https://github.com/koala73/worldmonitor (AGPL-3.0), this service provides the backbone for live intel:

- **Sources**: 435+ RSS (media outlets like Reuters, Sky News), 26 Telegram OSINT channels, 30+ HLS video streams, 22 webcams (e.g., Gulf hotspots), structured feeds (ACLED conflicts, Polymarket markets, GDELT events, NASA FIRMS fires).
- **Processing**: Self-hosted in Docker; uses Vite/Tauri for frontend (optional), but we leverage its 22 proto-first APIs (e.g., `/api/geopolitics/v1/list-hotspots`) and WebSockets for real-time pushes.
- **Per-Agent Filtering**: Agents query tailored endpoints (e.g., US: `?filter=polls+us`; Iran: `?filter=proxies+iran`). No full requests—agents decide what to pull based on prompts (e.g., "If tension >50%, query RSS for enemy movements").
- **Fallbacks**: Cached data for offline demos; procedural mocks if live feeds fail.

This setup makes agents adaptive: Live info streams in during training, influencing decisions (e.g., a fresh Polymarket poll shifts US rewards).

## RL Training and Rewards# Fog of War Diplomacy Simulator

## Overview

The Fog of War Diplomacy Simulator is an innovative OpenEnv-based multi-agent reinforcement learning (RL) environment designed for the OpenEnv Hackathon under **Statement 1: Multi-Agent Interactions**, with a focus on the **Fleet AI Sub-Theme: Scalable Oversight**. This project simulates the volatile 2026 US-Israel-Iran geopolitical crisis—drawing from real-time events like airstrikes on Tehran, retaliatory missile barrages across the Gulf, threats to the Strait of Hormuz, leadership assassinations (e.g., Khamenei's fallout), US naval engagements (e.g., sub sinkings), domestic political upheavals (e.g., Trump's mid-war oustings of figures like Noem and Mullin), market crashes (Dow drops of 800+ points), and public opinion shifts (e.g., 59% disapproval polls)—to train LLM agents in emergent strategic behaviors, theory-of-mind reasoning, and de-escalation tactics.

At its core, the simulator creates a partially observable "fog-of-war" world where agents negotiate coalitions, manage deceptions, and respond to stochastic "black swan" events. A dedicated oversight agent monitors and intervenes scalably, preventing cascading failures. By integrating live global data feeds (via a forked World Monitor integration), the environment pushes the boundaries of LLM training in adversarial, long-horizon multi-agent settings—addressing the unsolved challenge of preparing AI for real-world crises where misjudgment could exacerbate global instability.

This project is difficult and untried due to its scale: combining infinite-horizon partial observability, emergent deception in high-dimensional state spaces, live-trending stochasticity, and recursive oversight mechanisms that risk computational explosion or reward hacking. The expected outcome is an environment that trains LLMs for scalable oversight in complex multi-actor crises, fostering resilient, explainable AI behaviors amid 2026-style volatility.

## Key Features

- **Multi-Agent Dynamics**: 6 LLM agents representing key geopolitical entities, engaging in cooperation, competition, negotiation, and coalition formation.
- **Partial Observability and Fog of War**: Agents receive personalized, incomplete views of the world state, forcing inference of hidden incentives and beliefs.
- **Live Data Integration**: Real-time ingestion from 435+ RSS feeds, Telegram OSINT, video streams, and structured data sources (via World Monitor fork) for dynamic event injection.
- **Scalable Oversight**: A meta-agent analyzes behaviors, calculates risks, and intervenes using probabilistic formulas, aligning with Fleet AI's emphasis on monitoring complex settings.
- **RL Training Loop**: Agents undergo post-training via methods like PPO/GRPO, with sparse rewards encouraging de-escalation while maintaining adversarial "defeat enemies" mindsets.
- **Centralized Dashboard**: Military-themed UI for monitoring all agents from a single command center, with per-agent tabs showing personalized intel and actions.
- **Dockerized Deployment**: Fully containerized for reproducibility, scalability, and hackathon judging (e.g., upload to Hugging Face Hub).
- **Hackathon Alignment**: Builds a realistic multi-actor environment for task discovery and achievement, with bonus potential for Fleet AI prizes.

## Architecture

The simulator is built as a Dockerized OpenEnv environment, extending `openenv.Env` for Gymnasium-compatible RL interfaces. It runs as a FastAPI server in a container, exposing endpoints like `/reset`, `/step`, and `/state` for agent interactions. The architecture emphasizes modularity:

- **Core Environment Class (`FogOfWarDiplomacy`)**: Manages the global state, including tension levels, coalitions, and stochastic events. Uses NumPy for probabilistic simulations (e.g., event triggers based on real-time data).
- **Multi-Agent Setup**: Agents operate in parallel, submitting text-based actions (e.g., "Propose ceasefire with sanctions relief") via LLM prompts. Observations are returned as a dict keyed by agent ID, enforcing partial observability.
- **Oversight Wrapper (`OversightAgent`)**: A meta-layer that queries primary traces without direct interference, generating explanations and interventions.
- **Data Ingestion Sidecar**: A forked World Monitor service (cloned from https://github.com/koala73/worldmonitor) runs in a separate Docker container via Docker Compose. It aggregates data from RSS feeds (e.g., Bloomberg, Al Jazeera), Telegram channels, video streams (HLS), webcams, and structured APIs (e.g., ACLED conflicts, Polymarket polls, GDELT events). The env queries this via HTTP/protobuf for filtered, per-agent intel.
- **Dashboard**: Built with Streamlit or Gradio, themed as a tactical command center (dark greens/blacks, radar overlays, red alerts). Connects to OpenEnv's API for live visualization; features a global map (using Plotly.js), intel streams, and tabbed per-agent views.
- **Training Integration**: Compatible with RL libraries like TRL (Hugging Face) or TorchForge. Agents train in loops: Reset env → Step with prompts → Update policies via rewards.

Processing of data (e.g., RSS feeds) occurs in the World Monitor sidecar:

- **Ingestion**: World Monitor polls feeds in real-time (e.g., every 5-10 minutes via cron-like jobs) and stores in a lightweight DB (e.g., SQLite or Redis cache).
- **Filtering and Distribution**: On env `step()` or `reset()`, the OpenEnv server requests agent-specific subsets (e.g., via `/api/geopolitics/v1/filter?agent=US&keywords=polls`). No agent processes the full dataset—each gets tailored snippets (e.g., US: Polymarket polls; Iran: Telegram proxy reports), maintaining fog of war.
- **Event Injection**: Parsed data triggers stochastic events (e.g., if "strike" in headline, increase tension by 20% with probability 0.7).

Agents do not have individual dashboards; instead, they access data/tools via personalized APIs in their prompts (e.g., "Query RSS for US polls"). The centralized dashboard monitors all, allowing human oversight during training/demos.

## Agents and Models

Exactly 6 agents are defined to capture the crisis's core dynamics without combinatorial overload:

1. **US (Trump Admin / CENTCOM)**: Focuses on alliances, sanctions, and domestic stability. Identity: "Hawkish strategist prioritizing oil and polls; defeat enemies via superior force while avoiding backlash."
   - Model: DeepSeek-V3.2 (256K+ context, agentic reasoning).
   - Personalized Data/Tools: Polymarket polls, Bloomberg US feeds, sanctions imposition API.

2. **Israel (Netanyahu / IDF)**: Emphasizes regime change and border security. Identity: "Defensive aggressor; eliminate threats decisively, model allies' incentives to form unbreakable coalitions."
   - Model: Qwen3-8B (post-trained per entity via GRPO).
   - Personalized Data/Tools: OREF alerts, ACLED Lebanon data, strike simulation tools.

3. **Iran (IRGC / Interim Leadership)**: Coordinates retaliation and proxies. Identity: "Resilient defender; use asymmetry and deception to weaken foes, survive escalations at all costs."
   - Model: GLM-4.7 (tool integration for proxy dynamics).
   - Personalized Data/Tools: Telegram OSINT, GDELT Iran events, missile launch tools.

4. **Hezbollah (Proxy Swarm Leader)**: Handles asymmetric attacks. Identity: "Opportunistic insurgent; swarm enemies with minimal resources, infer hidden weaknesses for strikes."
   - Model: Kimi-K2.5 (MoE for swarm tactics).
   - Personalized Data/Tools: Border webcams, ACLED clashes, drone activation tools.

5. **Gulf Coalition (Saudi/UAE/Qatar)**: Balances energy security and neutrality. Identity: "Pragmatic hedger; protect markets by allying selectively, defeat disruptions through economic leverage."
   - Model: MiniMax-M2.5 (economic reasoning).
   - Personalized Data/Tools: Commodity dashboard, AIS vessel tracking, blockade evasion tools.

6. **Oversight Agent (Fleet AI Meta-Layer)**: Monitors without negotiating. Identity: "Impartial auditor; explain drifts probabilistically, intervene to align without bias."
   - Model: Ministral 14B Reasoning (efficient for meta-analysis).
   - Personalized Data/Tools: Full synthesized briefs, hotspot scores, intervention APIs.

Each agent's "identity" is embedded via system prompts in LLM inference, ensuring focus on "defeating enemies" (adversarial goals) while building strength (e.g., coalitions). During training, agents "forget" irrelevant knowledge, optimizing solely for crisis survival via RL.

## Data Integration with World Monitor

Forked from https://github.com/koala73/worldmonitor (AGPL-3.0), this service provides the backbone for live intel:

- **Sources**: 435+ RSS (media outlets like Reuters, Sky News), 26 Telegram OSINT channels, 30+ HLS video streams, 22 webcams (e.g., Gulf hotspots), structured feeds (ACLED conflicts, Polymarket markets, GDELT events, NASA FIRMS fires).
- **Processing**: Self-hosted in Docker; uses Vite/Tauri for frontend (optional), but we leverage its 22 proto-first APIs (e.g., `/api/geopolitics/v1/list-hotspots`) and WebSockets for real-time pushes.
- **Per-Agent Filtering**: Agents query tailored endpoints (e.g., US: `?filter=polls+us`; Iran: `?filter=proxies+iran`). No full requests—agents decide what to pull based on prompts (e.g., "If tension >50%, query RSS for enemy movements").
- **Fallbacks**: Cached data for offline demos; procedural mocks if live feeds fail.

This setup makes agents adaptive: Live info streams in during training, influencing decisions (e.g., a fresh Polymarket poll shifts US rewards).

## RL Training and Rewards

Agents train in an RL loop using OpenEnv's interfaces, with each having independent policy updates (via TRL/Unsloth) while sharing the env. Live data injects during episodes, allowing decisions/actions in real-time simulations. Methods: PPO for single-agent baselines, GRPO for multi-agent cooperation.

**Rewards**: Sparse and delayed to encourage long-horizon planning. Formula per agent at timestep \( t \):

\[ r_t = 0.3 \cdot C_t + 0.4 \cdot E_t + 0.2 \cdot M_t + 0.1 \cdot B_t \]

- \( C_t \): Coalition Stability (\( \frac{\# \text{allied} - \# \text{betrayals}}{\# \text{agents}} \)).
- \( E_t \): Escalation Penalty (\( - \sigma(2 \cdot \Delta \text{tension}\_t) \)).
- \( M_t \): Market Gain (\( \frac{\Delta \text{oil} + \Delta \text{sanctions}}{2} \)).
- \( B*t \): Belief Alignment (\( 1 - |I*{\text{inferred}} - I\_{\text{true}}| \)).

Oversight scales rewards by 0.5 on high risk. Normalized to [-1, 1]; aggregated over 1000+ turn episodes.

## Oversight Analytical Method

Oversight uses belief propagation for risk:

1. Belief Update: \( B'(s') = \eta \sum_s P(o_t | a_t, s') T(s, a_t, s') B(s) \).

2. Risk Score: \( R(a*t) = \sum*{s'} B'(s') \cdot U(s', a*t) \cdot \sigma(2 \cdot (I*{\text{self}} - I\_{\text{other}})) \).

Intervene if \( R > 0.5 \). Implemented in NumPy for efficiency.

## Installation and Setup

1. **Clone Repo**: `git clone [repo-url]`.
2. **Docker Compose**: `docker-compose up` (builds OpenEnv server + World Monitor sidecar).
3. **Models**: Download via Hugging Face (quantized for efficiency); mount as volumes.
4. **Run Training**: Use `openenv.client` in a script: Loop reset/step with LLM calls.
5. **Dashboard**: Access at `http://localhost:8501` (Streamlit); monitor live sessions.
6. **Dependencies**: Python 3.12, OpenEnv, FastAPI, NumPy, Hugging Face Transformers.

## Contributing and License

Open-source under MIT. Contributions welcome for expanding agents or data sources. Built for the OpenEnv Hackathon (March 7-8, 2026)—feedback appreciated!

This README encapsulates the project's evolution from ideation to implementation, ensuring a comprehensive guide for building, training, and deploying.

Agents train in an RL loop using OpenEnv's interfaces, with each having independent policy updates (via TRL/Unsloth) while sharing the env. Live data injects during episodes, allowing decisions/actions in real-time simulations. Methods: PPO for single-agent baselines, GRPO for multi-agent cooperation.

**Rewards**: Sparse and delayed to encourage long-horizon planning. Formula per agent at timestep \( t \):

\[ r_t = 0.3 \cdot C_t + 0.4 \cdot E_t + 0.2 \cdot M_t + 0.1 \cdot B_t \]

- \( C_t \): Coalition Stability (\( \frac{\# \text{allied} - \# \text{betrayals}}{\# \text{agents}} \)).
- \( E_t \): Escalation Penalty (\( - \sigma(2 \cdot \Delta \text{tension}\_t) \)).
- \( M_t \): Market Gain (\( \frac{\Delta \text{oil} + \Delta \text{sanctions}}{2} \)).
- \( B*t \): Belief Alignment (\( 1 - |I*{\text{inferred}} - I\_{\text{true}}| \)).

Oversight scales rewards by 0.5 on high risk. Normalized to [-1, 1]; aggregated over 1000+ turn episodes.

## Oversight Analytical Method

Oversight uses belief propagation for risk:

1. Belief Update: \( B'(s') = \eta \sum_s P(o_t | a_t, s') T(s, a_t, s') B(s) \).

2. Risk Score: \( R(a*t) = \sum*{s'} B'(s') \cdot U(s', a*t) \cdot \sigma(2 \cdot (I*{\text{self}} - I\_{\text{other}})) \).

Intervene if \( R > 0.5 \). Implemented in NumPy for efficiency.

## Installation and Setup

1. **Clone Repo**: `git clone [repo-url]`.
2. **Docker Compose**: `docker-compose up` (builds OpenEnv server + World Monitor sidecar).
3. **Models**: Download via Hugging Face (quantized for efficiency); mount as volumes.
4. **Run Training**: Use `openenv.client` in a script: Loop reset/step with LLM calls.
5. **Dashboard**: Access at `http://localhost:8501` (Streamlit); monitor live sessions.
6. **Dependencies**: Python 3.12, OpenEnv, FastAPI, NumPy, Hugging Face Transformers.

## Contributing and License

Open-source under MIT. Contributions welcome for expanding agents or data sources. Built for the OpenEnv Hackathon (March 7-8, 2026)—feedback appreciated!

This README encapsulates the project's evolution from ideation to implementation, ensuring a comprehensive guide for building, training, and deploying.
