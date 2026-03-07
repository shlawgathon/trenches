# TOOLS.md: Agent Tools and Function-Calling Interface in Fog of War Diplomacy Simulator

This document details the **tools** available to the 6 agents in the Fog of War Diplomacy Simulator. These tools enable agents to interact with the environment, query personalized intelligence feeds (from World Monitor integration), perform actions that affect the world state, and gather information consistent with their partial-observability constraints.

Tools are implemented as **text-based function-calling** within each agent's LLM inference loop (using Hugging Face Transformers or similar). When an agent needs information or wants to act, it outputs a structured function call in its response (e.g., JSON or XML-like format). The OpenEnv environment parses these calls, executes them, and injects results back into the next observation.

This design:
- Reinforces **theory-of-mind** (agents must infer when others might use tools)
- Supports **tool-use fine-tuning** during RL post-training
- Maintains **partial observability** (tools return only agent-specific data)
- Aligns with **entity identity** (some tools are role-exclusive)

## General Tool Usage Rules

- **Invocation Format**: Agents output function calls in a parseable format, e.g.:
  ```json
  {
    "tool": "query_intel",
    "parameters": {
      "keywords": "US_polls Iran_strike",
      "source": "polymarket"
    }
  }
  ```
  or XML-style if preferred by the prompt.

- **Execution**: In `step(actions)`, the env:
    1. Parses tool calls from agent text output
    2. Validates agent permissions (e.g., Iran cannot use "impose_sanctions")
    3. Executes (queries World Monitor API, simulates action outcome)
    4. Returns result in next obs dict (e.g., `obs["US"]["tool_result"]`)

- **Cost/Cooldown**: Most tools have simulated "cost" (e.g., -0.1 reward for heavy queries) or cooldown (e.g., query every 3 turns) to prevent spam.

- **Consistency**: Tools pull from World Monitor APIs (deployed on Vercel/Railway) → filtered JSON snippets → injected into prompt history for persistent context.

## Common Tools (Available to All Agents)

1. **query_intel**
    - Description: Request filtered intelligence from World Monitor feeds.
    - Parameters:
        - `keywords`: string (space-separated search terms, e.g., "oil Hormuz strike")
        - `source`: optional string (e.g., "polymarket", "acled", "telegram_osint", "commodity_dashboard")
        - `time_range`: optional string ("last_hour", "last_day")
    - Returns: Dict of snippets/headlines (e.g., {"headline": "...", "sentiment": 0.6, "source": "..."})
    - Usage: Core tool for maintaining situational awareness; agents decide what to query based on current tension.

2. **analyze_belief**
    - Description: Infer hidden incentives/beliefs of another agent (theory-of-mind).
    - Parameters:
        - `target_agent`: string (e.g., "Iran")
        - `evidence`: string (short summary of observed actions)
    - Returns: Dict {"inferred_incentive": "...", "confidence": 0.72}
    - Usage: Used to improve \( B_t \) reward component.

3. **propose_negotiation**
    - Description: Send a diplomatic proposal to one or more agents.
    - Parameters:
        - `recipients`: list[string] (e.g., ["US", "Gulf Coalition"])
        - `proposal_text`: string (e.g., "Ceasefire in exchange for sanctions relief")
    - Returns: Dict {"sent": true, "acknowledged_by": [...], "immediate_response": "..."}
    - Usage: Forms coalitions; can be deceptive.

## Agent-Specific Tools

### 1. US (Trump Admin / CENTCOM)
- **impose_sanctions**
    - Parameters: `target`: string, `severity`: float (0-1)
    - Effect: Increases tension for target, boosts US \( M_t \), risks backlash if overused.
- **deploy_assets**
    - Parameters: `location`: string (e.g., "Gulf"), `type`: string ("carrier", "cyber")
    - Effect: Deters escalation, visible to allies.
- **query_polls**
    - Shortcut to `query_intel(keywords="US approval rating Polymarket")`

### 2. Israel (Netanyahu / IDF)
- **launch_precise_strike**
    - Parameters: `target`: string (e.g., "IRGC facility"), `risk_level`: float
    - Effect: High escalation potential, strong \( E_t \) penalty if civilian risk high.
- **activate_iron_dome**
    - Parameters: `region`: string
    - Effect: Reduces damage from incoming rockets (reduces \( E_t \) penalty).
- **query_border_alerts**
    - Shortcut: `query_intel(keywords="OREF Lebanon rocket")`

### 3. Iran (IRGC / Interim Leadership)
- **activate_proxy**
    - Parameters: `proxy`: string (e.g., "Hezbollah"), `action_type`: string ("drone", "cyber")
    - Effect: Asymmetric retaliation, lower escalation visibility.
- **threaten_hormuz**
    - Parameters: `severity`: float
    - Effect: Spikes global oil price, strong \( M_t \) impact for Gulf.
- **query_proxy_status**
    - Shortcut: `query_intel(keywords="Hezbollah militia Telegram")`

### 4. Hezbollah (Proxy Swarm Leader)
- **launch_drone_swarm**
    - Parameters: `target`: string, `scale`: int (1-10)
    - Effect: High asymmetric damage, visible to Israel.
- **evade_detection**
    - Parameters: None
    - Effect: Reduces probability of being traced back to Iran.
- **query_border_streams**
    - Shortcut: `query_intel(keywords="MTV Lebanon webcam Hezbollah")`

### 5. Gulf Coalition (Saudi/UAE/Qatar)
- **adjust_oil_output**
    - Parameters: `delta`: float (-1 to +1)
    - Effect: Stabilizes or shocks markets, directly affects \( M_t \).
- **host_base_access**
    - Parameters: `ally`: string (e.g., "US")
    - Effect: Strengthens coalition, visible to Iran.
- **query_market_impact**
    - Shortcut: `query_intel(keywords="oil price Hormuz Bloomberg")`

### 6. Oversight Agent (Fleet AI Meta-Layer)
- **generate_explanation**
    - Parameters: `target_action`: string, `agent`: string
    - Returns: Natural language explanation + risk score.
- **calculate_risk**
    - Parameters: `action`: dict
    - Returns: Float risk score (0-1) using belief propagation formula.
- **intervene**
    - Parameters: `target_agent`: string, `action`: string ("force_rethink", "audit_beliefs")
    - Effect: Scales reward by 0.5 or forces re-action.
- **query_global_synthesis**
    - Shortcut: `query_intel(scope="global", keywords="hotspot escalation")`

## Implementation Notes

- **Tool Parsing**: Use structured output prompting (e.g., "Always respond with JSON tool call if using a tool") + regex/JSON parser in env.
- **Tool Result Injection**: Results appended to prompt history (e.g., "Tool result: [JSON]") for context window management.
- **RL Fine-Tuning**: Reward tool usage that leads to high entity-aligned outcomes (e.g., US sanctions → coalition strength).
- **Debugging**: Log all tool calls/results in dashboard for judging.
- **Security**: Validate parameters server-side to prevent invalid actions.

This toolset empowers agents to act intelligently within their roles while maintaining the simulator's core challenge: operating under incomplete, live-fed information in a high-stakes multi-agent crisis.