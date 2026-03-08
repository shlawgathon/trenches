import { getRuntimeEnv } from "./env";
import { HttpClient } from "./http";
import { SessionClient } from "./session-client";
import type { AgentId } from "./data-sources";
import { AGENT_SOURCE_REGISTRY, buildLiveSourcePlan, validateSourceRegistry } from "./data-sources";
import type { CapabilitiesResponse } from "./types";

export type PlatformRuntime = {
  bootedAt: string;
  env: ReturnType<typeof getRuntimeEnv>;
  backendStatus: "unknown" | "healthy" | "unreachable";
  capabilities: CapabilitiesResponse | null;
  sessionClient: SessionClient;
  sourceRegistry: typeof AGENT_SOURCE_REGISTRY;
  sourceValidation: ReturnType<typeof validateSourceRegistry>;
  liveSourcePlans: Record<AgentId, ReturnType<typeof buildLiveSourcePlan>>;
};

export async function createPlatformRuntime(): Promise<PlatformRuntime> {
  const env = getRuntimeEnv();
  const sessionClient = new SessionClient(new HttpClient(env.apiBaseUrl));
  let backendStatus: PlatformRuntime["backendStatus"] = "unknown";
  let capabilities: CapabilitiesResponse | null = null;

  try {
    capabilities = await sessionClient.capabilities();
    backendStatus = "healthy";
  } catch {
    try {
      await sessionClient.health();
      backendStatus = "healthy";
    } catch {
      backendStatus = "unreachable";
    }
  }

  return {
    liveSourcePlans: Object.fromEntries(
      Object.keys(AGENT_SOURCE_REGISTRY).map((agentId) => [agentId, buildLiveSourcePlan(agentId as AgentId)]),
    ) as Record<AgentId, ReturnType<typeof buildLiveSourcePlan>>,
    bootedAt: new Date().toISOString(),
    env,
    backendStatus,
    capabilities,
    sessionClient,
    sourceRegistry: AGENT_SOURCE_REGISTRY,
    sourceValidation: validateSourceRegistry(),
  };
}
