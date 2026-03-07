import { isAllowedUrl } from "./allowed-domains";
import { getAllSources } from "./registry";
import type { AgentId, SourceRegistryValidation } from "./types";

function getSourceKey(source: ReturnType<typeof getAllSources>[number]): string {
  switch (source.endpoint.kind) {
    case "url":
      return source.endpoint.url;
    case "telegram":
      return `telegram:${source.endpoint.handle}`;
    case "worldmonitor":
      return `worldmonitor:${source.endpoint.rpc}:${source.endpoint.selector ?? "*"}`;
    case "video":
      return `video:${source.endpoint.channel}`;
  }
}

export function validateSourceRegistry(): SourceRegistryValidation {
  const duplicateKeys = new Set<string>();
  const seenKeys = new Set<string>();
  const externalDomains = new Set<string>();
  const byAgent = {
    us: 0,
    israel: 0,
    iran: 0,
    hezbollah: 0,
    gulf: 0,
    oversight: 0,
  } satisfies Record<AgentId, number>;

  for (const source of getAllSources()) {
    byAgent[source.agentId] += 1;
    const key = getSourceKey(source);

    if (seenKeys.has(key)) {
      duplicateKeys.add(key);
    } else {
      seenKeys.add(key);
    }

    if (
      source.endpoint.kind === "url" &&
      source.allowlistStatus === "allowed" &&
      !isAllowedUrl(source.endpoint.url)
    ) {
      externalDomains.add(new URL(source.endpoint.url).hostname);
    }
  }

  return {
    duplicateKeys: [...duplicateKeys].sort(),
    externalDomains: [...externalDomains].sort(),
    byAgent,
  };
}
