export type AgentId = "us" | "israel" | "iran" | "hezbollah" | "gulf" | "oversight";

export type SourceKind = "rss" | "api" | "scrape" | "telegram" | "structured" | "video";

export type AuthMode = "none" | "apiKey" | "relay" | "session" | "manual";

export type AllowlistStatus = "allowed" | "external" | "not_applicable";

export type SourceDelivery = "training_core" | "live_demo";

export type SourceEndpoint =
  | {
      kind: "url";
      url: string;
      method?: "GET" | "POST";
    }
  | {
      kind: "worldmonitor";
      rpc: string;
      selector?: string;
    }
  | {
      kind: "telegram";
      handle: string;
    }
  | {
      kind: "video";
      channel: string;
    };

export type DataSourceSpec = {
  id: string;
  agentId: AgentId;
  delivery: SourceDelivery;
  name: string;
  kind: SourceKind;
  endpoint: SourceEndpoint;
  auth: AuthMode;
  allowlistStatus: AllowlistStatus;
  tags: string[];
  rationale: string;
  notes?: string;
};

export type SourceFetchResult =
  | {
      status: "ok";
      sourceId: string;
      payload: string | unknown;
      fetchedAt: string;
    }
  | {
      status: "placeholder" | "blocked";
      sourceId: string;
      reason: string;
      fetchedAt: string;
    };

export type SourceRegistryValidation = {
  duplicateKeys: string[];
  externalDomains: string[];
  byAgent: Record<AgentId, number>;
};

export type LiveSourcePlanItem = {
  sourceId: string;
  pollIntervalMs: number;
  warmStart: boolean;
  maxItemsPerPull: number;
};
