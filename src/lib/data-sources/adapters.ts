import type { DataSourceSpec, SourceFetchResult } from "./types";

export type DataSourceAdapter = {
  canFetch: boolean;
  fetch: () => Promise<SourceFetchResult>;
};

function ok(sourceId: string, payload: string | unknown): SourceFetchResult {
  return {
    status: "ok",
    sourceId,
    payload,
    fetchedAt: new Date().toISOString(),
  };
}

function placeholder(sourceId: string, reason: string): SourceFetchResult {
  return {
    status: "placeholder",
    sourceId,
    reason,
    fetchedAt: new Date().toISOString(),
  };
}

export function createDataSourceAdapter(source: DataSourceSpec): DataSourceAdapter {
  switch (source.endpoint.kind) {
    case "url": {
      const endpoint = source.endpoint;
      if (source.kind === "rss" || source.kind === "scrape") {
        return {
          canFetch: true,
          fetch: async () => {
            const response = await fetch(endpoint.url, { method: endpoint.method ?? "GET" });
            const payload = await response.text();
            return ok(source.id, payload);
          },
        };
      }

      return {
        canFetch: true,
        fetch: async () => {
          const response = await fetch(endpoint.url, { method: endpoint.method ?? "GET" });
          const payload = await response.json();
          return ok(source.id, payload);
        },
      };
    }

    case "worldmonitor": {
      const endpoint = source.endpoint;
      return {
        canFetch: false,
        fetch: async () =>
          placeholder(
            source.id,
            `World Monitor RPC adapter not wired yet: ${endpoint.rpc}. Route this through the Vercel proxy or Python sidecar.`,
          ),
      };
    }

    case "telegram": {
      const endpoint = source.endpoint;
      return {
        canFetch: false,
        fetch: async () =>
          placeholder(
            source.id,
            `Telegram relay not implemented yet for @${endpoint.handle}. Keep this behind a session/MTProto adapter.`,
          ),
      };
    }

    case "video": {
      const endpoint = source.endpoint;
      return {
        canFetch: false,
        fetch: async () =>
          placeholder(source.id, `Video channel ${endpoint.channel} is a monitoring source, not a fetchable feed.`),
      };
    }
  }
}
