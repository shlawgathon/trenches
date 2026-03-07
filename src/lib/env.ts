type RuntimeEnv = {
  apiBaseUrl: string;
  vercelApiBase: string;
  enableSourceLogic: boolean;
};

declare global {
  interface Window {
    __trenchesEnv?: Partial<Record<keyof RuntimeEnv | "mapboxToken", string | boolean>>;
  }
}

function toBoolean(value: string | boolean | undefined, fallback: boolean): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value !== "string") {
    return fallback;
  }
  return value.toLowerCase() === "true";
}

function readClientEnv(key: keyof RuntimeEnv | "mapboxToken"): string | boolean | undefined {
  if (typeof window !== "undefined" && window.__trenchesEnv && key in window.__trenchesEnv) {
    return window.__trenchesEnv[key];
  }
  return undefined;
}

export function getRuntimeEnv(): RuntimeEnv {
  const clientApiBase = readClientEnv("apiBaseUrl");
  const clientVercelBase = readClientEnv("vercelApiBase");
  const clientSourceLogic = readClientEnv("enableSourceLogic");
  return {
    apiBaseUrl: (typeof clientApiBase === "string" ? clientApiBase : process.env.NEXT_PUBLIC_API_BASE_URL) || "http://localhost:8000",
    vercelApiBase: (typeof clientVercelBase === "string" ? clientVercelBase : process.env.NEXT_PUBLIC_VERCEL_API_BASE) || "/api",
    enableSourceLogic: toBoolean(
      typeof clientSourceLogic === "string" || typeof clientSourceLogic === "boolean"
        ? clientSourceLogic
        : process.env.NEXT_PUBLIC_ENABLE_SOURCE_LOGIC,
      false,
    ),
  };
}

export function getMapboxToken(): string {
  const clientToken = readClientEnv("mapboxToken");
  if (typeof clientToken === "string") {
    return clientToken;
  }
  return process.env.NEXT_PUBLIC_MAPBOX_TOKEN || "";
}
