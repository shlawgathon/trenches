type RuntimeEnv = {
  apiBaseUrl: string;
  vercelApiBase: string;
  enableSourceLogic: boolean;
};

function toBoolean(value: string | boolean | undefined, fallback: boolean): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value !== "string") {
    return fallback;
  }
  return value.toLowerCase() === "true";
}

export function getRuntimeEnv(): RuntimeEnv {
  return {
    apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
    vercelApiBase: import.meta.env.VITE_VERCEL_API_BASE || "/api",
    enableSourceLogic: toBoolean(import.meta.env.VITE_ENABLE_SOURCE_LOGIC, false),
  };
}
