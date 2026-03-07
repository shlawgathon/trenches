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
    apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000",
    vercelApiBase: process.env.NEXT_PUBLIC_VERCEL_API_BASE || "/api",
    enableSourceLogic: toBoolean(process.env.NEXT_PUBLIC_ENABLE_SOURCE_LOGIC, false),
  };
}
