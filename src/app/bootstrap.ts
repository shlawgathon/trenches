import { createPlatformRuntime } from "../lib/platform";

declare global {
  interface Window {
    __trenches?: Awaited<ReturnType<typeof createPlatformRuntime>>;
  }
}

export async function bootstrapPlatform() {
  const runtime = await createPlatformRuntime();
  window.__trenches = runtime;
  return runtime;
}
