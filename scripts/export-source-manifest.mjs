import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { getAllSources } from "../src/lib/data-sources/registry.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const outputPath = resolve(__dirname, "../backend/src/trenches_env/source_manifest.json");

const payload = {
  generatedAt: new Date().toISOString(),
  sourceCount: getAllSources().length,
  sources: getAllSources(),
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");

process.stdout.write(`wrote ${payload.sourceCount} sources to ${outputPath}\n`);
