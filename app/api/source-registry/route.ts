import { NextResponse } from "next/server";

import { getAllSources, validateSourceRegistry } from "../../../src/lib/data-sources";

export const runtime = "edge";

export function GET() {
  return NextResponse.json({
    sources: getAllSources(),
    validation: validateSourceRegistry(),
  });
}
