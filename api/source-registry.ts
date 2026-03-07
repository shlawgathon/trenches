import { getAllSources, validateSourceRegistry } from "../src/lib/data-sources";

export const config = {
  runtime: "edge",
};

export default function handler(): Response {
  return Response.json({
    sources: getAllSources(),
    validation: validateSourceRegistry(),
  });
}
