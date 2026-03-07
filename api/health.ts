export const config = {
  runtime: "edge",
};

export default function handler(): Response {
  return Response.json({
    status: "ok",
    service: "trenches-vercel-api",
  });
}
