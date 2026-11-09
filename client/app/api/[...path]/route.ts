// Same-origin gateway. Only the configured API can be contacted; request paths
// cannot select arbitrary hosts or filesystem locations.
import { env } from "cloudflare:workers";

async function forward(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const route = path.join("/");
  if (!/^(health|summary|contexts(?:\/[a-f0-9]{64}(?:\/(?:source|rating))?)?)$/.test(route))
    return Response.json({ detail: "Endpoint not found." }, { status: 404 });
  if (request.method === "POST" && !/^contexts\/[a-f0-9]{64}\/source$/.test(route))
    return Response.json({ detail: "Method not allowed." }, { status: 405 });
  if (request.method === "PUT" && !/^contexts\/[a-f0-9]{64}\/rating$/.test(route))
    return Response.json({ detail: "Method not allowed." }, { status: 405 });
  const origin = env.DGILIT_API_ORIGIN || process.env.DGILIT_API_ORIGIN || "http://127.0.0.1:8000";
  const upstream = new URL(`/api/${route}`, origin);
  upstream.search = new URL(request.url).search;
  try {
    const body = request.method === "PUT" ? await request.text() : undefined;
    if (body && body.length > 1024)
      return Response.json({ detail: "Rating request is too large." }, { status: 413 });
    const result = await fetch(upstream, { method: request.method, body,
      headers: { Accept: "application/json", ...(request.method === "PUT" ? { "Content-Type": "application/json" } : {}) },
      cache: "no-store", signal: AbortSignal.timeout(60000) });
    return new Response(await result.text(), { status: result.status, headers: {
      "Content-Type": "application/json", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" } });
  } catch {
    return Response.json({ detail: "The local data service is unavailable. Start both services with npm run dev and try again." }, { status: 503 });
  }
}
export const GET = forward;
export const POST = forward;
export const PUT = forward;
