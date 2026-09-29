const defaultEdgeUrl = "http://127.0.0.1:8000";

export function edgeHttpUrl() {
  return process.env.NEXT_PUBLIC_EDGE_URL ?? defaultEdgeUrl;
}

export function edgeWsUrl(path: string) {
  const base = edgeHttpUrl().replace(/^http/, "ws");
  return `${base}${path}`;
}

export async function postCommand(path: string) {
  const response = await fetch(`${edgeHttpUrl()}${path}`, { method: "POST" });
  return response.json();
}

