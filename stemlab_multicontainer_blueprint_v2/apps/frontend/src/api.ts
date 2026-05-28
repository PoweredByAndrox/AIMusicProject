const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

function toAbsoluteUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }
  return new URL(pathOrUrl, `${API_BASE}/`).toString();
}

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(toAbsoluteUrl(path));
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(toAbsoluteUrl(path), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function postForm<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(toAbsoluteUrl(path), {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function mediaUrl(pathOrUrl?: string | null): string | null {
  if (!pathOrUrl) return null;
  return toAbsoluteUrl(pathOrUrl);
}
