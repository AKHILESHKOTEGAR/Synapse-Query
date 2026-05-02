const BASE =
  typeof window !== "undefined"
    ? "/api" // client-side: go through Next.js rewrite proxy
    : (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"); // SSR

export interface Source {
  text: string;
  source: string;
  page: number;
  similarity_score: number;
  bm25_score: number;
  rrf_score: number;
  rerank_score: number;
}

export interface UploadResult {
  message: string;
  source: string;
  chunks_stored: number;
  pages_processed: number;
  collection_total: number;
  status: string;
}

export interface HealthResult {
  status: string;
  total_chunks: number;
  collection_name: string;
}

export interface DocumentsResult {
  documents: string[];
  count: number;
  total_chunks: number;
}

// ---------------------------------------------------------------------------

export async function uploadPDF(file: File): Promise<UploadResult> {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

/**
 * Stream a RAG query.
 *
 * Yields answer tokens; calls `onSources` once with citation data as soon
 * as the backend emits the sources SSE event (before the first token).
 */
export async function* streamQuery(
  query: string,
  onSources: (s: Source[]) => void
): AsyncGenerator<string> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Query failed" }));
    throw new Error(err.detail ?? "Query failed");
  }

  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = JSON.parse(line.slice(6)) as {
        type: "sources" | "token" | "done";
        data?: unknown;
      };

      if (payload.type === "sources") {
        onSources(payload.data as Source[]);
      } else if (payload.type === "token") {
        yield payload.data as string;
      }
    }
  }
}

export async function getHealth(): Promise<HealthResult> {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

export async function getDocuments(): Promise<DocumentsResult> {
  const res = await fetch(`${BASE}/documents`);
  return res.json();
}
