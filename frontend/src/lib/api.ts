const BASE =
  typeof window !== "undefined"
    ? "/api"
    : (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000");

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

export interface DocumentDetail {
  name: string;
  chunks: number;
}

export interface DocumentsResult {
  documents: string[];
  document_details: DocumentDetail[];
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

export async function resetAll(): Promise<void> {
  const res = await fetch(`${BASE}/reset`, { method: "POST" });
  if (!res.ok) throw new Error("Reset failed");
}

export async function deleteDocument(filename: string): Promise<void> {
  const res = await fetch(
    `${BASE}/documents/${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Delete failed" }));
    throw new Error(err.detail ?? "Delete failed");
  }
}

/**
 * Stream a RAG query.
 * Yields answer tokens; calls `onSources` once with citation data.
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

export interface SummaryMeta {
  part: number;
  total_parts: number;
  total_chunks: number;
  sources: string[];
  source_label: string;
}

/**
 * Stream a plain-language summary for the given source(s).
 * Calls `onMeta` once with partition info, then yields tokens.
 */
export async function* streamSummarize(
  sources: string[],
  part: number,
  onMeta: (m: SummaryMeta) => void
): AsyncGenerator<string> {
  const res = await fetch(`${BASE}/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sources, part }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Summarize failed" }));
    throw new Error(err.detail ?? "Summarize failed");
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
        type: "meta" | "token" | "done" | "error";
        data?: unknown;
      };
      if (payload.type === "meta") onMeta(payload.data as SummaryMeta);
      else if (payload.type === "token") yield payload.data as string;
      else if (payload.type === "error") throw new Error((payload.data as string) ?? "Summary failed");
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
