"use client";

import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertCircle, Brain, ChevronRight, Loader2, Send } from "lucide-react";
import { Source, SummaryMeta, streamQuery, streamSummarize } from "@/lib/api";
import SourceCitation from "./SourceCitation";

export interface SummaryTrigger {
  sources: string[];
  label: string;
}

interface Props {
  pendingSummary?: SummaryTrigger | null;
  onSummaryConsumed?: () => void;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
  summaryMeta?: SummaryMeta;  // set on summary messages for continuation
}

const HINTS = [
  "What are the main contributions of this paper?",
  "Summarise the methodology in a table",
  "What results did the experiments produce?",
  "List all cited works and their relevance",
];

export default function ChatInterface({ pendingSummary, onSummaryConsumed }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const busyRef = useRef(false);

  useEffect(() => { busyRef.current = busy; }, [busy]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── standard RAG query ──────────────────────────────────────────────
  async function submit(e: FormEvent) {
    e.preventDefault();
    const query = input.trim();
    if (!query || busy) return;

    const userId = `u-${Date.now()}`;
    const asstId = `a-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: query },
      { id: asstId, role: "assistant", content: "", sources: [], streaming: true },
    ]);
    setInput("");
    setBusy(true);
    setError(null);

    try {
      const gen = streamQuery(query, (sources) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === asstId ? { ...m, sources } : m))
        )
      );
      for await (const token of gen) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstId ? { ...m, content: m.content + token } : m
          )
        );
      }
      setMessages((prev) =>
        prev.map((m) => (m.id === asstId ? { ...m, streaming: false } : m))
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Request failed";
      setError(msg);
      setMessages((prev) => prev.filter((m) => m.id !== asstId));
    } finally {
      setBusy(false);
    }
  }

  // ── summary streaming ────────────────────────────────────────────────
  const startSummary = useCallback(async (sources: string[], label: string, part: number) => {
    if (busyRef.current) return;

    const userLabel =
      part === 1
        ? `📄 Summarise: ${label}`
        : `📄 Continue summary — Part ${part}: ${label}`;

    const userId = `u-sum-${Date.now()}`;
    const asstId = `a-sum-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: userLabel },
      { id: asstId, role: "assistant", content: "", streaming: true },
    ]);
    setBusy(true);
    setError(null);

    try {
      let capturedMeta: SummaryMeta | null = null;

      const gen = streamSummarize(sources, part, (meta) => {
        capturedMeta = meta;
      });

      for await (const token of gen) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstId ? { ...m, content: m.content + token } : m
          )
        );
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === asstId
            ? { ...m, streaming: false, summaryMeta: capturedMeta ?? undefined }
            : m
        )
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Summary failed";
      setError(msg);
      setMessages((prev) => prev.filter((m) => m.id !== asstId));
    } finally {
      setBusy(false);
    }
  }, []);

  // ── watch for externally-triggered summary ───────────────────────────
  useEffect(() => {
    if (!pendingSummary) return;
    onSummaryConsumed?.();
    startSummary(pendingSummary.sources, pendingSummary.label, 1);
  }, [pendingSummary]); // eslint-disable-line react-hooks/exhaustive-deps

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(e as unknown as FormEvent);
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* ── message list ── */}
      <div className="flex-1 overflow-y-auto px-5 py-6 space-y-6">
        {messages.length === 0 ? (
          <EmptyState onHint={setInput} />
        ) : (
          messages.map((msg) => (
            <div key={msg.id}>
              {msg.role === "user" ? (
                <div className="flex justify-end">
                  <div className="max-w-[72%] bg-blue-600/90 text-white text-[13px] leading-relaxed rounded-2xl rounded-tr-sm px-4 py-3 shadow-lg shadow-blue-900/20">
                    {msg.content}
                  </div>
                </div>
              ) : (
                <div className="flex gap-3 items-start">
                  <div className="w-6 h-6 rounded-md bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shrink-0 mt-0.5 shadow-md shadow-blue-900/30">
                    <Brain className="w-3.5 h-3.5 text-white" />
                  </div>

                  <div className="flex-1 min-w-0">
                    {msg.content ? (
                      <div className="nexus-prose max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                        {msg.streaming && (
                          <span className="inline-block w-0.5 h-3.5 bg-blue-400/80 animate-pulse align-middle ml-0.5 rounded-full" />
                        )}
                      </div>
                    ) : msg.streaming ? (
                      <div className="flex items-center gap-2 py-1">
                        <div className="flex gap-1">
                          {[0, 1, 2].map((n) => (
                            <span
                              key={n}
                              className="w-1 h-1 rounded-full bg-blue-500/60 animate-bounce"
                              style={{ animationDelay: `${n * 120}ms` }}
                            />
                          ))}
                        </div>
                        <span className="text-[11px] text-gray-600">Thinking…</span>
                      </div>
                    ) : null}

                    {/* RAG citations */}
                    {!msg.streaming && msg.sources && msg.sources.length > 0 && (
                      <SourceCitation sources={msg.sources} />
                    )}

                    {/* Multi-part continuation */}
                    {!msg.streaming && msg.summaryMeta &&
                      msg.summaryMeta.part < msg.summaryMeta.total_parts && (
                      <button
                        onClick={() =>
                          startSummary(
                            msg.summaryMeta!.sources,
                            msg.summaryMeta!.source_label,
                            msg.summaryMeta!.part + 1
                          )
                        }
                        disabled={busy}
                        className="mt-3 flex items-center gap-2 px-3 py-2 rounded-lg border border-blue-500/20 bg-blue-500/5 text-blue-400 text-[12px] font-medium hover:bg-blue-500/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        <ChevronRight className="w-3.5 h-3.5" />
                        Continue — Part {msg.summaryMeta.part + 1} of {msg.summaryMeta.total_parts}
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))
        )}

        {error && (
          <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-red-500/5 border border-red-500/15 text-red-400 text-[12px]">
            <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── input bar ── */}
      <div className="px-5 pb-5 pt-3 border-t border-gray-800/60 bg-[#0a0a0f]">
        <form onSubmit={submit} className="flex gap-2.5 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask anything about your documents…"
            disabled={busy}
            rows={1}
            className={[
              "flex-1 bg-[#0d0d16] border border-white/[0.07] rounded-xl px-4 py-3",
              "text-[13px] text-gray-200 placeholder-gray-600",
              "resize-none overflow-auto focus:outline-none",
              "focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20",
              "transition-all duration-150 disabled:opacity-50",
            ].join(" ")}
            style={{ minHeight: "48px", maxHeight: "160px" }}
          />
          <button
            type="submit"
            disabled={!input.trim() || busy}
            className={[
              "w-11 h-11 rounded-xl flex items-center justify-center shrink-0",
              "bg-blue-600 hover:bg-blue-500 transition-colors",
              "disabled:opacity-30 disabled:cursor-not-allowed",
              "shadow-lg shadow-blue-900/30",
            ].join(" ")}
          >
            {busy
              ? <Loader2 className="w-4 h-4 text-white animate-spin" />
              : <Send className="w-4 h-4 text-white" />}
          </button>
        </form>
        <p className="text-[10px] text-gray-700 mt-2 text-center">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}

function EmptyState({ onHint }: { onHint: (v: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center px-6 gap-6">
      <div className="relative">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500/20 to-violet-600/20 border border-blue-500/15 flex items-center justify-center shadow-xl shadow-blue-900/20">
          <Brain className="w-7 h-7 text-blue-400" />
        </div>
        <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-emerald-500/90 border-2 border-[#0a0a0f] flex items-center justify-center">
          <span className="text-[7px] text-white font-bold">AI</span>
        </div>
      </div>

      <div className="space-y-1.5">
        <h2 className="text-[15px] font-semibold text-gray-200 tracking-tight">
          Ready to analyse
        </h2>
        <p className="text-[12px] text-gray-600 max-w-xs leading-relaxed">
          Upload PDFs on the left, then ask anything — or click Summarise on any document for a plain-language breakdown.
        </p>
      </div>

      <div className="w-full max-w-sm space-y-1.5">
        <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-wider mb-2">
          Try asking
        </p>
        {HINTS.map((hint) => (
          <button
            key={hint}
            onClick={() => onHint(hint)}
            className="w-full text-left px-3.5 py-2.5 rounded-xl border border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.04] hover:border-white/[0.09] text-[11px] text-gray-500 hover:text-gray-300 transition-all duration-150"
          >
            {hint}
          </button>
        ))}
      </div>
    </div>
  );
}
