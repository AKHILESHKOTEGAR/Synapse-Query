"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Brain,
  ChevronDown,
  ChevronRight,
  Database,
  FileText,
  GitMerge,
  Layers,
  RotateCcw,
  Search,
  Sparkles,
  Trash2,
  Zap,
} from "lucide-react";
import UploadZone from "@/components/UploadZone";
import ChatInterface from "@/components/ChatInterface";
import { deleteDocument, DocumentDetail, getDocuments, getHealth, resetAll } from "@/lib/api";

interface Stats {
  total_chunks: number;
  status: string;
}

const PIPELINE_STEPS = [
  {
    icon: <Search className="w-3.5 h-3.5" />,
    label: "Hybrid Search",
    detail: "BM25 + Vector · k=40",
    color: "text-blue-400",
    bg: "bg-blue-950/30",
    border: "border-blue-900/30",
  },
  {
    icon: <Layers className="w-3.5 h-3.5" />,
    label: "Cross-Encoder Re-rank",
    detail: "top-5 refined chunks",
    color: "text-violet-400",
    bg: "bg-violet-950/30",
    border: "border-violet-900/30",
  },
  {
    icon: <Sparkles className="w-3.5 h-3.5" />,
    label: "Grounded Generation",
    detail: "Nemotron · streamed",
    color: "text-emerald-400",
    bg: "bg-emerald-950/30",
    border: "border-emerald-900/30",
  },
] as const;

export default function Page() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [docs, setDocs] = useState<DocumentDetail[]>([]);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);
  const [pipelineOpen, setPipelineOpen] = useState(true);

  const refreshAll = useCallback(() => {
    getHealth()
      .then(setStats)
      .catch(() => {});
    getDocuments()
      .then((d) => setDocs(d.document_details ?? []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const handleReset = async () => {
    if (!confirm("Clear all documents and start fresh?")) return;
    setResetting(true);
    try {
      await resetAll();
      setDocs([]);
      setStats(null);
      refreshAll();
    } finally {
      setResetting(false);
    }
  };

  const handleDelete = async (name: string) => {
    setDeletingDoc(name);
    try {
      await deleteDocument(name);
      refreshAll();
    } catch {
      // silent — doc may not exist on disk
      refreshAll();
    } finally {
      setDeletingDoc(null);
    }
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#0a0a0f]">
      {/* ── Header ── */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-gray-800/80 shrink-0 bg-[#0d0d14]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-lg shadow-blue-900/30">
            <Brain className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-tight leading-none">
              Nexus
            </h1>
            <p className="text-[10px] text-gray-500 mt-0.5">
              Technical Document Intelligence
            </p>
          </div>
        </div>

        <div className="flex items-center gap-5 text-[11px]">
          {stats && (
            <>
              <div className="flex items-center gap-1.5 text-gray-400">
                <Database className="w-3 h-3 text-blue-400" />
                <span>{stats.total_chunks.toLocaleString()} chunks</span>
              </div>
              <div className="flex items-center gap-1.5 text-gray-400">
                <FileText className="w-3 h-3 text-violet-400" />
                <span>{docs.length} document{docs.length !== 1 ? "s" : ""}</span>
              </div>
            </>
          )}
          <div className="flex items-center gap-1.5 text-emerald-500/80">
            <Zap className="w-3 h-3" />
            <span className="text-gray-500">all-MiniLM · ms-marco · Nemotron</span>
          </div>
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 min-h-0">
        {/* ── Sidebar ── */}
        <aside className="w-72 border-r border-gray-800/80 flex flex-col bg-[#0d0d14] shrink-0 overflow-hidden">
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">

            {/* Pipeline */}
            <section>
              <button
                onClick={() => setPipelineOpen((v) => !v)}
                className="w-full flex items-center justify-between group"
              >
                <div className="flex items-center gap-1.5">
                  <GitMerge className="w-3 h-3 text-gray-500" />
                  <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">
                    Retrieval Pipeline
                  </span>
                </div>
                {pipelineOpen ? (
                  <ChevronDown className="w-3 h-3 text-gray-600" />
                ) : (
                  <ChevronRight className="w-3 h-3 text-gray-600" />
                )}
              </button>

              {pipelineOpen && (
                <div className="mt-2.5 space-y-1.5">
                  {PIPELINE_STEPS.map(({ icon, label, detail, color, bg, border }, i) => (
                    <div
                      key={i}
                      className={`flex items-center gap-2.5 px-3 py-2 rounded-lg border ${bg} ${border}`}
                    >
                      <span className={`shrink-0 ${color}`}>{icon}</span>
                      <div className="min-w-0">
                        <p className="text-[11px] font-medium text-gray-200">{label}</p>
                        <p className="text-[10px] text-gray-600">{detail}</p>
                      </div>
                      <span
                        className={`ml-auto text-[10px] font-bold font-mono ${color} opacity-60`}
                      >
                        {String(i + 1).padStart(2, "0")}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <Divider />

            {/* Upload */}
            <section>
              <SectionLabel>Upload Documents</SectionLabel>
              <div className="mt-2.5">
                <UploadZone onIngested={refreshAll} />
              </div>
            </section>

            {/* Document Library */}
            {docs.length > 0 && (
              <>
                <Divider />
                <section>
                  <SectionLabel>
                    Library
                    <span className="ml-1.5 px-1.5 py-0.5 rounded-full bg-gray-800 text-gray-400 text-[9px] font-semibold">
                      {docs.length}
                    </span>
                  </SectionLabel>
                  <div className="mt-2.5 space-y-1">
                    {docs.map((doc) => (
                      <div
                        key={doc.name}
                        className="group flex items-center gap-2 px-2.5 py-2 rounded-lg bg-gray-800/40 border border-gray-700/30 hover:border-gray-600/50 transition-colors"
                      >
                        <FileText className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                        <div className="min-w-0 flex-1">
                          <p className="text-[11px] text-gray-200 font-medium truncate">
                            {doc.name}
                          </p>
                          <p className="text-[10px] text-gray-600">
                            {doc.chunks} chunks
                          </p>
                        </div>
                        <button
                          onClick={() => handleDelete(doc.name)}
                          disabled={deletingDoc === doc.name}
                          className="shrink-0 p-1 rounded opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 hover:bg-red-950/40 transition-all disabled:opacity-40"
                          title={`Remove ${doc.name}`}
                        >
                          {deletingDoc === doc.name ? (
                            <span className="w-3 h-3 border border-gray-500 border-t-transparent rounded-full animate-spin block" />
                          ) : (
                            <Trash2 className="w-3 h-3" />
                          )}
                        </button>
                      </div>
                    ))}
                  </div>
                </section>
              </>
            )}
          </div>

          {/* Sidebar footer */}
          <div className="px-4 py-3 border-t border-gray-800/60 shrink-0 space-y-2.5">
            <button
              onClick={handleReset}
              disabled={resetting}
              className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-[11px] text-gray-500 hover:text-red-400 hover:bg-red-950/30 border border-gray-800 hover:border-red-900/50 transition-all disabled:opacity-40"
            >
              <RotateCcw className={`w-3 h-3 ${resetting ? "animate-spin" : ""}`} />
              {resetting ? "Clearing…" : "Start Fresh"}
            </button>
            <p className="text-[10px] text-gray-700 text-center">
              Adaptive Knowledge Store · Self-improving retrieval
            </p>
          </div>
        </aside>

        {/* ── Chat ── */}
        <main className="flex-1 min-w-0 bg-[#0a0a0f]">
          <ChatInterface />
        </main>
      </div>
    </div>
  );
}

function Divider() {
  return <div className="border-t border-gray-800/60" />;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">
      {children}
    </div>
  );
}
