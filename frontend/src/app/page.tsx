"use client";

import { useEffect, useState } from "react";
import { Brain, Database, GitMerge, Zap } from "lucide-react";
import UploadZone from "@/components/UploadZone";
import ChatInterface from "@/components/ChatInterface";
import { getHealth } from "@/lib/api";

interface Stats {
  total_chunks: number;
  status: string;
}

const PIPELINE_STEPS = [
  {
    step: "01",
    label: "Dense Vector Search",
    detail: "top-10 candidates",
    dotColor: "bg-blue-500",
    textColor: "text-blue-400",
    borderColor: "border-blue-900/40",
    bgColor: "bg-blue-950/20",
  },
  {
    step: "02",
    label: "Cross-Encoder Re-rank",
    detail: "top-3 refined chunks",
    dotColor: "bg-purple-500",
    textColor: "text-purple-400",
    borderColor: "border-purple-900/40",
    bgColor: "bg-purple-950/20",
  },
  {
    step: "03",
    label: "Grounded Generation",
    detail: "streamed via ",
    dotColor: "bg-emerald-500",
    textColor: "text-emerald-400",
    borderColor: "border-emerald-900/40",
    bgColor: "bg-emerald-950/20",
  },
] as const;

export default function Page() {
  const [stats, setStats] = useState<Stats | null>(null);

  const refreshStats = () =>
    getHealth()
      .then(setStats)
      .catch(() => { });

  useEffect(() => {
    refreshStats();
  }, []);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-gray-950">
      {/* ── Header ── */}
      <header className="flex items-center justify-between px-5 py-3.5 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <Brain className="w-4.5 h-4.5 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-tight">
              RAG System
            </h1>
            <p className="text-xs text-gray-500">
              bi-encoder · cross-encoder · Claude
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs">
          {stats && (
            <div className="flex items-center gap-1.5 text-gray-400">
              <Database className="w-3.5 h-3.5 text-blue-400" />
              <span>{stats.total_chunks.toLocaleString()} chunks indexed</span>
            </div>
          )}
          <div className="flex items-center gap-1.5 text-emerald-400">
            <Zap className="w-3 h-3" />
            <span>all-MiniLM-L6-v2 + ms-marco cross-encoder</span>
          </div>
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 min-h-0">
        {/* ── Sidebar ── */}
        <aside className="w-72 border-r border-gray-800 flex flex-col gap-6 p-4 overflow-y-auto shrink-0">
          {/* Pipeline diagram */}
          <section>
            <SectionTitle icon={<GitMerge className="w-3 h-3" />} label="Retrieval Pipeline" />
            <div className="space-y-1.5 mt-3">
              {PIPELINE_STEPS.map(
                ({ step, label, detail, dotColor, textColor, borderColor, bgColor }) => (
                  <div
                    key={step}
                    className={`flex items-center gap-3 p-2.5 rounded-lg border ${bgColor} ${borderColor}`}
                  >
                    <span
                      className={`text-xs font-bold font-mono w-6 text-center shrink-0 ${textColor}`}
                    >
                      {step}
                    </span>
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-gray-200 truncate">
                        {label}
                      </p>
                      <p className="text-xs text-gray-600">{detail}</p>
                    </div>
                    <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${dotColor}`} />
                  </div>
                )
              )}
            </div>
          </section>

          {/* Upload zone */}
          <section>
            <SectionTitle label="Document Ingestion" />
            <div className="mt-3">
              <UploadZone onIngested={refreshStats} />
            </div>
          </section>
        </aside>

        {/* ── Chat ── */}
        <main className="flex-1 min-w-0">
          <ChatInterface />
        </main>
      </div>
    </div>
  );
}

function SectionTitle({
  label,
  icon,
}: {
  label: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-1.5">
      {icon && <span className="text-gray-500">{icon}</span>}
      <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
        {label}
      </h2>
    </div>
  );
}
