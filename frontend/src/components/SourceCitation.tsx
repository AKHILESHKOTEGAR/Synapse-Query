"use client";

import { useState } from "react";
import { BookOpen, ChevronDown, ChevronRight, Cpu, Hash, Layers, Zap } from "lucide-react";
import { Source } from "@/lib/api";

interface Props {
  sources: Source[];
}

const SCORE_DEFS = [
  { label: "Re-rank",  text: "text-emerald-400", bg: "bg-emerald-400", Icon: Cpu,    norm: (s: Source) => 1 / (1 + Math.exp(-s.rerank_score)),     val: (s: Source) => s.rerank_score.toFixed(2)  },
  { label: "RRF",      text: "text-violet-400",  bg: "bg-violet-400",  Icon: Layers, norm: (s: Source) => Math.min(s.rrf_score / 0.034, 1),          val: (s: Source) => s.rrf_score.toFixed(4)     },
  { label: "Vector",   text: "text-blue-400",    bg: "bg-blue-400",    Icon: Zap,    norm: (s: Source) => s.similarity_score,                        val: (s: Source) => s.similarity_score.toFixed(3) },
  { label: "BM25",     text: "text-orange-400",  bg: "bg-orange-400",  Icon: Hash,   norm: (s: Source) => Math.min(s.bm25_score / 20, 1),             val: (s: Source) => s.bm25_score.toFixed(2)    },
] as const;

function MatchBadge({ rrf, rerank }: { rrf: number; rerank: number }) {
  const pct = Math.min(rrf / 0.034, 1);
  if (pct > 0.7 || rerank > 2)
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 tracking-wide uppercase">
        <Zap className="w-2 h-2" />Hybrid
      </span>
    );
  if (pct > 0.35)
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold bg-blue-500/10 text-blue-400 border border-blue-500/20 tracking-wide uppercase">
        <Layers className="w-2 h-2" />Vector
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold bg-orange-500/10 text-orange-400 border border-orange-500/20 tracking-wide uppercase">
      <Hash className="w-2 h-2" />Keyword
    </span>
  );
}

export default function SourceCitation({ sources }: Props) {
  const [open, setOpen] = useState(true);
  const [expanded, setExpanded] = useState<Set<number>>(new Set([0]));

  if (!sources.length) return null;

  const toggle = (i: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });

  return (
    <div className="mt-3 rounded-xl border border-white/[0.06] bg-[#0b0b12] overflow-hidden">
      {/* ── section header ── */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-white/[0.02] transition-colors group"
      >
        <BookOpen className="w-3 h-3 text-gray-600 shrink-0" />
        <span className="text-[10px] font-semibold text-gray-600 uppercase tracking-wider flex-1">
          {sources.length} source{sources.length > 1 ? "s" : ""} retrieved
        </span>
        {open
          ? <ChevronDown className="w-3 h-3 text-gray-700 group-hover:text-gray-500 transition-colors" />
          : <ChevronRight className="w-3 h-3 text-gray-700 group-hover:text-gray-500 transition-colors" />}
      </button>

      {open && (
        <div className="border-t border-white/[0.04]">
          {sources.map((src, i) => (
            <div key={i} className="border-b border-white/[0.04] last:border-0">
              {/* ── source row ── */}
              <button
                onClick={() => toggle(i)}
                className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left hover:bg-white/[0.02] transition-colors group"
              >
                <span className="w-4 h-4 rounded-full bg-blue-500/15 text-blue-400 text-[9px] font-bold flex items-center justify-center shrink-0">
                  {i + 1}
                </span>

                <div className="flex-1 min-w-0">
                  <p className="text-[11px] font-medium text-gray-300 truncate">{src.source}</p>
                  <p className="text-[9px] text-gray-600 mt-0.5">Page {src.page}</p>
                </div>

                <MatchBadge rrf={src.rrf_score} rerank={src.rerank_score} />

                {expanded.has(i)
                  ? <ChevronDown className="w-3 h-3 text-gray-600 shrink-0 group-hover:text-gray-400 transition-colors" />
                  : <ChevronRight className="w-3 h-3 text-gray-600 shrink-0 group-hover:text-gray-400 transition-colors" />}
              </button>

              {/* ── expanded detail ── */}
              {expanded.has(i) && (
                <div className="px-3 pb-3 space-y-2.5 bg-black/10">
                  {/* text excerpt */}
                  <p className="text-[11px] text-gray-400 leading-relaxed bg-black/20 rounded-lg px-3 py-2.5 border border-white/[0.04] font-mono">
                    {src.text}
                  </p>

                  {/* score bars */}
                  <div className="grid grid-cols-2 gap-x-5 gap-y-1.5">
                    {SCORE_DEFS.map(({ label, text, bg, Icon, norm, val }) => {
                      const pct = Math.max(0, Math.min(100, norm(src) * 100));
                      return (
                        <div key={label} className="flex items-center gap-1.5">
                          <Icon className={`w-2.5 h-2.5 ${text} shrink-0`} />
                          <span className="text-[9px] text-gray-600 w-11 shrink-0">{label}</span>
                          <div className="flex-1 h-0.5 bg-white/[0.06] rounded-full overflow-hidden">
                            <div className={`h-full ${bg} rounded-full`} style={{ width: `${pct}%` }} />
                          </div>
                          <span className={`text-[9px] ${text} tabular-nums w-9 text-right`}>{val(src)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
