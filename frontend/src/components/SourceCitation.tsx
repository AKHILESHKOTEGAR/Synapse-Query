import { BookOpen, Cpu, Hash, Layers, Zap } from "lucide-react";
import { Source } from "@/lib/api";

interface Props {
  sources: Source[];
}

function ScoreBar({
  value,
  color,
}: {
  value: number; // already normalised 0–1
  color: string;
}) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="flex-1 bg-gray-700/60 rounded-full h-1 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function RetrievalBadge({ rrf }: { rrf: number }) {
  // RRF max theoretical per list = 1/(60+1) ≈ 0.0164; two lists ≈ 0.033
  const pct = Math.min(rrf / 0.034, 1);
  if (pct > 0.7)
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-900/50 text-emerald-400 border border-emerald-800/60">
        <Zap className="w-2.5 h-2.5" />
        HYBRID MATCH
      </span>
    );
  if (pct > 0.35)
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-900/50 text-blue-400 border border-blue-800/60">
        <Layers className="w-2.5 h-2.5" />
        VECTOR MATCH
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-orange-900/50 text-orange-400 border border-orange-800/60">
      <Hash className="w-2.5 h-2.5" />
      KEYWORD MATCH
    </span>
  );
}

export default function SourceCitation({ sources }: Props) {
  if (!sources.length) return null;

  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

  return (
    <div className="space-y-2 mt-2">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wider">
        <BookOpen className="w-3 h-3" />
        {sources.length} chunk{sources.length > 1 ? "s" : ""} cited ·{" "}
        <span className="text-gray-600">k=25 → re-ranked → top-3</span>
      </div>

      {sources.map((src, i) => (
        <div
          key={i}
          className="bg-gray-800/70 border border-gray-700/40 rounded-xl p-3 space-y-2.5"
        >
          {/* ── Header ── */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-blue-600 text-white text-xs font-bold shrink-0">
              {i + 1}
            </span>
            <span className="text-xs font-medium text-gray-200 truncate flex-1 min-w-0">
              {src.source}
            </span>
            <span className="text-xs text-gray-500 shrink-0">p. {src.page}</span>
            <RetrievalBadge rrf={src.rrf_score} />
          </div>

          {/* ── Score panel ── */}
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs text-gray-500">
            {/* Re-rank (cross-encoder logit, sigmoid-normalised) */}
            <div className="flex items-center gap-1.5">
              <Cpu className="w-3 h-3 text-emerald-500 shrink-0" />
              <span className="w-14 shrink-0">Re-rank</span>
              <ScoreBar value={sigmoid(src.rerank_score)} color="bg-emerald-400" />
              <span className="w-11 text-right tabular-nums text-emerald-400">
                {src.rerank_score.toFixed(2)}
              </span>
            </div>

            {/* RRF fusion score */}
            <div className="flex items-center gap-1.5">
              <Layers className="w-3 h-3 text-violet-400 shrink-0" />
              <span className="w-14 shrink-0">RRF</span>
              <ScoreBar value={src.rrf_score / 0.034} color="bg-violet-400" />
              <span className="w-11 text-right tabular-nums text-violet-400">
                {src.rrf_score.toFixed(4)}
              </span>
            </div>

            {/* Vector similarity */}
            <div className="flex items-center gap-1.5">
              <Zap className="w-3 h-3 text-blue-400 shrink-0" />
              <span className="w-14 shrink-0">Vector</span>
              <ScoreBar value={src.similarity_score} color="bg-blue-400" />
              <span className="w-11 text-right tabular-nums text-blue-400">
                {src.similarity_score.toFixed(3)}
              </span>
            </div>

            {/* BM25 keyword score (normalised to ≤ 30 for bar) */}
            <div className="flex items-center gap-1.5">
              <Hash className="w-3 h-3 text-orange-400 shrink-0" />
              <span className="w-14 shrink-0">BM25</span>
              <ScoreBar value={Math.min(src.bm25_score / 20, 1)} color="bg-orange-400" />
              <span className="w-11 text-right tabular-nums text-orange-400">
                {src.bm25_score.toFixed(2)}
              </span>
            </div>
          </div>

          {/* ── Text excerpt ── */}
          <p className="text-xs text-gray-400 leading-relaxed border-t border-gray-700/50 pt-2 line-clamp-4 font-mono">
            {src.text}
          </p>
        </div>
      ))}
    </div>
  );
}
