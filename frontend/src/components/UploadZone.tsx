"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, CheckCircle2, XCircle, Loader2, X } from "lucide-react";
import { deleteDocument, uploadPDF, UploadResult } from "@/lib/api";

type FileStatus = "uploading" | "done" | "error";

interface FileState {
  id: string;
  name: string;
  status: FileStatus;
  result?: UploadResult;
  error?: string;
}

interface Props {
  onIngested?: () => void;
}

export default function UploadZone({ onIngested }: Props) {
  const [files, setFiles] = useState<FileState[]>([]);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      if (!accepted.length) return;

      const entries: FileState[] = accepted.map((f) => ({
        id: `${f.name}-${f.size}-${Date.now()}`,
        name: f.name,
        status: "uploading",
      }));
      setFiles((prev) => [...entries, ...prev]);

      await Promise.allSettled(
        accepted.map(async (f, i) => {
          const id = entries[i].id;
          try {
            const result = await uploadPDF(f);
            setFiles((prev) =>
              prev.map((s) => (s.id === id ? { ...s, status: "done", result } : s))
            );
            onIngested?.();
          } catch (err) {
            setFiles((prev) =>
              prev.map((s) =>
                s.id === id
                  ? { ...s, status: "error", error: err instanceof Error ? err.message : "Upload failed" }
                  : s
              )
            );
          }
        })
      );
    },
    [onIngested]
  );

  const handleRemove = async (id: string, name: string, isDone: boolean) => {
    if (isDone) {
      try {
        await deleteDocument(name);
        onIngested?.();
      } catch {
        // may already be gone — still remove from list
      }
    }
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const busy = files.some((f) => f.status === "uploading");

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: true,
    disabled: busy,
  });

  return (
    <div className="space-y-2">
      <div
        {...getRootProps()}
        className={[
          "relative border-2 border-dashed rounded-xl p-5 text-center cursor-pointer select-none",
          "transition-all duration-200",
          isDragActive
            ? "border-blue-500/70 bg-blue-500/5 scale-[1.01]"
            : "border-white/[0.07] hover:border-white/[0.14] hover:bg-white/[0.02]",
          busy ? "pointer-events-none opacity-50" : "",
        ].join(" ")}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-2">
          {busy ? (
            <div className="w-8 h-8 rounded-full bg-blue-500/10 flex items-center justify-center">
              <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
            </div>
          ) : (
            <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${
              isDragActive ? "bg-blue-500/20" : "bg-white/[0.04]"
            }`}>
              <Upload className={`w-4 h-4 ${isDragActive ? "text-blue-400" : "text-gray-500"}`} />
            </div>
          )}
          <div>
            <p className="text-xs font-medium text-gray-300">
              {isDragActive ? "Release to upload" : "Drop PDFs here"}
            </p>
            <p className="text-[10px] text-gray-600 mt-0.5">
              Multiple files · up to 50 MB each
            </p>
          </div>
        </div>
      </div>

      {/* Per-file rows */}
      {files.length > 0 && (
        <div className="space-y-1 max-h-52 overflow-y-auto">
          {files.map((f) => (
            <div
              key={f.id}
              className={[
                "group flex items-center gap-2.5 px-3 py-2 rounded-lg border text-[11px] transition-all",
                f.status === "uploading" && "bg-blue-500/5 border-blue-500/15 text-blue-300",
                f.status === "done"    && "bg-emerald-500/5 border-emerald-500/15 text-emerald-300",
                f.status === "error"   && "bg-red-500/5 border-red-500/15 text-red-300",
              ].filter(Boolean).join(" ")}
            >
              <span className="shrink-0">
                {f.status === "uploading" && <Loader2 className="w-3 h-3 animate-spin" />}
                {f.status === "done"    && <CheckCircle2 className="w-3 h-3" />}
                {f.status === "error"   && <XCircle className="w-3 h-3" />}
              </span>

              <div className="min-w-0 flex-1">
                <p className="font-medium truncate">{f.name}</p>
                <p className="opacity-60 mt-0.5">
                  {f.status === "uploading" && "Processing…"}
                  {f.status === "done" && f.result && `${f.result.pages_processed} pages · ${f.result.chunks_stored} chunks indexed`}
                  {f.status === "error" && f.error}
                </p>
              </div>

              {/* Remove / delete button */}
              {f.status !== "uploading" && (
                <button
                  onClick={() => handleRemove(f.id, f.name, f.status === "done")}
                  className={[
                    "shrink-0 p-1 rounded opacity-0 group-hover:opacity-100 transition-all",
                    f.status === "done"
                      ? "text-gray-500 hover:text-red-400 hover:bg-red-500/10"
                      : "text-gray-500 hover:text-gray-300 hover:bg-white/5",
                  ].join(" ")}
                  title={f.status === "done" ? "Remove from library" : "Dismiss"}
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
