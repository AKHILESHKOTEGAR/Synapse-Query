"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  FileText,
  CheckCircle2,
  XCircle,
  Loader2,
} from "lucide-react";
import { uploadPDF, UploadResult } from "@/lib/api";

type Status = "idle" | "uploading" | "success" | "error";

interface UploadState {
  status: Status;
  message?: string;
  result?: UploadResult;
}

interface Props {
  onIngested?: () => void;
}

export default function UploadZone({ onIngested }: Props) {
  const [state, setState] = useState<UploadState>({ status: "idle" });
  const [history, setHistory] = useState<string[]>([]);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      const file = accepted[0];
      if (!file) return;

      setState({ status: "uploading", message: `Ingesting ${file.name}…` });

      try {
        const result = await uploadPDF(file);
        setState({ status: "success", result, message: result.message });
        setHistory((h) => [file.name, ...h.filter((n) => n !== file.name)]);
        onIngested?.();
      } catch (err) {
        setState({
          status: "error",
          message: err instanceof Error ? err.message : "Upload failed",
        });
      }
    },
    [onIngested]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
    disabled: state.status === "uploading",
  });

  const statusStyles: Record<Status, string> = {
    idle: "",
    uploading: "bg-blue-950/40 border-blue-700 text-blue-300",
    success: "bg-green-950/40 border-green-700 text-green-300",
    error: "bg-red-950/40 border-red-700 text-red-300",
  };

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={[
          "border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-150 select-none",
          isDragActive
            ? "border-blue-500 bg-blue-950/20"
            : "border-gray-700 hover:border-gray-500 hover:bg-gray-800/30",
          state.status === "uploading" ? "pointer-events-none opacity-60" : "",
        ].join(" ")}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-2">
          {state.status === "uploading" ? (
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
          ) : (
            <Upload
              className={`w-8 h-8 ${
                isDragActive ? "text-blue-400" : "text-gray-500"
              }`}
            />
          )}
          <p className="text-sm font-medium text-gray-300">
            {isDragActive ? "Drop PDF here" : "Drag & drop PDF"}
          </p>
          <p className="text-xs text-gray-600">or click to browse · max 50 MB</p>
        </div>
      </div>

      {/* Status */}
      {state.status !== "idle" && (
        <div
          className={`flex items-start gap-2 p-3 rounded-lg border text-xs ${statusStyles[state.status]}`}
        >
          {state.status === "uploading" && (
            <Loader2 className="w-3.5 h-3.5 mt-0.5 shrink-0 animate-spin" />
          )}
          {state.status === "success" && (
            <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          )}
          {state.status === "error" && (
            <XCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          )}
          <div className="space-y-0.5">
            <p>{state.message}</p>
            {state.result && (
              <p className="opacity-70">
                {state.result.pages_processed} pages →{" "}
                {state.result.chunks_stored} chunks indexed
              </p>
            )}
          </div>
        </div>
      )}

      {/* Ingested files */}
      {history.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Ingested
          </p>
          {history.map((name) => (
            <div
              key={name}
              className="flex items-center gap-2 bg-gray-800/50 rounded-lg px-3 py-2"
            >
              <FileText className="w-3.5 h-3.5 text-blue-400 shrink-0" />
              <span className="text-xs text-gray-300 truncate">{name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
