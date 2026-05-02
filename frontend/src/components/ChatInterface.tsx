"use client";

import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import { AlertCircle, Bot, Loader2, Send, User } from "lucide-react";
import { Source, streamQuery } from "@/lib/api";
import SourceCitation from "./SourceCitation";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
        prev.map((m) =>
          m.id === asstId ? { ...m, streaming: false } : m
        )
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Request failed";
      setError(msg);
      setMessages((prev) => prev.filter((m) => m.id !== asstId));
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(e as unknown as FormEvent);
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-5 space-y-6">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          messages.map((msg) => (
            <div key={msg.id}>
              <div
                className={`flex gap-3 ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {msg.role === "assistant" && <Avatar role="assistant" />}

                <div
                  className={[
                    "max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
                    msg.role === "user"
                      ? "bg-blue-600 text-white rounded-tr-sm"
                      : "bg-gray-800 text-gray-100 rounded-tl-sm",
                  ].join(" ")}
                >
                  {msg.role === "assistant" ? (
                    <div className="prose prose-invert prose-sm max-w-none">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                      {msg.streaming && (
                        <span className="inline-block w-1.5 h-4 bg-blue-400 animate-pulse align-middle ml-0.5" />
                      )}
                    </div>
                  ) : (
                    msg.content
                  )}
                </div>

                {msg.role === "user" && <Avatar role="user" />}
              </div>

              {/* Citations appear after stream completes */}
              {msg.role === "assistant" &&
                !msg.streaming &&
                msg.sources &&
                msg.sources.length > 0 && (
                  <div className="ml-10 mt-2">
                    <SourceCitation sources={msg.sources} />
                  </div>
                )}
            </div>
          ))
        )}

        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-950/50 border border-red-800 text-red-300 text-sm">
            <AlertCircle className="w-4 h-4 shrink-0" />
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="px-4 pb-4 pt-3 border-t border-gray-800/60">
        <form onSubmit={submit} className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask anything about your documents…"
            disabled={busy}
            rows={1}
            className={[
              "flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3",
              "text-sm text-gray-100 placeholder-gray-600",
              "resize-none overflow-auto focus:outline-none focus:border-blue-500",
              "focus:ring-1 focus:ring-blue-500 transition-colors",
              "disabled:opacity-50",
            ].join(" ")}
            style={{ minHeight: "48px", maxHeight: "160px" }}
          />
          <button
            type="submit"
            disabled={!input.trim() || busy}
            className="w-11 h-11 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors shrink-0"
          >
            {busy ? (
              <Loader2 className="w-4 h-4 text-white animate-spin" />
            ) : (
              <Send className="w-4 h-4 text-white" />
            )}
          </button>
        </form>
        <p className="text-xs text-gray-700 mt-1.5 text-center">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}

function Avatar({ role }: { role: "user" | "assistant" }) {
  return (
    <div
      className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-1 ${
        role === "assistant" ? "bg-blue-600" : "bg-gray-700"
      }`}
    >
      {role === "assistant" ? (
        <Bot className="w-4 h-4 text-white" />
      ) : (
        <User className="w-4 h-4 text-gray-300" />
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-center gap-3 text-gray-600">
      <Bot className="w-12 h-12 text-gray-700" />
      <div>
        <p className="font-medium text-gray-400">Ready to answer</p>
        <p className="text-sm mt-1">Upload a PDF on the left, then ask anything</p>
      </div>
      <div className="mt-4 grid grid-cols-1 gap-2 text-xs max-w-sm w-full">
        {[
          "What are the main findings of this paper?",
          "Summarise section 3 in bullet points",
          "What methodology was used?",
        ].map((hint) => (
          <div
            key={hint}
            className="border border-gray-800 rounded-lg px-3 py-2 text-gray-600 cursor-default"
          >
            {hint}
          </div>
        ))}
      </div>
    </div>
  );
}
