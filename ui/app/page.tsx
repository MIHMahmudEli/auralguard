"use client";

import { useRef, useState } from "react";
import ResultCard, { DetectResult } from "@/components/ResultCard";

export default function Home() {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<DetectResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function detect(file: Blob, name: string) {
    setBusy(true);
    setError(null);
    setResult(null);
    setFileName(name);
    try {
      const fd = new FormData();
      fd.append("file", file, name);
      const res = await fetch("/api/detect", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error ?? "detection failed");
      setResult(data);
    } catch (e: any) {
      setError(e?.message ?? "unknown error");
    } finally {
      setBusy(false);
    }
  }

  function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) detect(f, f.name);
  }

  async function toggleRecord() {
    if (recording) {
      mediaRef.current?.stop();
      setRecording(false);
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream);
    chunksRef.current = [];
    rec.ondataavailable = (ev) => chunksRef.current.push(ev.data);
    rec.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      stream.getTracks().forEach((t) => t.stop());
      detect(blob, "recording.webm");
    };
    rec.start();
    mediaRef.current = rec;
    setRecording(true);
  }

  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-8 px-6 py-16">
      <header className="text-center">
        <h1 className="text-4xl font-bold tracking-tight">🛡️ AuralGuard</h1>
        <p className="mt-2 text-slate-500 dark:text-slate-400">
          Is this voice <span className="font-semibold">AI-generated</span> or human?
          Upload a clip or record one.
        </p>
      </header>

      <section className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <label className="flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed border-slate-300 p-8 text-center transition hover:border-indigo-400 dark:border-slate-700">
          <span className="text-sm text-slate-500 dark:text-slate-400">
            Click to upload (wav, mp3, flac, m4a, ogg)
          </span>
          <input
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={onUpload}
            disabled={busy}
          />
        </label>

        <div className="flex items-center gap-3">
          <div className="h-px flex-1 bg-slate-200 dark:bg-slate-800" />
          <span className="text-xs text-slate-400">or</span>
          <div className="h-px flex-1 bg-slate-200 dark:bg-slate-800" />
        </div>

        <button
          onClick={toggleRecord}
          disabled={busy}
          className={`rounded-xl px-4 py-3 font-medium text-white transition ${
            recording ? "bg-red-600 hover:bg-red-700" : "bg-indigo-600 hover:bg-indigo-700"
          } disabled:opacity-50`}
        >
          {recording ? "◼ Stop & analyze" : "● Record from microphone"}
        </button>
      </section>

      {busy && (
        <p className="text-center text-slate-500">Analyzing {fileName}…</p>
      )}
      {error && (
        <p className="rounded-lg bg-red-50 p-4 text-center text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}
      {result && <ResultCard result={result} fileName={fileName} />}

      <footer className="text-center text-xs text-slate-400">
        Probabilistic tool. Not a sole basis for high-stakes decisions. Model:{" "}
        {result?.model_version ?? "AuralGuard"}.
      </footer>
    </main>
  );
}
