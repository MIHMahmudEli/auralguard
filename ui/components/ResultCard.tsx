export type DetectResult = {
  verdict: "ai_generated" | "human";
  p_ai_generated: number;
  confidence: "low" | "medium" | "high";
  score: number;
  threshold: number;
  windows: number;
  model_version: string;
  duration_s: number;
  demo?: boolean;
};

export default function ResultCard({
  result,
  fileName,
}: {
  result: DetectResult;
  fileName: string | null;
}) {
  const isAI = result.verdict === "ai_generated";
  const pct = Math.round(result.p_ai_generated * 100);

  return (
    <section
      className={`rounded-2xl border p-6 shadow-sm ${
        isAI
          ? "border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40"
          : "border-emerald-300 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950/40"
      }`}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-500 dark:text-slate-400">{fileName}</p>
          <h2 className="text-2xl font-bold">
            {isAI ? "🤖 Likely AI-generated" : "🧑 Likely human"}
          </h2>
        </div>
        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold uppercase ${
            result.confidence === "high"
              ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
              : "bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-200"
          }`}
        >
          {result.confidence} confidence
        </span>
      </div>

      <div className="mt-4">
        <div className="flex justify-between text-sm">
          <span>Probability AI-generated</span>
          <span className="font-mono font-semibold">{pct}%</span>
        </div>
        <div className="mt-1 h-3 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
          <div
            className={`h-full ${isAI ? "bg-red-500" : "bg-emerald-500"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-2 text-sm text-slate-600 dark:text-slate-300">
        <div className="flex justify-between">
          <dt>Raw score</dt>
          <dd className="font-mono">{result.score}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Threshold</dt>
          <dd className="font-mono">{result.threshold}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Duration</dt>
          <dd className="font-mono">{result.duration_s}s</dd>
        </div>
        <div className="flex justify-between">
          <dt>Windows</dt>
          <dd className="font-mono">{result.windows}</dd>
        </div>
      </dl>

      {result.demo && (
        <p className="mt-4 rounded-lg bg-amber-100 p-2 text-center text-xs text-amber-800 dark:bg-amber-950 dark:text-amber-300">
          ⚠️ DEMO mode — no trained weights loaded; score is not meaningful.
        </p>
      )}
    </section>
  );
}
