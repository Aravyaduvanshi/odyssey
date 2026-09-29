interface MetricProps {
  label: string;
  value: string;
  tone?: "normal" | "good" | "warn";
}

export function Metric({ label, value, tone = "normal" }: MetricProps) {
  const toneClass = tone === "good" ? "text-signal" : tone === "warn" ? "text-caution" : "text-ink";
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-xl font-semibold ${toneClass}`}>{value}</div>
    </div>
  );
}

