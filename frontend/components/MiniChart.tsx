"use client";

export function MiniChart({ title, values, color = "#22d3ee" }: { title: string; values: number[]; color?: string }) {
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  const range = Math.max(0.001, max - min);
  const path = values
    .map((value, index) => {
      const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * 360;
      const y = 120 - ((value - min) / range) * 100;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <section className="rounded-md border border-cyan-400/15 bg-[#0d131c] p-4">
      <h2 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">{title}</h2>
      <svg className="mt-3 h-36 w-full" viewBox="0 0 360 140">
        <path d="M 0 120 H 360" stroke="#334155" />
        <path d={path} fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
      </svg>
    </section>
  );
}

