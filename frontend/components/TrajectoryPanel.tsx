"use client";

import type { NavigationState } from "@/types/navigation";

export function TrajectoryPanel({ samples }: { samples: NavigationState[] }) {
  const points = samples.map((sample) => ({ x: sample.local_x, y: sample.local_y }));
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(0, ...xs);
  const maxX = Math.max(1, ...xs);
  const minY = Math.min(0, ...ys);
  const maxY = Math.max(1, ...ys);
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const path = points
    .map((point, index) => {
      const x = 40 + ((point.x - minX) / width) * 520;
      const y = 320 - ((point.y - minY) / height) * 260;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  const latest = points[points.length - 1];

  return (
    <section className="h-full min-h-[360px] rounded-md border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-600">Navigation Map</h2>
        <span className="text-xs text-slate-500">Local ENU, no external tiles</span>
      </div>
      <svg className="mt-4 h-[300px] w-full rounded bg-panel" viewBox="0 0 600 360" role="img">
        <path d="M 40 300 C 160 260, 240 260, 340 205 S 500 120, 560 85" fill="none" stroke="#cbd5e1" strokeWidth="18" strokeLinecap="round" />
        <path d="M 40 300 C 160 260, 240 260, 340 205 S 500 120, 560 85" fill="none" stroke="#334155" strokeWidth="2" strokeDasharray="8 10" />
        {path && <path d={path} fill="none" stroke="#0f766e" strokeWidth="4" strokeLinecap="round" />}
        {latest && (
          <circle
            cx={40 + ((latest.x - minX) / width) * 520}
            cy={320 - ((latest.y - minY) / height) * 260}
            r="7"
            fill="#dc2626"
          />
        )}
      </svg>
    </section>
  );
}

