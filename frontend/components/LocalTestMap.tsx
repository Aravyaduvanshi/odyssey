"use client";

import { useMemo, useState } from "react";
import type { NavigationState } from "@/types/navigation";
import { Eye, EyeOff, Layers, MapPin, Navigation } from "lucide-react";

interface LocalTestMapProps {
  samples: NavigationState[];
  state: NavigationState | null;
}

export function LocalTestMap({ samples, state }: LocalTestMapProps) {
  const [showTruth, setShowTruth] = useState(true);
  const [showEkf, setShowEkf] = useState(true);
  const [showIns, setShowIns] = useState(true);
  const [showGnss, setShowGnss] = useState(true);
  const [showGrid, setShowGrid] = useState(true);

  // Compute bounding box for ENU coordinate system
  const bounds = useMemo(() => {
    let minX = -20;
    let maxX = 80;
    let minY = -20;
    let maxY = 120;

    for (const s of samples) {
      if (s.local_x < minX) minX = s.local_x - 15;
      if (s.local_x > maxX) maxX = s.local_x + 15;
      if (s.local_y < minY) minY = s.local_y - 15;
      if (s.local_y > maxY) maxY = s.local_y + 15;
      if (s.ins_position) {
        if (s.ins_position.E < minX) minX = s.ins_position.E - 15;
        if (s.ins_position.E > maxX) maxX = s.ins_position.E + 15;
        if (s.ins_position.N < minY) minY = s.ins_position.N - 15;
        if (s.ins_position.N > maxY) maxY = s.ins_position.N + 15;
      }
      if (s.ground_truth) {
        if (s.ground_truth.local_x < minX) minX = s.ground_truth.local_x - 15;
        if (s.ground_truth.local_x > maxX) maxX = s.ground_truth.local_x + 15;
        if (s.ground_truth.local_y < minY) minY = s.ground_truth.local_y - 15;
        if (s.ground_truth.local_y > maxY) maxY = s.ground_truth.local_y + 15;
      }
    }

    // Keep aspect ratio reasonable
    const rangeX = Math.max(80, maxX - minX);
    const rangeY = Math.max(80, maxY - minY);
    return { minX, maxX, minY, maxY, rangeX, rangeY };
  }, [samples]);

  // Coordinate projector from ENU meters to SVG canvas [900 x 560]
  const svgWidth = 900;
  const svgHeight = 560;
  const padding = 50;

  const project = (x: number, y: number) => {
    const px = padding + ((x - bounds.minX) / bounds.rangeX) * (svgWidth - 2 * padding);
    const py = (svgHeight - padding) - ((y - bounds.minY) / bounds.rangeY) * (svgHeight - 2 * padding);
    return { x: px, y: py };
  };

  // Build SVG path strings
  const ekfPathString = useMemo(() => {
    if (samples.length === 0) return "";
    return samples
      .map((s, i) => {
        const x = s.ekf_position ? s.ekf_position.E : s.local_x;
        const y = s.ekf_position ? s.ekf_position.N : s.local_y;
        const pt = project(x, y);
        return `${i === 0 ? "M" : "L"} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`;
      })
      .join(" ");
  }, [samples, bounds]);

  const insPathString = useMemo(() => {
    const insSamples = samples.filter((s) => s.ins_position);
    if (insSamples.length === 0) return "";
    return insSamples
      .map((s, i) => {
        const pt = project(s.ins_position!.E, s.ins_position!.N);
        return `${i === 0 ? "M" : "L"} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`;
      })
      .join(" ");
  }, [samples, bounds]);

  const truthPathString = useMemo(() => {
    const pts = samples.filter((s) => s.ground_truth).map((s) => project(s.ground_truth!.local_x, s.ground_truth!.local_y));
    if (pts.length === 0) return "";
    return pts
      .map((pt, i) => `${i === 0 ? "M" : "L"} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`)
      .join(" ");
  }, [samples, bounds]);

  const gnssPoints = useMemo(() => {
    return samples
      .filter((s) => s.gnss_position && s.gnss_health > 0.5)
      .map((s) => project(s.local_x, s.local_y));
  }, [samples, bounds]);

  const currentMarker = state ? project(state.local_x, state.local_y) : project(0, 0);
  const headingDeg = state?.heading_deg ?? 0;
  const isBlackout = state?.navigation_mode === "DEAD_RECKONING";

  return (
    <div className="relative flex-1 min-h-[440px] lg:min-h-[540px] rounded-lg border border-cyan-500/20 bg-[#080c14] overflow-hidden flex flex-col shadow-2xl">
      {/* Tactical Map Header Bar */}
      <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 pointer-events-auto">
          <div className="px-3 py-1 rounded bg-[#0a0f1b]/90 border border-cyan-500/30 text-cyan-300 font-bold text-xs tracking-[0.2em] shadow-lg flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${isBlackout ? "bg-rose-400 animate-ping" : "bg-cyan-400 animate-ping"}`} />
            {state?.map_label ?? "LOCAL TEST MAP"}
          </div>
          {isBlackout && (
            <div className="px-2.5 py-1 rounded bg-rose-950/80 border border-rose-500/60 text-rose-300 text-[11px] font-mono font-bold tracking-wider animate-pulse flex items-center gap-1.5">
              <span>GNSS OUTAGE</span>
            </div>
          )}
          <div className="hidden sm:flex px-2.5 py-1 rounded bg-[#0a0f1b]/90 border border-slate-800 text-slate-400 text-[11px] font-mono">
            REF: ENU (X=EAST, Y=NORTH)
          </div>
        </div>

        {/* Layer Toggles */}
        <div className="flex items-center gap-1.5 pointer-events-auto bg-[#0a0f1b]/90 p-1 rounded border border-slate-800 backdrop-blur">
          <button
            onClick={() => setShowTruth(!showTruth)}
            title="Toggle Ground Truth Layer"
            className={`px-2 py-1 rounded text-[10px] font-bold tracking-wider flex items-center gap-1 transition-colors ${
              showTruth ? "bg-sky-500/20 text-sky-300 border border-sky-500/40" : "text-slate-500 hover:text-slate-300"
            }`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />
            TRUTH
          </button>
          <button
            onClick={() => setShowEkf(!showEkf)}
            title="Toggle EKF Fused Estimate Layer"
            className={`px-2 py-1 rounded text-[10px] font-bold tracking-wider flex items-center gap-1 transition-colors ${
              showEkf ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40" : "text-slate-500 hover:text-slate-300"
            }`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            EKF
          </button>
          <button
            onClick={() => setShowIns(!showIns)}
            title="Toggle Pure Strapdown INS Layer"
            className={`px-2 py-1 rounded text-[10px] font-bold tracking-wider flex items-center gap-1 transition-colors ${
              showIns ? "bg-amber-500/20 text-orange-300 border border-orange-500/40" : "text-slate-500 hover:text-slate-300"
            }`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-orange-400" />
            INS
          </button>
          <button
            onClick={() => setShowGnss(!showGnss)}
            title="Toggle GNSS Fixes"
            className={`px-2 py-1 rounded text-[10px] font-bold tracking-wider flex items-center gap-1 transition-colors ${
              showGnss ? "bg-yellow-500/20 text-yellow-300 border border-yellow-500/40" : "text-slate-500 hover:text-slate-300"
            }`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-yellow-400" />
            GNSS
          </button>
        </div>
      </div>

      {/* Multi-Layer Trajectory Legend */}
      <div className="absolute top-12 left-3 z-10 hidden sm:flex flex-wrap items-center gap-1.5 pointer-events-none text-[10px] font-mono">
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-black/60 border border-slate-800/80 text-sky-300 backdrop-blur">
          <span className="w-3 border-b-2 border-sky-400 border-dashed" />
          <span>TRUTH</span>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-black/60 border border-slate-800/80 text-emerald-300 backdrop-blur">
          <span className="w-3 h-1 bg-emerald-400 rounded-full" />
          <span>EKF FUSION</span>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-black/60 border border-slate-800/80 text-orange-300 backdrop-blur">
          <span className="w-3 border-b border-orange-400 border-dashed" />
          <span>INS DRIFT</span>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-black/60 border border-slate-800/80 text-yellow-300 backdrop-blur">
          <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full" />
          <span>GNSS FIX</span>
        </div>
      </div>

      {/* SVG Navigation Canvas */}
      <svg
        className="w-full h-full flex-1"
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Tactical Navigation Map"
      >
        <defs>
          <radialGradient id="grid-glow" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#080c14" stopOpacity="0" />
          </radialGradient>
          <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Ambient Grid Background */}
        <rect width={svgWidth} height={svgHeight} fill="url(#grid-glow)" />

        {/* Coordinate Reference Crosshairs & Grid */}
        {showGrid && (
          <g stroke="rgba(56, 189, 248, 0.09)" strokeWidth="1">
            {/* Horizontal Grid lines */}
            {[0.2, 0.4, 0.6, 0.8].map((ratio) => (
              <line key={`h-${ratio}`} x1="0" y1={svgHeight * ratio} x2={svgWidth} y2={svgHeight * ratio} />
            ))}
            {/* Vertical Grid lines */}
            {[0.2, 0.4, 0.6, 0.8].map((ratio) => (
              <line key={`v-${ratio}`} x1={svgWidth * ratio} y1="0" x2={svgWidth * ratio} y2={svgHeight} />
            ))}
            {/* Origin indicator */}
            {(() => {
              const o = project(0, 0);
              return (
                <g>
                  <circle cx={o.x} cy={o.y} r="4" fill="none" stroke="#06b6d4" strokeWidth="1.5" opacity="0.6" />
                  <text x={o.x + 8} y={o.y + 4} fill="#64748b" fontSize="9" fontFamily="monospace">
                    (0, 0) ORIGIN
                  </text>
                </g>
              );
            })()}
          </g>
        )}

        {/* Ground Truth Trajectory */}
        {showTruth && truthPathString && (
          <g>
            <path
              d={truthPathString}
              fill="none"
              stroke="#0284c7"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray="6 6"
              opacity="0.8"
            />
          </g>
        )}

        {/* Pure Strapdown INS Trajectory (Drift Baseline) */}
        {showIns && insPathString && (
          <g>
            <path
              d={insPathString}
              fill="none"
              stroke="#f97316"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray="4 4"
              opacity="0.75"
            />
          </g>
        )}

        {/* Fused EKF / OdysseyV1 Estimated Trajectory */}
        {showEkf && ekfPathString && (
          <g>
            <path
              d={ekfPathString}
              fill="none"
              stroke="#10b981"
              strokeWidth="3.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              filter="url(#neon-glow)"
            />
          </g>
        )}

        {/* GNSS Fix Scatter */}
        {showGnss && (
          <g>
            {gnssPoints.map((pt, idx) => (
              <g key={idx}>
                <circle cx={pt.x} cy={pt.y} r="3" fill="#eab308" opacity="0.6" />
                <circle cx={pt.x} cy={pt.y} r="5" fill="none" stroke="#eab308" strokeWidth="0.5" opacity="0.3" />
              </g>
            ))}
          </g>
        )}

        {/* Current Vehicle Position & Orientation Marker */}
        {state && (
          <g
            transform={`translate(${currentMarker.x}, ${currentMarker.y}) rotate(${headingDeg})`}
            filter="url(#neon-glow)"
          >
            {/* Pulse Ring */}
            <circle cx="0" cy="0" r="18" fill="none" stroke="#06b6d4" strokeWidth="1.5" opacity="0.3" className="animate-ping" />
            {/* Heading Vector */}
            <line x1="0" y1="0" x2="0" y2="-28" stroke="#38bdf8" strokeWidth="2" strokeDasharray="3 3" />
            {/* Automotive Vehicle Chevron Icon */}
            <path
              d="M 0 -16 L 10 12 L 0 6 L -10 12 Z"
              fill="#f8fafc"
              stroke="#06b6d4"
              strokeWidth="2.5"
              strokeLinejoin="round"
            />
            <circle cx="0" cy="0" r="3" fill="#06b6d4" />
          </g>
        )}
      </svg>

      {/* Floating Bottom Telemetry Badge */}
      <div className="absolute bottom-3 left-3 right-3 z-10 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        <div className="flex items-center gap-3 px-3 py-1.5 rounded-md bg-[#0a0f1b]/95 border border-slate-800 text-xs font-mono pointer-events-auto backdrop-blur shadow-lg">
          <div>
            <span className="text-slate-500">EAST: </span>
            <span className="text-cyan-300 font-bold">{state ? `${state.local_x.toFixed(1)} m` : "0.0 m"}</span>
          </div>
          <div>
            <span className="text-slate-500">NORTH: </span>
            <span className="text-cyan-300 font-bold">{state ? `${state.local_y.toFixed(1)} m` : "0.0 m"}</span>
          </div>
          <div>
            <span className="text-slate-500">HEADING: </span>
            <span className="text-emerald-300 font-bold">{state ? `${state.heading_deg.toFixed(1)}°` : "0.0°"}</span>
          </div>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#0a0f1b]/95 border border-slate-800 text-[11px] font-mono pointer-events-auto backdrop-blur text-slate-400">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          <span>VEHICLE: </span>
          <span className="text-white font-bold">{state?.scenario_phase ?? "STATIONARY"}</span>
        </div>
      </div>
    </div>
  );
}
