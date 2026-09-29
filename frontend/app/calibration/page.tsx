"use client";

import { useState } from "react";
import { useNavigationSocket } from "@/hooks/useNavigationSocket";
import { AppShell } from "@/components/AppShell";
import { postCommand } from "@/lib/edge";
import { CheckCircle2, Compass, Cpu, HelpCircle, Loader2, RefreshCw, RotateCcw, ShieldCheck, Zap } from "lucide-react";

function radToDeg(rad: number | null | undefined): string {
  if (rad === null || rad === undefined || isNaN(rad)) return "0.0";
  return ((rad * 180) / Math.PI).toFixed(1);
}

function formatVal(val: number | null | undefined, digits = 4): string {
  if (val === null || val === undefined || isNaN(val)) return "0.0000";
  return val.toFixed(digits);
}

const STAGES = [
  { id: "INITIALIZING", number: 1, name: "SENSOR INITIALIZATION", desc: "Verifying IMU stream integrity and SI units" },
  { id: "ESTIMATING_GRAVITY", number: 2, name: "GRAVITY ESTIMATION", desc: "Averaging stationary accelerometer samples to isolate roll & pitch" },
  { id: "ESTIMATING_BIAS", number: 3, name: "GYRO BIAS ESTIMATION", desc: "Estimating stationary rate gyro zero-offsets (bx, by, bz)" },
  { id: "ESTIMATING_HEADING", number: 4, name: "VEHICLE DIRECTION ESTIMATION", desc: "Deriving vehicle forward direction from GNSS course while moving" },
  { id: "VALIDATING", number: 5, name: "MAGNETOMETER CHECK", desc: "Cross-validating orientation against ambient magnetic flux" },
  { id: "COMPLETE", number: 6, name: "ALIGNMENT COMPLETE", desc: "Phone-to-vehicle transformation matrix active" },
];

function getActiveStageIndex(status: string | undefined): number {
  switch (status) {
    case "INITIALIZING":
      return 0;
    case "ESTIMATING_GRAVITY":
      return 1;
    case "ESTIMATING_BIAS":
      return 2;
    case "ESTIMATING_HEADING":
      return 3;
    case "VALIDATING":
      return 4;
    case "COMPLETE":
      return 5;
    default:
      return 0;
  }
}

export default function CalibrationPage() {
  const { state, connected } = useNavigationSocket();
  const [resetting, setResetting] = useState(false);

  const calibration = state?.calibration;
  const alignment = calibration?.alignment;
  const currentStatus = alignment?.status ?? "INITIALIZING";
  const activeStageIdx = getActiveStageIndex(currentStatus);
  const confidence = Math.round((alignment?.confidence ?? 0) * 100);

  const handleReset = async () => {
    setResetting(true);
    try {
      await postCommand("/calibration/reset");
    } finally {
      setTimeout(() => setResetting(false), 500);
    }
  };

  return (
    <AppShell edgeOnline={connected} rateHz={state?.rates?.navigation_hz} mode={state?.navigation_mode}>
      <div className="flex flex-col gap-5 max-w-6xl mx-auto w-full">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-cyan-500/20 pb-3">
          <div>
            <h1 className="text-xl font-bold tracking-[0.25em] text-white flex items-center gap-2 font-mono">
              <Compass className="text-cyan-400" size={22} />
              AUTOMATIC PHONE → VEHICLE ALIGNMENT
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              ESTIMATING 3D ROTATION QUATERNION & GYROSCOPE BIASES WITHOUT PRE-ALIGNED FIXTURES
            </p>
          </div>

          <button
            onClick={handleReset}
            disabled={resetting}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg border border-cyan-500/40 bg-cyan-950/40 text-cyan-300 hover:bg-cyan-900/40 transition-colors text-xs font-mono font-bold tracking-wider"
          >
            {resetting ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
            RESTART CALIBRATION
          </button>
        </div>

        {/* Primary Alignment State Readouts (ROLL | PITCH | YAW | CONFIDENCE) */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* ROLL */}
          <div className="p-4 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 text-xs tracking-widest font-mono">
              <span>ESTIMATED ROLL</span>
              <span className="text-sky-400">φ (X)</span>
            </div>
            <div className="mt-3 flex items-baseline gap-2">
              <span className="text-4xl font-bold font-mono text-white">
                {radToDeg(alignment?.roll_rad)}°
              </span>
              <span className="text-xs text-sky-400 font-mono">DEG</span>
            </div>
            <div className="mt-1 text-xs text-slate-500 font-mono">
              {formatVal(alignment?.roll_rad, 4)} rad
            </div>
          </div>

          {/* PITCH */}
          <div className="p-4 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 text-xs tracking-widest font-mono">
              <span>ESTIMATED PITCH</span>
              <span className="text-emerald-400">θ (Y)</span>
            </div>
            <div className="mt-3 flex items-baseline gap-2">
              <span className="text-4xl font-bold font-mono text-white">
                {radToDeg(alignment?.pitch_rad)}°
              </span>
              <span className="text-xs text-emerald-400 font-mono">DEG</span>
            </div>
            <div className="mt-1 text-xs text-slate-500 font-mono">
              {formatVal(alignment?.pitch_rad, 4)} rad
            </div>
          </div>

          {/* YAW */}
          <div className="p-4 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 text-xs tracking-widest font-mono">
              <span>ESTIMATED YAW</span>
              <span className="text-amber-400">ψ (Z)</span>
            </div>
            <div className="mt-3 flex items-baseline gap-2">
              <span className="text-4xl font-bold font-mono text-white">
                {radToDeg(alignment?.yaw_rad)}°
              </span>
              <span className="text-xs text-amber-400 font-mono">DEG</span>
            </div>
            <div className="mt-1 text-xs text-slate-500 font-mono">
              {formatVal(alignment?.yaw_rad, 4)} rad
            </div>
          </div>

          {/* CONFIDENCE */}
          <div className="p-4 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 text-xs tracking-widest font-mono">
              <span>CONFIDENCE</span>
              <ShieldCheck size={16} className="text-cyan-400" />
            </div>
            <div className="mt-3 flex items-baseline gap-2">
              <span className="text-4xl font-bold font-mono text-cyan-300">
                {confidence}%
              </span>
              <span className="text-xs text-cyan-400 font-mono">ACCURACY</span>
            </div>
            <div className="mt-1 text-xs text-slate-500 font-mono">
              STATUS: {currentStatus}
            </div>
          </div>
        </section>

        {/* 6-Stage Visual Pipeline Progress */}
        <section className="rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 p-5 flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-400 font-mono">
              Alignment Stages & Automated Validation
            </h2>
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
              <span>CURRENT STAGE:</span>
              <span className="text-cyan-300 font-bold px-2 py-0.5 rounded bg-cyan-950/50 border border-cyan-500/30">
                {currentStatus}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
            {STAGES.map((st, idx) => {
              const isPast = idx < activeStageIdx;
              const isCurrent = idx === activeStageIdx;
              const isComplete = currentStatus === "COMPLETE" || isPast;

              return (
                <div
                  key={st.id}
                  className={`p-3.5 rounded-lg border transition-all duration-200 flex flex-col justify-between gap-2 ${
                    isCurrent
                      ? "border-cyan-400 bg-cyan-950/25 shadow-[0_0_15px_rgba(6,182,212,0.2)]"
                      : isPast
                      ? "border-emerald-500/30 bg-emerald-950/15 text-slate-300"
                      : "border-slate-800 bg-black/20 text-slate-500"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-bold tracking-widest text-slate-400">
                      STAGE 0{st.number}
                    </span>
                    {isComplete ? (
                      <CheckCircle2 size={16} className="text-emerald-400" />
                    ) : isCurrent ? (
                      <span className="h-2 w-2 rounded-full bg-cyan-400 animate-ping" />
                    ) : (
                      <span className="h-2 w-2 rounded-full border border-slate-600" />
                    )}
                  </div>

                  <div>
                    <h3 className={`text-xs font-bold font-mono tracking-wider ${isCurrent ? "text-cyan-300" : isPast ? "text-emerald-300" : "text-slate-400"}`}>
                      {st.name}
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-1 font-sans leading-snug">
                      {st.desc}
                    </p>
                  </div>

                  <div className="text-[10px] font-mono pt-2 border-t border-slate-800/80 flex justify-between">
                    <span>STATE:</span>
                    <span className={isComplete ? "text-emerald-400 font-bold" : isCurrent ? "text-cyan-300 font-bold" : "text-slate-600"}>
                      {isComplete ? "VERIFIED" : isCurrent ? "IN PROGRESS" : "PENDING"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {calibration?.message && (
            <div className="p-3 rounded bg-black/30 border border-slate-800 text-xs font-mono text-slate-300 flex items-center gap-2">
              <Zap size={14} className="text-cyan-400 flex-shrink-0" />
              <span>{calibration.message}</span>
            </div>
          )}
        </section>

        {/* Gyro Bias Offsets & Diagnostic Statistics */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Gyro Biases */}
          <div className="p-4 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col gap-3">
            <h3 className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-400 font-mono border-b border-slate-800 pb-2">
              Estimated Gyroscope Biases
            </h3>
            <div className="grid grid-cols-3 gap-2 font-mono text-center">
              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500">BIAS X (rad/s)</div>
                <div className="text-sm font-bold text-white mt-1">{formatVal(alignment?.gyro_bias_x, 5)}</div>
              </div>
              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500">BIAS Y (rad/s)</div>
                <div className="text-sm font-bold text-white mt-1">{formatVal(alignment?.gyro_bias_y, 5)}</div>
              </div>
              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500">BIAS Z (rad/s)</div>
                <div className="text-sm font-bold text-white mt-1">{formatVal(alignment?.gyro_bias_z, 5)}</div>
              </div>
            </div>
            <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
              Temporal averaging during stationary window isolates temperature and factory drift. Calibrated measurements subtract these offsets prior to vehicle frame transformation.
            </p>
          </div>

          {/* Sample Acceptance & Consistency */}
          <div className="p-4 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col gap-3">
            <h3 className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-400 font-mono border-b border-slate-800 pb-2">
              Sample Quality & Auxiliary Checks
            </h3>
            <div className="grid grid-cols-2 gap-2 font-mono">
              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500">ACCEPTED SAMPLES</div>
                <div className="text-sm font-bold text-white mt-1">
                  {alignment?.accepted_samples ?? 0} / 80
                </div>
              </div>
              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500">MAGNETOMETER CHECK</div>
                <div className={`text-sm font-bold mt-1 ${alignment?.magnetometer_consistent ? "text-emerald-400" : "text-amber-400"}`}>
                  {alignment?.magnetometer_consistent === null ? "PENDING" : alignment?.magnetometer_consistent ? "CONSISTENT" : "ANOMALOUS"}
                </div>
              </div>
            </div>
            <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
              Samples with acceleration magnitude deviating from 9.81 m/s² by more than 0.35 m/s² are rejected to prevent dynamic vehicle accelerations from corrupting gravity roll/pitch vectors.
            </p>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
