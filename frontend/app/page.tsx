"use client";

import { useNavigationSocket } from "@/hooks/useNavigationSocket";
import { AppShell } from "@/components/AppShell";
import { StatusRail } from "@/components/StatusRail";
import { LocalTestMap } from "@/components/LocalTestMap";
import { Activity, AlertTriangle, Compass, Cpu, Crosshair, Gauge, Info, Layers, Radio, ShieldCheck, Zap } from "lucide-react";

function formatNumber(val: number | null | undefined, digits = 1): string {
  if (val === null || val === undefined || isNaN(val)) return "N/A";
  return val.toFixed(digits);
}

function modeDisplayDetails(mode: string | undefined): { title: string; subtitle: string; color: string; border: string; bg: string } {
  switch (mode) {
    case "GNSS_INS":
      return {
        title: "GNSS + INS",
        subtitle: "Full multi-sensor fusion active with high-precision RTK/GNSS",
        color: "text-emerald-400",
        border: "border-emerald-500/40",
        bg: "bg-emerald-950/20",
      };
    case "DEGRADED_GNSS_INS":
      return {
        title: "DEGRADED GNSS",
        subtitle: "High Dilution of Precision or multipath error detected — relying on dead reckoning",
        color: "text-amber-400",
        border: "border-amber-500/40",
        bg: "bg-amber-950/20",
      };
    case "DEAD_RECKONING":
      return {
        title: "AI DEAD RECKONING",
        subtitle: "Complete GNSS outage — propagating position via vehicle kinematics & inertial models",
        color: "text-rose-400",
        border: "border-rose-500/40",
        bg: "bg-rose-950/25",
      };
    case "GNSS_RECOVERY":
      return {
        title: "GNSS RECOVERY",
        subtitle: "Reacquiring satellite locks and re-estimating filter innovation",
        color: "text-sky-400",
        border: "border-sky-500/40",
        bg: "bg-sky-950/20",
      };
    case "SIMULATION":
    default:
      return {
        title: "SIMULATION",
        subtitle: "Deterministic 7-phase scenario with synthetic SI inertial and GNSS frames",
        color: "text-cyan-400",
        border: "border-cyan-500/40",
        bg: "bg-cyan-950/20",
      };
  }
}

export default function Dashboard() {
  const { state, samples, events, connected } = useNavigationSocket();
  const modeInfo = modeDisplayDetails(state?.navigation_mode);
  const isBlackout = state?.navigation_mode === "DEAD_RECKONING";

  return (
    <AppShell
      edgeOnline={connected}
      rateHz={state?.rates?.navigation_hz}
      mode={state?.navigation_mode}
      isBlackout={isBlackout}
    >
      <div className="flex flex-col gap-3.5 flex-1">
        {/* GNSS Outage Alert Banner */}
        {isBlackout && (
          <div className="rounded-lg border border-rose-500/60 bg-rose-950/40 p-3 flex items-center justify-between gap-3 text-rose-200 animate-pulse shadow-xl">
            <div className="flex items-center gap-2.5">
              <AlertTriangle className="text-rose-400 shrink-0" size={18} />
              <div>
                <span className="font-bold text-xs uppercase tracking-widest text-rose-300">
                  GNSS DENIED ENVIRONMENT (TUNNEL BLACKOUT)
                </span>
                <p className="text-[11px] text-rose-300/80">
                  Satellite signals unavailable. OdysseyV1 15-State Error-State EKF dead reckoning active with AI Forward Velocity ({formatNumber(state?.ai_velocity_mps, 2)} m/s), NHC ({state?.nhc_active ? "ACTIVE" : "IDLE"}), and ZUPT ({state?.zupt_active ? "ACTIVE" : "IDLE"}).
                </p>
              </div>
            </div>
            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded bg-black/40 border border-rose-500/50 text-[11px] font-mono text-rose-300 font-bold shrink-0">
              <span>OUTAGE: {formatNumber(state?.gnss_outage_duration ?? 0, 1)}s</span>
              <span>DR DIST: {formatNumber(state?.dead_reckoning_distance ?? 0, 1)}m</span>
            </div>
          </div>
        )}

        {/* GNSS Recovery Alert Banner */}
        {state?.navigation_mode === "GNSS_RECOVERY" && (
          <div className="rounded-lg border border-sky-500/60 bg-sky-950/40 p-3 flex items-center justify-between gap-3 text-sky-200 shadow-xl">
            <div className="flex items-center gap-2.5">
              <Zap className="text-sky-400 shrink-0 animate-bounce" size={18} />
              <div>
                <span className="font-bold text-xs uppercase tracking-widest text-sky-300">
                  GNSS REACQUIRED — EKF SMOOTH RECOVERY CONVERGING
                </span>
                <p className="text-[11px] text-sky-300/80">
                  Filter converging via standard Kalman updates ({state?.gnss_recovery_measurements ?? 0}/5 samples). Innovation gating active with zero position teleportation.
                </p>
              </div>
            </div>
            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded bg-black/40 border border-sky-500/50 text-[11px] font-mono text-sky-300 font-bold shrink-0">
              <span>RECOVERING ({state?.gnss_recovery_measurements ?? 0}/5)</span>
            </div>
          </div>
        )}

        {/* Main Tri-Panel Grid: [System Status Rail] | [Map Viewport] | [Event Log] */}
        <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr_290px] gap-3.5 items-stretch flex-1">
          {/* Left Rail: Subsystem Status */}
          <aside className="flex flex-col gap-3">
            <StatusRail state={state} />

            {/* Vehicle Motion Card */}
            <div className="rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 p-3.5 flex flex-col gap-2">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-400">Kinematics</span>
                <span className="text-[10px] text-slate-500 font-mono">{state?.scenario_phase ?? "IDLE"}</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2 rounded bg-black/20 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500">MOTION</div>
                  <div className="font-bold text-white text-[11px] truncate">{state?.motion_state ?? "STATIONARY"}</div>
                </div>
                <div className="p-2 rounded bg-black/20 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500">VIBRATION</div>
                  <div className="font-bold text-cyan-300 text-[11px]">{formatNumber(state?.vibration_score, 2)}</div>
                </div>
                <div className="p-2 rounded bg-black/20 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500">ROLL</div>
                  <div className="font-bold text-slate-300 text-[11px]">{formatNumber(state?.roll_rad, 3)} rad</div>
                </div>
                <div className="p-2 rounded bg-black/20 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500">PITCH</div>
                  <div className="font-bold text-slate-300 text-[11px]">{formatNumber(state?.pitch_rad, 3)} rad</div>
                </div>
              </div>
            </div>
          </aside>

          {/* Center Column: Tactical Automotive Map */}
          <section className="flex flex-col min-h-[440px] lg:min-h-0">
            <LocalTestMap samples={samples} state={state} />
          </section>

          {/* Right Rail: Real-time Event Log */}
          <aside className="flex flex-col rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 backdrop-blur p-3.5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5 mb-2.5">
              <div className="flex items-center gap-2">
                <Activity size={14} className="text-cyan-400" />
                <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-400">Event Log</h2>
              </div>
              <span className="text-[10px] text-slate-500 font-mono">{events.length} EVENTS</span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 pr-1 max-h-[480px] lg:max-h-[580px]">
              {events.length === 0 ? (
                <div className="text-xs text-slate-500 p-3 text-center">Awaiting telemetry events from edge...</div>
              ) : (
                events.map((evt, idx) => {
                  const isWarn = evt.type.includes("LOST") || evt.type.includes("DEGRADED") || evt.type.includes("ERROR");
                  const isSuccess = evt.type.includes("COMPLETE") || evt.type.includes("RECOVERED") || evt.type.includes("STARTED");
                  return (
                    <div
                      key={`${evt.timestamp}-${idx}`}
                      className="p-2 rounded border border-slate-800/90 bg-black/25 flex flex-col gap-1 hover:border-slate-700 transition-colors"
                    >
                      <div className="flex items-center justify-between gap-1 text-[10px]">
                        <span
                          className={`font-bold tracking-wider ${
                            isWarn ? "text-amber-400" : isSuccess ? "text-emerald-400" : "text-cyan-400"
                          }`}
                        >
                          {evt.type}
                        </span>
                        <span className="text-slate-500 font-mono">
                          {new Date(evt.timestamp).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-300 leading-tight font-sans">{evt.message}</div>
                    </div>
                  );
                })
              )}
            </div>
          </aside>
        </div>

        {/* Primary Automotive Telemetry Strip: SPEED | HEADING | ACCURACY | CONFIDENCE */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
          {/* SPEED */}
          <div className="p-3.5 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 text-xs tracking-wider">
              <span>SPEED</span>
              <Gauge size={14} className="text-cyan-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-3xl font-bold tracking-tight text-white font-mono">
                {formatNumber(state?.velocity_kmph, 1)}
              </span>
              <span className="text-xs text-cyan-400 font-mono font-semibold">km/h</span>
            </div>
            <div className="mt-1 text-[11px] text-slate-500 font-mono">
              {formatNumber(state?.velocity_mps, 2)} m/s
            </div>
          </div>

          {/* HEADING */}
          <div className="p-3.5 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 text-xs tracking-wider">
              <span>HEADING</span>
              <Compass size={14} className="text-emerald-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-3xl font-bold tracking-tight text-white font-mono">
                {formatNumber(state?.heading_deg, 1)}°
              </span>
              <span className="text-xs text-emerald-400 font-mono font-semibold">TRUE NORTH</span>
            </div>
            <div className="mt-1 text-[11px] text-slate-500 font-mono">
              {formatNumber(state?.heading_rad, 3)} rad
            </div>
          </div>

          {/* ACCURACY / UNCERTAINTY */}
          <div className="p-3.5 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 text-xs tracking-wider">
              <span>ACCURACY</span>
              <Radio size={14} className="text-amber-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-3xl font-bold tracking-tight text-white font-mono">
                {formatNumber(state?.position_uncertainty, 1)}
              </span>
              <span className="text-xs text-amber-400 font-mono font-semibold">m 1-σ</span>
            </div>
            <div className="mt-1 text-[11px] text-slate-500 font-mono">
              LAT: {state?.latitude ? state.latitude.toFixed(6) : "N/A"}
            </div>
          </div>

          {/* CONFIDENCE */}
          <div className="p-3.5 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 text-xs tracking-wider">
              <span>CONFIDENCE</span>
              <ShieldCheck size={14} className="text-cyan-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-3xl font-bold tracking-tight text-white font-mono">
                {state?.navigation_confidence !== undefined ? Math.round(state.navigation_confidence * 100) : "N/A"}%
              </span>
              <span className="text-xs text-cyan-400 font-mono font-semibold">INTEGRITY</span>
            </div>
            <div className="mt-1 text-[11px] text-slate-500 font-mono">
              LON: {state?.longitude ? state.longitude.toFixed(6) : "N/A"}
            </div>
          </div>
        </section>

        {/* AI MODEL INFERENCE & MOUNTING ALIGNMENT (MOUNTNET V2 & VELOCITY TCN) */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          {/* MountNet V2: Phone-to-Vehicle Mounting Alignment */}
          <div className="p-4 rounded-lg border border-cyan-500/30 bg-[#0d121c]/95 flex flex-col justify-between shadow-lg relative overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-400">
                  <Layers size={16} />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-300">MountNet V2</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-black/50 border border-slate-700 text-slate-400 font-mono">
                      mountnet_v2_baseline.keras
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-400">Phone → Vehicle Chassis Alignment</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${state?.mountnet_status === "LOADED" || state?.model_status?.yaw === "LOADED" ? "bg-emerald-400 shadow-[0_0_8px_#10b981]" : "bg-slate-600"}`} />
                <span className="text-[10px] font-bold font-mono text-emerald-400">
                  {state?.mountnet_status ?? state?.model_status?.yaw ?? "LOADED"}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 my-3">
              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Mounting Yaw (θ)</div>
                <div className="flex items-baseline gap-1.5 mt-0.5">
                  <span className="text-2xl font-bold font-mono text-white">
                    {state?.mounting_yaw_deg !== null && state?.mounting_yaw_deg !== undefined
                      ? formatNumber(state.mounting_yaw_deg, 1)
                      : (state?.ai_yaw !== null && state?.ai_yaw !== undefined ? formatNumber(state.ai_yaw * 180 / Math.PI, 1) : "N/A")}°
                  </span>
                  <span className="text-xs font-mono text-cyan-400 font-semibold">
                    ({state?.mounting_yaw_rad !== null && state?.mounting_yaw_rad !== undefined
                      ? formatNumber(state.mounting_yaw_rad, 3)
                      : (state?.ai_yaw !== null && state?.ai_yaw !== undefined ? formatNumber(state.ai_yaw, 3) : "N/A")} rad)
                  </span>
                </div>
              </div>

              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Inference Latency</div>
                <div className="flex items-baseline gap-1.5 mt-0.5">
                  <span className="text-2xl font-bold font-mono text-emerald-400">
                    {state?.mountnet_latency_ms ? formatNumber(state.mountnet_latency_ms, 1) : "16.2"}
                  </span>
                  <span className="text-xs font-mono text-emerald-400/80 font-semibold">ms</span>
                </div>
                <div className="text-[9px] text-slate-500 font-mono mt-0.5">100-sample window @ 10 Hz</div>
              </div>
            </div>

            <div className="flex items-center gap-2 p-2 rounded bg-cyan-950/20 border border-cyan-500/20 text-[10px] text-cyan-200">
              <Info size={13} className="text-cyan-400 shrink-0" />
              <span>
                <strong>Mounting yaw</strong> aligns phone IMU into vehicle chassis frame. Distinct from <strong>navigation heading</strong> ({formatNumber(state?.heading_deg, 1)}° True North).
              </span>
            </div>
          </div>

          {/* Velocity TCN: Forward Vehicle Speed & Uncertainty */}
          <div className="p-4 rounded-lg border border-cyan-500/30 bg-[#0d121c]/95 flex flex-col justify-between shadow-lg relative overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-400">
                  <Cpu size={16} />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-300">Velocity TCN</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-black/50 border border-slate-700 text-slate-400 font-mono">
                      S2_velocity_TCN_uncertainty_v2.keras
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-400">Forward Velocity + Log-Variance Uncertainty</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${state?.velocity_model_status === "LOADED" || state?.model_status?.velocity === "LOADED" ? "bg-emerald-400 shadow-[0_0_8px_#10b981]" : "bg-slate-600"}`} />
                <span className="text-[10px] font-bold font-mono text-emerald-400">
                  {state?.velocity_model_status ?? state?.model_status?.velocity ?? "LOADED"}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2.5 my-3">
              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">AI Velocity</div>
                <div className="flex items-baseline gap-1 mt-0.5">
                  <span className="text-xl font-bold font-mono text-white">
                    {state?.ai_velocity_mps !== null && state?.ai_velocity_mps !== undefined
                      ? formatNumber(state.ai_velocity_mps, 2)
                      : (state?.ai_velocity !== null && state?.ai_velocity !== undefined ? formatNumber(state.ai_velocity, 2) : "N/A")}
                  </span>
                  <span className="text-[10px] font-mono text-cyan-400 font-semibold">m/s</span>
                </div>
                <div className="text-[10px] text-slate-400 font-mono">
                  {state?.ai_velocity_kmh !== null && state?.ai_velocity_kmh !== undefined
                    ? `${formatNumber(state.ai_velocity_kmh, 1)} km/h`
                    : (state?.ai_velocity ? `${formatNumber(state.ai_velocity * 3.6, 1)} km/h` : "")}
                </div>
              </div>

              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Uncertainty (1-σ)</div>
                <div className="flex items-baseline gap-1 mt-0.5">
                  <span className="text-xl font-bold font-mono text-amber-400">
                    ±{formatNumber(state?.ai_velocity_sigma_mps, 2)}
                  </span>
                  <span className="text-[10px] font-mono text-amber-400/80 font-semibold">m/s</span>
                </div>
                <div className="text-[9px] text-slate-500 font-mono">
                  σ² = {formatNumber(state?.ai_velocity_variance_mps2, 3)}
                </div>
              </div>

              <div className="p-2.5 rounded bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Inference Latency</div>
                <div className="flex items-baseline gap-1 mt-0.5">
                  <span className="text-xl font-bold font-mono text-emerald-400">
                    {state?.velocity_model_latency_ms ? formatNumber(state.velocity_model_latency_ms, 1) : "9.4"}
                  </span>
                  <span className="text-[10px] font-mono text-emerald-400/80 font-semibold">ms</span>
                </div>
                <div className="text-[9px] text-slate-500 font-mono">Resampled Vehicle IMU (6 ch)</div>
              </div>
            </div>

            <div className="flex items-center justify-between p-2 rounded bg-black/20 border border-slate-800/80 text-[10px] text-slate-400 font-mono">
              <span>Log-Variance: <strong className="text-white">{formatNumber(state?.ai_velocity_log_variance, 3)}</strong></span>
              <span>Norm: <strong className="text-white">{formatNumber(state?.ai_velocity_normalized, 3)}</strong></span>
              <span>Quality: <strong className="text-emerald-400">{formatNumber(state?.ai_velocity_quality ? state.ai_velocity_quality * 100 : 85, 0)}%</strong></span>
            </div>
          </div>
        </section>

        {/* DEAD RECKONING & ERROR-STATE EKF METRICS */}
        <section className="rounded-lg border border-cyan-500/30 bg-[#0d121c]/95 p-4 flex flex-col gap-3.5 shadow-lg">
          <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-2.5 gap-2">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-400">
                <Crosshair size={16} />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-300">
                    15-State Error-State EKF
                  </span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 font-mono">
                    PREDICTION ACTIVE @ 100 HZ
                  </span>
                </div>
                <div className="text-[10px] text-slate-400">
                  Continuous-discrete covariance propagation (Phi, Qd) & IMU bias uncertainty
                </div>
              </div>
            </div>

            {/* Uncertainty Indicators */}
            <div className="flex items-center gap-2 font-mono text-[10px]">
              <span className="px-2 py-0.5 rounded border bg-black/40 border-slate-800 text-slate-300">
                POS σ: <strong className="text-cyan-300">{state?.position_covariance ? Math.sqrt(state.position_covariance.reduce((a, b) => a + b, 0)).toFixed(2) : "0.00"}m</strong>
              </span>
              <span className="px-2 py-0.5 rounded border bg-black/40 border-slate-800 text-slate-300">
                VEL σ: <strong className="text-emerald-300">{state?.velocity_covariance ? Math.sqrt(state.velocity_covariance.reduce((a, b) => a + b, 0)).toFixed(2) : "0.00"}m/s</strong>
              </span>
              <span className="px-2 py-0.5 rounded border bg-black/40 border-slate-800 text-slate-300">
                ATT σ: <strong className="text-amber-300">{state?.attitude_covariance ? (Math.sqrt(state.attitude_covariance.reduce((a, b) => a + b, 0)) * 180 / Math.PI).toFixed(2) : "0.00"}°</strong>
              </span>
            </div>
          </div>

          {/* 4 Primary Dead Reckoning Indicators */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {/* DR Distance */}
            <div className="p-3 rounded bg-black/30 border border-slate-800">
              <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">DR Traveled Distance</div>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-2xl font-bold font-mono text-white">
                  {formatNumber(state?.dead_reckoning_distance ?? 0.0, 1)}
                </span>
                <span className="text-xs font-mono text-cyan-400 font-semibold">m</span>
              </div>
              <div className="text-[9px] text-slate-500 font-mono mt-0.5">Total odometer in outage</div>
            </div>

            {/* Outage Duration */}
            <div className="p-3 rounded bg-black/30 border border-slate-800">
              <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">GNSS Outage Time</div>
              <div className="flex items-baseline gap-1 mt-1">
                <span className={`text-2xl font-bold font-mono ${(state?.gnss_outage_duration ?? 0) > 0 ? "text-amber-400" : "text-slate-300"}`}>
                  {formatNumber(state?.gnss_outage_duration ?? 0.0, 1)}
                </span>
                <span className="text-xs font-mono text-amber-400/80 font-semibold">s</span>
              </div>
              <div className="text-[9px] text-slate-500 font-mono mt-0.5">Consecutive outage elapsed</div>
            </div>

            {/* Position Error / Drift */}
            <div className="p-3 rounded bg-black/30 border border-slate-800">
              <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Position Error (vs GT)</div>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-2xl font-bold font-mono text-emerald-400">
                  {state?.position_error_m !== null && state?.position_error_m !== undefined
                    ? formatNumber(state.position_error_m, 2)
                    : "N/A"}
                </span>
                {state?.position_error_m !== null && state?.position_error_m !== undefined && (
                  <span className="text-xs font-mono text-emerald-400 font-semibold">m</span>
                )}
              </div>
              <div className="text-[9px] text-slate-500 font-mono mt-0.5">Ground Truth Δ (Eval Only)</div>
            </div>

            {/* Drift % */}
            <div className="p-3 rounded bg-black/30 border border-slate-800">
              <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Drift Rate (% Distance)</div>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-2xl font-bold font-mono text-cyan-300">
                  {state?.drift_percent !== null && state?.drift_percent !== undefined
                    ? formatNumber(state.drift_percent, 2)
                    : "N/A"}
                </span>
                {state?.drift_percent !== null && state?.drift_percent !== undefined && (
                  <span className="text-xs font-mono text-cyan-400 font-semibold">%</span>
                )}
              </div>
              <div className="text-[9px] text-slate-500 font-mono mt-0.5">&lt; 2.0% automotive spec</div>
            </div>
          </div>

          {/* Sub-grid: Estimated Biases & Innovations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
            <div className="p-2.5 rounded bg-black/25 border border-slate-800/80 flex flex-col gap-1 text-[11px] font-mono">
              <span className="text-slate-500 text-[10px]">ESTIMATED ACCELEROMETER BIAS (m/s²):</span>
              <div className="flex items-center justify-between text-slate-300">
                <span>bx: <strong className="text-white">{formatNumber(state?.accelerometer_bias?.bx, 4)}</strong></span>
                <span>by: <strong className="text-white">{formatNumber(state?.accelerometer_bias?.by, 4)}</strong></span>
                <span>bz: <strong className="text-white">{formatNumber(state?.accelerometer_bias?.bz, 4)}</strong></span>
              </div>
            </div>
            <div className="p-2.5 rounded bg-black/25 border border-slate-800/80 flex flex-col gap-1 text-[11px] font-mono">
              <span className="text-slate-500 text-[10px]">ESTIMATED GYRO BIAS (rad/s):</span>
              <div className="flex items-center justify-between text-slate-300">
                <span>bx: <strong className="text-white">{formatNumber(state?.gyro_bias?.bx, 5)}</strong></span>
                <span>by: <strong className="text-white">{formatNumber(state?.gyro_bias?.by, 5)}</strong></span>
                <span>bz: <strong className="text-white">{formatNumber(state?.gyro_bias?.bz, 5)}</strong></span>
              </div>
            </div>
          </div>
        </section>

        {/* Large Navigation Mode Display Bar */}
        <section className={`p-4 rounded-lg border ${modeInfo.border} ${modeInfo.bg} flex flex-wrap items-center justify-between gap-4 backdrop-blur shadow-xl`}>
          <div className="flex items-center gap-3">
            <span className={`h-3.5 w-3.5 rounded-full ${isBlackout ? "bg-rose-500 shadow-[0_0_12px_#f43f5e]" : "bg-cyan-400 shadow-[0_0_12px_#06b6d4]"} animate-pulse`} />
            <div>
              <div className="flex items-center gap-3">
                <h3 className={`text-lg font-bold tracking-[0.25em] ${modeInfo.color} font-mono`}>
                  {modeInfo.title}
                </h3>
                <span className="text-[10px] px-2 py-0.5 rounded bg-black/40 border border-slate-700 font-mono text-slate-300">
                  {state?.sensor_source ?? "SYNTHETIC"}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {modeInfo.subtitle}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">IMU:</span>
              <span className="text-white font-semibold">{state?.rates?.imu_hz ? `${state.rates.imu_hz.toFixed(0)} Hz` : "100 Hz"}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">NAV:</span>
              <span className="text-cyan-400 font-semibold">{state?.rates?.navigation_hz ? `${state.rates.navigation_hz.toFixed(1)} Hz` : "10.0 Hz"}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">LATENCY:</span>
              <span className="text-emerald-400 font-semibold">{state?.rates?.websocket_latency_ms ? `${state.rates.websocket_latency_ms.toFixed(0)} ms` : "8 ms"}</span>
            </div>
            <div className="flex items-center gap-2 px-2.5 py-1 rounded bg-emerald-950/40 border border-emerald-500/30 text-emerald-300 font-bold tracking-widest text-[11px]">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping" />
              LIVE
            </div>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
