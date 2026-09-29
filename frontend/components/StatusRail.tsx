"use client";

import type { NavigationState } from "@/types/navigation";

type Status = "ACTIVE" | "NOT_LOADED" | "DEGRADED" | "ERROR";

function renderBadge(status: Status) {
  switch (status) {
    case "ACTIVE":
      return (
        <span className="flex items-center gap-1.5 text-emerald-400">
          <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981]" />
          <span className="text-[10px] font-bold tracking-wider">ACTIVE</span>
        </span>
      );
    case "DEGRADED":
      return (
        <span className="flex items-center gap-1.5 text-amber-400">
          <span className="text-xs">△</span>
          <span className="text-[10px] font-bold tracking-wider">DEGRADED</span>
        </span>
      );
    case "ERROR":
      return (
        <span className="flex items-center gap-1.5 text-rose-400">
          <span className="text-xs">✕</span>
          <span className="text-[10px] font-bold tracking-wider">ERROR</span>
        </span>
      );
    case "NOT_LOADED":
    default:
      return (
        <span className="flex items-center gap-1.5 text-slate-500">
          <span className="h-2 w-2 rounded-full border border-slate-500" />
          <span className="text-[10px] tracking-wider">NOT LOADED</span>
        </span>
      );
  }
}

export function StatusRail({ state }: { state: NavigationState | null }) {
  const gnssStatus: Status = state
    ? state.navigation_mode === "DEAD_RECKONING" || state.gnss_available === false
      ? "ERROR"
      : state.navigation_mode === "GNSS_RECOVERY"
      ? "DEGRADED"
      : state.gnss_health > 0.5 && state.latitude !== null
      ? "ACTIVE"
      : "DEGRADED"
    : "DEGRADED";

  const imuStatus: Status = state ? "ACTIVE" : "DEGRADED";
  const aiSpeedStatus: Status = state?.velocity_model_status === "LOADED" || state?.model_status?.velocity === "LOADED" ? "ACTIVE" : "NOT_LOADED";
  const aiYawStatus: Status = state?.mountnet_status === "LOADED" || state?.model_status?.yaw === "LOADED" ? "ACTIVE" : "NOT_LOADED";
  const insStatus: Status = state?.ins_position ? "ACTIVE" : "NOT_LOADED";
  const ekfStatus: Status = state?.ekf_position ? "ACTIVE" : "NOT_LOADED";
  const nhcStatus: Status = state?.nhc_active ? "ACTIVE" : "NOT_LOADED";
  const zuptStatus: Status = state?.zupt_active ? "ACTIVE" : "NOT_LOADED";
  const mapStatus: Status = "NOT_LOADED";

  const gnssDetail = state?.navigation_mode === "DEAD_RECKONING" || state?.gnss_available === false
    ? `DENIED (${(state?.gnss_outage_duration ?? 0).toFixed(1)}s)`
    : state?.navigation_mode === "GNSS_RECOVERY"
    ? `RECOVERING (${state?.gnss_recovery_measurements ?? 0}/5)`
    : state?.latitude !== null && state?.latitude !== undefined
    ? `CONNECTED (${state.latitude.toFixed(4)}°)`
    : "DENIED";

  const aiSpeedDetail = state?.ai_velocity_mps !== null && state?.ai_velocity_mps !== undefined
    ? `${state.ai_velocity_mps.toFixed(2)} m/s (±${state.ai_velocity_sigma_mps?.toFixed(2) ?? "0.00"})`
    : (state?.ai_velocity !== null && state?.ai_velocity !== undefined ? `${state.ai_velocity.toFixed(2)} m/s` : "Unloaded");

  const aiYawDetail = state?.mounting_yaw_deg !== null && state?.mounting_yaw_deg !== undefined
    ? `Mount: ${state.mounting_yaw_deg.toFixed(1)}°`
    : (state?.ai_yaw !== null && state?.ai_yaw !== undefined ? `Mount: ${(state.ai_yaw * 180 / Math.PI).toFixed(1)}°` : "Unloaded");

  const insDetail = state?.ins_position
    ? `E:${state.ins_position.E.toFixed(1)} N:${state.ins_position.N.toFixed(1)}m`
    : "Strapdown INS";

  const ekfDetail = state?.ekf_position
    ? `E:${state.ekf_position.E.toFixed(1)} N:${state.ekf_position.N.toFixed(1)}m`
    : "15-State ES-EKF";

  const items: Array<{ name: string; status: Status; detail: string; confidence?: number }> = [
    {
      name: "GNSS",
      status: gnssStatus,
      detail: gnssDetail,
      confidence: (state?.gnss_health ?? 0) * 100,
    },
    {
      name: "IMU",
      status: imuStatus,
      detail: state ? `${state.sensor_source} @ 100 Hz` : "Awaiting Data",
      confidence: state ? 100 : 0,
    },
    {
      name: "VELOCITY TCN",
      status: aiSpeedStatus,
      detail: aiSpeedDetail,
      confidence: state?.ai_velocity_quality !== undefined && state.ai_velocity_quality !== null ? state.ai_velocity_quality * 100 : (state?.ai_velocity_confidence ?? 0) * 100,
    },
    {
      name: "MOUNTNET V2",
      status: aiYawStatus,
      detail: aiYawDetail,
      confidence: state?.mounting_yaw_deg !== null && state?.mounting_yaw_deg !== undefined ? 95 : (state?.ai_yaw_confidence ?? 0) * 100,
    },
    {
      name: "INS",
      status: insStatus,
      detail: insDetail,
      confidence: state?.ins_position ? 98 : 0,
    },
    {
      name: "EKF",
      status: ekfStatus,
      detail: ekfDetail,
      confidence: state?.ekf_position ? 99 : 0,
    },
    {
      name: "NHC",
      status: nhcStatus,
      detail: state?.nhc_active ? "Non-Holonomic Active" : "Suppressed / Inactive",
    },
    {
      name: "ZUPT",
      status: zuptStatus,
      detail: state?.zupt_active ? "Zero Velocity Active" : "Moving / Inactive",
    },
    {
      name: "MAP",
      status: mapStatus,
      detail: state?.map_label ?? "LOCAL TEST MAP",
    },
  ];

  return (
    <section className="flex flex-col rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 backdrop-blur p-3.5 shadow-lg">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5 mb-3">
        <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-400">System Status</h2>
        <span className="text-[10px] text-slate-500 uppercase tracking-widest">SUBSYSTEM HEALTH</span>
      </div>

      <div className="flex flex-col gap-2">
        {items.map((item) => (
          <div
            key={item.name}
            className="flex items-center justify-between gap-3 p-2 rounded border border-slate-800/80 bg-black/20 hover:border-cyan-500/30 transition-all duration-150"
          >
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-white tracking-wide">{item.name}</span>
                {item.confidence !== undefined && item.confidence > 0 && (
                  <span className="text-[9px] px-1 py-0.2 rounded bg-cyan-950/60 text-cyan-300 font-mono">
                    {Math.round(item.confidence)}%
                  </span>
                )}
              </div>
              <div className="text-[10px] text-slate-400 font-mono truncate max-w-[140px]">{item.detail}</div>
            </div>
            <div>{renderBadge(item.status)}</div>
          </div>
        ))}
      </div>

      {state && state.gnss_position_innovation && (
        <div className="mt-3 pt-2.5 border-t border-slate-800 text-[10px] font-mono">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-slate-400 font-bold uppercase tracking-wider">GNSS Position Update</span>
            <span className={state.gnss_position_accepted ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>
              {state.gnss_position_accepted ? "ACCEPTED" : "REJECTED"}
            </span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Innovation (E/N/U):</span>
            <span className="text-slate-200">
              {state.gnss_position_innovation.map((v) => v.toFixed(2)).join(", ")} m
            </span>
          </div>
          {state.gnss_position_mahalanobis_distance !== null && state.gnss_position_mahalanobis_distance !== undefined && (
            <div className="flex justify-between text-slate-400 mt-0.5">
              <span>Mahalanobis d:</span>
              <span className="text-slate-200">{state.gnss_position_mahalanobis_distance.toFixed(2)}</span>
            </div>
          )}
        </div>
      )}

      {state && state.gnss_velocity_innovation && (
        <div className="mt-2 pt-2 border-t border-slate-800/80 text-[10px] font-mono">
          <div className="flex items-center justify-between mb-1">
            <span className="text-slate-400 font-bold uppercase tracking-wider">GNSS Velocity Update</span>
            <span className={state.gnss_velocity_accepted ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>
              {state.gnss_velocity_accepted ? "ACCEPTED" : "REJECTED"}
            </span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Innovation (E/N/U):</span>
            <span className="text-slate-200">
              {state.gnss_velocity_innovation.map((v) => v.toFixed(2)).join(", ")} m/s
            </span>
          </div>
          {state.gnss_velocity_mahalanobis_distance !== null && state.gnss_velocity_mahalanobis_distance !== undefined && (
            <div className="flex justify-between text-slate-400 mt-0.5">
              <span>Mahalanobis d:</span>
              <span className="text-slate-200">{state.gnss_velocity_mahalanobis_distance.toFixed(2)}</span>
            </div>
          )}
        </div>
      )}

      {state && (
        <div className="mt-2 pt-2 border-t border-slate-800/80 text-[10px] font-mono">
          <div className="flex items-center justify-between mb-1">
            <span className="text-slate-400 font-bold uppercase tracking-wider">Velocity TCN</span>
            <span className={
              state.velocity_model_status === "READY"
                ? "text-emerald-400 font-bold"
                : state.velocity_model_status === "WARMING_UP" || state.velocity_model_status === "LOADING"
                ? "text-amber-400 font-bold"
                : state.velocity_model_status === "ERROR"
                ? "text-rose-400 font-bold"
                : "text-slate-500 font-bold"
            }>
              {state.velocity_model_status === "READY" ? "READY" : state.velocity_model_status === "ERROR" ? "ERROR" : "WARMING UP"}
            </span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>AI velocity:</span>
            <span className="text-slate-200">
              {state.ai_velocity_mps !== null && state.ai_velocity_mps !== undefined
                ? `${state.ai_velocity_mps.toFixed(1)} m/s`
                : (state.ai_velocity !== null && state.ai_velocity !== undefined ? `${state.ai_velocity.toFixed(1)} m/s` : "--")}
            </span>
          </div>
          <div className="flex justify-between text-slate-400 mt-0.5">
            <span>AI uncertainty:</span>
            <span className="text-slate-200">
              {state.ai_velocity_sigma_mps !== null && state.ai_velocity_sigma_mps !== undefined
                ? `±${state.ai_velocity_sigma_mps.toFixed(1)} m/s`
                : "--"}
            </span>
          </div>
          <div className="flex justify-between text-slate-400 mt-0.5">
            <span>Innovation:</span>
            <span className="text-slate-200">
              {state.ai_velocity_innovation !== null && state.ai_velocity_innovation !== undefined
                ? `${state.ai_velocity_innovation.toFixed(1)} m/s`
                : "--"}
            </span>
          </div>
          <div className="flex justify-between text-slate-400 mt-0.5">
            <span>EKF update:</span>
            <span className={state.ai_velocity_accepted ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>
              {state.ai_velocity_accepted ? "ACCEPTED" : "REJECTED"}
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
