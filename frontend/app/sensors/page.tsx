"use client";

import { useMemo } from "react";
import { useNavigationSocket } from "@/hooks/useNavigationSocket";
import { AppShell } from "@/components/AppShell";
import { Activity, Cpu, Gauge, Radio, Waves, Zap } from "lucide-react";

function format(val: number | null | undefined, digits = 3): string {
  if (val === null || val === undefined || isNaN(val)) return "N/A";
  return val.toFixed(digits);
}

// Multi-line strip chart component for 3 axes (X, Y, Z)
function MultiAxisChart({
  title,
  unit,
  data,
  labels = ["X", "Y", "Z"],
  colors = ["#38bdf8", "#34d399", "#f59e0b"],
}: {
  title: string;
  unit: string;
  data: Array<{ x: number; y: number; z: number }>;
  labels?: [string, string, string];
  colors?: [string, string, string];
}) {
  const width = 600;
  const height = 180;
  const padding = 25;

  const { pathX, pathY, pathZ, minY, maxY } = useMemo(() => {
    if (data.length < 2) {
      return { pathX: "", pathY: "", pathZ: "", minY: -1, maxY: 1 };
    }
    const allVals = data.flatMap((d) => [d.x, d.y, d.z]);
    let min = Math.min(...allVals);
    let max = Math.max(...allVals);
    if (max - min < 0.1) {
      min -= 0.5;
      max += 0.5;
    }
    const range = max - min;

    const toY = (v: number) => (height - padding) - ((v - min) / range) * (height - 2 * padding);
    const toX = (i: number) => padding + (i / (data.length - 1)) * (width - 2 * padding);

    const px = data.map((d, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(1)} ${toY(d.x).toFixed(1)}`).join(" ");
    const py = data.map((d, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(1)} ${toY(d.y).toFixed(1)}`).join(" ");
    const pz = data.map((d, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(1)} ${toY(d.z).toFixed(1)}`).join(" ");

    return { pathX: px, pathY: py, pathZ: pz, minY: min, maxY: max };
  }, [data]);

  const latest = data[data.length - 1] ?? { x: 0, y: 0, z: 0 };

  return (
    <div className="rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-400 font-mono">{title}</h3>
          <span className="text-[10px] text-slate-500 font-mono">[{unit}]</span>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <span className="flex items-center gap-1 text-sky-400">
            <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />
            {labels[0]}: {format(latest.x, 2)}
          </span>
          <span className="flex items-center gap-1 text-emerald-400">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            {labels[1]}: {format(latest.y, 2)}
          </span>
          <span className="flex items-center gap-1 text-amber-400">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
            {labels[2]}: {format(latest.z, 2)}
          </span>
        </div>
      </div>

      {/* SVG Plot */}
      <svg className="w-full h-36 mt-1" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        {/* Horizontal grid lines */}
        <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="#1e293b" strokeDasharray="3 3" />
        <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="#1e293b" strokeWidth="0.5" />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#1e293b" strokeWidth="0.5" />

        {/* Traces */}
        {pathX && <path d={pathX} fill="none" stroke={colors[0]} strokeWidth="2" strokeLinecap="round" opacity="0.85" />}
        {pathY && <path d={pathY} fill="none" stroke={colors[1]} strokeWidth="2" strokeLinecap="round" opacity="0.85" />}
        {pathZ && <path d={pathZ} fill="none" stroke={colors[2]} strokeWidth="2" strokeLinecap="round" opacity="0.85" />}
      </svg>

      <div className="flex justify-between text-[9px] font-mono text-slate-500 px-1">
        <span>MIN: {format(minY, 2)}</span>
        <span>WINDOW: 60 SAMPLES</span>
        <span>MAX: {format(maxY, 2)}</span>
      </div>
    </div>
  );
}

export default function SensorsPage() {
  const { state, samples, connected } = useNavigationSocket();

  // Extract recent 60 samples for real-time charts
  const history = useMemo(() => samples.slice(-60), [samples]);

  const rawAccel = useMemo(
    () =>
      history.map((s) => ({
        x: s.raw_imu?.ax ?? 0,
        y: s.raw_imu?.ay ?? 0,
        z: s.raw_imu?.az ?? 0,
      })),
    [history]
  );

  const rawGyro = useMemo(
    () =>
      history.map((s) => ({
        x: s.raw_imu?.gx ?? 0,
        y: s.raw_imu?.gy ?? 0,
        z: s.raw_imu?.gz ?? 0,
      })),
    [history]
  );

  const vehicleAccel = useMemo(
    () =>
      history.map((s) => ({
        x: s.vehicle_frame_imu?.ACC_VEH_X ?? s.vehicle_frame_imu?.ax ?? 0,
        y: s.vehicle_frame_imu?.ACC_VEH_Y ?? s.vehicle_frame_imu?.ay ?? 0,
        z: s.vehicle_frame_imu?.ACC_VEH_Z ?? s.vehicle_frame_imu?.az ?? 0,
      })),
    [history]
  );

  const vehicleGyro = useMemo(
    () =>
      history.map((s) => ({
        x: s.vehicle_frame_imu?.GYRO_Roll_corrected ?? s.vehicle_frame_imu?.gx ?? 0,
        y: s.vehicle_frame_imu?.GYRO_Pitch_corrected ?? s.vehicle_frame_imu?.gy ?? 0,
        z: s.vehicle_frame_imu?.GYRO_Yaw_corrected ?? s.vehicle_frame_imu?.gz ?? 0,
      })),
    [history]
  );

  const rawMag = useMemo(
    () =>
      history.map((s) => ({
        x: s.raw_imu?.mx ?? 0,
        y: s.raw_imu?.my ?? 0,
        z: s.raw_imu?.mz ?? 0,
      })),
    [history]
  );

  return (
    <AppShell edgeOnline={connected} rateHz={state?.rates?.navigation_hz} mode={state?.navigation_mode}>
      <div className="flex flex-col gap-4">
        {/* Page Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-cyan-500/20 pb-3">
          <div>
            <h1 className="text-xl font-bold tracking-[0.25em] text-white flex items-center gap-2">
              <Cpu className="text-cyan-400" size={20} />
              SENSOR DIAGNOSTICS & TELEMETRY
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              REAL-TIME HIGH-FREQUENCY INERTIAL & MAGNETIC CHANNEL ANALYSIS
            </p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-[#0d121c] border border-slate-800 text-xs font-mono">
            <span className="text-slate-500">SOURCE:</span>
            <span className="text-cyan-300 font-bold">{state?.sensor_source ?? "SYNTHETIC"}</span>
          </div>
        </div>

        {/* Section 15: Engineering Status Panel (ONLY measured values; N/A if unavailable) */}
        <section className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
          <div className="p-3 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90">
            <div className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">IMU RATE</div>
            <div className="mt-1 text-2xl font-bold font-mono text-cyan-300">
              {state?.rates?.imu_hz !== null && state?.rates?.imu_hz !== undefined
                ? `${state.rates.imu_hz.toFixed(0)} Hz`
                : "N/A"}
            </div>
            <div className="text-[10px] text-slate-500 font-mono">Nominal: 100 Hz</div>
          </div>

          <div className="p-3 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90">
            <div className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">GNSS RATE</div>
            <div className="mt-1 text-2xl font-bold font-mono text-amber-300">
              {state?.rates?.gnss_hz !== null && state?.rates?.gnss_hz !== undefined
                ? `${state.rates.gnss_hz.toFixed(1)} Hz`
                : "N/A"}
            </div>
            <div className="text-[10px] text-slate-500 font-mono">Epoch Update</div>
          </div>

          <div className="p-3 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90">
            <div className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">FUSION RATE</div>
            <div className="mt-1 text-2xl font-bold font-mono text-emerald-400">
              {state?.rates?.navigation_hz !== null && state?.rates?.navigation_hz !== undefined
                ? `${state.rates.navigation_hz.toFixed(1)} Hz`
                : "N/A"}
            </div>
            <div className="text-[10px] text-slate-500 font-mono">Navigation Engine</div>
          </div>

          <div className="p-3 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90">
            <div className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">WS LATENCY</div>
            <div className="mt-1 text-2xl font-bold font-mono text-sky-400">
              {state?.rates?.websocket_latency_ms !== null && state?.rates?.websocket_latency_ms !== undefined
                ? `${state.rates.websocket_latency_ms.toFixed(0)} ms`
                : "N/A"}
            </div>
            <div className="text-[10px] text-slate-500 font-mono">Client-Server Ping</div>
          </div>

          <div className="p-3 rounded-lg border border-cyan-500/20 bg-[#0d121c]/90">
            <div className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">EDGE LATENCY</div>
            <div className="mt-1 text-2xl font-bold font-mono text-purple-400">
              {state?.rates?.edge_latency_ms !== null && state?.rates?.edge_latency_ms !== undefined
                ? `${state.rates.edge_latency_ms.toFixed(1)} ms`
                : "N/A"}
            </div>
            <div className="text-[10px] text-slate-500 font-mono">Internal Process</div>
          </div>
        </section>

        {/* Real-time Charts Grid */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Raw Acceleration Plot */}
          <MultiAxisChart
            title="Raw Phone Acceleration"
            unit="m/s²"
            data={rawAccel}
            labels={["ax", "ay", "az"]}
            colors={["#38bdf8", "#34d399", "#f59e0b"]}
          />

          {/* Raw Angular Velocity Plot */}
          <MultiAxisChart
            title="Raw Phone Gyroscope"
            unit="rad/s"
            data={rawGyro}
            labels={["gx", "gy", "gz"]}
            colors={["#06b6d4", "#a855f7", "#ec4899"]}
          />

          {/* Vehicle-Frame Acceleration Plot */}
          <MultiAxisChart
            title="Vehicle Acceleration (ACC_VEH)"
            unit="m/s²"
            data={vehicleAccel}
            labels={["X (Fwd)", "Y (Lat)", "Z (Up)"]}
            colors={["#10b981", "#3b82f6", "#f97316"]}
          />

          {/* Vehicle-Frame Gyroscope Plot */}
          <MultiAxisChart
            title="Vehicle Gyro (GYRO_corrected)"
            unit="rad/s"
            data={vehicleGyro}
            labels={["Roll Rate", "Pitch Rate", "Yaw Rate"]}
            colors={["#ec4899", "#a855f7", "#06b6d4"]}
          />

          {/* Magnetometer Plot */}
          <div className="md:col-span-2 lg:col-span-2">
            <MultiAxisChart
              title="Magnetometer Flux"
              unit="normalized"
              data={rawMag}
              labels={["mx", "my", "mz"]}
              colors={["#facc15", "#4ade80", "#60a5fa"]}
            />
          </div>
        </section>

        {/* Channel Readout Table */}
        <section className="rounded-lg border border-cyan-500/20 bg-[#0d121c]/90 p-4">
          <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-400 font-mono mb-3">
            Instantaneous Channel Matrix & Feature Mapping
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-slate-500 border-b border-slate-800 text-left">
                  <th className="pb-2">FRAME / CHANNEL</th>
                  <th className="pb-2">X AXIS</th>
                  <th className="pb-2">Y AXIS</th>
                  <th className="pb-2">Z AXIS</th>
                  <th className="pb-2">AI MODEL CONTRACT / STATUS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                <tr>
                  <td className="py-2.5 font-bold text-sky-400">Raw Phone Accel (ax, ay, az) [m/s²]</td>
                  <td className="py-2.5 text-white">{format(state?.raw_imu?.ax)}</td>
                  <td className="py-2.5 text-white">{format(state?.raw_imu?.ay)}</td>
                  <td className="py-2.5 text-white">{format(state?.raw_imu?.az)}</td>
                  <td className="py-2.5 text-slate-400">MountNet V2 Ch 0-2 (Resampled 10Hz)</td>
                </tr>
                <tr>
                  <td className="py-2.5 font-bold text-sky-300">Raw Phone Gyro (gx, gy, gz) [rad/s]</td>
                  <td className="py-2.5 text-white">{format(state?.raw_imu?.gx)}</td>
                  <td className="py-2.5 text-white">{format(state?.raw_imu?.gy)}</td>
                  <td className="py-2.5 text-white">{format(state?.raw_imu?.gz)}</td>
                  <td className="py-2.5 text-slate-400">MountNet V2 Ch 3-5 (Resampled 10Hz)</td>
                </tr>
                <tr>
                  <td className="py-2.5 font-bold text-emerald-400">Vehicle Accel (ACC_VEH_X, Y, Z) [m/s²]</td>
                  <td className="py-2.5 text-white font-bold">{format(state?.vehicle_frame_imu?.ACC_VEH_X ?? state?.vehicle_frame_imu?.ax)}</td>
                  <td className="py-2.5 text-white font-bold">{format(state?.vehicle_frame_imu?.ACC_VEH_Y ?? state?.vehicle_frame_imu?.ay)}</td>
                  <td className="py-2.5 text-white font-bold">{format(state?.vehicle_frame_imu?.ACC_VEH_Z ?? state?.vehicle_frame_imu?.az)}</td>
                  <td className="py-2.5 text-emerald-300">Velocity TCN Ch 0-2 [X=Fwd, Y=Lat, Z=Up]</td>
                </tr>
                <tr>
                  <td className="py-2.5 font-bold text-purple-400">Vehicle Gyro Corrected [rad/s]</td>
                  <td className="py-2.5 text-white font-bold">{format(state?.vehicle_frame_imu?.GYRO_Roll_corrected ?? state?.vehicle_frame_imu?.gx)}</td>
                  <td className="py-2.5 text-white font-bold">{format(state?.vehicle_frame_imu?.GYRO_Pitch_corrected ?? state?.vehicle_frame_imu?.gy)}</td>
                  <td className="py-2.5 text-white font-bold">{format(state?.vehicle_frame_imu?.GYRO_Yaw_corrected ?? state?.vehicle_frame_imu?.gz)}</td>
                  <td className="py-2.5 text-purple-300">Velocity TCN Ch 3-5 [Yaw, Pitch, Roll]</td>
                </tr>
                <tr>
                  <td className="py-2.5 font-bold text-amber-400">Magnetometer Auxiliary (normalized)</td>
                  <td className="py-2.5 text-white">{format(state?.raw_imu?.mx)}</td>
                  <td className="py-2.5 text-white">{format(state?.raw_imu?.my)}</td>
                  <td className="py-2.5 text-white">{format(state?.raw_imu?.mz)}</td>
                  <td className="py-2.5 text-amber-300">
                    {state?.calibration?.alignment?.magnetometer_consistent ? "Consistent" : "Distorted / Auxiliary"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* AI MODEL PIPELINE ARCHITECTURE CARD */}
        <section className="p-4 rounded-lg border border-cyan-500/30 bg-[#0d121c]/95 flex flex-col gap-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <span className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-400 font-mono">
              AI Pipeline Execution Pipeline
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 font-mono">
              100-Sample Buffer @ 10 Hz Resampling
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
            <div className="p-3 rounded bg-black/30 border border-slate-800 flex flex-col gap-1.5">
              <div className="text-cyan-400 font-bold uppercase text-[11px]">1. Phone Frame IMU</div>
              <div className="text-[10px] text-slate-400">
                100 Hz Raw IMU synchronized & resampled to 10 Hz into rolling buffer.
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                Input: <code className="text-slate-300">[ax, ay, az, gx, gy, gz]</code>
              </div>
            </div>

            <div className="p-3 rounded bg-black/30 border border-slate-800 flex flex-col gap-1.5">
              <div className="text-cyan-400 font-bold uppercase text-[11px]">2. MountNet V2 Alignment</div>
              <div className="text-[10px] text-slate-400">
                Predicts mounting yaw between phone & vehicle chassis (every 50 samples).
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                Output: <code className="text-emerald-300">θ_mount = {state?.mounting_yaw_deg ? `${state.mounting_yaw_deg.toFixed(1)}°` : "0.0°"}</code>
              </div>
            </div>

            <div className="p-3 rounded bg-black/30 border border-slate-800 flex flex-col gap-1.5">
              <div className="text-cyan-400 font-bold uppercase text-[11px]">3. Velocity TCN Inference</div>
              <div className="text-[10px] text-slate-400">
                Infers vehicle forward speed and uncertainty from vehicle-aligned frame (every 10 samples).
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                Output: <code className="text-amber-300">{state?.ai_velocity_mps ? `${state.ai_velocity_mps.toFixed(2)} m/s` : "N/A"}</code>
              </div>
            </div>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
