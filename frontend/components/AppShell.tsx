"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Compass, Cpu, Gauge, Play, RotateCcw, Satellite, Sliders, Square } from "lucide-react";
import { postCommand } from "@/lib/edge";

interface AppShellProps {
  children: React.ReactNode;
  edgeOnline: boolean;
  rateHz?: number | null;
  mode?: string;
  isBlackout?: boolean;
}

export function AppShell({ children, edgeOnline, rateHz, mode, isBlackout }: AppShellProps) {
  const pathname = usePathname();

  const navItems = [
    { label: "DASHBOARD", href: "/", icon: Gauge },
    { label: "SENSORS", href: "/sensors", icon: Cpu },
    { label: "CALIBRATION", href: "/calibration", icon: Compass },
  ];

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 flex flex-col font-mono selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Automotive Mission-Control Top Bar */}
      <header className="sticky top-0 z-50 border-b border-cyan-500/20 bg-[#0a0e17]/95 backdrop-blur-md px-4 py-2.5">
        <div className="max-w-[1720px] mx-auto flex flex-wrap items-center justify-between gap-4">
          
          {/* Brand & System Mode */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2.5">
              <div className="h-8 w-8 rounded border border-cyan-400/40 bg-cyan-950/40 flex items-center justify-center text-cyan-400 font-bold tracking-widest text-sm shadow-[0_0_12px_rgba(6,182,212,0.3)]">
                Ω1
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold tracking-[0.25em] text-white text-base">ODYSSEY<span className="text-cyan-400">V1</span></span>
                  <span className="text-[10px] uppercase px-1.5 py-0.5 rounded border border-cyan-500/30 bg-cyan-950/30 text-cyan-300 tracking-wider">
                    {mode ?? "SIMULATION"}
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 tracking-wider">EDGE NAVIGATION RESEARCH ENGINE</div>
              </div>
            </div>

            {/* Navigation Tabs */}
            <nav className="hidden md:flex items-center gap-1 ml-4 pl-4 border-l border-slate-800">
              {navItems.map((item) => {
                const active = pathname === item.href;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium tracking-wider transition-all duration-150 ${
                      active
                        ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                        : "text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent"
                    }`}
                  >
                    <Icon size={14} />
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Quick Controls & Live Telemetry Indicators */}
          <div className="flex items-center gap-3 text-xs">
            {/* Edge Engine Live Status */}
            <div className="flex items-center gap-2 px-3 py-1 rounded border border-slate-800 bg-[#0d121c]">
              <span
                className={`h-2 w-2 rounded-full ${
                  edgeOnline
                    ? "bg-emerald-400 shadow-[0_0_10px_#10b981] animate-pulse"
                    : "bg-rose-500 shadow-[0_0_10px_#f43f5e]"
                }`}
              />
              <span className={`text-[11px] font-semibold tracking-wider ${edgeOnline ? "text-emerald-400" : "text-rose-400"}`}>
                {edgeOnline ? "EDGE ONLINE" : "EDGE OFFLINE"}
              </span>
            </div>

            {/* Publish Rate */}
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded border border-slate-800 bg-[#0d121c] text-slate-300 text-[11px]">
              <span className="text-slate-500">PUB:</span>
              <span className="text-cyan-400 font-bold">{rateHz ? `${rateHz.toFixed(1)} Hz` : "10 Hz"}</span>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-1.5 border-l border-slate-800 pl-3">
              <button
                onClick={() => postCommand("/session/start")}
                title="Start Navigation Session"
                className="flex items-center gap-1 px-2.5 py-1 rounded border border-emerald-500/30 bg-emerald-950/20 text-emerald-300 hover:bg-emerald-900/40 transition-colors text-[11px]"
              >
                <Play size={12} />
                <span className="hidden sm:inline">START</span>
              </button>

              <button
                onClick={() => postCommand("/session/stop")}
                title="Stop Navigation Session"
                className="flex items-center gap-1 px-2.5 py-1 rounded border border-slate-700 bg-slate-900/50 text-slate-300 hover:bg-slate-800 transition-colors text-[11px]"
              >
                <Square size={12} />
                <span className="hidden sm:inline">STOP</span>
              </button>

              {isBlackout ? (
                <button
                  onClick={() => postCommand("/gnss/restore")}
                  title="Restore GNSS Reception"
                  className="flex items-center gap-1 px-2.5 py-1 rounded border border-cyan-500/40 bg-cyan-950/30 text-cyan-300 hover:bg-cyan-900/40 transition-colors text-[11px]"
                >
                  <RotateCcw size={12} />
                  <span>RESTORE GNSS</span>
                </button>
              ) : (
                <button
                  onClick={() => postCommand("/gnss/blackout")}
                  title="Simulate GNSS Blackout"
                  className="flex items-center gap-1 px-2.5 py-1 rounded border border-amber-500/40 bg-amber-950/30 text-amber-300 hover:bg-amber-900/40 transition-colors text-[11px]"
                >
                  <Satellite size={12} />
                  <span>SIM BLACKOUT</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Viewport */}
      <main className="flex-1 max-w-[1720px] w-full mx-auto p-3 md:p-5 flex flex-col">
        {children}
      </main>
    </div>
  );
}
