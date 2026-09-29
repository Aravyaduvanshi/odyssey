"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { edgeWsUrl } from "@/lib/edge";
import type { NavigationState, OdysseyEvent } from "@/types/navigation";

export function useNavigationSocket() {
  const [state, setState] = useState<NavigationState | null>(null);
  const [events, setEvents] = useState<OdysseyEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [samples, setSamples] = useState<NavigationState[]>([]);
  const reconnectTimer = useRef<number | null>(null);

  useEffect(() => {
    let navigationSocket: WebSocket | null = null;
    let eventSocket: WebSocket | null = null;
    let disposed = false;

    const connect = () => {
      navigationSocket = new WebSocket(edgeWsUrl("/ws/navigation"));
      eventSocket = new WebSocket(edgeWsUrl("/ws/events"));

      navigationSocket.onopen = () => setConnected(true);
      navigationSocket.onclose = () => {
        setConnected(false);
        if (!disposed) {
          reconnectTimer.current = window.setTimeout(connect, 1500);
        }
      };
      navigationSocket.onerror = () => setConnected(false);
      navigationSocket.onmessage = (message) => {
        const next = JSON.parse(message.data) as NavigationState;
        setState(next);
        setSamples((current) => [...current.slice(-179), next]);
      };

      eventSocket.onmessage = (message) => {
        const event = JSON.parse(message.data) as OdysseyEvent;
        setEvents((current) => [event, ...current].slice(0, 40));
      };
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer.current) {
        window.clearTimeout(reconnectTimer.current);
      }
      navigationSocket?.close();
      eventSocket?.close();
    };
  }, []);

  return useMemo(() => ({ state, samples, events, connected }), [state, samples, events, connected]);
}
