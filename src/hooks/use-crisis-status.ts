import { useEffect, useRef, useState } from 'react';
import { createMockStatus } from '../lib/crisis-mock';
import type { StatusPayload } from '../lib/crisis-types';

const ENDPOINT = '/crisis/status';
const POLL_MS = 500;

export function useCrisisStatus() {
  const [data, setData] = useState<StatusPayload>(() => createMockStatus());
  const [connected, setConnected] = useState(false);
  const failures = useRef(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      try {
        const ctrl = new AbortController();
        const abortTimer = setTimeout(() => ctrl.abort(), 900);
        const res = await fetch(ENDPOINT, { signal: ctrl.signal });
        clearTimeout(abortTimer);
        if (!res.ok) throw new Error('bad status');

        const json = (await res.json()) as Partial<StatusPayload>;
        if (cancelled) return;

        failures.current = 0;
        setConnected(true);
        setData((prev) => ({
          ...prev,
          ...json,
          metrics: { ...prev.metrics, ...(json.metrics ?? {}) },
          videos: json.videos ?? prev.videos,
          activeVideo: json.activeVideo ?? prev.activeVideo,
          systemHealth: { ...prev.systemHealth, ...(json.systemHealth ?? {}) },
          alerts: json.alerts ?? prev.alerts,
          logs: json.logs ?? prev.logs,
          confidenceExplanation: json.confidenceExplanation ?? prev.confidenceExplanation,
          signals: json.signals ?? prev.signals,
        }));
      } catch {
        failures.current += 1;
        if (failures.current > 2) {
          setConnected(false);
          setData(createMockStatus());
        }
      } finally {
        if (!cancelled) timer = setTimeout(tick, POLL_MS);
      }
    };

    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  const override = async (action: string) => {
    setData((prev) => ({
      ...prev,
      decision: action,
      lifecycleState: action === 'IGNORE' ? 'MONITORING' : 'DISPATCHED',
      incidentLocked: action !== 'IGNORE',
      decisionReason: `Manual override by operator: ${action}`,
      logs: [
        {
          id: `local-${Date.now()}`,
          timestamp: new Date().toISOString(),
          level: 'INFO' as const,
          message: `[Manual] Operator overrode system with: ${action}`,
        },
        ...prev.logs,
      ].slice(0, 80),
    }));

    try {
      await fetch('/crisis/override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
    } catch {
      // Keep the local override visible even when the backend is offline.
    }
  };

  const selectVideo = async (video: string) => {
    try {
      const res = await fetch('/crisis/videos/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video }),
      });
      if (!res.ok) throw new Error('bad select');
      const json = (await res.json()) as Partial<StatusPayload>;
      setData((prev) => ({
        ...prev,
        ...json,
        metrics: { ...prev.metrics, ...(json.metrics ?? {}) },
        videos: json.videos ?? prev.videos,
        activeVideo: json.activeVideo ?? video,
        systemHealth: { ...prev.systemHealth, ...(json.systemHealth ?? {}) },
        alerts: json.alerts ?? prev.alerts,
        logs: json.logs ?? prev.logs,
        confidenceExplanation: json.confidenceExplanation ?? prev.confidenceExplanation,
        signals: json.signals ?? prev.signals,
      }));
      setConnected(true);
    } catch {
      // Local view stays on the current video when the backend is unavailable.
    }
  };

  return { data, connected, override, selectVideo };
}
