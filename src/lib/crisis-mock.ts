import type {
  AlertItem,
  AlertType,
  LogItem,
  Metrics,
  Severity,
  StatusPayload,
  SystemStatus,
} from './crisis-types';

const ALERT_TYPES: AlertType[] = ['FIRE', 'FALL', 'INTRUSION', 'SMOKE', 'WEAPON', 'CROWD'];
const SEVERITIES: Severity[] = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
const CAMERAS = ['CAM-01', 'CAM-02', 'CAM-03', 'CAM-04', 'CAM-07', 'CAM-12'];
const LOG_LEVELS: LogItem['level'][] = ['INFO', 'INFO', 'AGENT', 'AGENT', 'WARN'];
const LOCATIONS = ['Main Entrance', 'North Hall', 'Emergency Stairwell', 'Loading Dock', 'Lobby'];
const VIDEO_LABELS = ['fire1.mp4', 'fire2.mp4', 'fall1.mp4', 'Live Camera Feed'];

const AGENT_MESSAGES = [
  'Analyzing motion vectors across CAM-02...',
  'Vision model: object detection pass complete',
  'Confidence delta: +2.4% within expected band',
  'Cross-referencing thermal signature with baseline',
  'Heuristic check: no anomaly detected',
  'Reasoning: scene state remains stable',
  'Tracking 4 entities across viewport',
  'Frame buffer flushed at 30 FPS sustained',
  'Agent: monitoring perimeter sensors',
  'LLM context window updated',
  'No correlated alerts in last 30s window',
  'Audio classifier: ambient profile nominal',
];

const decisions = ['ALERT_AMBULANCE', 'ALERT_FIRE_ENGINE', 'IGNORE'];

let alertCounter = 0;
let logCounter = 0;
let tick = 0;
let logs: LogItem[] = [];
let alerts: AlertItem[] = [];
let lastDecision: string | null = null;

function pick<T>(arr: readonly T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function nowIso() {
  return new Date().toISOString();
}

function buildMetrics(status: SystemStatus): Metrics {
  const baseConfidence = status === 'SAFE' ? 95 : status === 'MONITOR' ? 82 : 68;
  const confidence = Math.max(40, Math.min(99, baseConfidence + (Math.random() * 6 - 3)));

  return {
    confidence,
    fps: 28 + Math.random() * 4,
    latencyMs: 40 + Math.random() * 20,
    modelsActive: 4,
    uptime: '12d 04:22:18',
  };
}

function buildSystemHealth() {
  return {
    model_status: 'ACTIVE',
    camera_status: 'CONNECTED',
    api_status: 'ONLINE',
    latency: `${Math.round(40 + Math.random() * 12)}ms`,
  };
}

export function createMockStatus(): StatusPayload {
  tick += 1;

  if (tick % 3 === 0) {
    logs = [
      {
        id: `log-${logCounter++}`,
        timestamp: nowIso(),
        level: pick(LOG_LEVELS),
        message: pick(AGENT_MESSAGES),
      },
      ...logs,
    ].slice(0, 80);
  }

  if (Math.random() < 0.04 && alerts.length < 8) {
    alerts = [
      {
        id: `alert-${alertCounter++}`,
        type: pick(ALERT_TYPES),
        severity: pick(SEVERITIES),
        cameraId: pick(CAMERAS),
        timestamp: nowIso(),
        message: 'Event escalated from live vision pipeline',
      },
      ...alerts,
    ].slice(0, 12);
  }

  if (Math.random() < 0.02 && alerts.length > 0) {
    alerts = alerts.slice(0, -1);
  }

  const hasCritical = alerts.some((alert) => alert.severity === 'CRITICAL' || alert.severity === 'HIGH');
  const status: SystemStatus = hasCritical ? 'ALERT' : alerts.length > 0 ? 'MONITOR' : 'SAFE';

  if (!lastDecision || Math.random() < 0.18) {
    lastDecision = pick(decisions);
  }

  return {
    frame: null,
    status,
    alerts,
    logs,
    metrics: buildMetrics(status),
    videos: VIDEO_LABELS,
    activeVideo: VIDEO_LABELS[0],
    location: pick(LOCATIONS),
    currentVideo: pick(VIDEO_LABELS),
    lifecycleState: status === 'ALERT' ? 'DISPATCHED' : status === 'MONITOR' ? 'DETECTED' : 'MONITORING',
    confidenceExplanation: [
      'Multi-model ensemble agrees on the scene classification',
      'Temporal stability remains within tolerance',
      'No secondary escalation from social confirmation layer',
    ],
    decisionReason:
      status === 'ALERT'
        ? 'Confirmed abnormal activity from correlated signals'
        : 'Deterministic fallback while the live backend is unavailable',
    signals: alerts.slice(0, 3).map((alert) => `${alert.type} @ ${alert.cameraId}`),
    llmSummary: status === 'SAFE' ? 'No active threat confirmed by the reasoning layer.' : 'Operator review recommended for the current incident stream.',
    llmLink: 'https://github.com/Mohiee661/ai-crisis-response',
    llmConfirmation: status === 'SAFE' ? 'NO CORRELATION' : 'REVIEW REQUIRED',
    systemHealth: buildSystemHealth(),
    decision: lastDecision,
    incidentLocked: status !== 'SAFE',
  };
}
