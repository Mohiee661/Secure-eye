export type SystemStatus = 'SAFE' | 'MONITOR' | 'ALERT';
export type Severity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type AlertType = 'FIRE' | 'FALL' | 'INTRUSION' | 'SMOKE' | 'WEAPON' | 'CROWD';

export interface AlertItem {
  id: string;
  type: AlertType;
  severity: Severity;
  cameraId: string;
  timestamp: string;
  message?: string;
}

export interface LogItem {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARN' | 'ERROR' | 'AGENT';
  message: string;
}

export interface Metrics {
  confidence: number;
  fps: number;
  latencyMs: number;
  modelsActive: number;
  uptime: string;
}

export interface StatusPayload {
  frame: string | null;
  status: SystemStatus;
  alerts: AlertItem[];
  logs: LogItem[];
  metrics: Metrics;
  videos: string[];
  activeVideo: string | null;
  location: string;
  currentVideo: string;
  lifecycleState: string;
  confidenceExplanation: string[];
  decisionReason: string | null;
  signals: string[];
  llmSummary: string | null;
  llmLink: string | null;
  llmConfirmation: string | null;
  systemHealth: {
    model_status: string;
    camera_status: string;
    api_status: string;
    latency: string;
  };
  decision: string | null;
  incidentLocked: boolean;
}
