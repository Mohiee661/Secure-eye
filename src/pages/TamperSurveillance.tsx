import { useEffect, useState, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import {
  Shield, Activity, Eye, Settings, Volume2,
  AlertTriangle, Wifi, WifiOff,
} from 'lucide-react';

const BACKEND = '';         // relative — proxied through Vite to localhost:5000

void BACKEND;
type SystemStatus = 'SECURE' | 'MONITORING' | 'ALERT' | 'CONNECTING';

interface DetectionData {
  blur:       { detected: boolean; variance: number };
  shake:      { detected: boolean; magnitude: number };
  reposition: { detected: boolean; magnitude: number; shift_x: number; shift_y: number; alert_active: boolean };
  glare:      { detected: boolean; dark_pct: number; mid_pct: number; bright_pct: number };
  liveness:   { frozen: boolean; blackout: boolean; status: string; mean_diff: number; mean_brightness: number; is_active: boolean };
}

interface LogEntry {
  id: number;
  time: string;
  message: string;
  type: 'info' | 'warn' | 'alert';
}

interface SensorStates { [key: string]: boolean }

const defaultDetection: DetectionData = {
  blur:       { detected: false, variance: 0 },
  shake:      { detected: false, magnitude: 0 },
  reposition: { detected: false, magnitude: 0, shift_x: 0, shift_y: 0, alert_active: false },
  glare:      { detected: false, dark_pct: 0, mid_pct: 0, bright_pct: 0 },
  liveness:   { frozen: false, blackout: false, status: 'ACTIVE', mean_diff: 0, mean_brightness: 0, is_active: true },
};

const defaultSensors: SensorStates = {
  blur: true, shake: true, reposition: true, glare: true,
  liveness: true, audio: true, watermark: true, forensic: true,
};

const sensorLabels: Record<string, string> = {
  blur: 'Blur Detection', shake: 'Shake Detection', reposition: 'Repositioning',
  glare: 'Glare / CLAHE', liveness: 'Liveness Check', audio: 'Audio Analysis',
  watermark: 'HMAC Watermark', forensic: 'Forensic Logging',
};

let _id = 0;
const nextId = () => ++_id;

function statusColor(status: SystemStatus, connected: boolean) {
  if (!connected) return 'var(--text-muted)';
  if (status === 'ALERT')      return 'var(--alert)';
  if (status === 'MONITORING') return 'var(--monitor)';
  return 'var(--safe)';
}

export default function TamperSurveillance() {
  const [connected, setConnected]         = useState(false);
  const [status, setStatus]               = useState<SystemStatus>('CONNECTING');
  const [detection, setDetection]         = useState<DetectionData>(defaultDetection);
  const [logs, setLogs]                   = useState<LogEntry[]>([]);
  const [subtitles, setSubtitles]         = useState<LogEntry[]>([]);
  const [sensors, setSensors]             = useState<SensorStates>(defaultSensors);
  const [repositionAlert, setReposition]  = useState(false);
  const [glareMode, setGlareMode]         = useState<'CLAHE' | 'MSR'>('CLAHE');
  const socketRef = useRef<Socket | null>(null);
  const rawRef    = useRef<HTMLImageElement>(null);
  const procRef   = useRef<HTMLImageElement>(null);

  const addLog = (message: string, type: LogEntry['type'] = 'info') => {
    const time = new Date().toTimeString().slice(0, 8);
    setLogs(prev => [{ id: nextId(), time, message, type }, ...prev].slice(0, 60));
  };

  useEffect(() => {
    const socket = io(undefined, { transports: ['websocket', 'polling'] });
    socketRef.current = socket;

    socket.on('connect', () => {
      setConnected(true);
      setStatus('SECURE');
      addLog('Connected to AEGIS defense system', 'info');
      socket.emit('get_sensor_states');
    });

    socket.on('disconnect', () => {
      setConnected(false);
      setStatus('CONNECTING');
      addLog('Connection to AEGIS system lost', 'warn');
    });

    socket.on('status_update', ({ status: s, message }: { status: string; message: string }) => {
      setStatus(s === 'alert' ? 'ALERT' : s === 'monitoring' ? 'MONITORING' : 'SECURE');
      addLog(message, s === 'alert' ? 'alert' : 'info');
    });

    socket.on('detection_update', (data: DetectionData) => {
      setDetection(data);
      if (data.reposition?.alert_active) setReposition(true);
      const flags: string[] = [];
      if (data.blur.detected)      flags.push('BLUR');
      if (data.shake.detected)     flags.push('SHAKE');
      if (data.glare.detected)     flags.push('GLARE');
      if (data.liveness.frozen)    flags.push('FROZEN FEED');
      if (data.liveness.blackout)  flags.push('BLACKOUT');
      if (flags.length) {
        addLog(flags.join(' · ') + ' DETECTED', 'alert');
        setStatus('ALERT');
      } else {
        setStatus(prev => prev === 'ALERT' ? 'SECURE' : prev);
      }
    });

    socket.on('alert',       ({ message }: { message: string }) => { addLog(message, 'alert'); setStatus('ALERT'); });
    socket.on('alert_clear', () => { setStatus('SECURE'); addLog('Alert cleared — system nominal', 'info'); });
    socket.on('subtitle',    ({ text, type }: { text: string; type: string }) => {
      const time = new Date().toTimeString().slice(0, 8);
      const entryType: LogEntry['type'] = type === 'blackbox' ? 'warn' : 'info';
      setSubtitles(prev => [{ id: nextId(), time, message: text, type: entryType }, ...prev].slice(0, 40));
    });
    socket.on('sensor_states', (states: SensorStates) => setSensors(states));

    socket.on('reposition_alert', (data: { magnitude: number; shift_x: number; shift_y: number }) => {
      setDetection(prev => ({
        ...prev,
        reposition: { ...prev.reposition, magnitude: data.magnitude, shift_x: data.shift_x, shift_y: data.shift_y, alert_active: true },
      }));
      setReposition(true);
      addLog('CAMERA REPOSITIONING DETECTED', 'alert');
    });

    return () => { socket.disconnect(); };
  }, []);

  useEffect(() => {
    const tick = () => {
      const t = Date.now();
      if (rawRef.current)  rawRef.current.src  = `/video_frame?t=${t}`;
      if (procRef.current) procRef.current.src = `/processed_frame?t=${t}`;
    };
    tick();
    const id = setInterval(tick, 200);
    return () => clearInterval(id);
  }, []);

  const toggleSensor = (sensor: string, enabled: boolean) => {
    setSensors(prev => ({ ...prev, [sensor]: enabled }));
    socketRef.current?.emit('set_sensor_enabled', { sensor, enabled });
  };

  const switchGlareMode = (mode: 'CLAHE' | 'MSR') => {
    setGlareMode(mode);
    socketRef.current?.emit('set_glare_mode', { mode });
  };

  const dismissReposition = () => {
    setReposition(false);
    socketRef.current?.emit('dismiss_reposition_alert');
  };


  const sc = statusColor(status, connected);

  const metrics = [
    { label: 'BLUR',     value: detection.blur.variance.toFixed(2),   unit: 'var', detected: detection.blur.detected },
    { label: 'SHAKE',    value: detection.shake.magnitude.toFixed(2),  unit: 'mag', detected: detection.shake.detected },
    { label: 'GLARE',    value: detection.glare.bright_pct.toFixed(1), unit: '%',   detected: detection.glare.detected },
    { label: 'LIVENESS', value: detection.liveness.status,             unit: '',    detected: !detection.liveness.is_active },
  ];

  return (
    <div className="tamper-container">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="tamper-header panel">
        <div className="tamper-header-left">
          <Shield size={18} color="#00e676" />
          <span className="tamper-title">ACTIVE DEFENSE SYSTEM</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            type="button"
            onClick={() => { window.location.href = '/crisis.html'; }}
            style={{
              padding: '8px 14px',
              borderRadius: 10,
              border: '1px solid rgba(0, 230, 118, 0.28)',
              background: 'rgba(0, 230, 118, 0.08)',
              color: 'var(--text)',
              fontFamily: 'inherit',
              fontSize: 12,
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              cursor: 'pointer',
            }}
          >
            Open Crisis
          </button>
          {!connected && (
            <div className="conn-banner">
              <WifiOff size={12} />
              BACKEND OFFLINE — START python app.py
            </div>
          )}
          <div className="status-badge-container">
            <div
              className="status-dot-large"
              style={{
                backgroundColor: sc,
                boxShadow: `0 0 10px ${sc}80`,
                animation: status === 'ALERT' ? 'pulse-alert 1.5s infinite' : 'none',
              }}
            />
            <span className="status-text" style={{ color: sc }}>
              {connected ? status : 'CONNECTING'}
            </span>
            {connected ? <Wifi size={13} color="var(--text-muted)" /> : <WifiOff size={13} color="var(--alert)" />}
          </div>
        </div>
      </div>

      {/* ── Video Panels ───────────────────────────────────────────────────── */}
      <div className="video-grid">
        {/* Raw Feed */}
        <div className={`video-panel panel ${detection.blur.detected || detection.shake.detected ? 'glow-alert' : 'glow-safe'}`}>
          <div className="video-panel-label">
            <span className="mono-label">RAW FEED</span>
            <span className="mono-label" style={{ color: 'var(--monitor)' }}>UNPROTECTED</span>
          </div>
          <div className="video-aspect">
            <img
              ref={rawRef}
              alt="Raw surveillance feed"
              className="video-img"
            />
            <div className="video-placeholder">
              <Eye size={28} color="var(--border)" />
              <span className="mono-label">FEED UNAVAILABLE</span>
              <div className="scan-line" />
            </div>
          </div>
        </div>

        {/* Processed Feed */}
        <div className={`video-panel panel ${status === 'ALERT' ? 'glow-alert' : status === 'MONITORING' ? 'glow-monitor' : 'glow-safe'}`}>
          <div className="video-panel-label">
            <span className="mono-label">AEGIS PROCESSED</span>
            <span className="mono-label" style={{ color: 'var(--safe)' }}>DEFENDED</span>
          </div>
          <div className="video-aspect">
            <img
              ref={procRef}
              alt="AEGIS processed feed"
              className="video-img"
            />
            <div className="video-placeholder">
              <Shield size={28} color="var(--border)" />
              <span className="mono-label">FEED UNAVAILABLE</span>
              <div className="scan-line" />
            </div>
          </div>
          {/* Glare mode toggle */}
          <div style={{ display: 'flex', gap: 6, padding: '8px 14px', borderTop: '1px solid rgba(70,58,65,0.4)' }}>
            <span className="mono-label" style={{ marginRight: 4 }}>ENHANCE MODE</span>
            {(['CLAHE', 'MSR'] as const).map(m => (
              <button
                key={m}
                onClick={() => switchGlareMode(m)}
                style={{
                  padding: '3px 10px',
                  borderRadius: 6,
                  border: 'none',
                  background: glareMode === m ? 'linear-gradient(135deg,var(--primary),var(--accent))' : 'var(--border)',
                  color: glareMode === m ? '#fff' : 'var(--text-muted)',
                  fontFamily: 'Ubuntu Mono, monospace',
                  fontSize: 9,
                  letterSpacing: '0.15em',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Metrics Strip ──────────────────────────────────────────────────── */}
      <div className="metrics-strip panel">
        {metrics.map(({ label, value, unit, detected }) => (
          <div key={label} className="metric-card">
            <span className="mono-label metric-label">{label}</span>
            <div
              className="metric-indicator"
              style={{
                backgroundColor: detected ? 'var(--alert)' : 'var(--safe)',
                boxShadow: detected ? '0 0 8px rgba(255,68,68,0.6)' : '0 0 4px rgba(0,230,118,0.25)',
              }}
            />
            <span className="metric-value" style={{ color: detected ? 'var(--alert)' : 'var(--text)' }}>
              {value}
              {unit && <span className="metric-unit"> {unit}</span>}
            </span>
            <span className="metric-status" style={{ color: detected ? 'var(--alert)' : 'var(--text-muted)' }}>
              {detected ? 'DETECTED' : 'NOMINAL'}
            </span>
          </div>
        ))}
      </div>

      {/* ── Console Grid ───────────────────────────────────────────────────── */}
      <div className="console-grid">

        {/* Detection Log */}
        <div className="panel console-panel">
          <div className="console-header">
            <Activity size={13} color="var(--primary)" />
            <span className="mono-label">DETECTION LOG</span>
            <span className="mono-label" style={{ marginLeft: 'auto', color: 'rgba(0,230,118,0.35)' }}>
              {logs.length} EVENTS
            </span>
          </div>
          <div className="log-scroll">
            {logs.length === 0 && (
              <span className="mono-label" style={{ color: 'var(--border)', padding: '4px 0' }}>
                AWAITING EVENTS...
              </span>
            )}
            {logs.map(e => (
              <div key={e.id} className={`log-entry log-${e.type}`}>
                <span className="log-time">{e.time}</span>
                <span className="log-msg">{e.message}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Audio Log */}
        <div className="panel console-panel">
          <div className="console-header">
            <Volume2 size={13} color="var(--text-muted)" />
            <span className="mono-label">AUDIO LOG</span>
            <span className="mono-label" style={{ marginLeft: 'auto', color: 'rgba(0,230,118,0.35)' }}>
              {subtitles.length} ENTRIES
            </span>
          </div>
          <div className="log-scroll">
            {subtitles.length === 0 && (
              <span className="mono-label" style={{ color: 'var(--border)', padding: '4px 0' }}>
                LISTENING...
              </span>
            )}
            {subtitles.map(e => (
              <div key={e.id} className={`log-entry log-${e.type}`}>
                <span className="log-time">{e.time}</span>
                <span className="log-msg">{e.message}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Sensor Config */}
        <div className="panel console-panel">
          <div className="console-header">
            <Settings size={13} color="var(--text-muted)" />
            <span className="mono-label">SENSOR CONFIG</span>
          </div>
          <div className="sensor-list">
            {Object.entries(sensorLabels).map(([key, label]) => (
              <div key={key} className="sensor-row">
                <span className="sensor-label">{label}</span>
                <button
                  className={`sensor-toggle ${sensors[key] ? 'active' : ''}`}
                  onClick={() => toggleSensor(key, !sensors[key])}
                  aria-label={`Toggle ${label}`}
                >
                  <div className="toggle-thumb" />
                </button>
              </div>
            ))}
          </div>
        </div>

      </div>


      {/* ── Repositioning Alert Modal ───────────────────────────────────────── */}
      {repositionAlert && (
        <div className="modal-overlay">
          <div className="modal panel glow-alert">
            <div className="modal-icon">
              <AlertTriangle size={44} color="var(--alert)" />
            </div>
            <h2 className="modal-title">CAMERA REPOSITIONED</h2>
            <p className="modal-body">
              Physical camera movement detected. The camera may have been repositioned
              or tampered with. Verify the camera position immediately.
            </p>
            <div className="modal-metrics">
              <div className="modal-metric">
                <span className="mono-label">SHIFT X</span>
                <span className="modal-metric-val">{detection.reposition.shift_x.toFixed(1)}px</span>
              </div>
              <div className="modal-metric">
                <span className="mono-label">SHIFT Y</span>
                <span className="modal-metric-val">{detection.reposition.shift_y.toFixed(1)}px</span>
              </div>
              <div className="modal-metric">
                <span className="mono-label">MAGNITUDE</span>
                <span className="modal-metric-val">{detection.reposition.magnitude.toFixed(2)}</span>
              </div>
            </div>
            <button className="modal-dismiss" onClick={dismissReposition}>
              ACKNOWLEDGE &amp; DISMISS
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
