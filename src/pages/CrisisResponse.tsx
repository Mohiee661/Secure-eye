import { useEffect, useRef } from 'react';
import {
  Activity,
  AlertTriangle,
  Camera,
  Flame,
  ShieldAlert,
  UserRound,
  Wifi,
  WifiOff,
  Wind,
  Crosshair,
  Users,
} from 'lucide-react';
import { useCrisisStatus } from '../hooks/use-crisis-status';
import type { AlertItem, AlertType, Severity, SystemStatus } from '../lib/crisis-types';

const ALERT_ICON: Record<AlertType, typeof Flame> = {
  FIRE: Flame,
  FALL: UserRound,
  INTRUSION: ShieldAlert,
  SMOKE: Wind,
  WEAPON: Crosshair,
  CROWD: Users,
};

const STATUS_LABELS: Record<SystemStatus, string> = {
  SAFE: 'SAFE',
  MONITOR: 'MONITOR',
  ALERT: 'ALERT',
};

const SEVERITY_CLASS: Record<Severity, string> = {
  LOW: 'crisis-sev-low',
  MEDIUM: 'crisis-sev-med',
  HIGH: 'crisis-sev-high',
  CRITICAL: 'crisis-sev-critical',
};

function deploymentLabel(decision: string | null) {
  if (decision === 'ALERT_AMBULANCE') return 'DEPLOYED AMBULANCE';
  if (decision === 'ALERT_FIRE_ENGINE') return 'DEPLOYED FIRE ENGINE';
  return 'DEPLOYMENT STANDBY';
}

function DeploymentBadge({ decision }: { decision: string | null }) {
  const label = deploymentLabel(decision);
  const deployed = decision === 'ALERT_AMBULANCE' || decision === 'ALERT_FIRE_ENGINE';
  return <div className={`crisis-pill deployment ${deployed ? 'active' : 'idle'}`}>{label}</div>;
}

function StatusBadge({ status }: { status: SystemStatus }) {
  return <div className={`crisis-pill status ${status.toLowerCase()}`}>SYSTEM - {STATUS_LABELS[status]}</div>;
}

function ActiveAlert({ alert }: { alert?: AlertItem }) {
  if (!alert) {
    return (
      <div className="crisis-empty-card">
        <ShieldAlert size={34} />
        <div>
          <div className="crisis-empty-title">NO ACTIVE ALERTS</div>
          <div className="crisis-empty-sub">Waiting for a detection event</div>
        </div>
      </div>
    );
  }

  const Icon = ALERT_ICON[alert.type];
  return (
    <div className={`crisis-alert-card ${SEVERITY_CLASS[alert.severity]}`}>
      <div className="crisis-alert-head">
        <div className="crisis-alert-icon">
          <Icon size={18} />
        </div>
        <div className="crisis-alert-copy">
          <div className="crisis-alert-title">{alert.type}</div>
          <div className="crisis-alert-meta">
            <span>{alert.cameraId}</span>
            <span>{new Date(alert.timestamp).toLocaleTimeString([], { hour12: false })}</span>
          </div>
        </div>
        <div className="crisis-pill severity">{alert.severity}</div>
      </div>

      {alert.message && <div className="crisis-alert-message">{alert.message}</div>}
    </div>
  );
}

function VideoViewport({
  src,
  label,
  location,
}: {
  src: string | null;
  label: string;
  location: string;
}) {
  return (
    <div className="crisis-video-card">
      <div className="crisis-card-top">
        <div>
          <div className="crisis-card-title">LIVE VIDEO</div>
          <div className="crisis-card-subtitle">{label} - {location}</div>
        </div>
        <div className="crisis-live">
          <span className="crisis-live-dot" />
          REC
        </div>
      </div>

      <div className="crisis-video-frame">
        {src ? (
          <img src={src} alt="Selected crisis video" className="crisis-video-img" />
        ) : (
          <div className="crisis-video-placeholder">
            <Camera size={28} />
            <div>
              <div className="crisis-placeholder-title">NO FRAME</div>
              <div className="crisis-placeholder-sub">Upload a video into /videos</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function VideoControls({
  videos,
  activeVideo,
  onSelect,
}: {
  videos: string[];
  activeVideo: string | null;
  onSelect: (video: string) => void;
}) {
  return (
    <div className="crisis-video-controls">
      <div className="crisis-card-title">SELECT VIDEO</div>
      <div className="crisis-control-grid">
        {videos.map((video) => (
          <button
            key={video}
            className={`crisis-video-btn ${activeVideo === video ? 'active' : ''}`}
            onClick={() => onSelect(video)}
            type="button"
          >
            {video}
          </button>
        ))}
      </div>
    </div>
  );
}

function CleanHeader({ status, connected, decision }: { status: SystemStatus; connected: boolean; decision: string | null }) {
  return (
    <header className="crisis-header clean">
      <div className="crisis-brand">
        <div className="crisis-brand-mark">
          <Activity size={18} />
        </div>
        <div>
          <h1>AI Crisis Command Center</h1>
          <p>separate crisis runtime</p>
        </div>
      </div>
      <div className="crisis-header-right">
        <button
          type="button"
          onClick={() => { window.location.href = '/'; }}
          className="crisis-action-btn ghost"
          style={{ minHeight: 36 }}
        >
          Open Tamper
        </button>
        <DeploymentBadge decision={decision} />
        <div className="crisis-connection">
          {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
          <span>{connected ? 'ONLINE' : 'OFFLINE'}</span>
        </div>
        <StatusBadge status={status} />
      </div>
    </header>
  );
}

export default function CrisisResponse() {
  const { data, connected, override, selectVideo } = useCrisisStatus();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.scrollTo({ top: 0 });
  }, [data.activeVideo]);

  const src = data.frame
    ? data.frame.startsWith('data:')
      ? data.frame
      : `data:image/jpeg;base64,${data.frame}`
    : null;

  return (
    <div className="crisis-shell clean" ref={ref}>
      <CleanHeader status={data.status} connected={connected} decision={data.decision} />

      <main className="crisis-clean-grid">
        <section className="crisis-main-column">
          <VideoViewport
            src={src}
            label={data.activeVideo ?? data.currentVideo}
            location={data.location}
          />
          <VideoControls
            videos={data.videos}
            activeVideo={data.activeVideo ?? data.currentVideo}
            onSelect={selectVideo}
          />
        </section>

        <aside className="crisis-side-column">
          <div className="crisis-deployment-card">
            <div className="crisis-card-top">
              <div>
                <div className="crisis-card-title">DEPLOYMENT</div>
                <div className="crisis-card-subtitle">live response state</div>
              </div>
            </div>
            <div className={`crisis-deployment-readout ${data.decision === 'ALERT_AMBULANCE' || data.decision === 'ALERT_FIRE_ENGINE' ? 'active' : ''}`}>
              {deploymentLabel(data.decision)}
            </div>
          </div>

          <div className="crisis-alert-panel">
            <div className="crisis-card-top">
              <div>
                <div className="crisis-card-title">ACTIVE ALERT</div>
                <div className="crisis-card-subtitle">single live alert</div>
              </div>
              <AlertTriangle className={data.status === 'ALERT' ? 'crisis-alert-icon-alert' : 'crisis-alert-icon-muted'} />
            </div>
            <ActiveAlert alert={data.alerts[0]} />
          </div>

          <div className="crisis-actions-panel">
            <div className="crisis-card-title">OPERATOR ACTION</div>
            <div className="crisis-action-row">
              <button className="crisis-action-btn primary" onClick={() => void override('ALERT_AMBULANCE')} type="button">
                Dispatch Ambulance
              </button>
              <button className="crisis-action-btn primary" onClick={() => void override('ALERT_FIRE_ENGINE')} type="button">
                Dispatch Fire Engine
              </button>
              <button className="crisis-action-btn ghost" onClick={() => void override('IGNORE')} type="button">
                Ignore Alert
              </button>
            </div>
            <div className="crisis-note">
              Use the video selector to switch between fire and fall samples. The backend keeps one active alert card only.
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
