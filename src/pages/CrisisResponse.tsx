import { useEffect, useRef, useState } from 'react';
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
  if (decision === 'ALERT_AMBULANCE')    return 'DEPLOYED AMBULANCE';
  if (decision === 'ALERT_FIRE_ENGINE')  return 'DEPLOYED FIRE ENGINE';
  if (decision === 'ONSITE_TEAM')        return 'ON-SITE TEAM DISPATCHED';
  if (decision === 'GATE2_PENDING')      return 'AWAITING CONFIRMATION';
  if (decision === 'GATE1_MONITORING')   return 'MONITORING — RECOVERY WINDOW';
  return 'DEPLOYMENT STANDBY';
}

function deploymentClass(decision: string | null) {
  if (decision === 'ALERT_AMBULANCE' || decision === 'ALERT_FIRE_ENGINE' || decision === 'ONSITE_TEAM') return 'active';
  if (decision === 'GATE2_PENDING')    return 'pending';
  if (decision === 'GATE1_MONITORING') return 'monitoring';
  return 'idle';
}

function DeploymentBadge({ decision }: { decision: string | null }) {
  return <div className={`crisis-pill deployment ${deploymentClass(decision)}`}>{deploymentLabel(decision)}</div>;
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

function VideoViewport({ label, location }: { label: string; location: string }) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [hasFrame, setHasFrame] = useState(false);

  useEffect(() => {
    const tick = () => {
      if (imgRef.current) imgRef.current.src = `/crisis/frame?t=${Date.now()}`;
    };
    tick();
    const id = setInterval(tick, 100);
    return () => clearInterval(id);
  }, []);

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
        <img
          ref={imgRef}
          alt="Crisis feed"
          className="crisis-video-img"
          style={{ display: hasFrame ? 'block' : 'none' }}
          onLoad={() => setHasFrame(true)}
          onError={() => setHasFrame(false)}
        />
        {!hasFrame && (
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
      {videos.length === 0 ? (
        <p className="crisis-note" style={{ marginTop: 10 }}>No videos found — add .mp4 files to the /videos directory</p>
      ) : (
        <select
          className="crisis-video-select"
          value={activeVideo ?? ''}
          onChange={(e) => { if (e.target.value) onSelect(e.target.value); }}
        >
          {!activeVideo && <option value="">— select a video to evaluate —</option>}
          {videos.map((video) => (
            <option key={video} value={video}>{video}</option>
          ))}
        </select>
      )}
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

  return (
    <div className="crisis-shell clean" ref={ref}>
      <CleanHeader status={data.status} connected={connected} decision={data.decision} />

      <main className="crisis-clean-grid">
        <section className="crisis-main-column">
          <VideoViewport
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
            <div className={`crisis-deployment-readout ${deploymentClass(data.decision)}`}>
              {deploymentLabel(data.decision)}
            </div>
            {data.decision === 'GATE1_MONITORING' && (
              <div className="crisis-note" style={{ marginTop: 8, color: 'var(--warn)' }}>
                AI watching for 9s — suppresses if person recovers
              </div>
            )}
            {data.decision === 'GATE2_PENDING' && (
              <div className="crisis-note" style={{ marginTop: 8, color: 'var(--blue)' }}>
                Email sent to authority — waiting for response
              </div>
            )}
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
              Gate 1 pre-screens events — fire requires sustained detection, fall watches for 3s recovery.
              Gate 2 emails the authority a 7-second clip for final confirmation.
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
