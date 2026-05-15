import { useEffect, useState } from 'react';
import './home.css';

function useUtcClock() {
  const [ts, setTs] = useState('');
  useEffect(() => {
    const pad = (n: number) => String(n).padStart(2, '0');
    function tick() {
      const d = new Date();
      setTs(
        `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ` +
        `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())} UTC`
      );
    }
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return ts;
}

export default function Home() {
  const ts = useUtcClock();

  return (
    <>
      {/* NAV */}
      <header className="h-nav">
        <div className="h-brand" aria-label="AV26-001">
          <div className="h-brand-mark" aria-hidden="true" />
          <span className="h-brand-label">AV26-001</span>
        </div>
        <div className="h-nav-status" aria-live="polite">
          <span className="h-pulse" aria-hidden="true">
            <span className="h-pulse-core" />
            <span className="h-pulse-ring" />
          </span>
          <span>SYSTEMS ONLINE</span>
        </div>
      </header>

      {/* HERO */}
      <section className="h-hero">
        <div className="h-eyebrow">
          <span className="h-eyebrow-tick" aria-hidden="true" />
          <span>ENTRY POINT · OPERATIONS</span>
        </div>
        <h1 className="h-headline">
          Two systems. <em>One operations deck.</em>
        </h1>
        <p className="h-subtext">
          A unified entry point for tamper-resistance monitoring and crisis-response dispatch.
          Pick a module to open its live console.
        </p>
        <div className="h-hint">
          <span>SELECT MODULE</span>
          <span className="h-hint-arrow" aria-hidden="true">↓</span>
        </div>
      </section>

      {/* SPLIT MODULES */}
      <main className="h-split">
        <a className="h-module h-module--tamper" href="/tamper.html" aria-label="Open Tamper Resistance dashboard">
          <div className="h-module-head">
            <div className="h-module-index">MODULE 01 / TAMPER</div>
            <div className="h-module-meta">
              STATUS<br /><b>● ACTIVE</b>
            </div>
          </div>
          <h2 className="h-module-title">Tamper Resistance</h2>
          <p className="h-module-desc">
            Continuous integrity monitoring of capture feeds with cryptographic
            watermark validation and forensic evidence handling.
          </p>
          <ul className="h-feature-list">
            <li>Live monitoring <span>· feed integrity</span></li>
            <li>Watermark validation <span>· cryptographic</span></li>
            <li>Video evidence handling <span>· chain-of-custody</span></li>
            <li>Glare + low-light support <span>· adaptive</span></li>
          </ul>
          <span className="h-cta">
            Open Dashboard
            <span className="h-cta-arrow" aria-hidden="true">→</span>
          </span>
        </a>

        <a className="h-module h-module--crisis" href="/crisis.html" aria-label="Open Crisis Response dashboard">
          <div className="h-module-head">
            <div className="h-module-index">MODULE 02 / CRISIS</div>
            <div className="h-module-meta">
              STATUS<br /><b>● ARMED</b>
            </div>
          </div>
          <h2 className="h-module-title">Crisis Response</h2>
          <p className="h-module-desc">
            Real-time fire and fall detection with automated emergency dispatch.
            Single-glance situational awareness for first responders.
          </p>
          <ul className="h-feature-list">
            <li>Fire detection <span>· best.pt</span></li>
            <li>Fall detection <span>· yolov8n.pt</span></li>
            <li>Ambulance &amp; fire dispatch <span>· auto-route</span></li>
            <li>Live alert card <span>· broadcast</span></li>
          </ul>
          <span className="h-cta">
            Open Dashboard
            <span className="h-cta-arrow" aria-hidden="true">→</span>
          </span>
        </a>
      </main>

      {/* INFO STRIP */}
      <section className="h-info-wrap" aria-label="System diagnostics">
        <div className="h-info-label">SYSTEM DIAGNOSTICS</div>
        <div className="h-info-strip">
          <div className="h-cell">
            <div className="h-cell-key">FRONTEND</div>
            <div className="h-cell-val">:3000</div>
            <div className="h-cell-status"><span className="h-cell-dot" />ONLINE</div>
          </div>
          <div className="h-cell">
            <div className="h-cell-key">TAMPER BACKEND</div>
            <div className="h-cell-val">:8001</div>
            <div className="h-cell-status"><span className="h-cell-dot" />ONLINE</div>
          </div>
          <div className="h-cell">
            <div className="h-cell-key">CRISIS BACKEND</div>
            <div className="h-cell-val">:8002</div>
            <div className="h-cell-status"><span className="h-cell-dot" />ONLINE</div>
          </div>
          <div className="h-cell">
            <div className="h-cell-key">DEV SERVER</div>
            <div className="h-cell-val">:5173</div>
            <div className="h-cell-status"><span className="h-cell-dot" />ONLINE</div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="h-footer">
        <div className="h-footer-col">
          <span>AV26-001</span>
          <span className="h-footer-sep">/</span>
          <span>DUAL DASHBOARD SECURITY PLATFORM</span>
        </div>
        <div className="h-footer-col h-footer-col--right">
          <span>BUILD #1042</span>
          <span className="h-footer-sep">/</span>
          <span>{ts}</span>
        </div>
      </footer>
    </>
  );
}
