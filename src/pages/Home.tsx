import type { CSSProperties } from 'react';
import { ArrowRight, Shield, Flame, UserRound, Cpu } from 'lucide-react';

function DemoCard({
  title,
  description,
  href,
  accent,
  Icon,
  bullets,
}: {
  title: string;
  description: string;
  href: string;
  accent: string;
  Icon: typeof Shield;
  bullets: string[];
}) {
  return (
    <a className="home-demo-card" href={href} style={{ '--home-accent': accent } as CSSProperties}>
      <div className="home-demo-top">
        <div className="home-demo-icon">
          <Icon size={18} />
        </div>
        <ArrowRight size={16} className="home-demo-arrow" />
      </div>
      <div className="home-demo-title">{title}</div>
      <p className="home-demo-description">{description}</p>
      <ul className="home-demo-bullets">
        {bullets.map((bullet) => (
          <li key={bullet}>{bullet}</li>
        ))}
      </ul>
    </a>
  );
}

export default function Home() {
  return (
    <div className="home-shell">
      <section className="home-hero panel">
        <div className="home-kicker">Unified Intelligence Platform</div>
        <h1>Two isolated demos, one launch screen.</h1>
        <p className="home-copy">
          This project demonstrates two separate real-time systems: tamper resistance for live camera monitoring and
          crisis response for fire/fall detection. Pick a demo below to open the dashboard you want.
        </p>
        <div className="home-metrics">
          <div>
            <span>01</span>
            <strong>Tamper Resistance</strong>
          </div>
          <div>
            <span>02</span>
            <strong>Crisis Response</strong>
          </div>
          <div>
            <span>GPU</span>
            <strong>CUDA Runtime</strong>
          </div>
        </div>
      </section>

      <section className="home-grid">
        <DemoCard
          title="Tamper Resistance"
          description="Live camera defense UI with blur, shake, liveness, glare, and repositioning checks."
          href="/tamper.html"
          accent="#4ade80"
          Icon={Shield}
          bullets={['Camera monitoring', 'Alert stream', 'Operator controls']}
        />
        <DemoCard
          title="Crisis Response"
          description="Fire and fall detection dashboard with a single alert, deployment state, and clip controls."
          href="/crisis.html"
          accent="#60a5fa"
          Icon={Flame}
          bullets={['Fire dispatch', 'Fall dispatch', 'Video selector']}
        />
      </section>

      <section className="home-footer panel">
        <div className="home-footer-item">
          <Cpu size={16} />
          <span>CUDA-backed inference when available</span>
        </div>
        <div className="home-footer-item">
          <UserRound size={16} />
          <span>Separate runtime and UI for each demo</span>
        </div>
      </section>
    </div>
  );
}
