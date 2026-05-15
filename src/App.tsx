import { useState } from 'react';
import { Shield, AlertTriangle, Activity } from 'lucide-react';
import TamperSurveillance from './pages/TamperSurveillance';
import CrisisResponse from './pages/CrisisResponse';

type Tab = 'tamper' | 'crisis';

export default function App() {
  const [tab, setTab] = useState<Tab>('tamper');

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-brand">
          <div className="app-brand-icon">
            <Activity size={16} />
          </div>
          <div>
            <div className="app-brand-name">SecureEye</div>
            <div className="app-brand-sub">Intelligence Platform</div>
          </div>
        </div>

        <nav className="tab-nav">
          <button
            type="button"
            className={`tab-btn ${tab === 'tamper' ? 'active' : ''}`}
            onClick={() => setTab('tamper')}
          >
            <Shield size={14} />
            Tamper Surveillance
          </button>
          <button
            type="button"
            className={`tab-btn ${tab === 'crisis' ? 'active' : ''}`}
            onClick={() => setTab('crisis')}
          >
            <AlertTriangle size={14} />
            Crisis Response
          </button>
        </nav>

        <div className="app-header-right" />
      </header>

      <main className="app-main">
        <div style={{ display: tab === 'tamper' ? 'block' : 'none', height: '100%' }}>
          <TamperSurveillance />
        </div>
        <div style={{ display: tab === 'crisis' ? 'block' : 'none', height: '100%' }}>
          <CrisisResponse />
        </div>
      </main>
    </div>
  );
}
