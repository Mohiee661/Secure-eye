import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import CrisisResponse from './pages/CrisisResponse';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <CrisisResponse />
  </StrictMode>,
);
