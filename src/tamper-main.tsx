import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import TamperSurveillance from './pages/TamperSurveillance';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TamperSurveillance />
  </StrictMode>,
);
