import { createRoot } from 'react-dom/client';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Root element not found');
}

createRoot(rootElement).render(
  <div style={{ fontFamily: 'system-ui, sans-serif', padding: '1rem' }}>
    Flood Warning System
  </div>
);
