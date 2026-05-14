import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const rootDir = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(rootDir, 'index.html'),
        tamper: resolve(rootDir, 'tamper.html'),
        crisis: resolve(rootDir, 'crisis.html'),
      },
    },
  },
  server: {
    port: 4173,
    proxy: {
      '/crisis/status': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 0,
        rewrite: (path) => path.replace(/^\/crisis/, ''),
      },
      '/crisis/override': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 0,
        rewrite: (path) => path.replace(/^\/crisis/, ''),
      },
      '/crisis/videos/select': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 0,
        rewrite: (path) => path.replace(/^\/crisis/, ''),
      },
      '/crisis/videos': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 0,
        rewrite: (path) => path.replace(/^\/crisis/, ''),
      },
      '/video_feed':      { target: 'http://localhost:5000', changeOrigin: true, timeout: 0 },
      '/processed_feed':  { target: 'http://localhost:5000', changeOrigin: true, timeout: 0 },
      '/video_frame':     { target: 'http://localhost:5000', changeOrigin: true },
      '/processed_frame': { target: 'http://localhost:5000', changeOrigin: true },
      '/api':             { target: 'http://localhost:5000', changeOrigin: true },
      '/socket.io':       { target: 'http://localhost:5000', changeOrigin: true, ws: true },
    },
  },
});
