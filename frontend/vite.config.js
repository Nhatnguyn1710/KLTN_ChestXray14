import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const isDev = process.env.NODE_ENV !== 'production';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: isDev ? '/' : '/static/dist/',
  build: {
    // Vite output → src/api/static/dist (served by FastAPI /static mount)
    outDir: path.resolve(__dirname, '../src/api/static/dist'),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
      '/static/gradcam': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
    },
  },
});
