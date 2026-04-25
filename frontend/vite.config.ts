import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Ports from environment (set by dev.sh from appregistry) or defaults
const FRONTEND_PORT = parseInt(process.env.PORT || '3000', 10)
const BACKEND_PORT = parseInt(process.env.BACKEND_PORT || '9005', 10)
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`

// Base path for sub-path deployments (e.g. /codeloom).
// Set via VITE_BASE_PATH at build time for Docker; defaults to '/' for local dev.
const BASE_PATH = process.env.VITE_BASE_PATH || '/'
const normalizedBase = BASE_PATH === '/' ? '/' : `${BASE_PATH.replace(/\/+$/, '')}/`

// https://vite.dev/config/
export default defineConfig({
  base: normalizedBase,
  plugins: [react(), tailwindcss()],
  server: {
    port: FRONTEND_PORT,
    proxy: {
      '/api': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/mcp': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/chat': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/upload': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/image': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
    },
  },
})
