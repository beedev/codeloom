import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true,
      },
      '/chat': {
        target: 'http://localhost:7860',
        changeOrigin: true,
      },
      '/upload': {
        target: 'http://localhost:7860',
        changeOrigin: true,
      },
      '/image': {
        target: 'http://localhost:7860',
        changeOrigin: true,
      },
    },
  },
})
