import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          framework: ['vue', 'vue-router'],
          markdown: ['marked', 'dompurify'],
          math: ['katex', 'marked-katex-extension'],
          highlight: ['highlight.js/lib/common'],
        },
      },
    },
  },
  server: {
    host: '127.0.0.1', port: 5174, strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8002', changeOrigin: true } },
  },
})
