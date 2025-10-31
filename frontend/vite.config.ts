import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

/**
 * Vite Configuration for Dual RAG LLM Frontend
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3001,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'ui-vendor': ['lucide-react', 'react-markdown', 'react-syntax-highlighter'],
        },
      },
    },
  },
})

