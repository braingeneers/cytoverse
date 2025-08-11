import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { viteStaticCopy } from 'vite-plugin-static-copy'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  optimizeDeps: {
    exclude: ['onnxruntime-web', 'parquet-wasm'],
  },
  base: './', // For gh-pages
  server: {
    cors: true,
    watch: {
      usePolling: false,
      ignored: ['**/.venv/**', '**/node_modules/**', '**/data/**'],
    },
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  esbuild: {
    // Ensure .wasm files are not bundled by esbuild
    target: 'esnext',
  },
  worker: {
    format: 'es',
    plugins: () => [vue()],
  },
  preview: {
    cors: true,
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  plugins: [
    vue(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'assets',
        },
        {
          src: 'node_modules/parquet-wasm/esm/*.wasm',
          dest: 'assets',
        },
        {
          src: 'public/*',
          dest: './',
        },
        {
          src: 'public/models/*',
          dest: './models',
        },
      ],
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@ivfpq': '/ivfpq/typescript/src',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue'],
        },
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash].[ext]',
      },
      onwarn(warning, warn) {
        if (warning.message.includes('"use client"')) {
          return
        }
        warn(warning)
      },
    },
  },
})
