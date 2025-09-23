import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const parseRequiredPort = (value: string | undefined, key: string): number => {
  if (!value) {
    throw new Error(`[vite-config] Required environment variable "${key}" is missing. Set it in your .env file.`)
  }
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`[vite-config] ${key} must be a positive integer (received "${value}").`)
  }
  return parsed
}

const buildWsUrl = (httpUrl: string): string => {
  const url = new URL(httpUrl)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const explicitBackendUrl = env.VITE_BACKEND_URL ?? env.BACKEND_URL

  let backendUrl: string
  if (explicitBackendUrl) {
    let url: URL
    try {
      url = new URL(explicitBackendUrl)
    } catch (error) {
      throw new Error(`[vite-config] VITE_BACKEND_URL is invalid: ${error instanceof Error ? error.message : String(error)}`)
    }
    if (!url.port) {
      throw new Error('[vite-config] VITE_BACKEND_URL must include an explicit port (e.g., http://127.0.0.1:8001).')
    }
    backendUrl = url.toString().replace(/\/?$/, '')
  } else {
    const backendHost = env.BACKEND_HOST ?? env.VITE_BACKEND_HOST ?? '127.0.0.1'
    const backendPort = parseRequiredPort(env.BACKEND_PORT ?? env.VITE_BACKEND_PORT, 'BACKEND_PORT')
    backendUrl = `http://${backendHost}:${backendPort}`
  }

  const backendWsUrl = buildWsUrl(backendUrl)

  const devServerPort = env.VITE_DEV_SERVER_PORT ?? env.VITE_PORT ?? env.PORT
  const resolvedDevPort = devServerPort ? parseRequiredPort(devServerPort, 'VITE_DEV_SERVER_PORT') : 3000

  return {
    plugins: [react()],
    server: {
      port: resolvedDevPort,
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/health': {
          target: backendUrl,
          changeOrigin: true,
        },
        // Proxy WebSocket traffic during development so the frontend can use
        // same-origin WS URLs (ws://localhost:<devServerPort>/ws/...) and still reach the backend.
        '/ws': {
          target: backendWsUrl,
          changeOrigin: true,
          ws: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  }
})
