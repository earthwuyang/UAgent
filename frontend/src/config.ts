const trimTrailingSlash = (value: string): string => value.replace(/\/$/, '');

const runtimeDetectHttpBase = (): string => {
  // 1) Explicit override from env wins
  const envUrl = import.meta.env.VITE_BACKEND_HTTP_URL as string | undefined;
  if (envUrl && typeof envUrl === 'string') return trimTrailingSlash(envUrl);

  // 2) Derive from current location to support remote hosts
  try {
    const loc = window.location;
    const host = loc.hostname;
    const isLocal = host === 'localhost' || host === '127.0.0.1';
    // Allow optional port override; default to 8001
    const configuredPort = (import.meta.env.VITE_BACKEND_PORT as string | undefined) || '8001';
    const port = isLocal ? '8001' : configuredPort;
    return trimTrailingSlash(`${loc.protocol}//${host}:${port}`);
  } catch {
    // 3) Final fallback for SSR/build contexts
    return 'http://localhost:8001';
  }
};

export const HTTP_BASE_URL = runtimeDetectHttpBase();

const deriveWsUrl = (httpUrl: string): string => {
  try {
    const url = new URL(httpUrl);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    return trimTrailingSlash(url.toString());
  } catch {
    return 'ws://localhost:8001';
  }
};

const runtimeDetectWsBase = (): string => {
  const envWs = import.meta.env.VITE_BACKEND_WS_URL as string | undefined;
  if (envWs && typeof envWs === 'string') return trimTrailingSlash(envWs);

  try {
    // During dev (vite at :3000), prefer same-origin WS so proxy can handle it.
    const loc = window.location;
    if (loc.port === '3000') {
      const wsProto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
      return trimTrailingSlash(`${wsProto}//${loc.host}`);
    }
  } catch {}

  return deriveWsUrl(HTTP_BASE_URL);
};

export const WS_BASE_URL = runtimeDetectWsBase();
