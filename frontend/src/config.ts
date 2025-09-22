const trimTrailingSlash = (value: string): string => value.replace(/\/$/, '');

const resolveConfiguredPort = (): string | undefined => {
  const raw = import.meta.env.VITE_BACKEND_PORT as string | undefined;
  if (!raw || typeof raw !== 'string') return undefined;
  const trimmed = raw.trim();
  return trimmed ? trimmed : undefined;
};

const runtimeDetectHttpBase = (): string => {
  // 1) Explicit override from env wins
  const envUrl = import.meta.env.VITE_BACKEND_HTTP_URL as string | undefined;
  if (envUrl && typeof envUrl === 'string') return trimTrailingSlash(envUrl);

  // 2) Derive from current location to support remote hosts
  try {
    const loc = window.location;
    const configuredPort = resolveConfiguredPort();
    const locationPort = loc.port && loc.port !== '3000' ? loc.port : undefined;
    const port = configuredPort ?? locationPort ?? '8000';
    const host = loc.hostname;
    return trimTrailingSlash(`${loc.protocol}//${host}:${port}`);
  } catch {
    // 3) Final fallback for SSR/build contexts
    const fallbackPort = resolveConfiguredPort() ?? '8000';
    return `http://localhost:${fallbackPort}`;
  }
};

export const HTTP_BASE_URL = runtimeDetectHttpBase();

const deriveWsUrl = (httpUrl: string): string => {
  try {
    const url = new URL(httpUrl);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    return trimTrailingSlash(url.toString());
  } catch {
    const fallbackPort = resolveConfiguredPort() ?? '8000';
    return `ws://localhost:${fallbackPort}`;
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
