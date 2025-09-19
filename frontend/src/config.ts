const trimTrailingSlash = (value: string): string => value.replace(/\/$/, '');

export const HTTP_BASE_URL = trimTrailingSlash(
  (import.meta.env.VITE_BACKEND_HTTP_URL as string | undefined) ?? 'http://localhost:8000'
);

const deriveWsUrl = (httpUrl: string): string => {
  try {
    const url = new URL(httpUrl);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    return url.toString().replace(/\/$/, '');
  } catch (error) {
    return 'ws://localhost:8000';
  }
};

export const WS_BASE_URL = trimTrailingSlash(
  (import.meta.env.VITE_BACKEND_WS_URL as string | undefined) ?? deriveWsUrl(HTTP_BASE_URL)
);
