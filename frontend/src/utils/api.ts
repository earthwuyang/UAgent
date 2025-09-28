/**
 * API utilities for UAgent frontend
 */

export interface FetchOptions extends RequestInit {
  timeout?: number;
}

/**
 * Enhanced fetch function with proxy bypass and timeout support
 */
export const apiFetch = async (url: string, options: FetchOptions = {}): Promise<Response> => {
  const { timeout = 15000, ...fetchOptions } = options;

  // Create AbortController for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    // Check if this is a localhost URL to potentially bypass proxy
    const isLocalhost = url.includes('localhost') || url.includes('127.0.0.1');

    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        ...fetchOptions.headers,
      },
      // Bypass proxy for localhost
      ...(isLocalhost && {
        mode: 'cors' as RequestMode,
      }),
      ...fetchOptions,
    });

    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
};

/**
 * Get the backend base URL from environment
 */
export const getBackendUrl = (): string => {
  return import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';
};

/**
 * Enhanced API call with error handling
 */
export const apiCall = async <T>(
  endpoint: string,
  options: FetchOptions = {}
): Promise<{ data?: T; error?: string }> => {
  try {
    const url = `${getBackendUrl()}${endpoint}`;
    const response = await apiFetch(url, options);

    if (!response.ok) {
      return {
        error: `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data = await response.json();
    return { data };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return {
      error: errorMessage.includes('aborted') ? 'Request timeout' : errorMessage,
    };
  }
};

/**
 * Retry function with exponential backoff
 */
export const retryApiCall = async <T>(
  apiCallFn: () => Promise<{ data?: T; error?: string }>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<{ data?: T; error?: string }> => {
  let lastError: string = 'Unknown error';

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const result = await apiCallFn();

    if (result.data !== undefined) {
      return result;
    }

    lastError = result.error || 'Unknown error';

    // If this is the last attempt, return the error
    if (attempt === maxRetries) {
      break;
    }

    // Wait before retrying with exponential backoff
    const delay = baseDelay * Math.pow(2, attempt - 1);
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  return { error: lastError };
};