import { useEffect, useRef } from 'react';
import { useResearchTreeStore, ResearchNodeEvent } from '#/state/research-tree-store';

interface SSEMessagePayload {
  event_type: string;
  experiment_id: string;
  branch_id?: string | null;
  node_id?: string | null;
  timestamp: string;
  message?: string | null;
  recoverable?: boolean;
  data?: Record<string, unknown>;
}

const createEventId = () => {
  const globalCrypto = typeof globalThis !== 'undefined' ? (globalThis as typeof globalThis & { crypto?: Crypto }).crypto : undefined;
  if (globalCrypto && typeof globalCrypto.randomUUID === 'function') {
    return globalCrypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
};

export function useResearchEventStream(experimentId?: string | null) {
  const experimentRef = useRef<string | null>(null);

  useEffect(() => {
    if (!experimentId) {
      return undefined;
    }

    if (experimentRef.current === experimentId) {
      return undefined;
    }

    experimentRef.current = experimentId;
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const host = import.meta.env.VITE_BACKEND_BASE_URL || window.location.host;
    const url = `${protocol}//${host}/api/research/experiments/${experimentId}/events/stream`;

    const source = new EventSource(url, { withCredentials: false });

    source.addEventListener('research_event', (event) => {
      try {
        const payload: SSEMessagePayload = JSON.parse((event as MessageEvent).data);
        if (!payload?.node_id) {
          return;
        }

        const nodeEvent: ResearchNodeEvent = {
          id: createEventId(),
          node_id: payload.node_id,
          branch_id: payload.branch_id ?? null,
          event_type: payload.event_type,
          message: payload.message ?? null,
          timestamp: payload.timestamp,
          data: payload.data ?? {},
        };

        const store = useResearchTreeStore.getState();
        store.addNodeEvent(nodeEvent);

        if (payload.event_type === 'error') {
          const node = store.nodes.get(payload.node_id);
          const existingMetadata = (node?.metadata ?? {}) as Record<string, unknown>;
          const existingErrors = Array.isArray(existingMetadata.errors)
            ? [...(existingMetadata.errors as unknown[])]
            : [];

          const errorEntry = {
            message: payload.message ?? (payload.data?.['message'] as string | undefined) ?? 'Unknown error',
            timestamp: payload.timestamp,
            traceback: payload.data?.['traceback'],
            recoverable: payload.data?.['recoverable'] as boolean | undefined,
          };

          existingErrors.push(errorEntry);

          store.mergeNodeData(payload.node_id, {
            status: 'failed',
            metadata: {
              ...existingMetadata,
              errors: existingErrors.slice(-20),
              last_error: errorEntry,
            },
          });
        }
      } catch (error) {
        console.error('[Research SSE] Failed to process event', error);
      }
    });

    source.onerror = (error) => {
      console.error('[Research SSE] Error', error);
      useResearchTreeStore.getState().setError('Lost connection to research event stream');
    };

    return () => {
      source.close();
    };
  }, [experimentId]);
}
