/**
 * Mock WebSocket Handler for Node Events
 * 
 * This mock simulates the backend WebSocket endpoint for node event streaming
 * during development when the backend endpoint is not yet implemented.
 * 
 * Usage: Enable in development mode only
 * Backend endpoint: ws://localhost:2999/ws/research/{experimentId}
 */

import { WebSocketHandler, ws } from 'msw';

interface SubscriptionMessage {
  type: 'subscribe' | 'unsubscribe';
  node_id: string;
  experiment_id: string;
}

interface NodeEventMessage {
  type: 'node_event' | 'subscription_confirmed' | 'error';
  data: {
    id: string;
    node_id: string;
    experiment_id: string;
    event_type: string;
    message?: string;
    content?: Record<string, unknown>;
    timestamp: string;
  };
}

// Mock event types to simulate
const mockEventTypes = [
  'task_start',
  'observation',
  'action',
  'task_complete',
  'message',
];

let eventCounter = 0;

/**
 * Generate a mock node event
 */
function generateMockEvent(nodeId: string, experimentId: string): NodeEventMessage {
  const eventType = mockEventTypes[eventCounter % mockEventTypes.length];
  eventCounter++;

  return {
    type: 'node_event',
    data: {
      id: `mock-event-${Date.now()}-${eventCounter}`,
      node_id: nodeId,
      experiment_id: experimentId,
      event_type: eventType,
      message: `Mock ${eventType} event for node ${nodeId}`,
      content: {
        mock: true,
        eventNumber: eventCounter,
        timestamp: new Date().toISOString(),
      },
      timestamp: new Date().toISOString(),
    },
  };
}

/**
 * Mock WebSocket handler for node events
 * Matches URL pattern: ws://{host}/ws/research/{experimentId}
 */
export const nodeEventWsHandlers: WebSocketHandler[] = [
  ws.link('ws://*/ws/research/:experimentId', {
    connect({ params, client }) {
      const experimentId = params.experimentId as string;
      console.log(`[Mock NodeEventWS] Client connected to experiment: ${experimentId}`);

      // Store active subscriptions
      const subscriptions = new Map<string, NodeJS.Timeout>();

      // Handle incoming messages
      client.addEventListener('message', (event) => {
        try {
          const message: SubscriptionMessage = JSON.parse(event.data as string);
          
          if (message.type === 'subscribe') {
            console.log(`[Mock NodeEventWS] Subscribe to node: ${message.node_id}`);
            
            // Send confirmation
            client.send(JSON.stringify({
              type: 'subscription_confirmed',
              data: {
                node_id: message.node_id,
                message: `Subscribed to node ${message.node_id}`,
              },
            }));

            // Start sending mock events every 3 seconds
            const interval = setInterval(() => {
              const mockEvent = generateMockEvent(message.node_id, message.experiment_id);
              client.send(JSON.stringify(mockEvent));
              console.log(`[Mock NodeEventWS] Sent event for node ${message.node_id}:`, mockEvent.data.event_type);
            }, 3000);

            subscriptions.set(message.node_id, interval);
          } 
          else if (message.type === 'unsubscribe') {
            console.log(`[Mock NodeEventWS] Unsubscribe from node: ${message.node_id}`);
            
            // Clear interval
            const interval = subscriptions.get(message.node_id);
            if (interval) {
              clearInterval(interval);
              subscriptions.delete(message.node_id);
            }
          }
        } catch (error) {
          console.error('[Mock NodeEventWS] Error parsing message:', error);
        }
      });

      // Cleanup on disconnect
      client.addEventListener('close', () => {
        console.log(`[Mock NodeEventWS] Client disconnected from experiment: ${experimentId}`);
        // Clear all intervals
        subscriptions.forEach(interval => clearInterval(interval));
        subscriptions.clear();
      });
    },
  }),
];

/**
 * Development-only flag to enable mock WebSocket
 * Set to true to use mock instead of real backend endpoint
 */
export const ENABLE_NODE_EVENT_MOCK = import.meta.env.DEV && 
  import.meta.env.VITE_MOCK_NODE_EVENTS === 'true';
