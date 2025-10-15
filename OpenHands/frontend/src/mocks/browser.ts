import { setupWorker } from "msw/browser";
import { handlers as wsHandlers } from "./handlers.ws";
import { handlers } from "./handlers";
import { nodeEventWsHandlers, ENABLE_NODE_EVENT_MOCK } from "./node-event-ws-handlers";

// Conditionally include node event mock handlers
const allHandlers = ENABLE_NODE_EVENT_MOCK
  ? [...handlers, ...wsHandlers, ...nodeEventWsHandlers]
  : [...handlers, ...wsHandlers];

export const worker = setupWorker(...allHandlers);
