import { setCurrentAgentState } from "#/stores/agent-store";
import { setUrl, setScreenshotSrc } from "#/stores/browser-store";
import { ObservationMessage } from "#/types/message";
import { appendOutput } from "#/state/command-store";
import { appendJupyterOutput } from "#/state/jupyter-store";
import ObservationType from "#/types/observation-type";

export function handleObservationMessage(message: ObservationMessage) {
  switch (message.observation) {
    case ObservationType.RUN: {
      if (message.extras.hidden) break;
      let { content } = message;

      if (content.length > 5000) {
        const halfLength = 2500;
        const head = content.slice(0, halfLength);
        const tail = content.slice(content.length - halfLength);
        content = `${head}\r\n\n... (truncated ${message.content.length - 5000} characters) ...\r\n\n${tail}`;
      }

      appendOutput(content);
      break;
    }
    case ObservationType.RUN_IPYTHON:
      appendJupyterOutput({
        content: message.content,
        imageUrls: Array.isArray(message.extras?.image_urls)
          ? message.extras.image_urls
          : undefined,
      });
      break;
    case ObservationType.BROWSE:
    case ObservationType.BROWSE_INTERACTIVE:
      if (message.extras?.screenshot) {
        setScreenshotSrc(message.extras?.screenshot);
      }
      if (message.extras?.url) {
        setUrl(message.extras.url);
      }
      break;
    case ObservationType.AGENT_STATE_CHANGED:
      setCurrentAgentState(message.extras.agent_state);
      break;
    case ObservationType.DELEGATE:
    case ObservationType.READ:
    case ObservationType.EDIT:
    case ObservationType.THINK:
    case ObservationType.NULL:
    case ObservationType.RECALL:
    case ObservationType.ERROR:
    case ObservationType.MCP:
    case ObservationType.TASK_TRACKING:
      break; // We don't display the default message for these observations
    default:
      break;
  }
  if (!message.extras?.hidden) {
    // Convert the message to the appropriate observation type
    const { observation } = message;

    switch (observation) {
      case "browse":
        if (message.extras?.screenshot) {
          setScreenshotSrc(message.extras.screenshot);
        }
        if (message.extras?.url) {
          setUrl(message.extras.url);
        }
        break;
      case "browse_interactive":
        if (message.extras?.screenshot) {
          setScreenshotSrc(message.extras.screenshot);
        }
        if (message.extras?.url) {
          setUrl(message.extras.url);
        }
        break;
      default:
        // For any unhandled observation types, just ignore them
        break;
    }
  }
}
