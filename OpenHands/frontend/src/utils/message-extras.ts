import { ObservationMessage } from "#/types/message";
import { AgentState } from "#/types/agent-state";

/**
 * Safely extracts a string value from a union type of string | Record<string, unknown>
 * @param value - The value to extract from (can be string or object)
 * @returns The extracted string value or undefined if extraction fails
 */
export function extractStringValue(value: string | Record<string, unknown> | undefined): string | undefined {
  if (typeof value === "string") {
    return value;
  }
  
  if (value && typeof value === "object" && !Array.isArray(value)) {
    // Try to find a string property that might contain the actual value
    const possibleValues = Object.values(value);
    const stringValue = possibleValues.find(v => typeof v === "string");
    if (typeof stringValue === "string") {
      return stringValue;
    }
    
    // If no string property found, try JSON serialization as fallback
    try {
      return JSON.stringify(value);
    } catch {
      return undefined;
    }
  }
  
  return undefined;
}

/**
 * Safely extracts screenshot URL from observation message extras
 * @param extras - The extras object from ObservationMessage
 * @returns Screenshot URL string or undefined if not found/invalid
 */
export function extractScreenshot(extras: ObservationMessage["extras"]): string | undefined {
  return extractStringValue(extras?.screenshot);
}

/**
 * Safely extracts URL from observation message extras
 * @param extras - The extras object from ObservationMessage
 * @returns URL string or undefined if not found/invalid
 */
export function extractUrl(extras: ObservationMessage["extras"]): string | undefined {
  return extractStringValue(extras?.url);
}

/**
 * Safely extracts agent state from observation message extras
 * @param extras - The extras object from ObservationMessage
 * @returns AgentState value or undefined if not found/invalid
 */
export function extractAgentState(extras: ObservationMessage["extras"]): AgentState | undefined {
  const agentStateValue = extractStringValue(extras?.agent_state);
  
  if (!agentStateValue) {
    return undefined;
  }
  
  // Validate that the extracted string is a valid AgentState
  const validStates = Object.values(AgentState);
  if (validStates.includes(agentStateValue as AgentState)) {
    return agentStateValue as AgentState;
  }
  
  return undefined;
}

/**
 * Safely extracts hidden flag from observation message extras
 * @param extras - The extras object from ObservationMessage
 * @returns Boolean value or undefined if not found/invalid
 */
export function extractHidden(extras: ObservationMessage["extras"]): boolean | undefined {
  const hiddenValue = extras?.hidden;
  
  if (typeof hiddenValue === "boolean") {
    return hiddenValue;
  }
  
  if (typeof hiddenValue === "string") {
    return hiddenValue.toLowerCase() === "true";
  }
  
  if (hiddenValue && typeof hiddenValue === "object" && !Array.isArray(hiddenValue)) {
    // Check if object has a boolean property
    const booleanValue = Object.values(hiddenValue).find(v => typeof v === "boolean");
    if (typeof booleanValue === "boolean") {
      return booleanValue;
    }
  }
  
  return undefined;
}

/**
 * Safely extracts image URLs array from observation message extras
 * @param extras - The extras object from ObservationMessage
 * @returns Array of image URL strings or undefined if not found/invalid
 */
export function extractImageUrls(extras: ObservationMessage["extras"]): string[] | undefined {
  const imageUrls = extras?.image_urls;
  
  if (Array.isArray(imageUrls) && imageUrls.every(url => typeof url === "string")) {
    return imageUrls;
  }
  
  return undefined;
}