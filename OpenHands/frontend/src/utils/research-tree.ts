export type ResearchNodeMetadata = Record<string, unknown> | undefined | null;

const CANDIDATE_KEYS = [
  "conversation_id",
  "conversationId",
  "session_id",
  "sessionId",
  "run_conversation_id",
  "runConversationId",
];

const NESTED_KEYS = ["conversation", "session", "run"];

export function extractConversationId(metadata: ResearchNodeMetadata): string | null {
  if (!metadata || typeof metadata !== "object") {
    return null;
  }

  for (const key of CANDIDATE_KEYS) {
    const value = (metadata as Record<string, unknown>)[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }

  for (const containerKey of NESTED_KEYS) {
    const container = (metadata as Record<string, unknown>)[containerKey];
    if (container && typeof container === "object") {
      for (const key of CANDIDATE_KEYS.concat(["id", "ID"])) {
        const value = (container as Record<string, unknown>)[key];
        if (typeof value === "string" && value.trim()) {
          return value.trim();
        }
      }
    }
  }

  const branch = (metadata as Record<string, unknown>)["branch_id"];
  if (typeof branch === "string" && branch.trim()) {
    return branch.trim();
  }

  return null;
}
