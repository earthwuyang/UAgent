import { useSelector } from "react-redux";
import { RootState } from "#/store";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { useActiveConversation } from "./query/use-active-conversation";

/**
 * Hook to determine if the runtime is ready for operations
 *
 * @returns boolean indicating if the runtime is ready
 */
export const useRuntimeIsReady = (): boolean => {
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useSelector((state: RootState) => state.agent);

  // Runtime is ready if:
  // 1. Conversation runtime_status is STATUS$READY (backend confirms runtime is ready)
  // 2. OR agent state is not in inactive states AND conversation status is RUNNING
  const backendRuntimeReady = conversation?.runtime_status === "STATUS$READY";
  const agentAvailable = conversation?.status === "RUNNING" && !RUNTIME_INACTIVE_STATES.includes(curAgentState);

  return backendRuntimeReady || agentAvailable;
};
