/**
 * Hybrid Redux + Zustand store configuration
 *
 * Some parts of the app use Zustand (newer), some use Redux (legacy)
 * This file provides Redux store for compatibility while Zustand stores
 * coexist in /stores/ and /state/ directories
 */

import { combineReducers, configureStore, createSlice } from "@reduxjs/toolkit";

// Re-export Zustand store hooks for convenience
export { useAgentStore } from "./stores/agent-store";
export { useBrowserStore } from "./stores/browser-store";
export { useInitialQueryStore } from "./stores/initial-query-store";
export { useCommandStore } from "./state/command-store";
export { useJupyterStore } from "./state/jupyter-store";
export { useSecurityAnalyzerStore } from "./stores/security-analyzer-store";
export { useStatusStore } from "./state/status-store";
export { useMetricsStore } from "./stores/metrics-store";
export { useMicroagentManagementStore } from "./state/microagent-management-store";
export { useConversationStore } from "./state/conversation-store";
export { useEventMessageStore } from "./stores/event-message-store";

// Create minimal Redux slices for legacy components
// These provide valid initial state so useSelector doesn't crash
// The actual state management happens in Zustand stores
const agentSlice = createSlice({
  name: "agent",
  initialState: { curAgentState: "LOADING" },
  reducers: {},
});

const statusSlice = createSlice({
  name: "status",
  initialState: { curStatusMessage: null },
  reducers: {},
});

const commandSlice = createSlice({
  name: "cmd",
  initialState: { commands: [] },
  reducers: {},
});

const browserSlice = createSlice({
  name: "browser",
  initialState: { url: "", screenshotSrc: "" },
  reducers: {},
});

const jupyterSlice = createSlice({
  name: "jupyter",
  initialState: { cells: [] },
  reducers: {},
});

const metricsSlice = createSlice({
  name: "metrics",
  initialState: { cost: null, usage: null, max_budget_per_task: null },
  reducers: {},
});

const securityAnalyzerSlice = createSlice({
  name: "securityAnalyzer",
  initialState: { logs: [] },
  reducers: {},
});

const microagentManagementSlice = createSlice({
  name: "microagentManagement",
  initialState: {
    addMicroagentModalVisible: false,
    updateMicroagentModalVisible: false,
    learnThisRepoModalVisible: false,
    selectedRepository: null,
    personalRepositories: [],
    organizationRepositories: [],
    repositories: [],
    selectedMicroagentItem: null,
  },
  reducers: {},
});

const conversationSlice = createSlice({
  name: "conversation",
  initialState: {
    isRightPanelShown: true,
    selectedTab: "editor",
    shouldShownAgentLoading: false,
    shouldHideSuggestions: false,
    images: [],
    files: [],
    loadingFiles: [],
    loadingImages: [],
    messageToSend: null,
    submittedMessage: null,
    hasRightPanelToggled: true,
  },
  reducers: {},
});

const eventMessageSlice = createSlice({
  name: "eventMessage",
  initialState: { submittedEventIds: [] },
  reducers: {},
});

const initialQuerySlice = createSlice({
  name: "initialQuery",
  initialState: {
    files: [],
    selectedRepository: null,
    initialPrompt: "",
    selectedRepositoryProvider: null,
    replayJson: null,
  },
  reducers: {},
});

export const rootReducer = combineReducers({
  agent: agentSlice.reducer,
  status: statusSlice.reducer,
  cmd: commandSlice.reducer,
  browser: browserSlice.reducer,
  jupyter: jupyterSlice.reducer,
  metrics: metricsSlice.reducer,
  securityAnalyzer: securityAnalyzerSlice.reducer,
  microagentManagement: microagentManagementSlice.reducer,
  conversation: conversationSlice.reducer,
  eventMessage: eventMessageSlice.reducer,
  initialQuery: initialQuerySlice.reducer,
});

const store = configureStore({
  reducer: rootReducer,
});

// NOTE: Redux and Zustand stores are separate
// Redux is used by legacy components with useSelector
// Zustand is used by newer components and action functions
// They don't need to be synced - each manages its own state

export type RootState = ReturnType<typeof store.getState>;
export type AppStore = typeof store;
export type AppDispatch = typeof store.dispatch;

export default store;
