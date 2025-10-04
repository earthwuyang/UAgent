import { create } from "zustand";
import { Provider } from "#/types/settings";
import { GitRepository } from "#/types/git";

interface InitialQueryState {
  files: string[]; // base64 encoded images
  initialPrompt: string | null;
  selectedRepository: GitRepository | null;
  selectedRepositoryProvider: Provider | null;
  replayJson: string | null;
}

interface InitialQueryActions {
  addFile: (file: string) => void;
  removeFile: (index: number) => void;
  clearFiles: () => void;
  setInitialPrompt: (prompt: string) => void;
  clearInitialPrompt: () => void;
  setSelectedRepository: (repository: GitRepository | null) => void;
  clearSelectedRepository: () => void;
  setSelectedRepositoryProvider: (provider: Provider | null) => void;
  setReplayJson: (replayJson: string | null) => void;
  reset: () => void;
}

type InitialQueryStore = InitialQueryState & InitialQueryActions;

const initialState: InitialQueryState = {
  files: [],
  initialPrompt: null,
  selectedRepository: null,
  selectedRepositoryProvider: null,
  replayJson: null,
};

export const useInitialQueryStore = create<InitialQueryStore>((set) => ({
  ...initialState,

  addFile: (file: string) =>
    set((state) => ({
      files: [...state.files, file],
    })),

  removeFile: (index: number) =>
    set((state) => ({
      files: state.files.filter((_, i) => i !== index),
    })),

  clearFiles: () =>
    set(() => ({
      files: [],
    })),

  setInitialPrompt: (prompt: string) =>
    set(() => ({
      initialPrompt: prompt,
    })),

  clearInitialPrompt: () =>
    set(() => ({
      initialPrompt: null,
    })),

  setSelectedRepository: (repository: GitRepository | null) =>
    set(() => ({
      selectedRepository: repository,
    })),

  clearSelectedRepository: () =>
    set(() => ({
      selectedRepository: null,
    })),

  setSelectedRepositoryProvider: (provider: Provider | null) =>
    set(() => ({
      selectedRepositoryProvider: provider,
    })),

  setReplayJson: (replayJson: string | null) =>
    set(() => ({
      replayJson,
    })),

  reset: () => set(() => initialState),
}));

// Compatibility exports for direct calling
export const addFile = (file: string) => {
  useInitialQueryStore.getState().addFile(file);
};

export const removeFile = (index: number) => {
  useInitialQueryStore.getState().removeFile(index);
};

export const clearFiles = () => {
  useInitialQueryStore.getState().clearFiles();
};

export const setInitialPrompt = (prompt: string) => {
  useInitialQueryStore.getState().setInitialPrompt(prompt);
};

export const clearInitialPrompt = () => {
  useInitialQueryStore.getState().clearInitialPrompt();
};

export const setInitialQuerySelectedRepository = (
  repository: GitRepository | null,
) => {
  useInitialQueryStore.getState().setSelectedRepository(repository);
};

export const clearInitialQuerySelectedRepository = () => {
  useInitialQueryStore.getState().clearSelectedRepository();
};

export const setSelectedRepositoryProvider = (provider: Provider | null) => {
  useInitialQueryStore.getState().setSelectedRepositoryProvider(provider);
};

export const setReplayJson = (replayJson: string | null) => {
  useInitialQueryStore.getState().setReplayJson(replayJson);
};

export const resetInitialQueryStore = () => {
  useInitialQueryStore.getState().reset();
};
