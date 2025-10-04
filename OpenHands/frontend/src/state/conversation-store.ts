import { create } from "zustand";
import { devtools } from "zustand/middleware";

export type ConversationTab =
  | "editor"
  | "browser"
  | "jupyter"
  | "served"
  | "vscode"
  | "terminal";

export interface IMessageToSend {
  text: string;
  timestamp: number;
}

interface ConversationState {
  isRightPanelShown: boolean;
  selectedTab: ConversationTab | null;
  images: File[];
  files: File[];
  loadingFiles: string[]; // File names currently being processed
  loadingImages: string[]; // Image names currently being processed
  messageToSend: IMessageToSend | null;
  shouldShownAgentLoading: boolean;
  submittedMessage: string | null;
  shouldHideSuggestions: boolean; // New state to hide suggestions when input expands
  hasRightPanelToggled: boolean;
}

interface ConversationActions {
  setIsRightPanelShown: (isRightPanelShown: boolean) => void;
  setSelectedTab: (selectedTab: ConversationTab | null) => void;
  setShouldShownAgentLoading: (shouldShownAgentLoading: boolean) => void;
  setShouldHideSuggestions: (shouldHideSuggestions: boolean) => void;
  addImages: (images: File[]) => void;
  addFiles: (files: File[]) => void;
  removeImage: (index: number) => void;
  removeFile: (index: number) => void;
  clearImages: () => void;
  clearFiles: () => void;
  clearAllFiles: () => void;
  addFileLoading: (fileName: string) => void;
  removeFileLoading: (fileName: string) => void;
  addImageLoading: (imageName: string) => void;
  removeImageLoading: (imageName: string) => void;
  clearAllLoading: () => void;
  setMessageToSend: (text: string) => void;
  setSubmittedMessage: (message: string | null) => void;
  resetConversationState: () => void;
  setHasRightPanelToggled: (hasRightPanelToggled: boolean) => void;
}

type ConversationStore = ConversationState & ConversationActions;

// Helper function to get initial right panel state from localStorage
const getInitialRightPanelState = (): boolean => {
  const stored = localStorage.getItem("conversation-right-panel-shown");
  return stored !== null ? JSON.parse(stored) : true;
};

export const useConversationStore = create<ConversationStore>()(
  devtools(
    (set) => ({
      // Initial state
      isRightPanelShown: getInitialRightPanelState(),
      selectedTab: "editor" as ConversationTab,
      images: [],
      files: [],
      loadingFiles: [],
      loadingImages: [],
      messageToSend: null,
      shouldShownAgentLoading: false,
      submittedMessage: null,
      shouldHideSuggestions: false,
      hasRightPanelToggled: true,

      // Actions
      setIsRightPanelShown: (isRightPanelShown) =>
        set({ isRightPanelShown }, false, "setIsRightPanelShown"),

      setSelectedTab: (selectedTab) =>
        set({ selectedTab }, false, "setSelectedTab"),

      setShouldShownAgentLoading: (shouldShownAgentLoading) =>
        set({ shouldShownAgentLoading }, false, "setShouldShownAgentLoading"),

      setShouldHideSuggestions: (shouldHideSuggestions) =>
        set({ shouldHideSuggestions }, false, "setShouldHideSuggestions"),

      addImages: (images) =>
        set(
          (state) => ({ images: [...state.images, ...images] }),
          false,
          "addImages",
        ),

      addFiles: (files) =>
        set(
          (state) => ({ files: [...state.files, ...files] }),
          false,
          "addFiles",
        ),

      removeImage: (index) =>
        set(
          (state) => {
            const newImages = [...state.images];
            newImages.splice(index, 1);
            return { images: newImages };
          },
          false,
          "removeImage",
        ),

      removeFile: (index) =>
        set(
          (state) => {
            const newFiles = [...state.files];
            newFiles.splice(index, 1);
            return { files: newFiles };
          },
          false,
          "removeFile",
        ),

      clearImages: () => set({ images: [] }, false, "clearImages"),

      clearFiles: () => set({ files: [] }, false, "clearFiles"),

      clearAllFiles: () =>
        set(
          {
            images: [],
            files: [],
            loadingFiles: [],
            loadingImages: [],
          },
          false,
          "clearAllFiles",
        ),

      addFileLoading: (fileName) =>
        set(
          (state) => {
            if (!state.loadingFiles.includes(fileName)) {
              return { loadingFiles: [...state.loadingFiles, fileName] };
            }
            return state;
          },
          false,
          "addFileLoading",
        ),

      removeFileLoading: (fileName) =>
        set(
          (state) => ({
            loadingFiles: state.loadingFiles.filter(
              (name) => name !== fileName,
            ),
          }),
          false,
          "removeFileLoading",
        ),

      addImageLoading: (imageName) =>
        set(
          (state) => {
            if (!state.loadingImages.includes(imageName)) {
              return { loadingImages: [...state.loadingImages, imageName] };
            }
            return state;
          },
          false,
          "addImageLoading",
        ),

      removeImageLoading: (imageName) =>
        set(
          (state) => ({
            loadingImages: state.loadingImages.filter(
              (name) => name !== imageName,
            ),
          }),
          false,
          "removeImageLoading",
        ),

      clearAllLoading: () =>
        set({ loadingFiles: [], loadingImages: [] }, false, "clearAllLoading"),

      setMessageToSend: (text) =>
        set(
          {
            messageToSend: {
              text,
              timestamp: Date.now(),
            },
          },
          false,
          "setMessageToSend",
        ),

      setSubmittedMessage: (submittedMessage) =>
        set({ submittedMessage }, false, "setSubmittedMessage"),

      resetConversationState: () =>
        set({ shouldHideSuggestions: false }, false, "resetConversationState"),

      setHasRightPanelToggled: (hasRightPanelToggled) =>
        set({ hasRightPanelToggled }, false, "setHasRightPanelToggled"),
    }),
    {
      name: "conversation-store",
    },
  ),
);

// Compatibility exports for direct calling
export const setIsRightPanelShown = (isRightPanelShown: boolean) => {
  useConversationStore.getState().setIsRightPanelShown(isRightPanelShown);
};

export const setSelectedTab = (selectedTab: ConversationTab | null) => {
  useConversationStore.getState().setSelectedTab(selectedTab);
};

export const setShouldShownAgentLoading = (
  shouldShownAgentLoading: boolean,
) => {
  useConversationStore.getState().setShouldShownAgentLoading(
    shouldShownAgentLoading,
  );
};

export const setShouldHideSuggestions = (shouldHideSuggestions: boolean) => {
  useConversationStore.getState().setShouldHideSuggestions(
    shouldHideSuggestions,
  );
};

export const addImages = (images: File[]) => {
  useConversationStore.getState().addImages(images);
};

export const addFiles = (files: File[]) => {
  useConversationStore.getState().addFiles(files);
};

export const removeImage = (index: number) => {
  useConversationStore.getState().removeImage(index);
};

export const removeFile = (index: number) => {
  useConversationStore.getState().removeFile(index);
};

export const clearImages = () => {
  useConversationStore.getState().clearImages();
};

export const clearFiles = () => {
  useConversationStore.getState().clearFiles();
};

export const clearAllFiles = () => {
  useConversationStore.getState().clearAllFiles();
};

export const addFileLoading = (fileName: string) => {
  useConversationStore.getState().addFileLoading(fileName);
};

export const removeFileLoading = (fileName: string) => {
  useConversationStore.getState().removeFileLoading(fileName);
};

export const addImageLoading = (imageName: string) => {
  useConversationStore.getState().addImageLoading(imageName);
};

export const removeImageLoading = (imageName: string) => {
  useConversationStore.getState().removeImageLoading(imageName);
};

export const clearAllLoading = () => {
  useConversationStore.getState().clearAllLoading();
};

export const setMessageToSend = (text: string) => {
  useConversationStore.getState().setMessageToSend(text);
};

export const setSubmittedMessage = (message: string | null) => {
  useConversationStore.getState().setSubmittedMessage(message);
};

export const resetConversationState = () => {
  useConversationStore.getState().resetConversationState();
};

export const setHasRightPanelToggled = (hasRightPanelToggled: boolean) => {
  useConversationStore.getState().setHasRightPanelToggled(hasRightPanelToggled);
};
