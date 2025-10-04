import { create } from "zustand";

export type Cell = {
  content: string;
  type: "input" | "output";
  imageUrls?: string[];
};

interface JupyterState {
  cells: Cell[];
  appendJupyterInput: (content: string) => void;
  appendJupyterOutput: (payload: {
    content: string;
    imageUrls?: string[];
  }) => void;
  clearJupyter: () => void;
}

export const useJupyterStore = create<JupyterState>((set) => ({
  cells: [],
  appendJupyterInput: (content: string) =>
    set((state) => ({
      cells: [...state.cells, { content, type: "input" }],
    })),
  appendJupyterOutput: (payload: { content: string; imageUrls?: string[] }) =>
    set((state) => ({
      cells: [
        ...state.cells,
        {
          content: payload.content,
          type: "output",
          imageUrls: payload.imageUrls,
        },
      ],
    })),
  clearJupyter: () =>
    set(() => ({
      cells: [],
    })),
}));

// Compatibility exports for direct calling
export const appendJupyterInput = (content: string) => {
  useJupyterStore.getState().appendJupyterInput(content);
};

export const appendJupyterOutput = (payload: {
  content: string;
  imageUrls?: string[];
}) => {
  useJupyterStore.getState().appendJupyterOutput(payload);
};

export const clearJupyter = () => {
  useJupyterStore.getState().clearJupyter();
};
