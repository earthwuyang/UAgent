import { create } from "zustand";

export type Command = {
  content: string;
  type: "input" | "output";
};

interface CommandState {
  commands: Command[];
  appendInput: (content: string) => void;
  appendOutput: (content: string) => void;
  clearTerminal: () => void;
}

export const useCommandStore = create<CommandState>((set) => ({
  commands: [],
  appendInput: (content: string) =>
    set((state) => ({
      commands: [...state.commands, { content, type: "input" }],
    })),
  appendOutput: (content: string) =>
    set((state) => ({
      commands: [...state.commands, { content, type: "output" }],
    })),
  clearTerminal: () => set({ commands: [] }),
}));

// Compatibility exports for Redux-style usage
export const appendInput = (content: string) => {
  useCommandStore.getState().appendInput(content);
};

export const appendOutput = (content: string) => {
  useCommandStore.getState().appendOutput(content);
};

export const clearTerminal = () => {
  useCommandStore.getState().clearTerminal();
};
