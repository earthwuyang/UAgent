import { useEffect } from 'react';

export interface KeyboardShortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  callback: () => void;
  description?: string;
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[], enabled: boolean = true) {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      const key = event.key.toLowerCase();

      // Skip shortcuts when typing in input fields, except for Escape
      if (
        key !== 'escape' &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.contentEditable === 'true')
      ) {
        return;
      }

      const ctrl = event.ctrlKey;
      const shift = event.shiftKey;
      const alt = event.altKey;
      const meta = event.metaKey;

      for (const shortcut of shortcuts) {
        if (
          key === shortcut.key.toLowerCase() &&
          ctrl === !!shortcut.ctrl &&
          shift === !!shortcut.shift &&
          alt === !!shortcut.alt &&
          meta === !!shortcut.meta
        ) {
          event.preventDefault();
          shortcut.callback();
          break;
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [shortcuts, enabled]);
}