'use client';

import { useState, useEffect, useCallback } from 'react';
import { triggerWaterFlowTransition } from '@/utils/waterFlowTransition';

export type Theme = 'dark' | 'light';

export function useTheme() {
  const [theme, setTheme] = useState<Theme>('light');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    try {
      const savedTheme = localStorage.getItem('jru_theme') as Theme | null;
      if (savedTheme === 'light' || savedTheme === 'dark') {
        setTheme(savedTheme);
        document.documentElement.setAttribute('data-theme', savedTheme);
      } else {
        setTheme('light');
        document.documentElement.setAttribute('data-theme', 'light');
      }
    } catch {
      // Fallback to light
      document.documentElement.setAttribute('data-theme', 'light');
    }
    setMounted(true);
  }, []);

  const toggleTheme = useCallback(
    (event?: React.MouseEvent | MouseEvent) => {
      // Determine origin coordinates for water flood wave surge
      let x = typeof window !== 'undefined' ? window.innerWidth / 2 : 0;
      let y = typeof window !== 'undefined' ? window.innerHeight / 2 : 0;

      if (event && 'clientX' in event && typeof event.clientX === 'number' && event.clientX > 0) {
        x = event.clientX;
        y = event.clientY;
      } else if (typeof document !== 'undefined') {
        const activeEl = document.activeElement;
        if (activeEl && typeof activeEl.getBoundingClientRect === 'function') {
          const rect = activeEl.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0) {
            x = rect.left + rect.width / 2;
            y = rect.top + rect.height / 2;
          }
        }
      }

      const next: Theme = theme === 'dark' ? 'light' : 'dark';

      try {
        localStorage.setItem('jru_theme', next);
      } catch {
        // Storage restricted
      }

      document.documentElement.classList.add('theme-transitioning');

      // Trigger Cinematic Water Flow / Flood Wave Transition
      triggerWaterFlowTransition(
        x,
        y,
        next,
        () => {
          document.documentElement.setAttribute('data-theme', next);
          setTheme(next);
        },
        () => {
          document.documentElement.classList.remove('theme-transitioning');
        }
      );
    },
    [theme]
  );

  return { theme, toggleTheme, mounted };
}
