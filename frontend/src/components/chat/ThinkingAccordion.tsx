'use client';

import React, { useEffect, useId, useRef, useState } from 'react';
import { Brain, ChevronDown } from 'lucide-react';

interface ThinkingAccordionProps {
  thought: string;
  isStreaming?: boolean;
  isThinkingActive?: boolean;
}

const TICK_MS = 1000;

function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

/** Collapsible reasoning shown above an answer; open while thinking, folds away when the answer starts. */
export const ThinkingAccordion: React.FC<ThinkingAccordionProps> = ({ thought, isThinkingActive = false }) => {
  const [isOpen, setIsOpen] = useState<boolean>(isThinkingActive);
  const [userToggled, setUserToggled] = useState<boolean>(false);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const startTimeRef = useRef<number | null>(null);
  const wasActiveRef = useRef<boolean>(isThinkingActive);
  const bodyId = useId();

  // Whole-second stopwatch: one render per second instead of ten
  useEffect(() => {
    if (!isThinkingActive) return;
    if (!startTimeRef.current) startTimeRef.current = Date.now();
    const timer = setInterval(() => {
      if (startTimeRef.current) setElapsedSeconds(Math.round((Date.now() - startTimeRef.current) / 1000));
    }, TICK_MS);
    return () => clearInterval(timer);
  }, [isThinkingActive]);

  // Open when thinking starts; fold away when it ends, unless the user chose otherwise
  useEffect(() => {
    if (isThinkingActive && !userToggled) {
      setIsOpen(true);
    } else if (!isThinkingActive && wasActiveRef.current && !userToggled) {
      setIsOpen(false);
    }
    wasActiveRef.current = isThinkingActive;
  }, [isThinkingActive, userToggled]);

  const handleToggle = () => {
    setUserToggled(true);
    setIsOpen((prev) => !prev);
  };

  const timeText = elapsedSeconds > 0 ? formatSeconds(elapsedSeconds) : '';
  const label = isThinkingActive
    ? `Thinking${timeText ? ` · ${timeText}` : '…'}`
    : timeText
      ? `Thought for ${timeText}`
      : 'Reasoning';

  return (
    <div className={`thinking-accordion ${isThinkingActive ? 'active-thinking' : 'completed-thinking'}`}>
      <button
        type="button"
        className="thinking-header"
        onClick={handleToggle}
        aria-expanded={isOpen}
        aria-controls={bodyId}
      >
        <span className="thinking-header-left">
          <Brain
            className={`thinking-brain-svg ${isThinkingActive ? 'pulse' : ''}`}
            size={15}
            strokeWidth={1.75}
            aria-hidden="true"
          />
          <span className="thinking-title">{label}</span>
        </span>
        <ChevronDown className={`thinking-chevron ${isOpen ? 'open' : ''}`} size={14} strokeWidth={2} aria-hidden="true" />
      </button>

      {isOpen && (
        <div className="thinking-body" id={bodyId}>
          <div className="thinking-content">
            {thought}
            {isThinkingActive && <span className="thinking-stream-dot" />}
          </div>
        </div>
      )}
    </div>
  );
};
