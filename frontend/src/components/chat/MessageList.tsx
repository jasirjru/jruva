'use client';

import React, { useEffect, useRef, useState } from 'react';
import { MessageItem } from './MessageItem';
import { ChatInput } from './ChatInput';
import { JruLogo } from './JruLogo';
import { Message, AttachmentItem } from '@/types/chat';
import { haptics } from '@/utils/haptics';

interface MessageListProps {
  messages: Message[];
  onSend: (message: string, attachments?: AttachmentItem[], webSearch?: boolean) => void;
  onStop: () => void;
  isStreaming: boolean;
  onRegenerate?: () => void;
  onEditAndResend?: (messageId: string, newContent: string) => void;
  onSwitchBranch?: (messageId: string, targetIndex: number) => void;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  onSend,
  onStop,
  isStreaming,
  onRegenerate,
  onEditAndResend,
  onSwitchBranch,
}) => {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const heroRef = useRef<HTMLDivElement | null>(null);
  const [isHeroSpinning, setIsHeroSpinning] = useState(true);

  // Initial in-place rotation on mount, settles after 2.2s
  useEffect(() => {
    const timer = setTimeout(() => setIsHeroSpinning(false), 2200);
    return () => clearTimeout(timer);
  }, []);

  const triggerHeroSpin = () => {
    setIsHeroSpinning(false);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setIsHeroSpinning(true);
        setTimeout(() => setIsHeroSpinning(false), 2200);
      });
    });
  };

  // Auto-scroll as tokens stream in, but only when the user is already near the
  // bottom. Forcing scroll on every token makes earlier turns unreadable mid-stream.
  useEffect(() => {
    const anchor = bottomRef.current;
    if (!anchor) return;
    const scroller = anchor.closest('.jru-workspace-content') as HTMLElement | null;
    if (scroller) {
      const distanceFromBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
      const lastMsg = messages[messages.length - 1];
      const justSentByUser = messages.length >= 2 && messages[messages.length - 2]?.role === 'user' && !lastMsg?.content;
      if (distanceFromBottom > 200 && !justSentByUser) return;
    }
    anchor.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!heroRef.current) return;
    const rect = heroRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const deltaX = (e.clientX - centerX) / (rect.width / 2);
    const deltaY = (e.clientY - centerY) / (rect.height / 2);
    // Tilt via CSS variables on the stage: no React re-render per mouse move.
    // Set here too: mouseenter never fires if the pointer was already inside on mount.
    heroRef.current.setAttribute('data-tilting', '');
    heroRef.current.style.setProperty('--tilt-x', String(Math.max(-1, Math.min(1, deltaX))));
    heroRef.current.style.setProperty('--tilt-y', String(Math.max(-1, Math.min(1, deltaY))));
  };

  const handleMouseEnter = () => {
    heroRef.current?.setAttribute('data-tilting', '');
  };

  const handleMouseLeave = () => {
    heroRef.current?.style.setProperty('--tilt-x', '0');
    heroRef.current?.style.setProperty('--tilt-y', '0');
    heroRef.current?.removeAttribute('data-tilting');
  };

  // Find the last assistant message index for regenerate targeting
  const lastAssistantIdx = messages.findLastIndex((m) => m.role === 'assistant');

  if (messages.length === 0) {
    return (
      <div
        ref={heroRef}
        className="jru-hero-stage"
        onMouseMove={handleMouseMove}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {/* Integrated Brand Emblem with Subtle 3D Tilt */}
        <div
          className="jru-hero-emblem-lockup"
          style={{ transformStyle: 'preserve-3d' }}
        >
          <div style={{ position: 'relative', width: 88, height: 88, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div
              className="jru-icon-orbital-ring"
              style={{
                position: 'absolute',
                inset: -14,
                borderRadius: '50%',
                border: '1px dashed rgba(180, 108, 85, 0.22)',
                pointerEvents: 'none',
                transform: 'translate(calc(var(--tilt-x, 0) * 3px), calc(var(--tilt-y, 0) * 3px))',
                transition: 'transform 100ms ease-out',
              }}
            />
            <div
              className="jru-icon-orbital-ring-reverse"
              style={{
                position: 'absolute',
                inset: -7,
                borderRadius: '50%',
                border: '0.75px solid rgba(180, 108, 85, 0.14)',
                pointerEvents: 'none',
                transform: 'translate(calc(var(--tilt-x, 0) * -3px), calc(var(--tilt-y, 0) * -3px))',
                transition: 'transform 100ms ease-out',
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  top: -2.5,
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: 5,
                  height: 5,
                  borderRadius: '50%',
                  background: 'var(--accent)',
                  boxShadow: '0 0 8px rgba(180, 108, 85, 0.9)',
                }}
              />
            </div>
            <div
              className={`jru-hero-logo-spin-wrap ${isHeroSpinning ? 'spinning' : ''}`}
              onClick={triggerHeroSpin}
              title="Jruva"
              style={{ cursor: 'pointer' }}
            >
              <JruLogo size={88} glow={true} variant="render" animMode="hero" />
            </div>
          </div>
        </div>

        {/* Clean, Minimalist Heading */}
        <h1 className="jru-hero-greeting">Meet Jruva.</h1>

        {/* Hero Input Console */}
        <div className="jru-hero-input-container">
          <ChatInput
            onSend={onSend}
            onStop={onStop}
            isStreaming={isStreaming}
            isHero={true}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="chat-main" role="log" aria-label="Conversation">
      {messages.map((message, idx) => (
        <MessageItem
          key={message.id}
          message={message}
          onRegenerate={idx === lastAssistantIdx && onRegenerate ? onRegenerate : undefined}
          onEditAndResend={message.role === 'user' ? onEditAndResend : undefined}
          onSwitchBranch={onSwitchBranch}
        />
      ))}
      <div ref={bottomRef} style={{ height: '1px' }} />
    </div>
  );
};
