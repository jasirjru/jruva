'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { Pencil } from 'lucide-react';
import { Message } from '@/types/chat';
import '@/styles/answer.css';
import { MarkdownRenderer } from './MarkdownRenderer';
import { SourcesTray } from './SourcesTray';
import { JruLogo } from './JruLogo';
import { ThinkingAccordion } from './ThinkingAccordion';
import { AttachmentIcon, formatFileSize } from './AttachmentIcon';
import { AssistantActions } from './MessageActions';


interface MessageItemProps {
  message: Message;
  onRegenerate?: () => void;
  onEditAndResend?: (messageId: string, newContent: string) => void;
  onSwitchBranch?: (messageId: string, targetIndex: number) => void;
}

export const MessageItem: React.FC<MessageItemProps> = ({ message, onRegenerate, onEditAndResend, onSwitchBranch }) => {
  const isUser = message.role === 'user';
  const isStreaming = message.status === 'streaming';
  const isError = message.status === 'error';
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);

  const editTextareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-focus and resize edit textarea
  useEffect(() => {
    if (isEditing && editTextareaRef.current) {
      editTextareaRef.current.focus();
      editTextareaRef.current.style.height = 'auto';
      editTextareaRef.current.style.height = `${editTextareaRef.current.scrollHeight}px`;
    }
  }, [isEditing]);

  const formatTime = (date: Date) => {
    try {
      const d = date instanceof Date ? date : new Date(date);
      return new Intl.DateTimeFormat('default', {
        hour: '2-digit',
        minute: '2-digit',
      }).format(d);
    } catch {
      return '';
    }
  };

  const handleStartEdit = () => {
    setEditContent(message.content);
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditContent(message.content);
  };

  const handleSubmitEdit = () => {
    const trimmed = editContent.trim();
    if (trimmed && onEditAndResend) {
      setIsEditing(false);
      onEditAndResend(message.id, trimmed);
    }
  };

  const handleEditKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmitEdit();
    }
    if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };

  return (
    <div
      className={`message-wrapper ${message.role} ${isUser ? 'jru-message-user-animate-in' : 'jru-message-assistant-animate-in'}`}
    >
      {!isUser && (
        <div className="message-meta">
          <JruLogo size={16} animMode="chat" className={`ai-avatar-logo ${isStreaming ? 'streaming' : ''}`.trim()} />
          <span className="message-sender-jru">Jruva</span>
          {formatTime(message.createdAt) && (
            <time className="message-time">{formatTime(message.createdAt)}</time>
          )}
        </div>
      )}

      <div className={`message-bubble ${isError ? 'error' : ''}`}>
        {/* Display attachments if present */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="message-attachments-grid">
            {message.attachments.map((att, idx) => {
              const isImage = att.mime_type.startsWith('image/');
              const src = att.previewUrl || (att.data ? `data:${att.mime_type};base64,${att.data}` : null);

              if (isImage && src) {
                return (
                  <div key={idx} className="msg-attachment-image-card">
                    <img src={src} alt={att.name} className="msg-attachment-img" loading="lazy" />
                    <span className="msg-attachment-name">{att.name}</span>
                  </div>
                );
              }

              return (
                <div key={idx} className="msg-attachment-file-card">
                  <span className="msg-file-icon">
                    <AttachmentIcon name={att.name} mimeType={att.mime_type} size={18} />
                  </span>
                  <div className="msg-file-info">
                    <span className="msg-file-name">{att.name}</span>
                    {att.size ? <span className="msg-file-size">{formatFileSize(att.size)}</span> : null}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* User message: show edit mode or static content */}
        {isUser ? (
          isEditing ? (
            <div className="msg-edit-container">
              <textarea
                ref={editTextareaRef}
                className="msg-edit-textarea"
                value={editContent}
                onChange={(e) => {
                  setEditContent(e.target.value);
                  e.target.style.height = 'auto';
                  e.target.style.height = `${e.target.scrollHeight}px`;
                }}
                onKeyDown={handleEditKeyDown}
                rows={2}
              />
              <div className="msg-edit-actions">
                <button
                  type="button"
                  className="msg-edit-submit-btn"
                  onClick={handleSubmitEdit}
                  disabled={!editContent.trim()}
                >
                  Send
                </button>
                <button
                  type="button"
                  className="msg-edit-cancel-btn"
                  onClick={handleCancelEdit}
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div style={{ whiteSpace: 'pre-wrap' }}>{message.content}</div>
          )
        ) : (
          <div>
            {!isUser && message.content && (
              message.content.includes('OCR EXTRACTED') ||
              message.content.includes('Scanned Grade Card') ||
              message.content.includes('Scanned Fee Receipt') ||
              message.content.includes('Grade Point Calculation') ||
              message.content.includes('Attendance Eligibility Assessment') ||
              message.content.includes('Examination Schedule & Hall')
            ) && (
              <div className="jru-verified-document-badge">
                <span className="jru-verified-dot" />
                <span>AI-generated analysis • check against your official records</span>
              </div>
            )}
            {/* Thinking Accordion (Deep Reasoning CoT with live stopwatch & pulsing brain) */}
            {message.thought ? (
              <ThinkingAccordion
                thought={message.thought}
                isStreaming={isStreaming}
                isThinkingActive={message.isThinkingActive}
              />
            ) : null}

            {isStreaming && !message.content && !message.thought && (
              <div className="processing-indicator">
                <div className="processing-spinner" />
                <span className="processing-text">{message.statusText || 'Thinking...'}</span>
              </div>
            )}
            {message.content ? (
              <MarkdownRenderer content={message.content} isStreaming={isStreaming} />
            ) : null}
            {isError && message.errorMessage && message.errorMessage !== message.content ? (
              <p className="message-inline-error" role="alert">{message.errorMessage}</p>
            ) : null}
            {isStreaming && message.content && <span className="typing-cursor" aria-hidden="true" />}
          </div>
        )}

        {/* Display live web search citations if present */}
        {message.sources && message.sources.length > 0 && (
          <SourcesTray sources={message.sources} />
        )}
      </div>

      {/* Action Strip for USER Messages */}
      {isUser && !isStreaming && message.content && !isEditing && onEditAndResend && (
        <div className="message-actions-strip">
          <Tooltip content="Edit message" placement="below">
            <button type="button" onClick={handleStartEdit} className="msg-action-btn" aria-label="Edit message">
              <Pencil size={15} strokeWidth={1.75} aria-hidden="true" />
            </button>
          </Tooltip>
        </div>
      )}

      {/* Action Strip for ASSISTANT Messages */}
      {!isUser && !isStreaming && message.content && (
        <AssistantActions message={message} onRegenerate={onRegenerate} onSwitchBranch={onSwitchBranch} />
      )}
    </div>
  );
};
