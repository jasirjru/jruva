import { useEffect, useRef, useState } from 'react';
import {
  DropdownMenu,
  DropdownMenuDivider,
  DropdownMenuItem,
} from '@astryxdesign/core/DropdownMenu';
import { Ellipsis, MessageSquare, Pencil, Pin, PinOff, Trash2 } from 'lucide-react';
import { ChatSession } from '@/hooks/useChatStream';

const MAX_TITLE_LENGTH = 200;
const MENU_ICON_SIZE = 15;

type SessionRowProps = {
  session: ChatSession;
  isCurrent: boolean;
  isPinned: boolean;
  canRename: boolean;
  onSelect: (id: string) => void;
  onTogglePin: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onRequestDelete: (session: ChatSession) => void;
};

/** One conversation in the sidebar history: open it, or pin, rename or delete it from its menu. */
export function SessionRow({
  session,
  isCurrent,
  isPinned,
  canRename,
  onSelect,
  onTogglePin,
  onRename,
  onRequestDelete,
}: SessionRowProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [draftTitle, setDraftTitle] = useState(session.title);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Focus after the menu has closed and returned focus to its trigger.
  useEffect(() => {
    if (!isEditing) return;
    const frame = requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
    return () => cancelAnimationFrame(frame);
  }, [isEditing]);

  const startRename = () => {
    if (!canRename) return;
    setDraftTitle(session.title);
    setIsEditing(true);
  };

  const commitRename = () => {
    const trimmed = draftTitle.trim().slice(0, MAX_TITLE_LENGTH);
    if (trimmed && trimmed !== session.title) onRename(session.id, trimmed);
    setIsEditing(false);
  };

  const RowIcon = isPinned ? Pin : MessageSquare;

  return (
    <div className={`history-chat-entry ${isCurrent ? 'active' : ''}`}>
      {isEditing ? (
        <div className="history-entry-main">
          <RowIcon className="history-entry-bubble" size={14} strokeWidth={1.75} aria-hidden="true" />
          <input
            ref={inputRef}
            className="history-entry-rename-input"
            value={draftTitle}
            maxLength={MAX_TITLE_LENGTH}
            aria-label="Chat name"
            onChange={(e) => setDraftTitle(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') commitRename();
              if (e.key === 'Escape') setIsEditing(false);
            }}
          />
        </div>
      ) : (
        <button
          type="button"
          className="history-entry-main"
          onClick={() => onSelect(session.id)}
          onDoubleClick={startRename}
          aria-current={isCurrent ? 'page' : undefined}
          title={session.title}
        >
          <RowIcon className="history-entry-bubble" size={14} strokeWidth={1.75} aria-hidden="true" />
          <span className="history-entry-title">{session.title}</span>
        </button>
      )}

      <DropdownMenu
        menuWidth={176}
        placement="below"
        alignment="end"
        renderTrigger={(triggerProps) => (
          <button
            type="button"
            {...triggerProps}
            className="history-entry-more-btn"
            aria-label={`Options for ${session.title}`}
          >
            <Ellipsis size={16} strokeWidth={2} aria-hidden="true" />
          </button>
        )}
      >
        <DropdownMenuItem
          icon={isPinned ? <PinOff size={MENU_ICON_SIZE} /> : <Pin size={MENU_ICON_SIZE} />}
          label={isPinned ? 'Unpin' : 'Pin to top'}
          onClick={() => onTogglePin(session.id)}
        />
        {canRename && (
          <DropdownMenuItem icon={<Pencil size={MENU_ICON_SIZE} />} label="Rename" onClick={startRename} />
        )}
        <DropdownMenuDivider />
        <DropdownMenuItem
          icon={<Trash2 size={MENU_ICON_SIZE} />}
          label="Delete"
          variant="destructive"
          onClick={() => onRequestDelete(session)}
        />
      </DropdownMenu>
    </div>
  );
}
