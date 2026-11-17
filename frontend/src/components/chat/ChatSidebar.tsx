'use client';

import React, { useState, useEffect, useRef } from 'react';
import { AlertDialog } from '@astryxdesign/core/AlertDialog';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import {
  BookOpen,
  Bookmark,
  ChevronDown,
  FolderOpen,
  History,
  Image as ImageIcon,
  MessageSquare,
  PanelLeft,
  Puzzle,
  Search,
  SearchX,
  SquarePen,
  X,
} from 'lucide-react';
import { JruLogo } from './JruLogo';
import { SessionRow } from './SessionRow';
import { ChatSession } from '@/hooks/useChatStream';
import { Theme } from '@/hooks/useTheme';
import { WorkspaceView } from '@/types/workspace';
import { useMemory } from '@/context/MemoryContext';
import { useAuth } from '@/context/AuthContext';

interface ChatSidebarProps {
  /** null before mount: CSS decides the first paint by screen width. */
  isOpen: boolean | null;
  onToggle: () => void;
  sessions: ChatSession[];
  currentSessionId: string;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession?: (id: string, title: string) => void;
  theme: Theme;
  onToggleTheme: (event?: React.MouseEvent) => void;
  activeView?: WorkspaceView;
  onSelectView?: (view: WorkspaceView) => void;
  onOpenCommandPalette?: () => void;
}

interface SessionGroup {
  label: string;
  items: ChatSession[];
}

function groupSessionsByDate(sessions: ChatSession[]): SessionGroup[] {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterdayStart = todayStart - 86400000;
  const last7DaysStart = todayStart - 7 * 86400000;
  const last30DaysStart = todayStart - 30 * 86400000;

  const groups: SessionGroup[] = [
    { label: 'Today', items: [] },
    { label: 'Yesterday', items: [] },
    { label: 'Previous 7 Days', items: [] },
    { label: 'Previous 30 Days', items: [] },
    { label: 'Older', items: [] },
  ];

  for (const session of sessions) {
    const time =
      typeof session.updatedAt === 'string'
        ? new Date(session.updatedAt).getTime()
        : session.updatedAt || 0;

    if (!time || isNaN(time)) {
      groups[0].items.push(session);
    } else if (time >= todayStart) {
      groups[0].items.push(session);
    } else if (time >= yesterdayStart) {
      groups[1].items.push(session);
    } else if (time >= last7DaysStart) {
      groups[2].items.push(session);
    } else if (time >= last30DaysStart) {
      groups[3].items.push(session);
    } else {
      groups[4].items.push(session);
    }
  }

  return groups.filter((g) => g.items.length > 0);
}

const HISTORY_OPEN_KEY = 'jru_history_open';
const ICON_SIZE = 17;
const ICON_STROKE = 1.75;
const MOBILE_BREAKPOINT = 768;

export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  isOpen,
  onToggle,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onRenameSession,
  activeView = 'chat',
  onSelectView,
  onOpenCommandPalette,
}) => {
  const { memories, openMemoryModal } = useMemory();
  const { user } = useAuth();
  // Recent chats are visible by default; hiding them is remembered.
  const [isHistoryOpen, setIsHistoryOpen] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [mounted, setMounted] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<ChatSession | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  const accountName = user?.full_name || user?.username || 'Guest';
  const accountDetail = user ? user.email || 'Signed in' : 'Not signed in';
  const pinnedStorageKey = user?.id ? `jru_pinned_sessions_${user.id}` : 'jru_pinned_sessions_guest';
  const [pinnedIds, setPinnedIds] = useState<string[]>([]);

  useEffect(() => {
    setMounted(true);
    try {
      if (localStorage.getItem(HISTORY_OPEN_KEY) === 'false') setIsHistoryOpen(false);
    } catch {
      // Storage blocked: keep the default
    }
  }, []);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(pinnedStorageKey) || localStorage.getItem('jru_pinned_sessions');
      setPinnedIds(stored ? JSON.parse(stored) : []);
    } catch {
      setPinnedIds([]);
    }
  }, [pinnedStorageKey]);

  // Auto-close the drawer on mobile after navigating
  const handleNavClick = (action: () => void) => {
    action();
    if (typeof window !== 'undefined' && window.innerWidth < MOBILE_BREAKPOINT) {
      onToggle();
    }
  };

  const toggleHistory = () => {
    const next = !isHistoryOpen;
    setIsHistoryOpen(next);
    try {
      localStorage.setItem(HISTORY_OPEN_KEY, String(next));
    } catch {
      // Storage blocked: the choice lasts for this visit only
    }
  };

  const togglePin = (id: string) => {
    setPinnedIds((prev) => {
      const next = prev.includes(id) ? prev.filter((p) => p !== id) : [...prev, id];
      try {
        localStorage.setItem(pinnedStorageKey, JSON.stringify(next));
      } catch {
        // Ignore localStorage quota/permission errors
      }
      return next;
    });
  };

  const openSearch = () => {
    if (onOpenCommandPalette) {
      onOpenCommandPalette();
      return;
    }
    setIsHistoryOpen(true);
    setTimeout(() => searchInputRef.current?.focus(), 120);
  };

  const query = searchQuery.trim().toLowerCase();
  const filteredSessions = query ? sessions.filter((s) => s.title.toLowerCase().includes(query)) : sessions;
  const pinnedSessions = filteredSessions.filter((s) => pinnedIds.includes(s.id));
  const unpinnedSessions = filteredSessions.filter((s) => !pinnedIds.includes(s.id));
  const groupedSessions = groupSessionsByDate(unpinnedSessions);

  const renderSessionItem = (session: ChatSession) => (
    <SessionRow
      key={session.id}
      session={session}
      isCurrent={mounted && session.id === currentSessionId}
      isPinned={pinnedIds.includes(session.id)}
      canRename={Boolean(onRenameSession)}
      onSelect={(id) =>
        handleNavClick(() => {
          onSelectView?.('chat');
          onSelectSession(id);
        })
      }
      onTogglePin={togglePin}
      onRename={(id, title) => onRenameSession?.(id, title)}
      onRequestDelete={setPendingDelete}
    />
  );

  const navItems: { view: WorkspaceView; label: string; Icon: typeof ImageIcon }[] = [
    { view: 'images', label: 'Images', Icon: ImageIcon },
    { view: 'library', label: 'Library', Icon: BookOpen },
    { view: 'plugins', label: 'Plugins', Icon: Puzzle },
    { view: 'projects', label: 'Projects', Icon: FolderOpen },
  ];

  return (
    <aside
      className={`jru-workspace-sidebar ${isOpen === null ? 'pending' : isOpen ? 'open' : 'collapsed'}`}
      aria-label="Sidebar"
      aria-hidden={isOpen === false}
    >
      {/* Right-edge pull handle */}
      <div
        className="sidebar-drag-edge"
        title="Toggle sidebar"
        onClick={(e) => {
          e.stopPropagation();
          onToggle();
        }}
      >
        <div className="sidebar-drag-pill" />
      </div>

      <div className="sidebar-inner-content" inert={isOpen === false}>
        <div className="sidebar-chatgpt-header">
          <button
            type="button"
            className="sidebar-chatgpt-brand"
            onClick={() => handleNavClick(() => onSelectView?.('chat'))}
            aria-label="Jruva home"
          >
            <JruLogo size={20} />
            <span className="sidebar-chatgpt-title">Jruva</span>
          </button>

          <div className="sidebar-chatgpt-header-actions">
            <Tooltip content="Search (Ctrl+K)" placement="below">
              <button type="button" className="sidebar-header-action-btn" onClick={openSearch} aria-label="Search">
                <Search size={16} strokeWidth={2} aria-hidden="true" />
              </button>
            </Tooltip>
            <Tooltip content="Close sidebar (Ctrl+H)" placement="below">
              <button type="button" className="sidebar-header-action-btn" onClick={onToggle} aria-label="Close sidebar">
                <PanelLeft size={16} strokeWidth={2} aria-hidden="true" />
              </button>
            </Tooltip>
          </div>
        </div>

        <div className="sidebar-newchat-container">
          <Tooltip content="Ctrl+Shift+O" placement="end">
            <button
              type="button"
              onClick={() =>
                handleNavClick(() => {
                  onSelectView?.('chat');
                  onNewChat();
                })
              }
              className={`sidebar-chatgpt-newchat-btn ${activeView === 'chat' ? 'active' : ''}`}
            >
              <SquarePen size={ICON_SIZE} strokeWidth={ICON_STROKE} aria-hidden="true" />
              <span className="sidebar-chatgpt-newchat-text">New chat</span>
            </button>
          </Tooltip>
        </div>

        <nav className="sidebar-chatgpt-nav-list" aria-label="Main">
          {navItems.map(({ view, label, Icon }) => (
            <button
              key={view}
              type="button"
              className={`sidebar-chatgpt-nav-item ${activeView === view ? 'active' : ''}`}
              onClick={() => handleNavClick(() => onSelectView?.(view))}
              aria-current={activeView === view ? 'page' : undefined}
            >
              <Icon size={ICON_SIZE} strokeWidth={ICON_STROKE} aria-hidden="true" />
              <span className="nav-item-label">{label}</span>
            </button>
          ))}

          <button
            type="button"
            className="sidebar-chatgpt-nav-item"
            onClick={() => handleNavClick(() => openMemoryModal())}
          >
            <Bookmark size={ICON_SIZE} strokeWidth={ICON_STROKE} aria-hidden="true" />
            <span className="nav-item-label">Memory Card</span>
            {memories.length > 0 && (
              <span className="nav-item-badge-updated" aria-label={`${memories.length} memories`}>
                {memories.length}
              </span>
            )}
          </button>

          <button
            type="button"
            className={`sidebar-chatgpt-nav-item sidebar-history-trigger-btn ${isHistoryOpen ? 'active' : ''}`}
            onClick={toggleHistory}
            aria-expanded={isHistoryOpen}
            aria-controls="sidebar-history-panel"
          >
            <History size={ICON_SIZE} strokeWidth={ICON_STROKE} aria-hidden="true" />
            <span className="nav-item-label">History</span>
            <span className="sidebar-history-meta-right">
              {sessions.length > 0 && <span className="sidebar-history-count-pill">{sessions.length}</span>}
              <ChevronDown
                className={`history-chevron ${isHistoryOpen ? 'open' : ''}`}
                size={14}
                strokeWidth={2.25}
                aria-hidden="true"
              />
            </span>
          </button>
        </nav>

        {isHistoryOpen && (
          <div className="sidebar-history-expanded-panel" id="sidebar-history-panel">
            {sessions.length > 0 && (
              <div className="history-search-container">
                <div className="history-search-wrapper">
                  <Search className="history-search-icon" size={14} strokeWidth={2} aria-hidden="true" />
                  <input
                    ref={searchInputRef}
                    type="search"
                    className="history-search-input"
                    placeholder="Search chats"
                    aria-label="Search chats"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Escape') setSearchQuery('');
                    }}
                  />
                  {searchQuery && (
                    <button
                      type="button"
                      className="history-search-clear"
                      onClick={() => {
                        setSearchQuery('');
                        searchInputRef.current?.focus();
                      }}
                      aria-label="Clear search"
                    >
                      <X size={13} strokeWidth={2.25} aria-hidden="true" />
                    </button>
                  )}
                </div>
              </div>
            )}

            <div className="history-recent-chats-scroll">
              {sessions.length === 0 ? (
                <div className="history-empty-placeholder">
                  <MessageSquare className="empty-placeholder-icon" size={20} strokeWidth={1.5} aria-hidden="true" />
                  <span className="empty-placeholder-text">Your chats will appear here</span>
                </div>
              ) : filteredSessions.length === 0 ? (
                <div className="history-empty-placeholder" role="status">
                  <SearchX className="empty-placeholder-icon" size={20} strokeWidth={1.5} aria-hidden="true" />
                  <span className="empty-placeholder-text">No chats match &ldquo;{searchQuery.trim()}&rdquo;</span>
                </div>
              ) : (
                <>
                  {pinnedSessions.length > 0 && (
                    <div className="history-date-group history-pinned-group">
                      <div className="history-date-group-title">Pinned</div>
                      <div className="history-date-group-items">{pinnedSessions.map(renderSessionItem)}</div>
                    </div>
                  )}
                  {groupedSessions.map((group) => (
                    <div key={group.label} className="history-date-group">
                      <div className="history-date-group-title">{group.label}</div>
                      <div className="history-date-group-items">{group.items.map(renderSessionItem)}</div>
                    </div>
                  ))}
                </>
              )}
            </div>
          </div>
        )}

        <div className="sidebar-bottom-account-bar">
          <div className="sidebar-user-pill">
            <div className="sidebar-user-avatar" aria-hidden="true">
              {accountName.charAt(0).toUpperCase()}
            </div>
            <div className="sidebar-user-meta">
              <span className="sidebar-user-name">{accountName}</span>
              <span className="sidebar-user-tier">{accountDetail}</span>
            </div>
          </div>
        </div>
      </div>

      <AlertDialog
        isOpen={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
        title="Delete this chat?"
        description={`"${pendingDelete?.title ?? ''}" and its messages will be removed. This cannot be undone.`}
        cancelLabel="Cancel"
        actionLabel="Delete"
        actionVariant="destructive"
        onAction={() => {
          if (pendingDelete) onDeleteSession(pendingDelete.id);
          setPendingDelete(null);
        }}
      />
    </aside>
  );
};
