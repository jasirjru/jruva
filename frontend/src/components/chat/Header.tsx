'use client';

import React, { useEffect, useRef, useState } from 'react';
import { DropdownMenu, DropdownMenuDivider, DropdownMenuItem } from '@astryxdesign/core/DropdownMenu';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { Bookmark, Check, ChevronDown, Download, LogIn, LogOut, PanelLeft, Plus } from 'lucide-react';
import { Theme } from '@/hooks/useTheme';
import { ModelOption, PersonaOption, UserTier } from '@/types/chat';
import { useAuth } from '@/context/AuthContext';
import { useMemory } from '@/context/MemoryContext';

interface HeaderProps {
  onClear: () => void;
  onExport?: () => void;
  messageCount: number;
  theme: Theme;
  onToggleTheme: (event?: React.MouseEvent) => void;
  isSidebarOpen: boolean;
  onToggleSidebar: () => void;
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
  availableModels: ModelOption[];
  selectedTier?: UserTier;
  onSelectTier?: (tier: UserTier) => void;
  selectedPersona?: string;
  onSelectPersona?: (personaId: string) => void;
  availablePersonas?: PersonaOption[];
  sessionsCount?: number;
  onOpenExecutive?: () => void;
}

const ICON_SIZE = 16;
const DEFAULT_PERSONA_ID = 'general';

function firstName(fullName?: string | null, email?: string | null): string {
  if (fullName) return fullName.split(' ')[0];
  if (email) return email.split('@')[0];
  return 'Account';
}

export const Header: React.FC<HeaderProps> = ({
  onClear,
  onExport,
  messageCount,
  theme,
  onToggleTheme,
  isSidebarOpen,
  onToggleSidebar,
  selectedPersona,
  onSelectPersona,
  availablePersonas,
}) => {
  const { user, isAuthenticated, openAuthModal, logout } = useAuth();
  const { memories, openMemoryModal } = useMemory();

  const [isPersonaOpen, setIsPersonaOpen] = useState(false);
  const personaDropdownRef = useRef<HTMLDivElement | null>(null);

  // Close the answer-style menu on outside click or Escape
  useEffect(() => {
    if (!isPersonaOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (personaDropdownRef.current && !personaDropdownRef.current.contains(e.target as Node)) {
        setIsPersonaOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsPersonaOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isPersonaOpen]);

  const activePersonaId = selectedPersona || DEFAULT_PERSONA_ID;
  const activePersona = availablePersonas?.find((p) => p.id === activePersonaId);
  const hasPersonas = Boolean(availablePersonas && availablePersonas.length > 0 && onSelectPersona);
  const sidebarLabel = isSidebarOpen ? 'Close sidebar' : 'Open sidebar';
  const themeLabel = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
  const displayName = firstName(user?.full_name, user?.email);
  const avatarInitial = displayName.charAt(0).toUpperCase();

  return (
    <header className="workspace-top-bar">
      <div className="workspace-top-left">
        <Tooltip content={`${sidebarLabel} (Ctrl+H)`} placement="below">
          <button
            type="button"
            onClick={onToggleSidebar}
            className={`sidebar-icon-toggle-btn ${isSidebarOpen ? 'active' : ''}`}
            aria-label={sidebarLabel}
            aria-expanded={isSidebarOpen}
          >
            <PanelLeft size={18} strokeWidth={1.75} aria-hidden="true" />
          </button>
        </Tooltip>

        {hasPersonas && availablePersonas && onSelectPersona && (
          <div className="persona-selector-container" ref={personaDropdownRef}>
            <button
              type="button"
              className={`persona-selector-pill ${isPersonaOpen ? 'active' : ''}`}
              onClick={() => setIsPersonaOpen((open) => !open)}
              aria-label={`Answer style: ${activePersona?.name ?? 'General'}`}
              aria-haspopup="listbox"
              aria-expanded={isPersonaOpen}
            >
              <span className="persona-pill-icon" aria-hidden="true">{activePersona?.icon ?? '🧠'}</span>
              <span className="persona-pill-name">{activePersona?.name ?? 'General'}</span>
              <ChevronDown
                size={13}
                strokeWidth={2.25}
                className={`model-pill-chevron ${isPersonaOpen ? 'open' : ''}`}
                aria-hidden="true"
              />
            </button>

            {isPersonaOpen && (
              <div className="persona-dropdown-menu" role="listbox" aria-label="Answer style">
                <div className="model-dropdown-list">
                  {availablePersonas.map((persona) => {
                    const isSelected = persona.id === activePersonaId;
                    return (
                      <button
                        key={persona.id}
                        type="button"
                        className={`model-dropdown-card ${isSelected ? 'selected' : ''}`}
                        onClick={() => {
                          onSelectPersona(persona.id);
                          setIsPersonaOpen(false);
                        }}
                        role="option"
                        aria-selected={isSelected}
                      >
                        <div className="persona-avatar-box" aria-hidden="true">
                          <span style={{ fontSize: '15px' }}>{persona.icon}</span>
                        </div>
                        <div className="model-card-content">
                          <div className="model-card-title-line">
                            <span className="model-card-name">{persona.name}</span>
                          </div>
                        </div>
                        <div className="model-card-trailing">
                          <div className={`model-radio-circle ${isSelected ? 'selected' : ''}`}>
                            {isSelected && <Check size={10} strokeWidth={3.2} aria-hidden="true" />}
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="workspace-top-right">
        <Tooltip content={themeLabel} placement="below">
          <button
            type="button"
            onClick={(e) => onToggleTheme(e)}
            className={`workspace-icon-action theme-toggle-btn ${theme}`}
            aria-label={themeLabel}
          >
            <div className="theme-toggle-icon-stage" aria-hidden="true">
              <svg
                className={`theme-icon-sun ${theme === 'dark' ? 'active' : 'inactive'}`}
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" />
                <line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" />
                <line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </svg>
              <svg
                className={`theme-icon-moon ${theme === 'light' ? 'active' : 'inactive'}`}
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            </div>
          </button>
        </Tooltip>

        {messageCount > 0 && onExport && (
          <Tooltip content="Download this chat as Markdown" placement="below">
            <button type="button" onClick={onExport} className="workspace-export-btn">
              <Download size={14} strokeWidth={2} aria-hidden="true" />
              <span>Export</span>
            </button>
          </Tooltip>
        )}

        {/* Only when the sidebar, which has its own "New chat", is closed */}
        {messageCount > 0 && !isSidebarOpen && (
          <Tooltip content="New chat (Ctrl+Shift+O)" placement="below">
            <button type="button" onClick={onClear} className="workspace-new-chat-btn">
              <Plus size={14} strokeWidth={2.25} aria-hidden="true" />
              <span>New</span>
            </button>
          </Tooltip>
        )}

        {isAuthenticated && user ? (
          <DropdownMenu
            menuWidth={240}
            placement="below"
            alignment="end"
            header={
              <div className="jru-account-menu-header">
                <span className="jru-account-menu-name">{user.full_name || displayName}</span>
                {user.email && <span className="jru-account-menu-email">{user.email}</span>}
              </div>
            }
            renderTrigger={(triggerProps) => (
              <button
                type="button"
                {...triggerProps}
                className="header-user-pill"
                aria-label={`Account menu for ${displayName}`}
              >
                <span className="header-user-avatar" aria-hidden="true">{avatarInitial}</span>
                <span className="header-user-name">{displayName}</span>
                <ChevronDown
                  size={12}
                  strokeWidth={2.25}
                  className={`model-pill-chevron ${triggerProps['aria-expanded'] ? 'open' : ''}`}
                  aria-hidden="true"
                />
              </button>
            )}
          >
            <DropdownMenuItem
              icon={<Bookmark size={ICON_SIZE} strokeWidth={1.75} />}
              label="Memory Card"
              endContent={<span className="jru-menu-count">{memories.length}</span>}
              onClick={() => openMemoryModal()}
            />
            <DropdownMenuDivider />
            <DropdownMenuItem
              icon={<LogOut size={ICON_SIZE} strokeWidth={1.75} />}
              label="Log out"
              onClick={() => logout()}
            />
          </DropdownMenu>
        ) : (
          <button type="button" className="header-auth-signin-pill" onClick={() => openAuthModal('signin')}>
            <LogIn size={14} strokeWidth={2} aria-hidden="true" />
            <span>Sign in</span>
          </button>
        )}
      </div>
    </header>
  );
};
