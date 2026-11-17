'use client';

import React, { useState, useEffect } from 'react';
import { useChatStream } from '@/hooks/useChatStream';
import { useTheme } from '@/hooks/useTheme';
import { Theme } from '@astryxdesign/core/theme';
import { chocolateTheme } from '@astryxdesign/theme-chocolate/built';
import { ArtifactProvider, useArtifact } from '@/context/ArtifactContext';
import { AuthProvider, useAuth } from '@/context/AuthContext';
import { MemoryProvider } from '@/context/MemoryContext';
import { ChatSidebar } from './ChatSidebar';
import { Header } from './Header';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { AmbientBackground } from './AmbientBackground';
import { ArtifactCanvas } from './ArtifactCanvas';
import { AuthModal } from './AuthModal';
import { MemoryCardModal } from './MemoryCardModal';
import { InitialLogoSplash } from './InitialLogoSplash';
import { WorkspaceView, ProjectItem, LibraryFileItem } from '@/types/workspace';
import { ImagesWorkspace } from '@/components/workspaces/ImagesWorkspace';
import { LibraryWorkspace } from '@/components/workspaces/LibraryWorkspace';
import { ScheduledWorkspace } from '@/components/workspaces/ScheduledWorkspace';
import { PluginsWorkspace } from '@/components/workspaces/PluginsWorkspace';
import { ProjectsWorkspace } from '@/components/workspaces/ProjectsWorkspace';
import { CommandPalette } from '@/components/workspaces/CommandPalette';

// Escape inside these closes the element itself, not the sidebar or a running answer.
const LOCAL_ESCAPE_SELECTOR =
  '[role="menu"], [role="dialog"], dialog, [popover], .history-search-input, .history-entry-rename-input, .msg-edit-textarea';

const ChatContainerInner: React.FC = () => {
  const {
    messages,
    isStreaming,
    sessionId,
    sessions,
    error,
    selectedModel,
    setSelectedModel,
    selectedTier,
    setSelectedTier,
    availableModels,
    selectedPersona,
    setSelectedPersona,
    availablePersonas,
    sendMessage,
    stopGeneration,
    clearMessages,
    switchSession,
    deleteSession,
    renameSession,
    editAndResend,
    regenerateLastResponse,
    switchMessageVariation,
  } = useChatStream();

  const { theme, toggleTheme } = useTheme();
  const { activeArtifact, closeArtifact } = useArtifact();
  // null until mounted: CSS picks the first paint (open on wide screens), so nothing shifts on load.
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean | null>(null);
  const [showScrollBottom, setShowScrollBottom] = useState(false);
  const scrollContainerRef = React.useRef<HTMLDivElement | null>(null);

  const handleScroll = React.useCallback(() => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isUp = scrollHeight - scrollTop - clientHeight > 160;
    setShowScrollBottom(isUp);
  }, []);

  const scrollToBottom = React.useCallback(() => {
    if (!scrollContainerRef.current) return;
    scrollContainerRef.current.scrollTo({
      top: scrollContainerRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, []);

  // Open the sidebar on wide screens (>= 1024px) and keep it a drawer on narrow ones.
  // Only re-decide when the breakpoint is crossed, so a user's choice survives ordinary resizes.
  useEffect(() => {
    const wide = window.matchMedia('(min-width: 1024px)');
    const sync = () => setIsSidebarOpen(wide.matches);
    sync();
    wide.addEventListener('change', sync);
    return () => wide.removeEventListener('change', sync);
  }, []);

  const [showInitialSplash, setShowInitialSplash] = useState(true);
  const [activeView, setActiveView] = useState<WorkspaceView>('chat');
  const [activeProject, setActiveProject] = useState<ProjectItem | null>(null);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const hasMessages = messages.length > 0;

  // Global Keyboard Shortcuts (Cmd/Ctrl+K, Cmd/Ctrl+H, Escape, Cmd/Ctrl+Shift+O)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl+K: Toggle Global Search Command Palette
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
      // Cmd/Ctrl+H: Toggle Chat History drawer
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'h') {
        e.preventDefault();
        setIsSidebarOpen((prev) => !prev);
      }
      // Cmd/Ctrl+Shift+O: New chat session. Chrome reserves Ctrl+Shift+N (incognito)
      // and Ctrl+O (open file), so pages never receive those.
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        clearMessages();
      }
      // Escape: Stop stream if generating, or close artifact canvas, or close history drawer.
      // Skip when Escape belongs to a menu, dialog, popover or the history search.
      const target = e.target instanceof Element ? e.target : null;
      const isLocalEscape =
        e.defaultPrevented || Boolean(target?.closest(LOCAL_ESCAPE_SELECTOR));
      if (e.key === 'Escape' && !isLocalEscape) {
        if (isCommandPaletteOpen) {
          e.preventDefault();
          setIsCommandPaletteOpen(false);
        } else if (isStreaming) {
          stopGeneration();
        } else if (activeArtifact) {
          closeArtifact();
        } else if (isSidebarOpen) {
          setIsSidebarOpen(false);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isStreaming, activeArtifact, isSidebarOpen, isCommandPaletteOpen, stopGeneration, closeArtifact, clearMessages]);

  // Export conversation to Markdown file
  const handleExportMarkdown = () => {
    if (messages.length === 0) return;
    const dateStr = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    let md = `# Conversation with Jruva\n*Exported on ${dateStr}*\n\n---\n\n`;

    for (const msg of messages) {
      const author = msg.role === 'user' ? 'You' : 'Jruva';
      md += `### ${author}\n\n${msg.content}\n\n`;
      if (msg.sources && msg.sources.length > 0) {
        md += `**Sources:**\n`;
        msg.sources.forEach((s) => {
          md += `- [${s.title}](${s.url})\n`;
        });
        md += `\n`;
      }
      md += `---\n\n`;
    }

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `jru-chat-${new Date().toISOString().slice(0, 10)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Theme theme={chocolateTheme} mode={theme}>
    <div
      className={`jru-workspace-root ${activeArtifact ? 'has-active-artifact' : ''}`}
      data-jruva-skin=""
    >
      {/* 0. 10x Initial Opening Splash Animation — Jruva Logo Rotating In Place */}
      {showInitialSplash && (
        <InitialLogoSplash onComplete={() => setShowInitialSplash(false)} />
      )}

      {/* 1. Silky Ambient Background Animation */}
      <AmbientBackground />

      {/* Backdrop overlay for smooth dismiss when history drawer is open */}
      {isSidebarOpen && (
        <div
          className="sidebar-backdrop-overlay"
          onClick={() => setIsSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* 2. Collapsible Navigation Sidebar (AI Workspace OS Hub) */}
      <ChatSidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen((open) => !open)}
        sessions={sessions}
        currentSessionId={sessionId}
        onSelectSession={switchSession}
        onNewChat={clearMessages}
        onDeleteSession={deleteSession}
        onRenameSession={renameSession}
        theme={theme}
        onToggleTheme={toggleTheme}
        activeView={activeView}
        onSelectView={setActiveView}
        onOpenCommandPalette={() => setIsCommandPaletteOpen(true)}
      />

      {/* 3. Main Stage */}
      <main className="jru-workspace-main">
        <Header
          onClear={clearMessages}
          onExport={handleExportMarkdown}
          messageCount={messages.length}
          theme={theme}
          onToggleTheme={toggleTheme}
          isSidebarOpen={isSidebarOpen === true}
          onToggleSidebar={() => setIsSidebarOpen((open) => !open)}
          sessionsCount={sessions.length}
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
          availableModels={availableModels}
          selectedTier={selectedTier}
          onSelectTier={setSelectedTier}
          selectedPersona={selectedPersona}
          onSelectPersona={setSelectedPersona}
          availablePersonas={availablePersonas}
        />

{error && (
          <div className="jru-error-banner" role="alert">
            <span>⚠️ {error}</span>
          </div>
        )}

        {/* Active Project Context Banner */}
        {activeProject && activeView === 'chat' && (
          <div className="workspace-active-project-bar">
            <span className="project-bar-icon">📂</span>
            <span className="project-bar-label">Project: <strong>{activeProject.name}</strong></span>
            {activeProject.instructions && (
              <span className="project-bar-hint" title={activeProject.instructions}>
                (Instructions Active)
              </span>
            )}
            <button
              type="button"
              className="project-bar-clear"
              onClick={() => setActiveProject(null)}
              title="Detach project context"
            >
              ✕ Detach
            </button>
          </div>
        )}

        {/* Dynamic Workspace Stage Routing */}
        {activeView === 'chat' && (
          <>
            <div className="jru-workspace-content" ref={scrollContainerRef} onScroll={handleScroll}>
              <MessageList
                messages={messages}
                onSend={(msg, atts, webSearch) => {
                  const finalMsg = activeProject?.instructions
                    ? `[Project Context: ${activeProject.name} - ${activeProject.instructions}]\n\n${msg}`
                    : msg;
                  sendMessage(finalMsg, atts, webSearch);
                }}
                onStop={stopGeneration}
                isStreaming={isStreaming}
                onRegenerate={regenerateLastResponse}
                onEditAndResend={(msgId, newContent) => editAndResend(msgId, newContent)}
                onSwitchBranch={switchMessageVariation}
              />
            </div>

            {hasMessages && showScrollBottom && (
              <button
                type="button"
                className="jru-scroll-bottom-btn"
                onClick={scrollToBottom}
                aria-label="Scroll to latest message"
                title="Scroll to latest message"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <polyline points="19 12 12 19 5 12" />
                </svg>
                {isStreaming && <span className="jru-scroll-bottom-pulse" aria-hidden="true" />}
              </button>
            )}

            {hasMessages && (
              <div className="jru-docked-footer">
                <ChatInput
                  onSend={(msg, atts, webSearch) => {
                    const finalMsg = activeProject?.instructions
                      ? `[Project Context: ${activeProject.name} - ${activeProject.instructions}]\n\n${msg}`
                      : msg;
                    sendMessage(finalMsg, atts, webSearch);
                  }}
                  onStop={stopGeneration}
                  isStreaming={isStreaming}
                  isHero={false}
                />
              </div>
            )}
          </>
        )}

        {activeView === 'images' && (
          <ImagesWorkspace
            onBackToChat={() => setActiveView('chat')}
            onSendToChat={(url, p) => {
              sendMessage(`Here is an image generated with prompt: "${p}"\n\n![${p}](${url})`);
              setActiveView('chat');
            }}
          />
        )}

        {activeView === 'library' && (
          <LibraryWorkspace
            onBackToChat={() => setActiveView('chat')}
            onAttachFileToChat={(file) => {
              sendMessage(`Please analyze and summarize this document from my knowledge base: "${file.filename}"`);
              setActiveView('chat');
            }}
          />
        )}

        {activeView === 'scheduled' && (
          <ScheduledWorkspace onBackToChat={() => setActiveView('chat')} />
        )}

        {activeView === 'plugins' && (
          <PluginsWorkspace onBackToChat={() => setActiveView('chat')} />
        )}

        {activeView === 'projects' && (
          <ProjectsWorkspace
            onBackToChat={() => setActiveView('chat')}
            onSelectProjectForChat={(proj) => {
              setActiveProject(proj);
              setSelectedModel(proj.preferred_model || 'auto');
              setActiveView('chat');
            }}
          />
        )}
      </main>

      {/* 4. Interactive Split-Screen Artifact Canvas Drawer */}
      {activeArtifact && (
        <ArtifactCanvas
          artifact={activeArtifact}
          onClose={closeArtifact}
        />
      )}


      {/* 6. Global Workspace Command Palette (Ctrl+K) */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onSelectView={(view) => setActiveView(view)}
        onSelectSession={(sessionId) => {
          switchSession(sessionId);
          setActiveView('chat');
        }}
        onNewChat={clearMessages}
        onToggleTheme={toggleTheme}
        theme={theme}
      />
    </div>
    </Theme>
  );
};

const ScopedChatState: React.FC = () => {
  const { user } = useAuth();
  return (
    <MemoryProvider key={user?.id || 'anonymous'}>
      <ArtifactProvider>
        <ChatContainerInner />
        <AuthModal />
        <MemoryCardModal />
      </ArtifactProvider>
    </MemoryProvider>
  );
};

export const ChatContainer: React.FC = () => {
  return (
    <AuthProvider>
      <ScopedChatState />
    </AuthProvider>
  );
};
