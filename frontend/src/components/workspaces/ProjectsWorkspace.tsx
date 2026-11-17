'use client';

import '@/styles/pages/plugins-projects.css';
import { useCallback, useEffect, useState } from 'react';
import { AlertDialog } from '@astryxdesign/core/AlertDialog';
import { FolderOpen, Plus } from 'lucide-react';
import { ProjectItem } from '@/types/workspace';
import { useAuth } from '@/context/AuthContext';
import { ProjectCard } from './ProjectCard';
import { ProjectFormDialog } from './ProjectFormDialog';
import { errorMessage, isSignedOutError, requestJson } from './workspaceApi';
import { SignInPrompt, SkeletonRows, WorkspaceEmpty, WorkspaceNotice, WorkspacePageHead } from './WorkspacePageHead';

interface ProjectsWorkspaceProps {
  onBackToChat: () => void;
  onSelectProjectForChat?: (project: ProjectItem) => void;
}

type LoadState = 'loading' | 'ready' | 'error' | 'signed-out';
type FormState = { project: ProjectItem | null; key: number } | null;
type Notice = { tone: 'error' | 'success'; message: string } | null;

export function ProjectsWorkspace({ onBackToChat, onSelectProjectForChat }: ProjectsWorkspaceProps) {
  const { isAuthenticated, openAuthModal } = useAuth();
  const [projects, setProjects] = useState<ProjectItem[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [loadError, setLoadError] = useState('');
  // Guests are signed out by definition; skip the request the API would answer with 401.
  const shownState: LoadState = isAuthenticated ? loadState : 'signed-out';
  const [form, setForm] = useState<FormState>(null);
  const [pendingDelete, setPendingDelete] = useState<ProjectItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [notice, setNotice] = useState<Notice>(null);

  const loadProjects = useCallback(
    () =>
      requestJson<{ projects?: ProjectItem[] }>('/api/v1/projects').then(
        (data) => {
          setProjects(data.projects ?? []);
          setLoadState('ready');
        },
        (err: unknown) => {
          setLoadError(errorMessage(err));
          setLoadState(isSignedOutError(err) ? 'signed-out' : 'error');
        },
      ),
    [],
  );

  // Reload when the account changes, e.g. after signing in from this page.
  useEffect(() => {
    if (!isAuthenticated) return;
    void loadProjects();
  }, [loadProjects, isAuthenticated]);

  const retry = () => {
    setLoadState('loading');
    void loadProjects();
  };

  const openForm = (project: ProjectItem | null) => {
    setNotice(null);
    setForm((prev) => ({ project, key: (prev?.key ?? 0) + 1 }));
  };

  const handleSaved = (message: string) => {
    setForm(null);
    setNotice({ tone: 'success', message });
    void loadProjects();
  };

  const handleDelete = async () => {
    if (!pendingDelete) return;
    const target = pendingDelete;
    setIsDeleting(true);
    try {
      await requestJson(`/api/v1/projects/${target.id}`, { method: 'DELETE' });
      setProjects((prev) => prev.filter((p) => p.id !== target.id));
      setNotice({ tone: 'success', message: `Deleted "${target.name}".` });
    } catch (err) {
      setNotice({ tone: 'error', message: `Could not delete "${target.name}". ${errorMessage(err)}` });
    } finally {
      setIsDeleting(false);
      setPendingDelete(null);
    }
  };

  const startChat = onSelectProjectForChat
    ? (project: ProjectItem) => {
        onSelectProjectForChat(project);
        onBackToChat();
      }
    : undefined;

  const newButton = (
    <button type="button" className="pp-btn pp-btn-primary" onClick={() => openForm(null)}>
      <Plus size={16} strokeWidth={1.75} aria-hidden="true" />
      New project
    </button>
  );

  return (
    <div className="workspace-panel-container pp-page">
      <WorkspacePageHead
        title="Projects"
        description="Chats grouped around one goal."
        onBack={onBackToChat}
        actions={shownState === 'ready' && projects.length > 0 ? newButton : undefined}
      />

      {notice && (
        <WorkspaceNotice tone={notice.tone} message={notice.message} onDismiss={() => setNotice(null)} />
      )}

      {shownState === 'loading' && <SkeletonRows count={3} label="Loading projects" />}

      {shownState === 'error' && (
        <WorkspaceNotice tone="error" message={`Could not load projects. ${loadError}`} actionLabel="Try again" onAction={retry} />
      )}

      {shownState === 'signed-out' && (
        <SignInPrompt
          title="Sign in to use projects"
          description="Sign in to save projects."
          onSignIn={() => openAuthModal('signin')}
        />
      )}

      {shownState === 'ready' && projects.length === 0 && (
        <WorkspaceEmpty
          icon={<FolderOpen size={22} strokeWidth={1.75} />}
          title="No projects yet"
          description="Give a set of chats shared instructions."
          action={newButton}
        />
      )}

      {shownState === 'ready' && projects.length > 0 && (
        <ul className="pp-project-grid" aria-label="Your projects">
          {projects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onStartChat={startChat}
              onEdit={(p) => openForm(p)}
              onRequestDelete={setPendingDelete}
            />
          ))}
        </ul>
      )}

      {form && (
        <ProjectFormDialog key={form.key} project={form.project} onClose={() => setForm(null)} onSaved={handleSaved} />
      )}

      <AlertDialog
        isOpen={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open && !isDeleting) setPendingDelete(null);
        }}
        title="Delete this project?"
        description={`"${pendingDelete?.name ?? ''}" and its instructions will be removed. This cannot be undone.`}
        cancelLabel="Cancel"
        actionLabel="Delete"
        actionVariant="destructive"
        isActionLoading={isDeleting}
        onAction={handleDelete}
      />
    </div>
  );
}
