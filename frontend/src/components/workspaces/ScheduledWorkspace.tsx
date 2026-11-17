'use client';

import '@/styles/pages/plugins-projects.css';
import { useCallback, useEffect, useState } from 'react';
import { AlertDialog } from '@astryxdesign/core/AlertDialog';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { CalendarClock, Info, RefreshCw } from 'lucide-react';
import { ScheduledTaskItem } from '@/types/workspace';
import { useAuth } from '@/context/AuthContext';
import { NewTaskForm } from './NewTaskForm';
import { ScheduledTaskCard, type TaskFeedback } from './ScheduledTaskCard';
import { TaskHistoryDialog } from './TaskHistoryDialog';
import { errorMessage, isSignedOutError, requestJson } from './workspaceApi';
import { SignInPrompt, SkeletonRows, WorkspaceEmpty, WorkspaceNotice, WorkspacePageHead } from './WorkspacePageHead';

interface ScheduledWorkspaceProps {
  onBackToChat: () => void;
}

type LoadState = 'loading' | 'ready' | 'error' | 'signed-out';
type RunResponse = { success?: boolean; error?: string };

export function ScheduledWorkspace({ onBackToChat }: ScheduledWorkspaceProps) {
  const { isAuthenticated, openAuthModal } = useAuth();
  const [tasks, setTasks] = useState<ScheduledTaskItem[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [loadError, setLoadError] = useState('');
  // Guests are signed out by definition; skip the request the API would answer with 401.
  const shownState: LoadState = isAuthenticated ? loadState : 'signed-out';
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [runningId, setRunningId] = useState<string | null>(null);
  const [togglingId, setTogglingId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Record<string, TaskFeedback>>({});
  const [pageNotice, setPageNotice] = useState<TaskFeedback | null>(null);
  const [historyTask, setHistoryTask] = useState<ScheduledTaskItem | null>(null);
  const [pendingDelete, setPendingDelete] = useState<ScheduledTaskItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const loadTasks = useCallback(
    () =>
      requestJson<{ tasks?: ScheduledTaskItem[] }>('/api/v1/scheduled/tasks')
        .then(
          (data) => {
            setTasks(data.tasks ?? []);
            setLoadState('ready');
          },
          (err: unknown) => {
            setLoadError(errorMessage(err));
            setLoadState(isSignedOutError(err) ? 'signed-out' : 'error');
          },
        )
        .finally(() => setIsRefreshing(false)),
    [],
  );

  // Reload when the account changes, e.g. after signing in from this page.
  useEffect(() => {
    if (!isAuthenticated) return;
    void loadTasks();
  }, [loadTasks, isAuthenticated]);

  const setTaskFeedback = (taskId: string, next: TaskFeedback | null) => {
    setFeedback((prev) => {
      const rest = Object.fromEntries(Object.entries(prev).filter(([id]) => id !== taskId));
      return next ? { ...rest, [taskId]: next } : rest;
    });
  };

  const refresh = () => {
    setIsRefreshing(true);
    void loadTasks();
  };

  const retry = () => {
    setLoadState('loading');
    void loadTasks();
  };

  const handleRun = async (task: ScheduledTaskItem) => {
    setRunningId(task.id);
    setTaskFeedback(task.id, null);
    try {
      const data = await requestJson<RunResponse>(`/api/v1/scheduled/tasks/${task.id}/run`, { method: 'POST' });
      if (data.success) {
        setTaskFeedback(task.id, { tone: 'success', message: 'Run finished.' });
        void loadTasks();
      } else {
        setTaskFeedback(task.id, { tone: 'error', message: `The run failed: ${data.error || 'no reason given'}. Try again.` });
      }
    } catch (err) {
      setTaskFeedback(task.id, { tone: 'error', message: `Could not run the task. ${errorMessage(err)}` });
    } finally {
      setRunningId(null);
    }
  };

  const handleToggle = async (task: ScheduledTaskItem) => {
    setTogglingId(task.id);
    setTaskFeedback(task.id, null);
    try {
      const data = await requestJson<{ status: ScheduledTaskItem['status'] }>(
        `/api/v1/scheduled/tasks/${task.id}/toggle`,
        { method: 'PATCH' },
      );
      setTasks((prev) => prev.map((t) => (t.id === task.id ? { ...t, status: data.status } : t)));
    } catch (err) {
      setTaskFeedback(task.id, { tone: 'error', message: `Could not change the task. ${errorMessage(err)}` });
    } finally {
      setTogglingId(null);
    }
  };

  const handleDelete = async () => {
    if (!pendingDelete) return;
    const target = pendingDelete;
    setIsDeleting(true);
    try {
      await requestJson(`/api/v1/scheduled/tasks/${target.id}`, { method: 'DELETE' });
      setTasks((prev) => prev.filter((t) => t.id !== target.id));
      setPageNotice({ tone: 'success', message: `Deleted "${target.name}".` });
    } catch (err) {
      setPageNotice({ tone: 'error', message: `Could not delete "${target.name}". ${errorMessage(err)}` });
    } finally {
      setIsDeleting(false);
      setPendingDelete(null);
    }
  };

  const handleCreated = (name: string) => {
    setPageNotice({ tone: 'success', message: `Saved "${name}".` });
    void loadTasks();
  };

  const refreshButton = (
    <Tooltip content="Refresh" placement="below">
      <button type="button" className="pp-icon-btn" onClick={refresh} disabled={isRefreshing} aria-label="Refresh tasks">
        <RefreshCw size={16} strokeWidth={1.75} className={isRefreshing ? 'pp-spin' : undefined} aria-hidden="true" />
      </button>
    </Tooltip>
  );

  return (
    <div className="workspace-panel-container pp-page">
      <WorkspacePageHead
        title="Scheduled"
        description="Saved prompts you can run any time."
        onBack={onBackToChat}
        actions={shownState === 'ready' ? refreshButton : undefined}
      />

      {shownState === 'loading' && <SkeletonRows count={3} label="Loading tasks" />}

      {shownState === 'error' && (
        <WorkspaceNotice tone="error" message={`Could not load tasks. ${loadError}`} actionLabel="Try again" onAction={retry} />
      )}

      {shownState === 'signed-out' && (
        <SignInPrompt
          title="Sign in to use scheduled tasks"
          description="Sign in to save tasks."
          onSignIn={() => openAuthModal('signin')}
        />
      )}

      {shownState === 'ready' && (
        <>
          <p className="pp-info">
            <Info size={16} strokeWidth={1.75} aria-hidden="true" />
            <span>Tasks run when you press Run now.</span>
          </p>

          {pageNotice && (
            <WorkspaceNotice tone={pageNotice.tone} message={pageNotice.message} onDismiss={() => setPageNotice(null)} />
          )}

          <NewTaskForm onCreated={handleCreated} />

          <section className="pp-section" aria-labelledby="tasks-title">
            <h2 id="tasks-title" className="pp-section-title">
              Your tasks
            </h2>
            {tasks.length === 0 ? (
              <WorkspaceEmpty
                icon={<CalendarClock size={22} strokeWidth={1.75} />}
                title="No tasks yet"
                description="Tasks you save above show up here."
              />
            ) : (
              <ul className="pp-task-list">
                {tasks.map((task) => (
                  <ScheduledTaskCard
                    key={task.id}
                    task={task}
                    isRunning={runningId === task.id}
                    isToggling={togglingId === task.id}
                    feedback={feedback[task.id]}
                    onRun={handleRun}
                    onToggle={handleToggle}
                    onShowHistory={setHistoryTask}
                    onRequestDelete={setPendingDelete}
                    onDismissFeedback={(id) => setTaskFeedback(id, null)}
                  />
                ))}
              </ul>
            )}
          </section>
        </>
      )}

      {historyTask && <TaskHistoryDialog key={historyTask.id} task={historyTask} onClose={() => setHistoryTask(null)} />}

      <AlertDialog
        isOpen={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open && !isDeleting) setPendingDelete(null);
        }}
        title="Delete this task?"
        description={`"${pendingDelete?.name ?? ''}" and its run history will be removed. This cannot be undone.`}
        cancelLabel="Cancel"
        actionLabel="Delete"
        actionVariant="destructive"
        isActionLoading={isDeleting}
        onAction={handleDelete}
      />
    </div>
  );
}
