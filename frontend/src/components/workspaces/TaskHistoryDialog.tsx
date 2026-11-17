import { useCallback, useEffect, useState } from 'react';
import { Dialog, DialogHeader } from '@astryxdesign/core/Dialog';
import { Spinner } from '@astryxdesign/core/Spinner';
import { AlertCircle, CheckCircle2, LoaderCircle } from 'lucide-react';
import { ScheduledTaskItem, TaskExecutionItem } from '@/types/workspace';
import { errorMessage, formatDate, requestJson } from './workspaceApi';
import { WorkspaceNotice } from './WorkspacePageHead';

const STATUS_TEXT: Record<TaskExecutionItem['status'], string> = {
  success: 'Finished',
  failure: 'Failed',
  running: 'Running',
};

const STATUS_ICON = {
  success: CheckCircle2,
  failure: AlertCircle,
  running: LoaderCircle,
} as const;

type TaskHistoryDialogProps = {
  task: ScheduledTaskItem;
  onClose: () => void;
};

/** Past runs of one task, newest first as the backend returns them. */
export function TaskHistoryDialog({ task, onClose }: TaskHistoryDialogProps) {
  const [runs, setRuns] = useState<TaskExecutionItem[]>([]);
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [error, setError] = useState('');

  const loadRuns = useCallback(
    () =>
      requestJson<{ history?: TaskExecutionItem[] }>(`/api/v1/scheduled/tasks/${task.id}/history`).then(
        (data) => {
          setRuns(data.history ?? []);
          setState('ready');
        },
        (err: unknown) => {
          setError(errorMessage(err));
          setState('error');
        },
      ),
    [task.id],
  );

  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  const close = (open: boolean) => {
    if (!open) onClose();
  };

  return (
    <Dialog isOpen onOpenChange={close} purpose="info" width={600} maxHeight="80vh" className="pp-dialog">
      <DialogHeader title="Run history" subtitle={task.name} onOpenChange={close} />
      <div className="pp-history">
        {state === 'loading' && (
          <div className="pp-history-loading">
            <Spinner size="md" label="Loading runs" />
          </div>
        )}
        {state === 'error' && (
          <WorkspaceNotice
            tone="error"
            message={`Could not load the run history. ${error}`}
            actionLabel="Try again"
            onAction={() => {
              setState('loading');
              void loadRuns();
            }}
          />
        )}
        {state === 'ready' && runs.length === 0 && (
          <p className="pp-muted">This task has not run yet. Use Run now to try it.</p>
        )}
        {state === 'ready' && runs.length > 0 && (
          <ol className="pp-history-list">
            {runs.map((run) => {
              const Icon = STATUS_ICON[run.status] ?? AlertCircle;
              return (
                <li key={run.id} className={`pp-run pp-run-${run.status}`}>
                  <div className="pp-run-head">
                    <span className="pp-run-status">
                      <Icon size={15} strokeWidth={1.75} aria-hidden="true" />
                      {STATUS_TEXT[run.status] ?? run.status}
                    </span>
                    <time className="pp-run-date" dateTime={run.executed_at}>
                      {formatDate(run.executed_at, true)}
                    </time>
                  </div>
                  {run.output && <div className="pp-run-output">{run.output}</div>}
                  {run.error_message && <p className="pp-field-error">{run.error_message}</p>}
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </Dialog>
  );
}
