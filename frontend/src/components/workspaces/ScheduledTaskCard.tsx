import { DropdownMenu, DropdownMenuItem } from '@astryxdesign/core/DropdownMenu';
import { Switch } from '@astryxdesign/core/Switch';
import { Clock, Ellipsis, History, Play, Trash2 } from 'lucide-react';
import { ScheduledTaskItem } from '@/types/workspace';
import { scheduleLabel } from './NewTaskForm';
import { formatDate } from './workspaceApi';
import { WorkspaceNotice } from './WorkspacePageHead';

const ICON = { size: 15, strokeWidth: 1.75 } as const;

export type TaskFeedback = { tone: 'error' | 'success'; message: string };

type ScheduledTaskCardProps = {
  task: ScheduledTaskItem;
  isRunning: boolean;
  isToggling: boolean;
  feedback?: TaskFeedback;
  onRun: (task: ScheduledTaskItem) => void;
  onToggle: (task: ScheduledTaskItem) => void;
  onShowHistory: (task: ScheduledTaskItem) => void;
  onRequestDelete: (task: ScheduledTaskItem) => void;
  onDismissFeedback: (taskId: string) => void;
};

/** One saved task: its prompt, schedule, last run and the actions on it. */
export function ScheduledTaskCard({
  task,
  isRunning,
  isToggling,
  feedback,
  onRun,
  onToggle,
  onShowHistory,
  onRequestDelete,
  onDismissFeedback,
}: ScheduledTaskCardProps) {
  const isActive = task.status === 'active';
  const lastRun = formatDate(task.last_run, true);
  // Run now and History repeat on every card, so they point screen readers at the task name.
  const nameId = `task-name-${task.id}`;

  return (
    <li className="pp-task">
      <div className="pp-task-top">
        <div className="pp-task-heading">
          <h3 id={nameId} className="pp-task-name">{task.name}</h3>
          <p className="pp-task-schedule">
            <Clock size={14} strokeWidth={1.75} aria-hidden="true" />
            {scheduleLabel(task.cron_expression)}
            <span className="pp-task-tz">({task.timezone})</span>
          </p>
        </div>
        <div className="pp-task-switch">
          <span className="pp-task-state" aria-hidden="true">
            {isActive ? 'Active' : 'Paused'}
          </span>
          <Switch
            label={`Keep ${task.name} active`}
            isLabelHidden
            size="sm"
            value={isActive}
            isLoading={isToggling}
            isDisabled={isToggling}
            onChange={() => onToggle(task)}
          />
        </div>
      </div>

      <p className="pp-task-prompt">{task.prompt}</p>

      <div className="pp-task-foot">
        <span className="pp-task-meta">{lastRun ? `Last run ${lastRun}` : 'Not run yet'}</span>
        <div className="pp-task-actions">
          <button
            type="button"
            className="pp-btn pp-btn-secondary pp-btn-sm"
            onClick={() => onRun(task)}
            aria-describedby={nameId}
            disabled={isRunning}
            aria-busy={isRunning}
          >
            <Play {...ICON} aria-hidden="true" />
            {isRunning ? 'Running…' : 'Run now'}
          </button>
          <button
            type="button"
            className="pp-btn pp-btn-quiet pp-btn-sm"
            onClick={() => onShowHistory(task)}
            aria-describedby={nameId}
          >
            <History {...ICON} aria-hidden="true" />
            History
          </button>
          <DropdownMenu
            menuWidth={160}
            placement="below"
            alignment="end"
            renderTrigger={(triggerProps) => (
              <button type="button" {...triggerProps} className="pp-icon-btn" aria-label={`More options for ${task.name}`}>
                <Ellipsis size={16} strokeWidth={1.75} aria-hidden="true" />
              </button>
            )}
          >
            <DropdownMenuItem
              icon={<Trash2 size={15} />}
              label="Delete"
              variant="destructive"
              onClick={() => onRequestDelete(task)}
            />
          </DropdownMenu>
        </div>
      </div>

      {feedback && (
        <WorkspaceNotice
          tone={feedback.tone}
          message={feedback.message}
          actionLabel={feedback.tone === 'success' ? 'View result' : undefined}
          onAction={feedback.tone === 'success' ? () => onShowHistory(task) : undefined}
          onDismiss={() => onDismissFeedback(task.id)}
        />
      )}
    </li>
  );
}
