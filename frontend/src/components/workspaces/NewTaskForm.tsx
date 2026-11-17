import { useState, type FormEvent } from 'react';
import { Selector } from '@astryxdesign/core/Selector';
import { TextArea } from '@astryxdesign/core/TextArea';
import { TextInput } from '@astryxdesign/core/TextInput';
import { errorMessage, jsonBody, requestJson } from './workspaceApi';
import { WorkspaceNotice } from './WorkspacePageHead';

// Limits match TaskCreateRequest in backend/app/api/v1/endpoints/scheduled.py
const NAME_MAX = 150;
const PROMPT_MIN = 2;
const PROMPT_MAX = 5000;
const DEFAULT_SCHEDULE = '0 8 * * *';

export const SCHEDULE_OPTIONS = [
  { value: '0 8 * * *', label: 'Every day at 8:00' },
  { value: '0 18 * * *', label: 'Every day at 18:00' },
  { value: '0 9 * * 1', label: 'Every Monday at 9:00' },
  { value: '0 * * * *', label: 'Every hour' },
  { value: '*/30 * * * *', label: 'Every 30 minutes' },
];

export function scheduleLabel(cron: string): string {
  return SCHEDULE_OPTIONS.find((o) => o.value === cron)?.label ?? `Custom schedule (${cron})`;
}

type FieldErrors = { name?: string; prompt?: string };

function validate(name: string, prompt: string): FieldErrors {
  const errors: FieldErrors = {};
  if (!name.trim()) errors.name = 'Give the task a name.';
  if (prompt.trim().length < PROMPT_MIN) errors.prompt = 'Write what Jruva should do.';
  return errors;
}

type NewTaskFormProps = {
  onCreated: (name: string) => void;
};

/** Form that saves a new task with a prompt and a schedule. */
export function NewTaskForm({ onCreated }: NewTaskFormProps) {
  const [name, setName] = useState('');
  const [prompt, setPrompt] = useState('');
  const [schedule, setSchedule] = useState(DEFAULT_SCHEDULE);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const found = validate(name, prompt);
    setErrors(found);
    if (found.name || found.prompt) return;

    setIsSaving(true);
    setSaveError(null);
    try {
      await requestJson(
        '/api/v1/scheduled/tasks',
        jsonBody('POST', {
          name: name.trim(),
          prompt: prompt.trim(),
          cron_expression: schedule,
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
        }),
      );
      onCreated(name.trim());
      setName('');
      setPrompt('');
    } catch (err) {
      setSaveError(`Could not save the task. ${errorMessage(err)}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <section className="pp-card" aria-labelledby="new-task-title">
      <h2 id="new-task-title" className="pp-card-title">
        New task
      </h2>
      <form className="pp-form" onSubmit={handleSubmit} noValidate>
        <TextInput
          label="Name"
          value={name}
          placeholder="For example: Morning news summary"
          onChange={(value) => {
            setName(value.slice(0, NAME_MAX));
            if (errors.name && value.trim()) setErrors((prev) => ({ ...prev, name: undefined }));
          }}
          status={errors.name ? { type: 'error', message: errors.name } : undefined}
        />
        <TextArea
          label="What should Jruva do?"
          rows={3}
          maxLength={PROMPT_MAX}
          value={prompt}
          placeholder="For example: Summarise today's top news in five points."
          onChange={(value) => {
            setPrompt(value);
            if (errors.prompt && value.trim().length >= PROMPT_MIN) setErrors((prev) => ({ ...prev, prompt: undefined }));
          }}
          status={errors.prompt ? { type: 'error', message: errors.prompt } : undefined}
        />
        <div className="pp-form-row">
          <Selector
            label="Schedule"
            options={SCHEDULE_OPTIONS}
            value={schedule}
            onChange={(value) => setSchedule(value)}
            width="100%"
          />
          <button type="submit" className="pp-btn pp-btn-primary" disabled={isSaving} aria-busy={isSaving}>
            {isSaving ? 'Saving…' : 'Save task'}
          </button>
        </div>
        {saveError && <WorkspaceNotice tone="error" message={saveError} onDismiss={() => setSaveError(null)} />}
      </form>
    </section>
  );
}
