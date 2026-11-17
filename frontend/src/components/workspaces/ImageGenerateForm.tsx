import { Selector } from '@astryxdesign/core/Selector';
import { Spinner } from '@astryxdesign/core/Spinner';
import { Switch } from '@astryxdesign/core/Switch';
import { TextArea } from '@astryxdesign/core/TextArea';
import { Sparkles } from 'lucide-react';
import { ICON, Notice } from './WorkspacePageParts';
import '@/styles/pages/images-library.css';

export const PROMPT_MIN = 2;
export const PROMPT_MAX = 2000;

const STYLE_OPTIONS = [
  { value: 'photorealistic', label: 'Photo' },
  { value: 'cinematic', label: 'Cinematic' },
  { value: 'architecture', label: 'Architecture' },
  { value: 'cyberpunk', label: 'Cyberpunk' },
  { value: 'anime', label: 'Anime' },
];

const MODEL_OPTIONS = [
  { value: 'flux', label: 'Automatic' },
  { value: 'schnell', label: 'Fast' },
];

export type GenerateSettings = {
  prompt: string;
  style: string;
  model: string;
  enhance: boolean;
};

type Props = {
  settings: GenerateSettings;
  onChange: (next: GenerateSettings) => void;
  onSubmit: () => void;
  isGenerating: boolean;
  error: string | null;
  onDismissError: () => void;
};

/** Prompt, style, model and prompt-improvement controls. Ctrl+Enter submits. */
export function ImageGenerateForm({ settings, onChange, onSubmit, isGenerating, error, onDismissError }: Props) {
  const trimmedLength = settings.prompt.trim().length;
  const canSubmit = !isGenerating && trimmedLength >= PROMPT_MIN;
  const update = (patch: Partial<GenerateSettings>) => onChange({ ...settings, ...patch });

  return (
    <form
      className="il-card il-gen"
      aria-labelledby="il-gen-title"
      onSubmit={(e) => {
        e.preventDefault();
        if (canSubmit) onSubmit();
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canSubmit) {
          e.preventDefault();
          onSubmit();
        }
      }}
    >
      <h2 id="il-gen-title" className="il-card-title">
        Create an image
      </h2>

      <TextArea
        label="Describe the image"
        isLabelHidden
        value={settings.prompt}
        onChange={(value) => update({ prompt: value })}
        placeholder="A quiet harbour at dawn, fishing boats, soft mist"
        rows={3}
        maxLength={PROMPT_MAX}
        isDisabled={isGenerating}
        controlClassName="il-gen-prompt"
      />

      <div className="il-gen-row">
        <div className="il-gen-field">
          <Selector
            label="Style"
            value={settings.style}
            onChange={(value) => update({ style: value })}
            options={STYLE_OPTIONS}
            isDisabled={isGenerating}
            width="100%"
          />
        </div>
        <div className="il-gen-field">
          <Selector
            label="Speed"
            value={settings.model}
            onChange={(value) => update({ model: value })}
            options={MODEL_OPTIONS}
            isDisabled={isGenerating}
            width="100%"
          />
        </div>
      </div>

      <div className="il-gen-footer">
        <Switch
          label="Improve my prompt"
          value={settings.enhance}
          onChange={(checked) => update({ enhance: checked })}
          isDisabled={isGenerating}
          size="sm"
        />
        <button type="submit" className="il-btn il-btn--primary il-gen-submit" disabled={!canSubmit} aria-busy={isGenerating}>
          {isGenerating ? <Spinner size="sm" shade="inherit" /> : <Sparkles {...ICON} aria-hidden="true" />}
          {isGenerating ? 'Creating…' : 'Create'}
          {!isGenerating && (
            <kbd className="il-kbd" aria-hidden="true">
              Ctrl Enter
            </kbd>
          )}
        </button>
      </div>

      {error && (
        <Notice tone="error" onDismiss={onDismissError}>
          {error}
        </Notice>
      )}
    </form>
  );
}
