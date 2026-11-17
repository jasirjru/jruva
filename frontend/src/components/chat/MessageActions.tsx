import { useState } from 'react';
import { DropdownMenu, DropdownMenuItem } from '@astryxdesign/core/DropdownMenu';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  FileCode,
  FileText,
  FileType,
  RotateCcw,
  ThumbsDown,
  ThumbsUp,
} from 'lucide-react';
import { Message } from '@/types/chat';
import { haptics } from '@/utils/haptics';
import { exportAnswerAsMarkdown } from '@/utils/exportDoc/download';

type ExportFormat = 'markdown' | 'pdf' | 'docx';
type Feedback = 'up' | 'down';

const EXPORT_OPTIONS: { format: ExportFormat; label: string; Icon: typeof FileText }[] = [
  { format: 'pdf', label: 'PDF', Icon: FileText },
  { format: 'docx', label: 'Word', Icon: FileType },
  { format: 'markdown', label: 'Markdown', Icon: FileCode },
];
const COPY_RESET_MS = 2000;
const ICON = { size: 15, strokeWidth: 1.75 } as const;

function postFeedback(messageId: string, signal: Feedback | 'regenerated') {
  return fetch('/api/v1/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_id: messageId, signal }),
  });
}

type AssistantActionsProps = {
  message: Message;
  onRegenerate?: () => void;
  onSwitchBranch?: (messageId: string, targetIndex: number) => void;
};

/** Copy, regenerate, download, feedback and variant switching under an answer. */
export function AssistantActions({ message, onRegenerate, onSwitchBranch }: AssistantActionsProps) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [exportBusy, setExportBusy] = useState<ExportFormat | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const variationCount = message.variations?.length ?? 0;
  const variationIndex = message.activeVariationIndex ?? Math.max(variationCount - 1, 0);

  const handleCopy = async () => {
    if (!message.content) return;
    try {
      await navigator.clipboard.writeText(message.content);
      setCopyState('copied');
    } catch {
      setCopyState('failed');
    }
    setTimeout(() => setCopyState('idle'), COPY_RESET_MS);
  };

  const handleFeedback = async (signal: Feedback) => {
    const next = feedback === signal ? null : signal;
    setFeedback(next);
    if (!next || !message.id) return;
    haptics.click();
    try {
      await postFeedback(message.id, next);
    } catch {
      // Feedback is best-effort; the choice still shows locally
    }
  };

  const handleRegenerate = () => {
    if (message.id) postFeedback(message.id, 'regenerated').catch(() => undefined);
    onRegenerate?.();
  };

  const handleExport = async (format: ExportFormat, label: string) => {
    if (exportBusy) return;
    haptics.click();
    setExportBusy(format);
    setStatus(`Preparing ${label} file…`);
    try {
      if (format === 'markdown') {
        exportAnswerAsMarkdown(message.content);
      } else if (format === 'pdf') {
        const { exportAnswerAsPdf } = await import('@/utils/exportDoc/pdfExport');
        await exportAnswerAsPdf(message.content);
      } else {
        const { exportAnswerAsDocx } = await import('@/utils/exportDoc/docxExport');
        await exportAnswerAsDocx(message.content);
      }
      setStatus(null);
    } catch {
      setStatus(`Could not create the ${label} file. Please try again.`);
    } finally {
      setExportBusy(null);
    }
  };

  const copyLabel = copyState === 'copied' ? 'Copied' : copyState === 'failed' ? 'Copy failed' : 'Copy';

  return (
    <div className="message-actions-strip">
      {variationCount > 1 && onSwitchBranch && (
        <div className="msg-branch-pager" role="group" aria-label="Answer versions">
          <button
            type="button"
            className="msg-branch-btn"
            onClick={() => onSwitchBranch(message.id, variationIndex - 1)}
            disabled={variationIndex <= 0}
            aria-label="Previous version"
          >
            <ChevronLeft size={14} strokeWidth={2} aria-hidden="true" />
          </button>
          <span className="msg-branch-indicator" aria-live="polite">
            {variationIndex + 1} / {variationCount}
          </span>
          <button
            type="button"
            className="msg-branch-btn"
            onClick={() => onSwitchBranch(message.id, variationIndex + 1)}
            disabled={variationIndex >= variationCount - 1}
            aria-label="Next version"
          >
            <ChevronRight size={14} strokeWidth={2} aria-hidden="true" />
          </button>
        </div>
      )}

      <Tooltip content={copyLabel} placement="below">
        <button type="button" onClick={handleCopy} className="msg-action-btn" aria-label={copyLabel}>
          {copyState === 'copied' ? <Check {...ICON} aria-hidden="true" /> : <Copy {...ICON} aria-hidden="true" />}
        </button>
      </Tooltip>

      {onRegenerate && (
        <Tooltip content="Try again" placement="below">
          <button type="button" onClick={handleRegenerate} className="msg-action-btn" aria-label="Try again">
            <RotateCcw {...ICON} aria-hidden="true" />
          </button>
        </Tooltip>
      )}

      <DropdownMenu
        menuWidth={168}
        placement="below"
        alignment="start"
        header={<span className="jru-menu-heading">Download as</span>}
        renderTrigger={(triggerProps) => (
          <button
            type="button"
            {...triggerProps}
            className="msg-action-btn"
            aria-label="Download this answer"
            aria-busy={exportBusy !== null}
          >
            <Download {...ICON} aria-hidden="true" />
          </button>
        )}
      >
        {EXPORT_OPTIONS.map(({ format, label, Icon }) => (
          <DropdownMenuItem
            key={format}
            icon={<Icon size={15} />}
            label={label}
            isDisabled={exportBusy !== null}
            onClick={() => handleExport(format, label)}
          />
        ))}
      </DropdownMenu>

      <Tooltip content="Good answer" placement="below">
        <button
          type="button"
          onClick={() => handleFeedback('up')}
          className={`msg-action-btn ${feedback === 'up' ? 'active-feedback' : ''}`}
          aria-label="Good answer"
          aria-pressed={feedback === 'up'}
        >
          <ThumbsUp {...ICON} fill={feedback === 'up' ? 'currentColor' : 'none'} aria-hidden="true" />
        </button>
      </Tooltip>

      <Tooltip content="Bad answer" placement="below">
        <button
          type="button"
          onClick={() => handleFeedback('down')}
          className={`msg-action-btn ${feedback === 'down' ? 'active-feedback' : ''}`}
          aria-label="Bad answer"
          aria-pressed={feedback === 'down'}
        >
          <ThumbsDown {...ICON} fill={feedback === 'down' ? 'currentColor' : 'none'} aria-hidden="true" />
        </button>
      </Tooltip>

      <span className="msg-export-error" role="status">
        {status ?? ''}
      </span>
    </div>
  );
}
