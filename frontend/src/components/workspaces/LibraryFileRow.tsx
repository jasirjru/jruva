import { Spinner } from '@astryxdesign/core/Spinner';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  File,
  FileCode,
  FileSpreadsheet,
  FileText,
  MessageSquare,
  RotateCcw,
  Trash2,
} from 'lucide-react';
import { LibraryFileItem } from '@/types/workspace';
import { ICON } from './WorkspacePageParts';
import '@/styles/pages/images-library.css';

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileKind(file: LibraryFileItem): { Icon: typeof File; label: string } {
  const name = file.filename.toLowerCase();
  const type = file.file_type.toLowerCase();
  const ext = name.includes('.') ? name.split('.').pop() ?? '' : '';
  if (type.includes('pdf') || ext === 'pdf') return { Icon: FileText, label: 'PDF' };
  if (type.includes('word') || ext === 'docx') return { Icon: FileText, label: 'Word' };
  if (type.includes('csv') || ext === 'csv') return { Icon: FileSpreadsheet, label: 'CSV' };
  if (['md', 'markdown'].includes(ext)) return { Icon: FileText, label: 'Markdown' };
  if (ext === 'txt' || type.startsWith('text/plain')) return { Icon: FileText, label: 'Text' };
  if (['json', 'py', 'ts', 'tsx', 'js', 'jsx'].includes(ext)) return { Icon: FileCode, label: ext.toUpperCase() };
  return { Icon: File, label: ext ? ext.toUpperCase() : 'File' };
}

const STATUS: Record<LibraryFileItem['status'], { label: string; tone: string }> = {
  indexed: { label: 'Ready', tone: 'ready' },
  pending: { label: 'Waiting', tone: 'busy' },
  processing: { label: 'Indexing', tone: 'busy' },
  failed: { label: 'Failed', tone: 'failed' },
  deleting: { label: 'Deleting', tone: 'busy' },
};

function StatusChip({ status }: { status: LibraryFileItem['status'] }) {
  const info = STATUS[status] ?? STATUS.failed;
  let icon = <Clock size={13} strokeWidth={2} aria-hidden="true" />;
  if (status === 'indexed') icon = <CheckCircle2 size={13} strokeWidth={2} aria-hidden="true" />;
  if (status === 'failed') icon = <AlertCircle size={13} strokeWidth={2} aria-hidden="true" />;
  if (status === 'processing') icon = <Spinner size="sm" shade="inherit" />;
  return (
    <span className={`il-chip il-chip--${info.tone}`}>
      {icon}
      {info.label}
    </span>
  );
}

type Props = {
  file: LibraryFileItem;
  isRetrying: boolean;
  onRetry: (file: LibraryFileItem) => void;
  onRequestDelete: (file: LibraryFileItem) => void;
  onUseInChat?: (file: LibraryFileItem) => void;
};

/** One document: type, name, size, chunk count, indexing status and its actions. */
export function LibraryFileRow({ file, isRetrying, onRetry, onRequestDelete, onUseInChat }: Props) {
  const { Icon, label } = fileKind(file);
  const chunks = file.status === 'indexed' ? ` · ${file.chunk_count} ${file.chunk_count === 1 ? 'section' : 'sections'}` : '';

  return (
    <li className="il-row">
      <span className="il-row-icon" aria-hidden="true">
        <Icon size={18} strokeWidth={1.75} />
      </span>
      <div className="il-row-main">
        <p className="il-row-name" title={file.filename}>
          {file.filename}
        </p>
        <p className="il-row-meta">
          {label} · {formatFileSize(file.file_size)}
          {chunks}
        </p>
      </div>
      <StatusChip status={file.status} />
      <div className="il-row-actions">
        {file.status === 'failed' && (
          <button
            type="button"
            className="il-btn il-btn--small"
            onClick={() => onRetry(file)}
            disabled={isRetrying}
            aria-busy={isRetrying}
            aria-label={`Retry indexing ${file.filename}`}
          >
            {isRetrying ? <Spinner size="sm" shade="inherit" /> : <RotateCcw {...ICON} aria-hidden="true" />}
            {isRetrying ? 'Retrying…' : 'Retry'}
          </button>
        )}
        {file.status === 'indexed' && onUseInChat && (
          <button
            type="button"
            className="il-btn il-btn--small"
            onClick={() => onUseInChat(file)}
            aria-label={`Summarize ${file.filename} in chat`}
          >
            <MessageSquare {...ICON} aria-hidden="true" />
            Summarize in chat
          </button>
        )}
        <Tooltip content="Delete" placement="below">
          <button
            type="button"
            className="il-icon-btn il-icon-btn--danger"
            onClick={() => onRequestDelete(file)}
            disabled={file.status === 'deleting'}
            aria-label={`Delete ${file.filename}`}
          >
            <Trash2 {...ICON} aria-hidden="true" />
          </button>
        </Tooltip>
      </div>
    </li>
  );
}

/** Placeholder rows shown while the library loads. */
export function LibrarySkeleton() {
  return (
    <ul className="il-rows" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <li key={i} className="il-row il-row--skeleton">
          <span className="il-row-icon il-shimmer" />
          <div className="il-row-main">
            <span className="il-bar il-shimmer" style={{ width: `${60 - i * 12}%` }} />
            <span className="il-bar il-bar--short il-shimmer" />
          </div>
        </li>
      ))}
    </ul>
  );
}
