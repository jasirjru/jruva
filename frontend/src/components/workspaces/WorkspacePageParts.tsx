import type { ReactNode } from 'react';
import { AlertCircle, ArrowLeft, CheckCircle2, X } from 'lucide-react';
import '@/styles/pages/images-library.css';

export const ICON = { size: 16, strokeWidth: 1.75 } as const;

type PageHeaderProps = {
  title: string;
  subtitle: string;
  onBack: () => void;
};

/** Back link, serif title and a one-line subtitle shared by the Images and Library pages. */
export function PageHeader({ title, subtitle, onBack }: PageHeaderProps) {
  return (
    <header className="il-head">
      <button type="button" className="il-back" onClick={onBack}>
        <ArrowLeft {...ICON} aria-hidden="true" />
        Back to chat
      </button>
      <h1 className="il-title">{title}</h1>
      <p className="il-subtitle">{subtitle}</p>
    </header>
  );
}

export type NoticeTone = 'error' | 'success';

type NoticeProps = {
  tone: NoticeTone;
  children: ReactNode;
  action?: ReactNode;
  onDismiss?: () => void;
};

/** Inline message shown next to the thing it is about, with an optional recovery action. */
export function Notice({ tone, children, action, onDismiss }: NoticeProps) {
  const Icon = tone === 'error' ? AlertCircle : CheckCircle2;
  return (
    <div className={`il-notice il-notice--${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <Icon className="il-notice-icon" {...ICON} aria-hidden="true" />
      <div className="il-notice-text">{children}</div>
      {action}
      {onDismiss && (
        <button type="button" className="il-icon-btn il-notice-close" onClick={onDismiss} aria-label="Dismiss">
          <X size={15} strokeWidth={1.75} aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

type EmptyStateProps = {
  icon: ReactNode;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
};

/** Centered message for empty, signed-out and failed lists. */
export function EmptyState({ icon, title, children, action }: EmptyStateProps) {
  return (
    <div className="il-empty">
      <span className="il-empty-icon" aria-hidden="true">
        {icon}
      </span>
      <p className="il-empty-title">{title}</p>
      {children && <p className="il-empty-text">{children}</p>}
      {action}
    </div>
  );
}

/** Reads a FastAPI-style error body without throwing on non-JSON responses. */
export async function readErrorDetail(res: Response, fallback: string): Promise<string> {
  const body = await res.json().catch(() => ({}));
  const detail = body?.detail ?? body?.error_message ?? body?.error;
  return typeof detail === 'string' && detail.trim() ? detail : fallback;
}
