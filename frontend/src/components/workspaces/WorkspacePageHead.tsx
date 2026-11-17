import type { ReactNode } from 'react';
import { AlertCircle, ArrowLeft, Check, LogIn, X } from 'lucide-react';

const ICON = { size: 16, strokeWidth: 1.75 } as const;

type PageHeadProps = {
  title: string;
  description: string;
  onBack: () => void;
  actions?: ReactNode;
};

/** Back link, page title, one-line description and optional page actions. */
export function WorkspacePageHead({ title, description, onBack, actions }: PageHeadProps) {
  return (
    <header className="pp-head">
      <button type="button" className="pp-back" onClick={onBack}>
        <ArrowLeft {...ICON} aria-hidden="true" />
        <span>Back to chat</span>
      </button>
      <div className="pp-head-row">
        <div className="pp-head-text">
          <h1 className="pp-title">{title}</h1>
          <p className="pp-lede">{description}</p>
        </div>
        {actions && <div className="pp-head-actions">{actions}</div>}
      </div>
    </header>
  );
}

type NoticeTone = 'error' | 'success';

type NoticeProps = {
  tone: NoticeTone;
  message: string;
  actionLabel?: string;
  onAction?: () => void;
  onDismiss?: () => void;
};

/** Inline message that says what happened and, for errors, how to recover. */
export function WorkspaceNotice({ tone, message, actionLabel, onAction, onDismiss }: NoticeProps) {
  const Icon = tone === 'error' ? AlertCircle : Check;
  return (
    <div className={`pp-notice pp-notice-${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <Icon {...ICON} className="pp-notice-icon" aria-hidden="true" />
      <p className="pp-notice-text">{message}</p>
      {actionLabel && onAction && (
        <button type="button" className="pp-btn pp-btn-quiet pp-btn-sm" onClick={onAction}>
          {actionLabel}
        </button>
      )}
      {onDismiss && (
        <button type="button" className="pp-icon-btn pp-icon-btn-sm" onClick={onDismiss} aria-label="Dismiss message">
          <X size={14} strokeWidth={1.75} aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

type SignInPromptProps = {
  title: string;
  description: string;
  onSignIn: () => void;
};

/** Shown when the page needs an account and nobody is signed in. */
export function SignInPrompt({ title, description, onSignIn }: SignInPromptProps) {
  return (
    <div className="pp-empty">
      <LogIn size={22} strokeWidth={1.75} className="pp-empty-icon" aria-hidden="true" />
      <h2 className="pp-empty-title">{title}</h2>
      <p className="pp-empty-text">{description}</p>
      <button type="button" className="pp-btn pp-btn-primary" onClick={onSignIn}>
        Sign in
      </button>
    </div>
  );
}

type EmptyProps = {
  icon: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
};

export function WorkspaceEmpty({ icon, title, description, action }: EmptyProps) {
  return (
    <div className="pp-empty">
      <span className="pp-empty-icon" aria-hidden="true">
        {icon}
      </span>
      <h2 className="pp-empty-title">{title}</h2>
      <p className="pp-empty-text">{description}</p>
      {action}
    </div>
  );
}

/** Placeholder rows while a list loads. */
export function SkeletonRows({ count, label }: { count: number; label: string }) {
  return (
    <div className="pp-skeleton-list" role="status" aria-label={label}>
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="pp-skeleton-row" aria-hidden="true">
          <span className="pp-skeleton-line pp-skeleton-line-title" />
          <span className="pp-skeleton-line" />
        </div>
      ))}
    </div>
  );
}
