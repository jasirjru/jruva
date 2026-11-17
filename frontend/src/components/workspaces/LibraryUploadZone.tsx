import { useRef, useState } from 'react';
import { Spinner } from '@astryxdesign/core/Spinner';
import { Upload } from 'lucide-react';
import { ICON } from './WorkspacePageParts';
import '@/styles/pages/images-library.css';

export const ACCEPTED_EXTENSIONS = '.pdf,.txt,.md,.markdown,.csv,.json,.py,.ts,.tsx,.js,.jsx,.docx';

export type UploadProgress = { done: number; total: number; current: string } | null;

type Props = {
  canUpload: boolean;
  isSignedIn: boolean;
  uploadsEnabled: boolean | null;
  maxFileLabel: string;
  progress: UploadProgress;
  onFiles: (files: File[]) => void;
};

/** Drop target plus a real file button. Explains why uploading is unavailable when it is. */
export function LibraryUploadZone({ canUpload, isSignedIn, uploadsEnabled, maxFileLabel, progress, onFiles }: Props) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const take = (list: FileList | null) => {
    const files = list ? Array.from(list) : [];
    if (inputRef.current) inputRef.current.value = '';
    if (files.length > 0 && canUpload) onFiles(files);
  };

  let title = 'Drop files here';
  let detail = `PDF, Word, Markdown, text, CSV, JSON or code. Up to ${maxFileLabel} each.`;
  if (!isSignedIn) {
    title = 'Sign in to upload documents';
    detail = 'Private to your account.';
  } else if (uploadsEnabled === false) {
    title = 'Uploads are turned off on this server';
    detail = 'You can still attach a file to a chat message.';
  } else if (progress) {
    title = `Uploading ${Math.min(progress.done + 1, progress.total)} of ${progress.total}`;
    detail = progress.current;
  }

  return (
    <div
      className={`il-drop${isDragOver ? ' is-over' : ''}${canUpload || progress ? '' : ' is-disabled'}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (canUpload) setIsDragOver(true);
      }}
      onDragLeave={(e) => {
        // Ignore leave events fired when the pointer moves onto a child element.
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setIsDragOver(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragOver(false);
        take(e.dataTransfer.files);
      }}
    >
      <span className="il-drop-icon" aria-hidden="true">
        {progress ? <Spinner size="sm" shade="inherit" /> : <Upload size={20} strokeWidth={1.75} />}
      </span>
      <div className="il-drop-text" aria-live="polite">
        <p className="il-drop-title">{title}</p>
        <p className="il-drop-detail">{detail}</p>
      </div>
      <input
        ref={inputRef}
        id="il-file-input"
        aria-label="Choose files to upload"
        type="file"
        multiple
        accept={ACCEPTED_EXTENSIONS}
        className="il-sr"
        tabIndex={-1}
        onChange={(e) => take(e.target.files)}
        disabled={!canUpload}
      />
      <button
        type="button"
        className="il-btn il-btn--primary"
        onClick={() => inputRef.current?.click()}
        disabled={!canUpload}
      >
        <Upload {...ICON} aria-hidden="true" />
        Choose files
      </button>
    </div>
  );
}
