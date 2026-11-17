'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertDialog } from '@astryxdesign/core/AlertDialog';
import { TextInput } from '@astryxdesign/core/TextInput';
import { BookOpen, LogIn, RefreshCw, Search } from 'lucide-react';
import { LibraryFileItem } from '@/types/workspace';
import { useAuth } from '@/context/AuthContext';
import { EmptyState, ICON, Notice, NoticeTone, PageHeader, readErrorDetail } from './WorkspacePageParts';
import { LibraryUploadZone, UploadProgress } from './LibraryUploadZone';
import { LibraryFileRow, LibrarySkeleton } from './LibraryFileRow';
import '@/styles/pages/images-library.css';

interface LibraryWorkspaceProps {
  onBackToChat: () => void;
  onAttachFileToChat?: (file: LibraryFileItem) => void;
}

type ListStatus = 'loading' | 'ready' | 'error' | 'signed-out';
type PageNotice = { tone: NoticeTone; text: string; detail?: string[] } | null;

const AUTH_STORAGE_KEYS = ['jru_auth_user', 'jru_auth_token', 'token'];
const DEFAULT_MAX_BYTES = 15 * 1024 * 1024;
const POLL_MS = 5000;
const SUCCESS_NOTICE_MS = 4000;

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('The file could not be read'));
    reader.readAsDataURL(file);
  });
}

const hasStoredUser = () => typeof window === 'undefined' || Boolean(localStorage.getItem('jru_auth_user'));

export function LibraryWorkspace({ onBackToChat, onAttachFileToChat }: LibraryWorkspaceProps) {
  const [files, setFiles] = useState<LibraryFileItem[]>([]);
  const [status, setStatus] = useState<ListStatus>('loading');
  const [query, setQuery] = useState('');
  const [progress, setProgress] = useState<UploadProgress>(null);
  // Server-reported upload policy; null until the file list has loaded.
  const [uploadsEnabled, setUploadsEnabled] = useState<boolean | null>(null);
  const [maxFileBytes, setMaxFileBytes] = useState<number | null>(null);
  const [retryingId, setRetryingId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<LibraryFileItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [notice, setNotice] = useState<PageNotice>(null);
  const authEpochRef = useRef(0);
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { isAuthenticated, isLoading: authLoading } = useAuth();

  const showNotice = useCallback((next: PageNotice) => {
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
    setNotice(next);
    if (next?.tone === 'success') noticeTimer.current = setTimeout(() => setNotice(null), SUCCESS_NOTICE_MS);
  }, []);

  const fetchFiles = useCallback(async ({ quiet = false } = {}) => {
    const authEpoch = authEpochRef.current;
    if (!hasStoredUser()) {
      setFiles([]);
      setStatus('signed-out');
      return;
    }
    if (!quiet) setStatus((prev) => (prev === 'ready' ? prev : 'loading'));
    try {
      const res = await fetch('/api/v1/library/files');
      if (authEpoch !== authEpochRef.current) return;
      if (res.status === 401) {
        setFiles([]);
        setStatus('signed-out');
        return;
      }
      if (!res.ok) {
        if (!quiet) setStatus('error');
        return;
      }
      const data = await res.json();
      if (authEpoch !== authEpochRef.current) return;
      setFiles(data.files || []);
      if (typeof data.uploads_enabled === 'boolean') setUploadsEnabled(data.uploads_enabled);
      if (typeof data.max_file_bytes === 'number') setMaxFileBytes(data.max_file_bytes);
      setStatus('ready');
    } catch {
      if (!quiet) setStatus('error');
    }
  }, []);

  useEffect(() => {
    if (authLoading) return;
    if (!isAuthenticated) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setFiles([]);
      setStatus('signed-out');
      return;
    }
    void fetchFiles();
    const handleAuthChange = () => {
      authEpochRef.current += 1;
      setFiles([]);
      showNotice(null);
      void fetchFiles();
    };
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key !== null && !AUTH_STORAGE_KEYS.includes(e.key)) return;
      if (e.key !== null && e.oldValue === e.newValue) return;
      handleAuthChange();
    };
    window.addEventListener('jru-auth-change', handleAuthChange);
    window.addEventListener('storage', handleStorageChange);
    return () => {
      window.removeEventListener('jru-auth-change', handleAuthChange);
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [isAuthenticated, authLoading, fetchFiles, showNotice]);

  // Keep indexing states fresh without a manual refresh.
  const isIndexing = files.some((f) => f.status === 'pending' || f.status === 'processing' || f.status === 'deleting');
  useEffect(() => {
    if (!isIndexing || progress) return;
    const timer = setTimeout(() => void fetchFiles({ quiet: true }), POLL_MS);
    return () => clearTimeout(timer);
  }, [isIndexing, progress, files, fetchFiles]);

  useEffect(() => () => {
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
  }, []);

  const maxBytes = maxFileBytes ?? DEFAULT_MAX_BYTES;
  const maxFileLabel = `${Math.round(maxBytes / (1024 * 1024))} MB`;
  const canUpload = isAuthenticated && uploadsEnabled !== false && !progress;

  const uploadFiles = async (picked: File[]) => {
    if (!canUpload) return;
    showNotice(null);
    const failures: string[] = [];
    for (let i = 0; i < picked.length; i++) {
      const file = picked[i];
      setProgress({ done: i, total: picked.length, current: file.name });
      if (file.size > maxBytes) {
        failures.push(`${file.name}: larger than ${maxFileLabel}`);
        continue;
      }
      try {
        const res = await fetch('/api/v1/library/upload-json', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: file.name,
            data: await readAsDataUrl(file),
            file_type: file.type || 'application/octet-stream',
          }),
        });
        if (!res.ok) throw new Error(await readErrorDetail(res, 'upload failed'));
      } catch (err) {
        failures.push(`${file.name}: ${err instanceof Error ? err.message : 'upload failed'}`);
      }
    }
    setProgress(null);
    const uploaded = picked.length - failures.length;
    if (failures.length === 0) {
      showNotice({ tone: 'success', text: uploaded === 1 ? 'Document uploaded.' : `${uploaded} documents uploaded.` });
    } else {
      const lead = uploaded > 0 ? `${uploaded} uploaded, ${failures.length} failed.` : 'Upload failed.';
      showNotice({ tone: 'error', text: `${lead} Check the file and try again.`, detail: failures });
    }
    await fetchFiles({ quiet: true });
  };

  const handleRetry = async (file: LibraryFileItem) => {
    setRetryingId(file.id);
    showNotice(null);
    try {
      const res = await fetch(`/api/v1/library/files/${file.id}/reindex`, { method: 'POST' });
      if (!res.ok) throw new Error(await readErrorDetail(res, 'Indexing failed again.'));
      showNotice({ tone: 'success', text: `${file.filename} is indexed.` });
      await fetchFiles({ quiet: true });
    } catch (err) {
      const reason = err instanceof Error ? err.message : 'Indexing failed again.';
      showNotice({ tone: 'error', text: `${file.filename} could not be indexed. ${reason}` });
    } finally {
      setRetryingId(null);
    }
  };

  const handleDelete = async () => {
    if (!pendingDelete) return;
    const target = pendingDelete;
    setIsDeleting(true);
    try {
      const res = await fetch(`/api/v1/library/files/${target.id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('incomplete');
      setFiles((prev) => prev.filter((f) => f.id !== target.id));
      showNotice({ tone: 'success', text: `${target.filename} deleted.` });
    } catch {
      showNotice({ tone: 'error', text: `${target.filename} was not fully deleted. Delete it again to finish.` });
      await fetchFiles({ quiet: true });
    } finally {
      setIsDeleting(false);
      setPendingDelete(null);
    }
  };

  const needle = query.trim().toLowerCase();
  const visible = needle ? files.filter((f) => f.filename.toLowerCase().includes(needle)) : files;
  const countText = needle
    ? `${visible.length} of ${files.length}`
    : `${files.length} ${files.length === 1 ? 'document' : 'documents'}`;

  const useInChat = onAttachFileToChat
    ? (file: LibraryFileItem) => {
        onAttachFileToChat(file);
        onBackToChat();
      }
    : undefined;

  return (
    <div className="il-page">
      <PageHeader title="Library" subtitle="Upload documents and use them in your chats." onBack={onBackToChat} />

      <LibraryUploadZone
        canUpload={canUpload}
        isSignedIn={isAuthenticated}
        uploadsEnabled={uploadsEnabled}
        maxFileLabel={maxFileLabel}
        progress={progress}
        onFiles={uploadFiles}
      />

      {notice && (
        <Notice tone={notice.tone} onDismiss={() => showNotice(null)}>
          <p>{notice.text}</p>
          {notice.detail && (
            <ul className="il-notice-list">
              {notice.detail.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          )}
        </Notice>
      )}

      <section className="il-section" aria-labelledby="il-docs-title">
        <div className="il-section-head">
          <h2 id="il-docs-title" className="il-card-title">
            Documents
            {status === 'ready' && files.length > 0 && <span className="il-count">{countText}</span>}
          </h2>
          {status === 'ready' && files.length > 0 && (
            <div className="il-search">
              <TextInput
                type="search"
                label="Search documents"
                isLabelHidden
                placeholder="Search by name"
                startIcon={<Search size={15} strokeWidth={1.75} />}
                value={query}
                onChange={setQuery}
                onKeyDown={(e) => {
                  if (e.key === 'Escape' && query) {
                    e.preventDefault();
                    e.stopPropagation();
                    setQuery('');
                  }
                }}
                width="100%"
              />
            </div>
          )}
        </div>

        {status === 'loading' && <LibrarySkeleton />}
        {status === 'signed-out' && (
          <EmptyState icon={<LogIn size={22} strokeWidth={1.75} />} title="Sign in to use your library">
            Private to your account.
          </EmptyState>
        )}
        {status === 'error' && (
          <EmptyState
            icon={<BookOpen size={22} strokeWidth={1.75} />}
            title="Your library could not be loaded"
            action={
              <button type="button" className="il-btn" onClick={() => void fetchFiles()}>
                <RefreshCw {...ICON} aria-hidden="true" />
                Try again
              </button>
            }
          >
            Check your connection, then try again.
          </EmptyState>
        )}
        {status === 'ready' && files.length === 0 && (
          <EmptyState icon={<BookOpen size={22} strokeWidth={1.75} />} title="No documents yet">
            Files you can ask about in chat.
          </EmptyState>
        )}
        {status === 'ready' && files.length > 0 && visible.length === 0 && (
          <EmptyState
            icon={<Search size={22} strokeWidth={1.75} />}
            title={`No documents match "${query.trim()}"`}
            action={
              <button type="button" className="il-btn" onClick={() => setQuery('')}>
                Clear search
              </button>
            }
          />
        )}
        {status === 'ready' && visible.length > 0 && (
          <ul className="il-rows" aria-label="Documents">
            {visible.map((file) => (
              <LibraryFileRow
                key={file.id}
                file={file}
                isRetrying={retryingId === file.id}
                onRetry={handleRetry}
                onRequestDelete={setPendingDelete}
                onUseInChat={useInChat}
              />
            ))}
          </ul>
        )}
      </section>

      <AlertDialog
        isOpen={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open && !isDeleting) setPendingDelete(null);
        }}
        title="Delete this document?"
        description={`"${pendingDelete?.filename ?? ''}" will be removed from your library. This cannot be undone.`}
        cancelLabel="Cancel"
        actionLabel="Delete"
        actionVariant="destructive"
        isActionLoading={isDeleting}
        onAction={handleDelete}
      />
    </div>
  );
}
