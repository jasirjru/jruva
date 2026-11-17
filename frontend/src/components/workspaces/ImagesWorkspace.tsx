'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { AlertDialog } from '@astryxdesign/core/AlertDialog';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { ImageIcon, LogIn, RefreshCw } from 'lucide-react';
import { GalleryImageItem } from '@/types/workspace';
import { EmptyState, ICON, Notice, NoticeTone, PageHeader, readErrorDetail } from './WorkspacePageParts';
import { GallerySkeleton, ImageGalleryGrid } from './ImageGalleryGrid';
import { GenerateSettings, ImageGenerateForm } from './ImageGenerateForm';
import { ImagePreviewDialog } from './ImagePreviewDialog';
import '@/styles/pages/images-library.css';

interface ImagesWorkspaceProps {
  onBackToChat: () => void;
  onSendToChat?: (imageUrl: string, prompt: string) => void;
}

type GalleryStatus = 'loading' | 'ready' | 'error' | 'signed-out';
type PageNotice = { tone: NoticeTone; text: string } | null;

const IMAGE_SIZE = 1024; // Only square generation is configured on the server.
const SUCCESS_NOTICE_MS = 4000;
const GENERATE_FAILED = 'The image could not be created. Try again, or change the prompt.';

function isUsableImage(img: GalleryImageItem | null | undefined): img is GalleryImageItem {
  return Boolean(img && img.file_url && !img.file_url.startsWith('error:'));
}

/** Persist before POST so a reload or lost response reuses the same intent; keeps only a digest, never the prompt. */
async function idempotencyKeyFor(body: string): Promise<{ storageKey: string; intentKey: string }> {
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(body));
  const digest = Array.from(new Uint8Array(hash))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
  const storageKey = `jruva:image-intent:${digest}`;
  const intentKey = sessionStorage.getItem(storageKey) || crypto.randomUUID();
  sessionStorage.setItem(storageKey, intentKey);
  return { storageKey, intentKey };
}

export function ImagesWorkspace({ onBackToChat, onSendToChat }: ImagesWorkspaceProps) {
  const [images, setImages] = useState<GalleryImageItem[]>([]);
  const { isAuthenticated } = useAuth();
  const [status, setStatus] = useState<GalleryStatus>('loading');
  // Guests are signed out by definition; skip the gallery request the API would answer with 401.
  const shownStatus: GalleryStatus = isAuthenticated ? status : 'signed-out';
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [settings, setSettings] = useState<GenerateSettings>({
    prompt: '',
    style: 'photorealistic',
    model: 'flux',
    enhance: true,
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [upscalingId, setUpscalingId] = useState<string | null>(null);
  const [selected, setSelected] = useState<GalleryImageItem | null>(null);
  const [pendingDelete, setPendingDelete] = useState<GalleryImageItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [notice, setNotice] = useState<PageNotice>(null);
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showNotice = useCallback((next: PageNotice) => {
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
    setNotice(next);
    if (next?.tone === 'success') noticeTimer.current = setTimeout(() => setNotice(null), SUCCESS_NOTICE_MS);
  }, []);

  useEffect(() => () => {
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
  }, []);

  const fetchGallery = useCallback(async (): Promise<GalleryImageItem[] | null> => {
    setIsRefreshing(true);
    try {
      const res = await fetch('/api/v1/image/gallery');
      if (res.status === 401 || res.status === 403) {
        setImages([]);
        setSelected(null);
        setStatus('signed-out');
        return null;
      }
      if (!res.ok) {
        setStatus('error');
        return null;
      }
      const data = await res.json();
      const valid = ((data.images || []) as GalleryImageItem[]).filter(isUsableImage);
      setImages(valid);
      setStatus('ready');
      return valid;
    } catch {
      setStatus('error');
      return null;
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    // Initial load; fetchGallery only sets state after the request resolves.
    if (!isAuthenticated) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void fetchGallery();
  }, [fetchGallery, isAuthenticated]);

  const handleGenerate = async () => {
    const prompt = settings.prompt.trim();
    if (!prompt || isGenerating) return;
    setIsGenerating(true);
    setGenerateError(null);
    try {
      const body = JSON.stringify({
        prompt,
        model: settings.model,
        style: settings.style,
        enhance: settings.enhance,
        width: IMAGE_SIZE,
        height: IMAGE_SIZE,
        tier: 'B',
      });
      const { storageKey, intentKey } = await idempotencyKeyFor(body);
      const res = await fetch('/api/v1/image/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Idempotency-Key': intentKey },
        body,
      });
      if (res.status === 401) throw new Error('Sign in to create images.');
      if (!res.ok) throw new Error(await readErrorDetail(res, GENERATE_FAILED));

      const data = await res.json();
      const isErrorUrl = typeof data.image_url === 'string' && data.image_url.startsWith('error:');
      if (!data.success || isErrorUrl) {
        let msg = data.error_message || data.detail;
        if (!msg && isErrorUrl) msg = decodeURIComponent(data.image_url.replace(/^error:[^:]*:/, ''));
        throw new Error(msg || GENERATE_FAILED);
      }

      sessionStorage.removeItem(storageKey);
      setSettings((prev) => ({ ...prev, prompt: '' }));
      showNotice({ tone: 'success', text: 'Image saved to your gallery.' });
      const fresh = await fetchGallery();
      const created = fresh?.find((img) => img.id === data.image_id);
      setSelected(
        created ?? {
          id: data.image_id,
          prompt: data.original_prompt || prompt,
          enhanced_prompt: data.enhanced_prompt,
          model: data.model,
          tier: data.tier,
          style: settings.style,
          width: data.width,
          height: data.height,
          file_url: data.image_url,
          created_at: new Date().toISOString(),
        },
      );
    } catch (err) {
      setGenerateError(err instanceof Error && err.message ? err.message : GENERATE_FAILED);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleUpscale = async (img: GalleryImageItem) => {
    if (upscalingId) return;
    setUpscalingId(img.id);
    showNotice(null);
    try {
      const res = await fetch('/api/v1/image/upscale', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: img.file_url, scale: 2 }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.detail || data.error || 'The 2x copy could not be made. Try again.');
      setSelected(null);
      await fetchGallery();
      showNotice({ tone: 'success', text: 'A 2x copy was added to your gallery.' });
    } catch (err) {
      showNotice({ tone: 'error', text: err instanceof Error ? err.message : 'The 2x copy could not be made. Try again.' });
    } finally {
      setUpscalingId(null);
    }
  };

  const handleDelete = async () => {
    if (!pendingDelete) return;
    const target = pendingDelete;
    setIsDeleting(true);
    try {
      const res = await fetch(`/api/v1/image/${target.id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(await readErrorDetail(res, 'The image could not be deleted. Try again.'));
      setImages((prev) => prev.filter((img) => img.id !== target.id));
      setSelected((current) => (current?.id === target.id ? null : current));
      showNotice({ tone: 'success', text: 'Image deleted.' });
    } catch (err) {
      showNotice({ tone: 'error', text: err instanceof Error ? err.message : 'The image could not be deleted. Try again.' });
    } finally {
      setIsDeleting(false);
      setPendingDelete(null);
    }
  };

  const selectedIndex = selected ? images.findIndex((img) => img.id === selected.id) : -1;
  const stepPreview = (delta: -1 | 1) => {
    const next = images[selectedIndex + delta];
    if (next) setSelected(next);
  };

  const countLabel = shownStatus === 'ready' && images.length > 0 ? String(images.length) : null;

  return (
    <div className="il-page">
      <PageHeader title="Images" subtitle="Create and keep your images." onBack={onBackToChat} />

      <ImageGenerateForm
        settings={settings}
        onChange={setSettings}
        onSubmit={handleGenerate}
        isGenerating={isGenerating}
        error={generateError}
        onDismissError={() => setGenerateError(null)}
      />

      <section className="il-section" aria-labelledby="il-gallery-title">
        <div className="il-section-head">
          <h2 id="il-gallery-title" className="il-card-title">
            Gallery
            {countLabel && <span className="il-count">{countLabel}</span>}
          </h2>
          <Tooltip content="Refresh gallery" placement="below">
            <button
              type="button"
              className="il-icon-btn"
              onClick={() => void fetchGallery()}
              disabled={isRefreshing}
              aria-label="Refresh gallery"
            >
              <RefreshCw {...ICON} className={isRefreshing ? 'il-spin' : undefined} aria-hidden="true" />
            </button>
          </Tooltip>
        </div>

        {notice && (
          <Notice tone={notice.tone} onDismiss={() => showNotice(null)}>
            {notice.text}
          </Notice>
        )}

        {shownStatus === 'loading' && <GallerySkeleton />}
        {shownStatus === 'signed-out' && (
          <EmptyState icon={<LogIn size={22} strokeWidth={1.75} />} title="Sign in to see your gallery">
            Your images are private to your account.
          </EmptyState>
        )}
        {shownStatus === 'error' && images.length === 0 && (
          <EmptyState
            icon={<ImageIcon size={22} strokeWidth={1.75} />}
            title="Your gallery could not be loaded"
            action={
              <button type="button" className="il-btn" onClick={() => void fetchGallery()} disabled={isRefreshing}>
                <RefreshCw {...ICON} aria-hidden="true" />
                Try again
              </button>
            }
          >
            Check your connection, then try again.
          </EmptyState>
        )}
        {shownStatus === 'ready' && images.length === 0 && !isGenerating && (
          <EmptyState icon={<ImageIcon size={22} strokeWidth={1.75} />} title="No images yet">
            Describe an image above and it will appear here.
          </EmptyState>
        )}
        {(images.length > 0 || isGenerating) && shownStatus !== 'signed-out' && shownStatus !== 'loading' && (
          <ImageGalleryGrid
            images={images}
            upscalingId={upscalingId}
            pendingTile={isGenerating}
            onOpen={setSelected}
            onUpscale={handleUpscale}
            onRequestDelete={setPendingDelete}
          />
        )}
      </section>

      <ImagePreviewDialog
        image={isUsableImage(selected) ? selected : null}
        position={selectedIndex >= 0 ? { index: selectedIndex, total: images.length } : null}
        isUpscaling={selected !== null && upscalingId === selected.id}
        onClose={() => setSelected(null)}
        onStep={stepPreview}
        onUpscale={handleUpscale}
        onRequestDelete={setPendingDelete}
        onSendToChat={
          onSendToChat
            ? (img) => {
                onSendToChat(img.file_url, img.prompt);
                onBackToChat();
              }
            : undefined
        }
      />

      <AlertDialog
        isOpen={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open && !isDeleting) setPendingDelete(null);
        }}
        title="Delete this image?"
        description="It will be removed from your gallery. This cannot be undone."
        cancelLabel="Cancel"
        actionLabel="Delete"
        actionVariant="destructive"
        isActionLoading={isDeleting}
        onAction={handleDelete}
      />
    </div>
  );
}
