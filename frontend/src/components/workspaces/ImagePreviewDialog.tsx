import { Dialog, DialogHeader } from '@astryxdesign/core/Dialog';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { ChevronLeft, ChevronRight, Download, MessageSquare, Scaling, Trash2 } from 'lucide-react';
import { GalleryImageItem } from '@/types/workspace';
import { canUpscale, downloadImage } from './ImageGalleryGrid';
import { ICON } from './WorkspacePageParts';
import '@/styles/pages/images-library.css';

const STYLE_LABELS: Record<string, string> = {
  photorealistic: 'Photo',
  cinematic: 'Cinematic',
  architecture: 'Architecture',
  cyberpunk: 'Cyberpunk',
  anime: 'Anime',
};

export function styleLabel(style: string | undefined): string {
  if (!style) return 'Photo';
  return STYLE_LABELS[style] ?? style;
}

type Props = {
  image: GalleryImageItem | null;
  position: { index: number; total: number } | null;
  isUpscaling: boolean;
  onClose: () => void;
  onStep: (delta: -1 | 1) => void;
  onUpscale: (img: GalleryImageItem) => void;
  onRequestDelete: (img: GalleryImageItem) => void;
  onSendToChat?: (img: GalleryImageItem) => void;
};

/** Full-size view of one image with its prompt, details and actions. Arrow keys move between images. */
export function ImagePreviewDialog({
  image,
  position,
  isUpscaling,
  onClose,
  onStep,
  onUpscale,
  onRequestDelete,
  onSendToChat,
}: Props) {
  const hasPrev = position !== null && position.index > 0;
  const hasNext = position !== null && position.index < position.total - 1;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest('[role="menu"], input, textarea')) return;
    if (e.key === 'ArrowLeft' && hasPrev) onStep(-1);
    if (e.key === 'ArrowRight' && hasNext) onStep(1);
  };

  const pager = position && position.total > 1 && (
    <div className="il-pager">
      <Tooltip content="Previous (Left arrow)" placement="below">
        <button
          type="button"
          className="il-icon-btn"
          onClick={() => onStep(-1)}
          disabled={!hasPrev}
          aria-label="Previous image"
        >
          <ChevronLeft {...ICON} aria-hidden="true" />
        </button>
      </Tooltip>
      <span className="il-pager-count">
        {position.index + 1} of {position.total}
      </span>
      <Tooltip content="Next (Right arrow)" placement="below">
        <button
          type="button"
          className="il-icon-btn"
          onClick={() => onStep(1)}
          disabled={!hasNext}
          aria-label="Next image"
        >
          <ChevronRight {...ICON} aria-hidden="true" />
        </button>
      </Tooltip>
    </div>
  );

  return (
    <Dialog
      isOpen={image !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      width={920}
      maxHeight="92vh"
      purpose="info"
      className="il-preview"
    >
      {image && (
        <div className="il-preview-body" onKeyDown={handleKeyDown}>
          <DialogHeader title="Image" onOpenChange={() => onClose()} endContent={pager || undefined} />
          <div className="il-preview-split">
            <div className="il-preview-media">
              <img
                src={image.file_url}
                alt={image.prompt}
                width={image.width || 1024}
                height={image.height || 1024}
                className="il-preview-img"
              />
            </div>

            <div className="il-preview-info">
              <section>
                <h3 className="il-label">Prompt</h3>
                <p className="il-prompt-text">{image.prompt}</p>
              </section>

              {image.enhanced_prompt && image.enhanced_prompt !== image.prompt && (
                <details className="il-enhanced">
                  <summary>Improved prompt</summary>
                  <p>{image.enhanced_prompt}</p>
                </details>
              )}

              <dl className="il-meta">
                <div>
                  <dt>Size</dt>
                  <dd>
                    {image.width} x {image.height}
                  </dd>
                </div>
                <div>
                  <dt>Style</dt>
                  <dd>{styleLabel(image.style)}</dd>
                </div>
              </dl>

              <div className="il-preview-actions">
                <button type="button" className="il-btn il-btn--primary" onClick={() => downloadImage(image)}>
                  <Download {...ICON} aria-hidden="true" />
                  Download
                </button>
                {onSendToChat && (
                  <button type="button" className="il-btn" onClick={() => onSendToChat(image)}>
                    <MessageSquare {...ICON} aria-hidden="true" />
                    Discuss in chat
                  </button>
                )}
                <button
                  type="button"
                  className="il-btn"
                  onClick={() => onUpscale(image)}
                  disabled={isUpscaling || !canUpscale(image)}
                  aria-busy={isUpscaling}
                >
                  <Scaling {...ICON} aria-hidden="true" />
                  {isUpscaling ? 'Making 2x copy…' : 'Make a 2x copy'}
                </button>
                <button type="button" className="il-btn il-btn--danger" onClick={() => onRequestDelete(image)}>
                  <Trash2 {...ICON} aria-hidden="true" />
                  Delete
                </button>
              </div>
              {!canUpscale(image) && <p className="il-hint">This image is already at the largest size.</p>}
            </div>
          </div>
        </div>
      )}
    </Dialog>
  );
}
