import { useState } from 'react';
import { DropdownMenu, DropdownMenuDivider, DropdownMenuItem } from '@astryxdesign/core/DropdownMenu';
import { Spinner } from '@astryxdesign/core/Spinner';
import { Download, Ellipsis, ImageOff, Maximize2, Scaling, Trash2 } from 'lucide-react';
import { GalleryImageItem } from '@/types/workspace';
import '@/styles/pages/images-library.css';

const MENU_ICON = { size: 15, strokeWidth: 1.75 } as const;
/** Server limit for resized images (see backend upscaler): width * height * 4 must stay within 2048 x 2048. */
const MAX_UPSCALE_PIXELS = 2048 * 2048;
const UPSCALE_FACTOR = 2;

export function canUpscale(img: GalleryImageItem): boolean {
  return img.width * img.height * UPSCALE_FACTOR * UPSCALE_FACTOR <= MAX_UPSCALE_PIXELS;
}

export type ImageActions = {
  onOpen: (img: GalleryImageItem) => void;
  onUpscale: (img: GalleryImageItem) => void;
  onRequestDelete: (img: GalleryImageItem) => void;
};

type TileProps = ImageActions & {
  img: GalleryImageItem;
  isUpscaling: boolean;
  isBusy: boolean;
};

function GalleryTile({ img, isUpscaling, isBusy, onOpen, onUpscale, onRequestDelete }: TileProps) {
  const [hasFailed, setHasFailed] = useState(false);

  return (
    <li className="il-tile">
      <button type="button" className="il-tile-open" onClick={() => onOpen(img)}>
        {hasFailed ? (
          <span className="il-tile-missing">
            <ImageOff size={20} strokeWidth={1.75} aria-hidden="true" />
            <span className="il-sr">Image unavailable: </span>
          </span>
        ) : (
          <img
            src={img.file_url}
            alt={img.prompt}
            width={img.width || 1024}
            height={img.height || 1024}
            loading="lazy"
            decoding="async"
            className="il-tile-img"
            onError={() => setHasFailed(true)}
          />
        )}
        <span className="il-tile-caption" aria-hidden="true">
          {img.prompt}
        </span>
        {hasFailed && <span className="il-sr">{img.prompt}</span>}
      </button>

      {isUpscaling && (
        <div className="il-tile-busy" role="status">
          <Spinner size="sm" shade="inherit" />
          <span>Resizing</span>
        </div>
      )}

      <DropdownMenu
        menuWidth={196}
        placement="below"
        alignment="end"
        renderTrigger={(triggerProps) => (
          <button
            type="button"
            {...triggerProps}
            className="il-icon-btn il-tile-menu"
            aria-label={`Options for image: ${img.prompt}`}
          >
            <Ellipsis size={16} strokeWidth={2} aria-hidden="true" />
          </button>
        )}
      >
        <DropdownMenuItem icon={<Maximize2 {...MENU_ICON} />} label="Open" onClick={() => onOpen(img)} />
        <DropdownMenuItem
          icon={<Download {...MENU_ICON} />}
          label="Download"
          onClick={() => downloadImage(img)}
        />
        <DropdownMenuItem
          icon={<Scaling {...MENU_ICON} />}
          label="Make a 2x copy"
          isDisabled={isBusy || !canUpscale(img)}
          onClick={() => onUpscale(img)}
        />
        <DropdownMenuDivider />
        <DropdownMenuItem
          icon={<Trash2 {...MENU_ICON} />}
          label="Delete"
          variant="destructive"
          onClick={() => onRequestDelete(img)}
        />
      </DropdownMenu>
    </li>
  );
}

/** Saves an image through a temporary same-origin link. */
export function downloadImage(img: GalleryImageItem) {
  const link = document.createElement('a');
  link.href = img.file_url;
  link.download = `jruva-image-${img.id.slice(0, 8)}.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

type GridProps = ImageActions & {
  images: GalleryImageItem[];
  upscalingId: string | null;
  pendingTile: boolean;
};

/** Square tiles that open a preview; every action is also in the tile menu. */
export function ImageGalleryGrid({ images, upscalingId, pendingTile, ...actions }: GridProps) {
  return (
    <ul className="il-grid" aria-label="Your images">
      {pendingTile && (
        <li className="il-tile il-tile--pending" aria-hidden="true">
          <div className="il-tile-shimmer" />
        </li>
      )}
      {images.map((img) => (
        <GalleryTile
          key={img.id}
          img={img}
          isUpscaling={upscalingId === img.id}
          isBusy={upscalingId !== null}
          {...actions}
        />
      ))}
    </ul>
  );
}

/** Placeholder tiles shown while the gallery loads for the first time. */
export function GallerySkeleton() {
  return (
    <ul className="il-grid" aria-hidden="true">
      {Array.from({ length: 6 }, (_, i) => (
        <li key={i} className="il-tile il-tile--pending">
          <div className="il-tile-shimmer" />
        </li>
      ))}
    </ul>
  );
}
