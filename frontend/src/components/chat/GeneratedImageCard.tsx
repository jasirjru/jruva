'use client';

import React, { useState, useEffect, useRef } from 'react';
import { haptics } from '@/utils/haptics';
import { isSafeImageUrl } from '@/utils/safeImageUrl';

interface GeneratedImageCardProps {
  src: string;
  alt?: string;
  errorType?: 'timeout' | 'rate_limited' | 'server_error' | 'content_filter' | 'network_error' | string;
  errorMessage?: string;
  tier?: string;
}

export const GeneratedImageCard: React.FC<GeneratedImageCardProps> = ({
  src,
  alt = 'Generated Image',
  errorType: initialErrorType,
  errorMessage: initialErrorMessage,
  tier,
}) => {
  // Check if src is an error protocol token: error:<type>:<message>
  const isDirectError = src.startsWith('error:') || src.startsWith('urn:error:');
  let inlineErrorType = initialErrorType;
  let inlineErrorMessage = initialErrorMessage;

  if (isDirectError) {
    const rawError = src.replace(/^(error|urn:error):/, '');
    const colonIdx = rawError.indexOf(':');
    if (colonIdx !== -1) {
      inlineErrorType = inlineErrorType || rawError.slice(0, colonIdx);
      inlineErrorMessage = inlineErrorMessage || decodeURIComponent(rawError.slice(colonIdx + 1));
    } else {
      inlineErrorType = inlineErrorType || rawError;
    }
  }

  const [currentSrc, setCurrentSrc] = useState(src);
  const [isLoading, setIsLoading] = useState(!isDirectError);
  const [hasError, setHasError] = useState(isDirectError);
  const [errorType, setErrorType] = useState<string | undefined>(inlineErrorType);
  const [errorMessage, setErrorMessage] = useState<string | undefined>(inlineErrorMessage);
  const [isLightboxOpen, setIsLightboxOpen] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [isUpscaling, setIsUpscaling] = useState(false);
  const [isUpscaled, setIsUpscaled] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isErrorToken = currentSrc.startsWith('error:') || currentSrc.startsWith('urn:error:');
  // Model output can carry any URL; only load images from trusted sources.
  const isBlocked = !isErrorToken && !isSafeImageUrl(currentSrc);

  // Watchdog timer: If image takes > 28s to load, trigger error state
  useEffect(() => {
    if (currentSrc.startsWith('error:') || currentSrc.startsWith('urn:error:')) {
      setIsLoading(false);
      setHasError(true);
      return;
    }
    if (!isSafeImageUrl(currentSrc)) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setHasError(false);

    if (timeoutRef.current) clearTimeout(timeoutRef.current);

    timeoutRef.current = setTimeout(() => {
      if (isLoading) {
        setIsLoading(false);
        setHasError(true);
        setErrorType('timeout');
        setErrorMessage('The stored image could not be loaded in time. Please retry.');
      }
    }, 28000);

    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [currentSrc]);

  const handleImageLoad = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsLoading(false);
    setHasError(false);
  };

  const handleImageError = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsLoading(false);
    setHasError(true);
    if (!errorType) {
      setErrorType('network_error');
      setErrorMessage('Jruva could not load this image. Please retry.');
    }
  };

  const handleRetry = (e: React.MouseEvent) => {
    e.stopPropagation();
    haptics.click();
    setIsLoading(true);
    setHasError(false);
    setErrorType(undefined);
    setErrorMessage(undefined);
    setRetryCount((prev) => prev + 1);

    // Do not retry if request was blocked by safety policy or is unavailable
    if (errorType === 'content_filter' || errorType === 'unavailable' || currentSrc.startsWith('error:') || currentSrc.startsWith('urn:error:')) {
      setIsLoading(false);
      setHasError(true);
      return;
    }

    // Append fresh cache-busting seed
    const newSeed = Math.floor(Math.random() * 9000000) + 1000000;
    let updatedUrl = currentSrc;
    if (updatedUrl.includes('seed=')) {
      updatedUrl = updatedUrl.replace(/seed=\d+/, `seed=${newSeed}`);
    } else {
      updatedUrl += (updatedUrl.includes('?') ? '&' : '?') + `seed=${newSeed}`;
    }
    setCurrentSrc(updatedUrl);
  };

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();
    haptics.click();
    if (!isSafeImageUrl(currentSrc)) return;
    try {
      const response = await fetch(currentSrc);
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      const cleanName = (alt || 'generated-image')
        .slice(0, 32)
        .replace(/[^a-zA-Z0-9_-]/g, '_')
        .toLowerCase();
      link.download = `${cleanName}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error('Image download failed:', err);
    }
  };

  const handleCopyPrompt = (e: React.MouseEvent) => {
    e.stopPropagation();
    haptics.click();
    navigator.clipboard.writeText(alt || currentSrc);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleEditClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    haptics.click();
    window.dispatchEvent(
      new CustomEvent('jru:edit-image', {
        detail: {
          prompt: `Modify this image: ${alt || ''}`,
          imageUrl: currentSrc,
        },
      })
    );
  };

  const handleUpscale = async (e: React.MouseEvent) => {
    e.stopPropagation();
    haptics.click();
    if (isUpscaling || isUpscaled) return;

    setIsUpscaling(true);
    try {
      const res = await fetch('/api/v1/image/upscale', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: currentSrc, scale: 2 }),
      });
      const data = await res.json();
      if (!res.ok || !data.success) { window.alert(data.detail || data.error || "Image resizing failed"); return; }
      if (data.success && data.upscaled_url) {
        setCurrentSrc(data.upscaled_url);
        setIsUpscaled(true);
      }
    } catch (err) {
      console.error('Upscale error:', err);
    } finally {
      setIsUpscaling(false);
    }
  };

  const downloadLabel = isUpscaled ? 'Download resized image' : 'Download image';
  const copyLabel = isCopied ? 'Prompt copied' : 'Copy prompt';
  const upscaleLabel = isUpscaling ? 'Upscaling...' : isUpscaled ? 'Upscaled 2x' : 'Upscale 2x';

  if (isBlocked) {
    return (
      <div className="generated-image-card-container">
        <div className="generated-image-error-card" role="img" aria-label={alt}>
          <div className="image-error-title">Image blocked</div>
          <div className="image-error-desc">
            {alt} (this image comes from an untrusted source and was not loaded)
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="generated-image-card-container">
      {/* Visual Image Stage */}
      <div
        className="generated-image-frame"
        onClick={() => {
          if (!hasError && !isLoading) {
            haptics.click();
            setIsLightboxOpen(true);
          }
        }}
        title={!hasError && !isLoading ? 'Click to view full screen' : undefined}
      >
        {/* Loading Skeleton */}
        {isLoading && (
          <div className="generated-image-skeleton">
            <div className="generated-image-spinner" />
            <div className="generated-image-loading-text">
              <span>Creating your image...</span>
              <span className="generated-image-sub-text">Jruva image</span>
            </div>
          </div>
        )}

        {/* Visible Error State (Never silent black box) */}
        {hasError ? (
          <div className="generated-image-error-card">
            <div className="image-error-icon">
              {errorType === 'timeout'
                ? '⏳'
                : errorType === 'rate_limited'
                ? '🚦'
                : errorType === 'content_filter'
                ? '🛡️'
                : errorType === 'unavailable'
                ? '🏛️'
                : errorType === 'network_error'
                ? '📡'
                : '⚠️'}
            </div>
            <div className="image-error-title">
              {errorType === 'timeout'
                ? 'Image Generation Timed Out'
                : errorType === 'rate_limited'
                ? 'Provider Rate Limit Reached (429)'
                : errorType === 'content_filter'
                ? 'Content Policy Notice'
                : errorType === 'unavailable'
                ? 'Image Unavailable'
                : errorType === 'server_error'
                ? 'Provider Server Error (5xx)'
                : errorType === 'network_error'
                ? 'Network Connection Interrupted'
                : 'Image Generation Delayed or Failed'}
            </div>
            <div className="image-error-desc">
              {errorMessage || 'This visual request is currently unavailable.'}
            </div>
            {tier && (
              <div style={{ fontSize: '0.74rem', color: 'rgba(255, 255, 255, 0.65)', marginTop: '2px', marginBottom: '4px' }}>
                Pipeline: <strong>Tier {tier}</strong>
              </div>
            )}
            {errorType !== 'content_filter' && errorType !== 'unavailable' && (
              <button
                type="button"
                className="image-retry-btn"
                onClick={handleRetry}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
                </svg>
                <span>Retry Generation {retryCount > 0 ? `(${retryCount + 1})` : ''}</span>
              </button>
            )}
          </div>
        ) : (
          <img
            src={currentSrc}
            alt={alt}
            className={`generated-image-img ${isLoading ? 'loading' : 'loaded'}`}
            onLoad={handleImageLoad}
            onError={handleImageError}
            loading="eager"
          />
        )}
      </div>

      {/* One quiet row of icon actions below the image (always visible, also on touch) */}
      {!isLoading && !hasError && (
        <div className="gic-actions" role="group" aria-label="Image actions">
          <button
            type="button"
            className="gic-action-btn"
            onClick={handleDownload}
            title={downloadLabel}
            aria-label={downloadLabel}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
          </button>

          <button
            type="button"
            className={`gic-action-btn${isCopied ? ' is-active' : ''}`}
            onClick={handleCopyPrompt}
            title={copyLabel}
            aria-label={copyLabel}
          >
            {isCopied ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
              </svg>
            )}
          </button>

          <button
            type="button"
            className="gic-action-btn"
            onClick={handleEditClick}
            title="Edit image"
            aria-label="Edit image"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
            </svg>
          </button>

          <button
            type="button"
            className={`gic-action-btn${isUpscaled ? ' is-active' : ''}`}
            onClick={handleUpscale}
            disabled={isUpscaling || isUpscaled}
            aria-busy={isUpscaling}
            title={upscaleLabel}
            aria-label={upscaleLabel}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="15 3 21 3 21 9" />
              <polyline points="9 21 3 21 3 15" />
              <line x1="21" y1="3" x2="14" y2="10" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          </button>
        </div>
      )}

      {/* Lightbox Fullscreen Preview */}
      {isLightboxOpen && (
        <div className="image-lightbox-backdrop" onClick={() => setIsLightboxOpen(false)}>
          <div className="image-lightbox-modal" onClick={(e) => e.stopPropagation()}>
            <img src={currentSrc} alt={alt} className="image-lightbox-img" />
            <div className="image-lightbox-header">
              <span className="image-lightbox-caption">{alt}</span>
              <div className="image-lightbox-btns">
                <button
                  type="button"
                  className="lightbox-btn"
                  onClick={handleDownload}
                  title="Download"
                >
                  Download
                </button>
                <button
                  type="button"
                  className="lightbox-btn close"
                  onClick={() => setIsLightboxOpen(false)}
                >
                  ✕
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
