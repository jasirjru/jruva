'use client';

import React from 'react';

interface JruLogoProps {
  size?: number;
  className?: string;
  glow?: boolean;
  variant?: 'render' | 'vector';
  /** 'hero' → dramatic entry + continuous breath. 'chat' → gentle pulse. 'static' → no animation. */
  animMode?: 'hero' | 'chat' | 'static';
}

/**
 * Brand identity mark for jru — S+++ Tier Animated Version.
 * Warm luxury palette: Terracotta #B46C55 | Chestnut #62362C | Espresso #190502
 * Taupe #A48272 | Sand #AB9D91 | Alabaster #DEDBD1 | Ivory #FDF0E9
 *
 * Animation modes:
 *  - 'hero'   → Dramatic scale+rotate entry, then continuous breathing glow
 *  - 'chat'   → Subtle breathing glow only (used inside chat messages)
 *  - 'static' → No animation (e.g., favicon-like contexts)
 */
export const JruLogo: React.FC<JruLogoProps> = ({
  size = 32,
  className = '',
  glow = false,
  variant = 'render',
  animMode = 'static',
}) => {
  const animClass =
    animMode === 'hero'
      ? 'jru-logo-hero-breath'
      : animMode === 'chat'
      ? 'jru-logo-chat-pulse'
      : '';

  if (variant === 'render') {
    return (
      <div
        className={`jru-logo-wrapper ${animClass} ${className}`}
        style={{
          width: size,
          height: size,
          position: 'relative',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          background: 'transparent',
          filter: glow
            ? 'drop-shadow(0 8px 24px rgba(180, 108, 85, 0.4)) drop-shadow(0 2px 8px rgba(79, 56, 44, 0.25))'
            : 'drop-shadow(0 2px 6px rgba(25, 5, 2, 0.12))',
        }}
        aria-label="Jruva logo"
      >
        <img
          src="/jru-icon-256.webp"
          srcSet="/jru-icon-256.webp 1x, /jru-icon-512.webp 2x"
          decoding="async"
          alt="Jruva logo"
          width={size}
          height={size}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            display: 'block',
            userSelect: 'none',
            pointerEvents: 'none',
          }}
          draggable={false}
        />
      </div>
    );
  }

  // ─── Precision Animated Vector Mark ───────────────────────────────────────
  const petalSpinDuration = animMode === 'hero' ? '22s' : '36s';
  const corePulseDuration = animMode === 'hero' ? '2.8s' : '4.2s';
  const sparkDuration     = animMode === 'hero' ? '3.5s' : '5s';
  const crossRayDuration  = animMode === 'hero' ? '2.2s' : '3.5s';

  return (
    <div
      className={`jru-logo-wrapper ${animClass} ${className}`}
      style={{
        width: size,
        height: size,
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        background: 'transparent',
        filter: glow
          ? 'drop-shadow(0 6px 20px rgba(180, 108, 85, 0.42)) drop-shadow(0 2px 8px rgba(98, 54, 44, 0.28))'
          : 'drop-shadow(0 1.5px 5px rgba(25, 5, 2, 0.14))',
      }}
      aria-label="Jruva logo"
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 64 64"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ overflow: 'visible', display: 'block' }}
      >
        <defs>
          {/* Petal 1: Warm Terracotta Gradient */}
          <linearGradient id="jru-warm-terracotta" x1="20" y1="6" x2="44" y2="42" gradientUnits="userSpaceOnUse">
            <stop offset="0%"   stopColor="#E28F77" />
            <stop offset="35%"  stopColor="#B46C55" />
            <stop offset="78%"  stopColor="#8A4838" />
            <stop offset="100%" stopColor="#62362C" />
          </linearGradient>

          {/* Petal 2: Alabaster & Ivory */}
          <linearGradient id="jru-warm-alabaster" x1="48" y1="18" x2="20" y2="52" gradientUnits="userSpaceOnUse">
            <stop offset="0%"   stopColor="#FFFFFF" />
            <stop offset="25%"  stopColor="#FDF0E9" />
            <stop offset="65%"  stopColor="#DEDBD1" />
            <stop offset="100%" stopColor="#A48272" />
          </linearGradient>

          {/* Petal 3: Espresso Chestnut */}
          <linearGradient id="jru-warm-espresso" x1="28" y1="56" x2="30" y2="18" gradientUnits="userSpaceOnUse">
            <stop offset="0%"   stopColor="#AB9D91" />
            <stop offset="22%"  stopColor="#62362C" />
            <stop offset="65%"  stopColor="#4F382C" />
            <stop offset="100%" stopColor="#190502" />
          </linearGradient>

          {/* Specular Rim */}
          <linearGradient id="jru-warm-specular" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%"   stopColor="#FFFFFF" stopOpacity="0.9" />
            <stop offset="50%"  stopColor="#FDF0E9" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#DEDBD1" stopOpacity="0" />
          </linearGradient>

          {/* Edge Contour */}
          <linearGradient id="jru-warm-edge-contour" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%"   stopColor="#190502" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#62362C" stopOpacity="0.05" />
          </linearGradient>

          {/* Core Glow Radial */}
          <radialGradient id="jru-warm-core-glow" cx="32" cy="32" r="14" gradientUnits="userSpaceOnUse">
            <stop offset="0%"   stopColor="#FFFFFF" stopOpacity="0.95" />
            <stop offset="35%"  stopColor="#FDF0E9" stopOpacity="0.7" />
            <stop offset="70%"  stopColor="#B46C55" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#B46C55" stopOpacity="0" />
          </radialGradient>

          {/* Soft Bloom Filter */}
          <filter id="jru-warm-bloom" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="1.6" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>

          {/* Animated glow filter for breathing */}
          <filter id="jru-warm-glow-anim" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="2.5" result="gblur" />
            <feComposite in="SourceGraphic" in2="gblur" operator="over" />
          </filter>
        </defs>

        {/* Ambient Warm Core Halo */}
        <circle
          cx="32"
          cy="32"
          r="18"
          fill="url(#jru-warm-core-glow)"
          opacity={glow ? 0.65 : 0.35}
        />

        {/* ─── Tri-Chiral Rotational Petals (animated spin) ─── */}
        <g
          style={{
            transformOrigin: '32px 32px',
            animation: animMode !== 'static'
              ? `jruPetalSpin ${petalSpinDuration} linear infinite`
              : 'none',
          }}
        >
          {/* Petal 1: Top Terracotta */}
          <g transform="rotate(0 32 32)">
            <path
              d="M 32 32 C 28 18, 20 10, 32 8 C 45 6, 52 18, 44 30 C 40 36, 33 33, 32 32 Z"
              fill="url(#jru-warm-terracotta)"
              stroke="url(#jru-warm-edge-contour)"
              strokeWidth="0.8"
            />
            <path
              d="M 31 10 C 42 8, 49 18, 42 28"
              stroke="url(#jru-warm-specular)"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </g>

          {/* Petal 2: Right Alabaster Ivory */}
          <g transform="rotate(120 32 32)">
            <path
              d="M 32 32 C 28 18, 20 10, 32 8 C 45 6, 52 18, 44 30 C 40 36, 33 33, 32 32 Z"
              fill="url(#jru-warm-alabaster)"
              stroke="#AB9D91"
              strokeOpacity="0.4"
              strokeWidth="0.8"
            />
            <path
              d="M 31 10 C 42 8, 49 18, 42 28"
              stroke="url(#jru-warm-specular)"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </g>

          {/* Petal 3: Left Espresso Chestnut */}
          <g transform="rotate(240 32 32)">
            <path
              d="M 32 32 C 28 18, 20 10, 32 8 C 45 6, 52 18, 44 30 C 40 36, 33 33, 32 32 Z"
              fill="url(#jru-warm-espresso)"
              stroke="rgba(222, 219, 209, 0.25)"
              strokeWidth="0.7"
            />
            <path
              d="M 31 10 C 42 8, 49 18, 42 28"
              stroke="url(#jru-warm-specular)"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </g>
        </g>

        {/* Core Shadow (pulsing) */}
        <circle
          cx="32"
          cy="32"
          r="5.5"
          fill="#190502"
          style={{
            animation: animMode !== 'static'
              ? `jruCorePulse ${corePulseDuration} ease-in-out infinite`
              : 'none',
            opacity: 0.32,
          }}
        />

        {/* Cross Rays (pulsing) */}
        <ellipse
          cx="32" cy="32" rx="1.2" ry="9.5"
          fill="#FFFFFF"
          style={{
            animation: animMode !== 'static'
              ? `jruCrossRayPulse ${crossRayDuration} ease-in-out infinite`
              : 'none',
            opacity: 0.9,
          }}
        />
        <ellipse
          cx="32" cy="32" rx="9.5" ry="1.2"
          fill="#FFFFFF"
          style={{
            animation: animMode !== 'static'
              ? `jruCrossRayPulse ${crossRayDuration} ease-in-out infinite`
              : 'none',
            animationDelay: '0.4s',
            opacity: 0.9,
          }}
        />

        {/* Diamond Spark (breathing) */}
        <path
          d="M 32 22 Q 32 32 22 32 Q 32 32 32 42 Q 32 32 42 32 Q 32 32 32 22 Z"
          fill="#FFFFFF"
          filter="url(#jru-warm-bloom)"
          style={{
            transformOrigin: '32px 32px',
            animation: animMode !== 'static'
              ? `jruSparkDiamond ${sparkDuration} ease-in-out infinite`
              : 'none',
          }}
        />

        {/* Pinpoint Core */}
        <circle cx="32" cy="32" r="1.5" fill="#FFFFFF" />
      </svg>
    </div>
  );
};
