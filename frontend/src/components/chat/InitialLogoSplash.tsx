'use client';

import React, { useState, useEffect, useRef } from 'react';
import { JruLogo } from './JruLogo';

interface InitialLogoSplashProps {
  onComplete: () => void;
}

/**
 * Cinematic Initial Opening Splash Sequence
 * Features the Jruva mark rotating at the exact center point
 * with perfectly aligned concentric celestial orbital rings.
 */
export const InitialLogoSplash: React.FC<InitialLogoSplashProps> = ({ onComplete }) => {
  const [isClosing, setIsClosing] = useState(false);
  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  });

  useEffect(() => {
    // Already played in this tab, or reduced motion: the head script hid it before paint.
    if (document.documentElement.hasAttribute('data-splash-seen')) {
      onCompleteRef.current();
      return;
    }
    try {
      sessionStorage.setItem('jru_splash_seen', '1');
    } catch {
      // Storage blocked: the splash simply plays again next time.
    }

    // In-place rotation sequence (~1.9s)
    const timer = setTimeout(() => {
      setIsClosing(true);
    }, 1900);

    // Seamless unmount at 2.25s after silky fade-out
    const exitTimer = setTimeout(() => {
      onCompleteRef.current();
    }, 2250);

    // Interactive skip on keypress or click
    const handleSkip = () => {
      setIsClosing(true);
      setTimeout(() => onCompleteRef.current(), 220);
    };

    window.addEventListener('keydown', handleSkip, { once: true });

    return () => {
      clearTimeout(timer);
      clearTimeout(exitTimer);
      window.removeEventListener('keydown', handleSkip);
    };
  }, []);

  return (
    <div
      className={`jru-initial-splash-overlay ${isClosing ? 'splash-closing' : ''}`}
      onClick={() => {
        setIsClosing(true);
        setTimeout(onComplete, 220);
      }}
      aria-label="Opening Jruva"
    >
      <div className="jru-splash-content">
        {/* Dedicated Logo Stage with Perfectly Concentric Orbital Rings */}
        <div className="jru-splash-logo-stage">
          {/* Ambient Backlight Aura */}
          <div className="jru-splash-ambient-glow" />

          {/* Counter-Rotating Celestial Orbital Rings centered on Logo */}
          <div className="jru-splash-orbit-outer">
            <div className="jru-splash-orbit-pip" />
          </div>
          <div className="jru-splash-orbit-inner" />

          {/* Core Jruva Logo Rotating In Place (Exact Centroid Pivot) */}
          <div className="jru-splash-logo-core">
            <JruLogo size={102} glow={true} variant="render" />
          </div>
        </div>

        {/* Clean Brand Title */}
        <div className="jru-splash-text-container">
          <div className="jru-splash-brand-title">Jruva</div>
        </div>
      </div>
    </div>
  );
};
