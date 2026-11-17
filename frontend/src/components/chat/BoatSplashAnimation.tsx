'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { JruLogo } from './JruLogo';

interface BoatSplashAnimationProps {
  onComplete: () => void;
}

/**
 * Cinematic "Boat & Flood" Opening Splash
 * 
 * Timeline:
 *   Phase 0 (0-500ms)     → Water surge builds from left edge
 *   Phase 1 (500-2200ms)  → Boat rides the flood wave carrying the Jruva logo to center
 *   Phase 2 (2200-2800ms) → Boat "drops" logo at exact center (logo detaches & settles)
 *   Phase 3 (2800-3800ms) → Boat continues right + water drains/recedes
 *   Phase 4 (3800-4200ms) → Orbital rings fade in, final composition
 * 
 * Total duration: ~4.2s with skip-on-click
 */
export const BoatSplashAnimation: React.FC<BoatSplashAnimationProps> = ({ onComplete }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState<'water-entry' | 'boat-riding' | 'logo-drop' | 'boat-exit' | 'settled'>('water-entry');
  const [boatX, setBoatX] = useState(-200);
  const [boatY, setBoatY] = useState(0);
  const [boatRotation, setBoatRotation] = useState(0);
  const [logoOpacity, setLogoOpacity] = useState(0);
  const [logoScale, setLogoScale] = useState(0.5);
  const [logoOnBoat, setLogoOnBoat] = useState(true);
  const [logoDropProgress, setLogoDropProgress] = useState(0);
  const [waterLevel, setWaterLevel] = useState(0);
  const [orbitsVisible, setOrbitsVisible] = useState(false);
  const [isExiting, setIsExiting] = useState(false);
  const animFrameRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const skippedRef = useRef(false);

  // Easing functions
  const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);
  const easeInOutQuart = (t: number) => t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2;
  const easeOutElastic = (t: number) => {
    const c4 = (2 * Math.PI) / 3;
    return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  };

  // Water canvas simulation
  const drawWater = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, time: number, wLevel: number, boatXPos: number, phase: string) => {
    ctx.clearRect(0, 0, w, h);
    if (wLevel <= 0) return;

    const waterBaseY = h * (1 - wLevel * 0.55); // Water covers up to 55% of screen at max

    // Deep water layer
    const deepGrad = ctx.createLinearGradient(0, waterBaseY, 0, h);
    deepGrad.addColorStop(0, 'rgba(15, 76, 117, 0.6)');
    deepGrad.addColorStop(0.4, 'rgba(27, 38, 59, 0.75)');
    deepGrad.addColorStop(1, 'rgba(13, 27, 42, 0.85)');
    ctx.fillStyle = deepGrad;
    ctx.fillRect(0, waterBaseY + 30, w, h - waterBaseY);

    // Main wave surface
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= w; x += 2) {
      // Multi-harmonic waves
      const wave1 = Math.sin(x * 0.008 + time * 2.5) * 18 * wLevel;
      const wave2 = Math.sin(x * 0.015 + time * 3.2 + 1.2) * 10 * wLevel;
      const wave3 = Math.sin(x * 0.003 + time * 1.1) * 28 * wLevel;
      // Boat wake effect - creates a V-shaped wake behind the boat
      const distFromBoat = Math.abs(x - boatXPos);
      const wakeEffect = distFromBoat < 200 ? Math.sin((x - boatXPos) * 0.04 + time * 5) * 15 * Math.max(0, 1 - distFromBoat / 200) * wLevel : 0;
      
      const y = waterBaseY + wave1 + wave2 + wave3 + wakeEffect;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();

    const waveGrad = ctx.createLinearGradient(0, waterBaseY - 30, 0, h);
    waveGrad.addColorStop(0, 'rgba(56, 163, 209, 0.7)');
    waveGrad.addColorStop(0.3, 'rgba(27, 100, 140, 0.8)');
    waveGrad.addColorStop(0.7, 'rgba(15, 52, 96, 0.85)');
    waveGrad.addColorStop(1, 'rgba(13, 27, 42, 0.9)');
    ctx.fillStyle = waveGrad;
    ctx.fill();

    // Specular foam crest highlights
    ctx.beginPath();
    for (let x = 0; x <= w; x += 2) {
      const wave1 = Math.sin(x * 0.008 + time * 2.5) * 18 * wLevel;
      const wave2 = Math.sin(x * 0.015 + time * 3.2 + 1.2) * 10 * wLevel;
      const wave3 = Math.sin(x * 0.003 + time * 1.1) * 28 * wLevel;
      const distFromBoat = Math.abs(x - boatXPos);
      const wakeEffect = distFromBoat < 200 ? Math.sin((x - boatXPos) * 0.04 + time * 5) * 15 * Math.max(0, 1 - distFromBoat / 200) * wLevel : 0;
      const y = waterBaseY + wave1 + wave2 + wave3 + wakeEffect;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = `rgba(255, 255, 255, ${0.5 * wLevel})`;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = 'rgba(100, 200, 255, 0.6)';
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Splash spray particles near boat
    if (phase === 'boat-riding' || phase === 'boat-exit') {
      const sprayCount = 12;
      for (let i = 0; i < sprayCount; i++) {
        const sx = boatXPos + (Math.random() - 0.5) * 100;
        const sy = waterBaseY + (Math.random() - 0.8) * 50 * wLevel;
        const sr = 1.5 + Math.random() * 3;
        const sa = 0.3 + Math.random() * 0.4;
        ctx.beginPath();
        ctx.arc(sx, sy, sr, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200, 230, 255, ${sa * wLevel})`;
        ctx.fill();
      }
    }

    // Foam bubbles on surface
    for (let i = 0; i < 20; i++) {
      const bx = (w * (i + Math.sin(time + i * 1.7) * 0.1)) / 20;
      const by = waterBaseY + Math.sin(bx * 0.01 + time * 2) * 15 * wLevel + Math.sin(bx * 0.02 + time * 3) * 8 * wLevel;
      const br = 2 + Math.sin(time * 3 + i) * 1.5;
      ctx.beginPath();
      ctx.arc(bx, by + 5, Math.abs(br), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${0.15 * wLevel})`;
      ctx.fill();
    }
  }, []);

  // Main animation loop
  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    startTimeRef.current = performance.now();
    const centerX = w / 2;

    const TOTAL_DURATION = 4200;
    const PHASE_TIMINGS = {
      waterEntry: { start: 0, end: 500 },
      boatRiding: { start: 300, end: 2200 },
      logoDrop: { start: 2200, end: 2800 },
      boatExit: { start: 2800, end: 3800 },
      settled: { start: 3800, end: TOTAL_DURATION },
    };

    function animate(now: number) {
      if (skippedRef.current) return;

      const elapsed = now - startTimeRef.current;
      const progress = Math.min(1, elapsed / TOTAL_DURATION);

      // Phase determination
      let currentPhase: typeof phase;
      if (elapsed < PHASE_TIMINGS.boatRiding.start) {
        currentPhase = 'water-entry';
      } else if (elapsed < PHASE_TIMINGS.logoDrop.start) {
        currentPhase = 'boat-riding';
      } else if (elapsed < PHASE_TIMINGS.boatExit.start) {
        currentPhase = 'logo-drop';
      } else if (elapsed < PHASE_TIMINGS.settled.start) {
        currentPhase = 'boat-exit';
      } else {
        currentPhase = 'settled';
      }
      setPhase(currentPhase);

      // Water level
      let wl = 0;
      if (elapsed < 500) {
        wl = easeOutCubic(elapsed / 500) * 0.6;
      } else if (elapsed < 2200) {
        wl = 0.6 + easeInOutQuart((elapsed - 500) / 1700) * 0.4;
      } else if (elapsed < 2800) {
        wl = 1.0;
      } else if (elapsed < 3800) {
        wl = 1.0 - easeOutCubic((elapsed - 2800) / 1000) * 1.0;
      } else {
        wl = 0;
      }
      setWaterLevel(wl);

      // Boat position
      let bx = -200;
      let by = 0;
      let bRot = 0;

      if (elapsed >= 300 && elapsed < 2200) {
        const boatProgress = easeInOutQuart((elapsed - 300) / 1900);
        bx = -200 + (centerX + 200) * boatProgress; // -200 to centerX
        // Boat bobbing on waves
        by = Math.sin(elapsed * 0.004) * 8 + Math.sin(elapsed * 0.007) * 5;
        bRot = Math.sin(elapsed * 0.003) * 4 + Math.sin(elapsed * 0.005) * 2;
      } else if (elapsed >= 2200 && elapsed < 2800) {
        // Hold at center briefly then start moving right
        const holdProgress = (elapsed - 2200) / 600;
        bx = centerX + holdProgress * 60;
        by = Math.sin(elapsed * 0.004) * 6;
        bRot = Math.sin(elapsed * 0.003) * 3;
      } else if (elapsed >= 2800 && elapsed < 3800) {
        const exitProgress = easeOutCubic((elapsed - 2800) / 1000);
        bx = centerX + 60 + (w - centerX + 200) * exitProgress;
        by = Math.sin(elapsed * 0.005) * 5 * (1 - exitProgress);
        bRot = -5 * exitProgress;
      }

      setBoatX(bx);
      setBoatY(by);
      setBoatRotation(bRot);

      // Logo on boat → detach at logo-drop phase
      if (elapsed < 2200) {
        setLogoOnBoat(true);
        setLogoOpacity(Math.min(1, elapsed / 600));
        setLogoScale(0.5 + easeOutCubic(Math.min(1, elapsed / 1500)) * 0.5);
        setLogoDropProgress(0);
      } else if (elapsed < 2800) {
        setLogoOnBoat(false);
        const dropProg = easeOutElastic((elapsed - 2200) / 600);
        setLogoDropProgress(dropProg);
        setLogoOpacity(1);
        setLogoScale(1 + Math.sin((elapsed - 2200) * 0.01) * 0.05 * (1 - dropProg));
      } else {
        setLogoOnBoat(false);
        setLogoDropProgress(1);
        setLogoOpacity(1);
        setLogoScale(1);
      }

      // Orbits fade in after logo settles
      if (elapsed >= 3200) {
        setOrbitsVisible(true);
      }

      // Draw water
      const time = elapsed * 0.001;
      drawWater(ctx!, w, h, time, wl, bx, currentPhase);

      if (elapsed >= TOTAL_DURATION) {
        // Trigger exit
        setIsExiting(true);
        setTimeout(() => {
          if (!skippedRef.current) {
            skippedRef.current = true;
            onComplete();
          }
        }, 500);
        return;
      }

      animFrameRef.current = requestAnimationFrame(animate);
    }

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [drawWater, onComplete]);

  // Skip handler
  const handleSkip = useCallback(() => {
    if (skippedRef.current) return;
    skippedRef.current = true;
    cancelAnimationFrame(animFrameRef.current);
    setIsExiting(true);
    setTimeout(onComplete, 300);
  }, [onComplete]);

  // Keyboard skip
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') handleSkip();
    };
    window.addEventListener('keydown', handler, { once: true });
    return () => window.removeEventListener('keydown', handler);
  }, [handleSkip]);

  // Calculate logo's final resting position (absolute center of viewport)
  const viewportCenterX = typeof window !== 'undefined' ? window.innerWidth / 2 : 500;
  const viewportCenterY = typeof window !== 'undefined' ? window.innerHeight / 2 - 60 : 300;

  // Logo position: either on boat or transitioning to center
  const logoX = logoOnBoat ? boatX : boatX + (viewportCenterX - boatX) * logoDropProgress;
  const logoY = logoOnBoat ? (viewportCenterY + boatY - 70) : (viewportCenterY + boatY - 70) + ((viewportCenterY) - (viewportCenterY + boatY - 70)) * logoDropProgress;

  return (
    <div
      ref={containerRef}
      className={`jru-boat-splash-overlay ${isExiting ? 'boat-splash-exiting' : ''}`}
      onClick={handleSkip}
      aria-label="Opening Jruva"
    >
      {/* Water Canvas Layer */}
      <canvas
        ref={canvasRef}
        className="jru-boat-water-canvas"
      />

      {/* Boat SVG */}
      <div
        className="jru-boat-vessel"
        style={{
          transform: `translate(${boatX - 80}px, ${viewportCenterY + boatY + 10}px) rotate(${boatRotation}deg)`,
          opacity: phase === 'water-entry' ? Math.min(1, waterLevel * 2) : phase === 'settled' ? 0 : 1,
          transition: phase === 'settled' ? 'opacity 0.3s ease' : 'none',
        }}
      >
        <svg width="160" height="80" viewBox="0 0 160 80" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Boat hull */}
          <defs>
            <linearGradient id="boatHull" x1="0" y1="30" x2="0" y2="80" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#5C3A28" />
              <stop offset="50%" stopColor="#3E2518" />
              <stop offset="100%" stopColor="#2A180E" />
            </linearGradient>
            <linearGradient id="boatDeck" x1="0" y1="20" x2="0" y2="40" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#8B6B4A" />
              <stop offset="100%" stopColor="#6B4C33" />
            </linearGradient>
            <linearGradient id="mastGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#DEC5A8" />
              <stop offset="100%" stopColor="#A48060" />
            </linearGradient>
            <linearGradient id="sailGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="rgba(255,248,240,0.95)" />
              <stop offset="50%" stopColor="rgba(240,225,205,0.9)" />
              <stop offset="100%" stopColor="rgba(210,190,165,0.85)" />
            </linearGradient>
          </defs>

          {/* Sail */}
          <path d="M 80 0 Q 115 15 110 38 L 82 38 Z" fill="url(#sailGrad)" stroke="rgba(180,160,130,0.5)" strokeWidth="0.5">
            <animateTransform
              attributeName="transform"
              type="rotate"
              values="-2 80 38;2 80 38;-2 80 38"
              dur="3s"
              repeatCount="indefinite"
            />
          </path>

          {/* Mast */}
          <rect x="78" y="2" width="4" height="38" rx="2" fill="url(#mastGrad)" />

          {/* Hull body */}
          <path d="M 10 40 Q 0 42 5 60 Q 20 75 80 78 Q 140 75 155 60 Q 160 42 150 40 Z" fill="url(#boatHull)" />

          {/* Deck planks */}
          <path d="M 18 40 Q 80 35 142 40 Q 135 42 80 44 Q 25 42 18 40 Z" fill="url(#boatDeck)" />

          {/* Deck line details */}
          <line x1="30" y1="43" x2="130" y2="43" stroke="rgba(90,60,40,0.4)" strokeWidth="0.5" />
          <line x1="35" y1="46" x2="125" y2="46" stroke="rgba(90,60,40,0.3)" strokeWidth="0.5" />

          {/* Bow ornament */}
          <circle cx="8" cy="48" r="3" fill="#B46C55" opacity="0.8" />

          {/* Waterline reflection */}
          <ellipse cx="80" cy="78" rx="60" ry="4" fill="rgba(56,163,209,0.2)" />
        </svg>
      </div>

      {/* Logo (rides on boat, then detaches to center) */}
      <div
        className="jru-boat-logo-carrier"
        style={{
          position: 'absolute',
          left: `${logoX - 52}px`,
          top: `${logoY - 52}px`,
          width: '104px',
          height: '104px',
          transform: `scale(${logoScale}) rotate(${logoOnBoat ? boatRotation * 0.5 : 0}deg)`,
          opacity: logoOpacity,
          transition: logoOnBoat ? 'none' : 'transform 0.3s ease-out',
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          filter: logoDropProgress >= 1 ? 'drop-shadow(0 10px 30px rgba(180, 108, 85, 0.45))' : 'none',
        }}
      >
        <JruLogo size={102} glow={true} variant="render" />
      </div>

      {/* Orbital rings (fade in after logo settles) */}
      {orbitsVisible && (
        <div
          className="jru-boat-orbits-container"
          style={{
            position: 'absolute',
            left: `${viewportCenterX - 100}px`,
            top: `${viewportCenterY - 100}px`,
            width: '200px',
            height: '200px',
          }}
        >
          <div className="jru-boat-orbit-outer">
            <div className="jru-boat-orbit-pip" />
          </div>
          <div className="jru-boat-orbit-inner" />
          {/* Ambient glow */}
          <div className="jru-boat-ambient-glow" />
        </div>
      )}

      {/* Brand text (appears after logo settles) */}
      <div
        className="jru-boat-brand-text"
        style={{
          position: 'absolute',
          left: '50%',
          top: `${viewportCenterY + 80}px`,
          transform: 'translateX(-50%)',
          opacity: logoDropProgress >= 1 ? 1 : 0,
          transition: 'opacity 0.6s ease',
        }}
      >
        <div className="jru-boat-brand-title">Jruva</div>
      </div>
    </div>
  );
};
