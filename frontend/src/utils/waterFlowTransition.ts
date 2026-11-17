/**
 * Cinematic Boat & Flood Water Theme Transition Engine
 * 
 * When toggling dark ↔ light mode:
 *   1. A flood of water surges from the left
 *   2. A wooden sailboat rides the wave carrying the Jruva logo
 *   3. The flood washes across the viewport, switching the theme underneath
 *   4. The boat exits to the right, water recedes
 * 
 * GPU-accelerated canvas rendering with multi-harmonic wave physics,
 * foam crests, spray droplets, and a hand-drawn SVG boat.
 */

export type Theme = 'dark' | 'light';

interface Droplet {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  alpha: number;
  decay: number;
}

interface FoamBubble {
  x: number;
  y: number;
  radius: number;
  speed: number;
  phase: number;
}

// Pre-create boat image so it's ready when animation starts
let boatImageCache: HTMLImageElement | null = null;
let logoImageCache: HTMLImageElement | null = null;

function getBoatSVG(isDark: boolean): string {
  const hullColor1 = isDark ? '#3E2518' : '#5C3A28';
  const hullColor2 = isDark ? '#2A180E' : '#3E2518';
  const deckColor1 = isDark ? '#6B4C33' : '#8B6B4A';
  const deckColor2 = isDark ? '#4A3525' : '#6B4C33';
  const sailColor = isDark ? 'rgba(255,240,220,0.85)' : 'rgba(255,248,240,0.95)';
  const mastColor = isDark ? '#C4A882' : '#DEC5A8';

  return `<svg xmlns="http://www.w3.org/2000/svg" width="140" height="70" viewBox="0 0 140 70">
    <defs>
      <linearGradient id="bh" x1="0" y1="30" x2="0" y2="70" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="${hullColor1}"/>
        <stop offset="100%" stop-color="${hullColor2}"/>
      </linearGradient>
      <linearGradient id="bd" x1="0" y1="20" x2="0" y2="40" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="${deckColor1}"/>
        <stop offset="100%" stop-color="${deckColor2}"/>
      </linearGradient>
    </defs>
    <path d="M 70 0 Q 100 12 96 32 L 72 32 Z" fill="${sailColor}" stroke="rgba(180,160,130,0.4)" stroke-width="0.5"/>
    <rect x="68" y="2" width="4" height="32" rx="2" fill="${mastColor}"/>
    <path d="M 8 34 Q 0 36 4 52 Q 16 65 70 68 Q 124 65 136 52 Q 140 36 132 34 Z" fill="url(#bh)"/>
    <path d="M 14 34 Q 70 30 126 34 Q 120 36 70 38 Q 20 36 14 34 Z" fill="url(#bd)"/>
    <circle cx="6" cy="42" r="2.5" fill="#B46C55" opacity="0.8"/>
  </svg>`;
}

function createBoatImage(isDark: boolean): HTMLImageElement {
  const img = new Image();
  const svg = getBoatSVG(isDark);
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  return img;
}

export function triggerWaterFlowTransition(
  originX: number,
  originY: number,
  nextTheme: Theme,
  onThemeSwitch: () => void,
  onComplete?: () => void
): void {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    onThemeSwitch();
    onComplete?.();
    return;
  }

  // Check for reduced motion
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    onThemeSwitch();
    onComplete?.();
    return;
  }

  const width = window.innerWidth;
  const height = window.innerHeight;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  const canvas = document.createElement('canvas');
  canvas.style.position = 'fixed';
  canvas.style.inset = '0';
  canvas.style.width = '100vw';
  canvas.style.height = '100vh';
  canvas.style.pointerEvents = 'none';
  canvas.style.zIndex = '9999999';
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);

  const rawCtx = canvas.getContext('2d');
  if (!rawCtx) {
    onThemeSwitch();
    onComplete?.();
    return;
  }
  const ctx: CanvasRenderingContext2D = rawCtx;
  ctx.scale(dpr, dpr);
  document.body.appendChild(canvas);

  // ─── Configuration ──────────────────────────────────────────────────
  const isTargetLight = nextTheme === 'light';
  const DURATION = 1800; // ms — total animation length

  // Theme flood fill color
  const targetBg = isTargetLight ? '#F4EFEB' : '#12100E';

  // Water palette — light mode uses warm cream/sand/ivory tones matching the theme
  const waterDeep = isTargetLight
    ? 'rgba(164, 130, 114, 0.65)'    // Warm taupe
    : 'rgba(45, 28, 18, 0.8)';
  const waterMid = isTargetLight
    ? 'rgba(196, 174, 158, 0.6)'     // Sandy warm
    : 'rgba(120, 72, 50, 0.65)';
  const waterSurface = isTargetLight
    ? 'rgba(222, 210, 198, 0.75)'    // Cream ivory
    : 'rgba(180, 108, 85, 0.6)';
  const foamColor = isTargetLight
    ? 'rgba(253, 240, 233, 0.92)'    // Ivory foam
    : 'rgba(235, 160, 120, 0.85)';
  const foamGlow = isTargetLight
    ? 'rgba(180, 108, 85, 0.45)'     // Terracotta glow
    : 'rgba(217, 119, 87, 0.7)';
  const sprayColor = isTargetLight
    ? 'rgba(253, 240, 233, '          // Ivory spray
    : 'rgba(235, 160, 120, ';

  // Create boat image
  const boatImg = createBoatImage(!isTargetLight);
  const boatW = 140;
  const boatH = 70;

  // Load the Jruva logo for the boat to carry
  const logoImg = new Image();
  logoImg.src = '/jru-icon-256.webp';
  const logoSize = 48;

  // Spray droplets
  const droplets: Droplet[] = [];
  for (let i = 0; i < 30; i++) {
    droplets.push({
      x: -50,
      y: height * 0.45,
      vx: 3 + Math.random() * 5,
      vy: -(2 + Math.random() * 6),
      radius: 1 + Math.random() * 3,
      alpha: 0.6 + Math.random() * 0.4,
      decay: 0.008 + Math.random() * 0.015,
    });
  }

  // Foam bubbles on water surface
  const foamBubbles: FoamBubble[] = [];
  for (let i = 0; i < 24; i++) {
    foamBubbles.push({
      x: Math.random() * width,
      y: 0,
      radius: 1.5 + Math.random() * 3,
      speed: 0.5 + Math.random() * 1.5,
      phase: Math.random() * Math.PI * 2,
    });
  }

  const startTime = performance.now();
  let themeSwitched = false;
  let animId: number;

  // ─── Easing ─────────────────────────────────────────────────────────
  function easeInOutCubic(t: number): number {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  function easeOutQuart(t: number): number {
    return 1 - Math.pow(1 - t, 4);
  }

  // ─── Wave surface path ─────────────────────────────────────────────
  function getWaveY(x: number, time: number, waveIntensity: number): number {
    const w1 = Math.sin(x * 0.006 + time * 3.0) * 22;
    const w2 = Math.sin(x * 0.013 + time * 4.2 + 1.5) * 12;
    const w3 = Math.sin(x * 0.003 + time * 1.5) * 30;
    const w4 = Math.sin(x * 0.025 + time * 5.5 + 0.8) * 6;
    return (w1 + w2 + w3 + w4) * waveIntensity;
  }

  // ─── Render ─────────────────────────────────────────────────────────
  function render(now: number) {
    const elapsed = now - startTime;
    const progress = Math.min(1, elapsed / DURATION);
    const time = elapsed * 0.001;

    // Switch theme when flood reaches ~45%
    if (!themeSwitched && progress >= 0.42) {
      themeSwitched = true;
      onThemeSwitch();
    }

    ctx.clearRect(0, 0, width, height);

    // ─── Phase calculations ───────────────────────────────────────────
    // Flood front X position (left to right sweep)
    const floodFrontX = easeInOutCubic(progress) * (width + 300) - 200;

    // Water surface base Y (center-ish of screen, rises then recedes)
    let waterBaseY: number;
    if (progress < 0.15) {
      waterBaseY = height; // Start offscreen
    } else if (progress < 0.4) {
      const riseP = (progress - 0.15) / 0.25;
      waterBaseY = height - easeOutQuart(riseP) * (height * 0.52);
    } else if (progress < 0.7) {
      waterBaseY = height * 0.48; // Hold at peak
    } else {
      const recedeP = (progress - 0.7) / 0.3;
      waterBaseY = height * 0.48 + easeInOutCubic(recedeP) * (height * 0.52);
    }

    const waveIntensity = progress < 0.15 ? 0 : progress < 0.4 ? easeOutQuart((progress - 0.15) / 0.25) : progress < 0.7 ? 1 : 1 - easeInOutCubic((progress - 0.7) / 0.3);

    // ─── Layer 1: Theme flood fill (behind water) ─────────────────────
    if (floodFrontX > 0) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, Math.min(floodFrontX + 50, width), height);
      ctx.fillStyle = targetBg;
      ctx.fill();
      ctx.restore();
    }

    // ─── Layer 2: Deep water body ─────────────────────────────────────
    if (waterBaseY < height) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(0, height);
      for (let x = 0; x <= Math.min(floodFrontX + 100, width); x += 3) {
        const wy = waterBaseY + getWaveY(x, time, waveIntensity) + 25;
        ctx.lineTo(x, wy);
      }
      ctx.lineTo(Math.min(floodFrontX + 100, width), height);
      ctx.closePath();
      ctx.fillStyle = waterDeep;
      ctx.fill();
      ctx.restore();
    }

    // ─── Layer 3: Mid-water translucent body ──────────────────────────
    if (waterBaseY < height) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(0, height);
      for (let x = 0; x <= Math.min(floodFrontX + 60, width); x += 3) {
        const wy = waterBaseY + getWaveY(x, time * 1.1, waveIntensity) + 12;
        ctx.lineTo(x, wy);
      }
      ctx.lineTo(Math.min(floodFrontX + 60, width), height);
      ctx.closePath();
      ctx.fillStyle = waterMid;
      ctx.fill();
      ctx.restore();
    }

    // ─── Layer 4: Surface wave with foam crest ────────────────────────
    if (waterBaseY < height) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(0, height);
      for (let x = 0; x <= Math.min(floodFrontX + 30, width); x += 2) {
        const wy = waterBaseY + getWaveY(x, time * 1.2, waveIntensity);
        ctx.lineTo(x, wy);
      }
      ctx.lineTo(Math.min(floodFrontX + 30, width), height);
      ctx.closePath();
      ctx.fillStyle = waterSurface;
      ctx.fill();
      ctx.restore();

      // Foam crest line
      ctx.save();
      ctx.beginPath();
      for (let x = 0; x <= Math.min(floodFrontX + 30, width); x += 2) {
        const wy = waterBaseY + getWaveY(x, time * 1.2, waveIntensity);
        if (x === 0) ctx.moveTo(x, wy);
        else ctx.lineTo(x, wy);
      }
      ctx.strokeStyle = foamColor;
      ctx.lineWidth = 3 * waveIntensity;
      ctx.shadowColor = foamGlow;
      ctx.shadowBlur = 14;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.restore();

      // Foam bubbles along the surface
      ctx.save();
      for (const b of foamBubbles) {
        const bx = b.x + Math.sin(time * b.speed + b.phase) * 20;
        if (bx > floodFrontX + 50) continue;
        const by = waterBaseY + getWaveY(bx, time * 1.2, waveIntensity) + 3 + Math.sin(time * 2 + b.phase) * 4;
        ctx.beginPath();
        ctx.arc(bx, by, b.radius * waveIntensity, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${0.2 * waveIntensity})`;
        ctx.fill();
      }
      ctx.restore();
    }

    // ─── Layer 5: Leading splash spray at flood front ─────────────────
    if (progress > 0.05 && progress < 0.85) {
      ctx.save();
      // Big splash at the leading edge
      const splashBaseY = waterBaseY + getWaveY(floodFrontX, time * 1.2, waveIntensity);
      for (let i = 0; i < 8; i++) {
        const sx = floodFrontX + 10 + Math.random() * 40;
        const sy = splashBaseY - 10 - Math.random() * 50 * waveIntensity;
        const sr = 1 + Math.random() * 3;
        ctx.beginPath();
        ctx.arc(sx, sy, sr, 0, Math.PI * 2);
        ctx.fillStyle = sprayColor + (0.3 + Math.random() * 0.4).toFixed(2) + ')';
        ctx.fill();
      }
      ctx.restore();
    }

    // ─── Layer 6: Spray droplets ──────────────────────────────────────
    if (progress > 0.08 && progress < 0.8) {
      ctx.save();
      for (const d of droplets) {
        // Spawn from flood front
        if (d.alpha <= 0.05) {
          d.x = floodFrontX + Math.random() * 30;
          d.y = waterBaseY + getWaveY(d.x, time, waveIntensity) - 10;
          d.vx = 1 + Math.random() * 4;
          d.vy = -(3 + Math.random() * 7);
          d.alpha = 0.5 + Math.random() * 0.5;
        }
        d.x += d.vx;
        d.y += d.vy;
        d.vy += 0.15; // gravity
        d.alpha = Math.max(0, d.alpha - d.decay);

        if (d.alpha > 0) {
          ctx.beginPath();
          ctx.arc(d.x, d.y, d.radius, 0, Math.PI * 2);
          ctx.fillStyle = sprayColor + d.alpha.toFixed(2) + ')';
          ctx.fill();
        }
      }
      ctx.restore();
    }

    // ─── Layer 7: BOAT ────────────────────────────────────────────────
    // Boat rides the flood wave from left → center → right
    let boatX: number;
    if (progress < 0.85) {
      boatX = easeInOutCubic(Math.min(1, progress / 0.85)) * (width + boatW + 100) - boatW - 50;
    } else {
      boatX = width + 100; // Offscreen
    }

    const boatSurfaceY = waterBaseY + getWaveY(boatX + boatW / 2, time * 1.2, waveIntensity) - boatH * 0.55;
    const boatTilt = Math.sin(time * 2.5) * 3 + getWaveY(boatX + boatW / 2 + 10, time * 1.2, waveIntensity) * 0.08;

    if (boatX > -boatW && boatX < width + boatW && waterBaseY < height) {
      ctx.save();
      ctx.translate(boatX + boatW / 2, boatSurfaceY + boatH / 2);
      ctx.rotate((boatTilt * Math.PI) / 180);

      // Draw boat shadow on water
      ctx.fillStyle = 'rgba(0,0,0,0.12)';
      ctx.beginPath();
      ctx.ellipse(0, boatH * 0.38, boatW * 0.4, 6, 0, 0, Math.PI * 2);
      ctx.fill();

      // Draw the boat
      ctx.drawImage(boatImg, -boatW / 2, -boatH / 2, boatW, boatH);

      // Draw Jruva logo on the boat (sitting on the deck)
      if (logoImg.complete && logoImg.naturalWidth > 0) {
        const logoDrawX = -logoSize / 2 + 2;
        const logoDrawY = -boatH / 2 - logoSize * 0.55;
        // Glow behind logo
        ctx.save();
        ctx.shadowColor = 'rgba(180, 108, 85, 0.6)';
        ctx.shadowBlur = 20;
        ctx.drawImage(logoImg, logoDrawX, logoDrawY, logoSize, logoSize);
        ctx.restore();
        // Logo itself (crisp)
        ctx.drawImage(logoImg, logoDrawX, logoDrawY, logoSize, logoSize);
      }

      ctx.restore();

      // Boat wake - V-shaped ripples behind the boat
      if (progress > 0.1 && progress < 0.8) {
        ctx.save();
        ctx.globalAlpha = 0.3 * waveIntensity;
        for (let w = 0; w < 3; w++) {
          const wakeStartX = boatX - 10 - w * 30;
          const wakeY = boatSurfaceY + boatH * 0.4 + w * 8;
          ctx.beginPath();
          ctx.moveTo(wakeStartX, wakeY);
          ctx.quadraticCurveTo(wakeStartX - 40, wakeY - 10 - w * 5, wakeStartX - 80, wakeY - 5);
          ctx.strokeStyle = foamColor;
          ctx.lineWidth = 1.5 - w * 0.3;
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(wakeStartX, wakeY);
          ctx.quadraticCurveTo(wakeStartX - 40, wakeY + 10 + w * 5, wakeStartX - 80, wakeY + 5);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.restore();
      }
    }

    // ─── Continue or finish ───────────────────────────────────────────
    if (progress < 1) {
      animId = requestAnimationFrame(render);
    } else {
      // Smooth fade out
      canvas.style.transition = 'opacity 250ms ease-out';
      canvas.style.opacity = '0';
      setTimeout(() => {
        canvas.remove();
        onComplete?.();
      }, 280);
    }
  }

  animId = requestAnimationFrame(render);
}
