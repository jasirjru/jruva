'use client';

import React from 'react';

/**
 * Luminous Ambient Background Canvas — S+++ Edition.
 * Organic, slow-morphing atmospheric light aura.
 * 4 orb system with cinematic film grain.
 */
export const AmbientBackground: React.FC = () => {
  return (
    <div className="ambient-background-root" aria-hidden="true">
      {/* Primary Atmospheric Aurora Mesh — 4 orb system */}
      <div className="ambient-glow ambient-glow-1" />
      <div className="ambient-glow ambient-glow-2" />
      <div className="ambient-glow ambient-glow-3" />
      <div className="ambient-glow ambient-glow-4" />
      {/* Cinematic film grain overlay for photographic depth */}
      <div className="ambient-grain" />
    </div>
  );
};
