'use client';

import '@/styles/pages/popups.css';
import React, { useEffect, useRef, useState } from 'react';
import { Dialog } from '@astryxdesign/core/Dialog';
import { Button } from '@astryxdesign/core/Button';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { AlertCircle, Loader2, X } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { JruLogo } from './JruLogo';

interface GoogleCredentialResponse {
  credential?: string;
}

interface GoogleIdentity {
  accounts?: {
    id?: {
      initialize: (config: {
        client_id: string;
        callback: (response: GoogleCredentialResponse) => void;
        auto_select?: boolean;
      }) => void;
      renderButton: (parent: HTMLElement, options: Record<string, string | number>) => void;
    };
  };
}

declare global {
  interface Window {
    google?: GoogleIdentity;
  }
}

const GOOGLE_CLIENT_ID =
  process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ||
  '285159042014-bd4vnhp6hblrhadrvbbdeop2feooce9r.apps.googleusercontent.com';

const SCRIPT_POLL_MS = 120;
// About 10 s: the Google script is fetched on demand when this dialog first opens.
const SCRIPT_POLL_LIMIT = 80;
const GSI_SCRIPT_ID = 'google-gsi-client';
const GSI_SCRIPT_SRC = 'https://accounts.google.com/gsi/client';

// Load Google's sign-in script only when someone opens this dialog; pages never pay for it otherwise.
function ensureGoogleScript(): void {
  if (window.google?.accounts?.id || document.getElementById(GSI_SCRIPT_ID)) return;
  const script = document.createElement('script');
  script.id = GSI_SCRIPT_ID;
  script.src = GSI_SCRIPT_SRC;
  script.async = true;
  // A failed load leaves a stale tag; drop it so "Try again" can fetch it afresh.
  script.onerror = () => script.remove();
  document.head.appendChild(script);
}
const GOOGLE_BUTTON_WIDTH = 280;

type ButtonStatus = 'loading' | 'ready' | 'failed';

/** Google sign-in. Open state lives in AuthContext (openAuthModal / closeAuthModal). */
export const AuthModal: React.FC = () => {
  const { isAuthModalOpen, closeAuthModal } = useAuth();

  return (
    <Dialog
      isOpen={isAuthModalOpen}
      onOpenChange={(open) => {
        if (!open) closeAuthModal();
      }}
      width={400}
      maxHeight="92dvh"
      padding={0}
      className="pop-dialog"
      aria-labelledby="auth-modal-title"
      data-jruva-skin=""
    >
      {/* Mounted only while open, so each open starts without an old error. */}
      {isAuthModalOpen ? <AuthBody onClose={closeAuthModal} /> : null}
    </Dialog>
  );
};

function AuthBody({ onClose }: { onClose: () => void }) {
  const { loginWithGoogle } = useAuth();
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [buttonStatus, setButtonStatus] = useState<ButtonStatus>('loading');
  const [attempt, setAttempt] = useState(0);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Fetch the Google Identity script if needed, wait for it, then draw its button.
  useEffect(() => {
    ensureGoogleScript();
    let polls = 0;
    const interval = setInterval(() => {
      polls += 1;
      const googleId = window.google?.accounts?.id;
      const container = containerRef.current;
      if (googleId && container) {
        clearInterval(interval);
        try {
          googleId.initialize({
            client_id: GOOGLE_CLIENT_ID,
            auto_select: false,
            callback: async (response) => {
              if (!response?.credential) return;
              setIsSubmitting(true);
              setErrorMessage(null);
              const res = await loginWithGoogle(response.credential);
              if (!res.success) setErrorMessage(res.error || 'Google did not confirm the sign-in.');
              setIsSubmitting(false);
            },
          });
          container.replaceChildren();
          const isDark = document.documentElement.dataset.theme === 'dark';
          googleId.renderButton(container, {
            theme: isDark ? 'filled_black' : 'outline',
            size: 'large',
            type: 'standard',
            shape: 'pill',
            text: 'signin_with',
            logo_alignment: 'left',
            width: GOOGLE_BUTTON_WIDTH,
          });
          setButtonStatus('ready');
        } catch (err) {
          console.warn('Google sign-in button failed to render:', err);
          setButtonStatus('failed');
        }
      } else if (polls > SCRIPT_POLL_LIMIT) {
        clearInterval(interval);
        setButtonStatus('failed');
      }
    }, SCRIPT_POLL_MS);
    return () => clearInterval(interval);
  }, [loginWithGoogle, attempt]);

  const retry = () => {
    setButtonStatus('loading');
    setAttempt((n) => n + 1);
  };

  return (
    <div className="pop-shell">
      <div className="pop-auth">
        <JruLogo size={44} animMode="static" />
        <h2 id="auth-modal-title" className="pop-title" tabIndex={-1} data-autofocus="dialog-title">
          Sign in to Jruva
        </h2>
        <p className="pop-subtitle">Save your chats and memory across devices.</p>

        {isSubmitting && (
          <p className="pop-status" role="status">
            <Loader2 size={16} strokeWidth={1.75} className="pop-spin" aria-hidden="true" />
            Signing you in…
          </p>
        )}
        {!isSubmitting && buttonStatus === 'loading' && (
          <p className="pop-status" role="status">
            <Loader2 size={16} strokeWidth={1.75} className="pop-spin" aria-hidden="true" />
            Loading Google sign-in…
          </p>
        )}

        {/* Always mounted: Google draws its button into this node once. */}
        <div
          ref={containerRef}
          className="pop-auth-google"
          hidden={isSubmitting || buttonStatus !== 'ready'}
        />

        {buttonStatus === 'failed' && (
          <>
            <p className="pop-error" role="alert">
              <AlertCircle size={15} strokeWidth={1.75} aria-hidden="true" />
              <span>
                Google sign-in didn&apos;t load. Check your connection, or allow accounts.google.com if a
                content blocker is on, then try again.
              </span>
            </p>
            <div className="pop-auth-retry">
              <Button label="Try again" variant="secondary" onClick={retry} />
            </div>
          </>
        )}

        {errorMessage && (
          <p className="pop-error" role="alert">
            <AlertCircle size={15} strokeWidth={1.75} aria-hidden="true" />
            <span>
              <strong>Sign-in failed.</strong> {errorMessage} Use the Google button to try again.
            </span>
          </p>
        )}
      </div>

      {/* After the content in DOM order: showModal() focuses the first focusable node,
          and a tooltip opened by that focus outlives the move to the title. */}
      <Tooltip content="Close" placement="below">
        <button type="button" className="pop-icon-btn pop-auth-close" onClick={onClose} aria-label="Close">
          <X size={18} strokeWidth={1.75} aria-hidden="true" />
        </button>
      </Tooltip>

      <div className="pop-foot pop-auth-foot">
        <Button label="Continue without signing in" variant="ghost" onClick={onClose} />
      </div>
    </div>
  );
}
