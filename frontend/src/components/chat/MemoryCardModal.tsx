'use client';

import '@/styles/pages/popups.css';
import React, { useState } from 'react';
import { AlertDialog } from '@astryxdesign/core/AlertDialog';
import { Button } from '@astryxdesign/core/Button';
import { Dialog } from '@astryxdesign/core/Dialog';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { AlertCircle, Brain, Loader2, Lock, X } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { useMemory } from '@/context/MemoryContext';
import { UserMemory } from '@/types/memory';
import { MemoryAddForm } from './MemoryAddForm';
import { MemoryFactList } from './MemoryFactList';
import { asSentence } from './memoryCategories';

type PendingDelete = { kind: 'one'; memory: UserMemory } | { kind: 'all' } | null;

const QUOTE_LENGTH = 80;

function quote(text: string): string {
  return text.length > QUOTE_LENGTH ? `${text.slice(0, QUOTE_LENGTH - 1).trimEnd()}…` : text;
}

/** What Jruva remembers about the signed-in user. Open state lives in MemoryContext. */
export const MemoryCardModal: React.FC = () => {
  const { isMemoryModalOpen, closeMemoryModal } = useMemory();

  return (
    <Dialog
      isOpen={isMemoryModalOpen}
      onOpenChange={(open) => {
        if (!open) closeMemoryModal();
      }}
      width={600}
      maxHeight="min(760px, 90dvh)"
      padding={0}
      className="pop-dialog"
      aria-labelledby="memory-card-title"
      data-jruva-skin=""
    >
      {isMemoryModalOpen ? <MemoryBody onClose={closeMemoryModal} /> : null}
    </Dialog>
  );
};

function MemoryBody({ onClose }: { onClose: () => void }) {
  const { isAuthenticated, openAuthModal } = useAuth();
  const { memories, isLoading, addMemory, deleteMemory, clearAllMemories } = useMemory();
  const [pending, setPending] = useState<PendingDelete>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const count = memories.length;
  const subtitle =
    isAuthenticated && count > 0
      ? `${count} ${count === 1 ? 'fact' : 'facts'} Jruva uses to personalise answers.`
      : 'Facts Jruva keeps about you to personalise answers.';

  const confirmDelete = async () => {
    if (!pending) return;
    setIsDeleting(true);
    setDeleteError(null);
    const res = pending.kind === 'all' ? await clearAllMemories() : await deleteMemory(pending.memory.id);
    setIsDeleting(false);
    setPending(null);
    if (!res.success) {
      const what = pending.kind === 'all' ? 'clear your memory' : 'forget that fact';
      setDeleteError(`Couldn't ${what}. ${asSentence(res.error, 'The server did not respond')} Check your connection and try again.`);
    }
  };

  const dialogCopy =
    pending?.kind === 'all'
      ? {
          title: 'Clear all memory?',
          description: `All ${count} ${count === 1 ? 'fact' : 'facts'} will be deleted and Jruva will stop using them. This cannot be undone.`,
          action: 'Clear all',
        }
      : {
          title: 'Forget this fact?',
          description: pending?.kind === 'one' ? `“${quote(pending.memory.fact)}” will be deleted.` : '',
          action: 'Forget',
        };

  return (
    <div className="pop-shell">
      <header className="pop-head">
        <div className="pop-head-text">
          <h2 id="memory-card-title" className="pop-title" tabIndex={-1} data-autofocus="dialog-title">
            Memory
          </h2>
          <p className="pop-subtitle">{subtitle}</p>
        </div>
        <Tooltip content="Close" placement="below">
          <button type="button" className="pop-icon-btn" onClick={onClose} aria-label="Close">
            <X size={18} strokeWidth={1.75} aria-hidden="true" />
          </button>
        </Tooltip>
      </header>

      <div className="pop-body">
        {!isAuthenticated ? (
          <div className="pop-empty">
            <Brain className="pop-empty-icon" size={28} strokeWidth={1.5} aria-hidden="true" />
            <p className="pop-empty-title">Sign in to use memory</p>
            <p className="pop-empty-text">
              Sign in so Jruva can remember you.
            </p>
            <div className="pop-actions">
              <Button
                label="Sign in with Google"
                variant="primary"
                onClick={() => {
                  onClose();
                  openAuthModal('signin');
                }}
              />
            </div>
          </div>
        ) : (
          <>
            <MemoryAddForm showExamples={!isLoading && count === 0} onAdd={addMemory} />
            {deleteError && (
              <p className="pop-error" role="alert">
                <AlertCircle size={15} strokeWidth={1.75} aria-hidden="true" />
                <span>{deleteError}</span>
              </p>
            )}
            {isLoading && count === 0 ? (
              <p className="pop-status" role="status">
                <Loader2 size={16} strokeWidth={1.75} className="pop-spin" aria-hidden="true" />
                Loading your memory…
              </p>
            ) : (
              <MemoryFactList memories={memories} onRequestDelete={(memory) => setPending({ kind: 'one', memory })} />
            )}
          </>
        )}
      </div>

      <footer className="pop-foot">
        <span className="pop-foot-note">
          <Lock size={14} strokeWidth={1.75} aria-hidden="true" />
          Private to your account.
        </span>
        {isAuthenticated && count > 0 && (
          <button type="button" className="pop-text-btn" onClick={() => setPending({ kind: 'all' })}>
            Clear all
          </button>
        )}
      </footer>

      <AlertDialog
        isOpen={pending !== null}
        onOpenChange={(open) => {
          if (!open && !isDeleting) setPending(null);
        }}
        title={dialogCopy.title}
        description={dialogCopy.description}
        cancelLabel="Cancel"
        actionLabel={dialogCopy.action}
        actionVariant="destructive"
        isActionLoading={isDeleting}
        onAction={confirmDelete}
        data-jruva-skin=""
      />
    </div>
  );
}
