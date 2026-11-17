import { useState, type FormEvent } from 'react';
import { Dialog, DialogHeader } from '@astryxdesign/core/Dialog';
import { TextArea } from '@astryxdesign/core/TextArea';
import { TextInput } from '@astryxdesign/core/TextInput';
import { ProjectItem } from '@/types/workspace';
import { errorMessage, jsonBody, requestJson } from './workspaceApi';
import { WorkspaceNotice } from './WorkspacePageHead';

// Limits match ProjectCreateRequest in backend/app/api/v1/endpoints/projects.py
const NAME_MAX = 150;
const DESCRIPTION_MAX = 1000;
const INSTRUCTIONS_MAX = 5000;

type ProjectFormDialogProps = {
  /** The project to edit, or null to create a new one. */
  project: ProjectItem | null;
  onClose: () => void;
  onSaved: (message: string) => void;
};

/** Create or edit a project. Mount it with a key per project so the fields start fresh. */
export function ProjectFormDialog({ project, onClose, onSaved }: ProjectFormDialogProps) {
  const [name, setName] = useState(project?.name ?? '');
  const [description, setDescription] = useState(project?.description ?? '');
  const [instructions, setInstructions] = useState(project?.instructions ?? '');
  const [isSaving, setIsSaving] = useState(false);
  const [showNameError, setShowNameError] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const isEditing = project !== null;
  const trimmedName = name.trim();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!trimmedName) {
      setShowNameError(true);
      return;
    }
    setIsSaving(true);
    setSaveError(null);
    const payload = {
      name: trimmedName,
      description: description.trim() || null,
      instructions: instructions.trim() || null,
      // The page has no model picker; keep whatever the project already had.
      preferred_model: project?.preferred_model || 'auto',
    };
    try {
      if (isEditing) {
        await requestJson(`/api/v1/projects/${project.id}`, jsonBody('PUT', payload));
      } else {
        await requestJson('/api/v1/projects', jsonBody('POST', payload));
      }
      onSaved(isEditing ? `Saved changes to "${trimmedName}".` : `Created "${trimmedName}".`);
    } catch (err) {
      setSaveError(`Could not save the project. ${errorMessage(err)}`);
      setIsSaving(false);
    }
  };

  return (
    <Dialog
      isOpen
      onOpenChange={(open) => {
        if (!open && !isSaving) onClose();
      }}
      purpose="form"
      width={520}
      className="pp-dialog"
    >
      <DialogHeader
        title={isEditing ? 'Edit project' : 'New project'}
        subtitle="Applies to every chat in this project."
        onOpenChange={(open) => {
          if (!open && !isSaving) onClose();
        }}
      />
      <form className="pp-form" onSubmit={handleSubmit} noValidate>
        <TextInput
          label="Name"
          value={name}
          hasAutoFocus
          isRequired
          placeholder="For example: Thesis research"
          onChange={(value) => {
            setName(value.slice(0, NAME_MAX));
            if (value.trim()) setShowNameError(false);
          }}
          status={showNameError ? { type: 'error', message: 'Give the project a name.' } : undefined}
        />
        <TextInput
          label="Description"
          isOptional
          value={description}
          placeholder="What this project is for"
          onChange={(value) => setDescription(value.slice(0, DESCRIPTION_MAX))}
        />
        <TextArea
          label="Instructions"
          isOptional
          rows={5}
          maxLength={INSTRUCTIONS_MAX}
          value={instructions}
          description="Tone, format or facts Jruva should keep in mind."
          placeholder="For example: Answer in British English and cite page numbers."
          onChange={(value) => setInstructions(value)}
        />

        {saveError && <WorkspaceNotice tone="error" message={saveError} />}

        <div className="pp-form-actions">
          <button type="button" className="pp-btn pp-btn-quiet" onClick={onClose} disabled={isSaving}>
            Cancel
          </button>
          <button type="submit" className="pp-btn pp-btn-primary" disabled={isSaving} aria-busy={isSaving}>
            {isSaving ? 'Saving…' : isEditing ? 'Save changes' : 'Create project'}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
