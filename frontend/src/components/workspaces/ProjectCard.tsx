import { DropdownMenu, DropdownMenuDivider, DropdownMenuItem } from '@astryxdesign/core/DropdownMenu';
import { Ellipsis, MessageSquarePlus, Pencil, Trash2 } from 'lucide-react';
import { ProjectItem } from '@/types/workspace';
import { formatDate } from './workspaceApi';

const MENU_ICON_SIZE = 15;

type ProjectCardProps = {
  project: ProjectItem;
  onStartChat?: (project: ProjectItem) => void;
  onEdit: (project: ProjectItem) => void;
  onRequestDelete: (project: ProjectItem) => void;
};

/** One project: what it is for, its instructions, and the ways to use or change it. */
export function ProjectCard({ project, onStartChat, onEdit, onRequestDelete }: ProjectCardProps) {
  const updated = formatDate(project.updated_at ?? project.created_at);
  const nameId = `project-name-${project.id}`;

  return (
    <li className="pp-project">
      <div className="pp-project-top">
        <h3 id={nameId} className="pp-project-name">{project.name}</h3>
        <DropdownMenu
          menuWidth={176}
          placement="below"
          alignment="end"
          renderTrigger={(triggerProps) => (
            <button
              type="button"
              {...triggerProps}
              className="pp-icon-btn"
              aria-label={`Options for ${project.name}`}
            >
              <Ellipsis size={16} strokeWidth={1.75} aria-hidden="true" />
            </button>
          )}
        >
          <DropdownMenuItem icon={<Pencil size={MENU_ICON_SIZE} />} label="Edit" onClick={() => onEdit(project)} />
          <DropdownMenuDivider />
          <DropdownMenuItem
            icon={<Trash2 size={MENU_ICON_SIZE} />}
            label="Delete"
            variant="destructive"
            onClick={() => onRequestDelete(project)}
          />
        </DropdownMenu>
      </div>

      {project.description && <p className="pp-project-desc">{project.description}</p>}

      {project.instructions ? (
        <div className="pp-project-instructions">
          <span className="pp-project-label">Instructions</span>
          <p className="pp-project-instructions-text">{project.instructions}</p>
        </div>
      ) : (
        <p className="pp-project-none">No instructions yet.</p>
      )}

      <div className="pp-project-foot">
        {updated && <span className="pp-project-date">Updated {updated}</span>}
        {onStartChat && (
          <button
            type="button"
            className="pp-btn pp-btn-secondary pp-btn-sm"
            onClick={() => onStartChat(project)}
            aria-describedby={nameId}
          >
            <MessageSquarePlus size={15} strokeWidth={1.75} aria-hidden="true" />
            Start a chat
          </button>
        )}
      </div>
    </li>
  );
}
