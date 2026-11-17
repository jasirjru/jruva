import {
  BookOpen,
  CalendarClock,
  FileText,
  FolderKanban,
  Image as ImageIcon,
  MessageSquare,
  Moon,
  Plug,
  SquarePen,
  Sun,
  type LucideIcon,
} from 'lucide-react';
import { WorkspaceView } from '@/types/workspace';

export interface SearchResults {
  chats?: Array<{ id: string; title: string; snippet?: string }>;
  projects?: Array<{ id: string; name: string; description?: string }>;
  files?: Array<{ id: string; filename: string; chunk_count?: number }>;
  images?: Array<{ id: string; prompt: string; model?: string }>;
}

export interface PaletteItem {
  id: string;
  group: string;
  label: string;
  sub?: string;
  Icon: LucideIcon;
  run: () => void;
}

export interface PaletteHandlers {
  onSelectView: (view: WorkspaceView) => void;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  onToggleTheme: () => void;
  theme: 'dark' | 'light';
}

const ACTIONS_GROUP = 'Actions';
const SNIPPET_LENGTH = 90;

function clip(text: string, max: number): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  return clean.length > max ? `${clean.slice(0, max - 1).trimEnd()}…` : clean;
}

/** Quick actions, filtered by the query so typing "image" still offers "Open Images". */
export function buildActions(query: string, h: PaletteHandlers): PaletteItem[] {
  const goTo = (view: WorkspaceView) => () => h.onSelectView(view);
  const actions: PaletteItem[] = [
    {
      id: 'act-new-chat',
      group: ACTIONS_GROUP,
      label: 'New chat',
      Icon: SquarePen,
      run: () => {
        h.onSelectView('chat');
        h.onNewChat();
      },
    },
    { id: 'act-images', group: ACTIONS_GROUP, label: 'Open Images', Icon: ImageIcon, run: goTo('images') },
    { id: 'act-library', group: ACTIONS_GROUP, label: 'Open Library', Icon: BookOpen, run: goTo('library') },
    { id: 'act-projects', group: ACTIONS_GROUP, label: 'Open Projects', Icon: FolderKanban, run: goTo('projects') },
    { id: 'act-scheduled', group: ACTIONS_GROUP, label: 'Open Scheduled', Icon: CalendarClock, run: goTo('scheduled') },
    { id: 'act-plugins', group: ACTIONS_GROUP, label: 'Open Plugins', Icon: Plug, run: goTo('plugins') },
    {
      id: 'act-theme',
      group: ACTIONS_GROUP,
      label: h.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
      Icon: h.theme === 'dark' ? Sun : Moon,
      run: h.onToggleTheme,
    },
  ];
  const needle = query.trim().toLowerCase();
  if (!needle) return actions;
  return actions.filter((a) => a.label.toLowerCase().includes(needle));
}

/** Search results from /api/v1/search/global, in display order. */
export function buildResults(results: SearchResults | null, h: PaletteHandlers): PaletteItem[] {
  if (!results) return [];
  const items: PaletteItem[] = [];
  for (const chat of results.chats ?? []) {
    items.push({
      id: `chat-${chat.id}`,
      group: 'Chats',
      label: chat.title || 'Untitled chat',
      sub: chat.snippet ? clip(chat.snippet, SNIPPET_LENGTH) : undefined,
      Icon: MessageSquare,
      run: () => {
        h.onSelectView('chat');
        h.onSelectSession(chat.id);
      },
    });
  }
  for (const project of results.projects ?? []) {
    items.push({
      id: `project-${project.id}`,
      group: 'Projects',
      label: project.name,
      sub: project.description ? clip(project.description, SNIPPET_LENGTH) : undefined,
      Icon: FolderKanban,
      run: () => h.onSelectView('projects'),
    });
  }
  for (const file of results.files ?? []) {
    items.push({
      id: `file-${file.id}`,
      group: 'Library',
      label: file.filename,
      sub: file.chunk_count ? `${file.chunk_count} sections indexed` : undefined,
      Icon: FileText,
      run: () => h.onSelectView('library'),
    });
  }
  for (const image of results.images ?? []) {
    items.push({
      id: `image-${image.id}`,
      group: 'Images',
      label: clip(image.prompt, 70),
      sub: image.model,
      Icon: ImageIcon,
      run: () => h.onSelectView('images'),
    });
  }
  return items;
}
