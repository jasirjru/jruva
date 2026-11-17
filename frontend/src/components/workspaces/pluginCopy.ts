import { Code2, Globe, ImageIcon, Puzzle, ScanText, type LucideIcon } from 'lucide-react';
import type { PluginItem } from '@/types/workspace';

/**
 * Plain-words copy for each plugin the backend lists (backend/app/api/v1/endpoints/plugins.py).
 *
 * Only web search has a working on/off setting: chat_service.account_web_search_enabled reads it.
 * The other toggles are stored but nothing reads them, so they are shown without a switch.
 */
export type PluginCopy = {
  name: string;
  description: string;
  Icon: LucideIcon;
  hasSetting: boolean;
  settingNote?: string;
};

const COPY: Record<string, PluginCopy> = {
  web_search: {
    name: 'Web search',
    description: 'Finds current information and cites sources.',
    Icon: Globe,
    hasSetting: true,
  },
  python_sandbox: {
    name: 'Code check',
    description: 'Checks how code in answers scales and flags common problems.',
    Icon: Code2,
    hasSetting: false,
  },
  document_ocr: {
    name: 'Text in images',
    description: 'Reads text in images you attach.',
    Icon: ScanText,
    hasSetting: false,
  },
  image_studio: {
    name: 'Image generation',
    description: 'Creates and upscales images.',
    Icon: ImageIcon,
    hasSetting: false,
  },
};

/** Statuses the backend uses for plugins that work right now. */
const WORKING_STATUSES = new Set(['ready', 'static_analysis_active']);

export type PluginGroup = 'working' | 'off';

/** Planned plugins do not work yet, so they are not listed at all. */
export function pluginGroup(plugin: PluginItem): PluginGroup | null {
  if (WORKING_STATUSES.has(plugin.status)) return 'working';
  if (plugin.status === 'unavailable') return 'off';
  return null;
}

export function pluginCopy(plugin: PluginItem): PluginCopy {
  return (
    COPY[plugin.id] ?? {
      name: plugin.name,
      description: plugin.description,
      Icon: Puzzle,
      hasSetting: false,
    }
  );
}
