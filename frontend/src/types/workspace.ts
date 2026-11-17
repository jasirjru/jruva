/**
 * AI Workspace Operating System Types.
 */

export type WorkspaceView = 'chat' | 'images' | 'library' | 'scheduled' | 'plugins' | 'projects';

export interface ProjectItem {
  id: string;
  user_id?: string;
  name: string;
  description?: string | null;
  instructions?: string | null;
  preferred_model: string;
  created_at?: string;
  updated_at?: string;
}

export interface LibraryFileItem {
  id: string;
  user_id?: string;
  project_id?: string | null;
  filename: string;
  file_type: string;
  file_size: number;
  storage_path: string;
  status: 'indexed' | 'pending' | 'processing' | 'failed' | 'deleting';
  chunk_count: number;
  created_at?: string;
}

export interface GalleryImageItem {
  id: string;
  user_id?: string;
  prompt: string;
  enhanced_prompt?: string | null;
  model: string;
  tier: string;
  style: string;
  width: number;
  height: number;
  file_url: string;
  created_at?: string;
}

export interface ScheduledTaskItem {
  id: string;
  user_id?: string;
  name: string;
  prompt: string;
  cron_expression: string;
  timezone: string;
  status: 'active' | 'paused';
  last_run?: string | null;
  next_run?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface TaskExecutionItem {
  id: string;
  task_id: string;
  status: 'success' | 'failure' | 'running';
  output?: string | null;
  error_message?: string | null;
  executed_at: string;
}

export interface PluginItem {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  status: string;
  tools: string[];
  is_enabled: boolean;
}
