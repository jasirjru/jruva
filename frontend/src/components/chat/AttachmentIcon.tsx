import { File, FileCode, FileText, Image as ImageIcon } from 'lucide-react';

const CODE_EXTENSIONS = ['.py', '.js', '.ts', '.tsx', '.json', '.csv'];
const TEXT_EXTENSIONS = ['.pdf', '.txt', '.md', '.doc', '.docx'];

type AttachmentIconProps = {
  name: string;
  mimeType?: string;
  size?: number;
};

/** Line icon for an attached file, chosen from its type and extension. */
export function AttachmentIcon({ name, mimeType = '', size = 16 }: AttachmentIconProps) {
  const lower = name.toLowerCase();
  const Icon = mimeType.startsWith('image/')
    ? ImageIcon
    : CODE_EXTENSIONS.some((ext) => lower.endsWith(ext))
      ? FileCode
      : TEXT_EXTENSIONS.some((ext) => lower.endsWith(ext))
        ? FileText
        : File;
  return <Icon size={size} strokeWidth={1.75} aria-hidden="true" />;
}

const KB = 1024;
const MB = KB * 1024;

/** Human file size: "<1 KB", "48 KB", "2.4 MB". */
export function formatFileSize(bytes: number): string {
  if (bytes < KB) return '<1 KB';
  if (bytes < MB) return `${Math.round(bytes / KB)} KB`;
  return `${(bytes / MB).toFixed(1)} MB`;
}
