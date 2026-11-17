import { Brain, Code, FlaskConical, Network, type LucideIcon } from 'lucide-react';

const PERSONA_ICONS: Record<string, LucideIcon> = {
  general: Brain,
  architect: Network,
  coder: Code,
  scientist: FlaskConical,
};

type PersonaIconProps = {
  personaId: string;
  size?: number;
  className?: string;
};

/** Line icon for an answer-style persona; unknown ids fall back to the General icon. */
export function PersonaIcon({ personaId, size = 16, className }: PersonaIconProps) {
  const Icon = PERSONA_ICONS[personaId] ?? Brain;
  return <Icon size={size} strokeWidth={1.75} className={className} aria-hidden="true" />;
}
