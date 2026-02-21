/**
 * EntryPointsSection — collapsible grouped list of discovered entry points.
 * Groups by entry_type, shows analyzed dot, type badge, name, file path, detection method.
 */

import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Globe,
  MessageSquare,
  Clock,
  Terminal,
  Zap,
  Power,
  Box,
  HelpCircle,
  Loader2,
} from 'lucide-react';
import type { EntryPoint } from '../../types/index.ts';

const ENTRY_TYPE_STYLES: Record<
  string,
  { bg: string; label: string; icon: typeof Globe }
> = {
  http_endpoint: { bg: 'bg-glow/10 text-glow', label: 'HE', icon: Globe },
  message_handler: {
    bg: 'bg-nebula/10 text-nebula-bright',
    label: 'MH',
    icon: MessageSquare,
  },
  scheduled_task: {
    bg: 'bg-warning/10 text-warning',
    label: 'ST',
    icon: Clock,
  },
  cli_command: {
    bg: 'bg-success/10 text-success',
    label: 'CC',
    icon: Terminal,
  },
  event_listener: {
    bg: 'bg-glow-bright/10 text-glow-bright',
    label: 'EL',
    icon: Zap,
  },
  startup_hook: { bg: 'bg-danger/10 text-danger', label: 'SH', icon: Power },
  public_api: { bg: 'bg-nebula/10 text-nebula', label: 'PA', icon: Box },
  unknown: {
    bg: 'bg-void-surface text-text-dim',
    label: '??',
    icon: HelpCircle,
  },
};

const TYPE_ORDER = [
  'http_endpoint',
  'message_handler',
  'scheduled_task',
  'cli_command',
  'event_listener',
  'startup_hook',
  'public_api',
  'unknown',
];

interface EntryPointsSectionProps {
  entryPoints: EntryPoint[];
  analyzedUnitIds: Set<string>;
  isLoading: boolean;
}

export function EntryPointsSection({
  entryPoints,
  analyzedUnitIds,
  isLoading,
}: EntryPointsSectionProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  // Group by entry_type
  const grouped = entryPoints.reduce<Record<string, EntryPoint[]>>(
    (acc, ep) => {
      const type = ep.entry_type || 'unknown';
      if (!acc[type]) acc[type] = [];
      acc[type].push(ep);
      return acc;
    },
    {},
  );

  const sortedTypes = Object.keys(grouped).sort(
    (a, b) =>
      (TYPE_ORDER.indexOf(a) === -1 ? 99 : TYPE_ORDER.indexOf(a)) -
      (TYPE_ORDER.indexOf(b) === -1 ? 99 : TYPE_ORDER.indexOf(b)),
  );

  const uniqueTypes = sortedTypes.length;

  return (
    <div className="rounded-lg border border-void-surface/50 bg-void-light/20">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex w-full items-center justify-between px-4 py-3"
      >
        <div className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronDown className="h-3.5 w-3.5 text-text-dim" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-text-dim" />
          )}
          <h3 className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
            Entry Points
          </h3>
          <span className="rounded-full bg-void-surface px-1.5 py-0.5 text-[9px] font-medium text-text-dim">
            {entryPoints.length}
          </span>
        </div>
        <span className="text-[10px] text-text-dim/60">
          {uniqueTypes} type{uniqueTypes !== 1 ? 's' : ''} detected
        </span>
      </button>

      {/* Body */}
      {isExpanded && (
        <div className="border-t border-void-surface/30 px-4 pb-3">
          {isLoading ? (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
            </div>
          ) : entryPoints.length === 0 ? (
            <p className="py-4 text-center text-[10px] text-text-dim/50">
              No entry points discovered yet. Run analysis to detect them.
            </p>
          ) : (
            <div className="mt-2 space-y-3">
              {sortedTypes.map((type) => {
                const style = ENTRY_TYPE_STYLES[type] ?? ENTRY_TYPE_STYLES.unknown;
                const Icon = style.icon;
                const items = grouped[type];

                return (
                  <div key={type}>
                    {/* Group subheader */}
                    <div className="mb-1 flex items-center gap-1.5 px-1">
                      <Icon className="h-3 w-3 text-text-dim/60" />
                      <span className="text-[9px] font-semibold uppercase tracking-wider text-text-dim/70">
                        {type.replace(/_/g, ' ')}
                      </span>
                      <span className="rounded-full bg-void-surface px-1.5 py-0.5 text-[8px] text-text-dim/50">
                        {items.length}
                      </span>
                    </div>

                    {/* Entry rows */}
                    {items.map((ep) => {
                      const isAnalyzed = analyzedUnitIds.has(ep.unit_id);
                      return (
                        <div
                          key={ep.unit_id}
                          className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-text-muted transition-colors hover:bg-void-surface hover:text-text"
                        >
                          {/* Analyzed dot */}
                          <span
                            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                              isAnalyzed ? 'bg-success' : 'bg-void-surface'
                            }`}
                          />
                          {/* Type badge */}
                          <span
                            className={`shrink-0 rounded px-1 py-0.5 text-[8px] font-bold uppercase ${style.bg}`}
                          >
                            {style.label}
                          </span>
                          {/* Name */}
                          <span className="truncate font-[family-name:var(--font-code)] text-[11px]">
                            {ep.name}
                          </span>
                          {/* File path */}
                          <span className="ml-auto shrink-0 text-[10px] text-text-dim/50">
                            {ep.file_path}
                          </span>
                          {/* Detection badge */}
                          <span className="shrink-0 rounded bg-void-surface/50 px-1 py-0.5 text-[8px] text-text-dim/40">
                            {ep.detected_by}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
