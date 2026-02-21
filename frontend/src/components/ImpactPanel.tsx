/**
 * ImpactPanel — Blast radius visualization for impact analysis.
 *
 * Renders structured ASG dependency data: affected code units,
 * edge types, traversal depth, and file counts.
 */

import { useState } from 'react';
import { Crosshair, ChevronDown, ChevronRight, FileCode2 } from 'lucide-react';
import type { ImpactEntry } from '../hooks/useCodeChat.ts';

interface ImpactPanelProps {
  impact: ImpactEntry[];
}

const EDGE_COLORS: Record<string, string> = {
  calls: 'text-glow bg-glow/10 border-glow/20',
  imports: 'text-nebula-bright bg-nebula/10 border-nebula/20',
  inherits: 'text-warning bg-warning/10 border-warning/20',
  implements: 'text-success bg-success/10 border-success/20',
  overrides: 'text-danger bg-danger/10 border-danger/20',
};

function edgeStyle(edgeType: string): string {
  return EDGE_COLORS[edgeType] || 'text-text-dim bg-void-surface border-void-surface';
}

function shortPath(filePath: string): string {
  const parts = filePath.split('/');
  if (parts.length <= 2) return filePath;
  return `.../${parts.slice(-2).join('/')}`;
}

export function ImpactPanel({ impact }: ImpactPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [expandedUnits, setExpandedUnits] = useState<Set<number>>(
    () => new Set(),
  );

  const toggleUnit = (idx: number) => {
    setExpandedUnits((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const totalDirect = impact.reduce((sum, e) => sum + e.direct, 0);
  const totalIndirect = impact.reduce((sum, e) => sum + e.indirect, 0);

  return (
    <div className="border-b border-void-surface">
      {/* Header */}
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="flex w-full items-center gap-2 px-4 py-3 text-left hover:bg-void-light/50 transition-colors"
      >
        <Crosshair className="h-3.5 w-3.5 text-amber-400 shrink-0" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-400 font-[family-name:var(--font-display)]">
          Impact Analysis
        </h3>
        <span className="rounded-full bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-400">
          {impact.length} units · {totalDirect} direct / {totalIndirect} indirect
        </span>
        <div className="flex-1" />
        {collapsed ? (
          <ChevronRight className="h-3 w-3 text-text-dim" />
        ) : (
          <ChevronDown className="h-3 w-3 text-text-dim" />
        )}
      </button>

      {/* Body */}
      {!collapsed && (
        <div className="max-h-[40vh] overflow-y-auto overscroll-contain px-3 pb-3 space-y-2">
          {impact.map((entry, idx) => (
            <ImpactCard
              key={idx}
              entry={entry}
              isExpanded={expandedUnits.has(idx)}
              onToggle={() => toggleUnit(idx)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ImpactCard({
  entry,
  isExpanded,
  onToggle,
}: {
  entry: ImpactEntry;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const directDeps = entry.dependents.filter((d) => d.depth === 1);
  const indirectDeps = entry.dependents.filter((d) => d.depth > 1);

  return (
    <div className="rounded-lg border border-void-surface bg-void-light/50">
      {/* Unit header */}
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-text-dim shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-text-dim shrink-0" />
        )}

        <div className="flex-1 min-w-0">
          <span className="text-xs font-medium text-text truncate block">
            {entry.unit_name}
          </span>
          <div className="flex items-center gap-2 mt-0.5">
            <FileCode2 className="h-2.5 w-2.5 text-text-dim shrink-0" />
            <span className="text-[10px] text-text-dim truncate">
              {shortPath(entry.file_path)}
            </span>
          </div>
        </div>

        {/* Summary badges */}
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-semibold text-amber-400">
            {entry.direct} direct
          </span>
          {entry.indirect > 0 && (
            <span className="rounded bg-void-surface px-1.5 py-0.5 text-[9px] font-semibold text-text-dim">
              {entry.indirect} indirect
            </span>
          )}
          <span className="rounded bg-void-surface px-1.5 py-0.5 text-[9px] text-text-dim">
            {entry.files_affected} files
          </span>
        </div>
      </button>

      {/* Expanded dependents list */}
      {isExpanded && entry.dependents.length > 0 && (
        <div className="border-t border-void-surface/60 px-3 py-2 space-y-2">
          {/* Direct dependents */}
          {directDeps.length > 0 && (
            <div>
              <span className="text-[9px] font-semibold uppercase tracking-wider text-amber-400/80">
                Direct ({directDeps.length})
              </span>
              <div className="mt-1 space-y-1">
                {directDeps.map((dep, i) => (
                  <DependentRow key={i} dep={dep} />
                ))}
              </div>
            </div>
          )}

          {/* Indirect dependents */}
          {indirectDeps.length > 0 && (
            <div>
              <span className="text-[9px] font-semibold uppercase tracking-wider text-text-dim">
                Indirect ({indirectDeps.length})
              </span>
              <div className="mt-1 space-y-1">
                {indirectDeps.map((dep, i) => (
                  <DependentRow key={i} dep={dep} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {isExpanded && entry.dependents.length === 0 && (
        <div className="border-t border-void-surface/60 px-3 py-2">
          <span className="text-[10px] text-text-dim">No dependents found in ASG</span>
        </div>
      )}
    </div>
  );
}

function DependentRow({
  dep,
}: {
  dep: { name: string; edge_type: string; depth: number };
}) {
  return (
    <div className="flex items-center gap-2 pl-2">
      <span className="text-[9px] text-text-dim/50 w-3 text-right shrink-0">
        D{dep.depth}
      </span>
      <span
        className={`rounded border px-1 py-px text-[8px] font-semibold uppercase tracking-wide shrink-0 ${edgeStyle(dep.edge_type)}`}
      >
        {dep.edge_type}
      </span>
      <span className="text-[10px] text-text-muted truncate font-[family-name:var(--font-code)]">
        {dep.name}
      </span>
    </div>
  );
}
