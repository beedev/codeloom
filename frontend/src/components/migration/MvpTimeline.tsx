/**
 * MvpTimeline — vertical sidebar listing MVPs with per-MVP phase progress.
 *
 * Shows each MVP as a card with:
 * - Priority badge + name
 * - Mini progress bar (4 phases: Analyze → Design → Transform → Test)
 * - Status indicator (discovered, in_progress, migrated)
 * - Click to select MVP for phase execution
 */

import { Check, Loader2, AlertCircle, ChevronRight } from 'lucide-react';
import type { FunctionalMvpSummary, MigrationPhaseInfo } from '../../types/index.ts';

interface MvpTimelineProps {
  mvps: FunctionalMvpSummary[];
  selectedMvpId: number | null;
  onSelectMvp: (mvpId: number) => void;
}

const MVP_PHASE_TYPES = ['analyze', 'design', 'transform', 'test'] as const;

export function MvpTimeline({ mvps, selectedMvpId, onSelectMvp }: MvpTimelineProps) {
  const sortedMvps = [...mvps].sort((a, b) => a.priority - b.priority);

  return (
    <div className="flex flex-col gap-1 p-3">
      <h3 className="mb-1 px-1 text-[10px] font-semibold uppercase tracking-wider text-text-dim">
        MVP Execution Queue
      </h3>

      {sortedMvps.map((mvp) => {
        const isSelected = mvp.mvp_id === selectedMvpId;
        const mvpPhases = mvp.phases.filter(p => p.mvp_id === mvp.mvp_id);
        const completedPhases = mvpPhases.filter(p => p.approved).length;
        const runningPhase = mvpPhases.find(p => p.status === 'running');
        const errorPhase = mvpPhases.find(p => p.status === 'error');
        const allDone = completedPhases === 4;

        return (
          <button
            key={mvp.mvp_id}
            onClick={() => onSelectMvp(mvp.mvp_id)}
            className={`flex items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-colors ${
              isSelected
                ? 'border-glow/50 bg-glow/10'
                : allDone
                  ? 'border-success/30 bg-success/5 hover:bg-success/10'
                  : 'border-void-surface/50 bg-void-light/30 hover:bg-void-light/50'
            }`}
          >
            {/* Priority badge */}
            <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] font-bold ${
              allDone
                ? 'bg-success/15 text-success'
                : isSelected
                  ? 'bg-glow/15 text-glow'
                  : 'bg-void-surface/50 text-text-dim'
            }`}>
              {allDone ? <Check className="h-3 w-3" /> : mvp.priority}
            </span>

            {/* Name + phase progress */}
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <span className={`truncate text-xs font-medium ${
                  isSelected ? 'text-glow' : allDone ? 'text-success' : 'text-text'
                }`}>
                  {mvp.name}
                </span>
                {runningPhase && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-glow" />}
                {errorPhase && <AlertCircle className="h-3 w-3 shrink-0 text-danger" />}
              </div>

              {/* Mini phase progress bar */}
              <div className="mt-1.5 flex gap-1">
                {MVP_PHASE_TYPES.map((type) => {
                  const phase = mvpPhases.find(p => p.phase_type === type);
                  return (
                    <PhaseChip
                      key={type}
                      label={type.charAt(0).toUpperCase()}
                      phase={phase}
                      isActive={isSelected}
                    />
                  );
                })}
              </div>
            </div>

            {/* Chevron */}
            <ChevronRight className={`h-3.5 w-3.5 shrink-0 ${
              isSelected ? 'text-glow' : 'text-text-dim/30'
            }`} />
          </button>
        );
      })}

      {mvps.length === 0 && (
        <p className="py-4 text-center text-xs text-text-dim">
          No MVPs. Complete Phase 1 first.
        </p>
      )}
    </div>
  );
}


function PhaseChip({
  label,
  phase,
  isActive,
}: {
  label: string;
  phase?: MigrationPhaseInfo;
  isActive: boolean;
}) {
  if (!phase) {
    return (
      <span className="flex h-4 w-6 items-center justify-center rounded-sm bg-void-surface/30 text-[8px] text-text-dim/30">
        {label}
      </span>
    );
  }

  if (phase.approved) {
    return (
      <span className="flex h-4 w-6 items-center justify-center rounded-sm bg-success/20 text-[8px] font-bold text-success">
        <Check className="h-2.5 w-2.5" />
      </span>
    );
  }

  if (phase.status === 'complete') {
    return (
      <span className="flex h-4 w-6 items-center justify-center rounded-sm bg-success/10 text-[8px] text-success/70">
        {label}
      </span>
    );
  }

  if (phase.status === 'running') {
    return (
      <span className="flex h-4 w-6 items-center justify-center rounded-sm bg-glow/15 text-[8px] text-glow">
        <Loader2 className="h-2.5 w-2.5 animate-spin" />
      </span>
    );
  }

  if (phase.status === 'error') {
    return (
      <span className="flex h-4 w-6 items-center justify-center rounded-sm bg-danger/15 text-[8px] text-danger">
        !
      </span>
    );
  }

  // pending
  return (
    <span className={`flex h-4 w-6 items-center justify-center rounded-sm text-[8px] ${
      isActive ? 'bg-void-surface/50 text-text-dim' : 'bg-void-surface/30 text-text-dim/40'
    }`}>
      {label}
    </span>
  );
}
