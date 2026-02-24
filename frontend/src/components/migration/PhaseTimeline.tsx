/**
 * PhaseTimeline — horizontal stepper showing 6 migration phases.
 *
 * States: pending (gray), running (blue pulse), complete (green),
 * approved (green check), error (red). Click completed phases to review.
 * Uses Material Symbols for phase type icons.
 */

import { Check, Loader2, AlertCircle } from 'lucide-react';
import type { MigrationPhaseInfo } from '../../types/index.ts';

const PHASE_LABELS: Record<string, string> = {
  discovery: 'Discovery',
  architecture: 'Architecture',
  analyze: 'Analyze',
  design: 'Design',
  transform: 'Transform',
  test: 'Test',
};

const PHASE_ICONS: Record<string, string> = {
  discovery: 'lightbulb',
  architecture: 'account_tree',
  analyze: 'target',
  design: 'design_services',
  transform: 'swap_horiz',
  test: 'bug_report',
};

interface PhaseTimelineProps {
  phases: MigrationPhaseInfo[];
  activePhase: number;
  onSelectPhase: (phaseNumber: number) => void;
  locked?: boolean;
}

export function PhaseTimeline({ phases, activePhase, onSelectPhase, locked }: PhaseTimelineProps) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto px-4 py-3">
      {phases.map((phase, idx) => {
        const isActive = phase.phase_number === activePhase;
        // Locked phases are still clickable for read-only viewing
        const hasOutput = phase.status === 'complete' || phase.approved || phase.status === 'error';
        const isClickable = hasOutput;
        const materialIcon = PHASE_ICONS[phase.phase_type];

        return (
          <div key={phase.phase_id} className="flex items-center">
            {/* Connector line */}
            {idx > 0 && (
              <div
                className={`h-px w-6 shrink-0 ${
                  phase.status === 'complete' || phase.approved
                    ? 'bg-success/50'
                    : 'bg-void-surface'
                }`}
              />
            )}

            {/* Phase node */}
            <button
              onClick={() => isClickable ? onSelectPhase(phase.phase_number) : undefined}
              disabled={!isClickable && !isActive}
              className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs transition-colors ${
                locked
                  ? isActive
                    ? 'border-text-dim/40 bg-void-light/60 text-text-muted cursor-pointer'
                    : 'border-void-surface/50 bg-void-light/30 text-text-dim hover:bg-void-light/50 cursor-pointer'
                  : isActive
                    ? 'border-glow bg-glow/10 text-glow'
                    : phase.approved
                      ? 'border-success/50 bg-success/5 text-success hover:bg-success/10 cursor-pointer'
                      : phase.status === 'complete'
                        ? 'border-success/30 bg-void-light/50 text-success hover:bg-void-light cursor-pointer'
                        : phase.status === 'running'
                          ? 'border-glow/50 bg-glow/5 text-glow'
                          : phase.status === 'error'
                            ? 'border-danger/50 bg-danger/5 text-danger hover:bg-danger/10 cursor-pointer'
                            : 'border-void-surface bg-void-light/30 text-text-dim'
              }`}
            >
              <PhaseIcon
                status={phase.status}
                approved={phase.approved}
                materialIcon={materialIcon}
              />
              <span className="whitespace-nowrap">
                {phase.phase_number}. {PHASE_LABELS[phase.phase_type] ?? phase.phase_type}
              </span>
            </button>
          </div>
        );
      })}
    </div>
  );
}

function PhaseIcon({
  status,
  approved,
  materialIcon,
}: {
  status: string;
  approved: boolean;
  materialIcon?: string;
}) {
  if (approved) {
    return <Check className="h-3.5 w-3.5 text-success" />;
  }
  if (status === 'complete') {
    return <Check className="h-3.5 w-3.5 text-success/70" />;
  }
  if (status === 'running') {
    return <Loader2 className="h-3.5 w-3.5 animate-spin text-glow" />;
  }
  if (status === 'error') {
    return <AlertCircle className="h-3.5 w-3.5 text-danger" />;
  }
  // pending — show material icon if available
  if (materialIcon) {
    return <span className="material-symbols-outlined text-[16px]">{materialIcon}</span>;
  }
  return <div className="h-3 w-3 rounded-full border border-void-surface bg-void-light" />;
}
