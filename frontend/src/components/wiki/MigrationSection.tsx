/**
 * MigrationSection -- Project Wiki migration panel.
 *
 * Renders the active migration plan: status, stack info,
 * MVP progress donut, and expandable plan-level phases.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Loader2,
  Inbox,
  AlertCircle,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type {
  MigrationPlan,
  MigrationPhaseOutput,
  MigrationPhaseInfo,
  FunctionalMvpSummary,
} from '../../types/index.ts';

/* ── Props ───────────────────────────────────────────────────────── */

interface Props {
  migrationPlan: MigrationPlan | null;
  loading?: boolean;
  error?: string | null;
  onLoad: () => Promise<void>;
  onLoadPhase: (phaseNumber: number, mvpId?: number) => Promise<MigrationPhaseOutput | null>;
}

/* ── Status badge helpers ────────────────────────────────────────── */

const PLAN_STATUS_COLORS: Record<string, string> = {
  draft: 'bg-void-surface/50 text-text-dim border-void-surface',
  in_progress: 'bg-glow/15 text-glow border-glow/30',
  complete: 'bg-success/15 text-success border-success/30',
  abandoned: 'bg-danger/15 text-danger border-danger/30',
};

const PHASE_STATUS_COLORS: Record<string, string> = {
  pending: 'bg-void-surface/50 text-text-dim border-void-surface',
  running: 'bg-glow/15 text-glow border-glow/30',
  complete: 'bg-success/15 text-success border-success/30',
  error: 'bg-danger/15 text-danger border-danger/30',
};

/* ── MVP donut colors ────────────────────────────────────────────── */

const MVP_STATUS_COLORS: Record<FunctionalMvpSummary['status'], string> = {
  discovered: 'var(--color-text-dim)',
  refined: 'var(--color-nebula)',
  in_progress: 'var(--color-glow)',
  migrated: 'var(--color-success)',
};

const MVP_STATUS_LEGEND_CLASSES: Record<FunctionalMvpSummary['status'], string> = {
  discovered: 'bg-text-dim',
  refined: 'bg-nebula',
  in_progress: 'bg-glow',
  migrated: 'bg-success',
};

/* ── Language pill ───────────────────────────────────────────────── */

function LangPill({ lang }: { lang: string }) {
  return (
    <span className="inline-flex items-center rounded-full border border-void-surface bg-void-light/50 px-2.5 py-0.5 text-[11px] font-medium text-text-muted">
      {lang}
    </span>
  );
}

/* ── SVG donut chart ─────────────────────────────────────────────── */

function MvpDonut({ mvps }: { mvps: FunctionalMvpSummary[] }) {
  const statusCounts: Record<FunctionalMvpSummary['status'], number> = {
    discovered: 0,
    refined: 0,
    in_progress: 0,
    migrated: 0,
  };
  for (const mvp of mvps) {
    statusCounts[mvp.status]++;
  }

  const total = mvps.length;
  const radius = 50;
  const strokeWidth = 12;
  const circumference = 2 * Math.PI * radius;

  // Build arcs
  let accumulated = 0;
  const arcs: Array<{ color: string; offset: number; length: number }> = [];
  for (const status of ['discovered', 'refined', 'in_progress', 'migrated'] as const) {
    const count = statusCounts[status];
    if (count === 0) continue;
    const fraction = count / total;
    arcs.push({
      color: MVP_STATUS_COLORS[status],
      offset: accumulated * circumference,
      length: fraction * circumference,
    });
    accumulated += fraction;
  }

  return (
    <div className="flex items-center gap-6">
      <div className="relative">
        <svg width="130" height="130" viewBox="0 0 130 130">
          {/* Background ring */}
          <circle
            cx="65"
            cy="65"
            r={radius}
            fill="none"
            stroke="var(--color-void-surface)"
            strokeWidth={strokeWidth}
          />
          {/* Data arcs */}
          {arcs.map((arc, i) => (
            <circle
              key={i}
              cx="65"
              cy="65"
              r={radius}
              fill="none"
              stroke={arc.color}
              strokeWidth={strokeWidth}
              strokeDasharray={`${arc.length} ${circumference - arc.length}`}
              strokeDashoffset={-arc.offset}
              strokeLinecap="round"
              transform="rotate(-90 65 65)"
              className="transition-all duration-500"
            />
          ))}
          {/* Center text */}
          <text
            x="65"
            y="60"
            textAnchor="middle"
            className="fill-text text-2xl font-semibold"
            style={{ fontSize: '28px' }}
          >
            {total}
          </text>
          <text
            x="65"
            y="80"
            textAnchor="middle"
            className="fill-text-muted"
            style={{ fontSize: '11px' }}
          >
            MVPs
          </text>
        </svg>
      </div>

      {/* Legend */}
      <div className="space-y-2">
        {(['discovered', 'refined', 'in_progress', 'migrated'] as const).map((status) => (
          <div key={status} className="flex items-center gap-2 text-xs">
            <span className={`h-2.5 w-2.5 rounded-full ${MVP_STATUS_LEGEND_CLASSES[status]}`} />
            <span className="text-text-muted capitalize">{status.replace('_', ' ')}</span>
            <span className="text-text font-medium">{statusCounts[status]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Expandable phase row ────────────────────────────────────────── */

function PhaseRow({
  phase,
  onLoadPhase,
}: {
  phase: MigrationPhaseInfo;
  onLoadPhase: (phaseNumber: number) => Promise<MigrationPhaseOutput | null>;
}) {
  const [expanded, setExpanded] = useState(false);
  const [output, setOutput] = useState<MigrationPhaseOutput | null>(null);
  const [loading, setLoading] = useState(false);

  const handleToggle = useCallback(async () => {
    if (expanded) {
      setExpanded(false);
      return;
    }
    setExpanded(true);
    if (!output) {
      setLoading(true);
      const result = await onLoadPhase(phase.phase_number);
      setOutput(result);
      setLoading(false);
    }
  }, [expanded, output, onLoadPhase, phase.phase_number]);

  const statusClass = PHASE_STATUS_COLORS[phase.status] ?? PHASE_STATUS_COLORS.pending;

  return (
    <div className="rounded-lg border border-void-surface bg-void/40">
      <button
        onClick={handleToggle}
        className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-void-surface/30"
      >
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-text-dim" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-text-dim" />
        )}

        <span className="text-xs font-medium text-text-muted w-6">
          #{phase.phase_number}
        </span>

        <span className="text-xs font-medium text-text capitalize flex-1">
          {phase.phase_type}
        </span>

        <span
          className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium ${statusClass}`}
        >
          {phase.status}
        </span>

        {phase.approved && (
          <span className="inline-flex items-center gap-1 text-success" title="Approved">
            <Check className="h-3.5 w-3.5" />
          </span>
        )}
      </button>

      {expanded && (
        <div className="border-t border-void-surface px-4 py-3">
          {loading ? (
            <div className="flex items-center gap-2 py-4 text-text-muted">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-xs">Loading phase output...</span>
            </div>
          ) : output?.output ? (
            <div className="prose prose-invert prose-sm max-w-none text-text-muted [&_h1]:text-text [&_h2]:text-text [&_h3]:text-text [&_strong]:text-text [&_code]:text-glow [&_a]:text-glow">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {output.output}
              </ReactMarkdown>
            </div>
          ) : output?.output_preview ? (
            <div className="prose prose-invert prose-sm max-w-none text-text-muted [&_h1]:text-text [&_h2]:text-text [&_h3]:text-text [&_strong]:text-text [&_code]:text-glow [&_a]:text-glow">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {output.output_preview}
              </ReactMarkdown>
            </div>
          ) : (
            <p className="text-xs text-text-dim py-2">No output available for this phase.</p>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Main component ──────────────────────────────────────────────── */

export function MigrationSection({
  migrationPlan,
  loading,
  error,
  onLoad,
  onLoadPhase,
}: Props) {
  useEffect(() => {
    if (migrationPlan === null) {
      void onLoad();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* Loading */
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-muted">
        <Loader2 className="h-6 w-6 animate-spin" />
        <span className="text-sm">Loading migration data...</span>
      </div>
    );
  }

  /* Error */
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-danger">
        <AlertCircle className="h-6 w-6" />
        <span className="text-sm">{error}</span>
      </div>
    );
  }

  /* Empty state */
  if (!migrationPlan) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-dim">
        <Inbox className="h-8 w-8" />
        <span className="text-sm">No migration plan yet. Create one from the project page.</span>
      </div>
    );
  }

  const plan = migrationPlan;
  const planPhases = plan.plan_phases.filter((p) => p.mvp_id === null);
  const pipelineLabel = plan.pipeline_version === 2 ? 'V2 (4-phase)' : 'V1 (6-phase)';
  const statusClass = PLAN_STATUS_COLORS[plan.status] ?? PLAN_STATUS_COLORS.draft;

  return (
    <div className="p-6 space-y-6">
      {/* Header card */}
      <div className="rounded-xl border border-void-surface bg-void-light/30 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-text">Migration Plan</h2>
          <div className="flex items-center gap-3">
            <span className="text-xs text-text-dim">Pipeline {pipelineLabel}</span>
            <span
              className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${statusClass}`}
            >
              {plan.status.replace('_', ' ')}
            </span>
          </div>
        </div>

        {plan.target_brief && (
          <p className="text-sm text-text-muted leading-relaxed">{plan.target_brief}</p>
        )}
      </div>

      {/* Stack info */}
      <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
        <h3 className="text-sm font-semibold text-text mb-3">Stack Transition</h3>
        <div className="flex items-center gap-4 flex-wrap">
          {/* Source */}
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
              Source
            </p>
            <div className="flex flex-wrap gap-1.5">
              {plan.source_stack?.primary_language && (
                <LangPill lang={plan.source_stack.primary_language} />
              )}
              {plan.source_stack?.languages
                .filter((l) => l !== plan.source_stack?.primary_language)
                .map((lang) => (
                  <LangPill key={lang} lang={lang} />
                ))}
            </div>
          </div>

          <ArrowRight className="h-5 w-5 text-glow shrink-0" />

          {/* Target */}
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
              Target
            </p>
            <div className="flex flex-wrap gap-1.5">
              {plan.target_stack.languages.map((lang) => (
                <LangPill key={lang} lang={lang} />
              ))}
              {plan.target_stack.frameworks.map((fw) => (
                <span
                  key={fw}
                  className="inline-flex items-center rounded-full border border-nebula/30 bg-nebula/10 px-2.5 py-0.5 text-[11px] font-medium text-nebula-bright"
                >
                  {fw}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* MVP Progress */}
      {plan.mvps.length > 0 && (
        <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
          <h3 className="text-sm font-semibold text-text mb-3">MVP Progress</h3>
          <MvpDonut mvps={plan.mvps} />
        </div>
      )}

      {/* Plan-level Phases */}
      {planPhases.length > 0 && (
        <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
          <h3 className="text-sm font-semibold text-text mb-3">Plan Phases</h3>
          <div className="space-y-2">
            {planPhases.map((phase) => (
              <PhaseRow
                key={phase.phase_id}
                phase={phase}
                onLoadPhase={(num) => onLoadPhase(num)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
