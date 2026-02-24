/**
 * OverviewSection -- Project Wiki overview panel.
 *
 * Renders project status badges, metric cards, language distribution,
 * and quick-link navigation when analytics data is available.
 */

import { Link } from 'react-router-dom';
import {
  Loader2,
  AlertCircle,
  FileText,
  Blocks,
  GitFork,
  Hash,
  ExternalLink,
  MessageSquare,
  BarChart3,
  Inbox,
} from 'lucide-react';
import type { ProjectAnalytics } from '../../types/index.ts';

interface Props {
  analytics: ProjectAnalytics | null;
  loading?: boolean;
  error?: string | null;
  projectId: string;
}

/* ── Status badge helpers ─────────────────────────────────────────── */

type StatusVariant = 'success' | 'glow' | 'danger' | 'dim';

function resolveVariant(status: string): StatusVariant {
  const lower = status.toLowerCase();
  if (['complete', 'completed'].includes(lower)) return 'success';
  if (['parsing', 'running', 'pending'].includes(lower)) return 'glow';
  if (['error', 'failed'].includes(lower)) return 'danger';
  return 'dim';
}

const VARIANT_CLASSES: Record<StatusVariant, string> = {
  success: 'bg-success/15 text-success border-success/30',
  glow: 'bg-glow/15 text-glow border-glow/30',
  danger: 'bg-danger/15 text-danger border-danger/30',
  dim: 'bg-void-surface/50 text-text-dim border-void-surface',
};

function StatusBadge({ label, value }: { label: string; value: string }) {
  const variant = resolveVariant(value);
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-text-dim">{label}:</span>
      <span
        className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${VARIANT_CLASSES[variant]}`}
      >
        {value}
      </span>
    </div>
  );
}

/* ── Metric card ─────────────────────────────────────────────────── */

function MetricCard({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof FileText;
  label: string;
  value: number;
}) {
  return (
    <div className="rounded-xl border border-void-surface bg-void-light/30 p-5 flex flex-col gap-1">
      <div className="flex items-center gap-2 text-text-muted">
        <Icon className="h-4 w-4" />
        <span className="text-xs font-medium">{label}</span>
      </div>
      <p className="text-2xl font-semibold text-text">{value.toLocaleString()}</p>
    </div>
  );
}

/* ── Language bar ─────────────────────────────────────────────────── */

function LanguageBar({
  language,
  count,
  maxCount,
}: {
  language: string;
  count: number;
  maxCount: number;
}) {
  const pct = maxCount > 0 ? (count / maxCount) * 100 : 0;
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 truncate text-xs font-medium text-text">
        {language}
      </span>
      <div className="flex-1 h-2 rounded-full bg-void-surface overflow-hidden">
        <div
          className="h-full rounded-full bg-glow transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-10 text-right text-xs tabular-nums text-text-muted">
        {count}
      </span>
    </div>
  );
}

/* ── Main component ───────────────────────────────────────────────── */

export function OverviewSection({ analytics, loading, error, projectId }: Props) {
  /* Loading state */
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-muted">
        <Loader2 className="h-6 w-6 animate-spin" />
        <span className="text-sm">Loading project overview...</span>
      </div>
    );
  }

  /* Error state */
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-danger">
        <AlertCircle className="h-6 w-6" />
        <span className="text-sm">{error}</span>
      </div>
    );
  }

  /* Empty state */
  if (!analytics) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-dim">
        <Inbox className="h-8 w-8" />
        <span className="text-sm">No analytics data available for this project.</span>
      </div>
    );
  }

  /* Derived metrics */
  const { project, code_breakdown } = analytics;
  const totalUnits = Object.values(code_breakdown.units_by_type).reduce(
    (sum, n) => sum + n,
    0,
  );
  const totalRelationships = Object.values(code_breakdown.edges_by_type).reduce(
    (sum, n) => sum + n,
    0,
  );

  const langEntries = Object.entries(code_breakdown.files_by_language).sort(
    ([, a], [, b]) => b - a,
  );
  const maxLangCount = langEntries.length > 0 ? langEntries[0][1] : 0;

  return (
    <div className="p-6 space-y-6">
      {/* Name + description */}
      <div>
        <h2 className="text-xl font-semibold text-text">{project.name}</h2>
        {project.primary_language && (
          <p className="mt-1 text-sm text-text-muted">
            Primary language: {project.primary_language}
          </p>
        )}
      </div>

      {/* Status badges */}
      <div className="flex flex-wrap items-center gap-4">
        <StatusBadge label="AST Status" value={project.ast_status} />
        <StatusBadge label="ASG Status" value={project.asg_status} />
        <StatusBadge label="Deep Analysis" value={project.deep_analysis_status} />
      </div>

      {/* Metric cards (2x2 on sm, 4-col on lg) */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard icon={FileText} label="Total Files" value={project.file_count} />
        <MetricCard icon={Blocks} label="Code Units" value={totalUnits} />
        <MetricCard icon={GitFork} label="Relationships" value={totalRelationships} />
        <MetricCard icon={Hash} label="Total Lines" value={project.total_lines} />
      </div>

      {/* Language distribution */}
      {langEntries.length > 0 && (
        <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
          <h3 className="text-sm font-semibold text-text mb-3">
            Language Distribution
          </h3>
          <div className="space-y-2.5">
            {langEntries.map(([lang, count]) => (
              <LanguageBar
                key={lang}
                language={lang}
                count={count}
                maxCount={maxLangCount}
              />
            ))}
          </div>
        </div>
      )}

      {/* Quick links */}
      <div className="flex flex-wrap gap-3">
        <Link
          to={`/project/${projectId}`}
          className="inline-flex items-center gap-1.5 rounded-lg border border-void-surface bg-void-light/30 px-4 py-2 text-xs font-medium text-text-muted transition-colors hover:border-glow/40 hover:text-glow"
        >
          <ExternalLink className="h-3.5 w-3.5" />
          Project Page
        </Link>
        <Link
          to={`/project/${projectId}/chat`}
          className="inline-flex items-center gap-1.5 rounded-lg border border-void-surface bg-void-light/30 px-4 py-2 text-xs font-medium text-text-muted transition-colors hover:border-glow/40 hover:text-glow"
        >
          <MessageSquare className="h-3.5 w-3.5" />
          Chat
        </Link>
        <Link
          to={`/project/${projectId}/analytics`}
          className="inline-flex items-center gap-1.5 rounded-lg border border-void-surface bg-void-light/30 px-4 py-2 text-xs font-medium text-text-muted transition-colors hover:border-glow/40 hover:text-glow"
        >
          <BarChart3 className="h-3.5 w-3.5" />
          Analytics
        </Link>
      </div>
    </div>
  );
}
