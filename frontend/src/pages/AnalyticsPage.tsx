/**
 * AnalyticsPage — Project analytics dashboard.
 *
 * Aggregates code breakdown, migration progress, understanding coverage,
 * and LLM gateway metrics into a single view.
 * Design based on Stitch-generated mockup.
 */

import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Loader2,
  AlertCircle,
  FileCode2,
  Boxes,
  Network,
  FileText,
  Activity,
  Brain,
  Zap,
  ChevronRight,
  BarChart3,
  ArrowRightLeft,
  Search,
  Database,
  Bot,
} from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import * as api from '../services/api.ts';
import type { ProjectAnalytics } from '../types/index.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function StatusBadge({ label, status }: { label: string; status: string }) {
  const isGood = ['complete', 'completed'].includes(status);
  const isRunning = ['parsing', 'running', 'pending'].includes(status);
  const colorClass = isGood
    ? 'bg-success/10 text-success border-success/20'
    : isRunning
      ? 'bg-glow/10 text-glow border-glow/20'
      : 'bg-danger/10 text-danger border-danger/20';

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium ${colorClass}`}>
      {isGood && <span className="material-symbols-outlined text-sm">check_circle</span>}
      {isRunning && <Loader2 className="h-3 w-3 animate-spin" />}
      {label}: {status}
    </span>
  );
}

function MetricCard({
  icon,
  label,
  value,
  subtitle,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  subtitle?: string;
}) {
  return (
    <div className="rounded-xl border border-void-surface bg-void-light p-5">
      <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-text-muted">
        {icon}
        {label}
      </div>
      <div className="font-code text-3xl font-bold text-text">{value}</div>
      {subtitle && (
        <p className="mt-2 text-xs text-text-dim">{subtitle}</p>
      )}
    </div>
  );
}

function HorizontalBar({
  items,
  colorClass = 'bg-glow',
}: {
  items: [string, number][];
  colorClass?: string;
}) {
  const maxVal = Math.max(...items.map(([, v]) => v), 1);
  return (
    <div className="space-y-2">
      {items.map(([label, count]) => (
        <div key={label} className="flex items-center gap-3">
          <span className="w-20 truncate text-right text-xs text-text-dim">{label}</span>
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-void-surface">
            <div
              className={`h-full rounded-full ${colorClass}`}
              style={{ width: `${(count / maxVal) * 100}%` }}
            />
          </div>
          <span className="w-12 text-right font-code text-xs text-text-muted">
            {count.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  );
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return n.toLocaleString();
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export function AnalyticsPage() {
  const { id: projectId } = useParams<{ id: string }>();
  const [data, setData] = useState<ProjectAnalytics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError(null);
      try {
        const result = await api.getProjectAnalytics(projectId!);
        if (!cancelled) setData(result);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load analytics');
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [projectId]);

  if (isLoading) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-glow" />
        </div>
      </Layout>
    );
  }

  if (error || !data) {
    return (
      <Layout>
        <div className="flex h-full flex-col items-center justify-center gap-3 text-text-muted">
          <AlertCircle className="h-8 w-8 text-danger" />
          <p>{error || 'No data available'}</p>
          <Link to="/" className="text-sm text-glow hover:underline">Back to Dashboard</Link>
        </div>
      </Layout>
    );
  }

  const { project, code_breakdown, migration, understanding, queries, llm } = data;
  const totalUnits = Object.values(code_breakdown.units_by_type).reduce((a, b) => a + b, 0);
  const totalEdges = Object.values(code_breakdown.edges_by_type).reduce((a, b) => a + b, 0);

  return (
    <Layout>
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[1400px] p-6 lg:p-8">

          {/* Breadcrumb + Project Header */}
          <nav className="mb-2 flex items-center gap-1.5 text-xs text-text-dim">
            <Link to="/" className="hover:text-glow">Projects</Link>
            <ChevronRight className="h-3 w-3" />
            <Link to={`/project/${projectId}`} className="hover:text-glow">{project.name}</Link>
            <ChevronRight className="h-3 w-3" />
            <span className="text-text">Analytics</span>
          </nav>

          <div className="mb-8 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <h1 className="text-2xl font-bold text-text">{project.name}</h1>
            <div className="flex flex-wrap gap-2">
              <StatusBadge label="AST" status={project.ast_status} />
              <StatusBadge label="ASG" status={project.asg_status} />
              <StatusBadge label="Deep Analysis" status={project.deep_analysis_status} />
            </div>
          </div>

          {/* Row 1: Key Metrics */}
          <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              icon={<FileCode2 className="h-3.5 w-3.5 text-glow" />}
              label="Total Files"
              value={project.file_count.toLocaleString()}
              subtitle={`across ${(project.languages || []).length} language${(project.languages || []).length !== 1 ? 's' : ''}`}
            />
            <MetricCard
              icon={<Boxes className="h-3.5 w-3.5 text-glow" />}
              label="Code Units"
              value={totalUnits.toLocaleString()}
              subtitle="classes, methods, functions"
            />
            <MetricCard
              icon={<Network className="h-3.5 w-3.5 text-glow" />}
              label="Relationships"
              value={totalEdges.toLocaleString()}
              subtitle="edges in ASG"
            />
            <MetricCard
              icon={<FileText className="h-3.5 w-3.5 text-glow" />}
              label="Total Lines"
              value={formatNumber(project.total_lines)}
              subtitle={project.file_count > 0 ? `avg ${Math.round(project.total_lines / project.file_count)} lines/file` : undefined}
            />
          </div>

          {/* Row 2: Code Breakdown + Migration Progress */}
          <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-2">

            {/* Code Breakdown */}
            <div className="rounded-xl border border-void-surface bg-void-light p-6">
              <h3 className="mb-6 flex items-center gap-2 font-semibold text-text">
                <BarChart3 className="h-4 w-4 text-glow" />
                Codebase Distribution
              </h3>

              <div className="space-y-6">
                {/* Units by Type */}
                <div>
                  <div className="mb-2 flex justify-between text-xs text-text-dim">
                    <span>Units by Type</span>
                    <span className="font-code">Total: {totalUnits.toLocaleString()}</span>
                  </div>
                  <HorizontalBar
                    items={Object.entries(code_breakdown.units_by_type).sort((a, b) => b[1] - a[1])}
                    colorClass="bg-glow"
                  />
                </div>

                {/* Edges by Type */}
                <div>
                  <div className="mb-2 flex justify-between text-xs text-text-dim">
                    <span>Edges by Type</span>
                    <span className="font-code">Total: {totalEdges.toLocaleString()}</span>
                  </div>
                  <HorizontalBar
                    items={Object.entries(code_breakdown.edges_by_type).sort((a, b) => b[1] - a[1])}
                    colorClass="bg-nebula-bright"
                  />
                </div>

                {/* Files by Language */}
                <div>
                  <div className="mb-2 text-xs text-text-dim">Files by Language</div>
                  <LanguageBar files={code_breakdown.files_by_language} />
                </div>
              </div>
            </div>

            {/* Migration Progress */}
            <div className="rounded-xl border border-void-surface bg-void-light p-6">
              <div className="mb-6 flex items-start justify-between">
                <h3 className="flex items-center gap-2 font-semibold text-text">
                  <ArrowRightLeft className="h-4 w-4 text-glow" />
                  Migration Progress
                </h3>
                {migration.active_plan && (
                  <span className="rounded bg-glow/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-glow">
                    {migration.active_plan.status}
                  </span>
                )}
              </div>

              {migration.plan_count === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-text-dim">
                  <ArrowRightLeft className="mb-3 h-8 w-8 opacity-40" />
                  <p className="text-sm">No migration plans yet</p>
                </div>
              ) : migration.active_plan && (
                <MigrationStats plan={migration.active_plan} />
              )}
            </div>
          </div>

          {/* Row 3: Understanding + LLM Gateway */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">

            {/* Understanding Coverage */}
            <div className="rounded-xl border border-void-surface bg-void-light p-6 lg:col-span-4">
              <h3 className="mb-6 flex items-center gap-2 font-semibold text-text">
                <Brain className="h-4 w-4 text-glow" />
                Analysis Coverage
              </h3>
              <div className="space-y-4">
                <StatRow icon={<Search className="h-4 w-4 text-glow" />} label="Analyses Run" value={understanding.analyses_count} />
                <StatRow icon={<Zap className="h-4 w-4 text-glow" />} label="Entry Points" value={understanding.entry_points_detected} />
                <StatRow icon={<Database className="h-4 w-4 text-glow" />} label="Total Queries" value={queries.total} />
              </div>
            </div>

            {/* LLM Gateway Metrics */}
            <div className="rounded-xl border border-void-surface bg-void-light p-6 lg:col-span-8">
              <div className="mb-6 flex items-center justify-between">
                <h3 className="flex items-center gap-2 font-semibold text-text">
                  <Bot className="h-4 w-4 text-glow" />
                  LLM Gateway Metrics
                </h3>
                {llm.model && (
                  <span className="rounded border border-void-surface bg-void px-2 py-0.5 font-code text-[10px] text-text-muted">
                    {llm.model}
                  </span>
                )}
              </div>

              {!llm.total_calls ? (
                <div className="flex flex-col items-center justify-center py-8 text-text-dim">
                  <Activity className="mb-3 h-8 w-8 opacity-40" />
                  <p className="text-sm">No LLM calls recorded yet</p>
                </div>
              ) : (
                <LLMMetrics llm={llm} />
              )}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-void/50 p-3">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded bg-glow/10">
          {icon}
        </div>
        <span className="text-sm font-medium text-text">{label}</span>
      </div>
      <span className="font-code font-bold text-text">{value.toLocaleString()}</span>
    </div>
  );
}

const LANGUAGE_COLORS: Record<string, string> = {
  java: 'bg-amber-400',
  python: 'bg-blue-500',
  javascript: 'bg-yellow-400',
  typescript: 'bg-sky-500',
  csharp: 'bg-violet-500',
  xml: 'bg-sky-400',
  properties: 'bg-slate-400',
  json: 'bg-emerald-400',
  html: 'bg-orange-500',
  css: 'bg-pink-400',
};

function LanguageBar({ files }: { files: Record<string, number> }) {
  const entries = Object.entries(files).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((a, [, v]) => a + v, 0) || 1;

  return (
    <div>
      <div className="mb-2 flex h-3 overflow-hidden rounded-full">
        {entries.map(([lang, count]) => (
          <div
            key={lang}
            className={`${LANGUAGE_COLORS[lang?.toLowerCase()] || 'bg-slate-500'}`}
            style={{ width: `${(count / total) * 100}%` }}
            title={`${lang}: ${count}`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {entries.map(([lang, count]) => (
          <div key={lang} className="flex items-center gap-1.5 text-[10px] text-text-dim">
            <span className={`inline-block h-2 w-2 rounded-full ${LANGUAGE_COLORS[lang?.toLowerCase()] || 'bg-slate-500'}`} />
            {lang} ({count})
          </div>
        ))}
      </div>
    </div>
  );
}

function MigrationStats({ plan }: { plan: NonNullable<ProjectAnalytics['migration']['active_plan']> }) {
  const mvpTotal = plan.mvps.total || 0;
  const mvpMigrated = plan.mvps.migrated || 0;
  const mvpInProgress = plan.mvps.in_progress || 0;
  const phasesTotal = plan.phases.total || 0;

  // Phase breakdown for stacked bar
  const phaseColors: Record<string, string> = {
    approved: 'bg-success',
    completed: 'bg-success',
    in_progress: 'bg-glow',
    pending: 'bg-void-surface',
    failed: 'bg-danger/60',
  };

  const phaseEntries = Object.entries(plan.phases)
    .filter(([k]) => k !== 'total')
    .sort(([a], [b]) => {
      const order = ['approved', 'completed', 'in_progress', 'pending', 'failed'];
      return order.indexOf(a) - order.indexOf(b);
    });

  return (
    <div className="space-y-5">
      {plan.migration_lane && (
        <p className="font-code text-xs text-text-dim">
          Lane: <span className="text-text-muted">{plan.migration_lane}</span>
          {' '}<span className="text-text-dim">({plan.pipeline_version})</span>
        </p>
      )}

      {/* MVP ring + stats */}
      <div className="flex items-center gap-6">
        <div className="relative h-28 w-28 shrink-0">
          <svg className="-rotate-90" viewBox="0 0 36 36" width="100%" height="100%">
            <circle cx="18" cy="18" r="15.9" fill="none" className="stroke-void-surface" strokeWidth="3" />
            {mvpTotal > 0 && (
              <circle
                cx="18" cy="18" r="15.9" fill="none"
                className="stroke-success"
                strokeWidth="3"
                strokeDasharray={`${(mvpMigrated / mvpTotal) * 100} ${100 - (mvpMigrated / mvpTotal) * 100}`}
                strokeLinecap="round"
              />
            )}
            {mvpTotal > 0 && mvpInProgress > 0 && (
              <circle
                cx="18" cy="18" r="15.9" fill="none"
                className="stroke-glow"
                strokeWidth="3"
                strokeDasharray={`${(mvpInProgress / mvpTotal) * 100} ${100 - (mvpInProgress / mvpTotal) * 100}`}
                strokeDashoffset={`${-(mvpMigrated / mvpTotal) * 100}`}
                strokeLinecap="round"
              />
            )}
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="font-code text-xl font-bold text-text">
              {mvpTotal > 0 ? Math.round((mvpMigrated / mvpTotal) * 100) : 0}%
            </span>
            <span className="text-[8px] uppercase tracking-wider text-text-dim">MVPs</span>
          </div>
        </div>

        <div className="flex-1 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            {Object.entries(plan.mvps)
              .filter(([k]) => k !== 'total')
              .map(([status, count]) => (
                <div key={status}>
                  <p className="text-[10px] uppercase text-text-dim">{status.replace('_', ' ')}</p>
                  <p className="font-code text-lg font-bold text-text">{count}</p>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Phase progress bar */}
      <div>
        <div className="mb-1.5 flex justify-between text-[10px] text-text-dim">
          <span>Phase Progress</span>
          <span>{phasesTotal} total</span>
        </div>
        <div className="flex h-2 gap-0.5 overflow-hidden rounded-full">
          {phaseEntries.map(([status, count]) => (
            <div
              key={status}
              className={`${phaseColors[status] || 'bg-void-surface'}`}
              style={{ flex: count }}
            />
          ))}
        </div>
        <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5">
          {phaseEntries.map(([status, count]) => (
            <span key={status} className="flex items-center gap-1 text-[10px] text-text-dim">
              <span className={`inline-block h-1.5 w-1.5 rounded-full ${phaseColors[status] || 'bg-void-surface'}`} />
              {status.replace('_', ' ')} ({count})
            </span>
          ))}
        </div>
      </div>

      {/* Confidence + Gate Pass Rate */}
      <div className="grid grid-cols-2 gap-4 border-t border-void-surface pt-4">
        <div className="text-center">
          <p className="mb-1 text-[10px] uppercase text-text-dim">Avg Confidence</p>
          <p className="font-code text-2xl font-bold text-glow">
            {plan.avg_confidence != null ? `${Math.round(plan.avg_confidence * 100)}%` : '--'}
          </p>
        </div>
        <div className="text-center">
          <p className="mb-1 text-[10px] uppercase text-text-dim">Gate Pass Rate</p>
          <p className="font-code text-2xl font-bold text-success">
            {plan.gates_pass_rate != null ? `${Math.round(plan.gates_pass_rate * 100)}%` : '--'}
          </p>
        </div>
      </div>
    </div>
  );
}

function LLMMetrics({ llm }: { llm: ProjectAnalytics['llm'] }) {
  const callsByPurpose = Object.entries(llm.calls_by_purpose || {})
    .filter(([k]) => !k.endsWith('_error'))
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-6">
      {/* Top stats row */}
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-[10px] uppercase text-text-dim">Total Calls</p>
          <p className="font-code text-xl font-bold text-text">{(llm.total_calls || 0).toLocaleString()}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase text-text-dim">Est. Cost</p>
          <p className="font-code text-xl font-bold text-success">${(llm.estimated_cost_usd || 0).toFixed(2)}</p>
          {(llm.total_calls || 0) > 0 && (
            <p className="text-[10px] text-text-dim">
              avg ${((llm.estimated_cost_usd || 0) / (llm.total_calls || 1)).toFixed(4)}/call
            </p>
          )}
        </div>
        <div>
          <p className="text-[10px] uppercase text-text-dim">Avg Latency</p>
          <p className="font-code text-xl font-bold text-text">{Math.round(llm.avg_latency_ms || 0)}ms</p>
          <p className="text-[10px] text-text-dim">
            {(llm.errors || 0)} errors, {(llm.retries || 0)} retries
          </p>
        </div>
      </div>

      {/* Token consumption + Call breakdown */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div>
          <p className="mb-3 text-xs font-semibold text-text-dim">Token Consumption</p>
          <div className="space-y-3">
            <TokenBar label="Input Tokens" value={llm.total_tokens_in || 0} max={Math.max(llm.total_tokens_in || 0, llm.total_tokens_out || 0)} />
            <TokenBar label="Output Tokens" value={llm.total_tokens_out || 0} max={Math.max(llm.total_tokens_in || 0, llm.total_tokens_out || 0)} />
          </div>
        </div>

        <div>
          <p className="mb-3 text-xs font-semibold text-text-dim">Calls by Purpose</p>
          {callsByPurpose.length > 0 ? (
            <HorizontalBar items={callsByPurpose} colorClass="bg-glow" />
          ) : (
            <p className="text-xs text-text-dim">No purpose data</p>
          )}
        </div>
      </div>
    </div>
  );
}

function TokenBar({ label, value, max }: { label: string; value: number; max: number }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div>
      <div className="mb-1 flex justify-between text-[10px]">
        <span className="text-text-dim">{label}</span>
        <span className="font-code text-text-muted">{formatNumber(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-void-surface">
        <div className="h-full rounded-full bg-glow/60" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
