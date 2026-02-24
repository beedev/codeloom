/**
 * ArchitectureSection -- Project Wiki architecture panel.
 *
 * Displays entry points grouped by type, edge distribution bars,
 * and code unit breakdown from the ASG graph overview.
 */

import { useEffect } from 'react';
import {
  Loader2,
  AlertCircle,
  Inbox,
  Globe,
  Mail,
  Clock,
  Terminal,
  Radio,
  Rocket,
  Package,
  HelpCircle,
} from 'lucide-react';
import type {
  ProjectAnalytics,
  EntryPoint,
  EntryPointType,
  GraphOverview,
} from '../../types/index.ts';

interface Props {
  projectId: string;
  analytics: ProjectAnalytics | null;
  entryPoints: EntryPoint[] | null;
  graphOverview: GraphOverview | null;
  loading?: boolean;
  error?: string | null;
  onLoad: () => Promise<void>;
}

/* ── Entry-type display config ────────────────────────────────────── */

const ENTRY_TYPE_META: Record<
  EntryPointType,
  { label: string; icon: typeof Globe }
> = {
  http_endpoint: { label: 'HTTP Endpoints', icon: Globe },
  message_handler: { label: 'Message Handlers', icon: Mail },
  scheduled_task: { label: 'Scheduled Tasks', icon: Clock },
  cli_command: { label: 'CLI Commands', icon: Terminal },
  event_listener: { label: 'Event Listeners', icon: Radio },
  startup_hook: { label: 'Startup Hooks', icon: Rocket },
  public_api: { label: 'Public API', icon: Package },
  unknown: { label: 'Unknown', icon: HelpCircle },
};

/* ── Horizontal bar ───────────────────────────────────────────────── */

function HorizontalBar({
  label,
  count,
  maxCount,
}: {
  label: string;
  count: number;
  maxCount: number;
}) {
  const pct = maxCount > 0 ? (count / maxCount) * 100 : 0;
  return (
    <div className="flex items-center gap-3">
      <span className="w-28 shrink-0 truncate text-xs font-medium text-text">
        {label}
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

export function ArchitectureSection({
  analytics,
  entryPoints,
  loading,
  error,
  onLoad,
}: Props) {
  /* Lazy-load on mount when data hasn't been fetched */
  useEffect(() => {
    if (entryPoints === null) {
      void onLoad();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* Loading state */
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-muted">
        <Loader2 className="h-6 w-6 animate-spin" />
        <span className="text-sm">Loading architecture data...</span>
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
  if (!entryPoints || entryPoints.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-dim">
        <Inbox className="h-8 w-8" />
        <span className="text-sm">No architecture data available. Build the ASG first.</span>
      </div>
    );
  }

  /* Group entry points by type */
  const grouped = new Map<EntryPointType, EntryPoint[]>();
  for (const ep of entryPoints) {
    const list = grouped.get(ep.entry_type) ?? [];
    list.push(ep);
    grouped.set(ep.entry_type, list);
  }

  /* Edge distribution from analytics */
  const edgeEntries = analytics
    ? Object.entries(analytics.code_breakdown.edges_by_type).sort(([, a], [, b]) => b - a)
    : [];
  const maxEdge = edgeEntries.length > 0 ? edgeEntries[0][1] : 0;

  /* Unit-type breakdown from analytics */
  const unitEntries = analytics
    ? Object.entries(analytics.code_breakdown.units_by_type).sort(([, a], [, b]) => b - a)
    : [];
  const maxUnit = unitEntries.length > 0 ? unitEntries[0][1] : 0;

  return (
    <div className="p-6 space-y-6">
      {/* ── Entry Points ─────────────────────────────────────────── */}
      <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
        <h3 className="text-sm font-semibold text-text mb-3">Entry Points</h3>
        <div className="space-y-5">
          {Array.from(grouped.entries()).map(([type, eps]) => {
            const meta = ENTRY_TYPE_META[type] ?? ENTRY_TYPE_META.unknown;
            const Icon = meta.icon;
            return (
              <div key={type}>
                <div className="flex items-center gap-2 mb-2">
                  <Icon className="h-3.5 w-3.5 text-glow" />
                  <span className="text-xs font-medium text-text">{meta.label}</span>
                  <span className="inline-flex items-center rounded-full bg-glow/15 px-2 py-0.5 text-[10px] font-medium text-glow">
                    {eps.length}
                  </span>
                </div>
                <ul className="ml-5 space-y-1">
                  {eps.map((ep) => (
                    <li key={ep.unit_id} className="text-xs">
                      <span className="font-mono text-text">{ep.qualified_name}</span>
                      <span className="ml-2 text-text-dim">{ep.file_path}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Edge Distribution ─────────────────────────────────────── */}
      {edgeEntries.length > 0 && (
        <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
          <h3 className="text-sm font-semibold text-text mb-3">Edge Distribution</h3>
          <div className="space-y-2.5">
            {edgeEntries.map(([edgeType, count]) => (
              <HorizontalBar
                key={edgeType}
                label={edgeType}
                count={count}
                maxCount={maxEdge}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Code Units by Type ────────────────────────────────────── */}
      {unitEntries.length > 0 && (
        <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
          <h3 className="text-sm font-semibold text-text mb-3">Code Units by Type</h3>
          <div className="space-y-2.5">
            {unitEntries.map(([unitType, count]) => (
              <HorizontalBar
                key={unitType}
                label={unitType}
                count={count}
                maxCount={maxUnit}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
