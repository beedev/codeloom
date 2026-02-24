/**
 * MvpCatalogSection -- Project Wiki MVP catalog panel.
 *
 * Renders a responsive card grid of functional MVPs with inline
 * expandable detail views (files, units, architecture mapping).
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Loader2,
  Inbox,
  Boxes,
  Link2,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import type {
  MigrationPlan,
  MvpDetail,
  FunctionalMvpSummary,
} from '../../types/index.ts';

/* ── Props ───────────────────────────────────────────────────────── */

interface Props {
  migrationPlan: MigrationPlan | null;
  loading?: boolean;
  onLoad: () => Promise<void>;
  onLoadDetail: (mvpId: number) => Promise<MvpDetail | null>;
}

/* ── Status badge colors ─────────────────────────────────────────── */

const MVP_STATUS_COLORS: Record<FunctionalMvpSummary['status'], string> = {
  discovered: 'bg-void-surface/50 text-text-dim border-void-surface',
  refined: 'bg-nebula/15 text-nebula-bright border-nebula/30',
  in_progress: 'bg-glow/15 text-glow border-glow/30',
  migrated: 'bg-success/15 text-success border-success/30',
};

/* ── Metric bar component ────────────────────────────────────────── */

function MetricBar({ label, value }: { label: string; value: number }) {
  const v = value ?? 0;
  const pct = Math.min(Math.max(v, 0), 1) * 100;
  const colorClass = v > 0.7 ? 'bg-success' : v > 0.4 ? 'bg-warning' : 'bg-danger';

  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium text-text-dim">{label}</span>
        <span className="text-[10px] tabular-nums text-text-muted">{v.toFixed(2)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-void-surface overflow-hidden">
        <div
          className={`h-full rounded-full ${colorClass} transition-all duration-300`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

/* ── MVP detail expansion ────────────────────────────────────────── */

function MvpDetailPanel({ detail }: { detail: MvpDetail }) {
  return (
    <div className="space-y-4">
      {/* Files table */}
      {detail.files.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text mb-2">
            Files ({detail.files.length})
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-void-surface text-left">
                  <th className="py-1.5 pr-4 font-medium text-text-dim">Path</th>
                  <th className="py-1.5 pr-4 font-medium text-text-dim">Language</th>
                  <th className="py-1.5 font-medium text-text-dim text-right">Lines</th>
                </tr>
              </thead>
              <tbody>
                {detail.files.map((f) => (
                  <tr key={f.file_id} className="border-b border-void-surface/50">
                    <td className="py-1.5 pr-4 font-mono text-text-muted truncate max-w-xs">
                      {f.file_path}
                    </td>
                    <td className="py-1.5 pr-4 text-text-dim">{f.language}</td>
                    <td className="py-1.5 text-right tabular-nums text-text-muted">
                      {f.line_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Units table */}
      {detail.units.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text mb-2">
            Units ({detail.units.length})
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-void-surface text-left">
                  <th className="py-1.5 pr-4 font-medium text-text-dim">Name</th>
                  <th className="py-1.5 pr-4 font-medium text-text-dim">Type</th>
                  <th className="py-1.5 pr-4 font-medium text-text-dim">File</th>
                  <th className="py-1.5 font-medium text-text-dim text-right">Lines</th>
                </tr>
              </thead>
              <tbody>
                {detail.units.map((u) => (
                  <tr key={u.unit_id} className="border-b border-void-surface/50">
                    <td className="py-1.5 pr-4 font-medium text-text">{u.name}</td>
                    <td className="py-1.5 pr-4 text-text-dim">{u.unit_type}</td>
                    <td className="py-1.5 pr-4 font-mono text-text-muted truncate max-w-xs">
                      {u.file_path}
                    </td>
                    <td className="py-1.5 text-right tabular-nums text-text-muted">
                      {u.start_line}&ndash;{u.end_line}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Architecture mapping table */}
      {detail.architecture_mapping && detail.architecture_mapping.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text mb-2">
            Architecture Mapping
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-void-surface text-left">
                  <th className="py-1.5 pr-4 font-medium text-text-dim">Source</th>
                  <th className="py-1.5 pr-4 font-medium text-text-dim">Target</th>
                  <th className="py-1.5 font-medium text-text-dim">Changes</th>
                </tr>
              </thead>
              <tbody>
                {detail.architecture_mapping.map((m, i) => (
                  <tr key={i} className="border-b border-void-surface/50">
                    <td className="py-1.5 pr-4 font-mono text-text-muted truncate max-w-xs">
                      {m.source_path}
                    </td>
                    <td className="py-1.5 pr-4 font-mono text-text-muted truncate max-w-xs">
                      {m.target_path}
                    </td>
                    <td className="py-1.5 text-text-dim">{m.changes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── MVP card ────────────────────────────────────────────────────── */

function MvpCard({
  mvp,
  isExpanded,
  onToggle,
  onLoadDetail,
}: {
  mvp: FunctionalMvpSummary;
  isExpanded: boolean;
  onToggle: () => void;
  onLoadDetail: (mvpId: number) => Promise<MvpDetail | null>;
}) {
  const [detail, setDetail] = useState<MvpDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const handleToggle = useCallback(async () => {
    onToggle();
    if (!isExpanded && !detail) {
      setLoadingDetail(true);
      const result = await onLoadDetail(mvp.mvp_id);
      setDetail(result);
      setLoadingDetail(false);
    }
  }, [isExpanded, detail, onLoadDetail, mvp.mvp_id, onToggle]);

  const statusClass = MVP_STATUS_COLORS[mvp.status] ?? MVP_STATUS_COLORS.discovered;

  return (
    <div
      className={`rounded-xl border bg-void-light/30 transition-colors ${
        isExpanded ? 'border-glow/30 col-span-full' : 'border-void-surface'
      }`}
    >
      {/* Card header / summary -- always visible */}
      <button
        onClick={handleToggle}
        className="w-full text-left p-5 transition-colors hover:bg-void-surface/20"
      >
        <div className="flex items-start justify-between gap-2 mb-2">
          <h3 className="text-sm font-semibold text-text leading-snug">{mvp.name}</h3>
          <div className="flex items-center gap-2 shrink-0">
            <span className="inline-flex items-center rounded-md bg-glow/10 px-1.5 py-0.5 text-[10px] font-semibold text-glow">
              #{mvp.priority}
            </span>
            <span
              className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium ${statusClass}`}
            >
              {mvp.status.replace('_', ' ')}
            </span>
          </div>
        </div>

        {mvp.description && (
          <p className="text-xs text-text-muted line-clamp-2 mb-3 leading-relaxed">
            {mvp.description}
          </p>
        )}

        {/* Metric bars */}
        <div className="space-y-1.5">
          <MetricBar label="Cohesion" value={mvp.metrics.cohesion} />
          <MetricBar label="Coupling" value={mvp.metrics.coupling} />
          <MetricBar label="Readiness" value={mvp.metrics.readiness} />
        </div>

        <div className="mt-3 flex items-center justify-between">
          <span className="text-[10px] text-text-dim">
            Size: {mvp.metrics.size ?? 0} units
          </span>

          <div className="flex items-center gap-1.5">
            {mvp.depends_on_mvp_ids.length > 0 && (
              <div className="flex items-center gap-1">
                <Link2 className="h-3 w-3 text-text-dim" />
                {mvp.depends_on_mvp_ids.map((depId) => (
                  <span
                    key={depId}
                    className="inline-flex items-center rounded-md bg-void-surface px-1.5 py-0.5 text-[9px] font-medium text-text-dim"
                  >
                    MVP-{depId}
                  </span>
                ))}
              </div>
            )}

            {isExpanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-text-dim" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-text-dim" />
            )}
          </div>
        </div>
      </button>

      {/* Expanded detail */}
      {isExpanded && (
        <div className="border-t border-void-surface px-5 py-4">
          {loadingDetail ? (
            <div className="flex items-center gap-2 py-6 text-text-muted justify-center">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-xs">Loading MVP details...</span>
            </div>
          ) : detail ? (
            <MvpDetailPanel detail={detail} />
          ) : (
            <p className="text-xs text-text-dim py-4 text-center">
              Unable to load details for this MVP.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Main component ──────────────────────────────────────────────── */

export function MvpCatalogSection({
  migrationPlan,
  loading,
  onLoad,
  onLoadDetail,
}: Props) {
  const [expandedMvpId, setExpandedMvpId] = useState<number | null>(null);

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
        <span className="text-sm">Loading MVP catalog...</span>
      </div>
    );
  }

  /* Empty state */
  if (!migrationPlan || migrationPlan.mvps.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-dim">
        <Boxes className="h-8 w-8" />
        <span className="text-sm">
          {migrationPlan
            ? 'No MVPs discovered yet. Run the discovery phase first.'
            : 'No migration plan available.'}
        </span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-text">MVP Catalog</h2>
        <span className="text-xs text-text-dim">
          {migrationPlan.mvps.length} MVP{migrationPlan.mvps.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {migrationPlan.mvps.map((mvp) => (
          <MvpCard
            key={mvp.mvp_id}
            mvp={mvp}
            isExpanded={expandedMvpId === mvp.mvp_id}
            onToggle={() =>
              setExpandedMvpId((prev) => (prev === mvp.mvp_id ? null : mvp.mvp_id))
            }
            onLoadDetail={onLoadDetail}
          />
        ))}
      </div>
    </div>
  );
}
