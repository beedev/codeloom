/**
 * UnderstandingSection -- Project Wiki deep-analysis panel.
 *
 * Shows analysis coverage, a grid of analysis summary cards,
 * and expandable chain details with business rules, data entities,
 * integrations, side effects, and cross-cutting concerns.
 */

import { useEffect, useState, useCallback } from 'react';
import {
  Loader2,
  AlertCircle,
  Inbox,
  ChevronDown,
  ChevronRight,
  FileCode,
} from 'lucide-react';
import type {
  AnalysisSummary,
  ChainDetail,
  EvidenceRef,
} from '../../types/index.ts';

interface Props {
  projectId: string;
  analysisResults: {
    analyses: AnalysisSummary[];
    count: number;
    coverage_pct: number;
  } | null;
  loading?: boolean;
  error?: string | null;
  onLoad: () => Promise<void>;
  onLoadChain: (analysisId: string) => Promise<ChainDetail | null>;
}

/* ── Severity badge mapping ───────────────────────────────────────── */

const SEVERITY_CLASSES: Record<string, string> = {
  low: 'text-text-dim',
  medium: 'text-warning',
  high: 'text-danger',
};

/* ── Small evidence link ──────────────────────────────────────────── */

function EvidenceLink({ evidence }: { evidence: EvidenceRef }) {
  return (
    <span className="inline-flex items-center gap-1 rounded bg-void-surface/60 px-1.5 py-0.5 font-code text-[10px] text-text-dim">
      <FileCode className="h-2.5 w-2.5 shrink-0" />
      <span className="truncate max-w-[200px]">{evidence.qualified_name}</span>
      <span className="text-text-dim/60">
        @ {evidence.file_path}:{evidence.start_line}
      </span>
    </span>
  );
}

/* ── Expandable chain detail ──────────────────────────────────────── */

function ChainDetailView({
  detail,
  loadingDetail,
}: {
  detail: ChainDetail | null;
  loadingDetail: boolean;
}) {
  if (loadingDetail) {
    return (
      <div className="flex items-center gap-2 py-4 text-text-muted">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="text-xs">Loading chain detail...</span>
      </div>
    );
  }

  if (!detail) {
    return (
      <p className="py-3 text-xs text-text-dim">
        Failed to load chain detail.
      </p>
    );
  }

  const { result } = detail;

  return (
    <div className="mt-3 space-y-4 border-t border-void-surface pt-4">
      {/* Business Rules */}
      {result.business_rules.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text mb-2">Business Rules</h4>
          <ol className="list-decimal list-inside space-y-2">
            {result.business_rules.map((rule, idx) => (
              <li key={idx} className="text-xs text-text-muted">
                <span>{rule.description}</span>
                {rule.evidence.length > 0 && (
                  <div className="mt-1 ml-4 flex flex-wrap gap-1">
                    {rule.evidence.map((ev, evIdx) => (
                      <EvidenceLink key={evIdx} evidence={ev} />
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Data Entities */}
      {result.data_entities.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text mb-2">Data Entities</h4>
          <ul className="space-y-1.5">
            {result.data_entities.map((entity, idx) => (
              <li key={idx} className="text-xs">
                <span className="font-medium text-text">{entity.name}</span>
                <span className="mx-1.5 text-text-dim">({entity.type})</span>
                <span className="text-text-muted">{entity.description}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Integrations */}
      {result.integrations.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text mb-2">Integrations</h4>
          <ul className="space-y-1.5">
            {result.integrations.map((integ, idx) => (
              <li key={idx} className="text-xs">
                <span className="font-medium text-text">{integ.name}</span>
                <span className="mx-1.5 text-text-dim">({integ.type})</span>
                <span className="text-text-muted">{integ.description}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Side Effects */}
      {result.side_effects.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text mb-2">Side Effects</h4>
          <ul className="space-y-1.5">
            {result.side_effects.map((effect, idx) => (
              <li key={idx} className="flex items-start gap-2 text-xs">
                <span
                  className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                    SEVERITY_CLASSES[effect.severity] ?? 'text-text-dim'
                  } ${
                    effect.severity === 'high'
                      ? 'border-danger/30 bg-danger/10'
                      : effect.severity === 'medium'
                        ? 'border-warning/30 bg-warning/10'
                        : 'border-void-surface bg-void-surface/50'
                  }`}
                >
                  {effect.severity}
                </span>
                <span className="text-text-muted">{effect.description}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Cross-Cutting Concerns */}
      {result.cross_cutting_concerns.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text mb-2">
            Cross-Cutting Concerns
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {result.cross_cutting_concerns.map((concern) => (
              <span
                key={concern}
                className="rounded-full border border-nebula/30 bg-nebula/10 px-2.5 py-0.5 text-[11px] font-medium text-nebula-bright"
              >
                {concern}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Analysis card ────────────────────────────────────────────────── */

function AnalysisCard({
  summary,
  expanded,
  onToggle,
  chainDetail,
  loadingDetail,
}: {
  summary: AnalysisSummary;
  expanded: boolean;
  onToggle: () => void;
  chainDetail: ChainDetail | null;
  loadingDetail: boolean;
}) {
  const Chevron = expanded ? ChevronDown : ChevronRight;
  const tierLabel = summary.tier.replace('_', ' ').toUpperCase();
  const confidencePct = Math.round(summary.confidence_score * 100);
  const narrativePreview =
    summary.narrative && summary.narrative.length > 200
      ? summary.narrative.slice(0, 200) + '...'
      : summary.narrative;

  return (
    <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-start gap-3 text-left"
      >
        <Chevron className="mt-0.5 h-4 w-4 shrink-0 text-text-dim" />
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-text truncate">
              {summary.entry_name}
            </span>
            <span className="inline-flex rounded-full border border-glow/30 bg-glow/10 px-2 py-0.5 text-[10px] font-medium text-glow">
              {summary.entry_type}
            </span>
            <span className="inline-flex rounded-full border border-nebula/30 bg-nebula/10 px-2 py-0.5 text-[10px] font-medium text-nebula-bright">
              {tierLabel}
            </span>
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-3 text-xs text-text-dim">
            <span>Confidence: {confidencePct}%</span>
            <span>Units: {summary.total_units}</span>
          </div>
          {!expanded && narrativePreview && (
            <p className="mt-2 text-xs text-text-muted leading-relaxed line-clamp-3">
              {narrativePreview}
            </p>
          )}
        </div>
      </button>

      {expanded && (
        <ChainDetailView detail={chainDetail} loadingDetail={loadingDetail} />
      )}
    </div>
  );
}

/* ── Main component ───────────────────────────────────────────────── */

export function UnderstandingSection({
  analysisResults,
  loading,
  error,
  onLoad,
  onLoadChain,
}: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [chainDetail, setChainDetail] = useState<ChainDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  /* Lazy-load on mount when data hasn't been fetched */
  useEffect(() => {
    if (analysisResults === null) {
      void onLoad();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* Fetch chain detail when a card is expanded */
  const handleToggle = useCallback(
    async (analysisId: string) => {
      if (expandedId === analysisId) {
        setExpandedId(null);
        setChainDetail(null);
        return;
      }
      setExpandedId(analysisId);
      setChainDetail(null);
      setLoadingDetail(true);
      try {
        const detail = await onLoadChain(analysisId);
        setChainDetail(detail);
      } finally {
        setLoadingDetail(false);
      }
    },
    [expandedId, onLoadChain],
  );

  /* Loading state */
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-muted">
        <Loader2 className="h-6 w-6 animate-spin" />
        <span className="text-sm">Loading analysis results...</span>
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

  /* Empty / no analyses */
  if (!analysisResults || analysisResults.analyses.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-dim">
        <Inbox className="h-8 w-8" />
        <span className="text-sm">
          No deep analysis has been run. Trigger analysis from the project page.
        </span>
      </div>
    );
  }

  const { analyses, coverage_pct } = analysisResults;
  const coverageRounded = Math.round(coverage_pct);

  return (
    <div className="p-6 space-y-6">
      {/* Coverage bar */}
      <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-text">Analysis Coverage</h3>
          <span className="text-sm font-medium text-glow">{coverageRounded}%</span>
        </div>
        <div className="h-2.5 w-full rounded-full bg-void-surface overflow-hidden">
          <div
            className="h-full rounded-full bg-glow transition-all duration-500"
            style={{ width: `${coverageRounded}%` }}
          />
        </div>
      </div>

      {/* Analysis cards grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {analyses.map((summary) => (
          <AnalysisCard
            key={summary.analysis_id}
            summary={summary}
            expanded={expandedId === summary.analysis_id}
            onToggle={() => void handleToggle(summary.analysis_id)}
            chainDetail={
              expandedId === summary.analysis_id ? chainDetail : null
            }
            loadingDetail={
              expandedId === summary.analysis_id && loadingDetail
            }
          />
        ))}
      </div>
    </div>
  );
}
