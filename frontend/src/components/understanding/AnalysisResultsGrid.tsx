/**
 * AnalysisResultsGrid — card grid of analysis results with coverage bar.
 * Each card shows tier, confidence, name, file, narrative excerpt, and stats.
 */

import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  BarChart3,
  Layers,
  Hash,
  Loader2,
} from 'lucide-react';
import type { AnalysisSummary } from '../../types/index.ts';

const TIER_STYLES: Record<string, string> = {
  tier_1: 'bg-success/10 text-success',
  tier_2: 'bg-warning/10 text-warning',
  tier_3: 'bg-danger/10 text-danger',
};

const ENTRY_TYPE_BADGE: Record<string, string> = {
  http_endpoint: 'bg-glow/10 text-glow',
  message_handler: 'bg-nebula/10 text-nebula-bright',
  scheduled_task: 'bg-warning/10 text-warning',
  cli_command: 'bg-success/10 text-success',
  event_listener: 'bg-glow-bright/10 text-glow-bright',
  startup_hook: 'bg-danger/10 text-danger',
  public_api: 'bg-nebula/10 text-nebula',
  unknown: 'bg-void-surface text-text-dim',
};

function confidenceColor(score: number): string {
  if (score >= 0.8) return 'text-success';
  if (score >= 0.5) return 'text-warning';
  return 'text-danger';
}

function confidenceBg(score: number): string {
  if (score >= 0.8) return 'bg-success';
  if (score >= 0.5) return 'bg-warning';
  return 'bg-danger';
}

function coverageBarColor(pct: number): string {
  if (pct >= 80) return 'bg-success';
  if (pct >= 50) return 'bg-warning';
  return 'bg-danger';
}

interface AnalysisResultsGridProps {
  analyses: AnalysisSummary[];
  coveragePct: number;
  isLoading: boolean;
  onSelectAnalysis: (analysisId: string) => void;
}

export function AnalysisResultsGrid({
  analyses,
  coveragePct,
  isLoading,
  onSelectAnalysis,
}: AnalysisResultsGridProps) {
  const [isExpanded, setIsExpanded] = useState(true);

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
            Analysis Results
          </h3>
          <span className="rounded-full bg-void-surface px-1.5 py-0.5 text-[9px] font-medium text-text-dim">
            {analyses.length}
          </span>
        </div>
        {analyses.length > 0 && (
          <span className="text-[10px] text-text-dim/60">
            {Math.round(coveragePct)}% code coverage
          </span>
        )}
      </button>

      {/* Body */}
      {isExpanded && (
        <div className="border-t border-void-surface/30 px-4 pb-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
            </div>
          ) : analyses.length === 0 ? (
            <p className="py-4 text-center text-[10px] text-text-dim/50">
              No analysis results yet. Start an analysis to generate chain
              narratives.
            </p>
          ) : (
            <>
              {/* Coverage bar */}
              <div className="mt-3 mb-4">
                <div className="flex items-center justify-between text-[10px] text-text-dim">
                  <span className="flex items-center gap-1">
                    <BarChart3 className="h-3 w-3" />
                    Overall Coverage
                  </span>
                  <span>{Math.round(coveragePct)}%</span>
                </div>
                <div className="mt-1 h-1 rounded-full bg-void-surface">
                  <div
                    className={`h-1 rounded-full ${coverageBarColor(coveragePct)} transition-all duration-500`}
                    style={{ width: `${Math.min(coveragePct, 100)}%` }}
                  />
                </div>
              </div>

              {/* Card grid */}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {analyses.map((a) => (
                  <AnalysisCard
                    key={a.analysis_id}
                    analysis={a}
                    onClick={() => onSelectAnalysis(a.analysis_id)}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function AnalysisCard({
  analysis,
  onClick,
}: {
  analysis: AnalysisSummary;
  onClick: () => void;
}) {
  const tierStyle =
    TIER_STYLES[analysis.tier] ?? 'bg-void-surface text-text-dim';
  const entryStyle =
    ENTRY_TYPE_BADGE[analysis.entry_type] ?? 'bg-void-surface text-text-dim';
  const confColor = confidenceColor(analysis.confidence_score);
  const confBg = confidenceBg(analysis.confidence_score);
  const confPct = Math.round(analysis.confidence_score * 100);

  return (
    <button
      onClick={onClick}
      className="group rounded-lg border border-void-surface/50 bg-void-light/40 p-4 text-left transition-all hover:border-void-surface hover:bg-void-light"
    >
      {/* Top row: tier badge + confidence bar */}
      <div className="flex items-center justify-between">
        <span
          className={`rounded px-1.5 py-0.5 text-[8px] font-bold uppercase ${tierStyle}`}
        >
          {analysis.tier.replace('_', ' ')}
        </span>
        <div className="flex items-center gap-1.5">
          <span className={`text-[10px] font-medium ${confColor}`}>
            {confPct}%
          </span>
          <div className="h-1 w-10 rounded-full bg-void-surface">
            <div
              className={`h-1 rounded-full ${confBg}`}
              style={{ width: `${confPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Name */}
      <p className="mt-2 truncate font-[family-name:var(--font-code)] text-[11px] font-medium text-text group-hover:text-glow-bright">
        {analysis.entry_name}
      </p>

      {/* File path */}
      <p className="mt-0.5 truncate text-[10px] text-text-dim">
        {analysis.entry_file}
      </p>

      {/* Narrative excerpt */}
      {analysis.narrative && (
        <p className="mt-2 line-clamp-2 text-[10px] leading-relaxed text-text-muted">
          {analysis.narrative}
        </p>
      )}

      {/* Bottom stats */}
      <div className="mt-3 flex items-center gap-2 text-[9px] text-text-dim/60">
        <span className="flex items-center gap-0.5">
          <Layers className="h-2.5 w-2.5" />
          {analysis.total_units} units
        </span>
        <span className="flex items-center gap-0.5">
          <Hash className="h-2.5 w-2.5" />
          {analysis.total_tokens.toLocaleString()} tokens
        </span>
        <span
          className={`ml-auto rounded px-1 py-0.5 text-[8px] font-bold uppercase ${entryStyle}`}
        >
          {analysis.entry_type.replace(/_/g, ' ').slice(0, 6)}
        </span>
      </div>
    </button>
  );
}
