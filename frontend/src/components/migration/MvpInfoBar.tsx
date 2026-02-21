/**
 * MvpInfoBar — collapsible detail bar for the selected MVP in execution view.
 *
 * Shows MVP name, metrics summary, and expandable sections for:
 * - Source files and code units that will be migrated
 * - Architecture mapping table (from plan-level Architecture phase output)
 * - Deep Analyze button + output (on-demand LLM analysis for V2 pipeline)
 *
 * Lazy-fetches resolved detail via getMvpDetail() on expand.
 */

import { useState, useEffect, useCallback } from 'react';
import { ChevronDown, ChevronRight, FileCode, Box, Loader2, Sparkles, Table2, LayoutGrid } from 'lucide-react';
import type { FunctionalMvpSummary, MvpDetail } from '../../types/index.ts';
import * as api from '../../services/api.ts';
import { MvpDiagramPanel } from './MvpDiagramPanel.tsx';

interface MvpInfoBarProps {
  planId: string;
  mvp: FunctionalMvpSummary;
}

export function MvpInfoBar({ planId, mvp }: MvpInfoBarProps) {
  const [expanded, setExpanded] = useState(false);
  const [detail, setDetail] = useState<MvpDetail | null>(null);
  const [loading, setLoading] = useState(false);

  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisOutput, setAnalysisOutput] = useState<string | null>(null);
  const [analysisAt, setAnalysisAt] = useState<string | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const [showUnits, setShowUnits] = useState(false);
  const [showMapping, setShowMapping] = useState(false);
  const [showDiagrams, setShowDiagrams] = useState(false);

  // Reset when MVP changes
  useEffect(() => {
    setDetail(null);
    setExpanded(false);
    setAnalysisOutput(null);
    setAnalysisAt(null);
    setShowFiles(false);
    setShowUnits(false);
    setShowAnalysis(false);
    setShowMapping(false);
    setShowDiagrams(false);
  }, [mvp.mvp_id]);

  // Load pre-existing analysis from MVP data
  useEffect(() => {
    if (mvp.analysis_output?.output) {
      setAnalysisOutput(mvp.analysis_output.output);
      setAnalysisAt(mvp.analysis_at ?? null);
    }
  }, [mvp.analysis_output, mvp.analysis_at]);

  const handleToggle = async () => {
    const nextExpanded = !expanded;
    setExpanded(nextExpanded);

    if (nextExpanded && !detail) {
      setLoading(true);
      try {
        const d = await api.getMvpDetail(planId, mvp.mvp_id);
        setDetail(d);
      } catch {
        // Silently fail — user still sees summary
      } finally {
        setLoading(false);
      }
    }
  };

  const handleAnalyze = useCallback(async () => {
    setIsAnalyzing(true);
    try {
      const result = await api.analyzeMvp(planId, mvp.mvp_id);
      setAnalysisOutput(result.output);
      setAnalysisAt(result.analysis_at);
      setShowAnalysis(true);
    } catch {
      // Error is non-critical; user can retry
    } finally {
      setIsAnalyzing(false);
    }
  }, [planId, mvp.mvp_id]);

  const metrics = mvp.metrics;
  const hasMapping = (detail?.architecture_mapping?.length ?? 0) > 0;

  return (
    <div className="border-b border-void-surface">
      {/* Summary row — always visible */}
      <button
        onClick={handleToggle}
        className="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-void-light/30 transition-colors"
      >
        {expanded
          ? <ChevronDown className="h-3 w-3 shrink-0 text-text-dim" />
          : <ChevronRight className="h-3 w-3 shrink-0 text-text-dim" />
        }

        <span className="text-xs font-medium text-text">{mvp.name}</span>

        <div className="ml-auto flex items-center gap-3 text-[10px] text-text-muted">
          <span><span className="opacity-70">Files:</span> {mvp.file_ids.length}</span>
          <span><span className="opacity-70">Units:</span> {mvp.unit_ids.length}</span>
          {metrics.cohesion != null && (
            <span className={metrics.cohesion > 0.5 ? 'text-success' : 'text-warning'}>
              <span className="opacity-60">Coh:</span> {metrics.cohesion.toFixed(2)}
            </span>
          )}
          {metrics.readiness != null && (
            <span className={Math.round(metrics.readiness * 100) >= 70 ? 'text-success' : 'text-warning'}>
              {Math.round(metrics.readiness * 100)}% ready
            </span>
          )}
          <span className="text-text-dim/40">
            {expanded ? 'Hide detail' : 'Show detail'}
          </span>
        </div>
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="border-t border-void-surface/50 bg-void-light/50 px-4 py-3 text-xs">
          {loading && !detail && (
            <div className="flex items-center gap-2 py-2 text-text-muted">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading files and units...
            </div>
          )}

          {detail && (
            <div className="space-y-4">
              {/* Source Files (collapsible) */}
              <div>
                <button
                  onClick={() => setShowFiles(!showFiles)}
                  className="flex items-center gap-1.5 font-medium text-text hover:text-glow transition-colors"
                >
                  <FileCode className="h-3.5 w-3.5 text-glow" />
                  Source Files
                  <span className="font-normal text-text-muted">({detail.files.length})</span>
                  {showFiles
                    ? <ChevronDown className="h-3 w-3 text-text-dim" />
                    : <ChevronRight className="h-3 w-3 text-text-dim" />
                  }
                </button>
                {showFiles && (
                  <div className="mt-1.5 max-h-64 overflow-y-auto rounded border border-void-surface/50 bg-void-light/30 p-2">
                    {detail.files.length > 0 ? (
                      <div className="flex flex-col gap-1.5">
                        {detail.files.map((f) => (
                          <div key={f.file_id} className="flex items-center gap-2">
                            <LanguageDot language={f.language} />
                            <span className="font-mono text-[11px] text-text truncate">{f.file_path}</span>
                            <span className="ml-auto shrink-0 text-[10px] text-text-muted">{f.line_count} lines</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-text-muted">No files resolved</p>
                    )}
                  </div>
                )}
              </div>

              {/* Code Units (collapsible) */}
              <div>
                <button
                  onClick={() => setShowUnits(!showUnits)}
                  className="flex items-center gap-1.5 font-medium text-text hover:text-glow transition-colors"
                >
                  <Box className="h-3.5 w-3.5 text-glow" />
                  Code Units
                  <span className="font-normal text-text-muted">({detail.units.length})</span>
                  {showUnits
                    ? <ChevronDown className="h-3 w-3 text-text-dim" />
                    : <ChevronRight className="h-3 w-3 text-text-dim" />
                  }
                </button>
                {showUnits && (
                  <div className="mt-1.5 max-h-64 overflow-y-auto rounded border border-void-surface/50 bg-void-light/30 p-2">
                    {detail.units.length > 0 ? (
                      <div className="flex flex-col gap-1.5">
                        {detail.units.map((u) => (
                          <div key={u.unit_id} className="flex items-center gap-2">
                            <UnitTypeTag type={u.unit_type} />
                            <span className="font-mono text-[11px] text-text truncate" title={u.qualified_name}>
                              {u.name}
                            </span>
                            {u.signature && (
                              <span className="font-mono text-[10px] text-text-muted truncate max-w-[200px]" title={u.signature}>
                                {u.signature}
                              </span>
                            )}
                            <span className="ml-auto shrink-0 text-[10px] text-text-muted">
                              L{u.start_line}–{u.end_line}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-text-muted">No units resolved</p>
                    )}
                  </div>
                )}
              </div>

              {/* Architecture Mapping table (from plan-level Architecture phase) */}
              {hasMapping && (
                <div>
                  <button
                    onClick={() => setShowMapping(!showMapping)}
                    className="flex items-center gap-1.5 font-medium text-text hover:text-glow transition-colors"
                  >
                    <Table2 className="h-3.5 w-3.5 text-glow" />
                    Architecture Mapping
                    <span className="font-normal text-text-muted">({detail.architecture_mapping!.length})</span>
                    {showMapping
                      ? <ChevronDown className="h-3 w-3 text-text-dim" />
                      : <ChevronRight className="h-3 w-3 text-text-dim" />
                    }
                  </button>
                  {showMapping && (
                    <div className="mt-1.5 overflow-x-auto rounded border border-void-surface/50">
                      <table className="w-full text-[11px]">
                        <thead>
                          <tr className="border-b border-void-surface/50 bg-void-light/30">
                            <th className="px-2 py-1.5 text-left font-medium text-text-muted">Source</th>
                            <th className="px-2 py-1.5 text-left font-medium text-text-muted">Target</th>
                            <th className="px-2 py-1.5 text-left font-medium text-text-muted">Changes</th>
                          </tr>
                        </thead>
                        <tbody>
                          {detail.architecture_mapping!.map((m, i) => (
                            <tr key={i} className="border-b border-void-surface/30 last:border-0">
                              <td className="px-2 py-1">
                                <span className="font-mono text-text">{m.source_path}</span>
                                {m.source_class && (
                                  <span className="ml-1 text-text-muted">({m.source_class})</span>
                                )}
                              </td>
                              <td className="px-2 py-1">
                                <span className="font-mono text-glow">{m.target_path}</span>
                                {m.target_class && (
                                  <span className="ml-1 text-text-muted">({m.target_class})</span>
                                )}
                              </td>
                              <td className="px-2 py-1 text-text-muted">{m.changes}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* UML Diagrams section */}
              <div>
                <button
                  onClick={() => setShowDiagrams(!showDiagrams)}
                  className="flex items-center gap-1.5 font-medium text-text hover:text-glow transition-colors"
                >
                  <LayoutGrid className="h-3.5 w-3.5 text-glow" />
                  UML Diagrams
                  {showDiagrams
                    ? <ChevronDown className="h-3 w-3 text-text-dim" />
                    : <ChevronRight className="h-3 w-3 text-text-dim" />
                  }
                </button>
                {showDiagrams && (
                  <div className="mt-2">
                    <MvpDiagramPanel planId={planId} mvpId={mvp.mvp_id} />
                  </div>
                )}
              </div>

              {/* Deep Analyze section */}
              <div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="flex items-center gap-1.5 rounded bg-glow/10 px-3 py-1.5 text-[11px] font-medium text-glow hover:bg-glow/20 disabled:opacity-50 transition-colors"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-3.5 w-3.5" />
                        {analysisAt ? 'Re-analyze' : 'Deep Analyze'}
                      </>
                    )}
                  </button>
                  {analysisAt && (
                    <span className="text-[10px] text-text-dim">
                      Analyzed: {new Date(analysisAt).toLocaleString()}
                    </span>
                  )}
                </div>

                {/* Analysis output (collapsible) */}
                {analysisOutput && (
                  <div className="mt-2">
                    <button
                      onClick={() => setShowAnalysis(!showAnalysis)}
                      className="flex items-center gap-1.5 text-[11px] font-medium text-text hover:text-glow transition-colors"
                    >
                      {showAnalysis
                        ? <ChevronDown className="h-3 w-3 text-text-dim" />
                        : <ChevronRight className="h-3 w-3 text-text-dim" />
                      }
                      Analysis Output
                    </button>
                    {showAnalysis && (
                      <div className="mt-1.5 max-h-80 overflow-y-auto rounded border border-void-surface/50 bg-void-light/30 p-3">
                        <pre className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-text-muted">
                          {analysisOutput}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {!loading && !detail && (
            <p className="text-text-muted">Could not load details.</p>
          )}
        </div>
      )}
    </div>
  );
}


// ── Helpers ──────────────────────────────────────────────────────────

function LanguageDot({ language }: { language: string }) {
  const colors: Record<string, string> = {
    python: 'bg-blue-400',
    javascript: 'bg-yellow-400',
    typescript: 'bg-blue-300',
    java: 'bg-orange-400',
    csharp: 'bg-purple-400',
    sql: 'bg-emerald-400',
  };
  const cls = colors[language.toLowerCase()] ?? 'bg-text-dim';
  return <span className={`h-2 w-2 shrink-0 rounded-full ${cls}`} title={language} />;
}

function UnitTypeTag({ type }: { type: string }) {
  const colors: Record<string, string> = {
    class: 'bg-purple-500/15 text-purple-400',
    function: 'bg-blue-500/15 text-blue-400',
    method: 'bg-cyan-500/15 text-cyan-400',
    interface: 'bg-green-500/15 text-green-400',
    module: 'bg-orange-500/15 text-orange-400',
  };
  const cls = colors[type.toLowerCase()] ?? 'bg-void-surface text-text-dim';
  return (
    <span className={`shrink-0 rounded px-1 py-0 text-[9px] font-medium capitalize ${cls}`}>
      {type.substring(0, 3)}
    </span>
  );
}
