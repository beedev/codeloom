/**
 * MvpDiscoveryPanel — shown after Phase 1 (Discovery) completes.
 *
 * Displays auto-detected Functional MVPs with:
 * - Priority-ordered card list with metrics (cohesion, coupling, readiness)
 * - Per-MVP actions: rename, update priority
 * - Merge/split controls
 * - Expandable detail: resolved files, code units, SP references
 * - "Finalize & Create Phases" button to proceed
 */

import { useState, useCallback } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Merge,
  ArrowUpDown,
  Check,
  Loader2,
  AlertCircle,
  FileCode,
  Box,
  Sparkles,
  LayoutGrid,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { FunctionalMvpSummary, SpReference, MvpDetail } from '../../types/index.ts';
import * as api from '../../services/api.ts';
import { MvpDiagramPanel } from './MvpDiagramPanel.tsx';

interface MvpDiscoveryPanelProps {
  planId: string;
  mvps: FunctionalMvpSummary[];
  onMvpsChanged: () => void;  // Callback to refresh plan data
  onCreatePhases: () => void; // Callback when user clicks "Create Phases"
  isCreatingPhases: boolean;
}

export function MvpDiscoveryPanel({
  planId,
  mvps,
  onMvpsChanged,
  onCreatePhases,
  isCreatingPhases,
}: MvpDiscoveryPanelProps) {
  const [expandedMvp, setExpandedMvp] = useState<number | null>(null);
  const [editingName, setEditingName] = useState<number | null>(null);
  const [nameValue, setNameValue] = useState('');
  const [mergeSelection, setMergeSelection] = useState<Set<number>>(new Set());
  const [isMerging, setIsMerging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAnalysis, setShowAnalysis] = useState<Record<number, boolean>>({});
  const [showDiagrams, setShowDiagrams] = useState<Record<number, boolean>>({});
  const [isAnalyzingAll, setIsAnalyzingAll] = useState(false);
  const [analyzeProgress, setAnalyzeProgress] = useState<string | null>(null);
  const [isAnalyzingIntegration, setIsAnalyzingIntegration] = useState(false);

  // Cache fetched details so we don't re-fetch on re-expand
  const [detailCache, setDetailCache] = useState<Record<number, MvpDetail>>({});
  const [loadingDetail, setLoadingDetail] = useState<number | null>(null);

  const sortedMvps = [...mvps].sort((a, b) => a.priority - b.priority);

  // ── Toggle expand with lazy fetch ──

  const toggleExpand = useCallback(async (mvpId: number) => {
    if (expandedMvp === mvpId) {
      setExpandedMvp(null);
      return;
    }
    setExpandedMvp(mvpId);

    // Fetch detail if not cached
    if (!detailCache[mvpId]) {
      setLoadingDetail(mvpId);
      try {
        const detail = await api.getMvpDetail(planId, mvpId);
        setDetailCache(prev => ({ ...prev, [mvpId]: detail }));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load MVP details');
      } finally {
        setLoadingDetail(null);
      }
    }
  }, [expandedMvp, detailCache, planId]);

  // ── Rename MVP ──

  const startRename = (mvp: FunctionalMvpSummary) => {
    setEditingName(mvp.mvp_id);
    setNameValue(mvp.name);
  };

  const saveRename = useCallback(async (mvpId: number) => {
    if (!nameValue.trim()) return;
    setError(null);
    try {
      await api.updateMvp(planId, mvpId, { name: nameValue.trim() });
      setEditingName(null);
      onMvpsChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rename MVP');
    }
  }, [planId, nameValue, onMvpsChanged]);

  // ── Merge MVPs ──

  const toggleMergeSelect = (mvpId: number) => {
    setMergeSelection(prev => {
      const next = new Set(prev);
      if (next.has(mvpId)) {
        next.delete(mvpId);
      } else {
        next.add(mvpId);
      }
      return next;
    });
  };

  const handleMerge = useCallback(async () => {
    if (mergeSelection.size < 2) return;
    setIsMerging(true);
    setError(null);
    try {
      await api.mergeMvps(planId, Array.from(mergeSelection));
      setMergeSelection(new Set());
      setDetailCache({}); // Invalidate cache after merge
      onMvpsChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to merge MVPs');
    } finally {
      setIsMerging(false);
    }
  }, [planId, mergeSelection, onMvpsChanged]);

  // ── Priority update ──

  const updatePriority = useCallback(async (mvpId: number, priority: number) => {
    setError(null);
    try {
      await api.updateMvp(planId, mvpId, { priority });
      onMvpsChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update priority');
    }
  }, [planId, onMvpsChanged]);

  // ── Deep Analyze All MVPs ──

  const handleAnalyzeAll = useCallback(async () => {
    setIsAnalyzingAll(true);
    setAnalyzeProgress('Starting deep analysis...');
    setError(null);
    try {
      const result = await api.analyzeAllMvps(planId);
      setAnalyzeProgress(null);
      setDetailCache({}); // Invalidate cache to pick up new analysis_output
      onMvpsChanged();
      if (result.analyzed === 0 && result.total === 0) {
        setAnalyzeProgress('All MVPs already analyzed');
        setTimeout(() => setAnalyzeProgress(null), 3000);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze MVPs');
      setAnalyzeProgress(null);
    } finally {
      setIsAnalyzingAll(false);
    }
  }, [planId, onMvpsChanged]);

  const unanalyzedCount = mvps.filter(m => !m.analysis_output?.output).length;

  // ── Integration MVP (MVP 99) ──

  const integrationMvp = mvps.find(m => m.metrics?.integration_mvp);
  const otherMvpsAnalyzed = mvps.filter(m => !m.metrics?.integration_mvp && m.analysis_output?.output).length;
  const otherMvpsTotal = mvps.filter(m => !m.metrics?.integration_mvp).length;

  const handleAnalyzeIntegration = useCallback(async () => {
    if (!integrationMvp) return;
    setIsAnalyzingIntegration(true);
    setError(null);
    try {
      await api.analyzeMvp(planId, integrationMvp.mvp_id);
      setDetailCache({});
      onMvpsChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze integration MVP');
    } finally {
      setIsAnalyzingIntegration(false);
    }
  }, [planId, integrationMvp, onMvpsChanged]);

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-text">
            Functional MVPs
            <span className="ml-2 text-xs font-normal text-text-muted">
              {mvps.length} cluster{mvps.length !== 1 ? 's' : ''} detected
            </span>
          </h2>
          <p className="mt-0.5 text-xs text-text-dim">
            Review, rename, merge, or reorder before creating per-MVP phases.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {mergeSelection.size >= 2 && (
            <button
              onClick={handleMerge}
              disabled={isMerging}
              className="flex items-center gap-1.5 rounded-lg border border-glow/50 bg-glow/10 px-3 py-1.5 text-xs font-medium text-glow hover:bg-glow/20 disabled:opacity-50"
            >
              {isMerging ? <Loader2 className="h-3 w-3 animate-spin" /> : <Merge className="h-3 w-3" />}
              Merge {mergeSelection.size} MVPs
            </button>
          )}
          <button
            onClick={handleAnalyzeAll}
            disabled={isAnalyzingAll || unanalyzedCount === 0}
            className="flex items-center gap-1.5 rounded-lg border border-amber-500/50 bg-amber-500/10 px-3 py-1.5 text-xs font-medium text-amber-400 hover:bg-amber-500/20 disabled:opacity-50"
            title={unanalyzedCount === 0 ? 'All MVPs already analyzed' : `Analyze ${unanalyzedCount} MVP${unanalyzedCount !== 1 ? 's' : ''}`}
          >
            {isAnalyzingAll ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}
            {isAnalyzingAll ? (analyzeProgress || 'Analyzing...') : `Deep Analyze${unanalyzedCount > 0 ? ` (${unanalyzedCount})` : ''}`}
          </button>
          {integrationMvp && (
            <button
              onClick={handleAnalyzeIntegration}
              disabled={isAnalyzingIntegration || otherMvpsAnalyzed < otherMvpsTotal}
              className="flex items-center gap-1.5 rounded-lg border border-purple-500/50 bg-purple-500/10 px-3 py-1.5 text-xs font-medium text-purple-400 hover:bg-purple-500/20 disabled:opacity-50"
              title={
                otherMvpsAnalyzed < otherMvpsTotal
                  ? `Analyze all other MVPs first (${otherMvpsAnalyzed}/${otherMvpsTotal} done)`
                  : 'Run integration analysis — aggregates all MVP requirements'
              }
            >
              {isAnalyzingIntegration ? <Loader2 className="h-3 w-3 animate-spin" /> : <LayoutGrid className="h-3 w-3" />}
              {isAnalyzingIntegration ? 'Integrating...' : 'Analyze Integration'}
            </button>
          )}
          <button
            onClick={onCreatePhases}
            disabled={isCreatingPhases || mvps.length === 0}
            className="flex items-center gap-1.5 rounded-lg bg-glow px-4 py-1.5 text-xs font-medium text-white hover:bg-glow/90 disabled:opacity-50"
          >
            {isCreatingPhases ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
            Finalize & Create Phases
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-xs text-danger">
          <AlertCircle className="h-3 w-3 shrink-0" />
          {error}
        </div>
      )}

      {/* MVP Cards */}
      <div className="flex flex-col gap-2">
        {sortedMvps.map((mvp) => (
          <MvpCard
            key={mvp.mvp_id}
            planId={planId}
            mvp={mvp}
            detail={detailCache[mvp.mvp_id] ?? null}
            isExpanded={expandedMvp === mvp.mvp_id}
            isLoadingDetail={loadingDetail === mvp.mvp_id}
            isEditing={editingName === mvp.mvp_id}
            editValue={nameValue}
            isMergeSelected={mergeSelection.has(mvp.mvp_id)}
            isAnalysisExpanded={!!showAnalysis[mvp.mvp_id]}
            isDiagramsExpanded={!!showDiagrams[mvp.mvp_id]}
            onToggleExpand={() => toggleExpand(mvp.mvp_id)}
            onStartRename={() => startRename(mvp)}
            onEditChange={setNameValue}
            onSaveRename={() => saveRename(mvp.mvp_id)}
            onCancelRename={() => setEditingName(null)}
            onToggleMerge={() => toggleMergeSelect(mvp.mvp_id)}
            onToggleAnalysis={() => setShowAnalysis(prev => ({ ...prev, [mvp.mvp_id]: !prev[mvp.mvp_id] }))}
            onToggleDiagrams={() => setShowDiagrams(prev => ({ ...prev, [mvp.mvp_id]: !prev[mvp.mvp_id] }))}
            onPriorityUp={() => updatePriority(mvp.mvp_id, Math.max(1, mvp.priority - 1))}
            onPriorityDown={() => updatePriority(mvp.mvp_id, mvp.priority + 1)}
          />
        ))}
      </div>

      {mvps.length === 0 && (
        <div className="py-8 text-center text-xs text-text-dim">
          No MVPs discovered. Run Discovery first.
        </div>
      )}
    </div>
  );
}


// ── MVP Card ────────────────────────────────────────────────────────────

interface MvpCardProps {
  planId: string;
  mvp: FunctionalMvpSummary;
  detail: MvpDetail | null;
  isExpanded: boolean;
  isLoadingDetail: boolean;
  isEditing: boolean;
  editValue: string;
  isMergeSelected: boolean;
  isAnalysisExpanded: boolean;
  isDiagramsExpanded: boolean;
  onToggleExpand: () => void;
  onStartRename: () => void;
  onEditChange: (val: string) => void;
  onSaveRename: () => void;
  onCancelRename: () => void;
  onToggleMerge: () => void;
  onToggleAnalysis: () => void;
  onToggleDiagrams: () => void;
  onPriorityUp: () => void;
  onPriorityDown: () => void;
}

function MvpCard({
  planId,
  mvp,
  detail,
  isExpanded,
  isLoadingDetail,
  isEditing,
  editValue,
  isMergeSelected,
  isAnalysisExpanded,
  isDiagramsExpanded,
  onToggleExpand,
  onStartRename,
  onEditChange,
  onSaveRename,
  onCancelRename,
  onToggleMerge,
  onToggleAnalysis,
  onToggleDiagrams,
  onPriorityUp,
  onPriorityDown,
}: MvpCardProps) {
  const metrics = mvp.metrics;
  const spCount = mvp.sp_references?.length ?? 0;
  const [showFiles, setShowFiles] = useState(false);
  const [showUnits, setShowUnits] = useState(false);

  return (
    <div
      className={`rounded-xl border transition-colors ${
        isMergeSelected
          ? 'border-glow/50 bg-glow/5'
          : 'border-void-surface/50 bg-void-light/30'
      }`}
    >
      {/* Header row */}
      <div className="flex items-center gap-3 px-4 py-3">
        {/* Merge checkbox */}
        <input
          type="checkbox"
          checked={isMergeSelected}
          onChange={onToggleMerge}
          className="h-3.5 w-3.5 rounded border-void-surface accent-glow"
        />

        {/* Expand toggle */}
        <button onClick={onToggleExpand} className="text-text-dim hover:text-text">
          {isExpanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        </button>

        {/* Priority badge */}
        <div className="flex items-center gap-0.5">
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-glow/15 text-[10px] font-bold text-glow">
            {mvp.priority}
          </span>
          <div className="flex flex-col">
            <button onClick={onPriorityUp} className="text-text-dim hover:text-text leading-none">
              <ArrowUpDown className="h-2.5 w-2.5 rotate-180" />
            </button>
            <button onClick={onPriorityDown} className="text-text-dim hover:text-text leading-none">
              <ArrowUpDown className="h-2.5 w-2.5" />
            </button>
          </div>
        </div>

        {/* Name (editable) */}
        {isEditing ? (
          <div className="flex items-center gap-1.5">
            <input
              value={editValue}
              onChange={(e) => onEditChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') onSaveRename();
                if (e.key === 'Escape') onCancelRename();
              }}
              className="rounded border border-void-surface bg-void px-2 py-0.5 text-xs text-text outline-none focus:border-glow"
              autoFocus
            />
            <button onClick={onSaveRename} className="text-xs text-glow hover:underline">Save</button>
            <button onClick={onCancelRename} className="text-xs text-text-dim hover:underline">Cancel</button>
          </div>
        ) : (
          <button
            onClick={onStartRename}
            className="text-xs font-medium text-text hover:text-glow"
            title="Click to rename"
          >
            {mvp.name}
          </button>
        )}

        {/* Metrics badges */}
        <div className="ml-auto flex items-center gap-3 text-[10px] text-text-dim">
          <MetricBadge label="Units" value={mvp.unit_ids.length} />
          <MetricBadge label="Files" value={mvp.file_ids.length} />
          {metrics.cohesion != null && (
            <MetricBadge label="Coh" value={metrics.cohesion.toFixed(2)} color={metrics.cohesion > 0.5 ? 'success' : 'warning'} />
          )}
          {metrics.coupling != null && (
            <MetricBadge label="Coup" value={metrics.coupling.toFixed(2)} color={metrics.coupling < 0.3 ? 'success' : 'warning'} />
          )}
          {metrics.readiness != null && (
            <ReadinessBadge value={metrics.readiness} />
          )}
          {spCount > 0 && (
            <span className="flex items-center gap-1 text-warning">
              <span className="material-symbols-outlined text-[12px]">database</span>
              {spCount} SP{spCount !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>

      {/* Expanded detail */}
      {isExpanded && (
        <div className="border-t border-void-surface/50 px-4 py-3 text-xs">
          {/* Loading state */}
          {isLoadingDetail && !detail && (
            <div className="flex items-center gap-2 py-4 text-text-dim">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading details...
            </div>
          )}

          {/* Description (rendered as markdown for Foundation MVP etc.) */}
          {mvp.description && (
            <div className="mb-3 prose prose-sm prose-invert max-w-none text-text-muted
                            prose-headings:text-text-dim prose-headings:text-xs prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-1
                            prose-li:text-text-muted prose-li:text-xs prose-li:my-0
                            prose-ul:my-1 prose-p:my-1 prose-p:text-xs">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{mvp.description}</ReactMarkdown>
            </div>
          )}

          {/* Resolved detail sections */}
          {detail && (
            <div className="flex flex-col gap-4">
              {/* Files section (collapsible) */}
              {detail.files.length > 0 && (
                <div>
                  <button
                    onClick={() => setShowFiles(!showFiles)}
                    className="mb-1.5 flex items-center gap-1.5 font-medium text-text-muted hover:text-text transition-colors"
                  >
                    <FileCode className="h-3.5 w-3.5 text-glow/70" />
                    Source Files
                    <span className="font-normal text-text-dim">({detail.files.length})</span>
                    {showFiles
                      ? <ChevronDown className="h-3 w-3 text-text-dim" />
                      : <ChevronRight className="h-3 w-3 text-text-dim" />}
                  </button>
                  {showFiles && (
                    <div className="rounded-lg border border-void-surface/30 bg-void/40">
                      <table className="w-full text-[11px]">
                        <thead>
                          <tr className="border-b border-void-surface/30 text-left text-text-dim">
                            <th className="px-3 py-1.5 font-medium">Path</th>
                            <th className="px-3 py-1.5 font-medium w-20">Language</th>
                            <th className="px-3 py-1.5 font-medium w-16 text-right">Lines</th>
                          </tr>
                        </thead>
                        <tbody>
                          {detail.files.map((f) => (
                            <tr key={f.file_id} className="border-b border-void-surface/20 last:border-0">
                              <td className="px-3 py-1.5 font-mono text-text-muted">{f.file_path}</td>
                              <td className="px-3 py-1.5">
                                <LanguageBadge language={f.language} />
                              </td>
                              <td className="px-3 py-1.5 text-right text-text-dim">{f.line_count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Units section (collapsible) */}
              {detail.units.length > 0 && (
                <div>
                  <button
                    onClick={() => setShowUnits(!showUnits)}
                    className="mb-1.5 flex items-center gap-1.5 font-medium text-text-muted hover:text-text transition-colors"
                  >
                    <Box className="h-3.5 w-3.5 text-glow/70" />
                    Code Units
                    <span className="font-normal text-text-dim">({detail.units.length})</span>
                    {showUnits
                      ? <ChevronDown className="h-3 w-3 text-text-dim" />
                      : <ChevronRight className="h-3 w-3 text-text-dim" />}
                  </button>
                  {showUnits && (
                    <div className="rounded-lg border border-void-surface/30 bg-void/40">
                      <table className="w-full text-[11px]">
                        <thead>
                          <tr className="border-b border-void-surface/30 text-left text-text-dim">
                            <th className="px-3 py-1.5 font-medium">Name</th>
                            <th className="px-3 py-1.5 font-medium w-20">Type</th>
                            <th className="px-3 py-1.5 font-medium">File</th>
                            <th className="px-3 py-1.5 font-medium w-20 text-right">Lines</th>
                          </tr>
                        </thead>
                        <tbody>
                          {detail.units.map((u) => (
                            <tr key={u.unit_id} className="border-b border-void-surface/20 last:border-0 group">
                              <td className="px-3 py-1.5">
                                <div className="flex flex-col gap-0.5">
                                  <span className="font-mono font-medium text-text">{u.name}</span>
                                  {u.signature && (
                                    <span className="font-mono text-[10px] text-text-dim truncate max-w-[400px]" title={u.signature}>
                                      {u.signature}
                                    </span>
                                  )}
                                </div>
                              </td>
                              <td className="px-3 py-1.5">
                                <UnitTypeBadge type={u.unit_type} />
                              </td>
                              <td className="px-3 py-1.5 font-mono text-text-dim truncate max-w-[250px]" title={u.file_path}>
                                {u.file_path.split('/').pop()}
                              </td>
                              <td className="px-3 py-1.5 text-right text-text-dim">
                                {u.start_line}&ndash;{u.end_line}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* SP References */}
          {spCount > 0 && (
            <div className="mt-3">
              <h4 className="mb-1 font-medium text-text-muted">Stored Procedure References</h4>
              <div className="flex flex-col gap-1">
                {mvp.sp_references.map((sp: SpReference, i: number) => (
                  <div key={i} className="flex items-center gap-2 text-text-dim">
                    <span className="material-symbols-outlined text-[12px] text-warning">database</span>
                    <span className="font-mono text-text-muted">{sp.sp_name}</span>
                    <span className="text-text-dim">({sp.call_sites.length} call site{sp.call_sites.length !== 1 ? 's' : ''})</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Analysis Output (auto-generated during discovery) */}
          {mvp.analysis_output?.output && (
            <div className="mt-3">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleAnalysis();
                }}
                className="flex items-center gap-1.5 text-[11px] font-medium text-text hover:text-glow transition-colors"
              >
                {isAnalysisExpanded
                  ? <ChevronDown className="h-3 w-3 text-text-dim" />
                  : <ChevronRight className="h-3 w-3 text-text-dim" />
                }
                <Sparkles className="h-3 w-3 text-glow" />
                Functional Requirements Register
              </button>
              {mvp.analysis_at && (
                <span className="ml-6 text-[10px] text-text-dim">
                  Analyzed: {new Date(mvp.analysis_at).toLocaleString()}
                </span>
              )}
              {isAnalysisExpanded && (
                <div className="mt-1.5 max-h-80 overflow-y-auto rounded border border-void-surface/50 bg-void-light/30 p-3">
                  <pre className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-text-muted">
                    {mvp.analysis_output.output}
                  </pre>
                </div>
              )}
            </div>
          )}

          {/* UML Diagrams */}
          <div className="mt-3">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleDiagrams();
              }}
              className="flex items-center gap-1.5 text-[11px] font-medium text-text hover:text-glow transition-colors"
            >
              {isDiagramsExpanded
                ? <ChevronDown className="h-3 w-3 text-text-dim" />
                : <ChevronRight className="h-3 w-3 text-text-dim" />
              }
              <LayoutGrid className="h-3 w-3 text-glow" />
              UML Diagrams
            </button>
            {isDiagramsExpanded && (
              <div className="mt-2">
                <MvpDiagramPanel planId={planId} mvpId={mvp.mvp_id} />
              </div>
            )}
          </div>

          {/* Depends on */}
          {mvp.depends_on_mvp_ids.length > 0 && (
            <p className="mt-2 text-text-dim">
              Depends on MVP IDs: {mvp.depends_on_mvp_ids.join(', ')}
            </p>
          )}
        </div>
      )}
    </div>
  );
}


// ── Small components ────────────────────────────────────────────────────

function MetricBadge({ label, value, color }: { label: string; value: string | number; color?: string }) {
  const colorClass = color === 'success' ? 'text-success' : color === 'warning' ? 'text-warning' : 'text-text-dim';
  return (
    <span className={colorClass}>
      <span className="opacity-60">{label}:</span> {value}
    </span>
  );
}

function ReadinessBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? 'text-success' : pct >= 40 ? 'text-warning' : 'text-danger';
  return (
    <span className={`font-medium ${color}`}>
      {pct}% ready
    </span>
  );
}

function LanguageBadge({ language }: { language: string }) {
  const colors: Record<string, string> = {
    python: 'bg-blue-500/15 text-blue-400',
    javascript: 'bg-yellow-500/15 text-yellow-400',
    typescript: 'bg-blue-400/15 text-blue-300',
    java: 'bg-orange-500/15 text-orange-400',
    csharp: 'bg-purple-500/15 text-purple-400',
    sql: 'bg-emerald-500/15 text-emerald-400',
  };
  const cls = colors[language.toLowerCase()] ?? 'bg-void-surface text-text-dim';
  return (
    <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium ${cls}`}>
      {language}
    </span>
  );
}

function UnitTypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    class: 'bg-purple-500/15 text-purple-400',
    function: 'bg-blue-500/15 text-blue-400',
    method: 'bg-cyan-500/15 text-cyan-400',
    interface: 'bg-green-500/15 text-green-400',
    module: 'bg-orange-500/15 text-orange-400',
    enum: 'bg-yellow-500/15 text-yellow-400',
    struct: 'bg-rose-500/15 text-rose-400',
  };
  const cls = colors[type.toLowerCase()] ?? 'bg-void-surface text-text-dim';
  return (
    <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium capitalize ${cls}`}>
      {type}
    </span>
  );
}
