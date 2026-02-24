/**
 * BatchExecutionPanel — Batch migration execution & monitoring UI.
 *
 * Three modes:
 *   1. Launch: Configure and start a batch run
 *   2. Monitor: Live-polling progress view during execution
 *   3. Results: Final summary with retry option
 *
 * Appears in MigrationWizard after Phase 2 (discovery) approval
 * and MVP phases creation, as an alternative to per-MVP manual execution.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Loader2,
  Play,
  RotateCcw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  ChevronDown,
  ChevronRight,
  Download,
  Eye,
} from 'lucide-react';
import * as api from '../../services/api.ts';
import type { BatchStatus, BatchMvpResult, BatchExecuteParams } from '../../services/api.ts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BatchExecutionPanelProps {
  planId: string;
  totalMvps: number;
  onMvpClick?: (mvpId: number) => void;
  onBatchComplete?: () => void;
  /** Controlled mode: null = launch, string = monitor that batch, undefined = uncontrolled */
  initialBatchId?: string | null;
  onRunsLoaded?: (runs: BatchStatus[]) => void;
  onBatchIdChange?: (batchId: string | null) => void;
}

type ApprovalPolicy = 'manual' | 'auto' | 'auto_non_blocking';

// ---------------------------------------------------------------------------
// Status badge colors
// ---------------------------------------------------------------------------

const STATUS_CONFIG: Record<string, { label: string; color: string; bg: string }> = {
  pending:      { label: 'Pending',      color: 'text-text-dim',    bg: 'bg-void-surface/50' },
  processing:   { label: 'Processing',   color: 'text-glow',        bg: 'bg-glow/10' },
  executed:     { label: 'Executed',      color: 'text-nebula',      bg: 'bg-nebula/10' },
  approved:     { label: 'Approved',      color: 'text-success',     bg: 'bg-success/10' },
  needs_review: { label: 'Needs Review', color: 'text-warning',     bg: 'bg-warning/10' },
  failed:       { label: 'Failed',        color: 'text-danger',      bg: 'bg-danger/10' },
  skipped:      { label: 'Skipped',       color: 'text-text-dim',    bg: 'bg-void-surface/30' },
};

function StatusBadge({ status }: { status: string }) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending;
  return (
    <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-medium ${config.color} ${config.bg}`}>
      {config.label}
    </span>
  );
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'approved':
      return <CheckCircle2 className="h-4 w-4 text-success" />;
    case 'failed':
      return <XCircle className="h-4 w-4 text-danger" />;
    case 'needs_review':
      return <AlertTriangle className="h-4 w-4 text-warning" />;
    case 'processing':
      return <Loader2 className="h-4 w-4 animate-spin text-glow" />;
    case 'skipped':
      return <Clock className="h-4 w-4 text-text-dim" />;
    default:
      return <div className="h-3 w-3 rounded-full border border-void-surface bg-void-light" />;
  }
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export function BatchExecutionPanel({
  planId,
  totalMvps,
  onMvpClick,
  onBatchComplete,
  initialBatchId,
  onRunsLoaded,
  onBatchIdChange,
}: BatchExecutionPanelProps) {
  // ── State ──
  const [mode, setMode] = useState<'launch' | 'monitor' | 'history'>('launch');
  const [approvalPolicy, setApprovalPolicy] = useState<ApprovalPolicy>('auto');
  const [runAll, setRunAll] = useState(true);
  const [isLaunching, setIsLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Active batch tracking
  const [activeBatchId, setActiveBatchId] = useState<string | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatus | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Expanded MVP rows
  const [expandedMvps, setExpandedMvps] = useState<Set<number>>(new Set());

  // History
  const [batchHistory, setBatchHistory] = useState<BatchStatus[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);

  // Controlled mode: parent drives batch selection via initialBatchId
  const isControlled = initialBatchId !== undefined;
  const effectiveMode = isControlled
    ? (initialBatchId === null ? 'launch' as const : 'monitor' as const)
    : mode;

  // ── Cleanup polling on unmount ──
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // ── Load batch history ──
  const loadHistory = useCallback(async () => {
    try {
      const history = await api.listBatchExecutions(planId);
      setBatchHistory(history);
      setHistoryLoaded(true);
      onRunsLoaded?.(history);

      // Auto-switch to running batch (only in uncontrolled mode)
      if (initialBatchId === undefined) {
        const running = history.find(b => b.status === 'running');
        if (running) {
          setActiveBatchId(running.batch_id);
          setBatchStatus(running);
          setMode('monitor');
          startPolling(running.batch_id);
        }
      }
    } catch {
      // Ignore — history is optional
    }
  }, [planId]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // ── Polling ──
  const startPolling = useCallback((batchId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);

    const poll = async () => {
      try {
        const status = await api.getBatchStatus(planId, batchId);
        setBatchStatus(status);

        if (status.status !== 'running') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          onBatchComplete?.();
          loadHistory();
        }
      } catch {
        // Keep polling on transient failures
      }
    };

    // Immediate first poll
    poll();
    pollRef.current = setInterval(poll, 5000);
  }, [planId, onBatchComplete, loadHistory]);

  // ── Controlled mode: sync polling with external batch ID ──
  // NOTE: intentionally omits startPolling from deps to avoid infinite loop
  // (onBatchComplete is an inline arrow → new ref each render → startPolling
  //  recreates → effect re-fires → poll calls onBatchComplete → render → loop)
  useEffect(() => {
    if (initialBatchId === undefined) return;
    if (initialBatchId === null) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      setActiveBatchId(null);
      setBatchStatus(null);
      return;
    }
    setActiveBatchId(initialBatchId);

    // Clear any existing poll, then fetch once — only start polling if still running
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    let cancelled = false;
    api.getBatchStatus(planId, initialBatchId).then((status) => {
      if (cancelled) return;
      setBatchStatus(status);
      if (status.status === 'running') {
        startPolling(initialBatchId);
      }
    }).catch(() => {});

    return () => {
      cancelled = true;
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialBatchId, planId]);

  // ── Launch batch ──
  const handleLaunch = useCallback(async () => {
    setIsLaunching(true);
    setError(null);
    try {
      const params: BatchExecuteParams = {
        approval_policy: approvalPolicy,
        run_all: runAll,
      };
      const result = await api.launchBatchExecution(planId, params);
      setActiveBatchId(result.batch_id);
      setMode('monitor');
      onBatchIdChange?.(result.batch_id);
      startPolling(result.batch_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to launch batch');
    } finally {
      setIsLaunching(false);
    }
  }, [planId, approvalPolicy, runAll, startPolling]);

  // ── Retry failed ──
  const handleRetry = useCallback(async (batchId: string) => {
    setIsLaunching(true);
    setError(null);
    try {
      const result = await api.retryBatchExecution(planId, batchId);
      setActiveBatchId(result.batch_id);
      setMode('monitor');
      onBatchIdChange?.(result.batch_id);
      startPolling(result.batch_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retry batch');
    } finally {
      setIsLaunching(false);
    }
  }, [planId, startPolling]);

  // ── Approve single MVP (manual approval) ──
  const handleApproveMvp = useCallback(async (mvpId: number, phaseNumber: number) => {
    try {
      await api.approveMigrationPhase(planId, phaseNumber, mvpId);
      // Re-poll to pick up the change
      if (activeBatchId) {
        const status = await api.getBatchStatus(planId, activeBatchId);
        setBatchStatus(status);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve phase');
    }
  }, [planId, activeBatchId]);

  // ── Toggle expanded MVP row ──
  const toggleExpanded = (mvpId: number) => {
    setExpandedMvps(prev => {
      const next = new Set(prev);
      if (next.has(mvpId)) next.delete(mvpId);
      else next.add(mvpId);
      return next;
    });
  };

  // ── Progress bar ──
  const ProgressBar = ({ batch }: { batch: BatchStatus }) => {
    const total = batch.total_mvps;
    const done = batch.completed + batch.failed + batch.skipped + batch.needs_review;
    const pct = total > 0 ? Math.round((done / total) * 100) : 0;

    return (
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-text-muted">
            {done}/{total} MVPs processed
          </span>
          <span className="font-mono text-text-dim">{pct}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-void-surface">
          <div className="flex h-full">
            {batch.completed > 0 && (
              <div
                className="bg-success transition-all duration-500"
                style={{ width: `${(batch.completed / total) * 100}%` }}
              />
            )}
            {batch.needs_review > 0 && (
              <div
                className="bg-warning transition-all duration-500"
                style={{ width: `${(batch.needs_review / total) * 100}%` }}
              />
            )}
            {batch.failed > 0 && (
              <div
                className="bg-danger transition-all duration-500"
                style={{ width: `${(batch.failed / total) * 100}%` }}
              />
            )}
            {batch.skipped > 0 && (
              <div
                className="bg-text-dim/30 transition-all duration-500"
                style={{ width: `${(batch.skipped / total) * 100}%` }}
              />
            )}
          </div>
        </div>
        <div className="flex gap-3 text-[10px] text-text-dim">
          {batch.completed > 0 && <span className="text-success">{batch.completed} approved</span>}
          {batch.needs_review > 0 && <span className="text-warning">{batch.needs_review} needs review</span>}
          {batch.failed > 0 && <span className="text-danger">{batch.failed} failed</span>}
          {batch.skipped > 0 && <span>{batch.skipped} skipped</span>}
        </div>
      </div>
    );
  };

  // ── MVP Results Table ──
  const MvpResultsTable = ({ results, batch }: { results: BatchMvpResult[]; batch: BatchStatus }) => (
    <div className="divide-y divide-void-surface/50">
      {results.map((mvp) => {
        const isExpanded = expandedMvps.has(mvp.mvp_id);
        return (
          <div key={mvp.mvp_id} className="group">
            {/* Row — click navigates to MVP, chevron toggles expand */}
            <div
              className="flex cursor-pointer items-center gap-3 px-4 py-2.5 hover:bg-void-light/30"
              onClick={() => onMvpClick ? onMvpClick(mvp.mvp_id) : toggleExpanded(mvp.mvp_id)}
            >
              {/* Expand chevron */}
              <button
                onClick={(e) => { e.stopPropagation(); toggleExpanded(mvp.mvp_id); }}
                className="shrink-0 text-text-dim hover:text-text"
              >
                {isExpanded
                  ? <ChevronDown className="h-3.5 w-3.5" />
                  : <ChevronRight className="h-3.5 w-3.5" />
                }
              </button>

              {/* Status icon */}
              <StatusIcon status={mvp.status} />

              {/* MVP name */}
              <span className="flex-1 truncate text-xs text-text">
                {mvp.name}
              </span>

              {/* Status badge */}
              <StatusBadge status={mvp.status} />

              {/* Phase indicator */}
              <span className="text-[10px] text-text-dim">
                Phase {mvp.current_phase}
              </span>

              {/* Gate results summary */}
              {mvp.gate_results.length > 0 && (
                <span className="text-[10px] text-text-dim">
                  Gates: {mvp.gate_results.filter(g => g.passed).length}/{mvp.gate_results.length}
                </span>
              )}

              {/* Action buttons */}
              <div className="flex items-center gap-1.5 opacity-0 group-hover:opacity-100">
                {mvp.status === 'needs_review' && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleApproveMvp(mvp.mvp_id, mvp.current_phase);
                    }}
                    className="rounded bg-success/10 px-2 py-0.5 text-[10px] font-medium text-success hover:bg-success/20"
                  >
                    Approve
                  </button>
                )}
                {onMvpClick && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onMvpClick(mvp.mvp_id);
                    }}
                    className="rounded bg-void-surface px-2 py-0.5 text-[10px] text-text-muted hover:bg-void-surface/80"
                  >
                    <Eye className="inline h-3 w-3" /> View
                  </button>
                )}
              </div>
            </div>

            {/* Expanded detail */}
            {isExpanded && (
              <div className="border-t border-void-surface/30 bg-void-light/20 px-8 py-3">
                {mvp.error && (
                  <div className="mb-2 rounded bg-danger/5 px-3 py-2 text-xs text-danger">
                    {mvp.error}
                  </div>
                )}
                {mvp.gate_results.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-[10px] font-medium uppercase tracking-wider text-text-dim">
                      Gate Results
                    </p>
                    {mvp.gate_results.map((gate, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-2 text-xs"
                      >
                        {gate.passed
                          ? <CheckCircle2 className="h-3.5 w-3.5 text-success" />
                          : <XCircle className="h-3.5 w-3.5 text-danger" />
                        }
                        <span className="text-text-muted">{gate.gate_name}</span>
                        {gate.blocking && !gate.passed && (
                          <span className="rounded bg-danger/10 px-1.5 py-0.5 text-[9px] text-danger">
                            blocking
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {!mvp.error && mvp.gate_results.length === 0 && (
                  mvp.status === 'pending' ? (
                    <p className="text-xs text-text-dim">Waiting to execute...</p>
                  ) : (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3 text-xs text-text-dim">
                        <span>Phase {mvp.current_phase} {mvp.status === 'approved' ? 'approved' : 'executed'}</span>
                        {mvp.completed_at && (
                          <span>&middot; {new Date(mvp.completed_at).toLocaleTimeString()}</span>
                        )}
                      </div>
                      {onMvpClick && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onMvpClick(mvp.mvp_id);
                          }}
                          className="flex items-center gap-1 rounded bg-void-surface px-2.5 py-1 text-[11px] text-text-muted hover:bg-void-surface/80 hover:text-text"
                        >
                          <Eye className="h-3 w-3" />
                          View Outputs
                        </button>
                      )}
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );

  // ── Render ──

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Error banner */}
      {error && (
        <div className="rounded-lg border border-danger/30 bg-danger/5 px-4 py-2 text-xs text-danger">
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-2 text-danger/60 hover:text-danger"
          >
            dismiss
          </button>
        </div>
      )}

      {/* ── Mode: Launch ── */}
      {effectiveMode === 'launch' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-medium text-text">Batch Migration</h3>
            <p className="mt-1 text-xs text-text-dim">
              Execute migration phases across all {totalMvps} MVPs in one operation.
            </p>
          </div>

          {/* Configuration */}
          <div className="space-y-3 rounded-lg border border-void-surface bg-void-light/20 p-4">
            {/* Approval policy */}
            <div>
              <label className="mb-1.5 block text-[11px] font-medium uppercase tracking-wider text-text-dim">
                Approval Policy
              </label>
              <div className="flex gap-2">
                {([
                  { value: 'auto', label: 'Auto-approve', desc: 'Approve if all gates pass' },
                  { value: 'auto_non_blocking', label: 'Auto (non-blocking)', desc: 'Approve unless blocking gates fail' },
                  { value: 'manual', label: 'Manual', desc: 'Review each MVP after execution' },
                ] as const).map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setApprovalPolicy(opt.value)}
                    className={`flex-1 rounded-lg border px-3 py-2 text-left transition-colors ${
                      approvalPolicy === opt.value
                        ? 'border-glow/50 bg-glow/5 text-glow'
                        : 'border-void-surface bg-void-light/30 text-text-muted hover:border-text-dim/30'
                    }`}
                  >
                    <div className="text-xs font-medium">{opt.label}</div>
                    <div className="mt-0.5 text-[10px] opacity-70">{opt.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Run all toggle */}
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-medium text-text-muted">Run full pipeline per MVP</div>
                <div className="text-[10px] text-text-dim">
                  Transform → Approve → Test in sequence for each MVP
                </div>
              </div>
              <button
                onClick={() => setRunAll(!runAll)}
                className={`relative h-5 w-9 rounded-full transition-colors ${
                  runAll ? 'bg-glow' : 'bg-void-surface'
                }`}
              >
                <div
                  className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
                    runAll ? 'translate-x-4' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>
          </div>

          {/* Launch button */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleLaunch}
              disabled={isLaunching}
              className="flex items-center gap-2 rounded-lg bg-glow px-5 py-2.5 text-sm font-medium text-white hover:bg-glow/90 disabled:opacity-50"
            >
              {isLaunching ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              Start Batch Migration
            </button>

            {/* History link (only in uncontrolled mode — sidebar handles this otherwise) */}
            {!isControlled && batchHistory.length > 0 && (
              <button
                onClick={() => setMode('history')}
                className="text-xs text-text-dim hover:text-text-muted"
              >
                View {batchHistory.length} previous run{batchHistory.length !== 1 ? 's' : ''}
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── Mode: Monitor (loading) ── */}
      {effectiveMode === 'monitor' && !batchStatus && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
        </div>
      )}

      {/* ── Mode: Monitor ── */}
      {effectiveMode === 'monitor' && batchStatus && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium text-text">
                Batch Migration
                {batchStatus.status === 'running' && (
                  <Loader2 className="ml-2 inline h-3.5 w-3.5 animate-spin text-glow" />
                )}
              </h3>
              <p className="mt-0.5 text-[10px] text-text-dim">
                Policy: {batchStatus.approval_policy} | Run all: {batchStatus.run_all ? 'yes' : 'no'}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {batchStatus.status !== 'running' && (
                <>
                  <button
                    onClick={() => { setMode('launch'); onBatchIdChange?.(null); }}
                    className="rounded bg-void-surface px-3 py-1.5 text-xs text-text-muted hover:bg-void-surface/80"
                  >
                    New Batch
                  </button>
                  {batchStatus.failed > 0 && (
                    <button
                      onClick={() => handleRetry(batchStatus.batch_id)}
                      disabled={isLaunching}
                      className="flex items-center gap-1.5 rounded bg-warning/10 px-3 py-1.5 text-xs font-medium text-warning hover:bg-warning/20 disabled:opacity-50"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      Retry Failed ({batchStatus.failed})
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Progress bar */}
          <ProgressBar batch={batchStatus} />

          {/* MVP results table */}
          <div className="overflow-hidden rounded-lg border border-void-surface">
            <div className="flex items-center gap-3 border-b border-void-surface bg-void-light/30 px-4 py-2 text-[10px] font-medium uppercase tracking-wider text-text-dim">
              <span className="w-5" />
              <span className="w-5" />
              <span className="flex-1">MVP</span>
              <span className="w-24 text-center">Status</span>
              <span className="w-16 text-center">Phase</span>
              <span className="w-20 text-center">Gates</span>
              <span className="w-24" />
            </div>
            <MvpResultsTable results={batchStatus.mvp_results} batch={batchStatus} />
          </div>

          {/* Batch-level final state */}
          {batchStatus.status !== 'running' && (
            <div className={`rounded-lg border p-4 ${
              batchStatus.status === 'complete'
                ? 'border-success/30 bg-success/5'
                : batchStatus.status === 'partial_failure'
                  ? 'border-danger/30 bg-danger/5'
                  : 'border-warning/30 bg-warning/5'
            }`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className={`text-xs font-medium ${
                    batchStatus.status === 'complete' ? 'text-success'
                    : batchStatus.status === 'partial_failure' ? 'text-danger'
                    : 'text-warning'
                  }`}>
                    {batchStatus.status === 'complete'
                      ? 'All MVPs migrated successfully'
                      : batchStatus.status === 'partial_failure'
                        ? `${batchStatus.failed} MVP${batchStatus.failed !== 1 ? 's' : ''} failed`
                        : `${batchStatus.needs_review} MVP${batchStatus.needs_review !== 1 ? 's' : ''} need review`
                    }
                  </p>
                  {batchStatus.completed_at && (
                    <p className="mt-0.5 text-[10px] text-text-dim">
                      Completed at {new Date(batchStatus.completed_at).toLocaleTimeString()}
                    </p>
                  )}
                </div>
                <div className="flex gap-2">
                  {!isControlled && batchHistory.length > 0 && (
                    <button
                      onClick={() => setMode('history')}
                      className="rounded bg-void-surface px-3 py-1.5 text-xs text-text-muted hover:text-text"
                    >
                      History
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Mode: History (uncontrolled only — sidebar handles this in controlled mode) ── */}
      {effectiveMode === 'history' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-text">Batch Execution History</h3>
            <button
              onClick={() => setMode('launch')}
              className="text-xs text-text-dim hover:text-text-muted"
            >
              Back to Launch
            </button>
          </div>

          {batchHistory.length === 0 ? (
            <p className="text-xs text-text-dim">No batch runs yet.</p>
          ) : (
            <div className="space-y-2">
              {batchHistory.map((batch) => (
                <div
                  key={batch.batch_id}
                  className="flex cursor-pointer items-center gap-3 rounded-lg border border-void-surface p-3 hover:bg-void-light/30"
                  onClick={() => {
                    setActiveBatchId(batch.batch_id);
                    setBatchStatus(batch);
                    setMode('monitor');
                    if (batch.status === 'running') {
                      startPolling(batch.batch_id);
                    }
                  }}
                >
                  <StatusIcon status={
                    batch.status === 'complete' ? 'approved'
                    : batch.status === 'partial_failure' ? 'failed'
                    : batch.status === 'running' ? 'processing'
                    : 'needs_review'
                  } />
                  <div className="flex-1">
                    <div className="text-xs text-text-muted">
                      {batch.total_mvps} MVPs &middot; {batch.approval_policy}
                      {batch.run_all ? ' &middot; full pipeline' : ''}
                    </div>
                    <div className="text-[10px] text-text-dim">
                      {batch.started_at ? new Date(batch.started_at).toLocaleString() : 'unknown'}
                    </div>
                  </div>
                  <div className="flex gap-2 text-[10px]">
                    {batch.completed > 0 && <span className="text-success">{batch.completed} done</span>}
                    {batch.failed > 0 && <span className="text-danger">{batch.failed} failed</span>}
                    {batch.needs_review > 0 && <span className="text-warning">{batch.needs_review} review</span>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
