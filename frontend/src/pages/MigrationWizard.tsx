/**
 * MigrationWizard — main page for the MVP-centric migration pipeline.
 *
 * URL: /migration/:planId
 *
 * Version-aware (plan.pipeline_version):
 *   V1 (6-phase): Phase 1 Discovery → Phase 2 Architecture → per-MVP: Analyze, Design, Transform, Test
 *   V2 (4-phase): Phase 1 Architecture → Phase 2 Discovery → per-MVP: Transform, Test
 *
 * Layout modes:
 *   1. New plan form (planId === 'new')
 *   2. Plan-level phases (Phase 1, Phase 2)
 *   3. MVP execution mode: MvpTimeline sidebar + PhaseViewer for selected MVP
 *
 * Also handles plan creation when accessed via /migration/new?project=<id>.
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useParams, useSearchParams, useNavigate, Link } from 'react-router-dom';
import { Loader2, Trash2, ArrowLeft, ArrowRight, Lock, Play } from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import { PhaseTimeline } from '../components/migration/PhaseTimeline.tsx';
import { PhaseViewer } from '../components/migration/PhaseViewer.tsx';
import { PlanCreator } from '../components/migration/PlanCreator.tsx';
import { MvpDiscoveryPanel } from '../components/migration/MvpDiscoveryPanel.tsx';
import { MvpTimeline } from '../components/migration/MvpTimeline.tsx';
import { MvpInfoBar } from '../components/migration/MvpInfoBar.tsx';
import { AssetInventory } from '../components/migration/AssetInventory.tsx';
import { BatchExecutionPanel } from '../components/migration/BatchExecutionPanel.tsx';
import * as api from '../services/api.ts';
import type { BatchStatus } from '../services/api.ts';
import type { MigrationPlan, MigrationPhaseOutput } from '../types/index.ts';

// Version-aware per-MVP phase constants
const MVP_PHASE_NUMBERS_V1 = [3, 4, 5, 6] as const;
const MVP_PHASE_NUMBERS_V2 = [3, 4] as const;

const MVP_PHASE_LABELS_V1: Record<number, string> = {
  3: 'Analyze',
  4: 'Design',
  5: 'Transform',
  6: 'Test',
};
const MVP_PHASE_LABELS_V2: Record<number, string> = {
  3: 'Transform',
  4: 'Test',
};

function getMvpPhaseNumbers(version: number): readonly number[] {
  return version === 2 ? MVP_PHASE_NUMBERS_V2 : MVP_PHASE_NUMBERS_V1;
}

function getMvpPhaseLabels(version: number): Record<number, string> {
  return version === 2 ? MVP_PHASE_LABELS_V2 : MVP_PHASE_LABELS_V1;
}

export function MigrationWizard() {
  const { planId } = useParams<{ planId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const isNewPlan = planId === 'new';
  const projectId = searchParams.get('project') ?? '';

  // ── Core state ──
  const [projectName, setProjectName] = useState<string>('');
  const [sourceProjectId, setSourceProjectId] = useState<string>(projectId);
  const [plan, setPlan] = useState<MigrationPlan | null>(null);
  const [isLoading, setIsLoading] = useState(!isNewPlan);
  const [isCreating, setIsCreating] = useState(false);
  const [isCreatingPhases, setIsCreatingPhases] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Plan-level phase state ──
  const [activePhase, setActivePhase] = useState(1);
  const [phaseOutput, setPhaseOutput] = useState<MigrationPhaseOutput | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);

  // ── MVP-level phase state ──
  const [selectedMvpId, setSelectedMvpId] = useState<number | null>(null);
  const [mvpActivePhase, setMvpActivePhase] = useState<number>(3);
  const [mvpPhaseOutput, setMvpPhaseOutput] = useState<MigrationPhaseOutput | null>(null);
  const [isMvpExecuting, setIsMvpExecuting] = useState(false);

  // ── Framework doc enrichment ──
  const [isEnriching, setIsEnriching] = useState(false);
  const [enrichResult, setEnrichResult] = useState<{ enriched: string[]; failed: string[] } | null>(null);

  // ── View mode ──
  // 'plan' = plan-level phases (1, 2)
  // 'mvp' = per-MVP phases (3-6) with MVP sidebar; no MVP selected → batch panel
  const [viewMode, setViewMode] = useState<'plan' | 'mvp'>('plan');

  // ── Batch run state (controls sidebar + panel when no MVP selected) ──
  const [batchRuns, setBatchRuns] = useState<BatchStatus[]>([]);
  const [selectedBatchRunId, setSelectedBatchRunId] = useState<string | null>(null);
  const batchInitRef = useRef(false);

  // ── Version-aware constants ──
  const version = plan?.pipeline_version ?? 1;
  const mvpPhaseNumbers = useMemo(() => getMvpPhaseNumbers(version), [version]);
  const mvpPhaseLabels = useMemo(() => getMvpPhaseLabels(version), [version]);
  const lastMvpPhase = mvpPhaseNumbers[mvpPhaseNumbers.length - 1];

  // V1: Discovery is Phase 1, Architecture is Phase 2
  // V2: Architecture is Phase 1, Discovery is Phase 2
  const discoveryPhase = version === 2 ? 2 : 1;

  // ── Derived state ──
  const phase1Info = plan?.plan_phases.find(p => p.phase_number === 1);
  const phase2Info = plan?.plan_phases.find(p => p.phase_number === 2);
  const phase1Complete = phase1Info?.status === 'complete';
  const phase1Approved = phase1Info?.approved ?? false;
  const phase2Complete = phase2Info?.status === 'complete';
  const phase2Approved = phase2Info?.approved ?? false;
  const hasMvps = (plan?.mvps?.length ?? 0) > 0;

  // The discovery phase is where MVPs appear; the second plan-level phase gates MVP phases
  const discoveryComplete = version === 2 ? phase2Complete : phase1Complete;
  const bothPlanPhasesApproved = phase1Approved && phase2Approved;

  // Check if MVP phases have been created (any mvp has phases array populated)
  const hasMvpPhases = plan?.mvps?.some(m => m.phases.length > 0) ?? false;

  // Migration is underway once both plan phases are approved and MVP phases exist
  const migrationUnderway = bothPlanPhasesApproved && hasMvpPhases;

  const selectedMvp = plan?.mvps?.find(m => m.mvp_id === selectedMvpId) ?? null;

  // Asset inventory gate: show AssetInventory before discovery if no strategies saved
  const needsAssetInventory = plan != null && !isNewPlan && !plan.asset_strategies && !discoveryComplete;

  // ── Load plan data ──
  const refreshPlan = useCallback(async () => {
    if (!planId || isNewPlan) return;
    try {
      const p = await api.getMigrationPlan(planId);
      setPlan(p);
      return p;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load plan');
      return null;
    }
  }, [planId, isNewPlan]);

  useEffect(() => {
    if (isNewPlan || !planId) return;
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError(null);
      try {
        const p = await api.getMigrationPlan(planId!);
        if (cancelled) return;
        setPlan(p);

        // Determine initial view mode: both plan phases approved + MVP phases exist
        const p1 = p.plan_phases.find(ph => ph.phase_number === 1);
        const p2 = p.plan_phases.find(ph => ph.phase_number === 2);
        const bothApproved = (p1?.approved ?? false) && (p2?.approved ?? false);
        const mvpPhasesExist = p.mvps?.some(m => m.phases.length > 0) ?? false;

        if (bothApproved && mvpPhasesExist) {
          // Migration is underway — route to the right execution view
          const anyMvpInProgress = p.mvps?.find(m =>
            m.phases.some(ph => ph.status === 'running' || ph.status === 'complete')
          );
          if (anyMvpInProgress) {
            setViewMode('mvp');
            setSelectedMvpId(anyMvpInProgress.mvp_id);
          } else {
            // Default to MVP execution — batch panel shows when no MVP selected
            setViewMode('mvp');
          }
        }

        // Set active plan-level phase
        const lastComplete = [...p.plan_phases].reverse().find(ph => ph.status === 'complete' || ph.approved);
        const firstPending = p.plan_phases.find(ph => ph.status === 'pending');
        if (lastComplete && !lastComplete.approved) {
          setActivePhase(lastComplete.phase_number);
        } else if (firstPending) {
          setActivePhase(firstPending.phase_number);
        } else if (lastComplete) {
          setActivePhase(lastComplete.phase_number);
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load plan');
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [planId, isNewPlan]);

  // Resolve project name
  useEffect(() => {
    const pid = plan?.source_project_id ?? projectId;
    if (!pid) return;
    setSourceProjectId(pid);
    let cancelled = false;
    api.getProject(pid).then((p) => {
      if (!cancelled) setProjectName(p.name);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [plan?.source_project_id, projectId]);

  // Load plan-level phase output
  useEffect(() => {
    if (!plan || !planId || isNewPlan || viewMode !== 'plan') return;

    const phaseInfo = plan.plan_phases.find(p => p.phase_number === activePhase);
    if (!phaseInfo || phaseInfo.status === 'pending') {
      setPhaseOutput(null);
      return;
    }

    let cancelled = false;
    async function loadPhase() {
      try {
        const output = await api.getMigrationPhaseOutput(planId!, activePhase);
        if (cancelled) return;
        setPhaseOutput(output);
      } catch {
        if (cancelled) return;
        setPhaseOutput(null);
      }
    }

    loadPhase();
    return () => { cancelled = true; };
  }, [plan, planId, activePhase, isNewPlan, viewMode]);

  // Load MVP phase output when selectedMvp or mvpActivePhase changes
  useEffect(() => {
    if (!plan || !planId || isNewPlan || viewMode !== 'mvp' || !selectedMvpId) return;

    const mvp = plan.mvps?.find(m => m.mvp_id === selectedMvpId);
    if (!mvp) {
      setMvpPhaseOutput(null);
      return;
    }

    const phaseInfo = mvp.phases.find(p => p.phase_number === mvpActivePhase);
    if (!phaseInfo || phaseInfo.status === 'pending') {
      setMvpPhaseOutput(null);
      return;
    }

    let cancelled = false;
    async function loadMvpPhase() {
      try {
        const output = await api.getMigrationPhaseOutput(planId!, mvpActivePhase, selectedMvpId!);
        if (cancelled) return;
        setMvpPhaseOutput(output);
      } catch {
        if (cancelled) return;
        setMvpPhaseOutput(null);
      }
    }

    loadMvpPhase();
    return () => { cancelled = true; };
  }, [plan, planId, selectedMvpId, mvpActivePhase, isNewPlan, viewMode]);

  // Auto-select first MVP active phase when MVP selection changes
  useEffect(() => {
    if (!selectedMvp) return;
    const mvpPhases = selectedMvp.phases;
    // Find the first non-approved phase, or the last approved one
    const firstPending = mvpPhases.find(p => p.status === 'pending');
    const lastComplete = [...mvpPhases].reverse().find(p => p.status === 'complete' && !p.approved);
    const lastApproved = [...mvpPhases].reverse().find(p => p.approved);

    if (lastComplete) {
      setMvpActivePhase(lastComplete.phase_number);
    } else if (firstPending) {
      setMvpActivePhase(firstPending.phase_number);
    } else if (lastApproved) {
      setMvpActivePhase(lastApproved.phase_number);
    } else {
      setMvpActivePhase(mvpPhaseNumbers[0]); // default to first MVP phase
    }
    setMvpPhaseOutput(null);
  }, [selectedMvpId, selectedMvp]);

  // ── Handlers: Plan creation ──
  const handleCreatePlan = useCallback(async (data: Parameters<typeof api.createMigrationPlan>[0]) => {
    setIsCreating(true);
    setError(null);
    try {
      const newPlan = await api.createMigrationPlan(data);
      navigate(`/migration/${newPlan.plan_id}`, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create plan');
    } finally {
      setIsCreating(false);
    }
  }, [navigate]);

  // ── Handlers: Plan-level phase execution ──
  const handleExecute = useCallback(async () => {
    if (!planId || isNewPlan) return;
    setIsExecuting(true);
    setError(null);
    try {
      let result: MigrationPhaseOutput;
      // Discovery is the special phase that runs clustering + LLM
      // V1: Phase 1 = Discovery. V2: Phase 2 = Discovery.
      if (activePhase === discoveryPhase) {
        const discovery = await api.runDiscovery(planId);
        result = {
          phase_id: '',
          phase_number: discoveryPhase,
          phase_type: 'discovery',
          status: 'complete',
          output: discovery.phase_output?.output ?? '',
          output_files: discovery.phase_output?.output_files ?? [],
          approved: false,
          approved_at: null,
          input_summary: null,
          mvp_id: null,
          phase_metadata: null,
        };
      } else {
        result = await api.executeMigrationPhase(planId, activePhase);
      }
      setPhaseOutput(result);
      await refreshPlan();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Phase execution failed');
      await refreshPlan();
    } finally {
      setIsExecuting(false);
    }
  }, [planId, activePhase, isNewPlan, discoveryPhase, refreshPlan]);

  // ── Handlers: Plan-level phase approval ──
  const handleApprove = useCallback(async () => {
    if (!planId || isNewPlan) return;
    setError(null);
    try {
      const updated = await api.approveMigrationPhase(planId, activePhase);
      setPlan(updated);

      // After Phase 2 approved (the second plan-level phase): auto-create per-MVP phases
      if (activePhase === 2 && updated.mvps?.length > 0) {
        try {
          await api.createMvpPhases(planId);
          await refreshPlan();
        } catch {
          // MVP phases may already exist
        }
      }

      // Auto-advance to next plan-level phase
      if (activePhase < 2) {
        setActivePhase(activePhase + 1);
        setPhaseOutput(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve phase');
    }
  }, [planId, activePhase, isNewPlan, refreshPlan]);

  // ── Handlers: MVP phase execution ──
  const handleMvpExecute = useCallback(async () => {
    if (!planId || !selectedMvpId) return;
    setIsMvpExecuting(true);
    setError(null);
    try {
      const result = await api.executeMigrationPhase(planId, mvpActivePhase, selectedMvpId);
      setMvpPhaseOutput(result);
      await refreshPlan();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'MVP phase execution failed');
      await refreshPlan();
    } finally {
      setIsMvpExecuting(false);
    }
  }, [planId, selectedMvpId, mvpActivePhase, refreshPlan]);

  // ── Handlers: MVP phase approval ──
  const handleMvpApprove = useCallback(async () => {
    if (!planId || !selectedMvpId) return;
    setError(null);
    try {
      const updated = await api.approveMigrationPhase(planId, mvpActivePhase, selectedMvpId);
      setPlan(updated);

      // Auto-advance to next MVP phase
      const nextPhase = mvpActivePhase + 1;
      if (nextPhase <= lastMvpPhase) {
        setMvpActivePhase(nextPhase);
        setMvpPhaseOutput(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve MVP phase');
    }
  }, [planId, selectedMvpId, mvpActivePhase, lastMvpPhase]);

  // ── Handlers: Create MVP phases ──
  const handleCreateMvpPhases = useCallback(async () => {
    if (!planId) return;
    setIsCreatingPhases(true);
    setError(null);
    try {
      await api.createMvpPhases(planId);
      await refreshPlan();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create MVP phases');
    } finally {
      setIsCreatingPhases(false);
    }
  }, [planId, refreshPlan]);

  // ── Handlers: Plan deletion ──
  const handleDelete = useCallback(async () => {
    if (!planId || isNewPlan) return;
    if (!confirm('Delete this migration plan? This cannot be undone.')) return;
    try {
      await api.deleteMigrationPlan(planId);
      navigate(sourceProjectId ? `/project/${sourceProjectId}` : '/', { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete plan');
    }
  }, [planId, isNewPlan, sourceProjectId, navigate]);

  // ── Handlers: Framework doc enrichment ──
  const handleEnrichDocs = useCallback(async () => {
    if (!planId || isNewPlan) return;
    setIsEnriching(true);
    setError(null);
    try {
      const result = await api.enrichFrameworkDocs(planId);
      setEnrichResult({ enriched: result.enriched, failed: result.failed });
      await refreshPlan();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to enrich docs');
    } finally {
      setIsEnriching(false);
    }
  }, [planId, isNewPlan, refreshPlan]);

  // ── Handlers: MVP selection ──
  const handleSelectMvp = useCallback((mvpId: number) => {
    setSelectedMvpId(mvpId);
    setViewMode('mvp');
    setMvpPhaseOutput(null);
  }, []);

  const handleBackToPlan = useCallback(() => {
    setViewMode('plan');
    setSelectedMvpId(null);
    setMvpPhaseOutput(null);
  }, []);

  // ── Batch runs loaded callback ──
  const handleBatchRunsLoaded = useCallback((runs: BatchStatus[]) => {
    setBatchRuns(runs);
    // Auto-select a running batch on first load
    if (!batchInitRef.current) {
      batchInitRef.current = true;
      const running = runs.find(r => r.status === 'running');
      if (running) setSelectedBatchRunId(running.batch_id);
    }
  }, []);

  // ── Computed: can execute ──
  const canExecutePlanPhase = (() => {
    if (!plan) return false;
    if (activePhase === 1) return true;
    const prevPhase = plan.plan_phases.find(p => p.phase_number === activePhase - 1);
    return prevPhase?.approved ?? false;
  })();

  const canExecuteMvpPhase = (() => {
    if (!selectedMvp) return false;
    // First MVP phase can run if both plan-level phases are approved
    if (mvpActivePhase === mvpPhaseNumbers[0]) return bothPlanPhasesApproved;
    // Other phases need previous MVP phase approved
    const prevMvpPhase = selectedMvp.phases.find(p => p.phase_number === mvpActivePhase - 1);
    return prevMvpPhase?.approved ?? false;
  })();

  const activePhaseInfo = plan?.plan_phases.find(p => p.phase_number === activePhase);
  const mvpActivePhaseInfo = selectedMvp?.phases.find(p => p.phase_number === mvpActivePhase);

  // ── Render ──

  if (isLoading) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-text-dim" />
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="flex h-full flex-col">
        {/* Top bar with breadcrumb navigation */}
        <div className="flex items-center justify-between border-b border-void-surface px-4 py-3">
          <div className="flex items-center gap-1.5 text-sm">
            <Link to="/" className="text-text-dim hover:text-text-muted">
              Migration Plans
            </Link>
            {sourceProjectId && (
              <>
                <span className="text-text-dim/40">&gt;</span>
                <Link
                  to={`/project/${sourceProjectId}`}
                  className="text-text-muted hover:text-text"
                >
                  {projectName || 'Project'}
                </Link>
              </>
            )}
            <span className="text-text-dim/40">&gt;</span>
            <span className={`font-medium ${viewMode !== 'plan' ? 'text-text-muted hover:text-text cursor-pointer' : 'text-text'}`}
              onClick={viewMode !== 'plan' ? handleBackToPlan : undefined}
            >
              {isNewPlan ? 'New Migration' : (plan?.target_brief ?? 'Migration')}
            </span>
            {viewMode === 'plan' && !isNewPlan && needsAssetInventory && (
              <>
                <span className="text-text-dim/40">&gt;</span>
                <span className="text-glow">Asset Inventory</span>
              </>
            )}
            {viewMode === 'plan' && !isNewPlan && !needsAssetInventory && (
              migrationUnderway ? (
                <>
                  <span className="text-text-dim/40">&gt;</span>
                  <span className="text-text-muted">Plan Overview</span>
                </>
              ) : activePhaseInfo ? (
                <>
                  <span className="text-text-dim/40">&gt;</span>
                  <span className="text-text-muted">
                    Phase {activePhase}: {activePhaseInfo.phase_type}
                  </span>
                </>
              ) : null
            )}
            {viewMode === 'mvp' && selectedMvp && (
              <>
                <span className="text-text-dim/40">&gt;</span>
                <span className="text-glow">{selectedMvp.name}</span>
                {mvpActivePhaseInfo && (
                  <>
                    <span className="text-text-dim/40">&gt;</span>
                    <span className="text-text-muted">
                      {mvpPhaseLabels[mvpActivePhase] ?? mvpActivePhaseInfo.phase_type}
                    </span>
                  </>
                )}
              </>
            )}
            {plan && (
              <span className={`ml-2 rounded px-2 py-0.5 text-xs ${
                plan.status === 'complete'
                  ? 'bg-success/10 text-success'
                  : plan.status === 'in_progress'
                    ? 'bg-glow/10 text-glow'
                    : 'bg-void-surface/50 text-text-muted'
              }`}>
                {plan.status}
              </span>
            )}
            {/* Framework docs enrichment badge */}
            {plan && !isNewPlan && (
              <span className="ml-2 flex items-center gap-1.5">
                {enrichResult?.enriched?.length ? (
                  <span className="rounded bg-success/10 px-2 py-0.5 text-xs text-success">
                    Docs: {enrichResult.enriched.length} framework{enrichResult.enriched.length !== 1 ? 's' : ''}
                  </span>
                ) : null}
                <button
                  onClick={handleEnrichDocs}
                  disabled={isEnriching}
                  className="rounded bg-void-surface px-2 py-0.5 text-xs text-text-muted hover:bg-void-surface/80 hover:text-text disabled:opacity-50"
                  title="Fetch/refresh target framework documentation"
                >
                  {isEnriching ? 'Enriching...' : 'Refresh Docs'}
                </button>
              </span>
            )}
          </div>
          {plan && (
            <button
              onClick={handleDelete}
              className="text-text-dim hover:text-danger"
              title="Delete plan"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Source → Target tech stack bar (hidden during Asset Inventory — it has its own context pill) */}
        {plan && !isNewPlan && !needsAssetInventory && (plan.source_stack || plan.target_stack) && (
          <div className="flex items-center gap-3 border-b border-void-surface/50 bg-void-light/30 px-4 py-1.5">
            {/* Source */}
            {plan.source_stack && (
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] font-medium uppercase tracking-wider text-text-dim">Source</span>
                {plan.source_stack.languages.length > 0 ? (
                  plan.source_stack.languages.map((lang) => (
                    <span key={lang} className="rounded bg-void-surface/80 px-1.5 py-0.5 text-[11px] font-medium text-text-muted">
                      {lang}
                    </span>
                  ))
                ) : plan.source_stack.primary_language ? (
                  <span className="rounded bg-void-surface/80 px-1.5 py-0.5 text-[11px] font-medium text-text-muted">
                    {plan.source_stack.primary_language}
                  </span>
                ) : null}
              </div>
            )}

            {/* Arrow */}
            {plan.source_stack && (
              <ArrowRight className="h-3 w-3 text-text-dim" />
            )}

            {/* Target — only show when populated (derived from asset strategies) */}
            {(plan.target_stack.languages.length > 0 || plan.target_stack.frameworks.length > 0) && (
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] font-medium uppercase tracking-wider text-text-dim">Target</span>
                {plan.target_stack.languages.map((lang) => (
                  <span key={lang} className="rounded bg-glow/10 px-1.5 py-0.5 text-[11px] font-medium text-glow">
                    {lang}
                  </span>
                ))}
                {plan.target_stack.frameworks.map((fw) => (
                  <span key={fw} className="rounded bg-nebula/10 px-1.5 py-0.5 text-[11px] font-medium text-nebula">
                    {fw}
                  </span>
                ))}
                {plan.target_stack.versions && Object.keys(plan.target_stack.versions).length > 0 && (
                  Object.entries(plan.target_stack.versions).map(([key, ver]) => (
                    <span key={key} className="text-[10px] text-text-dim">
                      {key} {ver}
                    </span>
                  ))
                )}
              </div>
            )}

            {/* Migration type badge */}
            {plan.migration_type && (
              <span className="ml-auto rounded bg-void-surface/60 px-1.5 py-0.5 text-[10px] text-text-dim">
                {plan.migration_type.replace(/_/g, ' ')}
              </span>
            )}
          </div>
        )}

        {/* Error banner */}
        {error && (
          <div className="border-b border-danger/30 bg-danger/5 px-4 py-2 text-xs text-danger">
            {error}
          </div>
        )}

        {/* New plan form */}
        {isNewPlan && (
          <div className="flex-1 overflow-y-auto p-8">
            <PlanCreator
              projectId={projectId}
              projectName={projectName || 'your project'}
              onCreatePlan={handleCreatePlan}
              isCreating={isCreating}
            />
          </div>
        )}

        {/* Asset Inventory step — shown before discovery when no strategies are saved */}
        {needsAssetInventory && plan && viewMode === 'plan' && (
          <div className="flex-1 overflow-y-auto p-6">
            <AssetInventory
              planId={planId!}
              plan={plan}
              onConfirm={async () => {
                // After saving strategies, refresh plan and proceed
                await refreshPlan();
              }}
            />
          </div>
        )}

        {/* View mode tab bar — shown when migration is underway (replaces old banner) */}
        {!isNewPlan && plan && !needsAssetInventory && migrationUnderway && (
          <div className="flex items-center gap-1 border-b border-void-surface px-4 py-2">
            {([
              { key: 'plan' as const, label: 'Plan Overview', icon: <Lock className="h-3 w-3" /> },
              { key: 'mvp' as const, label: 'MVP Execution' },
            ]).map(({ key, label, icon }) => (
              <button
                key={key}
                onClick={() => setViewMode(key)}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  viewMode === key
                    ? 'bg-glow/10 text-glow border-b-2 border-glow'
                    : 'text-text-muted hover:text-text hover:bg-void-light/50'
                }`}
              >
                {icon}
                {label}
              </button>
            ))}
            <span className="ml-auto text-[10px] text-text-dim">
              {plan.mvps.length} MVP{plan.mvps.length !== 1 ? 's' : ''} queued
            </span>
          </div>
        )}

        {/* Existing plan — Plan-level view */}
        {!isNewPlan && plan && !needsAssetInventory && viewMode === 'plan' && (
          <>
            {/* Phase timeline (plan-level phases only) */}
            <div className="border-b border-void-surface">
              <PhaseTimeline
                phases={plan.plan_phases}
                activePhase={activePhase}
                onSelectPhase={(n) => {
                  setActivePhase(n);
                  setPhaseOutput(null);
                }}
                locked={migrationUnderway}
              />
            </div>

            {/* Phase viewer + MVP Discovery panel */}
            <div className="flex flex-1 overflow-hidden">
              {/* Left: Phase output */}
              <div className={`flex flex-col overflow-hidden ${
                activePhase === discoveryPhase && discoveryComplete && hasMvps ? 'w-3/5 border-r border-void-surface' : 'flex-1'
              }`}>
                <PhaseViewer
                  phase={phaseOutput}
                  phaseNumber={activePhase}
                  phaseType={activePhaseInfo?.phase_type ?? 'unknown'}
                  canExecute={canExecutePlanPhase}
                  isExecuting={isExecuting}
                  onExecute={handleExecute}
                  onApprove={handleApprove}
                  planId={planId}
                  sourceProjectId={sourceProjectId}
                />
              </div>

              {/* Right: MVP Discovery panel (shown when discovery phase complete + has MVPs) */}
              {activePhase === discoveryPhase && discoveryComplete && hasMvps && (
                <div className="w-2/5 overflow-y-auto">
                  <MvpDiscoveryPanel
                    planId={planId!}
                    mvps={plan.mvps}
                    onMvpsChanged={() => refreshPlan()}
                    onCreatePhases={handleCreateMvpPhases}
                    isCreatingPhases={isCreatingPhases}
                  />
                </div>
              )}
            </div>
          </>
        )}

        {/* Existing plan — MVP execution view */}
        {!isNewPlan && plan && viewMode === 'mvp' && (
          <div className="flex flex-1 overflow-hidden">
            {/* Left sidebar: MVP Timeline or Batch Runs */}
            <div className="w-60 shrink-0 overflow-y-auto border-r border-void-surface">
              {selectedMvpId ? (
                <MvpTimeline
                  mvps={plan.mvps}
                  selectedMvpId={selectedMvpId}
                  onSelectMvp={handleSelectMvp}
                />
              ) : (
                <div className="flex flex-col">
                  <div className="p-3 text-[10px] font-medium uppercase tracking-wider text-text-dim">
                    Batch Runs
                  </div>
                  {/* New Batch option */}
                  <button
                    onClick={() => setSelectedBatchRunId(null)}
                    className={`mx-2 mb-1 flex items-center gap-2 rounded-lg px-3 py-2 text-xs transition-colors ${
                      selectedBatchRunId === null
                        ? 'border border-glow/30 bg-glow/10 text-glow'
                        : 'border border-transparent text-text-muted hover:bg-void-light/50'
                    }`}
                  >
                    <Play className="h-3.5 w-3.5" />
                    New Batch
                  </button>
                  {/* Past runs */}
                  {batchRuns.length > 0 && (
                    <div className="mt-1 border-t border-void-surface/50">
                      {batchRuns.map((run, idx) => (
                        <button
                          key={run.batch_id}
                          onClick={() => setSelectedBatchRunId(run.batch_id)}
                          className={`flex w-full items-center gap-2.5 px-4 py-2.5 text-left transition-colors ${
                            selectedBatchRunId === run.batch_id
                              ? 'border-l-2 border-glow bg-glow/5'
                              : 'border-l-2 border-transparent hover:bg-void-light/30'
                          }`}
                        >
                          <div className={`h-2 w-2 shrink-0 rounded-full ${
                            run.status === 'complete' ? 'bg-success' :
                            run.status === 'running' ? 'bg-glow animate-pulse' :
                            run.status === 'partial_failure' ? 'bg-danger' :
                            'bg-warning'
                          }`} />
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-xs text-text-muted">
                              Run #{batchRuns.length - idx}
                            </div>
                            <div className="text-[10px] text-text-dim">
                              {run.total_mvps} MVPs &middot; {run.completed} done
                              {run.failed > 0 && ` · ${run.failed} failed`}
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Center: Phase execution for selected MVP */}
            <div className="flex flex-1 flex-col overflow-hidden">
              {selectedMvp ? (
                <>
                  {/* Back to execution overview */}
                  <button
                    onClick={() => { setSelectedMvpId(null); setMvpPhaseOutput(null); }}
                    className="flex items-center gap-1.5 text-xs text-text-dim hover:text-text px-4 py-1.5 border-b border-void-surface"
                  >
                    <ArrowLeft className="h-3 w-3" />
                    Execution Overview
                  </button>

                  {/* Per-MVP phase stepper */}
                  <div className="flex items-center gap-1 border-b border-void-surface px-4 py-2">
                    {mvpPhaseNumbers.map((phaseNum, idx) => {
                      const phase = selectedMvp.phases.find(p => p.phase_number === phaseNum);
                      const isActive = mvpActivePhase === phaseNum;
                      const isClickable = phase && (phase.status === 'complete' || phase.approved || phase.status === 'error');

                      return (
                        <div key={phaseNum} className="flex items-center">
                          {idx > 0 && (
                            <div className={`h-px w-8 shrink-0 ${
                              phase?.approved ? 'bg-success/50' : 'bg-void-surface'
                            }`} />
                          )}
                          <button
                            onClick={() => {
                              if (isClickable || isActive) {
                                setMvpActivePhase(phaseNum);
                                setMvpPhaseOutput(null);
                              }
                            }}
                            disabled={!isClickable && !isActive}
                            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs transition-colors ${
                              isActive
                                ? 'bg-glow/10 text-glow border border-glow/30'
                                : phase?.approved
                                  ? 'text-success border border-success/30 hover:bg-success/5 cursor-pointer'
                                  : phase?.status === 'complete'
                                    ? 'text-success/70 border border-success/20 hover:bg-success/5 cursor-pointer'
                                    : phase?.status === 'error'
                                      ? 'text-danger border border-danger/30 hover:bg-danger/5 cursor-pointer'
                                      : 'text-text-dim border border-void-surface/50'
                            }`}
                          >
                            <MvpPhaseIcon phase={phase} />
                            {mvpPhaseLabels[phaseNum]}
                          </button>
                        </div>
                      );
                    })}

                    {/* Overall MVP progress */}
                    <div className="ml-auto text-[10px] text-text-dim">
                      {selectedMvp.phases.filter(p => p.approved).length}/{selectedMvp.phases.length} phases approved
                    </div>
                  </div>

                  {/* MVP info bar — expandable files + units detail */}
                  <MvpInfoBar planId={planId!} mvp={selectedMvp} />

                  {/* Phase viewer for this MVP's active phase */}
                  <PhaseViewer
                    phase={mvpPhaseOutput}
                    phaseNumber={mvpActivePhase}
                    phaseType={mvpActivePhaseInfo?.phase_type ?? 'unknown'}
                    canExecute={canExecuteMvpPhase}
                    isExecuting={isMvpExecuting}
                    onExecute={handleMvpExecute}
                    onApprove={handleMvpApprove}
                    planId={planId}
                    mvpId={selectedMvpId}
                    sourceProjectId={sourceProjectId}
                  />
                </>
              ) : (
                <div className="flex-1 overflow-y-auto">
                  <BatchExecutionPanel
                    planId={planId!}
                    totalMvps={plan.mvps.length}
                    onMvpClick={(mvpId) => handleSelectMvp(mvpId)}
                    onBatchComplete={() => refreshPlan()}
                    initialBatchId={selectedBatchRunId}
                    onRunsLoaded={handleBatchRunsLoaded}
                    onBatchIdChange={setSelectedBatchRunId}
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}


// ── MVP Phase Icon ──

function MvpPhaseIcon({ phase }: { phase?: { status: string; approved: boolean } }) {
  if (!phase) {
    return <div className="h-2.5 w-2.5 rounded-full border border-void-surface bg-void-light" />;
  }
  if (phase.approved) {
    return <span className="material-symbols-outlined text-[14px] text-success">check_circle</span>;
  }
  if (phase.status === 'complete') {
    return <span className="material-symbols-outlined text-[14px] text-success/70">check</span>;
  }
  if (phase.status === 'running') {
    return <Loader2 className="h-3 w-3 animate-spin text-glow" />;
  }
  if (phase.status === 'error') {
    return <span className="material-symbols-outlined text-[14px] text-danger">error</span>;
  }
  return <div className="h-2.5 w-2.5 rounded-full border border-void-surface/50 bg-void-light/30" />;
}
