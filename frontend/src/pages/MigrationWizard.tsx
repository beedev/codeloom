/**
 * MigrationWizard — main page for the MVP-centric migration pipeline.
 *
 * URL: /migration/:planId
 *
 * Version-aware (plan.pipeline_version):
 *   V1 (6-phase): Phase 1 Discovery → Phase 2 Architecture → per-MVP: Analyze, Design, Transform, Test
 *   V2 (4-phase): Phase 1 Architecture → Phase 2 Discovery → per-MVP: Transform
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
import { Loader2, Trash2, ArrowLeft, ArrowRight, Lock, Terminal, Save } from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import { PhaseTimeline } from '../components/migration/PhaseTimeline.tsx';
import { PhaseViewer } from '../components/migration/PhaseViewer.tsx';
import { PlanCreator } from '../components/migration/PlanCreator.tsx';
import { MvpDiscoveryPanel } from '../components/migration/MvpDiscoveryPanel.tsx';
import { MvpTimeline } from '../components/migration/MvpTimeline.tsx';
import { MvpInfoBar } from '../components/migration/MvpInfoBar.tsx';
import { AssetInventory } from '../components/migration/AssetInventory.tsx';
import { DraftBriefEditor } from '../components/migration/DraftBriefEditor.tsx';
import { BatchExecutionPanel } from '../components/migration/BatchExecutionPanel.tsx';
import { AccuracyPanel } from '../components/migration/AccuracyPanel.tsx';
import * as api from '../services/api.ts';
import type { BatchStatus } from '../services/api.ts';
import type { MigrationPlan, MigrationPhaseOutput } from '../types/index.ts';

// Version-aware per-MVP phase constants
const MVP_PHASE_NUMBERS_V1 = [3, 4, 5, 6] as const;
const MVP_PHASE_NUMBERS_V2 = [3] as const;

const MVP_PHASE_LABELS_V1: Record<number, string> = {
  3: 'Analyze',
  4: 'Design',
  5: 'Transform',
  6: 'Test',
};
const MVP_PHASE_LABELS_V2: Record<number, string> = {
  3: 'Transform',
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
  // ── MVP-level phase state ──
  const [selectedMvpId, setSelectedMvpId] = useState<number | null>(null);
  const [mvpActivePhase, setMvpActivePhase] = useState<number>(3);
  const [mvpPhaseOutput, setMvpPhaseOutput] = useState<MigrationPhaseOutput | null>(null);

  // ── Framework doc enrichment ──
  const [isEnriching, setIsEnriching] = useState(false);
  const [enrichResult, setEnrichResult] = useState<{ enriched: string[]; failed: string[] } | null>(null);

  // ── View mode ──
  // 'plan' = plan-level phases (1, 2)
  // 'mvp' = per-MVP phases (3-6) with MVP sidebar; no MVP selected → batch panel
  const [viewMode, setViewMode] = useState<'plan' | 'mvp'>('plan');

  // ── Batch run state (controls sidebar + panel when no MVP selected) ──
  const [, setBatchRuns] = useState<BatchStatus[]>([]);
  const [selectedBatchRunId, setSelectedBatchRunId] = useState<string | null>(null);
  const batchInitRef = useRef(false);

  // ── Version-aware constants ──
  const version = plan?.pipeline_version ?? 1;
  const mvpPhaseNumbers = useMemo(() => getMvpPhaseNumbers(version), [version]);
  const mvpPhaseLabels = useMemo(() => getMvpPhaseLabels(version), [version]);
  // Phase types to show in MvpTimeline (lowercase label values, e.g. ['transform'] for V2)
  const mvpPhaseTypes = useMemo(
    () => Object.values(mvpPhaseLabels).map(l => l.toLowerCase()),
    [mvpPhaseLabels]
  );

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

  // Plan is fully complete when all MVPs are migrated or plan status is 'complete'
  const planComplete =
    plan?.status === 'complete' ||
    (hasMvps && (plan?.mvps?.every(m => m.status === 'migrated') ?? false));

  const selectedMvp = plan?.mvps?.find(m => m.mvp_id === selectedMvpId) ?? null;

  // CLI-created draft plan waiting for business context via Web UI
  const isCliDraft =
    plan != null &&
    plan.status === 'draft' &&
    plan.discovery_metadata != null &&
    plan.discovery_metadata.orchestrator === 'claude_code_cli';

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
          // Only auto-select an MVP if one is actively running; otherwise land on Execution Overview
          const runningMvp = p.mvps?.find(m =>
            m.phases.some(ph => ph.status === 'running')
          );
          setViewMode('mvp');
          if (runningMvp) {
            setSelectedMvpId(runningMvp.mvp_id);
          }
          // else: no auto-select → Execution Overview shows with all MVPs + accuracy
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
    // Only consider phases that belong to this pipeline version
    const mvpPhases = selectedMvp.phases.filter(p =>
      (mvpPhaseNumbers as readonly number[]).includes(p.phase_number)
    );
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
  }, [selectedMvpId, selectedMvp, mvpPhaseNumbers]);

  // ── Handlers: Plan creation ──
  const handleCreatePlan = useCallback(async (data: Parameters<typeof api.createMigrationPlan>[0]) => {
    setIsCreating(true);
    setError(null);
    try {
      const newPlan = await api.createMigrationPlan(data);
      // Copy the CLI migrate command so the user can paste it immediately
      const cmd = `claude /migrate --project ${newPlan.plan_id}`;
      navigator.clipboard.writeText(cmd).catch(() => {});
      // Navigate to the plan so Asset Inventory shows before going to plans list
      navigate(`/migration/${newPlan.plan_id}`, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create plan');
    } finally {
      setIsCreating(false);
    }
  }, [navigate]);

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

        {/* CLI Draft: business context editor — shown when plan was created from CLI as draft */}
        {isCliDraft && plan && (
          <div className="flex-1 overflow-y-auto p-6">
            <DraftBriefEditor planId={planId!} plan={plan} onSaved={refreshPlan} />
          </div>
        )}

        {/* Asset Inventory step — shown before discovery when no strategies are saved */}
        {needsAssetInventory && !isCliDraft && plan && viewMode === 'plan' && (
          <div className="flex-1 overflow-y-auto p-6">
            <AssetInventory
              planId={planId!}
              plan={plan}
              onConfirm={async () => {
                await refreshPlan();
                navigate('/migrations', { replace: true });
              }}
            />
          </div>
        )}

        {/* View mode tab bar — shown when migration is underway (replaces old banner) */}
        {!isNewPlan && plan && !needsAssetInventory && !isCliDraft && migrationUnderway && (
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
        {!isNewPlan && plan && !needsAssetInventory && !isCliDraft && viewMode === 'plan' && (
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
            {/* Left sidebar: MVP Timeline — always visible in MVP execution view */}
            <div className="w-60 shrink-0 overflow-y-auto border-r border-void-surface">
              <MvpTimeline
                mvps={plan.mvps}
                selectedMvpId={selectedMvpId}
                onSelectMvp={handleSelectMvp}
                phaseTypes={mvpPhaseTypes}
              />
            </div>

            {/* Center: Phase execution for selected MVP */}
            <div className="flex flex-1 flex-col overflow-y-auto">
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
                    planId={planId}
                    mvpId={selectedMvpId}
                    sourceProjectId={sourceProjectId}
                  />
                </>
              ) : planComplete ? (
                /* Completed plan — show MVP status grid (Execution Overview) */
                <div className="p-6 space-y-4">
                  <h2 className="text-sm font-semibold text-text">Execution Overview</h2>
                  <div className="grid grid-cols-1 gap-3">
                    {plan.mvps.map((mvp) => {
                      const allPhasesApproved = mvp.phases.length > 0 && mvp.phases.every(ph => ph.approved);
                      const statusColor =
                        mvp.status === 'migrated' ? 'text-success border-success/30 bg-success/10' :
                        mvp.status === 'in_progress' ? 'text-glow border-glow/30 bg-glow/10' :
                        mvp.status === 'refined' ? 'text-nebula-bright border-nebula/30 bg-nebula/10' :
                        'text-text-dim border-void-surface bg-void-surface/50';
                      return (
                        <button
                          key={mvp.mvp_id}
                          onClick={() => handleSelectMvp(mvp.mvp_id)}
                          className="flex items-center gap-4 rounded-xl border border-void-surface bg-void-light/30 px-5 py-3.5 text-left transition-colors hover:border-glow/20 hover:bg-void-light/50"
                        >
                          <span className="inline-flex items-center rounded-md bg-glow/10 px-1.5 py-0.5 text-[10px] font-semibold text-glow shrink-0">
                            #{mvp.priority}
                          </span>
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium text-text truncate">{mvp.name}</div>
                            {mvp.description && (
                              <div className="text-xs text-text-dim truncate mt-0.5">{mvp.description}</div>
                            )}
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            {allPhasesApproved && (
                              <span className="material-symbols-outlined text-[16px] text-success">check_circle</span>
                            )}
                            <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium ${statusColor}`}>
                              {mvp.status.replace('_', ' ')}
                            </span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <BatchExecutionPanel
                  planId={planId!}
                  totalMvps={plan.mvps.length}
                  onMvpClick={(mvpId) => handleSelectMvp(mvpId)}
                  onBatchComplete={() => refreshPlan()}
                  initialBatchId={selectedBatchRunId}
                  onRunsLoaded={handleBatchRunsLoaded}
                  onBatchIdChange={setSelectedBatchRunId}
                />
              )}

              {/* Accuracy report — always visible below batch/phase panel */}
              <div className="border-t border-void-surface px-6 py-6">
                <AccuracyPanel planId={planId!} selectedMvpName={selectedMvp?.name} />
              </div>
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
