/**
 * Migration Plans List Page
 *
 * Shows all migration plans with status, source project, MVP count,
 * phase progress, and actions (Open, Delete). "New Migration" button
 * navigates to /migration/new.
 */

import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Plus,
  Loader2,
  AlertCircle,
  Trash2,
  Clock,
  ChevronRight,
} from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import * as api from '../services/api.ts';
import type { MigrationPlan, Project } from '../types/index.ts';

const STATUS_STYLES: Record<string, string> = {
  draft: 'bg-void-surface/50 text-text-dim',
  in_progress: 'bg-glow/10 text-glow',
  complete: 'bg-success/10 text-success',
  abandoned: 'bg-danger/10 text-danger',
};

const STATUS_LABELS: Record<string, string> = {
  draft: 'Draft',
  in_progress: 'In Progress',
  complete: 'Complete',
  abandoned: 'Abandoned',
};

export function MigrationPlans() {
  const navigate = useNavigate();
  const [plans, setPlans] = useState<MigrationPlan[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [planData, projectData] = await Promise.all([
        api.listMigrationPlans(),
        api.listProjects(),
      ]);
      setPlans(planData);
      setProjects(projectData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load migration plans');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDelete = useCallback(async (planId: string) => {
    try {
      await api.deleteMigrationPlan(planId);
      setPlans((prev) => prev.filter((p) => p.plan_id !== planId));
    } catch {
      // Silently fail — plan stays in list
    }
  }, []);

  const projectMap = new Map(projects.map((p) => [p.project_id, p]));

  // Stats
  const active = plans.filter((p) => p.status === 'draft' || p.status === 'in_progress').length;
  const completed = plans.filter((p) => p.status === 'complete').length;
  const totalMvps = plans.reduce((sum, p) => sum + (p.mvps?.length ?? 0), 0);

  return (
    <Layout>
      <div className="flex-1 overflow-y-auto">
        {/* Header */}
        <div className="border-b border-void-surface px-6 py-5">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-text">Migration Plans</h1>
              <p className="mt-1 text-sm text-text-muted">
                Manage code migration pipelines across your projects.
              </p>
            </div>
            <button
              onClick={() => navigate('/migration/new')}
              className="flex items-center gap-2 rounded-lg bg-glow px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-glow-dim"
            >
              <Plus className="h-4 w-4" />
              New Migration
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="h-6 w-6 animate-spin text-text-dim" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center gap-2 py-20 text-danger">
              <AlertCircle className="h-5 w-5" />
              <span className="text-sm">{error}</span>
            </div>
          ) : plans.length === 0 ? (
            <EmptyState onCreateClick={() => navigate('/migration/new')} />
          ) : (
            <>
              {/* Stat Cards */}
              <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
                <StatCard icon="swap_horiz" label="Total Plans" value={plans.length} />
                <StatCard icon="play_circle" label="Active" value={active} />
                <StatCard icon="check_circle" label="Completed" value={completed} />
                <StatCard icon="hub" label="Total MVPs" value={totalMvps} />
              </div>

              {/* Plan List */}
              <div className="space-y-3">
                {plans.map((plan) => (
                  <PlanCard
                    key={plan.plan_id}
                    plan={plan}
                    project={plan.source_project_id ? projectMap.get(plan.source_project_id) : undefined}
                    onOpen={() => navigate(`/migration/${plan.plan_id}`)}
                    onDelete={() => handleDelete(plan.plan_id)}
                  />
                ))}
              </div>

              <p className="mt-6 text-center text-xs text-text-dim">
                Showing all {plans.length} migration plan{plans.length !== 1 ? 's' : ''}.
              </p>
            </>
          )}
        </div>
      </div>
    </Layout>
  );
}

// ---------------------------------------------------------------------------
// StatCard
// ---------------------------------------------------------------------------

function StatCard({
  icon,
  label,
  value,
}: {
  icon: string;
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-xl border border-void-surface/50 bg-void-light/50 p-5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
          {label}
        </span>
        <span className="material-symbols-outlined text-[20px] text-text-dim">
          {icon}
        </span>
      </div>
      <p className="mt-2 text-2xl font-semibold text-text">{value}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PlanCard
// ---------------------------------------------------------------------------

function PlanCard({
  plan,
  project,
  onOpen,
  onDelete,
}: {
  plan: MigrationPlan;
  project?: Project;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const approvedPhases = plan.plan_phases.filter((p) => p.approved).length;
  const totalPhases = plan.plan_phases.length;
  const sourceLangs = plan.source_stack?.languages?.join(', ')
    || plan.source_stack?.primary_language
    || '';
  const targetLangs = plan.target_stack?.languages?.join(', ') ?? '';
  const targetFrameworks = plan.target_stack?.frameworks?.join(', ') ?? '';

  return (
    <div className="group rounded-lg border border-void-surface/50 bg-void-light/40 p-5 transition-all hover:border-void-surface hover:bg-void-light">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-[20px] text-glow">swap_horiz</span>
          <div>
            <h3 className="text-sm font-medium text-text">
              {project?.name ?? 'Unknown Project'}
            </h3>
            <p className="mt-0.5 line-clamp-1 text-xs text-text-muted">
              {plan.target_brief}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`rounded px-2 py-0.5 text-[10px] font-medium ${
              STATUS_STYLES[plan.status] ?? STATUS_STYLES.draft
            }`}
          >
            {STATUS_LABELS[plan.status] ?? plan.status}
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="rounded p-1 text-text-dim opacity-0 transition-opacity hover:bg-void-surface hover:text-danger group-hover:opacity-100"
            aria-label="Delete migration plan"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Details row */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-text-dim">
        {(sourceLangs || targetLangs) && (
          <span className="flex items-center gap-1.5">
            {sourceLangs && (
              <span className="rounded bg-void-surface/80 px-2 py-0.5 text-[10px] font-medium text-text-muted">
                {sourceLangs}
              </span>
            )}
            {sourceLangs && targetLangs && (
              <span className="text-text-dim">&rarr;</span>
            )}
            {targetLangs && (
              <span className="rounded bg-nebula/10 px-2 py-0.5 text-[10px] font-medium text-nebula">
                {targetLangs}
              </span>
            )}
          </span>
        )}
        {targetFrameworks && (
          <span className="rounded bg-void-surface/50 px-2 py-0.5 text-text-muted">
            {targetFrameworks}
          </span>
        )}
        <span className="flex items-center gap-1">
          <span className="material-symbols-outlined text-[14px]">hub</span>
          {plan.mvps?.length ?? 0} MVPs
        </span>
        <span className="flex items-center gap-1">
          <span className="material-symbols-outlined text-[14px]">task_alt</span>
          {approvedPhases}/{totalPhases} phases
        </span>
        {plan.created_at && (
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {new Date(plan.created_at).toLocaleDateString()}
          </span>
        )}
      </div>

      {/* Action buttons */}
      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={onOpen}
          className="flex items-center gap-1 rounded-lg bg-glow px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-glow-dim"
        >
          Open
          <ChevronRight className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// EmptyState
// ---------------------------------------------------------------------------

function EmptyState({ onCreateClick }: { onCreateClick: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <span className="material-symbols-outlined text-[48px] text-void-surface">swap_horiz</span>
      <h3 className="mt-4 text-sm font-medium text-text-muted">
        No migration plans yet
      </h3>
      <p className="mt-1 text-xs text-text-dim">
        Create a migration plan to start transforming your codebase.
      </p>
      <button
        onClick={onCreateClick}
        className="mt-6 flex items-center gap-2 rounded-lg bg-glow px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-glow-dim"
      >
        <Plus className="h-4 w-4" />
        New Migration
      </button>
    </div>
  );
}
