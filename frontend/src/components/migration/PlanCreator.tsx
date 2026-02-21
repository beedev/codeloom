/**
 * PlanCreator — form for creating a new migration plan.
 *
 * Collects: migration type, target description, target stack (languages, frameworks),
 * and optional constraints (timeline, team size, risk tolerance).
 */

import { useState, useEffect } from 'react';
import { ArrowUpCircle, GitBranch, Loader2, Rocket, RotateCcw } from 'lucide-react';
import * as api from '../../services/api.ts';

type MigrationType = 'version_upgrade' | 'framework_migration' | 'rewrite';

interface PlanCreatorProps {
  projectId: string;
  projectName: string;
  onCreatePlan: (data: {
    source_project_id: string;
    target_brief: string;
    target_stack: { languages: string[]; frameworks: string[] };
    constraints?: { timeline?: string; team_size?: number; risk_tolerance?: string };
    migration_type: MigrationType;
  }) => void;
  isCreating: boolean;
}

const MIGRATION_TYPES: { value: MigrationType; label: string; description: string; icon: typeof Rocket }[] = [
  {
    value: 'version_upgrade',
    label: 'Version Upgrade',
    description: 'Same framework, newer version (e.g., Java 11 to 21)',
    icon: ArrowUpCircle,
  },
  {
    value: 'framework_migration',
    label: 'Framework Migration',
    description: 'Switch frameworks (e.g., Flask to FastAPI)',
    icon: GitBranch,
  },
  {
    value: 'rewrite',
    label: 'Full Rewrite',
    description: 'Start fresh with a new architecture',
    icon: RotateCcw,
  },
];

const PLACEHOLDERS: Record<MigrationType, { brief: string }> = {
  version_upgrade: {
    brief: 'e.g., Upgrade from Java 11 to Java 21',
  },
  framework_migration: {
    brief: 'e.g., Migrate from Flask to FastAPI with async support',
  },
  rewrite: {
    brief: 'e.g., Rewrite legacy monolith as microservices in Go',
  },
};

export function PlanCreator({ projectId, projectName, onCreatePlan, isCreating }: PlanCreatorProps) {
  const [migrationType, setMigrationType] = useState<MigrationType>('framework_migration');
  const [targetBrief, setTargetBrief] = useState('');
  const [timeline, setTimeline] = useState('');
  const [teamSize, setTeamSize] = useState('');
  const [riskTolerance, setRiskTolerance] = useState('medium');

  // Project selector state — used when no projectId is provided via URL
  const [projects, setProjects] = useState<{ project_id: string; name: string }[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState(projectId);
  const needsProjectSelector = !projectId;

  useEffect(() => {
    if (!needsProjectSelector) return;
    api.listProjects().then(setProjects).catch(() => {});
  }, [needsProjectSelector]);

  const effectiveProjectId = needsProjectSelector ? selectedProjectId : projectId;
  const effectiveProjectName = needsProjectSelector
    ? (projects.find(p => p.project_id === selectedProjectId)?.name ?? 'your project')
    : projectName;

  const placeholders = PLACEHOLDERS[migrationType];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!targetBrief.trim() || !effectiveProjectId) return;

    const constraints: Record<string, string | number> = {};
    if (timeline.trim()) constraints.timeline = timeline.trim();
    if (teamSize.trim()) constraints.team_size = parseInt(teamSize, 10);
    if (riskTolerance) constraints.risk_tolerance = riskTolerance;

    onCreatePlan({
      source_project_id: effectiveProjectId,
      target_brief: targetBrief.trim(),
      target_stack: { languages: [], frameworks: [] },
      constraints: Object.keys(constraints).length > 0 ? constraints : undefined,
      migration_type: migrationType,
    });
  };

  return (
    <div className="mx-auto max-w-xl">
      <div className="text-center">
        <Rocket className="mx-auto h-10 w-10 text-glow" />
        <h2 className="mt-3 text-lg font-semibold text-text">Start Migration</h2>
        <p className="mt-1 text-sm text-text-muted">
          Migrate <span className="text-text">{effectiveProjectName}</span>
        </p>
      </div>

      <form onSubmit={handleSubmit} className="mt-8 space-y-5">
        {/* Project selector — shown when no project ID in URL */}
        {needsProjectSelector && (
          <div>
            <label className="block text-xs font-medium text-text-muted mb-1">
              Source Project
            </label>
            {projects.length === 0 ? (
              <p className="text-xs text-text-dim">No projects found. Upload a codebase first.</p>
            ) : (
              <select
                value={selectedProjectId}
                onChange={e => setSelectedProjectId(e.target.value)}
                className="w-full rounded-md border border-void-surface bg-void-light px-3 py-2 text-sm text-text focus:border-glow/50 focus:outline-none"
                required
              >
                <option value="">Select a project...</option>
                {projects.map(p => (
                  <option key={p.project_id} value={p.project_id}>{p.name}</option>
                ))}
              </select>
            )}
          </div>
        )}

        {/* Migration type selector */}
        <div>
          <label className="block text-xs font-medium text-text-muted mb-2">
            Migration Type
          </label>
          <div className="grid grid-cols-3 gap-2">
            {MIGRATION_TYPES.map(({ value, label, description, icon: Icon }) => (
              <button
                key={value}
                type="button"
                onClick={() => setMigrationType(value)}
                className={`flex flex-col items-center gap-1.5 rounded-md border px-3 py-3 text-center transition-colors ${
                  migrationType === value
                    ? 'border-glow/60 bg-glow/10 text-text'
                    : 'border-void-surface bg-void-light text-text-muted hover:border-void-surface/80 hover:text-text-dim'
                }`}
              >
                <Icon className={`h-4 w-4 ${migrationType === value ? 'text-glow' : ''}`} />
                <span className="text-xs font-medium">{label}</span>
                <span className="text-[10px] leading-tight text-text-dim">{description}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Target description */}
        <div>
          <label className="block text-xs font-medium text-text-muted">
            Target Description
          </label>
          <textarea
            value={targetBrief}
            onChange={e => setTargetBrief(e.target.value)}
            placeholder={placeholders.brief}
            rows={3}
            className="mt-1 w-full rounded-md border border-void-surface bg-void-light px-3 py-2 text-sm text-text placeholder:text-text-dim focus:border-glow/50 focus:outline-none"
            required
          />
        </div>

        {/* Constraints (collapsible) */}
        <details className="rounded-md border border-void-surface/50">
          <summary className="cursor-pointer px-3 py-2 text-xs font-medium text-text-muted hover:text-text-dim">
            Constraints (optional)
          </summary>
          <div className="grid grid-cols-3 gap-3 p-3 pt-1">
            <div>
              <label className="block text-[10px] font-medium text-text-dim">Timeline</label>
              <input
                type="text"
                value={timeline}
                onChange={e => setTimeline(e.target.value)}
                placeholder="e.g., 2 weeks"
                className="mt-1 w-full rounded border border-void-surface bg-void-light px-2 py-1.5 text-xs text-text placeholder:text-text-dim focus:border-glow/50 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-[10px] font-medium text-text-dim">Team Size</label>
              <input
                type="number"
                value={teamSize}
                onChange={e => setTeamSize(e.target.value)}
                placeholder="e.g., 3"
                min="1"
                className="mt-1 w-full rounded border border-void-surface bg-void-light px-2 py-1.5 text-xs text-text placeholder:text-text-dim focus:border-glow/50 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-[10px] font-medium text-text-dim">Risk Tolerance</label>
              <select
                value={riskTolerance}
                onChange={e => setRiskTolerance(e.target.value)}
                className="mt-1 w-full rounded border border-void-surface bg-void-light px-2 py-1.5 text-xs text-text focus:border-glow/50 focus:outline-none"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>
        </details>

        <button
          type="submit"
          disabled={isCreating || !targetBrief.trim() || !effectiveProjectId}
          className="flex w-full items-center justify-center gap-2 rounded-md bg-glow px-4 py-2.5 text-sm font-medium text-white hover:bg-glow-dim disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isCreating ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Creating plan...
            </>
          ) : (
            <>
              <Rocket className="h-4 w-4" />
              Create Migration Plan
            </>
          )}
        </button>
      </form>
    </div>
  );
}
