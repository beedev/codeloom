/**
 * Dashboard Page
 *
 * 4 stat cards, grid of project cards with Open/Chat buttons, and a "Create Project" action.
 */

import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Plus,
  Loader2,
  AlertCircle,
  Trash2,
  Hash,
  Clock,
  GitBranch,
  Upload,
  FolderOpen,
} from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import { useProjects } from '../hooks/useProjects.ts';
import * as api from '../services/api.ts';
import type { Project, MigrationPlan } from '../types/index.ts';

export function Dashboard() {
  const { projects, isLoading, error, createProject, deleteProject } =
    useProjects();
  const navigate = useNavigate();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [migrationPlans, setMigrationPlans] = useState<MigrationPlan[]>([]);

  // Fetch migration plans
  useEffect(() => {
    let cancelled = false;
    api.listMigrationPlans().then((plans) => {
      if (!cancelled) setMigrationPlans(plans);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // Compute stats
  const totalFiles = projects.reduce((sum, p) => sum + (p.file_count ?? 0), 0);
  const activeMigrations = migrationPlans.filter(
    (p) => p.status === 'draft' || p.status === 'in_progress',
  ).length;

  return (
    <Layout projects={projects} projectsLoading={isLoading}>
      <div className="flex-1 overflow-y-auto">
        {/* Header */}
        <div className="border-b border-void-surface px-6 py-5">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-text">Projects</h1>
              <p className="mt-1 text-sm text-text-muted">
                Upload a codebase, explore its structure, and chat with AI about your code.
              </p>
            </div>
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center gap-2 rounded-lg bg-glow px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-glow-dim"
            >
              <Plus className="h-4 w-4" />
              New Project
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
          ) : projects.length === 0 ? (
            <EmptyState onCreateClick={() => setShowCreateModal(true)} />
          ) : (
            <>
              {/* Stat Cards */}
              <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
                <StatCard
                  icon="folder_copy"
                  label="Total Projects"
                  value={projects.length}
                />
                <StatCard
                  icon="description"
                  label="Total Files"
                  value={totalFiles}
                />
                <StatCard
                  icon="swap_horiz"
                  label="Active Migrations"
                  value={activeMigrations}
                />
                <StatCard
                  icon="query_stats"
                  label="Code Queries"
                  value="-"
                />
              </div>

              {/* Project Grid */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {projects.map((project) => (
                  <ProjectCard
                    key={project.project_id}
                    project={project}
                    onOpen={() => navigate(`/project/${project.project_id}`)}
                    onChat={() => navigate(`/project/${project.project_id}/chat`)}
                    onDelete={() => deleteProject(project.project_id)}
                  />
                ))}
              </div>

              {/* Footer */}
              <p className="mt-6 text-center text-xs text-text-dim">
                Showing all {projects.length} active project{projects.length !== 1 ? 's' : ''}.
              </p>
            </>
          )}
        </div>
      </div>

      {/* Create Project Modal */}
      {showCreateModal && (
        <CreateProjectModal
          onClose={() => setShowCreateModal(false)}
          onCreate={createProject}
        />
      )}
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
// ProjectCard
// ---------------------------------------------------------------------------

function ProjectCard({
  project,
  onOpen,
  onChat,
  onDelete,
}: {
  project: Project;
  onOpen: () => void;
  onChat: () => void;
  onDelete: () => void;
}) {
  const statusColors: Record<string, string> = {
    pending: 'bg-void-surface/50 text-text-dim',
    parsing: 'bg-warning/10 text-warning',
    complete: 'bg-success/10 text-success',
    error: 'bg-danger/10 text-danger',
  };

  return (
    <div className="group rounded-lg border border-void-surface/50 bg-void-light/40 p-5 transition-all hover:border-void-surface hover:bg-void-light">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-[20px] text-glow">folder_copy</span>
          <h3 className="text-sm font-medium text-text">{project.name}</h3>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="rounded p-1 text-text-dim opacity-0 transition-opacity hover:bg-void-surface hover:text-danger group-hover:opacity-100"
          aria-label={`Delete project ${project.name}`}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      {project.description && (
        <p className="mt-2 line-clamp-2 text-xs text-text-muted">
          {project.description}
        </p>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-text-dim">
        {project.primary_language && (
          <span className="rounded bg-void-surface/50 px-2 py-0.5 text-text-muted">
            {project.primary_language}
          </span>
        )}
        <SourceBadge sourceType={project.source_type} />
        <span className="flex items-center gap-1">
          <Hash className="h-3 w-3" />
          {project.file_count} files
        </span>
        <span
          className={`rounded px-2 py-0.5 text-[10px] font-medium ${
            statusColors[project.ast_status] ?? statusColors.pending
          }`}
        >
          {project.ast_status}
        </span>
        {project.created_at && (
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {new Date(project.created_at).toLocaleDateString()}
          </span>
        )}
      </div>

      {/* Open + Chat buttons */}
      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={onOpen}
          className="rounded-lg bg-glow px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-glow-dim"
        >
          Open
        </button>
        <button
          onClick={onChat}
          className="rounded-lg border border-void-surface px-4 py-1.5 text-xs text-text-muted transition-colors hover:bg-void-surface hover:text-text"
        >
          Chat
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
      <span className="material-symbols-outlined text-[48px] text-void-surface">folder_copy</span>
      <h3 className="mt-4 text-sm font-medium text-text-muted">
        No projects yet
      </h3>
      <p className="mt-1 text-xs text-text-dim">
        Create a project and upload your codebase to get started.
      </p>
      <button
        onClick={onCreateClick}
        className="mt-6 flex items-center gap-2 rounded-lg bg-glow px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-glow-dim"
      >
        <Plus className="h-4 w-4" />
        Create Project
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SourceBadge
// ---------------------------------------------------------------------------

const SOURCE_CONFIG: Record<string, { icon: typeof Upload; label: string; color: string }> = {
  zip: { icon: Upload, label: 'Zip', color: 'bg-void-surface/50 text-text-dim' },
  git: { icon: GitBranch, label: 'Git', color: 'bg-nebula/10 text-nebula' },
  local: { icon: FolderOpen, label: 'Local', color: 'bg-warning/10 text-warning' },
};

function SourceBadge({ sourceType }: { sourceType: string }) {
  const cfg = SOURCE_CONFIG[sourceType] ?? SOURCE_CONFIG.zip;
  const Icon = cfg.icon;
  return (
    <span className={`flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium ${cfg.color}`}>
      <Icon className="h-2.5 w-2.5" />
      {cfg.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// CreateProjectModal
// ---------------------------------------------------------------------------

function CreateProjectModal({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (name: string, description?: string) => Promise<Project | null>;
}) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!name.trim()) return;
      setIsCreating(true);
      const project = await onCreate(name.trim(), description.trim() || undefined);
      setIsCreating(false);
      if (project) {
        onClose();
        navigate(`/project/${project.project_id}`);
      }
    },
    [name, description, onCreate, onClose, navigate],
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md rounded-xl border border-void-surface bg-void p-6 shadow-2xl">
        <h2 className="text-lg font-semibold text-text">New Project</h2>
        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          <div>
            <label
              htmlFor="project-name"
              className="block text-sm font-medium text-text-muted"
            >
              Project Name
            </label>
            <input
              id="project-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="mt-1 w-full rounded-lg border border-void-surface bg-void-light px-4 py-2.5 text-sm text-text placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50"
              placeholder="e.g. my-react-app"
              autoFocus
              disabled={isCreating}
            />
          </div>
          <div>
            <label
              htmlFor="project-desc"
              className="block text-sm font-medium text-text-muted"
            >
              Description (optional)
            </label>
            <textarea
              id="project-desc"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="mt-1 w-full resize-none rounded-lg border border-void-surface bg-void-light px-4 py-2.5 text-sm text-text placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50"
              placeholder="Brief description of the codebase"
              disabled={isCreating}
            />
          </div>
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg px-4 py-2 text-sm text-text-muted hover:text-text"
              disabled={isCreating}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim() || isCreating}
              className="flex items-center gap-2 rounded-lg bg-glow px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-glow-dim disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isCreating && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
