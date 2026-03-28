/**
 * Dashboard Page
 *
 * Compact table-based project dashboard with tabbed navigation between
 * Code Projects and Knowledge Base. Stats row, filter bar, sortable table,
 * and pagination.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Plus,
  Loader2,
  AlertCircle,
  Trash2,
  Clock,
  GitBranch,
  Upload,
  FolderOpen,
  BarChart3,
  BookOpen,
  Code2,
  Terminal,
  Search,
  Filter,
  List,
  LayoutGrid,
  MessageSquare,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  FolderArchive,
  CircleDot,
  Link2,
} from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import { useProjects } from '../hooks/useProjects.ts';
import * as api from '../services/api.ts';
import type { Project, MigrationPlan } from '../types/index.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatRelativeTime(dateStr: string | null): string {
  if (!dateStr) return '-';
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMs = now - then;
  if (diffMs < 0) return 'just now';

  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.floor(months / 12)}y ago`;
}

function extractLanguages(project: Project): string[] {
  const langs: string[] = [];
  if (project.languages && project.languages.length > 0) {
    return project.languages.slice(0, 5);
  }
  if (project.primary_language) {
    langs.push(project.primary_language);
  }
  return langs;
}

type SortKey = 'recent' | 'files_desc' | 'alpha';

function sortProjects(list: Project[], sortKey: SortKey): Project[] {
  const sorted = [...list];
  switch (sortKey) {
    case 'recent':
      return sorted.sort((a, b) => {
        const da = a.updated_at ?? a.created_at ?? '';
        const db = b.updated_at ?? b.created_at ?? '';
        return db.localeCompare(da);
      });
    case 'files_desc':
      return sorted.sort((a, b) => (b.file_count ?? 0) - (a.file_count ?? 0));
    case 'alpha':
      return sorted.sort((a, b) => a.name.localeCompare(b.name));
    default:
      return sorted;
  }
}

const ROWS_PER_PAGE = 15;

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export function Dashboard() {
  const { projects, isLoading, error, createProject, deleteProject } =
    useProjects();
  const navigate = useNavigate();

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [migrationPlans, setMigrationPlans] = useState<MigrationPlan[]>([]);

  // Tab state
  const [activeTab, setActiveTab] = useState<'code' | 'knowledge'>('code');

  // Filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [languageFilter, setLanguageFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortKey, setSortKey] = useState<SortKey>('recent');
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);

  // Fetch migration plans
  useEffect(() => {
    let cancelled = false;
    api
      .listMigrationPlans()
      .then((plans) => {
        if (!cancelled) setMigrationPlans(plans);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // Split projects into code vs knowledge
  const { codeProjects, knowledgeProjects } = useMemo(() => {
    const code: Project[] = [];
    const knowledge: Project[] = [];
    for (const p of projects) {
      if (p.project_type === 'knowledge') {
        knowledge.push(p);
      } else {
        code.push(p);
      }
    }
    return { codeProjects: code, knowledgeProjects: knowledge };
  }, [projects]);

  // Build a lookup map: project_id -> project name (for parent linking)
  const projectNameMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const p of projects) {
      map.set(p.project_id, p.name);
    }
    return map;
  }, [projects]);

  // Migration status map: project_id -> has active migration
  const activeMigrationMap = useMemo(() => {
    const map = new Map<string, boolean>();
    for (const plan of migrationPlans) {
      if (
        plan.source_project_id &&
        (plan.status === 'in_progress' || plan.status === 'draft')
      ) {
        map.set(plan.source_project_id, true);
      }
    }
    return map;
  }, [migrationPlans]);

  // Collect all unique languages for the filter dropdown
  const allLanguages = useMemo(() => {
    const langs = new Set<string>();
    for (const p of codeProjects) {
      if (p.primary_language) langs.add(p.primary_language);
      if (p.languages) p.languages.forEach((l) => langs.add(l));
    }
    return Array.from(langs).sort();
  }, [codeProjects]);

  // Compute stats
  const totalFiles = projects.reduce((sum, p) => sum + (p.file_count ?? 0), 0);
  const activeMigrations = migrationPlans.filter(
    (p) => p.status === 'draft' || p.status === 'in_progress',
  ).length;

  // Active tab projects with filters applied
  const filteredProjects = useMemo(() => {
    const base = activeTab === 'code' ? codeProjects : knowledgeProjects;
    let filtered = base;

    // Search filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter((p) => p.name.toLowerCase().includes(q));
    }

    // Language filter (code tab only)
    if (activeTab === 'code' && languageFilter !== 'all') {
      filtered = filtered.filter(
        (p) =>
          p.primary_language === languageFilter ||
          (p.languages && p.languages.includes(languageFilter)),
      );
    }

    // Status filter (code tab only)
    if (activeTab === 'code' && statusFilter !== 'all') {
      filtered = filtered.filter((p) => p.ast_status === statusFilter);
    }

    // Sort
    filtered = sortProjects(filtered, sortKey);

    return filtered;
  }, [
    activeTab,
    codeProjects,
    knowledgeProjects,
    searchQuery,
    languageFilter,
    statusFilter,
    sortKey,
  ]);

  // Paginated slice
  const totalPages = Math.max(1, Math.ceil(filteredProjects.length / ROWS_PER_PAGE));
  const safeCurrentPage = Math.min(currentPage, totalPages);
  const paginatedProjects = filteredProjects.slice(
    (safeCurrentPage - 1) * ROWS_PER_PAGE,
    safeCurrentPage * ROWS_PER_PAGE,
  );


  const handleDeleteProject = useCallback(
    async (projectId: string, projectName: string) => {
      if (!window.confirm(`Delete project "${projectName}"? This cannot be undone.`)) {
        return;
      }
      await deleteProject(projectId);
    },
    [deleteProject],
  );

  return (
    <Layout projects={projects} projectsLoading={isLoading}>
      <div className="flex-1 overflow-y-auto">
        {/* Header */}
        <div className="border-b border-void-surface px-6 py-5">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-text">Projects</h1>
              <p className="mt-1 text-sm text-text-muted">
                Upload a codebase, explore its structure, and chat with AI about
                your code.
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
                  label="Total Projects"
                  value={projects.length}
                  subtitle={`${codeProjects.length} code, ${knowledgeProjects.length} knowledge`}
                  accentColor="from-blue-500 to-cyan-500"
                />
                <StatCard
                  label="Total Files"
                  value={totalFiles.toLocaleString()}
                  subtitle="Across all projects"
                  accentColor="from-blue-500 to-indigo-500"
                />
                <StatCard
                  label="Active Migrations"
                  value={activeMigrations}
                  subtitle={activeMigrations > 0 ? 'In progress' : 'No active plans'}
                  accentColor="from-orange-500 to-amber-500"
                />
                <StatCard
                  label="Knowledge Docs"
                  value={knowledgeProjects.reduce((sum, p) => sum + (p.file_count ?? 0), 0)}
                  subtitle={`${knowledgeProjects.length} knowledge base${knowledgeProjects.length !== 1 ? 's' : ''}`}
                  accentColor="from-indigo-500 to-violet-500"
                />
              </div>

              {/* Tab Bar */}
              <div className="mb-4 flex items-center gap-6 border-b border-void-surface">
                <TabButton
                  active={activeTab === 'code'}
                  onClick={() => { setActiveTab('code'); setCurrentPage(1); }}
                  icon={<Terminal className="h-4 w-4" />}
                  label="Code Projects"
                  count={codeProjects.length}
                />
                <TabButton
                  active={activeTab === 'knowledge'}
                  onClick={() => { setActiveTab('knowledge'); setCurrentPage(1); }}
                  icon={<BookOpen className="h-4 w-4" />}
                  label="Knowledge Base"
                  count={knowledgeProjects.length}
                />
              </div>

              {/* Filter Bar */}
              <div className="mb-4 flex flex-wrap items-center gap-3">
                {/* Search */}
                <div className="relative flex-1 min-w-[200px] max-w-sm">
                  <Filter className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-dim" />
                  <input
                    type="text"
                    placeholder="Filter projects..."
                    value={searchQuery}
                    onChange={(e) => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                    className="h-9 w-full rounded-lg border border-[#2a3352] bg-[#0a0f1e] pl-9 pr-3 text-xs text-text placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50"
                  />
                </div>

                {/* Language dropdown (code tab only) */}
                {activeTab === 'code' && (
                  <select
                    value={languageFilter}
                    onChange={(e) => { setLanguageFilter(e.target.value); setCurrentPage(1); }}
                    className="h-9 rounded-lg border border-[#2a3352] bg-[#0a0f1e] px-3 text-xs text-text-muted focus:border-glow/50 focus:outline-none"
                  >
                    <option value="all">All Languages</option>
                    {allLanguages.map((lang) => (
                      <option key={lang} value={lang}>
                        {lang}
                      </option>
                    ))}
                  </select>
                )}

                {/* Status dropdown (code tab only) */}
                {activeTab === 'code' && (
                  <select
                    value={statusFilter}
                    onChange={(e) => { setStatusFilter(e.target.value); setCurrentPage(1); }}
                    className="h-9 rounded-lg border border-[#2a3352] bg-[#0a0f1e] px-3 text-xs text-text-muted focus:border-glow/50 focus:outline-none"
                  >
                    <option value="all">All Status</option>
                    <option value="complete">Complete</option>
                    <option value="parsing">Parsing</option>
                    <option value="pending">Pending</option>
                    <option value="error">Error</option>
                  </select>
                )}

                {/* Sort dropdown */}
                <select
                  value={sortKey}
                  onChange={(e) => { setSortKey(e.target.value as SortKey); setCurrentPage(1); }}
                  className="h-9 rounded-lg border border-[#2a3352] bg-[#0a0f1e] px-3 text-xs text-text-muted focus:border-glow/50 focus:outline-none"
                >
                  <option value="recent">Recent</option>
                  <option value="files_desc">Files (desc)</option>
                  <option value="alpha">Alphabetical</option>
                </select>

                {/* View toggle */}
                <div className="flex items-center rounded-lg border border-[#2a3352] bg-[#0a0f1e]">
                  <button
                    onClick={() => setViewMode('list')}
                    className={`flex items-center justify-center h-9 w-9 rounded-l-lg transition-colors ${
                      viewMode === 'list'
                        ? 'bg-[#1e2740] text-glow'
                        : 'text-text-dim hover:text-text'
                    }`}
                    aria-label="List view"
                  >
                    <List className="h-3.5 w-3.5" />
                  </button>
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`flex items-center justify-center h-9 w-9 rounded-r-lg transition-colors ${
                      viewMode === 'grid'
                        ? 'bg-[#1e2740] text-glow'
                        : 'text-text-dim hover:text-text'
                    }`}
                    aria-label="Grid view"
                  >
                    <LayoutGrid className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              {/* Data Table / Content */}
              {filteredProjects.length === 0 ? (
                <TabEmptyState
                  tab={activeTab}
                  hasFilter={
                    searchQuery.trim() !== '' ||
                    languageFilter !== 'all' ||
                    statusFilter !== 'all'
                  }
                  onCreateClick={() => setShowCreateModal(true)}
                  onClearFilters={() => {
                    setSearchQuery('');
                    setLanguageFilter('all');
                    setStatusFilter('all');
                  }}
                />
              ) : activeTab === 'code' ? (
                <CodeProjectTable
                  projects={paginatedProjects}
                  activeMigrationMap={activeMigrationMap}
                  onOpen={(id) => navigate(`/project/${id}`)}
                  onChat={(id) => navigate(`/project/${id}/chat`)}
                  onAnalytics={(id) => navigate(`/project/${id}/analytics`)}
                  onDelete={handleDeleteProject}
                />
              ) : (
                <KnowledgeProjectTable
                  projects={paginatedProjects}
                  projectNameMap={projectNameMap}
                  onOpen={(id) => navigate(`/project/${id}`)}
                  onChat={(id) => navigate(`/project/${id}/chat`)}
                  onDelete={handleDeleteProject}
                />
              )}

              {/* Pagination */}
              {filteredProjects.length > 0 && (
                <div className="mt-4 flex items-center justify-between text-xs text-text-dim">
                  <span>
                    Showing{' '}
                    {Math.min(
                      (safeCurrentPage - 1) * ROWS_PER_PAGE + 1,
                      filteredProjects.length,
                    )}
                    -{Math.min(safeCurrentPage * ROWS_PER_PAGE, filteredProjects.length)}{' '}
                    of {filteredProjects.length} project
                    {filteredProjects.length !== 1 ? 's' : ''}
                  </span>
                  {totalPages > 1 && (
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                        disabled={safeCurrentPage <= 1}
                        className="flex h-7 w-7 items-center justify-center rounded-md border border-[#2a3352] bg-[#0a0f1e] text-text-dim transition-colors hover:text-text disabled:opacity-30"
                        aria-label="Previous page"
                      >
                        <ChevronLeft className="h-3.5 w-3.5" />
                      </button>
                      {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                        (page) => (
                          <button
                            key={page}
                            onClick={() => setCurrentPage(page)}
                            className={`flex h-7 w-7 items-center justify-center rounded-md text-xs transition-colors ${
                              page === safeCurrentPage
                                ? 'bg-glow text-white'
                                : 'border border-[#2a3352] bg-[#0a0f1e] text-text-dim hover:text-text'
                            }`}
                          >
                            {page}
                          </button>
                        ),
                      )}
                      <button
                        onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                        disabled={safeCurrentPage >= totalPages}
                        className="flex h-7 w-7 items-center justify-center rounded-md border border-[#2a3352] bg-[#0a0f1e] text-text-dim transition-colors hover:text-text disabled:opacity-30"
                        aria-label="Next page"
                      >
                        <ChevronRight className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              )}
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
  label,
  value,
  subtitle,
  accentColor,
}: {
  label: string;
  value: string | number;
  subtitle: string;
  accentColor: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-xl bg-[#131b2e] p-5">
      {/* Gradient accent line at top */}
      <div
        className={`absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r ${accentColor}`}
      />
      <span className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
        {label}
      </span>
      <p className="mt-2 text-2xl font-semibold text-text">{value}</p>
      <p className="mt-1 text-[11px] text-text-dim">{subtitle}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TabButton
// ---------------------------------------------------------------------------

function TabButton({
  active,
  onClick,
  icon,
  label,
  count,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  count: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 border-b-2 px-1 pb-3 text-sm font-medium transition-colors ${
        active
          ? 'border-glow text-glow'
          : 'border-transparent text-text-dim hover:text-text-muted'
      }`}
    >
      {icon}
      <span>{label}</span>
      <span
        className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
          active
            ? 'bg-glow/15 text-glow'
            : 'bg-void-surface/60 text-text-dim'
        }`}
      >
        {count}
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Code Project Table
// ---------------------------------------------------------------------------

const SOURCE_ICONS: Record<string, { icon: typeof Upload; label: string }> = {
  zip: { icon: FolderArchive, label: 'Zip' },
  git: { icon: GitBranch, label: 'Git' },
  local: { icon: FolderOpen, label: 'Local' },
};

function CodeProjectTable({
  projects,
  activeMigrationMap,
  onOpen,
  onChat,
  onAnalytics,
  onDelete,
}: {
  projects: Project[];
  activeMigrationMap: Map<string, boolean>;
  onOpen: (id: string) => void;
  onChat: (id: string) => void;
  onAnalytics: (id: string) => void;
  onDelete: (id: string, name: string) => void;
}) {
  return (
    <div className="overflow-hidden rounded-lg border border-void-surface bg-[#131b2e]">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-void-surface text-[10px] font-semibold uppercase tracking-wider text-text-dim">
            <th className="px-4 py-3">Project</th>
            <th className="px-4 py-3 w-[70px]">Files</th>
            <th className="px-4 py-3 w-[70px]">Source</th>
            <th className="px-4 py-3 w-[120px]">AST Status</th>
            <th className="px-4 py-3 w-[120px]">Migration</th>
            <th className="px-4 py-3 w-[120px]">Last Updated</th>
            <th className="px-4 py-3 w-[120px] text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {projects.map((project) => {
            const languages = extractLanguages(project);
            const sourceCfg =
              SOURCE_ICONS[project.source_type] ?? SOURCE_ICONS.zip;
            const SourceIcon = sourceCfg.icon;
            const hasMigration = activeMigrationMap.get(project.project_id);

            return (
              <tr
                key={project.project_id}
                className="group h-12 border-b border-void-surface/50 transition-colors hover:bg-[#1e2740] cursor-pointer"
                onClick={() => onOpen(project.project_id)}
              >
                {/* PROJECT */}
                <td className="px-4 py-2.5">
                  <div className="flex items-center gap-3">
                    <div className="min-w-0">
                      <span className="block truncate text-sm font-medium text-text">
                        {project.name}
                      </span>
                      {languages.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {languages.map((lang) => (
                            <span
                              key={lang}
                              className="rounded bg-[#1e2740] px-1.5 py-0.5 text-[10px] font-bold uppercase text-text-dim"
                            >
                              {lang}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </td>

                {/* FILES */}
                <td className="px-4 py-2.5 text-text-muted tabular-nums">
                  {project.file_count ?? 0}
                </td>

                {/* SOURCE */}
                <td className="px-4 py-2.5">
                  <div
                    className="flex items-center gap-1.5 text-text-dim"
                    title={sourceCfg.label}
                  >
                    <SourceIcon className="h-3.5 w-3.5" />
                    <span className="text-[10px] uppercase">
                      {sourceCfg.label}
                    </span>
                  </div>
                </td>

                {/* AST STATUS */}
                <td className="px-4 py-2.5">
                  <AstStatusBadge status={project.ast_status} />
                </td>

                {/* MIGRATION */}
                <td className="px-4 py-2.5">
                  {hasMigration ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-glow/10 px-2.5 py-0.5 text-[10px] font-semibold text-glow">
                      <CircleDot className="h-2.5 w-2.5" />
                      In Progress
                    </span>
                  ) : (
                    <span className="text-text-dim">&mdash;</span>
                  )}
                </td>

                {/* LAST UPDATED */}
                <td className="px-4 py-2.5 text-text-dim">
                  <div className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatRelativeTime(project.updated_at ?? project.created_at)}
                  </div>
                </td>

                {/* ACTIONS */}
                <td className="px-4 py-2.5 text-right">
                  <div className="flex items-center justify-end gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    <ActionButton
                      icon={<ExternalLink className="h-3.5 w-3.5" />}
                      label="Open"
                      onClick={(e) => {
                        e.stopPropagation();
                        onOpen(project.project_id);
                      }}
                    />
                    <ActionButton
                      icon={<MessageSquare className="h-3.5 w-3.5" />}
                      label="Chat"
                      onClick={(e) => {
                        e.stopPropagation();
                        onChat(project.project_id);
                      }}
                    />
                    <ActionButton
                      icon={<BarChart3 className="h-3.5 w-3.5" />}
                      label="Analytics"
                      onClick={(e) => {
                        e.stopPropagation();
                        onAnalytics(project.project_id);
                      }}
                    />
                    <ActionButton
                      icon={<Trash2 className="h-3.5 w-3.5" />}
                      label="Delete"
                      variant="danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(project.project_id, project.name);
                      }}
                    />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Knowledge Project Table
// ---------------------------------------------------------------------------

function KnowledgeProjectTable({
  projects,
  projectNameMap,
  onOpen,
  onChat,
  onDelete,
}: {
  projects: Project[];
  projectNameMap: Map<string, string>;
  onOpen: (id: string) => void;
  onChat: (id: string) => void;
  onDelete: (id: string, name: string) => void;
}) {
  return (
    <div className="overflow-hidden rounded-lg border border-void-surface bg-[#131b2e]">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-void-surface text-[10px] font-semibold uppercase tracking-wider text-text-dim">
            <th className="px-4 py-3">Project</th>
            <th className="px-4 py-3 w-[80px]">Docs</th>
            <th className="px-4 py-3 w-[200px]">Linked Project</th>
            <th className="px-4 py-3 w-[120px]">Last Updated</th>
            <th className="px-4 py-3 w-[100px] text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {projects.map((project) => {
            const parentName = project.parent_project_id
              ? projectNameMap.get(project.parent_project_id) ?? null
              : null;

            return (
              <tr
                key={project.project_id}
                className="group h-12 border-b border-void-surface/50 transition-colors hover:bg-[#1e2740] cursor-pointer"
                onClick={() => onOpen(project.project_id)}
              >
                {/* PROJECT */}
                <td className="px-4 py-2.5">
                  <div className="flex items-center gap-2.5">
                    <BookOpen className="h-4 w-4 shrink-0 text-indigo-400" />
                    <span className="truncate text-sm font-medium text-text">
                      {project.name}
                    </span>
                  </div>
                </td>

                {/* DOCS */}
                <td className="px-4 py-2.5 text-text-muted tabular-nums">
                  {project.file_count ?? 0}
                </td>

                {/* LINKED PROJECT */}
                <td className="px-4 py-2.5">
                  {parentName ? (
                    <div className="flex items-center gap-1.5 text-indigo-400/80">
                      <Link2 className="h-3 w-3" />
                      <span className="truncate text-[11px]">{parentName}</span>
                    </div>
                  ) : (
                    <span className="text-text-dim">&mdash;</span>
                  )}
                </td>

                {/* LAST UPDATED */}
                <td className="px-4 py-2.5 text-text-dim">
                  <div className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatRelativeTime(project.updated_at ?? project.created_at)}
                  </div>
                </td>

                {/* ACTIONS */}
                <td className="px-4 py-2.5 text-right">
                  <div className="flex items-center justify-end gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    <ActionButton
                      icon={<ExternalLink className="h-3.5 w-3.5" />}
                      label="Open"
                      onClick={(e) => {
                        e.stopPropagation();
                        onOpen(project.project_id);
                      }}
                    />
                    <ActionButton
                      icon={<MessageSquare className="h-3.5 w-3.5" />}
                      label="Chat"
                      onClick={(e) => {
                        e.stopPropagation();
                        onChat(project.project_id);
                      }}
                    />
                    <ActionButton
                      icon={<Trash2 className="h-3.5 w-3.5" />}
                      label="Delete"
                      variant="danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(project.project_id, project.name);
                      }}
                    />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// AstStatusBadge
// ---------------------------------------------------------------------------

function AstStatusBadge({ status }: { status: Project['ast_status'] }) {
  const config: Record<
    string,
    { dotColor: string; textColor: string; label: string }
  > = {
    complete: {
      dotColor: 'bg-emerald-400',
      textColor: 'text-emerald-400',
      label: 'Complete',
    },
    parsing: {
      dotColor: 'bg-amber-400',
      textColor: 'text-amber-400',
      label: 'Parsing',
    },
    pending: {
      dotColor: 'bg-text-dim',
      textColor: 'text-text-dim',
      label: 'Pending',
    },
    error: {
      dotColor: 'bg-red-400',
      textColor: 'text-red-400',
      label: 'Error',
    },
  };

  const cfg = config[status] ?? config.pending;

  return (
    <div className={`flex items-center gap-1.5 ${cfg.textColor}`}>
      <span className={`inline-block h-1.5 w-1.5 rounded-full ${cfg.dotColor}`} />
      <span className="text-[11px] font-medium">{cfg.label}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ActionButton
// ---------------------------------------------------------------------------

function ActionButton({
  icon,
  label,
  variant = 'default',
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  variant?: 'default' | 'danger';
  onClick: (e: React.MouseEvent) => void;
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      className={`flex h-7 w-7 items-center justify-center rounded-md transition-colors ${
        variant === 'danger'
          ? 'text-text-dim hover:bg-red-500/10 hover:text-red-400'
          : 'text-text-dim hover:bg-void-surface hover:text-text'
      }`}
      aria-label={label}
    >
      {icon}
    </button>
  );
}

// ---------------------------------------------------------------------------
// EmptyState (no projects at all)
// ---------------------------------------------------------------------------

function EmptyState({ onCreateClick }: { onCreateClick: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-[#131b2e]">
        <Code2 className="h-8 w-8 text-text-dim" />
      </div>
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
// TabEmptyState (tab has no projects or filters returned nothing)
// ---------------------------------------------------------------------------

function TabEmptyState({
  tab,
  hasFilter,
  onCreateClick,
  onClearFilters,
}: {
  tab: 'code' | 'knowledge';
  hasFilter: boolean;
  onCreateClick: () => void;
  onClearFilters: () => void;
}) {
  if (hasFilter) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-void-surface/60 py-16">
        <Search className="h-8 w-8 text-text-dim" />
        <h3 className="mt-3 text-sm font-medium text-text-muted">
          No matching projects
        </h3>
        <p className="mt-1 text-xs text-text-dim">
          Try adjusting your filters or search query.
        </p>
        <button
          onClick={onClearFilters}
          className="mt-4 rounded-lg border border-void-surface px-4 py-2 text-xs text-text-muted transition-colors hover:bg-void-surface hover:text-text"
        >
          Clear Filters
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-void-surface/60 py-16">
      {tab === 'code' ? (
        <Code2 className="h-8 w-8 text-text-dim" />
      ) : (
        <BookOpen className="h-8 w-8 text-text-dim" />
      )}
      <h3 className="mt-3 text-sm font-medium text-text-muted">
        {tab === 'code'
          ? 'No code projects yet'
          : 'No knowledge base projects yet'}
      </h3>
      <p className="mt-1 text-xs text-text-dim">
        {tab === 'code'
          ? 'Create a project to upload and analyze a codebase.'
          : 'Create a knowledge project to upload documents and chat with AI.'}
      </p>
      <button
        onClick={onCreateClick}
        className="mt-4 flex items-center gap-2 rounded-lg bg-glow px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-glow-dim"
      >
        <Plus className="h-3.5 w-3.5" />
        Create Project
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CreateProjectModal (preserved from original)
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
  const [projectType, setProjectType] = useState<'code' | 'knowledge'>('code');
  const [isCreating, setIsCreating] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!name.trim()) return;
      setIsCreating(true);
      try {
        // Use the API directly to pass project_type
        const project = await api.createProject({
          name: name.trim(),
          description: description.trim() || undefined,
          project_type: projectType,
        });
        onClose();
        navigate(`/project/${project.project_id}`);
      } catch {
        // Fall back to the hook-based create (which refreshes the list)
        const project = await onCreate(
          name.trim(),
          description.trim() || undefined,
        );
        if (project) {
          onClose();
          navigate(`/project/${project.project_id}`);
        }
      } finally {
        setIsCreating(false);
      }
    },
    [name, description, projectType, onCreate, onClose, navigate],
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md rounded-xl border border-void-surface bg-void p-6 shadow-2xl">
        <h2 className="text-lg font-semibold text-text">New Project</h2>
        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          {/* Project Type Selector */}
          <div>
            <label className="block text-sm font-medium text-text-muted mb-2">
              Project Type
            </label>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => setProjectType('code')}
                className={`flex flex-1 items-center gap-2.5 rounded-lg border px-4 py-3 text-left transition-colors ${
                  projectType === 'code'
                    ? 'border-glow bg-glow/10 text-text'
                    : 'border-void-surface bg-void-light text-text-muted hover:border-void-surface hover:bg-void-light/80'
                }`}
              >
                <Code2
                  className={`h-5 w-5 ${projectType === 'code' ? 'text-glow' : 'text-text-dim'}`}
                />
                <div>
                  <p className="text-sm font-medium">Code</p>
                  <p className="text-[11px] text-text-dim">
                    Upload and analyze source code
                  </p>
                </div>
              </button>
              <button
                type="button"
                onClick={() => setProjectType('knowledge')}
                className={`flex flex-1 items-center gap-2.5 rounded-lg border px-4 py-3 text-left transition-colors ${
                  projectType === 'knowledge'
                    ? 'border-indigo-500 bg-indigo-500/10 text-text'
                    : 'border-void-surface bg-void-light text-text-muted hover:border-void-surface hover:bg-void-light/80'
                }`}
              >
                <BookOpen
                  className={`h-5 w-5 ${projectType === 'knowledge' ? 'text-indigo-400' : 'text-text-dim'}`}
                />
                <div>
                  <p className="text-sm font-medium">Knowledge</p>
                  <p className="text-[11px] text-text-dim">
                    Upload docs and chat with AI
                  </p>
                </div>
              </button>
            </div>
          </div>

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
              placeholder={
                projectType === 'knowledge'
                  ? 'e.g. company-docs'
                  : 'e.g. my-react-app'
              }
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
              placeholder={
                projectType === 'knowledge'
                  ? 'What kind of documents will this hold?'
                  : 'Brief description of the codebase'
              }
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
              {isCreating && (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              )}
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
