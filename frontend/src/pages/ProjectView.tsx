/**
 * ProjectView Page
 *
 * 3-panel layout: File tree | Code viewer | Structure panel
 * Top bar with breadcrumbs and ASG status badge
 * Bottom status bar with file info
 */

import { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Loader2,
  AlertCircle,
  MessageSquareCode,
  Upload as UploadIcon,
  Code2,
  Network,
  ArrowRightLeft,
  Brain,
} from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import { FileTree } from '../components/FileTree.tsx';
import { CodeViewer } from '../components/CodeViewer.tsx';
import { GraphViewer } from '../components/GraphViewer.tsx';
import { ProjectUpload } from '../components/ProjectUpload.tsx';
import { UnderstandingPanel } from '../components/understanding/UnderstandingPanel.tsx';
import { useProjects } from '../hooks/useProjects.ts';
import * as api from '../services/api.ts';
import type { Project, FileTreeNode, CodeFile, CodeUnit, GraphNeighbor } from '../types/index.ts';

export function ProjectView() {
  const { id: projectId } = useParams<{ id: string }>();
  const { projects, isLoading: projectsLoading } = useProjects();

  const [project, setProject] = useState<Project | null>(null);
  const [tree, setTree] = useState<FileTreeNode | null>(null);
  const [isLoadingTree, setIsLoadingTree] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Selected file state
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileInfo, setFileInfo] = useState<CodeFile | null>(null);
  const [fileUnits, setFileUnits] = useState<CodeUnit[]>([]);
  const [selectedUnit, setSelectedUnit] = useState<CodeUnit | null>(null);
  const [isLoadingFile, setIsLoadingFile] = useState(false);

  const [showUpload, setShowUpload] = useState(false);
  const [activeTab, setActiveTab] = useState<'code' | 'graph' | 'understanding'>('code');
  const [pendingUnitId, setPendingUnitId] = useState<string | null>(null);

  // Existing migration plan for this project (if any)
  const [existingPlanId, setExistingPlanId] = useState<string | null>(null);

  // Load project and file tree
  useEffect(() => {
    if (!projectId) return;

    let cancelled = false;

    async function load() {
      setIsLoadingTree(true);
      setError(null);
      try {
        const [proj, fileTree] = await Promise.all([
          api.getProject(projectId!),
          api.getFileTree(projectId!).catch(() => null),
        ]);
        if (cancelled) return;
        setProject(proj);
        setTree(fileTree);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load project');
      } finally {
        if (!cancelled) setIsLoadingTree(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  // Check for existing migration plans for this project
  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    api.listMigrationPlans(projectId).then((plans) => {
      if (cancelled) return;
      // Pick the most relevant plan: prefer active (in_progress/draft) over completed
      const active = plans.find((p) => p.status === 'in_progress' || p.status === 'draft');
      const best = active ?? plans[0] ?? null;
      setExistingPlanId(best?.plan_id ?? null);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [projectId]);

  // Load file content when selected
  useEffect(() => {
    if (!projectId || !selectedFilePath) return;

    let cancelled = false;

    async function loadFile() {
      setIsLoadingFile(true);
      setSelectedUnit(null);
      try {
        const res = await api.getFileContent(projectId!, selectedFilePath!);
        if (cancelled) return;
        // Reconstruct content from units and remap line numbers
        // so highlights align with the displayed content.
        const parts: string[] = [];
        const remappedUnits: typeof fileUnits = [];
        let curLine = 1;
        for (const u of res.units) {
          const srcLines = u.source.split('\n');
          const mappedStart = curLine;
          const mappedEnd = curLine + srcLines.length - 1;
          parts.push(u.source);
          remappedUnits.push({
            unit_id: u.unit_id,
            file_id: res.file_id,
            unit_type: u.unit_type,
            name: u.name,
            start_line: mappedStart,
            end_line: mappedEnd,
            signature: u.signature,
            source: u.source,
          });
          curLine = mappedEnd + 1;
        }
        const content = parts.join('\n');
        setFileContent(content);
        setFileInfo({
          file_id: res.file_id,
          project_id: projectId!,
          file_path: res.file_path,
          language: res.language,
          line_count: content.split('\n').length,
          size_bytes: 0,
          file_hash: '',
          created_at: null,
        });
        setFileUnits(remappedUnits);
      } catch (err) {
        if (cancelled) return;
        setFileContent(`Error loading file: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setFileInfo(null);
        setFileUnits([]);
      } finally {
        if (!cancelled) setIsLoadingFile(false);
      }
    }

    loadFile();
    return () => {
      cancelled = true;
    };
  }, [projectId, selectedFilePath]);

  // When file loads and there's a pending unit to select, find and select it
  useEffect(() => {
    if (!pendingUnitId || fileUnits.length === 0) return;
    const match = fileUnits.find((u) => u.unit_id === pendingUnitId);
    if (match) setSelectedUnit(match);
    setPendingUnitId(null);
  }, [pendingUnitId, fileUnits]);

  const handleSelectFile = useCallback((fileId: string, filePath: string) => {
    setSelectedFileId(fileId);
    setSelectedFilePath(filePath);
  }, []);

  // Navigate to a dependency: resolve file_id from tree, open file, queue unit selection
  const handleNavigateToDep = useCallback(
    (neighbor: GraphNeighbor) => {
      // If the dep is in the current file, just select the unit
      if (neighbor.file_id === selectedFileId) {
        const match = fileUnits.find((u) => u.unit_id === neighbor.unit_id);
        if (match) setSelectedUnit(match);
        return;
      }
      // Find file_path from the tree
      const filePath = findFilePathById(tree, neighbor.file_id);
      if (!filePath) return;
      setPendingUnitId(neighbor.unit_id);
      handleSelectFile(neighbor.file_id, filePath);
    },
    [selectedFileId, fileUnits, tree, handleSelectFile],
  );

  const handleUploadComplete = useCallback(() => {
    // Refresh project and tree after upload
    if (!projectId) return;
    setShowUpload(false);
    api.getProject(projectId).then(setProject).catch(() => {});
    api.getFileTree(projectId).then(setTree).catch(() => {});
  }, [projectId]);

  if (!projectId) {
    return (
      <Layout projects={projects} projectsLoading={projectsLoading}>
        <div className="flex h-full items-center justify-center text-text-muted">
          No project selected
        </div>
      </Layout>
    );
  }

  // Build breadcrumb segments from selected file path
  const breadcrumbSegments = selectedFilePath
    ? selectedFilePath.split('/').filter(Boolean)
    : [];

  return (
    <Layout projects={projects} projectsLoading={projectsLoading}>
      {/* Top Bar */}
      <div className="flex items-center justify-between border-b border-void-surface px-4 py-3">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-medium text-text">
            {project?.name ?? 'Loading...'}
          </h1>
          {project?.primary_language && (
            <span className="rounded bg-void-surface/50 px-2 py-0.5 text-xs text-text-muted">
              {project.primary_language}
            </span>
          )}
          {project?.asg_status === 'complete' && (
            <span className="rounded bg-success/10 px-2 py-0.5 text-[10px] font-medium text-success">
              ASG COMPLETE
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Code / Graph tabs */}
          <div className="flex rounded-md border border-void-surface">
            <button
              onClick={() => setActiveTab('code')}
              className={`flex items-center gap-1 px-2.5 py-1 text-xs ${
                activeTab === 'code'
                  ? 'bg-void-surface text-text'
                  : 'text-text-muted hover:text-text-dim'
              }`}
            >
              <Code2 className="h-3 w-3" />
              Code
            </button>
            <button
              onClick={() => setActiveTab('graph')}
              className={`flex items-center gap-1 px-2.5 py-1 text-xs ${
                activeTab === 'graph'
                  ? 'bg-void-surface text-text'
                  : 'text-text-muted hover:text-text-dim'
              }`}
            >
              <Network className="h-3 w-3" />
              Graph
            </button>
            <button
              onClick={() => setActiveTab('understanding')}
              className={`flex items-center gap-1 px-2.5 py-1 text-xs ${
                activeTab === 'understanding'
                  ? 'bg-void-surface text-text'
                  : 'text-text-muted hover:text-text-dim'
              }`}
            >
              <Brain className="h-3 w-3" />
              Understanding
            </button>
          </div>

          <button
            onClick={() => setShowUpload(!showUpload)}
            className="flex items-center gap-1.5 rounded-md border border-void-surface px-3 py-1.5 text-xs text-text-muted hover:bg-void-surface hover:text-text"
          >
            <UploadIcon className="h-3.5 w-3.5" />
            Ingest Code
          </button>
          {project?.asg_status === 'complete' && (
            <button
              onClick={() => setActiveTab('understanding')}
              className="flex items-center gap-1.5 rounded-md border border-glow/50 px-3 py-1.5 text-xs text-glow hover:bg-glow/10"
            >
              <Brain className="h-3.5 w-3.5" />
              Analyze
            </button>
          )}
          {project?.asg_status === 'complete' && (
            <Link
              to={existingPlanId
                ? `/migration/${existingPlanId}`
                : `/migration/new?project=${projectId}`}
              className="flex items-center gap-1.5 rounded-md border border-nebula/50 px-3 py-1.5 text-xs text-nebula hover:bg-nebula/10 hover:text-nebula-light"
            >
              <ArrowRightLeft className="h-3.5 w-3.5" />
              {existingPlanId ? 'Migration' : 'Migrate'}
            </Link>
          )}
          <Link
            to={`/project/${projectId}/chat`}
            className="flex items-center gap-1.5 rounded-md bg-glow px-3 py-1.5 text-xs font-medium text-white hover:bg-glow-dim"
          >
            <MessageSquareCode className="h-3.5 w-3.5" />
            Chat
          </Link>
        </div>
      </div>

      {/* Breadcrumb bar */}
      {selectedFilePath && activeTab === 'code' && (
        <div className="flex items-center gap-1.5 border-b border-void-surface/50 bg-void-light/30 px-4 py-1.5 text-xs text-text-dim">
          <span className="material-symbols-outlined text-[14px]">description</span>
          {breadcrumbSegments.map((segment, idx) => (
            <span key={idx} className="flex items-center gap-1.5">
              {idx > 0 && <span className="text-text-dim/40">&gt;</span>}
              <span className={idx === breadcrumbSegments.length - 1 ? 'text-text-muted' : ''}>
                {segment}
              </span>
            </span>
          ))}
        </div>
      )}

      {/* Upload section (collapsible) */}
      {showUpload && (
        <div className="border-b border-void-surface p-4">
          <ProjectUpload
            projectId={projectId}
            onUploadComplete={handleUploadComplete}
          />
        </div>
      )}

      {/* Main content area */}
      {activeTab === 'understanding' ? (
        <div className="flex-1 overflow-hidden">
          <UnderstandingPanel
            projectId={projectId}
            asgStatus={project?.asg_status ?? 'pending'}
          />
        </div>
      ) : activeTab === 'graph' ? (
        <div className="flex-1 overflow-hidden">
          <GraphViewer
            projectId={projectId}
            asgStatus={project?.asg_status ?? 'pending'}
          />
        </div>
      ) : (
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="flex flex-1 overflow-hidden">
            {/* File tree panel */}
            <div className="w-64 shrink-0 overflow-y-auto border-r border-void-surface p-2">
              {isLoadingTree ? (
                <div className="flex items-center justify-center py-10">
                  <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
                </div>
              ) : error ? (
                <div className="flex items-center gap-2 p-3 text-xs text-danger">
                  <AlertCircle className="h-4 w-4" />
                  {error}
                </div>
              ) : tree ? (
                <FileTree
                  tree={tree}
                  selectedFileId={selectedFileId ?? undefined}
                  onSelectFile={handleSelectFile}
                />
              ) : (
                <div className="p-3 text-center text-xs text-text-dim">
                  <p>No files yet.</p>
                  <p className="mt-1">Upload a codebase to get started.</p>
                </div>
              )}
            </div>

            {/* Code viewer panel */}
            <div className="flex-1 overflow-hidden">
              {isLoadingFile ? (
                <div className="flex h-full items-center justify-center">
                  <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
                </div>
              ) : selectedFileId && fileContent !== null ? (
                <CodeViewer
                  content={fileContent}
                  filePath={selectedFilePath ?? ''}
                  language={fileInfo?.language}
                  units={fileUnits}
                  selectedUnit={selectedUnit}
                  onSelectUnit={setSelectedUnit}
                />
              ) : (
                <ProjectStats project={project} />
              )}
            </div>

            {/* Structure panel (right side) */}
            {selectedFileId && fileUnits.length > 0 && (
              <StructurePanel
                projectId={projectId!}
                units={fileUnits}
                selectedUnit={selectedUnit}
                onSelectUnit={setSelectedUnit}
                onNavigateToDep={handleNavigateToDep}
              />
            )}
          </div>

          {/* Status bar */}
          <div className="flex h-7 shrink-0 items-center gap-4 border-t border-void-surface bg-void-light/50 px-4 text-[10px] text-text-dim">
            <span className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-success" />
              Synced
            </span>
            {fileInfo?.language && (
              <span>{fileInfo.language}</span>
            )}
            {fileInfo?.line_count != null && (
              <span>{fileInfo.line_count} lines</span>
            )}
            {project && (
              <span>{project.file_count} files in project</span>
            )}
          </div>
        </div>
      )}
    </Layout>
  );
}

// ---------------------------------------------------------------------------
// StructurePanel — right panel showing code units grouped by type
// ---------------------------------------------------------------------------

const UNIT_TYPE_STYLES: Record<string, string> = {
  class: 'bg-warning/10 text-warning',
  function: 'bg-glow/10 text-glow',
  method: 'bg-nebula/10 text-nebula-bright',
  module: 'bg-success/10 text-success',
  interface: 'bg-glow-bright/10 text-glow-bright',
};

function StructurePanel({
  projectId,
  units,
  selectedUnit,
  onSelectUnit,
  onNavigateToDep,
}: {
  projectId: string;
  units: CodeUnit[];
  selectedUnit: CodeUnit | null;
  onSelectUnit: (unit: CodeUnit) => void;
  onNavigateToDep: (neighbor: GraphNeighbor) => void;
}) {
  const [deps, setDeps] = useState<{ callers: GraphNeighbor[]; callees: GraphNeighbor[] }>({ callers: [], callees: [] });
  const [depsLoading, setDepsLoading] = useState(false);

  // Fetch dependencies when a unit is selected
  useEffect(() => {
    if (!selectedUnit) {
      setDeps({ callers: [], callees: [] });
      return;
    }
    let cancelled = false;
    setDepsLoading(true);
    Promise.all([
      api.getCallers(projectId, selectedUnit.unit_id).catch(() => ({ results: [] as GraphNeighbor[] })),
      api.getCallees(projectId, selectedUnit.unit_id).catch(() => ({ results: [] as GraphNeighbor[] })),
    ]).then(([c, e]) => {
      if (!cancelled) {
        setDeps({ callers: c.results, callees: e.results });
        setDepsLoading(false);
      }
    });
    return () => { cancelled = true; };
  }, [projectId, selectedUnit]);

  // Group units by type
  const grouped = units.reduce<Record<string, CodeUnit[]>>((acc, unit) => {
    const type = unit.unit_type || 'other';
    if (!acc[type]) acc[type] = [];
    acc[type].push(unit);
    return acc;
  }, {});

  const typeOrder = ['class', 'interface', 'function', 'method', 'module', 'other'];
  const sortedTypes = Object.keys(grouped).sort(
    (a, b) => (typeOrder.indexOf(a) === -1 ? 99 : typeOrder.indexOf(a)) - (typeOrder.indexOf(b) === -1 ? 99 : typeOrder.indexOf(b)),
  );

  const hasDeps = deps.callers.length > 0 || deps.callees.length > 0;

  return (
    <div className="w-64 shrink-0 overflow-y-auto border-l border-void-surface">
      {/* STRUCTURE section */}
      <div className="border-b border-void-surface px-3 py-2.5">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
          Structure
        </h4>
      </div>
      <div className="p-2">
        {sortedTypes.map((type) => (
          <div key={type} className="mb-2">
            <p className="mb-1 px-1 text-[9px] font-semibold uppercase tracking-wider text-text-dim/70">
              {type}s ({grouped[type].length})
            </p>
            {grouped[type].map((unit) => (
              <button
                key={unit.unit_id}
                onClick={() => onSelectUnit(unit)}
                className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors ${
                  selectedUnit?.unit_id === unit.unit_id
                    ? 'bg-glow/10 text-glow'
                    : 'text-text-muted hover:bg-void-surface hover:text-text'
                }`}
              >
                <span
                  className={`shrink-0 rounded px-1 py-0.5 text-[8px] font-bold uppercase ${
                    UNIT_TYPE_STYLES[unit.unit_type] ?? 'bg-void-surface text-text-dim'
                  }`}
                >
                  {unit.unit_type.slice(0, 3)}
                </span>
                <span className="truncate font-[family-name:var(--font-code)] text-[11px]">
                  {unit.name}
                </span>
              </button>
            ))}
          </div>
        ))}
      </div>

      {/* DEPENDENCIES section */}
      <div className="border-t border-void-surface px-3 py-2.5">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
          Dependencies
        </h4>
      </div>
      <div className="px-3 py-2">
        {!selectedUnit ? (
          <p className="text-[10px] text-text-dim/50">Select a unit to see dependencies.</p>
        ) : depsLoading ? (
          <p className="text-[10px] text-text-dim/50">Loading...</p>
        ) : !hasDeps ? (
          <p className="text-[10px] text-text-dim/50">No dependencies found.</p>
        ) : (
          <div className="flex flex-col gap-2">
            {deps.callees.length > 0 && (
              <div>
                <p className="mb-1 text-[9px] font-semibold uppercase tracking-wider text-text-dim/70">
                  Calls ({deps.callees.length})
                </p>
                {deps.callees.map((n) => (
                  <DepItem key={n.unit_id} neighbor={n} onClick={onNavigateToDep} />
                ))}
              </div>
            )}
            {deps.callers.length > 0 && (
              <div>
                <p className="mb-1 text-[9px] font-semibold uppercase tracking-wider text-text-dim/70">
                  Called by ({deps.callers.length})
                </p>
                {deps.callers.map((n) => (
                  <DepItem key={n.unit_id} neighbor={n} onClick={onNavigateToDep} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function DepItem({ neighbor, onClick }: { neighbor: GraphNeighbor; onClick: (n: GraphNeighbor) => void }) {
  const style = UNIT_TYPE_STYLES[neighbor.unit_type] ?? 'bg-void-surface text-text-dim';
  return (
    <button
      onClick={() => onClick(neighbor)}
      className="flex w-full items-center gap-1.5 rounded px-1 py-1 text-[10px] text-text-muted hover:bg-void-surface hover:text-text transition-colors text-left"
      title={neighbor.qualified_name}
    >
      <span className={`shrink-0 rounded px-1 py-0.5 text-[7px] font-bold uppercase ${style}`}>
        {neighbor.unit_type.slice(0, 3)}
      </span>
      <span className="truncate font-[family-name:var(--font-code)]">
        {neighbor.name}
      </span>
      <span className="ml-auto shrink-0 text-[8px] text-text-dim/50">{neighbor.edge_type}</span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Tree search helper — resolves file_id to file_path
// ---------------------------------------------------------------------------

function findFilePathById(node: FileTreeNode | null, fileId: string): string | null {
  if (!node) return null;
  if (node.file_id === fileId && node.file_path) return node.file_path;
  if (node.children) {
    for (const child of node.children) {
      const found = findFilePathById(child, fileId);
      if (found) return found;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// ProjectStats (shown when no file is selected)
// ---------------------------------------------------------------------------

function ProjectStats({ project }: { project: Project | null }) {
  if (!project) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
      </div>
    );
  }

  return (
    <div className="flex h-full items-center justify-center">
      <div className="max-w-sm text-center">
        <h2 className="text-lg font-semibold text-text">{project.name}</h2>
        {project.description && (
          <p className="mt-2 text-sm text-text-muted">{project.description}</p>
        )}
        <div className="mt-6 grid grid-cols-2 gap-4 text-left">
          <StatBox label="Files" value={project.file_count} />
          <StatBox label="Lines" value={project.total_lines.toLocaleString()} />
          <StatBox label="AST Status" value={project.ast_status} />
          <StatBox
            label="Languages"
            value={
              project.languages.length > 0
                ? project.languages.join(', ')
                : 'N/A'
            }
          />
          <StatBox label="Source" value={project.source_type ?? 'zip'} />
          {project.source_url && (
            <StatBox
              label={project.source_type === 'git' ? 'Repository' : 'Path'}
              value={project.source_url}
            />
          )}
          {project.repo_branch && (
            <StatBox label="Branch" value={project.repo_branch} />
          )}
          {project.last_synced_at && (
            <StatBox
              label="Last Synced"
              value={new Date(project.last_synced_at).toLocaleString()}
            />
          )}
        </div>
        <p className="mt-8 text-xs text-text-dim">
          Select a file from the tree to view its source code.
        </p>
      </div>
    </div>
  );
}

function StatBox({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-lg border border-void-surface/50 bg-void-light/50 p-3">
      <p className="text-[10px] font-medium uppercase tracking-wider text-text-dim">
        {label}
      </p>
      <p className="mt-1 text-sm font-medium text-text-dim">{value}</p>
    </div>
  );
}
