/**
 * CodeLoom API Client
 *
 * Single point of contact for all backend communication.
 * Uses cookie-based sessions (credentials: 'include').
 * All paths are relative -- Vite proxies /api to FastAPI on :9005.
 */

import type {
  User,
  Project,
  CodeFile,
  CodeUnit,
  FileTreeNode,
  IngestionResult,
  ModelProvider,
  RerankerConfig,
  RerankerOption,
  GraphOverview,
  GraphNeighbor,
  UnitDetail,
} from '../types/index.ts';

// ---------------------------------------------------------------------------
// Fetch wrapper
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const body = await response.text();
    let message: string;
    try {
      const json = JSON.parse(body) as { detail?: string; error?: string; message?: string };
      message = json.detail ?? json.error ?? json.message ?? body;
    } catch {
      message = body;
    }

    // Provide clear feedback for authorization failures
    if (response.status === 403) {
      message = message || 'Access denied. You do not have permission for this action.';
    }

    throw new ApiError(response.status, message);
  }

  // 204 No Content
  if (response.status === 204) {
    return undefined as unknown as T;
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export async function login(credentials: {
  username: string;
  password: string;
}): Promise<User> {
  return request<User>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify(credentials),
  });
}

export async function logout(): Promise<void> {
  return request<void>('/api/auth/logout', { method: 'POST' });
}

export async function getCurrentUser(): Promise<User> {
  return request<User>('/api/auth/me');
}

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

export async function listProjects(): Promise<Project[]> {
  return request<Project[]>('/api/projects');
}

export async function getProject(projectId: string): Promise<Project> {
  return request<Project>(`/api/projects/${projectId}`);
}

export async function createProject(data: {
  name: string;
  description?: string;
}): Promise<Project> {
  return request<Project>('/api/projects', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function deleteProject(projectId: string): Promise<void> {
  return request<void>(`/api/projects/${projectId}`, {
    method: 'DELETE',
  });
}

// ---------------------------------------------------------------------------
// Code Upload & Ingestion
// ---------------------------------------------------------------------------

export async function uploadCodebase(
  projectId: string,
  file: File,
): Promise<IngestionResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`/api/projects/${projectId}/upload`, {
    method: 'POST',
    credentials: 'include',
    body: formData,
    // Do NOT set Content-Type -- browser sets multipart boundary automatically
  });

  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(response.status, body);
  }

  return response.json() as Promise<IngestionResult>;
}

export async function ingestFromGit(
  projectId: string,
  repoUrl: string,
  branch: string = 'main',
): Promise<IngestionResult> {
  return request<IngestionResult>(`/api/projects/${projectId}/ingest/git`, {
    method: 'POST',
    body: JSON.stringify({ repo_url: repoUrl, branch }),
  });
}

export async function ingestFromLocal(
  projectId: string,
  dirPath: string,
): Promise<IngestionResult> {
  return request<IngestionResult>(`/api/projects/${projectId}/ingest/local`, {
    method: 'POST',
    body: JSON.stringify({ dir_path: dirPath }),
  });
}

// ---------------------------------------------------------------------------
// Files & Code Units
// ---------------------------------------------------------------------------

export async function getProjectFiles(
  projectId: string,
): Promise<CodeFile[]> {
  return request<CodeFile[]>(`/api/projects/${projectId}/files`);
}

export async function getFileTree(
  projectId: string,
): Promise<FileTreeNode> {
  return request<FileTreeNode>(`/api/projects/${projectId}/tree`);
}

export async function getProjectUnits(
  projectId: string,
  fileId?: string,
): Promise<CodeUnit[]> {
  const params = fileId ? `?file_id=${fileId}` : '';
  return request<CodeUnit[]>(`/api/projects/${projectId}/units${params}`);
}

export async function getFileContent(
  projectId: string,
  filePath: string,
): Promise<{
  file_id: string;
  file_path: string;
  language: string;
  line_count: number;
  units: Array<{
    unit_id: string;
    unit_type: string;
    name: string;
    start_line: number;
    end_line: number;
    signature: string;
    source: string;
  }>;
}> {
  return request(`/api/projects/${projectId}/file/${filePath}`);
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export interface ModelsResponse {
  providers: ModelProvider[];
  current: { provider: string; model: string };
  default_provider: string;
  default_model: string;
}

export async function getModels(): Promise<ModelsResponse> {
  return request<ModelsResponse>('/api/settings/models');
}

export async function setModel(provider: string, model: string): Promise<void> {
  return request<void>('/api/settings/models', {
    method: 'POST',
    body: JSON.stringify({ provider, model }),
  });
}

export interface RerankerResponse {
  config: RerankerConfig;
  available_models: RerankerOption[];
}

export async function getReranker(): Promise<RerankerResponse> {
  return request<RerankerResponse>('/api/settings/reranker');
}

export async function setReranker(
  enabled: boolean,
  model: string,
  top_n?: number,
): Promise<void> {
  return request<void>('/api/settings/reranker', {
    method: 'POST',
    body: JSON.stringify({ enabled, model, top_n: top_n ?? 10 }),
  });
}

// ---------------------------------------------------------------------------
// Graph / ASG
// ---------------------------------------------------------------------------

export async function getGraphOverview(
  projectId: string,
): Promise<GraphOverview> {
  return request<GraphOverview>(`/api/projects/${projectId}/graph/overview`);
}

export async function getCallers(
  projectId: string,
  unitId: string,
  depth: number = 1,
): Promise<{ unit_id: string; direction: string; depth: number; results: GraphNeighbor[] }> {
  return request(`/api/projects/${projectId}/graph/callers/${unitId}?depth=${depth}`);
}

export async function getCallees(
  projectId: string,
  unitId: string,
  depth: number = 1,
): Promise<{ unit_id: string; direction: string; depth: number; results: GraphNeighbor[] }> {
  return request(`/api/projects/${projectId}/graph/callees/${unitId}?depth=${depth}`);
}

export async function getDependencies(
  projectId: string,
  unitId: string,
  depth: number = 2,
): Promise<{ unit_id: string; direction: string; depth: number; results: GraphNeighbor[] }> {
  return request(`/api/projects/${projectId}/graph/dependencies/${unitId}?depth=${depth}`);
}

export async function getDependents(
  projectId: string,
  unitId: string,
  depth: number = 2,
): Promise<{ unit_id: string; direction: string; depth: number; results: GraphNeighbor[] }> {
  return request(`/api/projects/${projectId}/graph/dependents/${unitId}?depth=${depth}`);
}

export async function getFullGraph(
  projectId: string,
  edgeTypes: string = 'calls,contains,inherits,imports,implements,overrides',
): Promise<{
  nodes: Array<{ id: string; name: string; qualified_name: string; unit_type: string; language: string; file_id: string }>;
  links: Array<{ source: string; target: string; edge_type: string }>;
  node_count: number;
  edge_count: number;
}> {
  return request(`/api/projects/${projectId}/graph/full?edge_types=${edgeTypes}`);
}

export async function getUnitDetail(
  projectId: string,
  unitId: string,
): Promise<UnitDetail> {
  return request<UnitDetail>(`/api/projects/${projectId}/graph/unit/${unitId}`);
}

export async function buildASG(
  projectId: string,
): Promise<{ success: boolean; project_id: string; edges_created: number; elapsed_seconds: number }> {
  return request(`/api/projects/${projectId}/build-asg`, { method: 'POST' });
}

// ---------------------------------------------------------------------------
// Migration
// ---------------------------------------------------------------------------

import type {
  MigrationPlan,
  MigrationPhaseOutput,
  FunctionalMvpSummary,
  MvpDetail,
  DiscoveryResult,
  AssetInventoryResponse,
  AssetStrategySpec,
  MigrationLane,
} from '../types/index.ts';

// ── Plan CRUD ──

export async function createMigrationPlan(data: {
  source_project_id: string;
  target_brief: string;
  target_stack: object;
  constraints?: object;
  migration_type?: string;
}): Promise<MigrationPlan> {
  return request<MigrationPlan>('/api/migration/plan', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function listMigrationPlans(
  projectId?: string,
): Promise<MigrationPlan[]> {
  const params = projectId ? `?project_id=${projectId}` : '';
  return request<MigrationPlan[]>(`/api/migration/plans${params}`);
}

export async function getMigrationPlan(planId: string): Promise<MigrationPlan> {
  return request<MigrationPlan>(`/api/migration/${planId}`);
}

export async function deleteMigrationPlan(planId: string): Promise<void> {
  return request<void>(`/api/migration/${planId}`, { method: 'DELETE' });
}

// ── Asset Inventory ──

export async function getAssetInventory(
  planId: string,
): Promise<AssetInventoryResponse> {
  return request<AssetInventoryResponse>(`/api/migration/${planId}/asset-inventory`);
}

export async function refineAssetInventory(
  planId: string,
): Promise<AssetInventoryResponse> {
  return request<AssetInventoryResponse>(`/api/migration/${planId}/asset-inventory/refine`, {
    method: 'POST',
  });
}

export async function saveAssetStrategies(
  planId: string,
  strategies: Record<string, AssetStrategySpec>,
): Promise<{ status: string; plan_id: string }> {
  return request(`/api/migration/${planId}/asset-strategies`, {
    method: 'POST',
    body: JSON.stringify({ strategies }),
  });
}

export async function listMigrationLanes(): Promise<MigrationLane[]> {
  return request<MigrationLane[]>('/api/migration/lanes');
}

// ── Discovery + MVP Management ──

export async function runDiscovery(
  planId: string,
  clusteringParams?: Record<string, unknown>,
): Promise<DiscoveryResult> {
  return request<DiscoveryResult>(`/api/migration/${planId}/discover`, {
    method: 'POST',
    body: JSON.stringify(clusteringParams ? { clustering_params: clusteringParams } : {}),
  });
}

export async function listMvps(
  planId: string,
): Promise<FunctionalMvpSummary[]> {
  return request<FunctionalMvpSummary[]>(`/api/migration/${planId}/mvps`);
}

export async function getMvpDetail(
  planId: string,
  mvpId: number,
): Promise<MvpDetail> {
  return request<MvpDetail>(`/api/migration/${planId}/mvps/${mvpId}`);
}

export async function updateMvp(
  planId: string,
  mvpId: number,
  updates: {
    name?: string;
    description?: string;
    unit_ids?: string[];
    file_ids?: string[];
    priority?: number;
  },
): Promise<FunctionalMvpSummary> {
  return request<FunctionalMvpSummary>(`/api/migration/${planId}/mvps/${mvpId}`, {
    method: 'PUT',
    body: JSON.stringify(updates),
  });
}

export async function mergeMvps(
  planId: string,
  mvpIds: number[],
  newName?: string,
): Promise<FunctionalMvpSummary> {
  return request<FunctionalMvpSummary>(`/api/migration/${planId}/mvps/merge`, {
    method: 'POST',
    body: JSON.stringify({ mvp_ids: mvpIds, new_name: newName }),
  });
}

export async function splitMvp(
  planId: string,
  mvpId: number,
  unitIds: string[],
  newName: string,
): Promise<{ original: FunctionalMvpSummary; new_mvp: FunctionalMvpSummary }> {
  return request(`/api/migration/${planId}/mvps/${mvpId}/split`, {
    method: 'POST',
    body: JSON.stringify({ unit_ids: unitIds, new_name: newName }),
  });
}

export async function analyzeMvp(
  planId: string,
  mvpId: number,
): Promise<{ output: string; output_files: unknown[]; analysis_at: string }> {
  return request(`/api/migration/${planId}/mvps/${mvpId}/analyze`, {
    method: 'POST',
  });
}

export async function analyzeAllMvps(
  planId: string,
): Promise<{ analyzed: number; total: number; results: Record<string, { status: string }> }> {
  return request(`/api/migration/${planId}/mvps/analyze-all`, {
    method: 'POST',
  });
}

export async function createMvpPhases(
  planId: string,
): Promise<MigrationPlan> {
  return request<MigrationPlan>(`/api/migration/${planId}/mvps/create-phases`, {
    method: 'POST',
  });
}

// ── Phase Execution (plan-level + per-MVP) ──

export async function executeMigrationPhase(
  planId: string,
  phaseNumber: number,
  mvpId?: number,
): Promise<MigrationPhaseOutput> {
  const params = mvpId != null ? `?mvp_id=${mvpId}` : '';
  return request<MigrationPhaseOutput>(
    `/api/migration/${planId}/phase/${phaseNumber}/execute${params}`,
    { method: 'POST' },
  );
}

export async function getMigrationPhaseOutput(
  planId: string,
  phaseNumber: number,
  mvpId?: number,
): Promise<MigrationPhaseOutput> {
  const params = mvpId != null ? `?mvp_id=${mvpId}` : '';
  return request<MigrationPhaseOutput>(
    `/api/migration/${planId}/phase/${phaseNumber}${params}`,
  );
}

export async function approveMigrationPhase(
  planId: string,
  phaseNumber: number,
  mvpId?: number,
): Promise<MigrationPlan> {
  const params = mvpId != null ? `?mvp_id=${mvpId}` : '';
  return request<MigrationPlan>(
    `/api/migration/${planId}/phase/${phaseNumber}/approve${params}`,
    { method: 'POST' },
  );
}

export async function rejectMigrationPhase(
  planId: string,
  phaseNumber: number,
  mvpId?: number,
  feedback?: string,
): Promise<MigrationPhaseOutput> {
  const params = mvpId != null ? `?mvp_id=${mvpId}` : '';
  return request<MigrationPhaseOutput>(
    `/api/migration/${planId}/phase/${phaseNumber}/reject${params}`,
    {
      method: 'POST',
      body: JSON.stringify(feedback ? { feedback } : {}),
    },
  );
}

// ── Diff Context + Download ──

import type {
  DiffContext,
  EntryPoint,
  AnalysisSummary,
  AnalysisJobStatus,
  ChainDetail,
} from '../types/index.ts';

export async function getDiffContext(
  planId: string,
  phaseNumber: number,
  mvpId?: number,
): Promise<DiffContext> {
  const params = mvpId != null ? `?mvp_id=${mvpId}` : '';
  return request<DiffContext>(
    `/api/migration/${planId}/phase/${phaseNumber}/diff-context${params}`,
  );
}

export function getPhaseDownloadUrl(
  planId: string,
  phaseNumber: number,
  mvpId?: number,
  format: 'zip' | 'single' = 'zip',
  filePath?: string,
): string {
  const params = new URLSearchParams();
  if (mvpId != null) params.set('mvp_id', String(mvpId));
  params.set('format', format);
  if (filePath) params.set('file_path', filePath);
  return `/api/migration/${planId}/phase/${phaseNumber}/download?${params.toString()}`;
}

// ── MVP Diagrams ──

import type {
  DiagramResponse,
  DiagramAvailability,
} from '../types/index.ts';

export async function listMvpDiagrams(
  planId: string,
  mvpId: number,
): Promise<DiagramAvailability[]> {
  return request<DiagramAvailability[]>(`/api/migration/${planId}/mvps/${mvpId}/diagrams`);
}

export async function getMvpDiagram(
  planId: string,
  mvpId: number,
  diagramType: string,
): Promise<DiagramResponse> {
  return request<DiagramResponse>(`/api/migration/${planId}/mvps/${mvpId}/diagrams/${diagramType}`);
}

export async function refreshMvpDiagram(
  planId: string,
  mvpId: number,
  diagramType: string,
): Promise<DiagramResponse> {
  return request<DiagramResponse>(`/api/migration/${planId}/mvps/${mvpId}/diagrams/${diagramType}/refresh`, {
    method: 'POST',
  });
}

// ── Framework Doc Enrichment ──

export async function enrichFrameworkDocs(
  planId: string,
): Promise<{ enriched: string[]; failed: string[]; total_tokens: number }> {
  return request(`/api/migration/${planId}/enrich-docs`, { method: 'POST' });
}

// ── Batch Execution ──

export interface BatchExecuteParams {
  phase_number?: number | null;
  mvp_ids?: number[] | null;
  approval_policy?: 'manual' | 'auto' | 'auto_non_blocking';
  run_all?: boolean;
}

export interface BatchMvpResult {
  mvp_id: number;
  name: string;
  status: 'pending' | 'processing' | 'executed' | 'approved' | 'needs_review' | 'failed' | 'skipped';
  current_phase: number;
  error: string | null;
  completed_at: string | null;
  gate_results: Array<{ gate_name: string; passed: boolean; blocking: boolean; details?: string }>;
}

export interface BatchStatus {
  batch_id: string;
  plan_id: string;
  status: 'running' | 'complete' | 'partial_failure' | 'needs_review';
  approval_policy: string;
  run_all: boolean;
  starting_phase: number;
  total_mvps: number;
  completed: number;
  failed: number;
  skipped: number;
  needs_review: number;
  started_at: string;
  completed_at: string | null;
  mvp_results: BatchMvpResult[];
}

export interface BatchLaunchResult {
  batch_id: string;
  plan_id: string;
  total_mvps: number;
  mvp_list: Array<{ mvp_id: number; name: string }>;
  pipeline_version: number;
}

export async function launchBatchExecution(
  planId: string,
  params: BatchExecuteParams,
): Promise<BatchLaunchResult> {
  return request<BatchLaunchResult>(`/api/migration/${planId}/batch/execute`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getBatchStatus(
  planId: string,
  batchId: string,
): Promise<BatchStatus> {
  return request<BatchStatus>(`/api/migration/${planId}/batch/${batchId}`);
}

export async function listBatchExecutions(
  planId: string,
): Promise<BatchStatus[]> {
  return request<BatchStatus[]>(`/api/migration/${planId}/batch`);
}

export async function retryBatchExecution(
  planId: string,
  batchId: string,
  mvpIds?: number[],
): Promise<BatchLaunchResult> {
  return request<BatchLaunchResult>(`/api/migration/${planId}/batch/${batchId}/retry`, {
    method: 'POST',
    body: JSON.stringify(mvpIds ? { mvp_ids: mvpIds } : {}),
  });
}

// ---------------------------------------------------------------------------
// Understanding / Deep Analysis
// ---------------------------------------------------------------------------

export async function startAnalysis(
  projectId: string,
): Promise<{ job_id: string; status: string; project_id: string }> {
  return request(`/api/understanding/${projectId}/analyze`, { method: 'POST' });
}

export async function getAnalysisStatus(
  projectId: string,
  jobId: string,
): Promise<AnalysisJobStatus> {
  return request(`/api/understanding/${projectId}/status/${jobId}`);
}

export async function getEntryPoints(
  projectId: string,
): Promise<{ entry_points: EntryPoint[]; count: number }> {
  return request(`/api/understanding/${projectId}/entry-points`);
}

export async function getAnalysisResults(
  projectId: string,
): Promise<{ analyses: AnalysisSummary[]; count: number; coverage_pct: number }> {
  return request(`/api/understanding/${projectId}/results`);
}

export async function getChainDetail(
  projectId: string,
  analysisId: string,
): Promise<ChainDetail> {
  return request(`/api/understanding/${projectId}/chain/${analysisId}`);
}

// ---------------------------------------------------------------------------
// Analytics
// ---------------------------------------------------------------------------

export async function getProjectAnalytics(
  projectId: string,
): Promise<import('../types/index.ts').ProjectAnalytics> {
  return request(`/api/projects/${projectId}/analytics`);
}
