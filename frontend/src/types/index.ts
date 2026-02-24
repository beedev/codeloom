/**
 * CodeLoom TypeScript Interfaces
 *
 * Central type definitions for the entire frontend.
 * Every API response and UI data structure lives here.
 */

export interface Project {
  project_id: string;
  user_id: string;
  name: string;
  description: string;
  primary_language: string | null;
  languages: string[];
  file_count: number;
  total_lines: number;
  ast_status: 'pending' | 'parsing' | 'complete' | 'error';
  asg_status: string;
  deep_analysis_status: 'none' | 'pending' | 'running' | 'completed' | 'failed';
  source_type: 'zip' | 'git' | 'local';
  source_url: string | null;
  repo_branch: string | null;
  last_synced_at: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface CodeFile {
  file_id: string;
  project_id: string;
  file_path: string;
  language: string;
  line_count: number;
  size_bytes: number;
  file_hash: string;
  created_at: string | null;
}

export interface CodeUnit {
  unit_id: string;
  file_id: string;
  unit_type: string;
  name: string;
  start_line: number;
  end_line: number;
  signature: string;
  source: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
}

export interface ChatSource {
  filename: string;
  snippet: string;
  score: number;
  unit_name?: string;
  unit_type?: string;
  start_line?: number;
  end_line?: number;
  language?: string;
}

export interface User {
  user_id: string;
  username: string;
  email: string | null;
  roles: string[];
  api_key: string | null;
}

export interface FileTreeNode {
  name: string;
  type: 'file' | 'directory';
  children?: FileTreeNode[];
  file_id?: string;
  file_path?: string;
  language?: string;
  line_count?: number;
}

export interface IngestionResult {
  project_id: string;
  files_processed: number;
  files_skipped: number;
  units_extracted: number;
  chunks_created: number;
  embeddings_stored: number;
  errors: string[];
  elapsed_seconds: number;
}

// ---- Settings types ---------------------------------------------------------

export interface ModelProvider {
  name: string;
  type: string;
  models: ModelInfo[];
}

export interface ModelInfo {
  name: string;
  display_name: string;
}

export interface RerankerConfig {
  enabled: boolean;
  model: string;
  resolved_model: string | null;
  top_n: number;
  loaded: boolean;
  is_local: boolean;
  is_groq: boolean;
}

export interface RerankerOption {
  id: string;
  name: string;
  type: string;
  description: string;
}

// ---- Graph / ASG types ------------------------------------------------------

export interface GraphNode {
  id: string;
  name: string;
  qualified_name?: string;
  unit_type: string;
  language?: string;
  file_id?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  edge_type: string;
}

export interface GraphOverview {
  project_id: string;
  total_edges: number;
  edge_types: Record<string, number>;
}

export interface GraphNeighbor {
  unit_id: string;
  name: string;
  qualified_name: string;
  unit_type: string;
  language: string;
  file_id: string;
  edge_type: string;
  depth: number;
}

export interface EdgeUnit {
  unit_id: string;
  name: string;
  qualified_name: string;
  unit_type: string;
  language: string;
  edge_type: string;
}

export interface UnitDetail {
  unit_id: string;
  name: string;
  qualified_name: string;
  unit_type: string;
  language: string;
  file_id: string;
  file_path: string | null;
  start_line: number | null;
  end_line: number | null;
  line_count: number | null;
  signature: string | null;
  docstring: string | null;
  source: string | null;
  edges: {
    outgoing: Record<string, EdgeUnit[]>;
    incoming: Record<string, EdgeUnit[]>;
  };
}

// ---- Migration types --------------------------------------------------------

export type MigrationPhaseType =
  | 'discovery'
  | 'architecture'
  | 'analyze'
  | 'design'
  | 'transform'
  | 'test';

export interface MigrationPlan {
  plan_id: string;
  source_project_id: string | null;
  source_stack: { primary_language: string; languages: string[] } | null;
  target_brief: string;
  target_stack: { languages: string[]; frameworks: string[]; versions?: Record<string, string> };
  constraints: { timeline?: string; team_size?: number; risk_tolerance?: string } | null;
  status: 'draft' | 'in_progress' | 'complete' | 'abandoned';
  current_phase: number;
  migration_type?: 'version_upgrade' | 'framework_migration' | 'rewrite';
  pipeline_version?: number;  // 1=V1 (6-phase), 2=V2 (4-phase). Defaults to 1.
  asset_strategies?: Record<string, AssetStrategySpec> | null;
  discovery_metadata: Record<string, unknown> | null;
  plan_phases: MigrationPhaseInfo[];
  mvps: FunctionalMvpSummary[];
  created_at: string | null;
  updated_at: string | null;
}

export interface FunctionalMvpSummary {
  mvp_id: number;
  name: string;
  description: string | null;
  status: 'discovered' | 'refined' | 'in_progress' | 'migrated';
  priority: number;
  file_ids: string[];
  unit_ids: string[];
  depends_on_mvp_ids: number[];
  sp_references: SpReference[];
  metrics: MvpMetrics;
  current_phase: number;
  analysis_output?: { output: string } | null;  // On-demand deep analysis (V2)
  analysis_at?: string | null;                   // When analysis was last run
  phases: MigrationPhaseInfo[];
  created_at: string | null;
  updated_at: string | null;
}

export interface SpReference {
  sp_name: string;
  call_sites: SpCallSite[];
}

export interface SpCallSite {
  caller_name: string;
  file_path: string;
  line: number;
}

export interface MvpMetrics {
  cohesion: number;
  coupling: number;
  size: number;
  readiness: number;
}

export interface MigrationPhaseInfo {
  phase_id: string;
  phase_number: number;
  phase_type: MigrationPhaseType;
  status: 'pending' | 'running' | 'complete' | 'error';
  approved: boolean;
  approved_at: string | null;
  output_preview: string | null;
  mvp_id: number | null;
}

export interface MigrationPhaseOutput {
  phase_id: string;
  phase_number: number;
  phase_type: MigrationPhaseType;
  status: string;
  output: string | null;
  output_files: MigrationFile[];
  approved: boolean;
  approved_at: string | null;
  input_summary: string | null;
  mvp_id: number | null;
  phase_metadata: Record<string, unknown> | null;
}

export interface MigrationFile {
  file_path: string;
  language: string;
  content: string;
  is_sp_stub?: boolean;
  test_type?: 'unit' | 'integration' | 'equivalence' | 'sp_stub';
}

export interface MvpResolvedFile {
  file_id: string;
  file_path: string;
  language: string;
  line_count: number;
}

export interface MvpResolvedUnit {
  unit_id: string;
  name: string;
  qualified_name: string;
  unit_type: string;
  language: string;
  file_path: string;
  start_line: number;
  end_line: number;
  signature: string;
}

export interface ArchitectureMapping {
  source_path: string;
  source_class: string;
  target_path: string;
  target_class: string;
  changes: string;
}

export interface MvpDetail extends Omit<FunctionalMvpSummary, 'phases'> {
  files: MvpResolvedFile[];
  units: MvpResolvedUnit[];
  architecture_mapping?: ArchitectureMapping[];
  phases: Array<{
    phase_id: string;
    phase_number: number;
    phase_type: MigrationPhaseType;
    status: string;
    approved: boolean;
    approved_at: string | null;
    output_preview: string | null;
  }>;
}

export interface DiscoveryResult {
  plan_id: string;
  mvps: FunctionalMvpSummary[];
  sp_analysis: Record<string, unknown>;
  phase_output: {
    output: string;
    output_files: MigrationFile[];
  };
}

// ---- Asset Inventory types --------------------------------------------------

export type AssetStrategy =
  | 'version_upgrade'
  | 'framework_migration'
  | 'rewrite'
  | 'convert'
  | 'keep_as_is'
  | 'no_change';

export interface AssetSubType {
  unit_type: string;
  unit_count: number;
  file_count: number;
  sample_names: string[];
}

export interface AssetInventoryItem {
  language: string;
  file_count: number;
  unit_count: number;
  total_lines: number;
  sample_paths: string[];
  sub_types?: AssetSubType[];
}

export interface AssetStrategySpec {
  strategy: AssetStrategy;
  target: string | null;
  reason: string | null;
  lane_id?: string | null;
  sub_types?: Record<string, { strategy: AssetStrategy; lane_id?: string }>;
}

export interface MigrationLane {
  lane_id: string;
  display_name: string;
  source_frameworks: string[];
  target_frameworks: string[];
  confidence?: number;
}

export interface AssetInventoryResponse {
  inventory: AssetInventoryItem[];
  suggested_strategies: Record<string, AssetStrategySpec>;
  suggested_lanes?: Record<string, MigrationLane>;
  llm_refined: boolean;
}

// ---- Diff types -------------------------------------------------------------

export interface DiffSourceFile {
  file_path: string;
  language: string;
  content: string;
}

export interface DiffFileMapping {
  source_path: string;
  target_path: string;
  confidence: number;
}

export interface DiffContext {
  migrated_files: MigrationFile[];
  source_files: DiffSourceFile[];
  file_mapping: DiffFileMapping[];
}

export type DiffViewMode = 'side-by-side' | 'unified';

// ---- Understanding / Deep Analysis types ------------------------------------

export type EntryPointType =
  | 'http_endpoint' | 'message_handler' | 'scheduled_task'
  | 'cli_command' | 'event_listener' | 'startup_hook'
  | 'public_api' | 'unknown';

export type AnalysisTier = 'tier_1' | 'tier_2' | 'tier_3';

export interface EntryPoint {
  unit_id: string;
  name: string;
  qualified_name: string;
  file_path: string;
  entry_type: EntryPointType;
  language: string;
  detected_by: 'heuristic' | 'annotation' | 'both';
}

export interface AnalysisSummary {
  analysis_id: string;
  entry_unit_id: string;
  entry_name: string;
  entry_qualified_name: string;
  entry_file: string;
  entry_type: string;
  tier: AnalysisTier;
  total_units: number;
  total_tokens: number;
  confidence_score: number;
  coverage_pct: number;
  narrative: string | null;
  schema_version: number;
  analyzed_at: string | null;
}

export interface EvidenceRef {
  unit_id: string;
  qualified_name: string;
  file_path: string;
  start_line: number;
  end_line: number;
  snippet: string | null;
}

export interface ChainDetail {
  analysis_id: string;
  entry_point: {
    unit_id: string;
    name: string;
    qualified_name: string;
    file_path: string;
    entry_type: string;
  };
  tier: AnalysisTier;
  total_units: number;
  total_tokens: number;
  confidence_score: number;
  coverage_pct: number;
  narrative: string | null;
  schema_version: number;
  prompt_version: string;
  analyzed_at: string | null;
  result: {
    business_rules: Array<{ description: string; evidence: EvidenceRef[] }>;
    data_entities: Array<{ name: string; type: string; description: string; evidence: EvidenceRef[] }>;
    integrations: Array<{ name: string; type: string; description: string; evidence: EvidenceRef[] }>;
    side_effects: Array<{ description: string; severity: 'low' | 'medium' | 'high'; evidence: EvidenceRef[] }>;
    cross_cutting_concerns: string[];
  };
  units: AnalysisUnit[];
}

export interface AnalysisUnit {
  unit_id: string;
  name: string;
  qualified_name: string;
  unit_type: string;
  file_path: string;
  min_depth: number;
  path_count: number;
}

export interface AnalysisJobStatus {
  job_id: string;
  project_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: { total: number; completed: number };
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  retry_count: number;
  errors: Array<{ entry_point: string; error: string }> | null;
}

// ---- Diagram types ----------------------------------------------------------

export type DiagramType =
  | 'class' | 'package' | 'component'
  | 'sequence' | 'usecase' | 'activity' | 'deployment';

export type DiagramCategory = 'structural' | 'behavioral';

export interface DiagramResponse {
  diagram_type: DiagramType;
  category: DiagramCategory;
  puml: string;
  svg: string;
  title: string;
  cached: boolean;
  generated_at: string | null;
}

export interface DiagramAvailability {
  diagram_type: DiagramType;
  category: DiagramCategory;
  label: string;
  cached: boolean;
  generated_at: string | null;
}

// ---------------------------------------------------------------------------
// Analytics
// ---------------------------------------------------------------------------

export interface ProjectAnalytics {
  project: {
    name: string;
    file_count: number;
    total_lines: number;
    primary_language: string | null;
    languages: string[];
    ast_status: string;
    asg_status: string;
    deep_analysis_status: string;
  };
  code_breakdown: {
    units_by_type: Record<string, number>;
    edges_by_type: Record<string, number>;
    files_by_language: Record<string, number>;
  };
  migration: {
    plan_count: number;
    active_plan: {
      plan_id: string;
      status: string;
      pipeline_version: string;
      migration_lane: string | null;
      mvps: Record<string, number>;
      phases: Record<string, number>;
      avg_confidence: number | null;
      gates_pass_rate: number | null;
    } | null;
  };
  understanding: {
    analyses_count: number;
    entry_points_detected: number;
  };
  queries: {
    total: number;
  };
  llm: {
    total_calls?: number;
    total_tokens_in?: number;
    total_tokens_out?: number;
    total_latency_ms?: number;
    avg_latency_ms?: number;
    errors?: number;
    retries?: number;
    calls_by_purpose?: Record<string, number>;
    model?: string;
    estimated_cost_usd?: number;
  };
}
