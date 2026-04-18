/**
 * MCP Tool Explorer
 *
 * Interactive demo page for testing CodeLoom's 22 MCP tools.
 * Calls the MCP streamable-HTTP endpoint directly (/mcp) using
 * JSON-RPC over SSE transport.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Play,
  Loader2,
  ChevronDown,
  ChevronRight,
  Terminal,
  Clock,
  AlertCircle,
  CheckCircle2,
  Wifi,
  WifiOff,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Layout } from '../components/Layout.tsx';
import { listProjects } from '../services/api.ts';
import type { Project } from '../types/index.ts';

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

interface ToolParam {
  name: string;
  type: 'string' | 'number' | 'textarea' | 'project_id' | 'json';
  required: boolean;
  description: string;
  default?: string | number;
  placeholder?: string;
}

interface ToolDef {
  name: string;
  description: string;
  params: ToolParam[];
}

interface ToolCategory {
  label: string;
  tools: ToolDef[];
}

const TOOL_CATEGORIES: ToolCategory[] = [
  {
    label: 'Code Intelligence',
    tools: [
      {
        name: 'codeloom_chat',
        description: 'Ask a question about a codebase using RAG retrieval',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Project UUID' },
          { name: 'query', type: 'textarea', required: true, description: 'Natural language question', placeholder: 'How does authentication work?' },
          { name: 'max_sources', type: 'number', required: false, description: 'Max source chunks (default 6)', default: 6 },
        ],
      },
      {
        name: 'codeloom_blast_radius',
        description: 'Analyze impact of changes to a code unit',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Project UUID' },
          { name: 'unit_name', type: 'string', required: true, description: 'Code unit name', placeholder: 'e.g. create_app' },
          { name: 'depth', type: 'number', required: false, description: 'Transitive depth (default 3)', default: 3 },
        ],
      },
      {
        name: 'codeloom_search_codebase',
        description: 'Semantic search over an ingested codebase',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Project UUID' },
          { name: 'query', type: 'textarea', required: true, description: 'Search query', placeholder: 'database connection pooling' },
          { name: 'top_k', type: 'number', required: false, description: 'Number of results (default 5)', default: 5 },
        ],
      },
      {
        name: 'codeloom_search_knowledge',
        description: 'Search uploaded knowledge base documents',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Knowledge project UUID' },
          { name: 'query', type: 'textarea', required: true, description: 'Search query', placeholder: 'CICS transaction processing' },
          { name: 'top_k', type: 'number', required: false, description: 'Number of results (default 5)', default: 5 },
        ],
      },
    ],
  },
  {
    label: 'Project & Source',
    tools: [
      {
        name: 'codeloom_list_projects',
        description: 'List all projects with metadata',
        params: [
          { name: 'page', type: 'number', required: false, description: 'Page number', default: 1 },
          { name: 'page_size', type: 'number', required: false, description: 'Results per page', default: 20 },
        ],
      },
      {
        name: 'codeloom_get_project_intel',
        description: 'Get comprehensive project intelligence (AST, ASG, analysis)',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Project UUID' },
        ],
      },
      {
        name: 'codeloom_list_units',
        description: 'List all code units in a project (paginated)',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Project UUID' },
          { name: 'language', type: 'string', required: false, description: 'Filter by language', placeholder: 'python' },
          { name: 'unit_type', type: 'string', required: false, description: 'Filter by type', placeholder: 'function' },
          { name: 'page', type: 'number', required: false, description: 'Page number', default: 1 },
          { name: 'page_size', type: 'number', required: false, description: 'Results per page', default: 50 },
        ],
      },
      {
        name: 'codeloom_get_source_unit',
        description: 'Get full source code and metadata for a code unit',
        params: [
          { name: 'unit_id', type: 'string', required: true, description: 'Code unit UUID', placeholder: 'e.g. abc-123-def' },
        ],
      },
      {
        name: 'codeloom_get_import_graph',
        description: 'Analyze import relationships and shared files',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Project UUID' },
          { name: 'shared_threshold', type: 'number', required: false, description: 'Min importers to flag as shared (default 3)', default: 3 },
        ],
      },
    ],
  },
  {
    label: 'Migration',
    tools: [
      {
        name: 'codeloom_get_lane_info',
        description: 'Detect migration lane for a source/target pair',
        params: [
          { name: 'source_framework', type: 'string', required: true, description: 'Source framework', placeholder: 'struts' },
          { name: 'target_stack_json', type: 'string', required: true, description: 'Target stack JSON', placeholder: '{"framework": "spring_boot"}' },
        ],
      },
      {
        name: 'codeloom_save_plan',
        description: 'Save a migration plan to CodeLoom DB',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Source project UUID' },
          { name: 'target_brief', type: 'string', required: true, description: 'One-line migration target', placeholder: 'Migrate to TypeScript Express REST API' },
          { name: 'target_stack', type: 'string', required: true, description: 'Target stack JSON', placeholder: '{"languages":["typescript"],"frameworks":["expressjs"]}' },
        ],
      },
      {
        name: 'codeloom_save_mvps',
        description: 'Save MVP cluster definitions to DB',
        params: [
          { name: 'plan_id', type: 'string', required: true, description: 'Migration plan UUID' },
          { name: 'mvps', type: 'json', required: true, description: 'MVP definitions JSON array', placeholder: '[{"name":"Foundation","priority":0,"source_file_paths":["main.py"]}]' },
        ],
      },
      {
        name: 'codeloom_start_transform',
        description: 'Mark MVP transform phase as in-progress',
        params: [
          { name: 'plan_id', type: 'string', required: true, description: 'Plan UUID' },
          { name: 'mvp_id', type: 'number', required: true, description: 'MVP ID' },
        ],
      },
      {
        name: 'codeloom_complete_transform',
        description: 'Record MVP transform outcome',
        params: [
          { name: 'plan_id', type: 'string', required: true, description: 'Plan UUID' },
          { name: 'mvp_id', type: 'number', required: true, description: 'MVP ID' },
          { name: 'transform_summary', type: 'textarea', required: true, description: 'Markdown summary of transform', placeholder: '## Transform Complete\n- 5 files generated...' },
          { name: 'status', type: 'string', required: false, description: '"complete" or "failed"', default: 'complete' },
        ],
      },
      {
        name: 'codeloom_validate_output',
        description: 'Run ground truth advisory validation on output',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Project UUID' },
          { name: 'phase_type', type: 'string', required: true, description: 'Phase type', placeholder: 'transform' },
          { name: 'output_text', type: 'textarea', required: true, description: 'Output text to validate' },
        ],
      },
      {
        name: 'codeloom_save_accuracy_report',
        description: 'Persist migration accuracy report to DB',
        params: [
          { name: 'plan_id', type: 'string', required: true, description: 'Plan UUID' },
          { name: 'overall_score', type: 'number', required: true, description: 'Score before fixes (0-100)' },
          { name: 'fixed_score', type: 'number', required: true, description: 'Score after fixes (0-100)' },
          { name: 'fixes_applied', type: 'number', required: true, description: 'Fixes applied count' },
          { name: 'fixes_pending', type: 'number', required: true, description: 'Fixes pending count' },
          { name: 'report_markdown', type: 'textarea', required: true, description: 'Full report markdown' },
        ],
      },
    ],
  },
  {
    label: 'MVP Context',
    tools: [
      {
        name: 'codeloom_list_mvps',
        description: 'List all MVPs for a migration plan',
        params: [
          { name: 'plan_id', type: 'string', required: true, description: 'Migration plan UUID' },
        ],
      },
      {
        name: 'codeloom_get_mvp_context',
        description: 'Get rich context for an MVP (source, lane, analysis)',
        params: [
          { name: 'plan_id', type: 'string', required: true, description: 'Plan UUID' },
          { name: 'mvp_id', type: 'number', required: true, description: 'MVP ID' },
        ],
      },
      {
        name: 'codeloom_get_compiled_context',
        description: 'Get fully compiled migration context for an MVP phase',
        params: [
          { name: 'plan_id', type: 'string', required: true, description: 'Plan UUID' },
          { name: 'mvp_id', type: 'number', required: true, description: 'MVP ID' },
          { name: 'phase_type', type: 'string', required: false, description: 'Phase type (transform, analyze, design, test)', placeholder: 'transform' },
          { name: 'token_budget', type: 'number', required: false, description: 'Override token budget' },
        ],
      },
    ],
  },
  {
    label: 'Documentation',
    tools: [
      {
        name: 'codeloom_generate_reverse_doc',
        description: 'Generate 9-chapter reverse engineering document',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Project UUID' },
          { name: 'chapters', type: 'json', required: false, description: 'Chapter numbers to generate (e.g. [1,2,3])', placeholder: '[1, 2, 3]' },
        ],
      },
      {
        name: 'codeloom_get_reverse_doc',
        description: 'Get a reverse engineering document (by chapter)',
        params: [
          { name: 'doc_id', type: 'string', required: true, description: 'Document UUID' },
          { name: 'chapter', type: 'number', required: false, description: 'Chapter number (1-14)' },
        ],
      },
      {
        name: 'codeloom_list_reverse_docs',
        description: 'List all reverse engineering docs for a project',
        params: [
          { name: 'project_id', type: 'project_id', required: true, description: 'Project UUID' },
        ],
      },
    ],
  },
];

// Flatten for quick lookup
const ALL_TOOLS: ToolDef[] = TOOL_CATEGORIES.flatMap((c) => c.tools);

// ---------------------------------------------------------------------------
// MCP caller
// ---------------------------------------------------------------------------

interface MCPResult {
  data: unknown;
  durationMs: number;
  error?: string;
}

/** Shape of a JSON-RPC response from the MCP server. */
interface JsonRpcResponse {
  jsonrpc?: string;
  id?: number | string;
  result?: {
    content?: Array<{ type: string; text?: string }>;
    [key: string]: unknown;
  };
  error?: { code?: number; message?: string; data?: unknown };
  [key: string]: unknown;
}

/** Compute default form values for a tool's params. */
function buildDefaults(tool: ToolDef): Record<string, string> {
  const defaults: Record<string, string> = {};
  for (const p of tool.params) {
    if (p.default !== undefined) {
      defaults[p.name] = String(p.default);
    }
  }
  return defaults;
}

/**
 * Negotiate an MCP session and call a tool via streamable-HTTP transport.
 *
 * The MCP streamable-HTTP spec requires:
 * 1. POST an "initialize" request to get a session ID (from Mcp-Session header)
 * 2. POST a "tools/call" request with that session ID
 * 3. Parse SSE response for the result
 */
async function callMCPTool(toolName: string, args: Record<string, unknown>): Promise<MCPResult> {
  const start = performance.now();

  try {
    // Step 1: Initialize session
    const initRes = await fetch('/mcp', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
      },
      credentials: 'include',
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'initialize',
        params: {
          protocolVersion: '2025-03-26',
          capabilities: {},
          clientInfo: { name: 'codeloom-mcp-demo', version: '1.0.0' },
        },
      }),
    });

    // Extract session ID from response header
    const sessionId = initRes.headers.get('mcp-session-id');
    const initText = await initRes.text();

    // Parse SSE lines to find the initialize result
    const initData = parseSSEResponse(initText);
    if (!initData || initData.error) {
      throw new Error(
        (initData?.error?.message as string | undefined) || 'Failed to initialize MCP session',
      );
    }

    // Step 2: Send initialized notification (required by protocol)
    const notifHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream',
    };
    if (sessionId) notifHeaders['Mcp-Session-Id'] = sessionId;

    await fetch('/mcp', {
      method: 'POST',
      headers: notifHeaders,
      credentials: 'include',
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'notifications/initialized',
      }),
    });

    // Step 3: Call the tool
    const callHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream',
    };
    if (sessionId) callHeaders['Mcp-Session-Id'] = sessionId;

    const callRes = await fetch('/mcp', {
      method: 'POST',
      headers: callHeaders,
      credentials: 'include',
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 2,
        method: 'tools/call',
        params: { name: toolName, arguments: args },
      }),
    });

    const callText = await callRes.text();
    const callData = parseSSEResponse(callText);

    const durationMs = Math.round(performance.now() - start);

    if (callData?.error) {
      return {
        data: null,
        durationMs,
        error: callData.error.message || JSON.stringify(callData.error),
      };
    }

    // Extract text content from MCP response
    if (callData?.result?.content) {
      const textContent = callData.result.content.find(
        (c: { type: string; text?: string }) => c.type === 'text',
      );
      if (textContent?.text) {
        try {
          return { data: JSON.parse(textContent.text), durationMs };
        } catch {
          return { data: textContent.text, durationMs };
        }
      }
    }

    return { data: callData?.result ?? callData, durationMs };
  } catch (err) {
    const durationMs = Math.round(performance.now() - start);
    return {
      data: null,
      durationMs,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/**
 * Parse an SSE response body. Finds the last "data: {...}" line containing
 * a JSON-RPC response (has "id" field).
 */
function parseSSEResponse(text: string): JsonRpcResponse | null {
  const lines = text.split('\n');
  let lastData: JsonRpcResponse | null = null;

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      try {
        const parsed = JSON.parse(line.slice(6)) as JsonRpcResponse;
        if (parsed.id !== undefined || parsed.result || parsed.error) {
          lastData = parsed;
        }
      } catch {
        // skip malformed lines
      }
    }
  }

  // If no SSE lines found, try parsing the whole text as JSON (non-SSE response)
  if (!lastData) {
    try {
      const parsed = JSON.parse(text) as JsonRpcResponse;
      if (parsed.id !== undefined || parsed.result || parsed.error) {
        return parsed;
      }
    } catch {
      // not JSON either
    }
  }

  return lastData;
}

// ---------------------------------------------------------------------------
// Result display — renders markdown for text, structured cards for data
// ---------------------------------------------------------------------------

function ResultDisplay({ data, toolName }: { data: any; toolName: string }) {
  if (!data) return null;

  // Extract markdown-renderable content from common response shapes
  const response = data.response || data.explanation || data.content || null;
  const sources = data.sources || [];
  const chapters = data.chapters || [];
  const dependents = data.dependents || [];
  const hasStructuredData = data.unit || data.total_dependents !== undefined || data.impact_level || chapters.length > 0;

  return (
    <div className="space-y-4">
      {/* Structured summary cards */}
      {hasStructuredData && (
        <div className="rounded-lg border border-[#2a3352] bg-[#131b2e] p-4 space-y-3">
          {data.unit && (
            <div className="flex items-center gap-3">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-400">Unit</span>
              <span className="text-sm text-white font-semibold">{data.unit.name}</span>
              <span className="px-1.5 py-0.5 rounded text-[9px] bg-[#1e2740] text-gray-400 uppercase">{data.unit.unit_type}</span>
              <span className="px-1.5 py-0.5 rounded text-[9px] bg-[#1e2740] text-gray-400 uppercase">{data.unit.language}</span>
            </div>
          )}
          {data.impact_level && (
            <div className="flex gap-4 text-xs">
              <span className={`font-bold ${data.impact_level === 'high' ? 'text-red-400' : data.impact_level === 'medium' ? 'text-amber-400' : 'text-emerald-400'}`}>
                Impact: {data.impact_level.toUpperCase()}
              </span>
              <span className="text-gray-400">{data.total_dependents} dependents</span>
              <span className="text-gray-400">{data.direct_dependents} direct</span>
              <span className="text-gray-400">{data.transitive_dependents} transitive</span>
              <span className="text-gray-400">{data.files_affected} files</span>
            </div>
          )}
          {/* Chapter list (reverse doc metadata) */}
          {chapters.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs font-bold uppercase tracking-wider text-blue-400 mb-2">Chapters</p>
              {chapters.map((ch: any) => (
                <div key={ch.chapter} className="flex items-center gap-3 text-xs py-1 border-b border-[#1e2740] last:border-0">
                  <span className="text-blue-400 font-mono w-6">Ch{ch.chapter}</span>
                  <span className="text-white flex-1">{ch.title}</span>
                  <span className="text-gray-500">{ch.words?.toLocaleString()} words</span>
                </div>
              ))}
              {data.hint && <p className="text-[10px] text-gray-500 mt-2 italic">{data.hint}</p>}
            </div>
          )}
          {/* Dependents list */}
          {dependents.length > 0 && (
            <div className="space-y-1 max-h-48 overflow-auto">
              <p className="text-xs font-bold uppercase tracking-wider text-blue-400 mb-2">Dependents</p>
              {dependents.slice(0, 20).map((d: any, i: number) => (
                <div key={i} className="flex items-center gap-2 text-xs py-0.5">
                  <span className="text-gray-500 w-4">{i + 1}</span>
                  <span className="text-white font-medium">{d.name}</span>
                  <span className="px-1 py-0.5 rounded text-[9px] bg-[#1e2740] text-gray-400">{d.unit_type}</span>
                  <span className="text-gray-500">via {d.edge_type}</span>
                  <span className="text-gray-500">depth={d.depth}</span>
                </div>
              ))}
              {dependents.length > 20 && <p className="text-[10px] text-gray-500">... and {dependents.length - 20} more</p>}
            </div>
          )}
        </div>
      )}

      {/* Markdown response (chat, blast radius explanation, chapter content) */}
      {response && (
        <div className="rounded-lg border border-[#2a3352] bg-[#131b2e] p-5">
          <div className="prose prose-invert prose-sm max-w-none
            prose-headings:text-white prose-headings:font-semibold
            prose-h2:text-lg prose-h2:mt-4 prose-h2:mb-2
            prose-h3:text-base prose-h3:mt-3 prose-h3:mb-1
            prose-p:text-gray-300 prose-p:leading-relaxed
            prose-li:text-gray-300
            prose-strong:text-white
            prose-code:text-emerald-300 prose-code:bg-[#0a0f1e] prose-code:px-1 prose-code:rounded
            prose-pre:bg-[#0a0f1e] prose-pre:border prose-pre:border-[#2a3352]
            prose-a:text-blue-400
            prose-table:text-xs
            prose-th:text-gray-400 prose-th:font-medium
            prose-td:text-gray-300
          ">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown>
          </div>
        </div>
      )}

      {/* Sources */}
      {sources.length > 0 && (
        <div className="rounded-lg border border-[#2a3352] bg-[#131b2e] p-4">
          <p className="text-xs font-bold uppercase tracking-wider text-blue-400 mb-2">Sources ({sources.length})</p>
          <div className="space-y-1">
            {sources.map((s: any, i: number) => (
              <div key={i} className="flex items-center gap-2 text-xs py-1 border-b border-[#1e2740] last:border-0">
                <span className="text-blue-400 font-mono w-4">{i + 1}</span>
                <span className="text-white">{s.filename || s.file_path || 'unknown'}</span>
                {s.score && <span className="text-gray-500 ml-auto">{(s.score * 100).toFixed(0)}%</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Raw JSON fallback — for tools without markdown response */}
      {!response && !hasStructuredData && sources.length === 0 && (
        <pre className="max-h-[500px] overflow-auto rounded-lg border border-[#2a3352] bg-[#0a0f1e] p-4 text-xs text-emerald-300 font-mono leading-relaxed whitespace-pre-wrap break-words">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}

      {/* Always show raw JSON in a collapsible section */}
      {(response || hasStructuredData) && (
        <details className="group">
          <summary className="cursor-pointer text-[10px] text-gray-500 hover:text-gray-400">
            Show raw JSON
          </summary>
          <pre className="mt-2 max-h-[300px] overflow-auto rounded-lg border border-[#2a3352] bg-[#0a0f1e] p-3 text-[10px] text-gray-500 font-mono whitespace-pre-wrap break-words">
            {JSON.stringify(data, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function MCPDemo() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedTool, setSelectedTool] = useState<ToolDef>(ALL_TOOLS[0]);
  const [formValues, setFormValues] = useState<Record<string, string>>(() => buildDefaults(ALL_TOOLS[0]));
  const [result, setResult] = useState<MCPResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set());
  const [mcpAvailable, setMcpAvailable] = useState<boolean | null>(null);


  // Fetch projects for dropdown
  useEffect(() => {
    listProjects()
      .then(setProjects)
      .catch(() => setProjects([]));
  }, []);

  // Check MCP endpoint availability via JSON-RPC initialize
  useEffect(() => {
    fetch('/mcp', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' },
      body: JSON.stringify({
        jsonrpc: '2.0', id: 0, method: 'initialize',
        params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'mcp-demo', version: '1.0' } },
      }),
    })
      .then((r) => setMcpAvailable(r.ok))
      .catch(() => setMcpAvailable(false));
  }, []);

  const selectTool = useCallback((tool: ToolDef) => {
    setSelectedTool(tool);
    setFormValues(buildDefaults(tool));
    setResult(null);
  }, []);


  const toggleCategory = useCallback((label: string) => {
    setCollapsedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  }, []);

  const handleRun = useCallback(async () => {
    setIsRunning(true);
    setResult(null);

    // Build arguments
    const args: Record<string, unknown> = {};
    for (const p of selectedTool.params) {
      const val = formValues[p.name];
      if (val === undefined || val === '') continue;

      if (p.type === 'number') {
        args[p.name] = Number(val);
      } else if (p.type === 'json') {
        try {
          args[p.name] = JSON.parse(val);
        } catch {
          args[p.name] = val;
        }
      } else {
        args[p.name] = val;
      }
    }

    const res = await callMCPTool(selectedTool.name, args);
    setResult(res);
    setIsRunning(false);
  }, [selectedTool, formValues]);

  // Check if all required params are filled
  const canRun = useMemo(() => {
    return selectedTool.params
      .filter((p) => p.required)
      .every((p) => {
        const val = formValues[p.name];
        return val !== undefined && val !== '';
      });
  }, [selectedTool, formValues]);

  return (
    <Layout>
      <div className="flex flex-1 overflow-hidden">
        {/* Left Panel: Tool selector */}
        <aside className="flex w-[300px] shrink-0 flex-col border-r border-void-surface bg-void-light/50 overflow-y-auto">
          {/* Panel header */}
          <div className="sticky top-0 z-10 border-b border-void-surface bg-void-light/80 px-4 py-3 backdrop-blur-sm">
            <div className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-glow" />
              <span className="text-xs font-semibold text-text">22 Tools</span>
            </div>
          </div>

          {/* Tool categories */}
          <div className="flex-1 p-2">
            {TOOL_CATEGORIES.map((cat) => {
              const isCollapsed = collapsedCategories.has(cat.label);
              return (
                <div key={cat.label} className="mb-1">
                  <button
                    onClick={() => toggleCategory(cat.label)}
                    className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-text-dim hover:text-text transition-colors"
                  >
                    {isCollapsed ? (
                      <ChevronRight className="h-3 w-3" />
                    ) : (
                      <ChevronDown className="h-3 w-3" />
                    )}
                    {cat.label}
                    <span className="ml-auto text-[10px] text-text-dim/60">{cat.tools.length}</span>
                  </button>

                  {!isCollapsed && (
                    <div className="ml-1 space-y-0.5">
                      {cat.tools.map((tool) => (
                        <button
                          key={tool.name}
                          onClick={() => selectTool(tool)}
                          className={`w-full rounded-md px-2.5 py-1.5 text-left transition-colors ${
                            selectedTool.name === tool.name
                              ? 'bg-glow/10 border border-glow/20 text-glow'
                              : 'text-text-muted hover:bg-void-surface hover:text-text'
                          }`}
                        >
                          <div className="text-xs font-medium truncate">
                            {tool.name.replace('codeloom_', '')}
                          </div>
                          <div className="text-[10px] text-text-dim leading-tight mt-0.5 line-clamp-1">
                            {tool.description}
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </aside>

        {/* Right Panel: Tool execution */}
        <div className="flex flex-1 flex-col overflow-y-auto bg-void">
          {/* Page header */}
          <div className="border-b border-void-surface px-6 py-5">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-lg font-semibold text-text">MCP Tool Explorer</h1>
                <p className="text-xs text-text-dim mt-0.5">
                  Test CodeLoom's 22 MCP tools interactively
                </p>
              </div>
              {/* Connection indicator */}
              <div
                className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-[10px] font-medium ${
                  mcpAvailable === true
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : mcpAvailable === false
                      ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                      : 'bg-void-surface text-text-dim border border-void-surface'
                }`}
              >
                {mcpAvailable === true ? (
                  <Wifi className="h-3 w-3" />
                ) : mcpAvailable === false ? (
                  <WifiOff className="h-3 w-3" />
                ) : (
                  <Loader2 className="h-3 w-3 animate-spin" />
                )}
                {mcpAvailable === true
                  ? 'Streamable HTTP'
                  : mcpAvailable === false
                    ? 'MCP Unavailable'
                    : 'Checking...'}
              </div>
            </div>
          </div>

          {/* Tool detail area */}
          <div className="flex-1 p-6">
            {/* Tool header */}
            <div className="mb-5">
              <h2 className="text-sm font-semibold text-glow font-mono">
                {selectedTool.name}
              </h2>
              <p className="text-xs text-text-muted mt-1">{selectedTool.description}</p>
            </div>

            {/* Input form */}
            <div className="space-y-3 mb-5">
              {selectedTool.params.map((param) => (
                <div key={param.name}>
                  <label className="mb-1 flex items-center gap-1.5 text-xs text-text-muted">
                    <span className="font-medium text-text">
                      {param.name}
                    </span>
                    {param.required && (
                      <span className="text-[9px] text-red-400">required</span>
                    )}
                    <span className="text-text-dim">-- {param.description}</span>
                  </label>

                  {param.type === 'project_id' ? (
                    <select
                      value={formValues[param.name] ?? ''}
                      onChange={(e) =>
                        setFormValues((prev) => ({ ...prev, [param.name]: e.target.value }))
                      }
                      className="w-full rounded-lg border border-void-surface bg-void-light px-3 py-2 text-xs text-text focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50"
                    >
                      <option value="">Select a project...</option>
                      {projects.map((p) => (
                        <option key={p.project_id} value={p.project_id}>
                          {p.name} ({p.file_count} files, {p.primary_language ?? 'unknown'})
                        </option>
                      ))}
                    </select>
                  ) : param.type === 'textarea' ? (
                    <textarea
                      value={formValues[param.name] ?? ''}
                      onChange={(e) =>
                        setFormValues((prev) => ({ ...prev, [param.name]: e.target.value }))
                      }
                      placeholder={param.placeholder}
                      rows={3}
                      className="w-full rounded-lg border border-void-surface bg-void-light px-3 py-2 text-xs text-text font-mono placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50 resize-y"
                    />
                  ) : param.type === 'json' ? (
                    <textarea
                      value={formValues[param.name] ?? ''}
                      onChange={(e) =>
                        setFormValues((prev) => ({ ...prev, [param.name]: e.target.value }))
                      }
                      placeholder={param.placeholder}
                      rows={4}
                      className="w-full rounded-lg border border-void-surface bg-void-light px-3 py-2 text-xs text-text font-mono placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50 resize-y"
                    />
                  ) : (
                    <input
                      type={param.type === 'number' ? 'number' : 'text'}
                      value={formValues[param.name] ?? ''}
                      onChange={(e) =>
                        setFormValues((prev) => ({ ...prev, [param.name]: e.target.value }))
                      }
                      placeholder={param.placeholder ?? (param.default !== undefined ? String(param.default) : '')}
                      className="w-full rounded-lg border border-void-surface bg-void-light px-3 py-2 text-xs text-text font-mono placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50"
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Run button */}
            <button
              onClick={handleRun}
              disabled={isRunning || !canRun || mcpAvailable === false}
              className={`flex items-center gap-2 rounded-lg px-5 py-2.5 text-xs font-semibold transition-all ${
                isRunning || !canRun || mcpAvailable === false
                  ? 'bg-void-surface text-text-dim cursor-not-allowed'
                  : 'bg-glow text-white hover:bg-glow-dim shadow-glow/20 shadow-lg'
              }`}
            >
              {isRunning ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              {isRunning ? 'Running...' : 'Run Tool'}
            </button>

            {/* Result area */}
            {result && (
              <div className="mt-6">
                {/* Status bar */}
                <div className="flex items-center gap-3 mb-2">
                  {result.error ? (
                    <div className="flex items-center gap-1.5 text-red-400">
                      <AlertCircle className="h-3.5 w-3.5" />
                      <span className="text-xs font-medium">Error</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 text-emerald-400">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      <span className="text-xs font-medium">Success</span>
                    </div>
                  )}
                  <div className="flex items-center gap-1 text-text-dim">
                    <Clock className="h-3 w-3" />
                    <span className="text-[10px]">{result.durationMs}ms</span>
                  </div>
                </div>

                {/* Error message */}
                {result.error && (
                  <div className="mb-3 rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-3">
                    <p className="text-xs text-red-400 font-mono">{result.error}</p>
                  </div>
                )}

                {/* Rich result display */}
                {result.data !== null && (
                  <ResultDisplay data={result.data} toolName={selectedTool?.name ?? ''} />
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}
