/**
 * AgentStepCard — renders a single agent execution step.
 *
 * Step types:
 * - thinking: Italic text with brain indicator
 * - tool_call: Tool name badge + collapsible args
 * - tool_result: Duration badge + collapsible content (indented under tool_call)
 * - output: Full markdown rendering with syntax highlighting
 * - error: Red border with error message
 */

import { useState } from 'react';
import {
  Brain,
  Wrench,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  FileText,
  Clock,
} from 'lucide-react';
import type {
  ThinkingEvent,
  ToolCallEvent,
  ToolResultEvent,
  OutputEvent,
  AgentErrorEvent,
} from '../../types/index.ts';

/* ── Tool category colors ─────────────────────────────────────── */

const TOOL_COLORS: Record<string, string> = {
  read_source_file: 'bg-nebula/15 text-nebula-bright border-nebula/30',
  get_unit_details: 'bg-nebula/15 text-nebula-bright border-nebula/30',
  get_source_code: 'bg-nebula/15 text-nebula-bright border-nebula/30',
  get_functional_context: 'bg-glow/15 text-glow border-glow/30',
  get_dependencies: 'bg-glow/15 text-glow border-glow/30',
  get_module_graph: 'bg-glow/15 text-glow border-glow/30',
  get_deep_analysis: 'bg-glow/15 text-glow border-glow/30',
  search_codebase: 'bg-warning/15 text-warning border-warning/30',
  lookup_framework_docs: 'bg-success/15 text-success border-success/30',
  validate_syntax: 'bg-warning/15 text-warning border-warning/30',
};

const DEFAULT_TOOL_COLOR = 'bg-void-surface/50 text-text-dim border-void-surface';

/* ── Step type components ─────────────────────────────────────── */

export function ThinkingCard({ event }: { event: ThinkingEvent }) {
  return (
    <div className="flex gap-2.5 py-2">
      <Brain className="h-4 w-4 text-text-dim shrink-0 mt-0.5" />
      <p className="text-xs text-text-muted italic leading-relaxed">
        {event.content}
      </p>
    </div>
  );
}

export function ToolCallCard({
  event,
  result,
}: {
  event: ToolCallEvent;
  result?: ToolResultEvent;
}) {
  const [expanded, setExpanded] = useState(true);
  const colorClass = TOOL_COLORS[event.tool] ?? DEFAULT_TOOL_COLOR;

  return (
    <div className="space-y-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left py-1.5 group"
      >
        <Wrench className="h-3.5 w-3.5 text-text-dim shrink-0" />
        <span
          className={`inline-flex items-center rounded-md border px-1.5 py-0.5 text-[10px] font-semibold ${colorClass}`}
        >
          {event.tool}
        </span>

        {result && (
          <span className="inline-flex items-center gap-1 text-[10px] text-text-dim">
            <Clock className="h-3 w-3" />
            {(result.duration_ms / 1000).toFixed(1)}s
            {result.truncated && (
              <span className="text-warning">(truncated)</span>
            )}
          </span>
        )}

        <span className="ml-auto">
          {expanded ? (
            <ChevronDown className="h-3 w-3 text-text-dim" />
          ) : (
            <ChevronRight className="h-3 w-3 text-text-dim" />
          )}
        </span>
      </button>

      {expanded && (
        <div className="ml-6 space-y-2">
          {/* Args */}
          {Object.keys(event.args).length > 0 && (
            <div className="rounded-lg bg-void-surface/30 p-2.5">
              <p className="text-[10px] font-medium text-text-dim mb-1">Args</p>
              <pre className="text-[10px] text-text-muted font-mono whitespace-pre-wrap break-all">
                {JSON.stringify(event.args, null, 2)}
              </pre>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="rounded-lg bg-void-surface/30 p-2.5">
              <p className="text-[10px] font-medium text-text-dim mb-1">Result</p>
              <pre className="text-[10px] text-text-muted font-mono whitespace-pre-wrap break-all max-h-60 overflow-y-auto">
                {result.result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function OutputCard({ event }: { event: OutputEvent }) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="border border-glow/20 rounded-xl bg-glow/5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left px-4 py-2.5"
      >
        <FileText className="h-4 w-4 text-glow shrink-0" />
        <span className="text-xs font-semibold text-glow">Final Output</span>
        <span className="ml-auto">
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5 text-text-dim" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-text-dim" />
          )}
        </span>
      </button>
      {expanded && (
        <div className="px-4 pb-4">
          <pre className="text-xs text-text-muted font-mono whitespace-pre-wrap break-words max-h-96 overflow-y-auto leading-relaxed">
            {event.content}
          </pre>
        </div>
      )}
    </div>
  );
}

export function ErrorCard({ event }: { event: AgentErrorEvent }) {
  return (
    <div className="flex gap-2.5 py-2 px-3 rounded-lg border border-danger/30 bg-danger/5">
      <AlertTriangle className="h-4 w-4 text-danger shrink-0 mt-0.5" />
      <div>
        <p className="text-xs font-medium text-danger">{event.error}</p>
        {event.recoverable && (
          <p className="text-[10px] text-text-dim mt-0.5">This error may be recoverable.</p>
        )}
      </div>
    </div>
  );
}

export function DoneCard({
  turnsUsed,
  toolsCalled,
  totalMs,
}: {
  turnsUsed: number;
  toolsCalled: number;
  totalMs: number;
}) {
  return (
    <div className="flex items-center gap-3 py-2 px-3 rounded-lg bg-success/5 border border-success/20">
      <CheckCircle2 className="h-4 w-4 text-success shrink-0" />
      <span className="text-xs text-text-muted">
        Done: {turnsUsed} turn{turnsUsed !== 1 ? 's' : ''},{' '}
        {toolsCalled} tool{toolsCalled !== 1 ? 's' : ''},{' '}
        {(totalMs / 1000).toFixed(1)}s
      </span>
    </div>
  );
}
