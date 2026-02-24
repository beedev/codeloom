/**
 * AgentExecutionPanel — real-time visualization of agentic migration execution.
 *
 * Connects to the SSE stream via executeMigrationPhaseAgentic(), renders a
 * progress bar (turn N / max_turns), and shows step cards as events arrive.
 * ToolResultEvents are paired with their ToolCallEvent via call_id.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { Bot, XCircle, Loader2 } from 'lucide-react';
import {
  ThinkingCard,
  ToolCallCard,
  OutputCard,
  ErrorCard,
  DoneCard,
} from './AgentStepCard.tsx';
import { executeMigrationPhaseAgentic } from '../../services/api.ts';
import type {
  AgentEvent,
  AgentStartEvent,
  ThinkingEvent,
  ToolCallEvent,
  ToolResultEvent,
  OutputEvent,
  AgentDoneEvent,
  AgentErrorEvent,
} from '../../types/index.ts';

/* ── Discriminated-union step model ───────────────────────────── */

type Step =
  | { kind: 'thinking'; event: ThinkingEvent }
  | { kind: 'tool_call'; event: ToolCallEvent; result?: ToolResultEvent }
  | { kind: 'output'; event: OutputEvent }
  | { kind: 'error'; event: AgentErrorEvent };

/* ── Props ────────────────────────────────────────────────────── */

interface AgentExecutionPanelProps {
  planId: string;
  phaseNumber: number;
  phaseType: string;
  mvpId?: number | null;
  maxTurns?: number;
  onComplete: () => void;
  onCancel: () => void;
}

/* ── Phase label map ──────────────────────────────────────────── */

const PHASE_LABELS: Record<string, string> = {
  discovery: 'Discovery Analysis',
  architecture: 'Target Architecture',
  analyze: 'MVP Analysis',
  design: 'Detailed Design',
  transform: 'Code Transform',
  test: 'Test Generation',
};

/* ── Component ────────────────────────────────────────────────── */

export function AgentExecutionPanel({
  planId,
  phaseNumber,
  phaseType,
  mvpId,
  maxTurns = 10,
  onComplete,
  onCancel,
}: AgentExecutionPanelProps) {
  const [steps, setSteps] = useState<Step[]>([]);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [maxTurnsActual, setMaxTurnsActual] = useState(maxTurns);
  const [toolCount, setToolCount] = useState(0);
  const [doneEvent, setDoneEvent] = useState<AgentDoneEvent | null>(null);
  const [isRunning, setIsRunning] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Map call_id -> index in steps[] for pairing tool_result with tool_call
  const callIndexMap = useRef<Map<string, number>>(new Map());

  /* Auto-scroll to bottom as new steps arrive */
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [steps, doneEvent]);

  /* Handle a single SSE event */
  const handleEvent = useCallback((event: AgentEvent) => {
    switch (event.type) {
      case 'agent_start': {
        const e = event as AgentStartEvent;
        setMaxTurnsActual(e.max_turns);
        setToolCount(e.tool_count);
        setCurrentTurn(e.turn);
        break;
      }
      case 'thinking': {
        const e = event as ThinkingEvent;
        setCurrentTurn(e.turn);
        setSteps(prev => [...prev, { kind: 'thinking', event: e }]);
        break;
      }
      case 'tool_call': {
        const e = event as ToolCallEvent;
        setCurrentTurn(e.turn);
        setSteps(prev => {
          const idx = prev.length;
          callIndexMap.current.set(e.call_id, idx);
          return [...prev, { kind: 'tool_call', event: e }];
        });
        break;
      }
      case 'tool_result': {
        const e = event as ToolResultEvent;
        const idx = callIndexMap.current.get(e.call_id);
        if (idx != null) {
          setSteps(prev => {
            const updated = [...prev];
            const step = updated[idx];
            if (step && step.kind === 'tool_call') {
              updated[idx] = { ...step, result: e };
            }
            return updated;
          });
        }
        break;
      }
      case 'output': {
        const e = event as OutputEvent;
        setSteps(prev => [...prev, { kind: 'output', event: e }]);
        break;
      }
      case 'agent_done': {
        const e = event as AgentDoneEvent;
        setDoneEvent(e);
        setIsRunning(false);
        break;
      }
      case 'error': {
        const e = event as AgentErrorEvent;
        setSteps(prev => [...prev, { kind: 'error', event: e }]);
        if (!e.recoverable) {
          setIsRunning(false);
        }
        break;
      }
    }
  }, []);

  /* Start SSE connection */
  useEffect(() => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    executeMigrationPhaseAgentic(planId, phaseNumber, handleEvent, {
      mvpId: mvpId ?? undefined,
      maxTurns,
      signal: ctrl.signal,
    })
      .then(() => {
        setIsRunning(false);
      })
      .catch((err: unknown) => {
        if (ctrl.signal.aborted) return;
        const message = err instanceof Error ? err.message : 'Unknown error';
        setError(message);
        setIsRunning(false);
      });

    return () => {
      ctrl.abort();
    };
  }, [planId, phaseNumber, mvpId, maxTurns, handleEvent]);

  /* Cancel handler */
  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    setIsRunning(false);
    onCancel();
  }, [onCancel]);

  /* Complete handler — called when user clicks "Done" after execution finishes */
  const handleDone = useCallback(() => {
    onComplete();
  }, [onComplete]);

  /* Progress calculation */
  const progressPct = maxTurnsActual > 0
    ? Math.min(100, Math.round(((currentTurn + 1) / maxTurnsActual) * 100))
    : 0;

  const label = PHASE_LABELS[phaseType] ?? phaseType;

  return (
    <div className="flex h-full flex-col rounded-xl border border-nebula/20 bg-void-light/30">
      {/* ── Header with progress ── */}
      <div className="flex items-center gap-3 border-b border-void-surface px-5 py-3">
        <Bot className="h-4.5 w-4.5 text-nebula-bright shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-text">
              Agent Execution: {label}
            </span>
            {isRunning && (
              <Loader2 className="h-3 w-3 animate-spin text-nebula-bright" />
            )}
            <span className="ml-auto text-[10px] text-text-dim">
              Turn {currentTurn + 1} / {maxTurnsActual}
              {toolCount > 0 && ` | ${toolCount} tools`}
            </span>
          </div>

          {/* Progress bar */}
          <div className="mt-1.5 h-1 w-full rounded-full bg-void-surface overflow-hidden">
            <div
              className="h-full rounded-full bg-nebula-bright transition-all duration-300"
              style={{ width: `${doneEvent ? 100 : progressPct}%` }}
            />
          </div>
        </div>

        {/* Cancel button */}
        {isRunning && (
          <button
            onClick={handleCancel}
            className="rounded-md p-1 text-text-dim transition-colors hover:bg-danger/10 hover:text-danger"
            title="Cancel execution"
          >
            <XCircle className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* ── Step log ── */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-3 space-y-1">
        {steps.map((step, i) => {
          switch (step.kind) {
            case 'thinking':
              return <ThinkingCard key={i} event={step.event} />;
            case 'tool_call':
              return (
                <ToolCallCard key={i} event={step.event} result={step.result} />
              );
            case 'output':
              return <OutputCard key={i} event={step.event} />;
            case 'error':
              return <ErrorCard key={i} event={step.event} />;
          }
        })}

        {/* Done summary */}
        {doneEvent && (
          <DoneCard
            turnsUsed={doneEvent.turns_used}
            toolsCalled={doneEvent.tools_called}
            totalMs={doneEvent.total_ms}
          />
        )}

        {/* Connection error */}
        {error && !doneEvent && (
          <div className="flex gap-2 py-2 px-3 rounded-lg border border-danger/30 bg-danger/5">
            <XCircle className="h-4 w-4 text-danger shrink-0 mt-0.5" />
            <p className="text-xs text-danger">{error}</p>
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      {!isRunning && (
        <div className="flex items-center justify-end gap-2 border-t border-void-surface px-5 py-3">
          {error && !doneEvent && (
            <span className="text-[10px] text-danger mr-auto">
              Execution failed — review errors above
            </span>
          )}
          <button
            onClick={handleDone}
            className="rounded-md bg-glow px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-glow-dim"
          >
            {doneEvent ? 'View Results' : 'Close'}
          </button>
        </div>
      )}
    </div>
  );
}
