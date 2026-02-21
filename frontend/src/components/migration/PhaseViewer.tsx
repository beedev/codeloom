/**
 * PhaseViewer — renders phase output as markdown, with actions.
 *
 * Layout depends on phase type:
 * - transform: Full-width DiffViewer with file selector tabs
 * - test: TestFilePanel with grouped test files
 * - others: 60/40 split — left=markdown, right=generated files
 *
 * Bottom action bar: Reject, Request Changes, Approve Phase.
 * Download buttons in header when files exist.
 */

import { useState, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Highlight, themes } from 'prism-react-renderer';
import { Play, Check, Loader2, FileCode, AlertCircle, Download } from 'lucide-react';
import { DiffViewer } from './DiffViewer.tsx';
import { TestFilePanel } from './TestFilePanel.tsx';
import * as api from '../../services/api.ts';
import type { MigrationPhaseOutput, MigrationFile, DiffContext } from '../../types/index.ts';

const PHASE_LABELS: Record<string, string> = {
  discovery: 'Discovery Analysis',
  architecture: 'Target Architecture',
  analyze: 'MVP Analysis',
  design: 'Detailed Design',
  transform: 'Code Transform',
  test: 'Test Generation',
};

interface PhaseViewerProps {
  phase: MigrationPhaseOutput | null;
  phaseNumber: number;
  phaseType: string;
  canExecute: boolean;
  isExecuting: boolean;
  onExecute: () => void;
  onApprove: () => void;
  planId?: string;
  mvpId?: number | null;
  sourceProjectId?: string;
}

export function PhaseViewer({
  phase,
  phaseNumber,
  phaseType,
  canExecute,
  isExecuting,
  onExecute,
  onApprove,
  planId,
  mvpId,
  sourceProjectId: _sourceProjectId,
}: PhaseViewerProps) {
  void _sourceProjectId; // reserved for future use (VSCode integration)
  const [activeFileIdx, setActiveFileIdx] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [diffContext, setDiffContext] = useState<DiffContext | null>(null);
  const [isDiffLoading, setIsDiffLoading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const title = PHASE_LABELS[phaseType] ?? phaseType;
  const hasOutput = phase && phase.output;
  const hasFiles = phase && phase.output_files && phase.output_files.length > 0;
  const isComplete = phase?.status === 'complete';
  const isApproved = phase?.approved;
  const isError = phase?.status === 'error';

  const isTransform = phaseType === 'transform';
  const isTest = phaseType === 'test';

  // Fetch diff context when transform phase has files
  const shouldFetchDiff = isTransform && hasFiles && !!planId;

  // Reset file index when switching to a different phase
  const phaseIdForReset = phase?.phase_id;
  useEffect(() => {
    setActiveFileIdx(0); // eslint-disable-line react-hooks/set-state-in-effect -- intentional reset on phase change
  }, [phaseIdForReset]);

  useEffect(() => {
    if (!shouldFetchDiff) {
      setDiffContext(null); // eslint-disable-line react-hooks/set-state-in-effect -- reset when conditions change
      setIsDiffLoading(false);
      return;
    }

    let cancelled = false;
    setIsDiffLoading(true);
    api.getDiffContext(planId!, phaseNumber, mvpId ?? undefined)
      .then((ctx) => {
        if (!cancelled) setDiffContext(ctx);
      })
      .catch(() => {
        if (!cancelled) setDiffContext(null);
      })
      .finally(() => {
        if (!cancelled) setIsDiffLoading(false);
      });

    return () => { cancelled = true; };
  }, [shouldFetchDiff, planId, phaseNumber, mvpId]);

  // Download handlers
  const handleDownloadAll = useCallback(() => {
    if (!planId) return;
    setDownloadError(null);
    try {
      const url = api.getPhaseDownloadUrl(planId, phaseNumber, mvpId ?? undefined, 'zip');
      window.location.href = url;
    } catch {
      setDownloadError('Download failed');
    }
  }, [planId, phaseNumber, mvpId]);

  const handleDownloadFile = useCallback((filePath: string) => {
    if (!planId) return;
    setDownloadError(null);
    try {
      const url = api.getPhaseDownloadUrl(planId, phaseNumber, mvpId ?? undefined, 'single', filePath);
      window.location.href = url;
    } catch {
      setDownloadError('Download failed');
    }
  }, [planId, phaseNumber, mvpId]);

  // Keyboard shortcut: Cmd+Enter to approve
  useEffect(() => {
    if (!isComplete || isApproved) return;
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        onApprove();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isComplete, isApproved, onApprove]);

  // Find matching source file for the active migrated file
  const activeMigratedFile = hasFiles ? phase!.output_files![activeFileIdx] : null;
  const matchedSourceFile = (() => {
    if (!diffContext || !activeMigratedFile) return null;
    const mapping = diffContext.file_mapping.find(
      m => m.target_path === activeMigratedFile.file_path,
    );
    if (!mapping) return null;
    return diffContext.source_files.find(s => s.file_path === mapping.source_path) ?? null;
  })();

  return (
    <div className="flex h-full flex-col">
      {/* Phase header */}
      <div className="flex items-center justify-between border-b border-void-surface px-6 py-4">
        <div>
          <h2 className="text-base font-medium text-text">
            Phase {phaseNumber}: {title}
          </h2>
          {phase && (
            <p className="mt-0.5 text-xs text-text-dim">
              Status: {phase.status}
              {isApproved && ' (approved)'}
            </p>
          )}
          {typeof phase?.phase_metadata?.output_path === 'string' && (
            <p className="mt-0.5 text-xs text-text-dim font-mono">
              Saved to: {phase.phase_metadata.output_path}/code
            </p>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Download buttons */}
          {hasFiles && planId && (
            <>
              <button
                onClick={handleDownloadAll}
                className="flex items-center gap-1.5 rounded-md border border-void-surface px-3 py-1.5 text-xs text-text-muted transition-colors hover:border-glow/30 hover:text-text"
                title="Download all files as ZIP"
              >
                <Download className="h-3.5 w-3.5" />
                Download All
              </button>
            </>
          )}

          {/* VSCode placeholder */}
          {hasFiles && (
            <button
              className="flex items-center gap-1 rounded-md border border-void-surface px-3 py-1.5 text-xs text-text-dim cursor-not-allowed opacity-50"
              title="Coming Soon"
              disabled
            >
              <span className="material-symbols-outlined text-[14px]">code</span>
              Open in VSCode
            </button>
          )}

          {/* Execute button */}
          {canExecute && !isExecuting && (!phase || phase.status === 'pending' || isError) && (
            <button
              onClick={onExecute}
              className="flex items-center gap-1.5 rounded-md bg-glow px-4 py-2 text-xs font-medium text-white hover:bg-glow-dim"
            >
              <Play className="h-3.5 w-3.5" />
              Execute Phase
            </button>
          )}

          {isExecuting && (
            <div className="flex items-center gap-2 rounded-md bg-glow/10 px-4 py-2 text-xs text-glow">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Running...
            </div>
          )}

          {isApproved && (
            <span className="flex items-center gap-1.5 text-xs text-success">
              <Check className="h-3.5 w-3.5" />
              Approved
            </span>
          )}
        </div>
      </div>

      {/* Download error */}
      {downloadError && (
        <div className="border-b border-danger/30 bg-danger/5 px-4 py-1.5 text-xs text-danger">
          {downloadError}
        </div>
      )}

      {/* Phase content */}
      <div className="flex flex-1 overflow-hidden">
        {/* ── Pre-execution states ── */}
        {!hasOutput && !isExecuting && !hasFiles && (
          <div className="flex flex-1 items-center justify-center text-text-dim">
            <div className="text-center">
              {isError && phase?.output ? (
                <div className="mx-auto max-w-lg">
                  <AlertCircle className="mx-auto h-8 w-8 text-danger" />
                  <p className="mt-3 text-sm text-danger">Phase failed</p>
                  <p className="mt-2 text-xs text-text-dim">{phase.output}</p>
                </div>
              ) : (
                <>
                  <p className="text-sm">Phase not yet executed.</p>
                  {canExecute && (
                    <p className="mt-1 text-xs">Click "Execute Phase" to begin.</p>
                  )}
                  {!canExecute && (
                    <p className="mt-1 text-xs">Approve the previous phase first.</p>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {isExecuting && !hasOutput && !hasFiles && (
          <div className="flex flex-1 items-center justify-center">
            <div className="text-center">
              <Loader2 className="mx-auto h-8 w-8 animate-spin text-glow" />
              <p className="mt-3 text-sm text-text-muted">
                Analyzing codebase and generating output...
              </p>
              <p className="mt-1 text-xs text-text-dim">
                This may take a minute for large codebases.
              </p>
            </div>
          </div>
        )}

        {/* ── Transform phase: DiffViewer (full width) ── */}
        {isTransform && hasFiles && (
          <div className="flex flex-1 flex-col overflow-hidden">
            {/* File selector tabs */}
            <div className="flex items-center gap-1 overflow-x-auto border-b border-void-surface px-2">
              {phase!.output_files!.map((file: MigrationFile, idx: number) => (
                <button
                  key={idx}
                  onClick={() => setActiveFileIdx(idx)}
                  className={`flex items-center gap-1.5 whitespace-nowrap px-3 py-2 text-xs ${
                    idx === activeFileIdx
                      ? 'border-b-2 border-glow text-glow'
                      : 'text-text-dim hover:text-text-muted'
                  }`}
                >
                  <FileCode className="h-3 w-3" />
                  {file.file_path.split('/').pop()}
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDownloadFile(file.file_path); }}
                    className="ml-1 rounded p-0.5 text-text-dim/50 hover:text-text-muted"
                    title="Download file"
                  >
                    <Download className="h-2.5 w-2.5" />
                  </button>
                </button>
              ))}
            </div>

            {/* Diff loading */}
            {isDiffLoading && (
              <div className="flex flex-1 items-center justify-center">
                <Loader2 className="h-5 w-5 animate-spin text-glow" />
                <span className="ml-2 text-xs text-text-dim">Loading diff context...</span>
              </div>
            )}

            {/* DiffViewer */}
            {!isDiffLoading && activeMigratedFile && (
              <div className="flex-1 overflow-hidden">
                <DiffViewer
                  sourceFile={matchedSourceFile}
                  migratedFile={activeMigratedFile}
                  sourceLanguage={matchedSourceFile?.language ?? activeMigratedFile.language}
                  targetLanguage={activeMigratedFile.language}
                />
              </div>
            )}
          </div>
        )}

        {/* ── Test phase: TestFilePanel (full width) ── */}
        {isTest && hasFiles && (
          <div className="flex-1 overflow-hidden">
            <TestFilePanel
              files={phase!.output_files!}
              onDownloadAll={handleDownloadAll}
              onDownloadFile={handleDownloadFile}
            />
          </div>
        )}

        {/* ── Fallback: transform/test with output text but no files ── */}
        {(isTransform || isTest) && !hasFiles && hasOutput && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {phase!.output!}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {/* ── Other phases: original 60/40 split ── */}
        {!isTransform && !isTest && (hasOutput || hasFiles) && (
          <>
            {/* Left: Markdown output */}
            <div className={`overflow-y-auto ${hasFiles ? 'w-3/5 border-r border-void-surface' : 'flex-1'}`}>
              {hasOutput && (
                <div className="p-6">
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {phase!.output!}
                    </ReactMarkdown>
                  </div>
                </div>
              )}
            </div>

            {/* Right: Generated files (40%) */}
            {hasFiles && (
              <div className="flex w-2/5 flex-col overflow-hidden">
                <div className="flex items-center gap-1 overflow-x-auto border-b border-void-surface px-2">
                  {phase!.output_files!.map((file: MigrationFile, idx: number) => (
                    <button
                      key={idx}
                      onClick={() => setActiveFileIdx(idx)}
                      className={`flex items-center gap-1.5 whitespace-nowrap px-3 py-2 text-xs ${
                        idx === activeFileIdx
                          ? 'border-b-2 border-glow text-glow'
                          : 'text-text-dim hover:text-text-muted'
                      }`}
                    >
                      <FileCode className="h-3 w-3" />
                      {file.file_path}
                      {file.test_type && (
                        <span className="rounded bg-void-surface px-1 text-[10px]">
                          {file.test_type}
                        </span>
                      )}
                    </button>
                  ))}
                </div>
                <div className="flex-1 overflow-auto">
                  <GeneratedFileView file={phase!.output_files![activeFileIdx]} />
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Bottom action bar */}
      {isComplete && !isApproved && (
        <div className="flex items-center gap-3 border-t border-void-surface bg-void-light/50 px-6 py-3">
          <input
            type="text"
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Add a feedback comment for the migration team... (Markdown supported)"
            className="flex-1 rounded-lg border border-void-surface bg-void px-3 py-2 text-xs text-text placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50"
          />

          <button
            className="rounded-lg border border-danger/50 px-4 py-2 text-xs font-medium text-danger transition-colors hover:bg-danger/10"
          >
            Reject
          </button>

          <button
            onClick={onExecute}
            className="rounded-lg border border-warning/50 px-4 py-2 text-xs font-medium text-warning transition-colors hover:bg-warning/10"
          >
            Request Changes
          </button>

          <button
            onClick={onApprove}
            className="flex items-center gap-1.5 rounded-lg bg-glow px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-glow-dim"
          >
            <Check className="h-3.5 w-3.5" />
            Approve Phase
          </button>

          <span className="text-[10px] text-text-dim">
            <kbd className="rounded border border-void-surface bg-void-lighter px-1 py-0.5 text-[9px]">
              {navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'}
            </kbd>
            +
            <kbd className="rounded border border-void-surface bg-void-lighter px-1 py-0.5 text-[9px]">
              Enter
            </kbd>
            {' '}to approve
          </span>
        </div>
      )}
    </div>
  );
}

export function GeneratedFileView({ file }: { file: MigrationFile }) {
  if (!file) return null;

  const langMap: Record<string, string> = {
    python: 'python',
    java: 'java',
    javascript: 'javascript',
    typescript: 'typescript',
    csharp: 'csharp',
    go: 'go',
    rust: 'rust',
  };
  const lang = langMap[file.language] ?? file.language;

  return (
    <Highlight theme={themes.nightOwl} code={file.content} language={lang}>
      {({ style, tokens, getLineProps, getTokenProps }) => (
        <pre
          className="overflow-x-auto p-4 text-xs leading-relaxed"
          style={{ ...style, background: 'transparent' }}
        >
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })}>
              <span className="mr-4 inline-block w-8 text-right text-text-dim select-none">
                {i + 1}
              </span>
              {line.map((token, key) => (
                <span key={key} {...getTokenProps({ token })} />
              ))}
            </div>
          ))}
        </pre>
      )}
    </Highlight>
  );
}
