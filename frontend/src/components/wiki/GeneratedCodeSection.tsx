/**
 * GeneratedCodeSection -- Project Wiki generated code panel.
 *
 * Displays transform / test phase output with syntax-highlighted
 * source files in a split-pane layout (file list + code viewer).
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Loader2,
  Inbox,
  FileCode,
} from 'lucide-react';
import { Highlight, themes } from 'prism-react-renderer';
import type {
  MigrationPlan,
  MigrationPhaseOutput,
  MigrationFile,
  FunctionalMvpSummary,
} from '../../types/index.ts';

/* ── Props ───────────────────────────────────────────────────────── */

interface Props {
  migrationPlan: MigrationPlan | null;
  loading?: boolean;
  onLoad: () => Promise<void>;
  onLoadPhase: (phaseNumber: number, mvpId?: number) => Promise<MigrationPhaseOutput | null>;
}

/* ── Phase type mapping ──────────────────────────────────────────── */

type PhaseKind = 'transform' | 'test';

function getPhaseNumber(pipelineVersion: number | undefined, kind: PhaseKind): number {
  if (pipelineVersion === 2) {
    return kind === 'transform' ? 3 : 4;
  }
  // V1 (default)
  return kind === 'transform' ? 5 : 6;
}

/* ── Language to prism-react-renderer language mapping ────────────── */

function mapLanguage(lang: string): string {
  const lower = lang.toLowerCase();
  const mapping: Record<string, string> = {
    'c#': 'csharp',
    'cs': 'csharp',
    'csharp': 'csharp',
    'javascript': 'javascript',
    'js': 'javascript',
    'typescript': 'typescript',
    'ts': 'typescript',
    'tsx': 'tsx',
    'jsx': 'jsx',
    'python': 'python',
    'py': 'python',
    'java': 'java',
    'sql': 'sql',
    'xml': 'markup',
    'html': 'markup',
    'css': 'css',
    'json': 'json',
    'yaml': 'yaml',
    'yml': 'yaml',
    'go': 'go',
    'rust': 'rust',
    'kotlin': 'kotlin',
    'swift': 'swift',
    'ruby': 'ruby',
    'php': 'php',
  };
  return mapping[lower] ?? lower;
}

/* ── Basename helper ─────────────────────────────────────────────── */

function basename(filePath: string): string {
  const parts = filePath.split('/');
  return parts[parts.length - 1] ?? filePath;
}

/* ── MVP pill selector ───────────────────────────────────────────── */

function MvpPillBar({
  mvps,
  selectedId,
  onSelect,
}: {
  mvps: FunctionalMvpSummary[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {mvps.map((mvp) => {
        const active = selectedId === mvp.mvp_id;
        return (
          <button
            key={mvp.mvp_id}
            onClick={() => onSelect(mvp.mvp_id)}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
              active
                ? 'bg-glow/10 text-glow border-glow/20'
                : 'border-void-surface text-text-muted hover:border-void-surface hover:text-text'
            }`}
          >
            {mvp.name}
          </button>
        );
      })}
    </div>
  );
}

/* ── Language badge ───────────────────────────────────────────────── */

function LangBadge({ lang }: { lang: string }) {
  return (
    <span className="inline-flex items-center rounded-md bg-void-surface/60 px-1.5 py-0.5 text-[9px] font-medium text-text-dim">
      {lang}
    </span>
  );
}

/* ── Code viewer with prism highlighting ─────────────────────────── */

function CodeViewer({ file }: { file: MigrationFile }) {
  const language = mapLanguage(file.language);

  return (
    <div className="flex flex-col h-full">
      {/* File header */}
      <div className="flex items-center gap-2 border-b border-void-surface px-4 py-2.5 bg-void-light/50">
        <FileCode className="h-3.5 w-3.5 text-text-dim" />
        <span className="text-xs font-mono text-text-muted flex-1 truncate">
          {file.file_path}
        </span>
        <LangBadge lang={file.language} />
      </div>

      {/* Highlighted code */}
      <div className="flex-1 overflow-auto">
        <Highlight theme={themes.nightOwl} code={file.content} language={language}>
          {({ className, style, tokens, getLineProps, getTokenProps }) => (
            <pre
              className={`${className} text-xs leading-relaxed p-4 m-0`}
              style={{ ...style, background: 'transparent' }}
            >
              {tokens.map((line, i) => {
                const lineProps = getLineProps({ line, key: i });
                return (
                  <div key={i} {...lineProps} className={`${lineProps.className ?? ''} table-row`}>
                    <span className="table-cell pr-4 text-right select-none text-text-dim/50 w-10">
                      {i + 1}
                    </span>
                    <span className="table-cell">
                      {line.map((token, j) => {
                        const tokenProps = getTokenProps({ token, key: j });
                        return <span key={j} {...tokenProps} />;
                      })}
                    </span>
                  </div>
                );
              })}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}

/* ── Main component ──────────────────────────────────────────────── */

export function GeneratedCodeSection({
  migrationPlan,
  loading,
  onLoad,
  onLoadPhase,
}: Props) {
  const [selectedMvpId, setSelectedMvpId] = useState<number | null>(null);
  const [activeKind, setActiveKind] = useState<PhaseKind>('transform');
  const [phaseOutput, setPhaseOutput] = useState<MigrationPhaseOutput | null>(null);
  const [loadingPhase, setLoadingPhase] = useState(false);
  const [selectedFileIdx, setSelectedFileIdx] = useState(0);

  useEffect(() => {
    if (migrationPlan === null) {
      void onLoad();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* Load phase output when MVP or kind changes */
  const loadPhase = useCallback(
    async (mvpId: number | null, kind: PhaseKind) => {
      if (mvpId === null) return;
      setPhaseOutput(null);
      setSelectedFileIdx(0);
      setLoadingPhase(true);
      const phaseNum = getPhaseNumber(migrationPlan?.pipeline_version, kind);
      const result = await onLoadPhase(phaseNum, mvpId);
      setPhaseOutput(result);
      setLoadingPhase(false);
    },
    [migrationPlan?.pipeline_version, onLoadPhase],
  );

  const handleSelectMvp = useCallback(
    (mvpId: number) => {
      setSelectedMvpId(mvpId);
      void loadPhase(mvpId, activeKind);
    },
    [activeKind, loadPhase],
  );

  const handleSelectKind = useCallback(
    (kind: PhaseKind) => {
      setActiveKind(kind);
      void loadPhase(selectedMvpId, kind);
    },
    [selectedMvpId, loadPhase],
  );

  /* Loading */
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-muted">
        <Loader2 className="h-6 w-6 animate-spin" />
        <span className="text-sm">Loading generated code data...</span>
      </div>
    );
  }

  /* Empty state */
  if (!migrationPlan || migrationPlan.mvps.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-dim">
        <FileCode className="h-8 w-8" />
        <span className="text-sm">
          {migrationPlan
            ? 'No MVPs available for code generation.'
            : 'No migration plan available.'}
        </span>
      </div>
    );
  }

  const files = phaseOutput?.output_files ?? [];
  const activeFile = files[selectedFileIdx] ?? null;

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-lg font-semibold text-text">Generated Code</h2>

      {/* MVP selector */}
      <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
        <h3 className="text-sm font-semibold text-text mb-3">Select MVP</h3>
        <MvpPillBar
          mvps={migrationPlan.mvps}
          selectedId={selectedMvpId}
          onSelect={handleSelectMvp}
        />
      </div>

      {/* Phase toggle */}
      {selectedMvpId !== null && (
        <div className="flex gap-2">
          {(['transform', 'test'] as const).map((kind) => {
            const active = activeKind === kind;
            return (
              <button
                key={kind}
                onClick={() => handleSelectKind(kind)}
                className={`rounded-lg border px-4 py-2 text-xs font-medium capitalize transition-colors ${
                  active
                    ? 'bg-glow text-white border-glow'
                    : 'border-void-surface text-text-muted hover:text-text'
                }`}
              >
                {kind}
              </button>
            );
          })}
        </div>
      )}

      {/* Split pane: file list + code viewer */}
      {selectedMvpId !== null && (
        <div className="rounded-xl border border-void-surface bg-void-light/30 overflow-hidden">
          {loadingPhase ? (
            <div className="flex flex-col items-center justify-center gap-3 py-16 text-text-muted">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span className="text-xs">Loading phase output...</span>
            </div>
          ) : files.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-3 py-16 text-text-dim">
              <Inbox className="h-6 w-6" />
              <span className="text-xs">No generated code for this phase yet.</span>
            </div>
          ) : (
            <div className="flex min-h-[400px]">
              {/* File list sidebar */}
              <div className="w-48 shrink-0 border-r border-void-surface overflow-y-auto">
                {files.map((file, idx) => {
                  const isActive = idx === selectedFileIdx;
                  return (
                    <button
                      key={file.file_path}
                      onClick={() => setSelectedFileIdx(idx)}
                      className={`flex w-full items-center gap-2 px-3 py-2 text-left transition-colors ${
                        isActive
                          ? 'bg-void-surface text-text'
                          : 'text-text-muted hover:bg-void-surface/40 hover:text-text'
                      }`}
                    >
                      <div className="flex-1 min-w-0">
                        <p className="text-[11px] font-medium truncate">
                          {basename(file.file_path)}
                        </p>
                      </div>
                      <LangBadge lang={file.language} />
                    </button>
                  );
                })}
              </div>

              {/* Code viewer */}
              <div className="flex-1 min-w-0 bg-void/60">
                {activeFile ? (
                  <CodeViewer file={activeFile} />
                ) : (
                  <div className="flex items-center justify-center h-full text-text-dim text-xs">
                    Select a file to view
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
