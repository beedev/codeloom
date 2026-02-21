/**
 * SourcePanel Component
 *
 * Displays RAG retrieval sources alongside the chat.
 * Compact cards with click-to-expand code previews.
 * "View full file" opens a modal with complete file content.
 */

import { useState, useEffect, useRef } from 'react';
import { FileCode2, Copy, Check, ChevronRight, ExternalLink, X, Loader2 } from 'lucide-react';
import { Highlight, themes } from 'prism-react-renderer';
import type { ChatSource } from '../types/index.ts';

interface SourcePanelProps {
  sources: ChatSource[];
  projectId?: string;
}

// Map language strings to Prism language identifiers
function toPrismLanguage(lang?: string): string {
  if (!lang) return 'python';
  const map: Record<string, string> = {
    python: 'python', py: 'python',
    javascript: 'javascript', js: 'javascript',
    typescript: 'typescript', ts: 'typescript',
    java: 'java', csharp: 'csharp', cs: 'csharp',
    go: 'go', rust: 'rust', ruby: 'ruby', cpp: 'cpp', c: 'c',
  };
  return map[lang.toLowerCase()] || 'python';
}

// Extract just the filename from a path
function shortPath(filename: string): string {
  const parts = filename.split('/');
  if (parts.length <= 2) return filename;
  return `.../${parts.slice(-2).join('/')}`;
}

// Unit type label styling
function unitTypeStyle(unitType?: string): string {
  switch (unitType) {
    case 'function': return 'text-glow bg-glow/10';
    case 'method': return 'text-nebula-bright bg-nebula/10';
    case 'class': return 'text-warning bg-warning/10';
    case 'module': return 'text-success bg-success/10';
    default: return 'text-text-dim bg-void-surface';
  }
}

interface FileViewerState {
  filePath: string;
  language: string;
  highlightStart?: number;
  highlightEnd?: number;
}

export function SourcePanel({ sources, projectId }: SourcePanelProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [viewingFile, setViewingFile] = useState<FileViewerState | null>(null);

  if (sources.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="text-center">
          <FileCode2 className="mx-auto h-8 w-8 text-text-dim/40" />
          <p className="mt-3 text-xs text-text-dim">
            Source references will appear here after you send a query.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-void-surface px-4 py-3 shrink-0">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-text-muted font-[family-name:var(--font-display)]">
          Source References
        </h3>
        <span className="rounded-full bg-glow/10 px-2 py-0.5 text-[10px] font-semibold text-glow">
          {sources.length}
        </span>
      </div>

      {/* Scrollable source list */}
      <div className="flex-1 overflow-y-auto overscroll-contain">
        <div className="space-y-2 p-3">
          {sources.map((source, idx) => (
            <SourceCard
              key={idx}
              source={source}
              rank={idx + 1}
              isExpanded={expandedIdx === idx}
              onToggle={() => setExpandedIdx(expandedIdx === idx ? null : idx)}
              onViewFile={projectId ? () => setViewingFile({
                filePath: source.filename,
                language: source.language || 'python',
                highlightStart: source.start_line,
                highlightEnd: source.end_line,
              }) : undefined}
            />
          ))}
        </div>
      </div>

      {/* File viewer modal */}
      {viewingFile && projectId && (
        <FileViewerModal
          projectId={projectId}
          data={viewingFile}
          onClose={() => setViewingFile(null)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SourceCard — compact by default, expands on click
// ---------------------------------------------------------------------------

function SourceCard({
  source,
  rank,
  isExpanded,
  onToggle,
  onViewFile,
}: {
  source: ChatSource;
  rank: number;
  isExpanded: boolean;
  onToggle: () => void;
  onViewFile?: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const snippet = source.snippet || '';
  const prismLang = toPrismLanguage(source.language);

  async function handleCopy(e: React.MouseEvent) {
    e.stopPropagation();
    await navigator.clipboard.writeText(snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div
      className={`rounded-lg border transition-all ${
        isExpanded
          ? 'border-glow/30 bg-void-light'
          : 'border-void-surface bg-void-light/50 hover:border-glow/20 hover:bg-void-light'
      }`}
    >
      {/* Clickable header */}
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left"
      >
        {/* Rank */}
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-nebula/20 text-[9px] font-bold text-nebula-bright">
          {rank}
        </span>

        {/* File info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <FileCode2 className="h-3 w-3 shrink-0 text-text-dim" />
            <span className="truncate text-xs font-medium text-text" title={source.filename}>
              {shortPath(source.filename)}
            </span>
          </div>
          <div className="flex items-center gap-1.5 mt-1">
            {source.unit_type && (
              <span className={`rounded px-1 py-px text-[8px] font-semibold uppercase tracking-wide ${unitTypeStyle(source.unit_type)}`}>
                {source.unit_type}
              </span>
            )}
            {source.unit_name && (
              <span className="truncate text-[10px] font-[family-name:var(--font-code)] text-text-muted max-w-[120px]">
                {source.unit_name}
              </span>
            )}
            {source.start_line != null && (
              <span className="text-[9px] text-text-dim">
                L{source.start_line}{source.end_line != null && `\u2013${source.end_line}`}
              </span>
            )}
          </div>
        </div>

        {/* Score + expand indicator */}
        <div className="flex items-center gap-2 shrink-0">
          <ScoreBadge score={source.score} />
          <ChevronRight className={`h-3 w-3 text-text-dim transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
        </div>
      </button>

      {/* Expanded code snippet */}
      {isExpanded && snippet && (
        <div className="relative border-t border-void-surface/60">
          {/* Top bar: language + actions */}
          <div className="flex items-center justify-between px-3 py-1.5 bg-void/40">
            {source.language && (
              <span className="rounded bg-void-surface px-1.5 py-0.5 text-[8px] font-medium text-text-dim">
                {source.language}
              </span>
            )}
            <div className="flex items-center gap-1">
              {onViewFile && (
                <button
                  onClick={(e) => { e.stopPropagation(); onViewFile(); }}
                  className="flex items-center gap-1 rounded px-2 py-1 text-[10px] text-glow hover:bg-glow/10 transition-colors"
                  title="View full file"
                >
                  <ExternalLink className="h-3 w-3" />
                  View full file
                </button>
              )}
              <button
                onClick={handleCopy}
                className="rounded p-1 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
                title="Copy snippet"
              >
                {copied ? (
                  <Check className="h-3 w-3 text-success" />
                ) : (
                  <Copy className="h-3 w-3" />
                )}
              </button>
            </div>
          </div>

          <Highlight
            theme={themes.nightOwl}
            code={snippet.trimEnd()}
            language={prismLang}
          >
            {({ tokens, getLineProps, getTokenProps }) => (
              <pre className="overflow-x-auto p-3 text-[11px] leading-[1.6] bg-void/60">
                {tokens.map((line, i) => (
                  <div key={i} {...getLineProps({ line })} className="table-row">
                    <span className="table-cell select-none pr-3 text-right text-text-dim/50 text-[10px]">
                      {(source.start_line ?? 1) + i}
                    </span>
                    <span className="table-cell whitespace-pre-wrap break-all">
                      {line.map((token, key) => (
                        <span key={key} {...getTokenProps({ token })} />
                      ))}
                    </span>
                  </div>
                ))}
              </pre>
            )}
          </Highlight>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ScoreBadge — handles both 0-1 and >1 score ranges
// ---------------------------------------------------------------------------

function ScoreBadge({ score }: { score: number }) {
  const displayPercent = score > 1
    ? Math.round(score * 10) / 10
    : Math.round(score * 100);

  const isRawScore = score > 1;

  let colorClass = 'text-text-dim border-void-surface';
  if (score >= 3 || (!isRawScore && score >= 0.8)) {
    colorClass = 'text-success border-success/30 bg-success/5';
  } else if (score >= 2 || (!isRawScore && score >= 0.6)) {
    colorClass = 'text-glow border-glow/30 bg-glow/5';
  } else if (score >= 1 || (!isRawScore && score >= 0.4)) {
    colorClass = 'text-warning border-warning/30 bg-warning/5';
  }

  return (
    <span
      className={`rounded-md border px-1.5 py-0.5 text-[10px] font-semibold font-[family-name:var(--font-display)] ${colorClass}`}
      title={`Relevance score: ${score.toFixed(3)}`}
    >
      {isRawScore ? displayPercent.toFixed(1) : `${displayPercent}%`}
    </span>
  );
}

// ---------------------------------------------------------------------------
// FileViewerModal — fetches and displays full file with line highlighting
// ---------------------------------------------------------------------------

interface FileUnit {
  unit_id: string;
  unit_type: string;
  name: string;
  start_line: number;
  end_line: number;
  source: string;
}

function FileViewerModal({
  projectId,
  data,
  onClose,
}: {
  projectId: string;
  data: FileViewerState;
  onClose: () => void;
}) {
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [lineCount, setLineCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const highlightRef = useRef<HTMLDivElement>(null);
  const prismLang = toPrismLanguage(data.language);

  // Fetch file content
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setFetchError(null);

    fetch(`/api/projects/${projectId}/file/${data.filePath}`, {
      credentials: 'include',
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: { units: FileUnit[]; line_count: number }) => {
        if (cancelled) return;
        const fullSource = json.units.map((u) => u.source).join('\n\n');
        setFileContent(fullSource);
        setLineCount(json.line_count || fullSource.split('\n').length);
      })
      .catch((err) => {
        if (cancelled) return;
        setFetchError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [projectId, data.filePath]);

  // Scroll to highlighted lines
  useEffect(() => {
    if (fileContent && highlightRef.current) {
      setTimeout(() => {
        highlightRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 100);
    }
  }, [fileContent]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const handleCopy = async () => {
    if (!fileContent) return;
    await navigator.clipboard.writeText(fileContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-void/80 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative flex h-[85vh] w-[85vw] max-w-5xl flex-col rounded-xl border border-void-surface bg-void-light shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-void-surface px-5 py-3 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <FileCode2 className="h-4 w-4 shrink-0 text-glow" />
            <span className="truncate text-sm font-medium text-text font-[family-name:var(--font-code)]">
              {data.filePath}
            </span>
            <span className="shrink-0 rounded bg-void-surface px-2 py-0.5 text-[10px] font-medium text-text-dim">
              {data.language}
            </span>
            {lineCount > 0 && (
              <span className="shrink-0 text-[10px] text-text-dim">{lineCount} lines</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="rounded p-1.5 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
              title="Copy file"
            >
              {copied ? <Check className="h-4 w-4 text-success" /> : <Copy className="h-4 w-4" />}
            </button>
            <button
              onClick={onClose}
              className="rounded p-1.5 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
              title="Close (Esc)"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {loading && (
            <div className="flex h-full items-center justify-center">
              <Loader2 className="h-6 w-6 animate-spin text-glow" />
            </div>
          )}

          {fetchError && (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <p className="text-sm text-danger">{fetchError}</p>
                <button
                  onClick={onClose}
                  className="mt-3 rounded-lg border border-void-surface px-3 py-1.5 text-xs text-text-muted hover:bg-void-surface"
                >
                  Close
                </button>
              </div>
            </div>
          )}

          {fileContent && (
            <Highlight theme={themes.nightOwl} code={fileContent} language={prismLang}>
              {({ tokens, getLineProps, getTokenProps }) => (
                <pre className="p-4 text-[12px] leading-[1.7] font-[family-name:var(--font-code)]">
                  {tokens.map((line, i) => {
                    const lineNum = i + 1;
                    const isHighlighted =
                      data.highlightStart != null &&
                      data.highlightEnd != null &&
                      lineNum >= data.highlightStart &&
                      lineNum <= data.highlightEnd;

                    return (
                      <div
                        key={i}
                        {...getLineProps({ line })}
                        ref={lineNum === data.highlightStart ? highlightRef : undefined}
                        className={`table-row ${isHighlighted ? 'bg-glow/8' : ''}`}
                      >
                        <span
                          className={`table-cell select-none pr-4 text-right text-[11px] w-12 ${
                            isHighlighted ? 'text-glow font-medium' : 'text-text-dim/40'
                          }`}
                        >
                          {lineNum}
                        </span>
                        {isHighlighted && (
                          <span className="table-cell w-0.5 bg-glow/60" />
                        )}
                        <span className="table-cell pl-3 whitespace-pre-wrap">
                          {line.map((token, key) => (
                            <span key={key} {...getTokenProps({ token })} />
                          ))}
                        </span>
                      </div>
                    );
                  })}
                </pre>
              )}
            </Highlight>
          )}
        </div>
      </div>
    </div>
  );
}
