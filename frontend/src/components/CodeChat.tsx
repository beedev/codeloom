/**
 * CodeChat Component
 *
 * Chat interface for RAG-powered code queries.
 * - Message list with user/assistant bubbles
 * - Markdown rendering for assistant messages
 * - Inline source cards below assistant messages
 * - Full file viewer modal on source click
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Highlight, themes } from 'prism-react-renderer';
import {
  Send,
  Loader2,
  User,
  Bot,
  Trash2,
  FileCode2,
  ChevronDown,
  ChevronUp,
  X,
  ExternalLink,
  Copy,
  Check,
  Crosshair,
} from 'lucide-react';
import type { ChatMessage, ChatSource } from '../types/index.ts';

interface CodeChatProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  error: string | null;
  projectId?: string;
  hideInlineSources?: boolean;
  onSendMessage: (query: string, mode?: 'chat' | 'impact') => void;
  onClear: () => void;
}

export function CodeChat({
  messages,
  isStreaming,
  error,
  projectId,
  hideInlineSources,
  onSendMessage,
  onClear,
}: CodeChatProps) {
  const [input, setInput] = useState('');
  const [impactMode, setImpactMode] = useState(false);
  const [viewingFile, setViewingFile] = useState<FileViewerData | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = useCallback(
    (e?: FormEvent) => {
      e?.preventDefault();
      const query = input.trim();
      if (!query || isStreaming) return;
      setInput('');
      onSendMessage(query, impactMode ? 'impact' : 'chat');
    },
    [input, isStreaming, onSendMessage, impactMode],
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const handleViewFile = useCallback(
    (source: ChatSource) => {
      if (!projectId) return;
      setViewingFile({
        projectId,
        filePath: source.filename,
        language: source.language || 'python',
        highlightStart: source.start_line,
        highlightEnd: source.end_line,
      });
    },
    [projectId],
  );

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-void-surface px-4 py-3">
        <h2 className="text-sm font-medium text-text">Code Chat</h2>
        {messages.length > 0 && (
          <button
            onClick={onClear}
            className="flex items-center gap-1.5 rounded px-2 py-1 text-xs text-text-dim hover:bg-void-surface hover:text-danger"
          >
            <Trash2 className="h-3 w-3" />
            Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <Bot className="mx-auto h-10 w-10 text-text-dim/40" />
              <p className="mt-3 text-sm text-text-muted">
                Ask a question about your codebase
              </p>
              <p className="mt-1 text-xs text-text-dim">
                e.g. &quot;What does the auth module do?&quot;
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {messages.map((msg, idx) => (
              <MessageBubble
                key={idx}
                message={msg}
                hideInlineSources={hideInlineSources}
                onViewFile={handleViewFile}
              />
            ))}
            {isStreaming && (
              <div className="flex items-center gap-2 text-xs text-text-dim">
                <Loader2 className="h-3 w-3 animate-spin text-glow" />
                Thinking...
              </div>
            )}
            {error && (
              <div className="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                {error}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-void-surface p-4">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={impactMode ? "Ask about impact / blast radius..." : "Ask about your code..."}
            rows={1}
            className="flex-1 resize-none rounded-lg border border-void-surface bg-void-light px-4 py-2.5 text-sm text-text placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/30"
            disabled={isStreaming}
          />
          <button
            type="button"
            onClick={() => setImpactMode(!impactMode)}
            className={`rounded-lg p-2.5 transition-colors ${
              impactMode
                ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30'
                : 'bg-void-surface/50 text-text-dim hover:bg-void-surface hover:text-text-muted'
            }`}
            aria-label="Toggle impact analysis mode"
            title={impactMode ? "Impact mode ON — shows blast radius" : "Impact mode OFF"}
          >
            <Crosshair className="h-4 w-4" />
          </button>
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="rounded-lg bg-glow/20 p-2.5 text-glow transition-colors hover:bg-glow/30 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Send message"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </form>

      {/* File Viewer Modal */}
      {viewingFile && (
        <FileViewerModal
          data={viewingFile}
          onClose={() => setViewingFile(null)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Markdown component overrides (shared config)
// ---------------------------------------------------------------------------

const markdownComponents = {
  p: ({ children }: { children?: React.ReactNode }) => (
    <p className="text-text leading-relaxed mb-3 last:mb-0">{children}</p>
  ),
  code: ({ className, children, ...props }: { className?: string; children?: React.ReactNode }) => {
    const isInline = !className;
    return isInline ? (
      <code
        className="px-1.5 py-0.5 rounded bg-void-surface text-glow-bright font-[family-name:var(--font-code)] text-sm"
        {...props}
      >
        {children}
      </code>
    ) : (
      <code
        className="block p-4 rounded-lg bg-transparent text-text font-[family-name:var(--font-code)] text-sm overflow-x-auto"
        {...props}
      >
        {children}
      </code>
    );
  },
  pre: ({ children }: { children?: React.ReactNode }) => (
    <pre className="bg-void-lighter rounded-lg overflow-hidden my-3 border border-void-surface">{children}</pre>
  ),
  ul: ({ children }: { children?: React.ReactNode }) => (
    <ul className="list-disc list-inside text-text space-y-1 my-2">{children}</ul>
  ),
  ol: ({ children }: { children?: React.ReactNode }) => (
    <ol className="list-decimal list-inside text-text space-y-1 my-2">{children}</ol>
  ),
  li: ({ children }: { children?: React.ReactNode }) => (
    <li className="text-text">{children}</li>
  ),
  a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-glow hover:text-glow-bright underline underline-offset-2"
    >
      {children}
    </a>
  ),
  strong: ({ children }: { children?: React.ReactNode }) => (
    <strong className="font-semibold text-text">{children}</strong>
  ),
  em: ({ children }: { children?: React.ReactNode }) => (
    <em className="italic text-text-muted">{children}</em>
  ),
  blockquote: ({ children }: { children?: React.ReactNode }) => (
    <blockquote className="border-l-2 border-glow/30 pl-4 my-3 text-text-muted italic">
      {children}
    </blockquote>
  ),
  h1: ({ children }: { children?: React.ReactNode }) => (
    <h1 className="text-xl font-bold text-text mt-4 mb-2 font-[family-name:var(--font-display)]">
      {children}
    </h1>
  ),
  h2: ({ children }: { children?: React.ReactNode }) => (
    <h2 className="text-lg font-bold text-text mt-3 mb-2 font-[family-name:var(--font-display)]">
      {children}
    </h2>
  ),
  h3: ({ children }: { children?: React.ReactNode }) => (
    <h3 className="text-base font-bold text-text mt-2 mb-1 font-[family-name:var(--font-display)]">
      {children}
    </h3>
  ),
  table: ({ children }: { children?: React.ReactNode }) => (
    <div className="overflow-x-auto my-4">
      <table className="min-w-full border-collapse border border-void-surface rounded-lg overflow-hidden">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }: { children?: React.ReactNode }) => (
    <thead className="bg-void-surface/50">{children}</thead>
  ),
  tbody: ({ children }: { children?: React.ReactNode }) => (
    <tbody className="divide-y divide-void-surface">{children}</tbody>
  ),
  tr: ({ children }: { children?: React.ReactNode }) => (
    <tr className="hover:bg-void-light/30 transition-colors">{children}</tr>
  ),
  th: ({ children }: { children?: React.ReactNode }) => (
    <th className="px-4 py-2.5 text-left text-xs font-semibold text-glow uppercase tracking-wider border-b border-void-surface">
      {children}
    </th>
  ),
  td: ({ children }: { children?: React.ReactNode }) => (
    <td className="px-4 py-3 text-sm text-text border-r border-void-surface/50 last:border-r-0">
      {children}
    </td>
  ),
};

// ---------------------------------------------------------------------------
// MessageBubble
// ---------------------------------------------------------------------------

function MessageBubble({
  message,
  hideInlineSources,
  onViewFile,
}: {
  message: ChatMessage;
  hideInlineSources?: boolean;
  onViewFile: (source: ChatSource) => void;
}) {
  const isUser = message.role === 'user';

  return (
    <div className="flex gap-3">
      {/* Avatar */}
      <div
        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${
          isUser ? 'bg-nebula/20 text-nebula' : 'bg-glow/20 text-glow'
        }`}
      >
        {isUser ? (
          <User className="h-3.5 w-3.5" />
        ) : (
          <Bot className="h-3.5 w-3.5" />
        )}
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1">
        {isUser ? (
          <p className="text-sm text-text">{message.content}</p>
        ) : (
          <>
            <div className="prose prose-invert max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={markdownComponents as never}
              >
                {message.content}
              </ReactMarkdown>
            </div>

            {/* Inline sources (hidden when side panel is active) */}
            {!hideInlineSources && message.sources && message.sources.length > 0 && (
              <InlineSources
                sources={message.sources}
                onViewFile={onViewFile}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// InlineSources - collapsible source cards below assistant messages
// ---------------------------------------------------------------------------

function InlineSources({
  sources,
  onViewFile,
}: {
  sources: ChatSource[];
  onViewFile: (source: ChatSource) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const visibleSources = expanded ? sources : sources.slice(0, 3);

  return (
    <div className="mt-4 pt-3 border-t border-void-surface/60">
      <button
        onClick={() => setExpanded((e) => !e)}
        className="flex items-center gap-2 mb-3 text-xs font-medium text-text-muted hover:text-glow transition-colors"
      >
        <FileCode2 className="h-3.5 w-3.5" />
        <span>{sources.length} source{sources.length !== 1 ? 's' : ''} referenced</span>
        {sources.length > 3 && (
          expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
        )}
      </button>

      <div className="space-y-2">
        {visibleSources.map((source, idx) => (
          <SourceCard
            key={idx}
            source={source}
            rank={idx + 1}
            onViewFile={() => onViewFile(source)}
          />
        ))}
      </div>

      {!expanded && sources.length > 3 && (
        <button
          onClick={() => setExpanded(true)}
          className="mt-2 text-xs text-text-dim hover:text-glow transition-colors"
        >
          + {sources.length - 3} more source{sources.length - 3 !== 1 ? 's' : ''}
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SourceCard - compact inline source with expandable snippet
// ---------------------------------------------------------------------------

function shortPath(filename: string): string {
  const parts = filename.split('/');
  if (parts.length <= 3) return filename;
  return `.../${parts.slice(-3).join('/')}`;
}

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

function unitTypeStyle(unitType?: string): string {
  switch (unitType) {
    case 'function': return 'text-glow bg-glow/10';
    case 'method': return 'text-nebula-bright bg-nebula/10';
    case 'class': return 'text-warning bg-warning/10';
    case 'module': return 'text-success bg-success/10';
    default: return 'text-text-dim bg-void-surface';
  }
}

function SourceCard({
  source,
  rank,
  onViewFile,
}: {
  source: ChatSource;
  rank: number;
  onViewFile: () => void;
}) {
  const [showSnippet, setShowSnippet] = useState(false);
  const snippet = source.snippet || '';
  const prismLang = toPrismLanguage(source.language);

  return (
    <div className="rounded-lg border border-void-surface bg-void-light/40 transition-all hover:border-glow/15">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-nebula/20 text-[9px] font-bold text-nebula-bright">
          {rank}
        </span>

        <div className="flex-1 min-w-0 flex items-center gap-2">
          <span className="truncate text-xs font-medium text-text" title={source.filename}>
            {shortPath(source.filename)}
          </span>
          {source.unit_type && (
            <span className={`shrink-0 rounded px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wide ${unitTypeStyle(source.unit_type)}`}>
              {source.unit_type}
            </span>
          )}
          {source.unit_name && (
            <span className="hidden sm:inline truncate text-[10px] font-[family-name:var(--font-code)] text-text-dim max-w-[120px]">
              {source.unit_name}
            </span>
          )}
          {source.start_line != null && (
            <span className="shrink-0 text-[10px] text-text-dim">
              L{source.start_line}{source.end_line != null && `\u2013${source.end_line}`}
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 shrink-0">
          {snippet && (
            <button
              onClick={() => setShowSnippet((s) => !s)}
              className="rounded p-1 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
              title={showSnippet ? 'Hide snippet' : 'Show snippet'}
            >
              {showSnippet ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
          )}
          <button
            onClick={onViewFile}
            className="rounded p-1 text-text-dim hover:bg-glow/10 hover:text-glow transition-colors"
            title="View full file"
          >
            <ExternalLink className="h-3 w-3" />
          </button>
        </div>
      </div>

      {/* Expandable snippet */}
      {showSnippet && snippet && (
        <div className="border-t border-void-surface/50">
          <Highlight
            theme={themes.nightOwl}
            code={snippet.trimEnd()}
            language={prismLang}
          >
            {({ tokens, getLineProps, getTokenProps }) => (
              <pre className="overflow-x-auto p-3 text-[11px] leading-[1.6] bg-void/50 max-h-64 overflow-y-auto">
                {tokens.map((line, i) => (
                  <div key={i} {...getLineProps({ line })} className="table-row">
                    <span className="table-cell select-none pr-3 text-right text-text-dim/40 text-[10px]">
                      {(source.start_line ?? 1) + i}
                    </span>
                    <span className="table-cell whitespace-pre-wrap">
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
// FileViewer Modal - fetches and displays full file
// ---------------------------------------------------------------------------

interface FileViewerData {
  projectId: string;
  filePath: string;
  language: string;
  highlightStart?: number;
  highlightEnd?: number;
}

interface FileUnit {
  unit_id: string;
  unit_type: string;
  name: string;
  start_line: number;
  end_line: number;
  source: string;
}

function FileViewerModal({
  data,
  onClose,
}: {
  data: FileViewerData;
  onClose: () => void;
}) {
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [lineCount, setLineCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const highlightRef = useRef<HTMLDivElement>(null);

  // Fetch file content
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setFetchError(null);

    fetch(`/api/projects/${data.projectId}/file/${data.filePath}`, {
      credentials: 'include',
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: { units: FileUnit[]; line_count: number }) => {
        if (cancelled) return;
        // Reconstruct file from ordered units
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
  }, [data.projectId, data.filePath]);

  // Scroll to highlighted lines after content loads
  useEffect(() => {
    if (fileContent && highlightRef.current) {
      setTimeout(() => {
        highlightRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 100);
    }
  }, [fileContent]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: globalThis.KeyboardEvent) => {
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

  const prismLang = toPrismLanguage(data.language);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-void/80 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative flex h-[85vh] w-[85vw] max-w-5xl flex-col rounded-xl border border-void-surface bg-void-light shadow-2xl">
        {/* Modal header */}
        <div className="flex items-center justify-between border-b border-void-surface px-5 py-3">
          <div className="flex items-center gap-3 min-w-0">
            <FileCode2 className="h-4 w-4 shrink-0 text-glow" />
            <span className="truncate text-sm font-medium text-text font-[family-name:var(--font-code)]">
              {data.filePath}
            </span>
            <span className="shrink-0 rounded bg-void-surface px-2 py-0.5 text-[10px] font-medium text-text-dim">
              {data.language}
            </span>
            {lineCount > 0 && (
              <span className="shrink-0 text-[10px] text-text-dim">
                {lineCount} lines
              </span>
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
            <Highlight
              theme={themes.nightOwl}
              code={fileContent}
              language={prismLang}
            >
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
                        <span className={`table-cell pl-3 whitespace-pre-wrap ${isHighlighted ? '' : ''}`}>
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
