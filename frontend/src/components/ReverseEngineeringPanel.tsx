/**
 * ReverseEngineeringPanel -- generates and displays comprehensive
 * reverse engineering documentation for an ingested codebase.
 *
 * Calls the /api/reverse-engineer/{projectId}/ endpoints to generate,
 * poll, and display a 14-chapter documentation artifact rendered as
 * markdown with a chapter navigation sidebar.
 *
 * Includes ground truth validation: per-chapter confidence badges and
 * a summary bar showing verification metrics.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiPath } from '../services/api.ts';
import {
  BookOpen,
  Loader2,
  AlertCircle,
  Download,
  RefreshCw,
  CheckCircle2,
  Clock,
  XCircle,
  ShieldCheck,
  AlertTriangle,
  Flag,
  Sparkles,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DocSummary {
  doc_id: string;
  status: 'generating' | 'complete' | 'failed';
  progress: number;
  total_chapters: number;
  chapter_titles: string[];
  created_at: string;
}

interface ValidationChapter {
  confidence: number;
  issues: string[];
  verified_claims?: number;
  total_claims?: number;
  llm_generated?: boolean;
}

interface ValidationSummaryMetric {
  documented?: number;
  actual?: number;
  verified?: number;
  unverified?: number;
  total?: number;
  percentage: number;
}

interface ValidationData {
  overall_confidence: number;
  validated_at: string;
  chapters: Record<string, ValidationChapter>;
  summary: Record<string, ValidationSummaryMetric>;
  flagged_issues: string[];
}

interface DocFull {
  doc_id: string;
  status: 'generating' | 'complete' | 'failed';
  chapters: Record<string, string>;
  chapter_titles: string[];
  progress: number;
  total_chapters: number;
}

interface GenerateResponse {
  doc_id: string;
  status: string;
  chapters_generated: number;
  total_chapters: number;
}

interface Props {
  projectId: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const POLL_INTERVAL_MS = 4000;

const LLM_CHAPTERS = new Set([2, 5, 14]);

const STATUS_CONFIG: Record<string, { icon: typeof CheckCircle2; label: string; className: string }> = {
  complete: {
    icon: CheckCircle2,
    label: 'Complete',
    className: 'bg-success/10 text-success border-success/30',
  },
  generating: {
    icon: Clock,
    label: 'Generating',
    className: 'bg-nebula/10 text-nebula border-nebula/30',
  },
  failed: {
    icon: XCircle,
    label: 'Failed',
    className: 'bg-danger/10 text-danger border-danger/30',
  },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Return a confidence badge config based on percentage. */
function getConfidenceBadge(confidence: number): {
  color: string;
  bgColor: string;
  borderColor: string;
  icon: typeof CheckCircle2;
  label: string;
} {
  if (confidence >= 90) {
    return {
      color: 'text-success',
      bgColor: 'bg-success/10',
      borderColor: 'border-success/30',
      icon: CheckCircle2,
      label: `${confidence}%`,
    };
  }
  if (confidence >= 70) {
    return {
      color: 'text-amber-400',
      bgColor: 'bg-amber-400/10',
      borderColor: 'border-amber-400/30',
      icon: AlertTriangle,
      label: `${confidence}%`,
    };
  }
  return {
    color: 'text-danger',
    bgColor: 'bg-danger/10',
    borderColor: 'border-danger/30',
    icon: Flag,
    label: `${confidence}%`,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ReverseEngineeringPanel({ projectId }: Props) {
  const [doc, setDoc] = useState<DocFull | null>(null);
  const [selectedChapter, setSelectedChapter] = useState(1);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState({ current: 0, total: 14 });
  const [validation, setValidation] = useState<ValidationData | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---- API helpers --------------------------------------------------------

  const apiBase = `/api/reverse-engineer/${projectId}`;

  const loadLatestDoc = useCallback(async () => {
    try {
      const res = await fetch(apiPath(`${apiBase}/doc/latest`), {
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
      });

      if (res.status === 404) {
        // No doc yet -- that's fine
        setDoc(null);
        return;
      }

      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `HTTP ${res.status}`);
      }

      const data: DocFull = await res.json();
      setDoc(data);
      setProgress({ current: data.progress, total: data.total_chapters });

      // Extract validation from the _validation key in chapters
      const rawValidation = (data.chapters as Record<string, unknown>)?.['_validation'];
      if (rawValidation && typeof rawValidation === 'object') {
        setValidation(rawValidation as ValidationData);
      }

      if (data.status === 'generating') {
        setIsGenerating(true);
      } else {
        setIsGenerating(false);
      }
    } catch (err) {
      // Only set error if it's not a simple 404
      if (err instanceof Error && !err.message.includes('404')) {
        setError(err.message);
      }
    }
  }, [apiBase]);

  const handleGenerate = useCallback(async () => {
    setError(null);
    setIsGenerating(true);
    setProgress({ current: 0, total: 14 });

    try {
      const res = await fetch(apiPath(`${apiBase}/generate`), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `HTTP ${res.status}`);
      }

      const data: GenerateResponse = await res.json();
      setProgress({ current: data.chapters_generated, total: data.total_chapters });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start generation');
      setIsGenerating(false);
    }
  }, [apiBase]);

  const handleValidate = useCallback(async () => {
    if (!doc) return;

    setIsValidating(true);
    try {
      const res = await fetch(apiPath(`${apiBase}/validate`), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `HTTP ${res.status}`);
      }

      const data: ValidationData = await res.json();
      setValidation(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Validation failed');
    } finally {
      setIsValidating(false);
    }
  }, [apiBase, doc]);

  const handleDownload = useCallback(() => {
    if (!doc || !doc.chapters) return;

    const parts: string[] = [];
    const titles = doc.chapter_titles ?? [];

    for (let i = 1; i <= doc.total_chapters; i++) {
      const title = titles[i - 1] ?? `Chapter ${i}`;
      parts.push(`# Chapter ${i}: ${title}\n\n`);
      parts.push(doc.chapters[i] ?? '_Not yet generated._');
      parts.push('\n\n---\n\n');
    }

    const blob = new Blob([parts.join('')], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'reverse-engineering-doc.md';
    a.click();
    URL.revokeObjectURL(url);
  }, [doc]);

  // ---- Load on mount ------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    (async () => {
      setIsLoading(true);
      await loadLatestDoc();
      if (!cancelled) setIsLoading(false);
    })();

    return () => {
      cancelled = true;
    };
  }, [projectId, loadLatestDoc]);

  // ---- Poll during generation ---------------------------------------------

  useEffect(() => {
    if (!isGenerating) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(apiPath(`${apiBase}/doc/latest`), {
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) return;
        const data: DocFull = await res.json();

        setDoc(data);
        setProgress({ current: data.progress, total: data.total_chapters });

        // Check for validation data from auto-validate
        const rawValidation = (data.chapters as Record<string, unknown>)?.['_validation'];
        if (rawValidation && typeof rawValidation === 'object') {
          setValidation(rawValidation as ValidationData);
        }

        if (data.status !== 'generating') {
          setIsGenerating(false);
        }
      } catch {
        // Swallow poll errors
      }
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [isGenerating, apiBase]);

  // ---- Render: loading state ----------------------------------------------

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
        <span className="ml-2 text-xs text-text-dim">Loading documentation...</span>
      </div>
    );
  }

  // ---- Render: error state ------------------------------------------------

  if (error && !doc) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex items-center gap-2 rounded-xl border border-danger/20 bg-danger/5 px-4 py-3 text-xs text-danger">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      </div>
    );
  }

  // ---- Render: no doc, CTA ------------------------------------------------

  if (!doc && !isGenerating) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="max-w-md text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-xl bg-void-surface">
            <BookOpen className="h-7 w-7 text-text-dim" />
          </div>
          <h2 className="mt-4 text-sm font-semibold text-text">
            Reverse Engineering Documentation
          </h2>
          <p className="mt-2 text-xs text-text-muted">
            Generate a comprehensive 14-chapter technical document that reverse-engineers
            your codebase -- covering architecture, data models, business rules,
            integration points, and more.
          </p>
          <button
            onClick={handleGenerate}
            className="mt-6 inline-flex items-center gap-2 rounded-md bg-glow px-4 py-2 text-xs font-medium text-white hover:bg-glow-dim transition-colors"
          >
            <BookOpen className="h-3.5 w-3.5" />
            Generate Documentation
          </button>
        </div>
      </div>
    );
  }

  // ---- Render: generating (no doc content yet) ----------------------------

  if (isGenerating && !doc) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="max-w-sm text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-nebula" />
          <p className="mt-4 text-sm font-medium text-text">Generating documentation...</p>
          <p className="mt-1 text-xs text-text-muted">
            Chapter {progress.current} of {progress.total}
          </p>
          <div className="mx-auto mt-4 h-1.5 w-64 overflow-hidden rounded-full bg-void-surface">
            <div
              className="h-full rounded-full bg-nebula transition-all duration-500"
              style={{ width: `${(progress.current / progress.total) * 100}%` }}
            />
          </div>
        </div>
      </div>
    );
  }

  // ---- Render: main view --------------------------------------------------

  const titles = doc?.chapter_titles ?? [];
  const totalChapters = doc?.total_chapters ?? 14;
  const statusCfg = STATUS_CONFIG[doc?.status ?? 'generating'] ?? STATUS_CONFIG.generating;
  const StatusIcon = statusCfg.icon;

  const chapterContent = doc?.chapters?.[selectedChapter];

  // Compute validation summary text
  const validationSummaryText = validation
    ? [
        `${Math.round(validation.overall_confidence * 100)}% confidence`,
        validation.summary?.program_coverage
          ? `${validation.summary.program_coverage.documented}/${validation.summary.program_coverage.actual} programs`
          : null,
        validation.summary?.business_rules
          ? `${validation.summary.business_rules.verified}/${(validation.summary.business_rules.verified ?? 0) + (validation.summary.business_rules.unverified ?? 0)} rules verified`
          : null,
        validation.flagged_issues?.length
          ? `${validation.flagged_issues.length} issues flagged`
          : null,
      ].filter(Boolean).join(' | ')
    : null;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-void-surface/50 px-4 py-3">
        <div className="flex items-center gap-3">
          <BookOpen className="h-4 w-4 text-text-dim" />
          <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            Reverse Engineering Documentation
          </span>
          <span
            className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-medium ${statusCfg.className}`}
          >
            <StatusIcon className="h-3 w-3" />
            {statusCfg.label}
          </span>
          {isGenerating && (
            <span className="text-[10px] text-text-dim">
              Chapter {progress.current} / {progress.total}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {doc?.status === 'complete' && (
            <>
              <button
                onClick={handleValidate}
                disabled={isValidating}
                className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs transition-colors ${
                  isValidating
                    ? 'border-void-surface text-text-dim cursor-not-allowed opacity-50'
                    : 'border-void-surface text-text-muted hover:bg-void-surface hover:text-text'
                }`}
              >
                {isValidating ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <ShieldCheck className="h-3 w-3" />
                )}
                {isValidating ? 'Validating...' : 'Validate'}
              </button>
              <button
                onClick={handleDownload}
                className="flex items-center gap-1.5 rounded-md border border-void-surface px-2.5 py-1 text-xs text-text-muted hover:bg-void-surface hover:text-text transition-colors"
              >
                <Download className="h-3 w-3" />
                Download .md
              </button>
            </>
          )}
          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs transition-colors ${
              isGenerating
                ? 'border-void-surface text-text-dim cursor-not-allowed opacity-50'
                : 'border-void-surface text-text-muted hover:bg-void-surface hover:text-text'
            }`}
          >
            {isGenerating ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCw className="h-3 w-3" />
            )}
            {isGenerating ? 'Generating...' : 'Regenerate'}
          </button>
        </div>
      </div>

      {/* Validation summary bar */}
      {validation && (
        <div className="flex items-center gap-3 border-b border-void-surface/50 bg-void-light/30 px-4 py-2">
          <ShieldCheck className={`h-3.5 w-3.5 shrink-0 ${
            validation.overall_confidence >= 0.9
              ? 'text-success'
              : validation.overall_confidence >= 0.7
                ? 'text-amber-400'
                : 'text-danger'
          }`} />
          <span className="text-[11px] text-text-muted">
            Validation: {validationSummaryText}
          </span>
        </div>
      )}

      {/* Progress bar during generation */}
      {isGenerating && (
        <div className="h-1 w-full bg-void-surface">
          <div
            className="h-full bg-nebula transition-all duration-500"
            style={{ width: `${(progress.current / progress.total) * 100}%` }}
          />
        </div>
      )}

      {/* Main content: sidebar + reader */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chapter navigation sidebar */}
        <div className="w-72 shrink-0 overflow-y-auto border-r border-void-surface/50 bg-void-light/10">
          <div className="px-3 py-3">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
              Chapters
            </p>
          </div>
          <nav className="px-2 pb-4">
            {Array.from({ length: totalChapters }, (_, i) => i + 1).map((num) => {
              const title = titles[num - 1] ?? `Chapter ${num}`;
              const isSelected = selectedChapter === num;
              const hasContent = doc?.chapters?.[num] != null;

              // Validation badge for this chapter
              const chapterValidation = validation?.chapters?.[String(num)];
              const isLLM = LLM_CHAPTERS.has(num);

              let badgeEl: React.ReactNode = null;
              if (chapterValidation) {
                const badge = getConfidenceBadge(chapterValidation.confidence);
                const BadgeIcon = badge.icon;
                badgeEl = (
                  <span
                    className={`ml-auto flex items-center gap-0.5 rounded px-1 py-0.5 text-[9px] font-medium ${badge.bgColor} ${badge.color} ${badge.borderColor} border`}
                    title={chapterValidation.issues?.join('\n') || `${chapterValidation.confidence}% confidence`}
                  >
                    <BadgeIcon className="h-2.5 w-2.5" />
                    {badge.label}
                  </span>
                );
              } else if (isLLM && hasContent) {
                badgeEl = (
                  <span
                    className="ml-auto flex items-center gap-0.5 rounded border border-nebula/30 bg-nebula/10 px-1 py-0.5 text-[9px] font-medium text-nebula"
                    title="LLM-synthesized chapter -- manual review recommended"
                  >
                    <Sparkles className="h-2.5 w-2.5" />
                    LLM
                  </span>
                );
              }

              return (
                <button
                  key={num}
                  onClick={() => setSelectedChapter(num)}
                  className={`flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-xs transition-colors ${
                    isSelected
                      ? 'bg-glow/10 text-glow'
                      : hasContent
                        ? 'text-text-muted hover:bg-void-surface/50 hover:text-text'
                        : 'text-text-dim/50 hover:bg-void-surface/30'
                  }`}
                >
                  <span
                    className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded text-[10px] font-bold ${
                      isSelected
                        ? 'bg-glow/20 text-glow'
                        : hasContent
                          ? 'bg-void-surface/60 text-text-dim'
                          : 'bg-void-surface/30 text-text-dim/40'
                    }`}
                  >
                    {num}
                  </span>
                  <span className="leading-snug flex-1 truncate">{title}</span>
                  {badgeEl}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Content reader */}
        <div className="flex-1 overflow-y-auto">
          {chapterContent ? (
            <div className="mx-auto max-w-4xl px-8 py-6">
              <article className="prose prose-invert prose-sm max-w-none prose-headings:text-text prose-p:text-text-muted prose-strong:text-text prose-code:text-glow prose-code:bg-void-surface/50 prose-code:rounded prose-code:px-1.5 prose-code:py-0.5 prose-code:text-xs prose-pre:bg-void-surface/30 prose-pre:border prose-pre:border-void-surface/50 prose-a:text-glow prose-li:text-text-muted prose-th:text-text-muted prose-td:text-text-dim">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {chapterContent}
                </ReactMarkdown>
              </article>
            </div>
          ) : isGenerating ? (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <Loader2 className="mx-auto h-5 w-5 animate-spin text-text-dim" />
                <p className="mt-3 text-xs text-text-dim">
                  Generating chapter {selectedChapter}...
                </p>
              </div>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <BookOpen className="mx-auto h-6 w-6 text-text-dim/40" />
                <p className="mt-3 text-xs text-text-dim">
                  Select a chapter to view its content.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-7 shrink-0 items-center gap-4 border-t border-void-surface bg-void-light/50 px-4 text-[10px] text-text-dim">
        <span className="flex items-center gap-1">
          <BookOpen className="h-3 w-3" />
          Documentation
        </span>
        {doc && (
          <>
            <span>
              {Object.keys(doc.chapters).filter(k => !k.startsWith('_')).length} / {doc.total_chapters} chapters
            </span>
            {doc.status === 'complete' && (
              <span>
                {Object.entries(doc.chapters)
                  .filter(([k]) => !k.startsWith('_'))
                  .reduce((acc, [, c]) => acc + ((c as string)?.length ?? 0), 0)
                  .toLocaleString()} chars
              </span>
            )}
            {validation && (
              <span className="flex items-center gap-1">
                <ShieldCheck className="h-3 w-3" />
                {Math.round(validation.overall_confidence * 100)}% validated
              </span>
            )}
          </>
        )}
      </div>
    </div>
  );
}
