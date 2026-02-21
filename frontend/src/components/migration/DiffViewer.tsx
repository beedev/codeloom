/**
 * DiffViewer — side-by-side and unified diff view for original vs migrated code.
 *
 * Matches Stitch screen 242b65aa55e44ec8844715ab4dc03b42:
 * - Top bar: file path breadcrumb (source → target), language badges, view toggle
 * - Stats bar: additions/deletions/modifications with change navigation
 * - Split or unified diff with colored line backgrounds
 * - Collapsed unchanged sections
 */

import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import { Highlight, themes } from 'prism-react-renderer';
import { diffLines, type Change } from 'diff';
import { ChevronUp, ChevronDown, ArrowRight } from 'lucide-react';
import type { DiffViewMode } from '../../types/index.ts';

// ── Language label mapping ──────────────────────────────────────────
const LANG_LABELS: Record<string, string> = {
  java: 'Java',
  csharp: 'C#',
  python: 'Python',
  javascript: 'JavaScript',
  typescript: 'TypeScript',
  go: 'Go',
  rust: 'Rust',
};

const PRISM_LANG: Record<string, string> = {
  python: 'python',
  java: 'java',
  javascript: 'javascript',
  typescript: 'typescript',
  csharp: 'csharp',
  go: 'go',
  rust: 'rust',
};

// ── Types ───────────────────────────────────────────────────────────

interface DiffLine {
  type: 'add' | 'remove' | 'unchanged';
  content: string;
  oldLineNum: number | null;
  newLineNum: number | null;
}

interface CollapsedGroup {
  type: 'collapsed';
  count: number;
  startOldLine: number;
  startNewLine: number;
}

type DiffRow = DiffLine | CollapsedGroup;

interface DiffStats {
  additions: number;
  deletions: number;
  modifications: number;
}

// ── Props ───────────────────────────────────────────────────────────

interface DiffViewerProps {
  sourceFile: { file_path: string; language: string; content: string } | null;
  migratedFile: { file_path: string; language: string; content: string };
  sourceLanguage: string;
  targetLanguage: string;
}

// ── Collapse threshold ──────────────────────────────────────────────
const COLLAPSE_THRESHOLD = 3;

// ── Diff computation ────────────────────────────────────────────────

function computeDiffLines(source: string, target: string): DiffLine[] {
  const changes: Change[] = diffLines(source, target);
  const lines: DiffLine[] = [];
  let oldLine = 1;
  let newLine = 1;

  for (const change of changes) {
    const content = change.value.endsWith('\n')
      ? change.value.slice(0, -1)
      : change.value;
    const subLines = content.split('\n');

    for (const sub of subLines) {
      if (change.added) {
        lines.push({ type: 'add', content: sub, oldLineNum: null, newLineNum: newLine++ });
      } else if (change.removed) {
        lines.push({ type: 'remove', content: sub, oldLineNum: oldLine++, newLineNum: null });
      } else {
        lines.push({ type: 'unchanged', content: sub, oldLineNum: oldLine++, newLineNum: newLine++ });
      }
    }
  }

  return lines;
}

function collapseUnchanged(lines: DiffLine[]): DiffRow[] {
  const rows: DiffRow[] = [];
  let i = 0;

  while (i < lines.length) {
    if (lines[i].type !== 'unchanged') {
      rows.push(lines[i]);
      i++;
      continue;
    }

    // Count consecutive unchanged
    let j = i;
    while (j < lines.length && lines[j].type === 'unchanged') j++;
    const count = j - i;

    if (count > COLLAPSE_THRESHOLD * 2 + 1) {
      // Show first COLLAPSE_THRESHOLD, collapse middle, show last COLLAPSE_THRESHOLD
      for (let k = i; k < i + COLLAPSE_THRESHOLD; k++) rows.push(lines[k]);
      rows.push({
        type: 'collapsed',
        count: count - COLLAPSE_THRESHOLD * 2,
        startOldLine: lines[i + COLLAPSE_THRESHOLD].oldLineNum ?? 0,
        startNewLine: lines[i + COLLAPSE_THRESHOLD].newLineNum ?? 0,
      });
      for (let k = j - COLLAPSE_THRESHOLD; k < j; k++) rows.push(lines[k]);
    } else {
      for (let k = i; k < j; k++) rows.push(lines[k]);
    }
    i = j;
  }

  return rows;
}

function computeStats(lines: DiffLine[]): DiffStats {
  const rawAdds = lines.filter(l => l.type === 'add').length;
  const rawRemoves = lines.filter(l => l.type === 'remove').length;
  const modifications = Math.min(rawAdds, rawRemoves);

  return {
    additions: rawAdds - modifications,
    deletions: rawRemoves - modifications,
    modifications,
  };
}

// Find indices of change groups for navigation
function findChangeGroups(rows: DiffRow[]): number[] {
  const groups: number[] = [];
  let inChange = false;

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const isChange = 'type' in row && (row.type === 'add' || row.type === 'remove');
    if (isChange && !inChange) {
      groups.push(i);
      inChange = true;
    } else if (!isChange) {
      inChange = false;
    }
  }

  return groups;
}

// ── Component ───────────────────────────────────────────────────────

export function DiffViewer({
  sourceFile,
  migratedFile,
  sourceLanguage,
  targetLanguage,
}: DiffViewerProps) {
  const [viewMode, setViewMode] = useState<DiffViewMode>(
    () => window.matchMedia('(max-width: 1024px)').matches ? 'unified' : 'side-by-side',
  );
  const [currentChangeIdx, setCurrentChangeIdx] = useState(0);
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const unifiedRef = useRef<HTMLDivElement>(null);

  // Auto-switch to unified on narrow screens
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 1024px)');
    const handler = (e: MediaQueryListEvent) => {
      if (e.matches) setViewMode('unified');
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // Compute diff
  const sourceContent = sourceFile?.content ?? '';
  const targetContent = migratedFile.content;

  const diffLines_ = useMemo(
    () => computeDiffLines(sourceContent, targetContent),
    [sourceContent, targetContent],
  );
  const rows = useMemo(() => collapseUnchanged(diffLines_), [diffLines_]);
  const stats = useMemo(() => computeStats(diffLines_), [diffLines_]);
  const changeGroups = useMemo(() => findChangeGroups(rows), [rows]);

  const isIdentical = stats.additions === 0 && stats.deletions === 0 && stats.modifications === 0;
  const isLargeFile = diffLines_.length > 2000;

  // Synchronized scrolling for side-by-side
  const isSyncing = useRef(false);
  const handleScroll = useCallback((source: 'left' | 'right') => {
    if (isSyncing.current) return;
    isSyncing.current = true;
    const from = source === 'left' ? leftRef.current : rightRef.current;
    const to = source === 'left' ? rightRef.current : leftRef.current;
    if (from && to) {
      to.scrollTop = from.scrollTop;
    }
    requestAnimationFrame(() => { isSyncing.current = false; });
  }, []);

  // Change navigation
  const navigateChange = useCallback((direction: 'prev' | 'next') => {
    if (changeGroups.length === 0) return;
    let nextIdx: number;
    if (direction === 'next') {
      nextIdx = currentChangeIdx >= changeGroups.length - 1 ? 0 : currentChangeIdx + 1;
    } else {
      nextIdx = currentChangeIdx <= 0 ? changeGroups.length - 1 : currentChangeIdx - 1;
    }
    setCurrentChangeIdx(nextIdx);

    // Scroll to the change
    const rowIdx = changeGroups[nextIdx];
    const target = viewMode === 'side-by-side' ? leftRef.current : unifiedRef.current;
    if (target) {
      const lineEl = target.querySelector(`[data-row="${rowIdx}"]`);
      lineEl?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [changeGroups, currentChangeIdx, viewMode]);

  const sourcePath = sourceFile?.file_path ?? '(no original)';
  const targetPath = migratedFile.file_path;
  const srcLangLabel = LANG_LABELS[sourceLanguage] ?? sourceLanguage;
  const tgtLangLabel = LANG_LABELS[targetLanguage] ?? targetLanguage;

  return (
    <div className="flex h-full flex-col bg-void">
      {/* ── Top Bar ── */}
      <div className="flex items-center justify-between border-b border-void-surface px-4 py-2.5">
        <div className="flex items-center gap-2 text-xs">
          {/* Source badge */}
          {sourceFile && (
            <span className="rounded bg-void-light px-2 py-0.5 font-medium text-text-muted">
              {srcLangLabel}
            </span>
          )}
          <span className="font-mono text-text-muted" title={sourcePath}>
            {sourcePath.split('/').pop()}
          </span>
          <ArrowRight className="h-3 w-3 text-text-dim" />
          <span className="rounded bg-void-light px-2 py-0.5 font-medium text-text-muted">
            {tgtLangLabel}
          </span>
          <span className="font-mono text-text-muted" title={targetPath}>
            {targetPath.split('/').pop()}
          </span>
        </div>

        {/* View mode toggle */}
        <div className="flex items-center rounded-lg border border-void-surface">
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`px-3 py-1 text-[11px] font-medium transition-colors ${
              viewMode === 'side-by-side'
                ? 'bg-glow/10 text-glow'
                : 'text-text-dim hover:text-text-muted'
            }`}
          >
            Side by Side
          </button>
          <button
            onClick={() => setViewMode('unified')}
            className={`px-3 py-1 text-[11px] font-medium transition-colors ${
              viewMode === 'unified'
                ? 'bg-glow/10 text-glow'
                : 'text-text-dim hover:text-text-muted'
            }`}
          >
            Unified
          </button>
        </div>
      </div>

      {/* ── Stats Bar ── */}
      <div className="flex items-center justify-between border-b border-void-surface px-4 py-1.5 text-xs">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1 text-green-400">
            <span className="inline-block h-2 w-2 rounded-full bg-green-400" />
            {stats.additions} Addition{stats.additions !== 1 ? 's' : ''}
          </span>
          <span className="flex items-center gap-1 text-red-400">
            <span className="inline-block h-2 w-2 rounded-full bg-red-400" />
            {stats.deletions} Deletion{stats.deletions !== 1 ? 's' : ''}
          </span>
          <span className="flex items-center gap-1 text-yellow-400">
            <span className="inline-block h-2 w-2 rounded-full bg-yellow-400" />
            {stats.modifications} Modification{stats.modifications !== 1 ? 's' : ''}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-text-dim">
            Change {changeGroups.length > 0 ? currentChangeIdx + 1 : 0} of {changeGroups.length}
          </span>
          <button
            onClick={() => navigateChange('prev')}
            className="rounded p-0.5 text-text-dim hover:bg-void-light hover:text-text-muted"
            disabled={changeGroups.length === 0}
          >
            <ChevronUp className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => navigateChange('next')}
            className="rounded p-0.5 text-text-dim hover:bg-void-light hover:text-text-muted"
            disabled={changeGroups.length === 0}
          >
            <ChevronDown className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* ── Large file warning ── */}
      {isLargeFile && (
        <div className="border-b border-warning/20 bg-warning/5 px-4 py-1.5 text-[11px] text-warning">
          Large file ({diffLines_.length.toLocaleString()} lines). Diff may be truncated for performance.
        </div>
      )}

      {/* ── Identical files ── */}
      {isIdentical && (
        <div className="flex flex-1 items-center justify-center text-text-dim">
          <div className="text-center">
            <span className="material-symbols-outlined text-3xl text-success">check_circle</span>
            <p className="mt-2 text-sm">Files are identical</p>
            <p className="mt-1 text-xs">No differences found between source and migrated code.</p>
          </div>
        </div>
      )}

      {/* ── No source file ── */}
      {!sourceFile && !isIdentical && (
        <div className="flex flex-1 overflow-hidden">
          <div className="flex w-1/2 items-center justify-center border-r border-void-surface text-text-dim">
            <div className="text-center">
              <span className="material-symbols-outlined text-2xl">help_outline</span>
              <p className="mt-2 text-xs">No original source found</p>
              <p className="mt-1 text-[10px]">Source file could not be matched to this migrated file.</p>
            </div>
          </div>
          <div className="w-1/2 overflow-auto">
            <DiffCodePanel
              content={migratedFile.content}
              language={targetLanguage}
              header="MIGRATED"
              badge="AI Generated"
            />
          </div>
        </div>
      )}

      {/* ── Side by Side ── */}
      {sourceFile && !isIdentical && viewMode === 'side-by-side' && (
        <div className="flex flex-1 overflow-hidden">
          {/* Left: Original */}
          <div
            ref={leftRef}
            className="w-1/2 overflow-auto border-r border-void-surface"
            onScroll={() => handleScroll('left')}
          >
            <div className="sticky top-0 z-10 flex items-center justify-between bg-void-light/90 px-3 py-1.5 text-[11px] font-medium text-text-dim backdrop-blur">
              <span>ORIGINAL</span>
            </div>
            <SplitDiffPanel
              rows={rows}
              side="left"
              language={sourceLanguage}
            />
          </div>

          {/* Right: Migrated */}
          <div
            ref={rightRef}
            className="w-1/2 overflow-auto"
            onScroll={() => handleScroll('right')}
          >
            <div className="sticky top-0 z-10 flex items-center gap-2 bg-void-light/90 px-3 py-1.5 text-[11px] font-medium text-text-dim backdrop-blur">
              <span>MIGRATED</span>
              <span className="rounded bg-glow/10 px-1.5 py-0.5 text-[9px] text-glow">
                AI Generated
              </span>
            </div>
            <SplitDiffPanel
              rows={rows}
              side="right"
              language={targetLanguage}
            />
          </div>
        </div>
      )}

      {/* ── Unified ── */}
      {sourceFile && !isIdentical && viewMode === 'unified' && (
        <div ref={unifiedRef} className="flex-1 overflow-auto">
          <UnifiedDiffPanel
            rows={rows}
            sourceLanguage={sourceLanguage}
            targetLanguage={targetLanguage}
          />
        </div>
      )}
    </div>
  );
}

// ── Split Diff Panel ────────────────────────────────────────────────

function SplitDiffPanel({
  rows,
  side,
  language,
}: {
  rows: DiffRow[];
  side: 'left' | 'right';
  language: string;
}) {
  const lang = PRISM_LANG[language] ?? language;

  return (
    <div className="font-code text-xs leading-relaxed">
      {rows.map((row, idx) => {
        if ('count' in row && row.type === 'collapsed') {
          return (
            <div
              key={`c-${idx}`}
              data-row={idx}
              className="flex items-center gap-2 bg-void-lighter/30 px-3 py-1 text-[10px] text-text-dim italic"
            >
              <span className="inline-block w-8" />
              {row.count} unchanged lines collapsed
            </div>
          );
        }

        const line = row as DiffLine;
        const lineNum = side === 'left' ? line.oldLineNum : line.newLineNum;
        const showContent = side === 'left' ? line.type !== 'add' : line.type !== 'remove';

        if (!showContent) {
          return (
            <div
              key={idx}
              data-row={idx}
              className="flex h-[22px] bg-void-lighter/10"
            >
              <span className="inline-block w-12 shrink-0 text-right text-text-dim/30 select-none pr-2" />
            </div>
          );
        }

        const bgColor =
          line.type === 'add'
            ? 'bg-green-500/8'
            : line.type === 'remove'
              ? 'bg-red-500/8'
              : '';

        const borderColor =
          line.type === 'add'
            ? 'border-l-2 border-green-500/40'
            : line.type === 'remove'
              ? 'border-l-2 border-red-500/40'
              : 'border-l-2 border-transparent';

        return (
          <div
            key={idx}
            data-row={idx}
            className={`flex ${bgColor} ${borderColor}`}
          >
            <span className="inline-block w-12 shrink-0 text-right text-text-dim/50 select-none pr-2 pt-px">
              {lineNum}
            </span>
            <HighlightedLine content={line.content} language={lang} />
          </div>
        );
      })}
    </div>
  );
}

// ── Unified Diff Panel ──────────────────────────────────────────────

function UnifiedDiffPanel({
  rows,
  sourceLanguage,
  targetLanguage,
}: {
  rows: DiffRow[];
  sourceLanguage: string;
  targetLanguage: string;
}) {
  return (
    <div className="font-code text-xs leading-relaxed">
      {rows.map((row, idx) => {
        if ('count' in row && row.type === 'collapsed') {
          return (
            <div
              key={`c-${idx}`}
              data-row={idx}
              className="flex items-center gap-2 bg-void-lighter/30 px-3 py-1 text-[10px] text-text-dim italic"
            >
              <span className="inline-block w-16" />
              {row.count} unchanged lines collapsed
            </div>
          );
        }

        const line = row as DiffLine;
        const prefix = line.type === 'add' ? '+' : line.type === 'remove' ? '-' : ' ';
        const lang = PRISM_LANG[line.type === 'add' ? targetLanguage : sourceLanguage] ?? sourceLanguage;

        const bgColor =
          line.type === 'add'
            ? 'bg-green-500/8'
            : line.type === 'remove'
              ? 'bg-red-500/8'
              : '';

        const prefixColor =
          line.type === 'add'
            ? 'text-green-400'
            : line.type === 'remove'
              ? 'text-red-400'
              : 'text-text-dim/30';

        return (
          <div
            key={idx}
            data-row={idx}
            className={`flex ${bgColor}`}
          >
            <span className="inline-block w-8 shrink-0 text-right text-text-dim/40 select-none pr-1 pt-px">
              {line.oldLineNum ?? ''}
            </span>
            <span className="inline-block w-8 shrink-0 text-right text-text-dim/40 select-none pr-1 pt-px">
              {line.newLineNum ?? ''}
            </span>
            <span className={`inline-block w-4 shrink-0 text-center select-none ${prefixColor}`}>
              {prefix}
            </span>
            <HighlightedLine content={line.content} language={lang} />
          </div>
        );
      })}
    </div>
  );
}

// ── Code Panel (single file, no diff) ───────────────────────────────

function DiffCodePanel({
  content,
  language,
  header,
  badge,
}: {
  content: string;
  language: string;
  header: string;
  badge?: string;
}) {
  const lang = PRISM_LANG[language] ?? language;

  return (
    <div>
      <div className="sticky top-0 z-10 flex items-center gap-2 bg-void-light/90 px-3 py-1.5 text-[11px] font-medium text-text-dim backdrop-blur">
        <span>{header}</span>
        {badge && (
          <span className="rounded bg-glow/10 px-1.5 py-0.5 text-[9px] text-glow">{badge}</span>
        )}
      </div>
      <Highlight theme={themes.nightOwl} code={content} language={lang}>
        {({ style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className="font-code text-xs leading-relaxed"
            style={{ ...style, background: 'transparent' }}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })} className="flex">
                <span className="inline-block w-12 shrink-0 text-right text-text-dim/50 select-none pr-2">
                  {i + 1}
                </span>
                <span>
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
  );
}

// ── Highlighted Line (single line with Prism) ───────────────────────

function HighlightedLine({
  content,
  language,
}: {
  content: string;
  language: string;
}) {
  return (
    <Highlight theme={themes.nightOwl} code={content} language={language}>
      {({ style, tokens, getTokenProps }) => (
        <span style={{ ...style, background: 'transparent' }}>
          {tokens[0]?.map((token, key) => (
            <span key={key} {...getTokenProps({ token })} />
          ))}
        </span>
      )}
    </Highlight>
  );
}
