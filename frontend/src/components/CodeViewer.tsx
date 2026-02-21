/**
 * CodeViewer Component
 *
 * Syntax-highlighted, read-only code display with line numbers.
 * Scrolls to and highlights a selected code unit range.
 */

import { useEffect, useMemo, useRef } from 'react';
import { Highlight, themes } from 'prism-react-renderer';
import type { CodeUnit } from '../types/index.ts';

/** Map CodeLoom language identifiers to Prism grammar names. */
const LANG_MAP: Record<string, string> = {
  python: 'python',
  javascript: 'javascript',
  typescript: 'typescript',
  java: 'java',
  csharp: 'csharp',
  jsx: 'jsx',
  tsx: 'tsx',
};

interface CodeViewerProps {
  content: string;
  filePath: string;
  language?: string;
  units?: CodeUnit[];
  selectedUnit?: CodeUnit | null;
  onSelectUnit?: (unit: CodeUnit) => void;
}

export function CodeViewer({
  content,
  filePath,
  language,
  units,
  selectedUnit,
  onSelectUnit,
}: CodeViewerProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const scrollTargetLine = selectedUnit?.start_line ?? null;

  const lines = useMemo(() => content.split('\n'), [content]);

  // Build a set of highlighted line numbers from the selected unit
  const highlightedLines = useMemo(() => {
    if (!selectedUnit) return new Set<number>();
    const set = new Set<number>();
    for (let i = selectedUnit.start_line; i <= selectedUnit.end_line; i++) {
      set.add(i);
    }
    return set;
  }, [selectedUnit]);

  // Scroll to the selected unit when it changes
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [selectedUnit]);

  const prismLang = LANG_MAP[language ?? ''] ?? 'python';

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-void-surface bg-void-light/50 px-4 py-2">
        <span className="truncate text-xs text-text-muted">{filePath}</span>
        <div className="flex items-center gap-3 text-xs text-text-dim">
          {language && (
            <span className="rounded bg-void-surface/50 px-2 py-0.5">{language}</span>
          )}
          <span>{lines.length} lines</span>
        </div>
      </div>

      {/* Unit badges (if units provided) */}
      {units && units.length > 0 && (
        <div className="flex flex-wrap gap-1 border-b border-void-surface bg-void-light/30 px-4 py-2">
          {units.map((unit) => (
            <button
              key={unit.unit_id}
              onClick={() => onSelectUnit?.(unit)}
              className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                selectedUnit?.unit_id === unit.unit_id
                  ? 'bg-glow/20 text-glow'
                  : 'bg-void-surface/50 text-text-muted hover:bg-void-surface hover:text-text-dim'
              }`}
            >
              {unit.unit_type}: {unit.name}
            </button>
          ))}
        </div>
      )}

      {/* Code area with syntax highlighting */}
      <div className="flex-1 overflow-auto">
        <Highlight
          theme={themes.nightOwl}
          code={content}
          language={prismLang}
        >
          {({ tokens, getLineProps, getTokenProps }) => (
            <pre
              className="min-w-max text-xs leading-5"
              style={{ background: 'transparent' }}
            >
              {tokens.map((line, idx) => {
                const lineNum = idx + 1;
                const isHighlighted = highlightedLines.has(lineNum);
                const lineProps = getLineProps({ line });
                return (
                  <div
                    key={lineNum}
                    ref={lineNum === scrollTargetLine ? scrollRef : undefined}
                    {...lineProps}
                    className={`flex ${
                      isHighlighted
                        ? 'bg-glow/10'
                        : 'hover:bg-void-light/50'
                    }`}
                    style={undefined}
                  >
                    <span className="inline-block w-12 shrink-0 select-none pr-4 text-right text-text-dim">
                      {lineNum}
                    </span>
                    <span className="flex-1 whitespace-pre pr-4">
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
      </div>
    </div>
  );
}
