/**
 * AccuracyPanel — migration accuracy report with expandable per-program details.
 *
 * Fetches:
 *   GET /api/migration/{planId}/accuracy       → structured summary (scores, per-MVP table)
 *   GET /api/migration/{planId}/accuracy/report → raw MIGRATION_ACCURACY.md for program details
 *
 * Sections rendered:
 *   1. Score badges (pre-fix → post-fix) + fix counts
 *   2. Per-MVP summary table
 *   3. Expandable per-program analysis (parsed from markdown)
 */

import { useState, useEffect } from 'react';
import {
  Download, FileText, Loader2, ChevronDown, ChevronRight,
  Bug, AlertTriangle, CheckCircle2,
} from 'lucide-react';
import { getAccuracyReport, getAccuracyReportMarkdown, ApiError } from '../../services/api.ts';
import type { AccuracyReportData, PerMvpAccuracy } from '../../services/api.ts';

// ---------------------------------------------------------------------------
// Types for parsed markdown
// ---------------------------------------------------------------------------

interface BugItem {
  raw: string;          // display text (HTML stripped of ~~)
  isFixed: boolean;
  severity: string | null; // CRITICAL | HIGH | MEDIUM | LOW | null
}

interface FixItem {
  file: string;
  issue: string;
  fix: string;
  status: string;
}

interface ManualItem {
  file: string;
  issue: string;
  priority: string;
}

interface ParagraphRow {
  paragraph: string;
  match: string; // EXACT | DEVIATION | BUG | GAP | CORRECT
}

interface ParsedProgram {
  id: string;           // unique key
  header: string;       // "CALCBILL → `src/orders/billing.py`"
  sourceName: string;   // "CALCBILL"
  targetPath: string;   // "src/orders/billing.py"
  mvpSection: string;   // "MVP 0: Foundation & Data Models" — which MVP section this program belongs to
  verdict: string;      // worst match found (BUG > GAP > DEVIATION > EXACT)
  paragraphRows: ParagraphRow[];
  bugs: BugItem[];
  notes: string[];
}

// ---------------------------------------------------------------------------
// Markdown parser
// ---------------------------------------------------------------------------

function parseAccuracyMarkdown(md: string): ParsedProgram[] {
  const lines = md.split('\n');
  const programs: ParsedProgram[] = [];
  let currentMvpSection = '';

  // Walk through lines, tracking current ### MVP section and parsing #### program headers
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Track current MVP section (### MVP N: Name)
    if (line.startsWith('### ')) {
      const mvpHeader = line.replace(/^### /, '').trim();
      // Accept MVP section headers (e.g. "MVP 0: Foundation & Data Models")
      if (/^MVP\s+\d+/i.test(mvpHeader)) {
        currentMvpSection = mvpHeader;
      }
      i++;
      continue;
    }

    // Parse program headers: #### PROGRAM (source) → target OR #### PROGRAM → target
    if (!line.startsWith('#### ')) { i++; continue; }

    const headerLine = line.replace(/^#### /, '').trim();
    const arrowMatch = headerLine.match(/^([^\s→]+(?:\s*\([^)]*\))?)\s*→\s*`?([^`\n]+)`?/);
    if (!arrowMatch) { i++; continue; }

    const sourceName = arrowMatch[1].trim().replace(/\s*\([^)]*\)$/, '');
    const targetPath = arrowMatch[2].trim();
    i++;

    // Collect table rows (Construct | Status | Note) until next header or blank block
    const paragraphRows: ParagraphRow[] = [];
    let inTable = false;
    const progStartIdx = i;
    while (i < lines.length && !lines[i].startsWith('### ') && !lines[i].startsWith('#### ')) {
      const tl = lines[i];
      if (!tl.startsWith('|')) {
        if (inTable) { inTable = false; }
        i++;
        continue;
      }
      if (/^\|[-\s|]+$/.test(tl)) { inTable = true; i++; continue; } // separator
      inTable = true;
      const cells = tl.split('|').map(c => c.trim()).filter(Boolean);
      if (cells.length >= 2) {
        // Skip header rows
        if (/^(paragraph|construct|field)$/i.test(cells[0])) { i++; continue; }
        const para = cells[0].replace(/`/g, '');
        const status = cells.length >= 2 ? cells[1] : '';
        const note = cells.length >= 3 ? cells[2] : '';
        // Determine match from status column (✅ Correct, ⚠️ Gap, ❌ Bug, 📝 Deviation, ✅ FIXED)
        let match = 'CORRECT';
        const statusUpper = status.toUpperCase();
        if (statusUpper.includes('BUG') || statusUpper.includes('❌')) match = 'BUG';
        else if (statusUpper.includes('GAP') || statusUpper.includes('⚠️')) match = 'GAP';
        else if (statusUpper.includes('DEVIATION') || statusUpper.includes('📝')) match = 'DEVIATION';
        else if (statusUpper.includes('FIXED')) match = 'FIXED';
        paragraphRows.push({ paragraph: para, match: `${match}${note ? ' — ' + note : ''}` });
      }
      i++;
    }

    // Collect bugs and notes from the block between progStartIdx and current i
    const blockLines = lines.slice(progStartIdx, i);
    const bugs: BugItem[] = [];
    let inBugs = false;
    for (const bl of blockLines) {
      if (/^\*\*Bugs:\*\*/.test(bl)) { inBugs = true; continue; }
      if (/^\*\*Notes?:\*\*|^\*\*Fix:/.test(bl)) { inBugs = false; continue; }
      if (!inBugs || !bl.startsWith('- ')) continue;  // '- ' not just '-' (excludes --- hr)

      const isFixed = bl.includes('~~') || bl.includes('FIXED');
      // Strip markdown: leading "- ", strikethrough ~~...~~, bold **, backticks, FIXED marker
      let rawText = bl.replace(/^-\s+/, '').trim();
      // Remove strikethrough wrapper: ~~content~~ suffix
      rawText = rawText.replace(/^~~(.+?)~~.*$/, '$1').trim();
      // Remove bold markers but preserve asterisks inside parens like (1500-*)
      rawText = rawText.replace(/\*\*([^*]*)\*\*/g, '$1');
      rawText = rawText.replace(/`/g, '').trim();
      // Remove trailing FIXED markers
      rawText = rawText.replace(/✅\s*FIXED\s*$/, '').trim();
      const sevMatch = rawText.match(/^\[(\w+)\]/);
      const severity = sevMatch ? sevMatch[1] : null;
      const displayText = rawText.replace(/^\[\w+\]\s*/, '').trim();
      if (displayText || rawText) {
        bugs.push({ raw: displayText || rawText, isFixed, severity });
      }
    }

    const notes: string[] = [];
    let inNotes = false;
    for (const bl of blockLines) {
      if (/^\*\*Notes?:\*\*/.test(bl)) { inNotes = true; }
      if (inNotes && bl.startsWith('**') && !bl.includes('Notes')) { inNotes = false; }
      if (inNotes && bl.startsWith('**Notes')) {
        const afterColon = bl.replace(/^\*\*Notes?:\*\*\s*/, '').trim();
        if (afterColon) notes.push(afterColon);
        continue;
      }
      if (inNotes && bl.trim()) notes.push(bl.trim());
    }

    // Determine worst verdict
    const allMatches = paragraphRows.map(r => r.match.toUpperCase());
    let verdict = 'CORRECT';
    if (allMatches.some(m => m.includes('BUG'))) verdict = 'BUG';
    else if (allMatches.some(m => m.includes('GAP'))) verdict = 'GAP';
    else if (allMatches.some(m => m.includes('DEVIATION'))) verdict = 'DEVIATION';
    else if (allMatches.some(m => m.includes('EXACT'))) verdict = 'EXACT';
    if (bugs.some(b => !b.isFixed)) verdict = 'BUG';
    else if (bugs.length > 0 && bugs.every(b => b.isFixed)) {
      if (verdict === 'BUG') verdict = 'FIXED';
    }

    programs.push({
      id: `${sourceName}-${targetPath}`,
      header: headerLine,
      sourceName,
      targetPath,
      mvpSection: currentMvpSection,
      verdict,
      paragraphRows,
      bugs,
      notes: notes.filter(Boolean),
    });
  }

  return programs;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseFixesApplied(md: string): FixItem[] {
  const fixes: FixItem[] = [];
  const sectionMatch = md.match(/## Fixes Applied[^\n]*\n([\s\S]*?)(?=\n## |\n$)/);
  if (!sectionMatch) return fixes;
  const lines = sectionMatch[1].split('\n');
  for (const line of lines) {
    if (!line.startsWith('|') || /^\|[-\s|]+$/.test(line)) continue;
    const cells = line.split('|').map(c => c.trim()).filter(Boolean);
    if (cells.length < 5 || /^#$/i.test(cells[0])) continue;
    fixes.push({
      file: cells[1].replace(/`/g, ''),
      issue: cells[3],
      fix: cells[4],
      status: cells[5] ?? '',
    });
  }
  return fixes;
}

function parseManualItems(md: string): ManualItem[] {
  const items: ManualItem[] = [];
  const sectionMatch = md.match(/## Remaining Manual Items[^\n]*\n([\s\S]*?)(?=\n## |\n$)/);
  if (!sectionMatch) return items;
  const lines = sectionMatch[1].split('\n');
  for (const line of lines) {
    if (!line.startsWith('|') || /^\|[-\s|]+$/.test(line)) continue;
    const cells = line.split('|').map(c => c.trim()).filter(Boolean);
    if (cells.length < 3 || /^#$/i.test(cells[0])) continue;
    items.push({
      file: cells[1].replace(/`/g, ''),
      issue: cells[2],
      priority: cells[3] ?? '',
    });
  }
  return items;
}

function scoreTier(score: number | null): 'green' | 'amber' | 'red' | 'none' {
  if (score === null) return 'none';
  if (score >= 90) return 'green';
  if (score >= 70) return 'amber';
  return 'red';
}

const TIER_CLASSES: Record<string, string> = {
  green: 'bg-success/15 text-success border border-success/30',
  amber: 'bg-warning/15 text-warning border border-warning/30',
  red:   'bg-danger/15 text-danger border border-danger/30',
  none:  'bg-void-surface/50 text-text-dim border border-void-surface',
};

const TIER_DOT: Record<string, string> = {
  green: 'bg-success',
  amber: 'bg-warning',
  red:   'bg-danger',
  none:  'bg-text-dim/40',
};

const VERDICT_BADGE: Record<string, string> = {
  BUG:       'bg-danger/15 text-danger border border-danger/30',
  GAP:       'bg-warning/20 text-warning border border-warning/30',
  DEVIATION: 'bg-glow/10 text-glow border border-glow/20',
  FIXED:     'bg-success/10 text-success border border-success/20',
  EXACT:     'bg-success/10 text-success border border-success/20',
  CORRECT:   'bg-success/10 text-success border border-success/20',
};

const MATCH_BADGE: Record<string, string> = {
  BUG:       'text-danger',
  GAP:       'text-warning',
  DEVIATION: 'text-glow/80',
  EXACT:     'text-success',
  CORRECT:   'text-success',
};

function relativeTime(isoString: string): string {
  const then = new Date(isoString).getTime();
  const diffMs = Date.now() - then;
  if (diffMs < 0) return 'just now';
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return 'just now';
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return `${Math.floor(diffHr / 24)}d ago`;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ScoreBadge({ label, score, large }: { label: string; score: number | null; large?: boolean }) {
  const tier = scoreTier(score);
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="text-[10px] uppercase tracking-wider text-text-dim">{label}</span>
      <span className={`flex items-center gap-1.5 rounded-lg px-3 font-mono font-bold ${
        large ? 'py-2 text-xl' : 'py-1.5 text-base'
      } ${TIER_CLASSES[tier]}`}>
        <span className={`h-2 w-2 shrink-0 rounded-full ${TIER_DOT[tier]}`} />
        {score !== null ? `${score}/100` : '—'}
      </span>
    </div>
  );
}

function MvpAccuracyRow({ row }: { row: PerMvpAccuracy }) {
  const preTier = scoreTier(row.score);
  const postTier = scoreTier(row.fixed_score);
  return (
    <tr className="border-t border-void-surface/50 hover:bg-void-light/20 transition-colors">
      <td className="py-2 pl-3 pr-4 text-xs font-medium text-text">{row.mvp_name}</td>
      <td className="py-2 px-3 text-center">
        <span className={`rounded px-1.5 py-0.5 text-[11px] font-mono font-semibold ${TIER_CLASSES[preTier]}`}>
          {row.score}/100
        </span>
      </td>
      <td className="py-2 px-3 text-center">
        <span className={`rounded px-1.5 py-0.5 text-[11px] font-mono font-semibold ${TIER_CLASSES[postTier]}`}>
          {row.fixed_score}/100
        </span>
      </td>
      <td className="py-2 px-3 text-center text-xs text-text-muted">{row.constructs}</td>
      <td className="py-2 px-3 text-center text-xs text-danger/80">{row.bugs || <span className="text-text-dim">—</span>}</td>
      <td className="py-2 px-3 text-center text-xs text-text-dim">{row.gaps || <span className="text-text-dim">—</span>}</td>
    </tr>
  );
}

function ProgramRow({ prog, expanded, onToggle }: {
  prog: ParsedProgram;
  expanded: boolean;
  onToggle: () => void;
}) {
  const activeBugs = prog.bugs.filter(b => !b.isFixed);
  const fixedBugs = prog.bugs.filter(b => b.isFixed);
  const verdictClass = VERDICT_BADGE[prog.verdict] ?? VERDICT_BADGE.CORRECT;

  return (
    <div className="border-b border-void-surface/40 last:border-0">
      {/* Header row */}
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors hover:bg-void-light/20"
      >
        {expanded
          ? <ChevronDown className="h-3.5 w-3.5 shrink-0 text-text-dim" />
          : <ChevronRight className="h-3.5 w-3.5 shrink-0 text-text-dim" />
        }

        {/* Source → target */}
        <div className="min-w-0 flex-1">
          <span className="text-xs font-semibold text-text">{prog.sourceName}</span>
          <span className="mx-1.5 text-text-dim/40">→</span>
          <span className="font-mono text-[11px] text-text-muted">{prog.targetPath}</span>
        </div>

        {/* Verdict badge */}
        <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${verdictClass}`}>
          {prog.verdict}
        </span>

        {/* Bug/fixed counts */}
        {activeBugs.length > 0 && (
          <span className="flex items-center gap-1 text-[11px] text-danger/80 shrink-0">
            <Bug className="h-3 w-3" />
            {activeBugs.length}
          </span>
        )}
        {fixedBugs.length > 0 && (
          <span className="flex items-center gap-1 text-[11px] text-success/70 shrink-0">
            <CheckCircle2 className="h-3 w-3" />
            {fixedBugs.length} fixed
          </span>
        )}
      </button>

      {/* Expanded body */}
      {expanded && (
        <div className="border-t border-void-surface/30 bg-void/40 px-4 pb-4 pt-3">

          {/* Paragraph table */}
          {prog.paragraphRows.length > 0 && (
            <div className="mb-3 overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="border-b border-void-surface/50">
                    <th className="pb-1.5 pr-4 text-left font-semibold text-text-dim uppercase tracking-wider">Paragraph</th>
                    <th className="pb-1.5 text-left font-semibold text-text-dim uppercase tracking-wider">Match</th>
                  </tr>
                </thead>
                <tbody>
                  {prog.paragraphRows.map((row, i) => {
                    const matchKey = row.match.trim().toUpperCase().split(/\s/)[0];
                    const matchClass = MATCH_BADGE[matchKey] ?? 'text-text-muted';
                    return (
                      <tr key={i} className="border-b border-void-surface/30 last:border-0">
                        <td className="py-1 pr-4 font-mono text-text-muted">{row.paragraph}</td>
                        <td className={`py-1 font-semibold ${matchClass}`}>{row.match}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Active bugs */}
          {activeBugs.length > 0 && (
            <div className="mb-3">
              <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-danger/70">
                Bugs requiring attention
              </p>
              <ul className="space-y-1.5">
                {activeBugs.map((bug, i) => (
                  <li key={i} className="flex gap-2 rounded-md bg-danger/5 px-3 py-2 text-[11px] text-text-muted">
                    <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-danger/70" />
                    <span>
                      {bug.severity && (
                        <span className="mr-1.5 rounded bg-danger/15 px-1 py-0.5 text-[10px] font-bold text-danger/90">
                          {bug.severity}
                        </span>
                      )}
                      {bug.raw}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Fixed bugs */}
          {fixedBugs.length > 0 && (
            <div className="mb-2">
              <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-success/60">
                Auto-fixed
              </p>
              <ul className="space-y-1">
                {fixedBugs.map((bug, i) => (
                  <li key={i} className="flex gap-2 rounded-md bg-success/5 px-3 py-1.5 text-[11px] text-text-dim line-through">
                    <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-success/50 no-underline" style={{textDecoration: 'none'}} />
                    <span className="no-underline" style={{textDecoration: 'none'}}>
                      ✅ {bug.raw}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Notes (no bugs) */}
          {prog.bugs.length === 0 && prog.notes.length > 0 && (
            <p className="text-[11px] text-text-dim italic">{prog.notes[0]}</p>
          )}
          {prog.bugs.length === 0 && prog.notes.length === 0 && (
            <p className="text-[11px] text-success/70">✓ No bugs detected</p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface AccuracyPanelProps {
  planId: string;
  selectedMvpName?: string | null;
}

export function AccuracyPanel({ planId, selectedMvpName }: AccuracyPanelProps) {
  const [data, setData] = useState<AccuracyReportData | null>(null);
  const [programs, setPrograms] = useState<ParsedProgram[]>([]);
  const [fixesApplied, setFixesApplied] = useState<FixItem[]>([]);
  const [manualItems, setManualItems] = useState<ManualItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Auto-open program details and expand all when an MVP is selected
  useEffect(() => {
    if (selectedMvpName && programs.length > 0) {
      const filtered = programs.filter(p =>
        p.mvpSection.toLowerCase().includes(selectedMvpName.toLowerCase())
      );
      if (filtered.length > 0) {
        setDetailsOpen(true);
        setExpandedIds(new Set(filtered.map(p => p.id)));
      }
    }
  }, [selectedMvpName, programs]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError(null);
      try {
        const [result, mdText] = await Promise.allSettled([
          getAccuracyReport(planId),
          getAccuracyReportMarkdown(planId),
        ]);

        if (cancelled) return;

        if (result.status === 'fulfilled') {
          setData(result.value);
        } else if (result.reason instanceof ApiError && result.reason.status === 404) {
          setData(null);
        } else if (result.reason instanceof Error) {
          setError(result.reason.message);
        }

        if (mdText.status === 'fulfilled') {
          setPrograms(parseAccuracyMarkdown(mdText.value));
          setFixesApplied(parseFixesApplied(mdText.value));
          setManualItems(parseManualItems(mdText.value));
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [planId]);

  function toggleProgram(id: string) {
    setExpandedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function expandAll() {
    setExpandedIds(new Set(programs.map(p => p.id)));
  }

  function collapseAll() {
    setExpandedIds(new Set());
  }

  // ── Loading ──
  if (isLoading) {
    return (
      <div className="mt-6 flex items-center justify-center rounded-xl border border-void-surface/50 bg-void-light/20 px-6 py-8">
        <Loader2 className="h-4 w-4 animate-spin text-text-dim" />
        <span className="ml-2 text-xs text-text-dim">Loading accuracy report…</span>
      </div>
    );
  }

  // ── Error ──
  if (error) {
    return (
      <div className="mt-6 rounded-xl border border-danger/20 bg-danger/5 px-4 py-3 text-xs text-danger">
        {error}
      </div>
    );
  }

  // ── Null / no-report ──
  const hasData = data?.has_report && (
    data.accuracy_score !== null || data.accuracy_fixed_score !== null
  );

  if (!hasData) {
    return (
      <div className="mt-6 flex items-start gap-3 rounded-xl border border-void-surface/50 bg-void-light/20 px-5 py-4">
        <FileText className="mt-0.5 h-4 w-4 shrink-0 text-text-dim/50" />
        <div>
          <p className="text-sm font-medium text-text-muted">No accuracy report yet</p>
          <p className="mt-0.5 text-xs text-text-dim">
            Run{' '}
            <code className="rounded bg-void-surface/80 px-1 py-0.5 font-mono text-[11px] text-glow">
              /migrate run
            </code>{' '}
            to generate a comparison report.
          </p>
        </div>
      </div>
    );
  }

  const hasPerMvp = (data!.accuracy_per_mvp?.length ?? 0) > 0;

  // Filter programs by selected MVP when one is active
  const filteredPrograms = selectedMvpName
    ? programs.filter(p => p.mvpSection.toLowerCase().includes(selectedMvpName.toLowerCase()))
    : programs;
  const bugPrograms = filteredPrograms.filter(p => p.bugs.some(b => !b.isFixed));
  const allPrograms = filteredPrograms;

  return (
    <div className="mt-6 rounded-xl border border-void-surface/50 bg-void-light/20">

      {/* ── Header ── */}
      <div className="flex items-center justify-between border-b border-void-surface/50 px-4 py-3">
        <div className="flex items-center gap-2">
          <FileText className="h-3.5 w-3.5 text-text-dim" />
          <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            Accuracy Report
          </span>
        </div>
        <div className="flex items-center gap-3">
          {data!.accuracy_last_run && (
            <span className="text-[10px] text-text-dim">
              Last run: {relativeTime(data!.accuracy_last_run)}
            </span>
          )}
          <a
            href={`/api/migration/${planId}/accuracy/report`}
            download="MIGRATION_ACCURACY.md"
            className="flex items-center gap-1.5 rounded-md border border-void-surface bg-void-surface/50 px-2.5 py-1 text-xs text-text-muted transition-colors hover:border-text-dim/30 hover:bg-void-surface hover:text-text"
          >
            <Download className="h-3 w-3" />
            Download
          </a>
        </div>
      </div>

      {/* ── Score row ── */}
      <div className="flex items-center gap-8 px-5 py-5">
        <ScoreBadge label="Pre-fix" score={data!.accuracy_score} />

        <svg className="h-4 w-8 shrink-0 text-text-dim/40" viewBox="0 0 32 16" fill="none" aria-hidden>
          <path d="M0 8h28M22 2l8 6-8 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>

        <ScoreBadge label="After fixes" score={data!.accuracy_fixed_score} large />

        <div className="ml-auto flex flex-col gap-1 text-right">
          {data!.accuracy_fixes_applied !== null && (
            <span className="text-sm font-semibold text-success">
              {data!.accuracy_fixes_applied} auto-fixed
            </span>
          )}
          {(data!.accuracy_fixes_pending ?? 0) > 0 && (
            <span className="text-sm font-semibold text-warning">
              {data!.accuracy_fixes_pending} manual review
            </span>
          )}
        </div>
      </div>

      {/* ── Fix pill row ── */}
      <div className="flex items-center gap-2 border-t border-void-surface/30 px-5 py-2.5">
        {(data!.accuracy_fixes_applied ?? 0) > 0 && (
          <span className="rounded-full bg-success/10 px-2.5 py-0.5 text-[11px] font-medium text-success">
            {data!.accuracy_fixes_applied} auto-fixed
          </span>
        )}
        {(data!.accuracy_fixes_pending ?? 0) > 0 && (
          <span className="rounded-full bg-warning/10 px-2.5 py-0.5 text-[11px] font-medium text-warning">
            {data!.accuracy_fixes_pending} need manual review
          </span>
        )}
        {(data!.accuracy_fixes_applied ?? 0) === 0 && (data!.accuracy_fixes_pending ?? 0) === 0 && (
          <span className="text-[11px] text-text-dim">No fixes applied</span>
        )}
      </div>

      {/* ── Per-MVP summary table ── */}
      {hasPerMvp && (
        <div className="border-t border-void-surface/50">
          <table className="w-full">
            <thead>
              <tr className="border-b border-void-surface/50">
                {['MVP', 'Pre-Fix', 'Post-Fix', 'Constructs', 'Bugs', 'Gaps'].map(h => (
                  <th key={h} className={`py-2 text-[10px] font-semibold uppercase tracking-wider text-text-dim ${
                    h === 'MVP' ? 'pl-3 pr-4 text-left' : 'px-3 text-center'
                  }`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data!.accuracy_per_mvp!.map(row => (
                <MvpAccuracyRow key={row.mvp_name} row={row} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Fixes Applied ── */}
      {fixesApplied.length > 0 && (
        <div className="border-t border-void-surface/50 px-4 py-3">
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-success/70">
            Surgical Fixes Applied ({fixesApplied.length})
          </p>
          <div className="space-y-1.5">
            {fixesApplied.map((fix, i) => (
              <div key={i} className="flex gap-2 rounded-md bg-success/5 px-3 py-2 text-[11px]">
                <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-success/60" />
                <div>
                  <span className="font-mono text-text-muted">{fix.file}</span>
                  <span className="mx-1.5 text-text-dim">—</span>
                  <span className="text-text-dim">{fix.issue}</span>
                  {fix.fix && (
                    <p className="mt-0.5 text-success/70">{fix.fix}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Remaining Manual Items ── */}
      {manualItems.length > 0 && (
        <div className="border-t border-void-surface/50 px-4 py-3">
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-warning/70">
            Remaining Manual Items ({manualItems.length})
          </p>
          <div className="space-y-1.5">
            {manualItems.map((item, i) => (
              <div key={i} className="flex gap-2 rounded-md bg-warning/5 px-3 py-2 text-[11px]">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning/60" />
                <div>
                  <span className="font-mono text-text-muted">{item.file}</span>
                  <span className="mx-1.5 text-text-dim">—</span>
                  <span className="text-text-dim">{item.issue}</span>
                  {item.priority && (
                    <span className="ml-2 rounded bg-void-surface/80 px-1.5 py-0.5 text-[10px] text-text-dim">
                      {item.priority}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Per-Program Details (expandable) ── */}
      {allPrograms.length > 0 && (
        <div className="border-t border-void-surface/50">
          {/* Section header */}
          <button
            onClick={() => setDetailsOpen(v => !v)}
            className="flex w-full items-center gap-2 px-4 py-3 text-left transition-colors hover:bg-void-light/20"
          >
            {detailsOpen
              ? <ChevronDown className="h-3.5 w-3.5 text-text-dim" />
              : <ChevronRight className="h-3.5 w-3.5 text-text-dim" />
            }
            <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
              Program Details
            </span>
            <span className="ml-1.5 text-[11px] text-text-dim">
              {allPrograms.length} programs
            </span>
            {bugPrograms.length > 0 && (
              <span className="ml-1 flex items-center gap-1 rounded-full bg-danger/10 px-2 py-0.5 text-[10px] font-medium text-danger">
                <Bug className="h-2.5 w-2.5" />
                {bugPrograms.length} with bugs
              </span>
            )}
            {detailsOpen && (
              <div className="ml-auto flex items-center gap-3" onClick={e => e.stopPropagation()}>
                <button
                  onClick={expandAll}
                  className="text-[10px] text-glow/70 hover:text-glow"
                >
                  Expand all
                </button>
                <button
                  onClick={collapseAll}
                  className="text-[10px] text-text-dim hover:text-text-muted"
                >
                  Collapse all
                </button>
              </div>
            )}
          </button>

          {/* Program list */}
          {detailsOpen && (
            <div className="border-t border-void-surface/30">
              {allPrograms.map(prog => (
                <ProgramRow
                  key={prog.id}
                  prog={prog}
                  expanded={expandedIds.has(prog.id)}
                  onToggle={() => toggleProgram(prog.id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
