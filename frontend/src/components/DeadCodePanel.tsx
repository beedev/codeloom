/**
 * DeadCodePanel -- Shows potentially dead/unreachable functions.
 *
 * Lists code units that have zero detected callers in the ASG.
 * Includes a disclaimer banner since reflection, dynamic dispatch,
 * and external entry points can produce false positives.
 *
 * API: GET /api/projects/{projectId}/graph/dead-code?limit=50
 */

import { useState, useEffect, useMemo } from 'react';
import {
  Loader2,
  Skull,
  AlertCircle,
  AlertTriangle,
  FileCode2,
  Search,
  X,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DeadCodeUnit {
  unit_id: string;
  name: string;
  qualified_name: string;
  unit_type: string;
  language: string;
  file_path: string;
  start_line: number;
}

interface DeadCodeResponse {
  results: DeadCodeUnit[];
}

interface Props {
  projectId: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DeadCodePanel({ projectId }: Props) {
  const [data, setData] = useState<DeadCodeUnit[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  // ---- Fetch ----

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    fetch(`/api/projects/${projectId}/graph/dead-code?limit=200`, {
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
    })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.text();
          throw new Error(body || `HTTP ${res.status}`);
        }
        return res.json() as Promise<DeadCodeResponse>;
      })
      .then((json) => {
        if (!cancelled) setData(json.results ?? []);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load dead code data');
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => { cancelled = true; };
  }, [projectId]);

  // ---- Filtered data ----

  const filtered = useMemo(() => {
    if (!search.trim()) return data;
    const q = search.toLowerCase();
    return data.filter(
      (u) =>
        u.name.toLowerCase().includes(q) ||
        u.qualified_name.toLowerCase().includes(q) ||
        u.file_path.toLowerCase().includes(q) ||
        u.language.toLowerCase().includes(q),
    );
  }, [data, search]);

  // Group by language for the summary badges
  const langCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const u of data) {
      counts[u.language] = (counts[u.language] ?? 0) + 1;
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [data]);

  // ---- Render: states ----

  if (isLoading) {
    return (
      <div className="flex items-center justify-center rounded-xl border border-void-surface/50 bg-void-light/20 px-6 py-12">
        <Loader2 className="h-4 w-4 animate-spin text-text-dim" />
        <span className="ml-2 text-xs text-text-dim">Scanning for dead code...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-danger/20 bg-danger/5 px-4 py-3 text-xs text-danger">
        <AlertCircle className="h-4 w-4 shrink-0" />
        {error}
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-void-surface/50 bg-void-light/20 px-5 py-4">
        <Skull className="mt-0.5 h-4 w-4 shrink-0 text-text-dim/50" />
        <div>
          <p className="text-sm font-medium text-text-muted">No dead code detected</p>
          <p className="mt-0.5 text-xs text-text-dim">
            All functions have at least one detected caller in the ASG.
          </p>
        </div>
      </div>
    );
  }

  // ---- Render: main ----

  return (
    <div className="rounded-xl border border-void-surface/50 bg-void-light/20">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-void-surface/50 px-4 py-3">
        <div className="flex items-center gap-2">
          <Skull className="h-3.5 w-3.5 text-text-dim" />
          <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            Dead Code Detection
          </span>
          <span className="rounded-full bg-danger/10 px-2 py-0.5 text-[10px] font-semibold text-danger">
            {data.length}
          </span>
        </div>

        {/* Language breakdown pills */}
        <div className="flex items-center gap-1.5">
          {langCounts.slice(0, 5).map(([lang, count]) => (
            <span
              key={lang}
              className="rounded bg-void-surface/60 px-1.5 py-0.5 text-[10px] text-text-dim"
            >
              {lang} ({count})
            </span>
          ))}
        </div>
      </div>

      {/* Warning banner */}
      <div className="flex items-start gap-2.5 border-b border-void-surface/30 bg-warning/5 px-4 py-2.5">
        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning/70" />
        <p className="text-[11px] leading-relaxed text-text-dim">
          These functions have no detected callers in the ASG. They may still be called
          externally, via reflection, dynamic dispatch, or as framework entry points.
          Review before removing.
        </p>
      </div>

      {/* Search */}
      <div className="flex items-center gap-2 border-b border-void-surface/30 px-4 py-2">
        <Search className="h-3 w-3 text-text-dim" />
        <input
          type="text"
          placeholder="Filter by name, file, or language..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 bg-transparent text-xs text-text-muted placeholder-text-dim/60 outline-none"
        />
        {search && (
          <button onClick={() => setSearch('')} className="text-text-dim hover:text-text-muted">
            <X className="h-3 w-3" />
          </button>
        )}
        {search && (
          <span className="text-[10px] text-text-dim">
            {filtered.length} match{filtered.length !== 1 ? 'es' : ''}
          </span>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-void-surface/50">
              <th className="py-2 pl-4 pr-3 text-left text-[10px] font-semibold uppercase tracking-wider text-text-dim">
                Name
              </th>
              <th className="py-2 px-3 text-left text-[10px] font-semibold uppercase tracking-wider text-text-dim">
                File
              </th>
              <th className="py-2 px-3 text-left text-[10px] font-semibold uppercase tracking-wider text-text-dim">
                Language
              </th>
              <th className="py-2 px-3 text-center text-[10px] font-semibold uppercase tracking-wider text-text-dim">
                Line
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((unit) => (
              <tr
                key={unit.unit_id}
                className="border-t border-void-surface/30 transition-colors hover:bg-void-light/30"
              >
                <td className="py-2 pl-4 pr-3">
                  <span className="text-xs font-medium text-text">{unit.name}</span>
                  {unit.qualified_name && unit.qualified_name !== unit.name && (
                    <p className="mt-0.5 truncate text-[10px] text-text-dim">
                      {unit.qualified_name}
                    </p>
                  )}
                  <span className="ml-1.5 rounded bg-void-surface/60 px-1 py-0.5 text-[10px] font-mono text-text-dim">
                    {unit.unit_type}
                  </span>
                </td>
                <td className="py-2 px-3">
                  <span className="flex items-center gap-1.5 text-[11px] text-text-muted">
                    <FileCode2 className="h-3 w-3 shrink-0 text-text-dim/60" />
                    <span className="truncate font-mono">{unit.file_path}</span>
                  </span>
                </td>
                <td className="py-2 px-3">
                  <span className="text-xs text-text-muted">{unit.language}</span>
                </td>
                <td className="py-2 px-3 text-center">
                  <span className="font-mono text-[11px] text-text-dim">{unit.start_line}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="border-t border-void-surface/50 px-4 py-2.5 text-[10px] text-text-dim">
        Showing {filtered.length} of {data.length} unreferenced units
      </div>
    </div>
  );
}
