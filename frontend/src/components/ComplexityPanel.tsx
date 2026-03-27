/**
 * ComplexityPanel -- Cyclomatic complexity report for a project.
 *
 * Fetches the top-N most complex code units from the ASG complexity
 * endpoint and renders a sortable, filterable table with color-coded
 * severity badges.
 *
 * API: GET /api/projects/{projectId}/graph/complexity?sort=desc&limit=30
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Loader2,
  Gauge,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  SlidersHorizontal,
  FileCode2,
  AlertCircle,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ComplexityUnit {
  unit_id: string;
  name: string;
  qualified_name: string;
  unit_type: string;
  language: string;
  file_path: string;
  start_line: number;
  end_line: number;
  complexity: number;
}

interface ComplexityResponse {
  results: ComplexityUnit[];
}

interface Props {
  projectId: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

type SortField = 'name' | 'file_path' | 'language' | 'unit_type' | 'complexity';
type SortDir = 'asc' | 'desc';

const COMPLEXITY_TIERS: Array<{
  max: number;
  label: string;
  bg: string;
  text: string;
  border: string;
}> = [
  { max: 5,  label: 'Low',      bg: 'bg-success/15', text: 'text-success',  border: 'border-success/30' },
  { max: 10, label: 'Moderate', bg: 'bg-warning/15', text: 'text-warning',  border: 'border-warning/30' },
  { max: 20, label: 'High',     bg: 'bg-orange-500/15', text: 'text-orange-400', border: 'border-orange-500/30' },
  { max: Infinity, label: 'Very High', bg: 'bg-danger/15', text: 'text-danger', border: 'border-danger/30' },
];

function getTier(complexity: number) {
  return COMPLEXITY_TIERS.find((t) => complexity <= t.max) ?? COMPLEXITY_TIERS[3];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ComplexityPanel({ projectId }: Props) {
  const [data, setData] = useState<ComplexityUnit[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Sort
  const [sortField, setSortField] = useState<SortField>('complexity');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  // Filter: minimum complexity
  const [minComplexity, setMinComplexity] = useState(5);

  // ---- Fetch ----

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    fetch(`/api/projects/${projectId}/graph/complexity?sort=desc&limit=100`, {
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
    })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.text();
          throw new Error(body || `HTTP ${res.status}`);
        }
        return res.json() as Promise<ComplexityResponse>;
      })
      .then((json) => {
        if (!cancelled) setData(json.results ?? []);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load complexity data');
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => { cancelled = true; };
  }, [projectId]);

  // ---- Sort handler ----

  const handleSort = useCallback((field: SortField) => {
    setSortField((prev) => {
      if (prev === field) {
        setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
        return prev;
      }
      setSortDir(field === 'complexity' ? 'desc' : 'asc');
      return field;
    });
  }, []);

  // ---- Derived data ----

  const filtered = useMemo(() => {
    let items = data.filter((u) => u.complexity >= minComplexity);

    items.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case 'complexity': cmp = a.complexity - b.complexity; break;
        case 'name':       cmp = a.name.localeCompare(b.name); break;
        case 'file_path':  cmp = a.file_path.localeCompare(b.file_path); break;
        case 'language':   cmp = a.language.localeCompare(b.language); break;
        case 'unit_type':  cmp = a.unit_type.localeCompare(b.unit_type); break;
      }
      return sortDir === 'desc' ? -cmp : cmp;
    });

    return items;
  }, [data, minComplexity, sortField, sortDir]);

  // ---- Render: states ----

  if (isLoading) {
    return (
      <div className="flex items-center justify-center rounded-xl border border-void-surface/50 bg-void-light/20 px-6 py-12">
        <Loader2 className="h-4 w-4 animate-spin text-text-dim" />
        <span className="ml-2 text-xs text-text-dim">Loading complexity data...</span>
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
        <Gauge className="mt-0.5 h-4 w-4 shrink-0 text-text-dim/50" />
        <div>
          <p className="text-sm font-medium text-text-muted">No complexity data</p>
          <p className="mt-0.5 text-xs text-text-dim">
            Run ASG build first to compute cyclomatic complexity for code units.
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
          <Gauge className="h-3.5 w-3.5 text-text-dim" />
          <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            Cyclomatic Complexity
          </span>
          <span className="rounded-full bg-void-surface/60 px-2 py-0.5 text-[10px] font-medium text-text-dim">
            {filtered.length} / {data.length}
          </span>
        </div>

        {/* Min-complexity filter */}
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="h-3 w-3 text-text-dim" />
          <label className="text-[10px] text-text-dim">Min:</label>
          <input
            type="number"
            min={1}
            max={100}
            value={minComplexity}
            onChange={(e) => setMinComplexity(Math.max(1, parseInt(e.target.value) || 1))}
            className="w-14 rounded border border-void-surface bg-void px-2 py-1 text-xs text-text-muted outline-none focus:border-glow/40"
          />
        </div>
      </div>

      {/* Table */}
      {filtered.length === 0 ? (
        <div className="px-5 py-6 text-center text-xs text-text-dim">
          No units with complexity &ge; {minComplexity}. Try lowering the minimum.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-void-surface/50">
                <SortHeader field="name" label="Name" current={sortField} dir={sortDir} onClick={handleSort} />
                <SortHeader field="file_path" label="File" current={sortField} dir={sortDir} onClick={handleSort} />
                <SortHeader field="language" label="Language" current={sortField} dir={sortDir} onClick={handleSort} />
                <SortHeader field="unit_type" label="Type" current={sortField} dir={sortDir} onClick={handleSort} />
                <SortHeader field="complexity" label="Complexity" current={sortField} dir={sortDir} onClick={handleSort} align="center" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((unit) => {
                const tier = getTier(unit.complexity);
                return (
                  <tr
                    key={unit.unit_id}
                    className="border-t border-void-surface/30 transition-colors hover:bg-void-light/30"
                  >
                    <td className="py-2 pl-4 pr-3">
                      <span className="text-xs font-medium text-text">{unit.name}</span>
                      {unit.qualified_name && unit.qualified_name !== unit.name && (
                        <p className="mt-0.5 truncate text-[10px] text-text-dim">{unit.qualified_name}</p>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      <span className="flex items-center gap-1.5 text-[11px] text-text-muted">
                        <FileCode2 className="h-3 w-3 shrink-0 text-text-dim/60" />
                        <span className="truncate font-mono">{unit.file_path}</span>
                        <span className="shrink-0 text-[10px] text-text-dim">
                          :{unit.start_line}
                        </span>
                      </span>
                    </td>
                    <td className="py-2 px-3">
                      <span className="text-xs text-text-muted">{unit.language}</span>
                    </td>
                    <td className="py-2 px-3">
                      <span className="rounded bg-void-surface/60 px-1.5 py-0.5 text-[10px] font-mono text-text-dim">
                        {unit.unit_type}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-center">
                      <span
                        className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-mono font-semibold border ${tier.bg} ${tier.text} ${tier.border}`}
                      >
                        {unit.complexity}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-4 border-t border-void-surface/50 px-4 py-2.5">
        {COMPLEXITY_TIERS.map((tier) => (
          <div key={tier.label} className="flex items-center gap-1.5 text-[10px]">
            <span className={`inline-block h-2 w-2 rounded-full ${tier.bg} border ${tier.border}`} />
            <span className="text-text-dim">
              {tier.label}
              {tier.max === Infinity ? ' (21+)' : ` (1-${tier.max})`}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SortHeader
// ---------------------------------------------------------------------------

function SortHeader({
  field,
  label,
  current,
  dir,
  onClick,
  align = 'left',
}: {
  field: SortField;
  label: string;
  current: SortField;
  dir: SortDir;
  onClick: (f: SortField) => void;
  align?: 'left' | 'center';
}) {
  const isActive = current === field;
  const Icon = isActive ? (dir === 'asc' ? ArrowUp : ArrowDown) : ArrowUpDown;

  return (
    <th
      className={`py-2 px-3 text-[10px] font-semibold uppercase tracking-wider cursor-pointer select-none transition-colors hover:text-text-muted ${
        align === 'center' ? 'text-center' : 'text-left'
      } ${isActive ? 'text-text-muted' : 'text-text-dim'} ${field === 'name' ? 'pl-4' : ''}`}
      onClick={() => onClick(field)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <Icon className="h-2.5 w-2.5" />
      </span>
    </th>
  );
}
