/**
 * MetadataSearchPanel -- Search code units by decorator/annotation or return type.
 *
 * Provides a tabbed interface to query the ASG for units matching
 * specific metadata attributes, using two separate API endpoints.
 *
 * APIs:
 *   GET /api/projects/{projectId}/graph/units/by-decorator?decorator={name}
 *   GET /api/projects/{projectId}/graph/units/by-return-type?return_type={type}
 */

import { useState, useCallback } from 'react';
import { apiPath } from '../services/api.ts';
import {
  Loader2,
  Search,
  AtSign,
  Undo2,
  FileCode2,
  AlertCircle,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SearchResultUnit {
  unit_id: string;
  name: string;
  qualified_name: string;
  unit_type: string;
  language: string;
  file_path: string;
  start_line: number;
}

interface SearchResponse {
  results: SearchResultUnit[];
}

type SearchMode = 'decorator' | 'return_type';

interface Props {
  projectId: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function MetadataSearchPanel({ projectId }: Props) {
  const [mode, setMode] = useState<SearchMode>('decorator');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResultUnit[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  // ---- Search ----

  const handleSearch = useCallback(async () => {
    const trimmed = query.trim();
    if (!trimmed) return;

    setIsLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const endpoint =
        mode === 'decorator'
          ? `/api/projects/${projectId}/graph/units/by-decorator?decorator=${encodeURIComponent(trimmed)}`
          : `/api/projects/${projectId}/graph/units/by-return-type?return_type=${encodeURIComponent(trimmed)}`;

      const res = await fetch(apiPath(endpoint), {
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `HTTP ${res.status}`);
      }

      const json = (await res.json()) as SearchResponse;
      setResults(json.results ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, [projectId, mode, query]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') handleSearch();
    },
    [handleSearch],
  );

  const handleModeSwitch = useCallback((newMode: SearchMode) => {
    setMode(newMode);
    setResults([]);
    setHasSearched(false);
    setError(null);
  }, []);

  const handleClear = useCallback(() => {
    setQuery('');
    setResults([]);
    setHasSearched(false);
    setError(null);
  }, []);

  // ---- Render ----

  return (
    <div className="rounded-xl border border-void-surface/50 bg-void-light/20">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-void-surface/50 px-4 py-3">
        <Search className="h-3.5 w-3.5 text-text-dim" />
        <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
          Code Search
        </span>
        {hasSearched && results.length > 0 && (
          <span className="rounded-full bg-glow/10 px-2 py-0.5 text-[10px] font-medium text-glow">
            {results.length} result{results.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Mode tabs */}
      <div className="flex items-center border-b border-void-surface/30 px-4">
        <TabButton
          active={mode === 'decorator'}
          onClick={() => handleModeSwitch('decorator')}
          icon={<AtSign className="h-3 w-3" />}
          label="By Decorator"
        />
        <TabButton
          active={mode === 'return_type'}
          onClick={() => handleModeSwitch('return_type')}
          icon={<Undo2 className="h-3 w-3" />}
          label="By Return Type"
        />
      </div>

      {/* Search input */}
      <div className="flex items-center gap-2 border-b border-void-surface/30 px-4 py-2.5">
        <input
          type="text"
          placeholder={
            mode === 'decorator'
              ? 'e.g. @app.route, @Override, @Component...'
              : 'e.g. str, int, Promise<void>, List<String>...'
          }
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          className="flex-1 bg-transparent text-xs text-text-muted placeholder-text-dim/50 outline-none"
        />
        {query && (
          <button
            onClick={handleClear}
            className="rounded p-0.5 text-text-dim hover:bg-void-surface hover:text-text-muted"
            title="Clear"
          >
            <span className="text-[10px]">Clear</span>
          </button>
        )}
        <button
          onClick={handleSearch}
          disabled={!query.trim() || isLoading}
          className="flex items-center gap-1.5 rounded-md border border-glow/30 bg-glow/10 px-3 py-1.5 text-[11px] font-medium text-glow transition-colors hover:bg-glow/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isLoading ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Search className="h-3 w-3" />
          )}
          Search
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 border-b border-void-surface/30 bg-danger/5 px-4 py-2.5 text-xs text-danger">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          {error}
        </div>
      )}

      {/* Results */}
      {isLoading ? (
        <div className="flex items-center justify-center px-6 py-10">
          <Loader2 className="h-4 w-4 animate-spin text-text-dim" />
          <span className="ml-2 text-xs text-text-dim">Searching...</span>
        </div>
      ) : hasSearched && results.length === 0 ? (
        <div className="px-5 py-8 text-center">
          <p className="text-xs text-text-dim">
            No units found matching{' '}
            <span className="font-mono text-text-muted">
              {mode === 'decorator' ? `@${query}` : query}
            </span>
          </p>
          <p className="mt-1 text-[10px] text-text-dim/60">
            Try a different search term or check spelling.
          </p>
        </div>
      ) : results.length > 0 ? (
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
                  Type
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
              {results.map((unit) => (
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
                  </td>
                  <td className="py-2 px-3">
                    <span className="flex items-center gap-1.5 text-[11px] text-text-muted">
                      <FileCode2 className="h-3 w-3 shrink-0 text-text-dim/60" />
                      <span className="truncate font-mono">{unit.file_path}</span>
                    </span>
                  </td>
                  <td className="py-2 px-3">
                    <span className="rounded bg-void-surface/60 px-1.5 py-0.5 text-[10px] font-mono text-text-dim">
                      {unit.unit_type}
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
      ) : (
        /* Initial state -- no search performed */
        <div className="px-5 py-8 text-center">
          <p className="text-xs text-text-dim">
            {mode === 'decorator'
              ? 'Search for units by decorator or annotation name.'
              : 'Search for units by return type.'}
          </p>
          <p className="mt-1.5 text-[10px] text-text-dim/60">
            {mode === 'decorator'
              ? 'Examples: @app.route, @Override, @Component, @Transactional'
              : 'Examples: str, int, void, Promise, List<String>'}
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TabButton
// ---------------------------------------------------------------------------

function TabButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 border-b-2 px-3 py-2.5 text-[11px] font-medium transition-colors ${
        active
          ? 'border-glow text-glow'
          : 'border-transparent text-text-dim hover:text-text-muted'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}
