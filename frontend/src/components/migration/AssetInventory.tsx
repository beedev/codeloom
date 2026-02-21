/**
 * AssetInventory — file-type breakdown with per-type migration strategy selection.
 *
 * Shown after plan creation and before discovery. Presents auto-suggested
 * strategies (rule-based, optionally LLM-refined) that the user can
 * confirm or adjust per language/file-type.
 *
 * Design reference: Stitch project 9709894738079143127 screen 11973a412bb04d5db9a3d04705426622
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  FileCode,
  Box,
  AlignLeft,
  Layers,
  Sparkles,
  Loader2,
  ArrowRight,
} from 'lucide-react';
import * as api from '../../services/api.ts';
import type {
  AssetInventoryItem,
  AssetStrategy,
  AssetStrategySpec,
  AssetInventoryResponse,
  MigrationPlan,
} from '../../types/index.ts';

interface AssetInventoryProps {
  planId: string;
  plan: MigrationPlan;
  onConfirm: () => void;
  onCancel?: () => void;
}

const STRATEGY_OPTIONS: { value: AssetStrategy; label: string }[] = [
  { value: 'version_upgrade', label: 'Version Upgrade' },
  { value: 'framework_migration', label: 'Framework Migration' },
  { value: 'rewrite', label: 'Rewrite' },
  { value: 'convert', label: 'Convert' },
  { value: 'keep_as_is', label: 'Keep As-Is' },
  { value: 'no_change', label: 'No Change' },
];

const ACTIVE_STRATEGIES: Set<AssetStrategy> = new Set([
  'version_upgrade', 'framework_migration', 'rewrite',
]);

const PASSIVE_STRATEGIES: Set<AssetStrategy> = new Set([
  'keep_as_is', 'convert',
]);

const TARGET_ENABLED_STRATEGIES: Set<AssetStrategy> = new Set([
  'version_upgrade', 'framework_migration', 'rewrite', 'convert',
]);

const LANG_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  java: { bg: 'bg-green-500/10', text: 'text-green-400', border: 'border-green-500/20' },
  python: { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/20' },
  javascript: { bg: 'bg-yellow-500/10', text: 'text-yellow-400', border: 'border-yellow-500/20' },
  typescript: { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/20' },
  csharp: { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/20' },
  sql: { bg: 'bg-yellow-500/10', text: 'text-yellow-400', border: 'border-yellow-500/20' },
  xml: { bg: 'bg-orange-500/10', text: 'text-orange-400', border: 'border-orange-500/20' },
  json: { bg: 'bg-cyan-500/10', text: 'text-cyan-400', border: 'border-cyan-500/20' },
  yaml: { bg: 'bg-cyan-500/10', text: 'text-cyan-400', border: 'border-cyan-500/20' },
  properties: { bg: 'bg-void-surface/50', text: 'text-text-dim', border: 'border-void-surface' },
};

const DEFAULT_LANG_COLOR = { bg: 'bg-void-surface/50', text: 'text-text-muted', border: 'border-void-surface' };

function getLangColor(lang: string) {
  return LANG_COLORS[lang.toLowerCase()] ?? DEFAULT_LANG_COLOR;
}

function formatNumber(n: number): string {
  return n.toLocaleString();
}

export function AssetInventory({ planId, plan, onConfirm, onCancel }: AssetInventoryProps) {
  const [inventory, setInventory] = useState<AssetInventoryItem[]>([]);
  const [strategies, setStrategies] = useState<Record<string, AssetStrategySpec>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isRefining, setIsRefining] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [llmRefined, setLlmRefined] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load asset inventory on mount
  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    api.getAssetInventory(planId)
      .then((data: AssetInventoryResponse) => {
        if (cancelled) return;
        setInventory(data.inventory);
        setStrategies(data.suggested_strategies);
        setLlmRefined(data.llm_refined);
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load asset inventory');
      })
      .finally(() => { if (!cancelled) setIsLoading(false); });
    return () => { cancelled = true; };
  }, [planId]);

  // Derived stats
  const totalFiles = useMemo(() => inventory.reduce((s, i) => s + i.file_count, 0), [inventory]);
  const totalUnits = useMemo(() => inventory.reduce((s, i) => s + i.unit_count, 0), [inventory]);
  const totalLines = useMemo(() => inventory.reduce((s, i) => s + i.total_lines, 0), [inventory]);

  const activeCount = useMemo(() => {
    return Object.values(strategies).filter(s => s.strategy !== 'no_change').length;
  }, [strategies]);

  const impactedFiles = useMemo(() => {
    return inventory
      .filter(item => strategies[item.language]?.strategy !== 'no_change')
      .reduce((s, i) => s + i.file_count, 0);
  }, [inventory, strategies]);

  // Handlers
  const updateStrategy = useCallback((lang: string, strategy: AssetStrategy) => {
    setStrategies(prev => ({
      ...prev,
      [lang]: {
        ...prev[lang],
        strategy,
        target: strategy === 'no_change' || strategy === 'keep_as_is' ? null : prev[lang]?.target ?? null,
      },
    }));
  }, []);

  const updateTarget = useCallback((lang: string, target: string) => {
    setStrategies(prev => ({
      ...prev,
      [lang]: { ...prev[lang], target: target || null },
    }));
  }, []);

  const toggleActive = useCallback((lang: string) => {
    setStrategies(prev => {
      const current = prev[lang];
      if (!current) return prev;
      const isActive = current.strategy !== 'no_change';
      return {
        ...prev,
        [lang]: {
          ...current,
          strategy: isActive ? 'no_change' : 'keep_as_is',
          target: isActive ? null : current.target,
        },
      };
    });
  }, []);

  const handleRefine = useCallback(async () => {
    setIsRefining(true);
    setError(null);
    try {
      const refined = await api.refineAssetInventory(planId);
      setStrategies(refined.suggested_strategies);
      setLlmRefined(refined.llm_refined);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'AI refinement failed');
    } finally {
      setIsRefining(false);
    }
  }, [planId]);

  const handleConfirm = useCallback(async () => {
    setIsSaving(true);
    setError(null);
    try {
      await api.saveAssetStrategies(planId, strategies);
      onConfirm();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save strategies');
    } finally {
      setIsSaving(false);
    }
  }, [planId, strategies, onConfirm]);

  // Migration context label
  const migrationLabel = plan.migration_type
    ? plan.migration_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : 'Migration';

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-3">
        <Loader2 className="h-8 w-8 text-glow animate-spin" />
        <p className="text-sm text-text-muted">Loading asset inventory...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Page header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-text">Asset Inventory</h2>
          <p className="mt-1 text-sm text-text-muted max-w-2xl">
            Review file types and assign migration strategies before running discovery.
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-void-surface bg-void-light px-4 py-2 text-sm">
          <ArrowRight className="h-4 w-4 text-glow" />
          <span className="font-medium text-text">{migrationLabel}</span>
          {plan.target_brief && (
            <span className="text-text-dim">· {plan.target_brief}</span>
          )}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard icon={<FileCode className="h-5 w-5" />} label="Total Files" value={formatNumber(totalFiles)} />
        <StatCard icon={<Box className="h-5 w-5" />} label="Total Units" value={formatNumber(totalUnits)} />
        <StatCard icon={<AlignLeft className="h-5 w-5" />} label="Total Lines" value={formatNumber(totalLines)} />
        <StatCard icon={<Layers className="h-5 w-5" />} label="Active Types" value={`${activeCount} of ${inventory.length}`} />
      </div>

      {/* Error banner */}
      {error && (
        <div className="rounded-md border border-danger/30 bg-danger/10 px-4 py-2 text-sm text-danger">
          {error}
        </div>
      )}

      {/* Data table */}
      <div className="flex flex-col rounded-lg border border-void-surface bg-void-light overflow-hidden">
        {/* Table header bar */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-void-surface">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-text">Detected Assets</h3>
            <span className="rounded-full bg-void-surface px-2 py-0.5 text-[10px] text-text-dim">
              {inventory.length} Types
            </span>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-void-surface bg-void/50 text-[11px] uppercase tracking-wider text-text-dim">
                <th className="w-10 p-3 text-center" />
                <th className="p-3 font-medium">Language</th>
                <th className="p-3 font-medium">Files</th>
                <th className="p-3 font-medium">Units</th>
                <th className="p-3 font-medium">Lines</th>
                <th className="p-3 font-medium min-w-[180px]">Migration Strategy</th>
                <th className="p-3 font-medium min-w-[140px]">Target</th>
                <th className="p-3 font-medium">Sample Paths</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-void-surface text-sm">
              {inventory.map(item => {
                const spec = strategies[item.language];
                const isActive = spec?.strategy !== 'no_change';
                const color = getLangColor(item.language);
                const targetEnabled = TARGET_ENABLED_STRATEGIES.has(spec?.strategy);

                return (
                  <tr
                    key={item.language}
                    className="group transition-colors hover:bg-void/40"
                  >
                    {/* Checkbox */}
                    <td className="p-3 text-center">
                      <input
                        type="checkbox"
                        checked={isActive}
                        onChange={() => toggleActive(item.language)}
                        className="h-4 w-4 rounded border-void-surface bg-void text-glow focus:ring-glow/50 focus:ring-offset-void-light cursor-pointer"
                      />
                    </td>

                    {/* Language badge */}
                    <td className="p-3">
                      <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${color.bg} ${color.text} ${color.border}`}>
                        {item.language}
                      </span>
                    </td>

                    {/* Counts */}
                    <td className="p-3 text-text-muted">{formatNumber(item.file_count)}</td>
                    <td className="p-3 text-text-muted">{formatNumber(item.unit_count)}</td>
                    <td className="p-3 text-text-muted">{formatNumber(item.total_lines)}</td>

                    {/* Strategy dropdown */}
                    <td className="p-3">
                      <select
                        value={spec?.strategy ?? 'no_change'}
                        onChange={e => updateStrategy(item.language, e.target.value as AssetStrategy)}
                        className={`w-full rounded-md px-2.5 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-glow cursor-pointer ${
                          isActive
                            ? 'border-glow/60 bg-void text-text'
                            : 'border-void-surface bg-void/50 text-text-dim'
                        }`}
                      >
                        {STRATEGY_OPTIONS.map(opt => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    </td>

                    {/* Target input */}
                    <td className="p-3">
                      {targetEnabled && isActive ? (
                        <input
                          type="text"
                          value={spec?.target ?? ''}
                          onChange={e => updateTarget(item.language, e.target.value)}
                          placeholder="Target..."
                          className="w-full rounded-md border border-void-surface bg-void px-2.5 py-1.5 text-sm text-text placeholder:text-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow"
                        />
                      ) : (
                        <span className="px-2.5 py-1.5 text-sm text-text-dim">—</span>
                      )}
                    </td>

                    {/* Sample paths */}
                    <td className="p-3">
                      {item.sample_paths.length > 0 && (
                        <code className="block max-w-[200px] truncate rounded border border-void-surface bg-void/50 px-2 py-1 font-code text-[11px] text-text-dim">
                          {item.sample_paths[0]}
                        </code>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Table footer — auto-suggestion note + AI refine */}
        <div className="flex items-center justify-between border-t border-void-surface bg-void/50 px-5 py-2.5">
          <div className="flex items-center gap-2 text-xs text-text-dim">
            <Sparkles className="h-3.5 w-3.5 text-glow" />
            <span>
              Auto-suggested based on:{' '}
              <span className="font-medium text-text-muted">{migrationLabel}</span>
            </span>
            {llmRefined && (
              <span className="ml-2 rounded-full bg-nebula/10 px-2 py-0.5 text-[10px] text-nebula-bright">
                AI-Refined
              </span>
            )}
          </div>
          <button
            onClick={handleRefine}
            disabled={isRefining}
            className="flex items-center gap-1 text-xs font-medium text-nebula-bright hover:text-nebula transition-colors disabled:opacity-50"
          >
            {isRefining ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Refining...
              </>
            ) : (
              <>
                Refine with AI
                <Sparkles className="h-3 w-3" />
              </>
            )}
          </button>
        </div>

        {/* LLM reasons — show per-row if refined */}
        {llmRefined && Object.entries(strategies).some(([, s]) => s.reason) && (
          <div className="border-t border-void-surface bg-void/30 px-5 py-3 space-y-1">
            {Object.entries(strategies)
              .filter(([, s]) => s.reason)
              .map(([lang, s]) => (
                <p key={lang} className="text-[11px] text-text-dim">
                  <span className="font-medium text-text-muted">{lang}:</span> {s.reason}
                </p>
              ))}
          </div>
        )}
      </div>

      {/* Bottom action bar */}
      <div className="flex items-center justify-between rounded-lg border border-void-surface bg-void-light px-5 py-4">
        <div className="flex items-center gap-4">
          <p className="text-sm text-text">
            <span className="font-bold text-glow">{activeCount} of {inventory.length}</span>{' '}
            asset types selected
          </p>
          <span className="h-4 w-px bg-void-surface" />
          <p className="text-xs text-text-dim">
            Total impacted files: {formatNumber(impactedFiles)}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {onCancel && (
            <button
              onClick={onCancel}
              className="px-4 py-2 text-sm text-text-muted hover:text-text transition-colors"
            >
              Cancel
            </button>
          )}
          <button
            onClick={handleConfirm}
            disabled={isSaving || activeCount === 0}
            className="flex items-center gap-2 rounded-md bg-glow px-5 py-2.5 text-sm font-semibold text-white hover:bg-glow-dim disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isSaving ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                Confirm & Run Discovery
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Stat card sub-component ──

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="group rounded-lg border border-void-surface bg-void-light p-4 transition-colors hover:border-glow/30">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-text-dim">{label}</span>
        <span className="rounded-md bg-glow/10 p-1.5 text-glow group-hover:bg-glow group-hover:text-white transition-colors">
          {icon}
        </span>
      </div>
      <span className="text-xl font-bold text-text">{value}</span>
    </div>
  );
}
