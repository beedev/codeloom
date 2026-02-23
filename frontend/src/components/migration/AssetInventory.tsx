/**
 * AssetInventory — file-type breakdown with per-type migration strategy selection.
 *
 * Shown after plan creation and before discovery. Presents auto-suggested
 * strategies (rule-based, optionally LLM-refined) that the user can
 * confirm or adjust per language/file-type.
 *
 * Design reference: Stitch project 9709894738079143127 screen 11973a412bb04d5db9a3d04705426622
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  FileCode,
  Box,
  AlignLeft,
  Layers,
  Sparkles,
  Loader2,
  ArrowRight,
  ChevronDown,
  ChevronRight,
  Info,
} from 'lucide-react';
import * as api from '../../services/api.ts';
import type {
  AssetInventoryItem,
  AssetStrategy,
  AssetStrategySpec,
  AssetInventoryResponse,
  MigrationPlan,
  MigrationLane,
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
  const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({});
  const [lanes, setLanes] = useState<MigrationLane[]>([]);
  const [suggestedLanes, setSuggestedLanes] = useState<Record<string, MigrationLane>>({});

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
        if (data.suggested_lanes) setSuggestedLanes(data.suggested_lanes);
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load asset inventory');
      })
      .finally(() => { if (!cancelled) setIsLoading(false); });

    api.listMigrationLanes()
      .then((data) => {
        if (!cancelled) setLanes(data);
      })
      .catch(() => {}); // Non-critical — lanes are optional

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

  const toggleExpand = useCallback((lang: string) => {
    setExpandedRows(prev => ({ ...prev, [lang]: !prev[lang] }));
  }, []);

  const updateLaneId = useCallback((lang: string, laneId: string | null) => {
    setStrategies(prev => ({
      ...prev,
      [lang]: { ...prev[lang], lane_id: laneId },
    }));
  }, []);

  const updateSubTypeStrategy = useCallback((lang: string, unitType: string, strategy: AssetStrategy) => {
    setStrategies(prev => {
      const current = prev[lang] || { strategy: 'convert', target: null, reason: null };
      const subTypes = { ...(current.sub_types || {}) };
      subTypes[unitType] = { ...subTypes[unitType], strategy };
      return { ...prev, [lang]: { ...current, sub_types: subTypes } };
    });
  }, []);

  const updateSubTypeLaneId = useCallback((lang: string, unitType: string, laneId: string | null) => {
    setStrategies(prev => {
      const current = prev[lang] || { strategy: 'convert', target: null, reason: null };
      const subTypes = { ...(current.sub_types || {}) };
      subTypes[unitType] = { ...subTypes[unitType], lane_id: laneId || undefined };
      return { ...prev, [lang]: { ...current, sub_types: subTypes } };
    });
  }, []);

  // Migration context label
  const migrationLabel = plan.migration_type
    ? plan.migration_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : 'Migration';

  // Derive primary target framework from strategies for cross-row lane hints
  const primaryTargetFramework = useMemo(() => {
    for (const [, spec] of Object.entries(strategies)) {
      if (spec.strategy === 'framework_migration' && spec.target) {
        return spec.target.toLowerCase();
      }
    }
    return null;
  }, [strategies]);

  const TARGET_HINTS: Record<string, string> = {
    springboot: 'Spring Data JPA repositories + Spring services',
    spring: 'Spring Data JPA repositories + Spring services',
    dotnet_core: '.NET Core EF Core DbContext + services (coming soon)',
    django: 'Django models + service layer (coming soon)',
  };

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

      {/* Instructional banner */}
      <div className="rounded-lg border border-nebula/20 bg-nebula/5 px-5 py-4">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-nebula-bright shrink-0 mt-0.5" />
          <div className="text-xs text-text-muted space-y-1.5">
            <p className="font-medium text-text-dim text-sm">How Asset Inventory Works</p>
            <p>CodeLoom analyzed your codebase and detected the file types below. For each type:</p>
            <ul className="list-disc pl-4 space-y-0.5">
              <li>Choose a <span className="font-medium text-text-dim">Migration Strategy</span> (Framework Migration, Convert, Keep As-Is, etc.)</li>
              <li>For SQL and XML files, expand the row to see sub-categories — each can have a different strategy</li>
              <li>When a <span className="font-medium text-text-dim">Migration Lane</span> is available, it provides deterministic transforms and quality gates</li>
              <li>Files without a lane are migrated by the AI engine using your target stack context</li>
            </ul>
          </div>
        </div>
      </div>

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
                <th className="w-8 p-3" />
                <th className="p-3 font-medium">Language</th>
                <th className="p-3 font-medium">Files</th>
                <th className="p-3 font-medium">Units</th>
                <th className="p-3 font-medium">Lines</th>
                <th className="p-3 font-medium min-w-[180px]">Migration Strategy</th>
                <th className="p-3 font-medium min-w-[140px]">Target</th>
                <th className="p-3 font-medium min-w-[180px]">Lane</th>
                <th className="p-3 font-medium">Sample Paths</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-void-surface text-sm">
              {inventory.map(item => {
                const spec = strategies[item.language];
                const isActive = spec?.strategy !== 'no_change';
                const color = getLangColor(item.language);
                const targetEnabled = TARGET_ENABLED_STRATEGIES.has(spec?.strategy);
                const hasSubTypes = (item.sub_types?.length ?? 0) > 0;
                const isExpanded = expandedRows[item.language] ?? false;
                const langLane = suggestedLanes[item.language];
                const selectedLaneId = spec?.lane_id ?? langLane?.lane_id ?? null;

                return (
                  <React.Fragment key={item.language}>
                    <tr className="group transition-colors hover:bg-void/40">
                      {/* Checkbox */}
                      <td className="p-3 text-center">
                        <input
                          type="checkbox"
                          checked={isActive}
                          onChange={() => toggleActive(item.language)}
                          className="h-4 w-4 rounded border-void-surface bg-void text-glow focus:ring-glow/50 focus:ring-offset-void-light cursor-pointer"
                        />
                      </td>

                      {/* Expand chevron */}
                      <td className="p-3">
                        {hasSubTypes ? (
                          <button
                            onClick={() => toggleExpand(item.language)}
                            className="text-text-dim hover:text-text transition-colors"
                          >
                            {isExpanded ? (
                              <ChevronDown className="h-4 w-4" />
                            ) : (
                              <ChevronRight className="h-4 w-4" />
                            )}
                          </button>
                        ) : null}
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
                          <span className="px-2.5 py-1.5 text-sm text-text-dim">&mdash;</span>
                        )}
                      </td>

                      {/* Lane selector */}
                      <td className="p-3">
                        {isActive && lanes.length > 0 ? (
                          <select
                            value={selectedLaneId ?? ''}
                            onChange={e => updateLaneId(item.language, e.target.value || null)}
                            className="w-full rounded-md border border-void-surface bg-void px-2.5 py-1.5 text-sm text-text focus:outline-none focus:ring-1 focus:ring-glow cursor-pointer"
                          >
                            <option value="">No lane</option>
                            {lanes.map(l => (
                              <option key={l.lane_id} value={l.lane_id}>
                                {l.display_name}
                                {suggestedLanes[item.language]?.lane_id === l.lane_id ? ' (suggested)' : ''}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <span className="px-2.5 py-1.5 text-sm text-text-dim">&mdash;</span>
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

                    {/* Expanded sub-type rows */}
                    {isExpanded && item.sub_types?.map(st => {
                      const stKey = `${item.language}:${st.unit_type}`;
                      const stSpec = spec?.sub_types?.[st.unit_type];
                      const stStrategy = stSpec?.strategy ?? spec?.strategy ?? 'convert';
                      const stLaneSuggestion = suggestedLanes[stKey];
                      const stLaneId = stSpec?.lane_id ?? stLaneSuggestion?.lane_id ?? null;
                      const stIsActive = stStrategy !== 'no_change';
                      const targetHint = stLaneId && primaryTargetFramework
                        ? TARGET_HINTS[primaryTargetFramework]
                        : null;

                      return (
                        <tr
                          key={stKey}
                          className="bg-void/20 text-xs border-t border-void-surface/50"
                        >
                          <td className="p-2" />
                          <td className="p-2" />
                          <td className="p-2 pl-8">
                            <span className="inline-flex items-center gap-1.5 text-text-dim">
                              <span className="w-3 border-t border-l border-void-surface h-3 rounded-bl" />
                              <code className="rounded bg-void-surface/50 px-1.5 py-0.5 text-[10px] font-code">
                                {st.unit_type}
                              </code>
                            </span>
                          </td>
                          <td className="p-2 text-text-dim">{st.file_count}</td>
                          <td className="p-2 text-text-dim">{st.unit_count}</td>
                          <td className="p-2" />
                          <td className="p-2">
                            <select
                              value={stStrategy}
                              onChange={e => updateSubTypeStrategy(item.language, st.unit_type, e.target.value as AssetStrategy)}
                              className={`w-full rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-glow cursor-pointer ${
                                stIsActive
                                  ? 'border-glow/40 bg-void text-text'
                                  : 'border-void-surface bg-void/50 text-text-dim'
                              }`}
                            >
                              {STRATEGY_OPTIONS.map(opt => (
                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                              ))}
                            </select>
                          </td>
                          <td className="p-2" />
                          <td className="p-2">
                            {stIsActive && lanes.length > 0 ? (
                              <div>
                                <select
                                  value={stLaneId ?? ''}
                                  onChange={e => updateSubTypeLaneId(item.language, st.unit_type, e.target.value || null)}
                                  className="w-full rounded border border-void-surface bg-void px-2 py-1 text-xs text-text focus:outline-none focus:ring-1 focus:ring-glow cursor-pointer"
                                >
                                  <option value="">No lane</option>
                                  {lanes.map(l => (
                                    <option key={l.lane_id} value={l.lane_id}>
                                      {l.display_name}
                                      {stLaneSuggestion?.lane_id === l.lane_id ? ' (suggested)' : ''}
                                    </option>
                                  ))}
                                </select>
                                {targetHint && (
                                  <p className="mt-1 text-[10px] text-text-dim">
                                    Target: {targetHint}
                                  </p>
                                )}
                              </div>
                            ) : (
                              <span className="text-text-dim">&mdash;</span>
                            )}
                          </td>
                          <td className="p-2">
                            {st.sample_names.length > 0 && (
                              <code className="text-[10px] text-text-dim font-code">
                                {st.sample_names.slice(0, 2).join(', ')}
                              </code>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </React.Fragment>
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
