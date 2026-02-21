/**
 * ChainDetailDrawer — right-side slide-in panel showing full chain analysis.
 * Sections: narrative, business rules, data entities, integrations,
 * side effects, cross-cutting concerns, traced units.
 */

import { useState, useEffect } from 'react';
import {
  X,
  FileText,
  BookOpen,
  Database,
  Plug,
  AlertTriangle,
  GitBranch,
  Layers,
  Hash,
  FileCode2,
  ChevronDown,
  ChevronRight,
  Loader2,
} from 'lucide-react';
import type { ChainDetail, EvidenceRef } from '../../types/index.ts';

const TIER_STYLES: Record<string, string> = {
  tier_1: 'bg-success/10 text-success',
  tier_2: 'bg-warning/10 text-warning',
  tier_3: 'bg-danger/10 text-danger',
};

const SEVERITY_STYLES: Record<string, string> = {
  high: 'bg-danger/10 text-danger',
  medium: 'bg-warning/10 text-warning',
  low: 'bg-success/10 text-success',
};

function confidenceColor(score: number): string {
  if (score >= 0.8) return 'text-success';
  if (score >= 0.5) return 'text-warning';
  return 'text-danger';
}

interface ChainDetailDrawerProps {
  chain: ChainDetail;
  isLoading: boolean;
  onClose: () => void;
}

export function ChainDetailDrawer({
  chain,
  isLoading,
  onClose,
}: ChainDetailDrawerProps) {
  // Close on Escape
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const tierStyle =
    TIER_STYLES[chain.tier] ?? 'bg-void-surface text-text-dim';
  const confPct = Math.round(chain.confidence_score * 100);
  const covPct = Math.round(chain.coverage_pct);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/30"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed bottom-0 right-0 top-0 z-50 flex w-full flex-col border-l border-void-surface bg-void shadow-2xl sm:w-[480px]"
        style={{ animation: 'slide-in-right 0.3s ease-out' }}
      >
        {isLoading ? (
          <div className="flex flex-1 items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-text-dim" />
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="shrink-0 border-b border-void-surface px-5 py-4">
              <div className="flex items-start justify-between">
                <span
                  className={`rounded px-1.5 py-0.5 text-[8px] font-bold uppercase ${tierStyle}`}
                >
                  {chain.tier.replace('_', ' ')}
                </span>
                <button
                  onClick={onClose}
                  className="rounded p-1 text-text-dim hover:bg-void-surface hover:text-text"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <p className="mt-2 font-[family-name:var(--font-code)] text-sm font-semibold text-text">
                {chain.entry_point.name}
              </p>
              <p className="mt-0.5 text-[10px] text-text-dim">
                {chain.entry_point.qualified_name}
              </p>
              <p className="text-[10px] text-text-dim/60">
                {chain.entry_point.file_path}
              </p>
              <div className="mt-2 flex items-center gap-3 text-[10px] text-text-dim">
                <span className="flex items-center gap-0.5">
                  <Layers className="h-3 w-3" />
                  {chain.total_units} units
                </span>
                <span className="flex items-center gap-0.5">
                  <Hash className="h-3 w-3" />
                  {chain.total_tokens.toLocaleString()} tokens
                </span>
                <span className={confidenceColor(chain.confidence_score)}>
                  {confPct}% confidence
                </span>
                <span>{covPct}% coverage</span>
              </div>
            </div>

            {/* Scrollable body */}
            <div className="flex-1 overflow-y-auto">
              {/* Narrative */}
              {chain.narrative && (
                <DrawerSection
                  icon={FileText}
                  title="Narrative"
                  defaultExpanded
                >
                  <p className="whitespace-pre-wrap text-xs leading-relaxed text-text-muted">
                    {chain.narrative}
                  </p>
                </DrawerSection>
              )}

              {/* Business Rules */}
              {chain.result.business_rules.length > 0 && (
                <DrawerSection
                  icon={BookOpen}
                  title="Business Rules"
                  count={chain.result.business_rules.length}
                  defaultExpanded
                >
                  <div className="space-y-2">
                    {chain.result.business_rules.map((rule, i) => (
                      <ItemCard key={i} description={rule.description}>
                        {rule.evidence.length > 0 && (
                          <EvidenceList evidence={rule.evidence} />
                        )}
                      </ItemCard>
                    ))}
                  </div>
                </DrawerSection>
              )}

              {/* Data Entities */}
              {chain.result.data_entities.length > 0 && (
                <DrawerSection
                  icon={Database}
                  title="Data Entities"
                  count={chain.result.data_entities.length}
                  defaultExpanded
                >
                  <div className="space-y-2">
                    {chain.result.data_entities.map((entity, i) => (
                      <ItemCard
                        key={i}
                        title={entity.name}
                        subtitle={entity.type}
                        description={entity.description}
                      >
                        {entity.evidence.length > 0 && (
                          <EvidenceList evidence={entity.evidence} />
                        )}
                      </ItemCard>
                    ))}
                  </div>
                </DrawerSection>
              )}

              {/* Integrations */}
              {chain.result.integrations.length > 0 && (
                <DrawerSection
                  icon={Plug}
                  title="Integrations"
                  count={chain.result.integrations.length}
                >
                  <div className="space-y-2">
                    {chain.result.integrations.map((integ, i) => (
                      <ItemCard
                        key={i}
                        title={integ.name}
                        subtitle={integ.type}
                        description={integ.description}
                      >
                        {integ.evidence.length > 0 && (
                          <EvidenceList evidence={integ.evidence} />
                        )}
                      </ItemCard>
                    ))}
                  </div>
                </DrawerSection>
              )}

              {/* Side Effects */}
              {chain.result.side_effects.length > 0 && (
                <DrawerSection
                  icon={AlertTriangle}
                  title="Side Effects"
                  count={chain.result.side_effects.length}
                >
                  <div className="space-y-2">
                    {chain.result.side_effects.map((effect, i) => {
                      const sevStyle =
                        SEVERITY_STYLES[effect.severity] ??
                        'bg-void-surface text-text-dim';
                      return (
                        <ItemCard key={i} description={effect.description}>
                          <span
                            className={`mt-1 inline-block rounded px-1.5 py-0.5 text-[8px] font-bold uppercase ${sevStyle}`}
                          >
                            {effect.severity}
                          </span>
                          {effect.evidence.length > 0 && (
                            <EvidenceList evidence={effect.evidence} />
                          )}
                        </ItemCard>
                      );
                    })}
                  </div>
                </DrawerSection>
              )}

              {/* Cross-Cutting Concerns */}
              {chain.result.cross_cutting_concerns.length > 0 && (
                <DrawerSection
                  icon={GitBranch}
                  title="Cross-Cutting Concerns"
                  count={chain.result.cross_cutting_concerns.length}
                >
                  <div className="flex flex-wrap gap-1.5">
                    {chain.result.cross_cutting_concerns.map((concern, i) => (
                      <span
                        key={i}
                        className="rounded-full bg-void-surface px-2 py-0.5 text-[10px] text-text-muted"
                      >
                        {concern}
                      </span>
                    ))}
                  </div>
                </DrawerSection>
              )}

              {/* Traced Units */}
              {chain.units.length > 0 && (
                <DrawerSection
                  icon={FileCode2}
                  title="Traced Units"
                  count={chain.units.length}
                  defaultExpanded={false}
                >
                  <div className="overflow-x-auto">
                    <table className="w-full text-[10px]">
                      <thead>
                        <tr className="border-b border-void-surface/30 text-left text-text-dim/50">
                          <th className="pb-1 pr-3 font-medium">Name</th>
                          <th className="pb-1 pr-3 font-medium">Type</th>
                          <th className="pb-1 pr-3 font-medium text-right">
                            Depth
                          </th>
                          <th className="pb-1 font-medium text-right">
                            Paths
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {chain.units.map((unit) => (
                          <tr
                            key={unit.unit_id}
                            className="border-b border-void-surface/10"
                          >
                            <td
                              className="truncate py-1 pr-3 font-[family-name:var(--font-code)] text-text-muted"
                              title={unit.qualified_name}
                            >
                              {unit.name}
                            </td>
                            <td className="py-1 pr-3 text-text-dim/60">
                              {unit.unit_type}
                            </td>
                            <td className="py-1 pr-3 text-right text-text-dim/60">
                              {unit.min_depth}
                            </td>
                            <td className="py-1 text-right text-text-dim/60">
                              {unit.path_count}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </DrawerSection>
              )}
            </div>

            {/* Footer */}
            <div className="shrink-0 border-t border-void-surface px-5 py-2.5 text-[9px] text-text-dim/40">
              Schema v{chain.schema_version} · {chain.prompt_version}
              {chain.analyzed_at && (
                <>
                  {' · '}
                  {new Date(chain.analyzed_at).toLocaleDateString()}
                </>
              )}
            </div>
          </>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// DrawerSection — reusable collapsible section
// ---------------------------------------------------------------------------

function DrawerSection({
  icon: Icon,
  title,
  count,
  defaultExpanded = true,
  children,
}: {
  icon: typeof FileText;
  title: string;
  count?: number;
  defaultExpanded?: boolean;
  children: React.ReactNode;
}) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="border-b border-void-surface">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex w-full items-center gap-2 px-5 py-2.5 text-left"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-text-dim/50" />
        ) : (
          <ChevronRight className="h-3 w-3 text-text-dim/50" />
        )}
        <Icon className="h-3.5 w-3.5 text-text-dim" />
        <span className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
          {title}
        </span>
        {count != null && (
          <span className="rounded-full bg-void-surface px-1.5 py-0.5 text-[8px] font-medium text-text-dim/50">
            {count}
          </span>
        )}
      </button>
      {isExpanded && <div className="px-5 pb-3">{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ItemCard — card for rules, entities, effects
// ---------------------------------------------------------------------------

function ItemCard({
  title,
  subtitle,
  description,
  children,
}: {
  title?: string;
  subtitle?: string;
  description: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-void-surface/30 bg-void-light/20 p-3">
      {title && (
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-text">{title}</span>
          {subtitle && (
            <span className="rounded bg-void-surface/50 px-1 py-0.5 text-[8px] text-text-dim/50">
              {subtitle}
            </span>
          )}
        </div>
      )}
      <p className="mt-1 text-[11px] leading-relaxed text-text-muted">
        {description}
      </p>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// EvidenceList — collapsible evidence references
// ---------------------------------------------------------------------------

function EvidenceList({ evidence }: { evidence: EvidenceRef[] }) {
  const [showEvidence, setShowEvidence] = useState(false);

  return (
    <div className="mt-1.5">
      <button
        onClick={(e) => {
          e.stopPropagation();
          setShowEvidence(!showEvidence);
        }}
        className="text-[9px] text-glow hover:text-glow-bright"
      >
        {showEvidence ? 'Hide' : 'Show'} {evidence.length} evidence ref
        {evidence.length !== 1 ? 's' : ''}
      </button>
      {showEvidence && (
        <div className="mt-1 space-y-1">
          {evidence.map((ref, i) => (
            <EvidenceRefItem key={i} evidence={ref} />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// EvidenceRefItem — inline code reference
// ---------------------------------------------------------------------------

function EvidenceRefItem({ evidence }: { evidence: EvidenceRef }) {
  return (
    <div className="rounded border border-void-surface/20 bg-void/50 px-2.5 py-1.5">
      <div className="flex items-center gap-1.5 text-[9px]">
        <span className="font-[family-name:var(--font-code)] text-glow-bright">
          {evidence.qualified_name}
        </span>
        <span className="text-text-dim/40">
          {evidence.file_path}:{evidence.start_line}-{evidence.end_line}
        </span>
      </div>
      {evidence.snippet && (
        <pre className="mt-1 overflow-x-auto whitespace-pre-wrap font-[family-name:var(--font-code)] text-[9px] leading-relaxed text-text-dim">
          {evidence.snippet}
        </pre>
      )}
    </div>
  );
}
