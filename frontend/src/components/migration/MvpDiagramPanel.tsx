/**
 * MvpDiagramPanel — tab bar with 7 diagram types for an MVP.
 *
 * Split into Structural and Behavioral sections.
 * Fetches diagram on tab change, shows loading state for LLM-generated diagrams.
 * Refresh button for behavioral diagrams.
 */

import { useState, useEffect, useCallback } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import type { DiagramResponse, DiagramAvailability } from '../../types/index.ts';
import * as api from '../../services/api.ts';
import { DiagramViewer } from './DiagramViewer.tsx';

interface MvpDiagramPanelProps {
  planId: string;
  mvpId: number;
}

const STRUCTURAL_TABS = [
  { type: 'class', label: 'Class' },
  { type: 'package', label: 'Package' },
  { type: 'component', label: 'Component' },
] as const;

const BEHAVIORAL_TABS = [
  { type: 'sequence', label: 'Sequence' },
  { type: 'usecase', label: 'Use Case' },
  { type: 'activity', label: 'Activity' },
  { type: 'deployment', label: 'Deployment' },
] as const;

export function MvpDiagramPanel({ planId, mvpId }: MvpDiagramPanelProps) {
  const [activeTab, setActiveTab] = useState<string>('class');
  const [diagram, setDiagram] = useState<DiagramResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availability, setAvailability] = useState<DiagramAvailability[]>([]);

  // Load availability on mount / mvp change
  useEffect(() => {
    let cancelled = false;
    api.listMvpDiagrams(planId, mvpId)
      .then((data) => { if (!cancelled) setAvailability(data); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [planId, mvpId]);

  // Reset on MVP change
  useEffect(() => {
    setDiagram(null);
    setActiveTab('class');
    setError(null);
  }, [mvpId]);

  const fetchDiagram = useCallback(async (type: string, refresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const result = refresh
        ? await api.refreshMvpDiagram(planId, mvpId, type)
        : await api.getMvpDiagram(planId, mvpId, type);
      setDiagram(result);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to generate diagram';
      setError(msg);
      setDiagram(null);
    } finally {
      setLoading(false);
    }
  }, [planId, mvpId]);

  // Fetch when tab changes
  useEffect(() => {
    fetchDiagram(activeTab);
  }, [activeTab, fetchDiagram]);

  const isBehavioral = ['sequence', 'usecase', 'activity', 'deployment'].includes(activeTab);

  const getCacheStatus = (type: string) => {
    return availability.find((a) => a.diagram_type === type);
  };

  return (
    <div className="rounded border border-void-surface/50 bg-void-light/20">
      {/* Tab bar */}
      <div className="flex items-center gap-0.5 border-b border-void-surface/50 px-2 py-1.5">
        {/* Structural */}
        <span className="mr-1 text-[9px] font-medium uppercase tracking-wider text-text-dim">Structural</span>
        {STRUCTURAL_TABS.map((tab) => (
          <TabButton
            key={tab.type}
            label={tab.label}
            active={activeTab === tab.type}
            cached={getCacheStatus(tab.type)?.cached}
            onClick={() => setActiveTab(tab.type)}
          />
        ))}

        <div className="mx-2 h-3 w-px bg-void-surface/50" />

        {/* Behavioral */}
        <span className="mr-1 text-[9px] font-medium uppercase tracking-wider text-text-dim">Behavioral</span>
        {BEHAVIORAL_TABS.map((tab) => (
          <TabButton
            key={tab.type}
            label={tab.label}
            active={activeTab === tab.type}
            cached={getCacheStatus(tab.type)?.cached}
            onClick={() => setActiveTab(tab.type)}
          />
        ))}

        {/* Refresh button for behavioral diagrams */}
        {isBehavioral && diagram && !loading && (
          <button
            onClick={() => fetchDiagram(activeTab, true)}
            className="ml-auto rounded p-1 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
            title="Regenerate diagram"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {/* Content area */}
      <div className="min-h-[400px]">
        {loading && (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-text-muted">
            <Loader2 className="h-5 w-5 animate-spin text-glow" />
            <span className="text-[11px]">
              {isBehavioral ? 'Generating diagram via LLM...' : 'Generating diagram...'}
            </span>
          </div>
        )}

        {error && !loading && (
          <div className="flex flex-col items-center justify-center gap-2 py-12">
            <span className="text-[11px] text-error">{error}</span>
            <button
              onClick={() => fetchDiagram(activeTab)}
              className="rounded bg-void-surface px-3 py-1 text-[11px] text-text-muted hover:text-text transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {diagram && !loading && !error && (
          <DiagramViewer
            svg={diagram.svg}
            title={diagram.title}
            puml={diagram.puml}
          />
        )}
      </div>
    </div>
  );
}


function TabButton({
  label,
  active,
  cached,
  onClick,
}: {
  label: string;
  active: boolean;
  cached?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        relative rounded px-2 py-1 text-[11px] font-medium transition-colors
        ${active
          ? 'bg-glow/15 text-glow'
          : 'text-text-muted hover:bg-void-surface hover:text-text'
        }
      `}
    >
      {label}
      {cached && (
        <span className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-success" title="Cached" />
      )}
    </button>
  );
}
