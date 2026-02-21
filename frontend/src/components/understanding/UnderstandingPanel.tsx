/**
 * UnderstandingPanel — root orchestrator for the Understanding tab.
 * Composes StatusBanner, EntryPointsSection, AnalysisResultsGrid,
 * and ChainDetailDrawer into a full-width scrollable layout.
 */

import { useMemo } from 'react';
import { Brain } from 'lucide-react';
import { useUnderstanding } from '../../hooks/useUnderstanding.ts';
import { StatusBanner } from './StatusBanner.tsx';
import { EntryPointsSection } from './EntryPointsSection.tsx';
import { AnalysisResultsGrid } from './AnalysisResultsGrid.tsx';
import { ChainDetailDrawer } from './ChainDetailDrawer.tsx';

interface UnderstandingPanelProps {
  projectId: string;
  asgStatus: string;
}

export function UnderstandingPanel({
  projectId,
  asgStatus,
}: UnderstandingPanelProps) {
  const {
    entryPoints,
    analyses,
    coveragePct,
    jobStatus,
    selectedChain,
    isLoadingEntryPoints,
    isLoadingResults,
    isLoadingChain,
    triggerAnalysis,
    loadChainDetail,
    clearSelectedChain,
  } = useUnderstanding(projectId, asgStatus);

  // Build set of analyzed unit IDs for dot indicators
  const analyzedUnitIds = useMemo(
    () => new Set(analyses.map((a) => a.entry_unit_id)),
    [analyses],
  );

  // Not-ready state
  if (asgStatus !== 'complete') {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="max-w-sm text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-void-surface">
            <Brain className="h-6 w-6 text-text-dim" />
          </div>
          <p className="mt-4 text-sm text-text-muted">
            Deep Understanding requires a completed ASG.
          </p>
          <p className="mt-1 text-[10px] text-text-dim">
            Build the ASG first, then return here to analyze your codebase.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Sticky header */}
      <StatusBanner
        jobStatus={jobStatus}
        analysisCount={analyses.length}
        hasResults={analyses.length > 0}
        onTriggerAnalysis={triggerAnalysis}
      />

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-5xl space-y-6 px-6 py-6">
          <EntryPointsSection
            entryPoints={entryPoints}
            analyzedUnitIds={analyzedUnitIds}
            isLoading={isLoadingEntryPoints}
          />
          <AnalysisResultsGrid
            analyses={analyses}
            coveragePct={coveragePct}
            isLoading={isLoadingResults}
            onSelectAnalysis={loadChainDetail}
          />
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-7 shrink-0 items-center gap-4 border-t border-void-surface bg-void-light/50 px-4 text-[10px] text-text-dim">
        <span className="flex items-center gap-1">
          <Brain className="h-3 w-3" />
          Understanding
        </span>
        {analyses.length > 0 && (
          <>
            <span>
              {analyses.length} chain{analyses.length !== 1 ? 's' : ''}
            </span>
            <span>{Math.round(coveragePct)}% coverage</span>
          </>
        )}
        {entryPoints.length > 0 && (
          <span>
            {entryPoints.length} entry point
            {entryPoints.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Chain detail drawer */}
      {selectedChain && (
        <ChainDetailDrawer
          chain={selectedChain}
          isLoading={isLoadingChain}
          onClose={clearSelectedChain}
        />
      )}
    </div>
  );
}
