/**
 * StatusBanner — header bar for the Understanding tab.
 * Shows brain icon, title, action button, progress bar, and error state.
 */

import { Brain, Play, RefreshCw, Loader2, AlertCircle } from 'lucide-react';
import type { AnalysisJobStatus } from '../../types/index.ts';

interface StatusBannerProps {
  jobStatus: AnalysisJobStatus | null;
  analysisCount: number;
  hasResults: boolean;
  onTriggerAnalysis: () => void;
}

export function StatusBanner({
  jobStatus,
  analysisCount,
  hasResults,
  onTriggerAnalysis,
}: StatusBannerProps) {
  const isRunning =
    jobStatus?.status === 'pending' || jobStatus?.status === 'running';
  const isFailed = jobStatus?.status === 'failed';
  const progress = jobStatus?.progress;
  const pct =
    progress && progress.total > 0
      ? Math.round((progress.completed / progress.total) * 100)
      : 0;

  return (
    <div className="shrink-0 border-b border-void-surface bg-void-light/30 px-6 py-4">
      <div className="mx-auto flex max-w-5xl items-center justify-between">
        {/* Left: icon + title */}
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-nebula/10">
            <Brain className="h-4 w-4 text-nebula" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-text">
              Deep Understanding
            </h2>
            <p className="text-[10px] text-text-dim">
              {hasResults
                ? `${analysisCount} chain${analysisCount !== 1 ? 's' : ''} analyzed`
                : 'AI-powered codebase comprehension'}
            </p>
          </div>
        </div>

        {/* Right: action button */}
        <button
          onClick={onTriggerAnalysis}
          disabled={isRunning}
          className="flex items-center gap-1.5 rounded-md bg-glow px-3 py-1.5 text-xs font-medium text-white hover:bg-glow-dim disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isRunning ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Analyzing...
            </>
          ) : hasResults ? (
            <>
              <RefreshCw className="h-3.5 w-3.5" />
              Re-analyze
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5" />
              Start Analysis
            </>
          )}
        </button>
      </div>

      {/* Progress bar */}
      {isRunning && progress && progress.total > 0 && (
        <div className="mx-auto mt-3 max-w-5xl">
          <div className="flex items-center justify-between text-[10px] text-text-dim">
            <span>
              {progress.completed}/{progress.total} entry points
            </span>
            <span>{pct}%</span>
          </div>
          <div className="mt-1 h-1.5 rounded-full bg-void-surface">
            <div
              className="h-1.5 rounded-full bg-glow transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {/* Error banner */}
      {isFailed && jobStatus.errors && jobStatus.errors.length > 0 && (
        <div className="mx-auto mt-3 max-w-5xl rounded-md border border-danger/30 bg-danger/5 px-4 py-2.5">
          <div className="flex items-center gap-2 text-xs text-danger">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />
            <span className="font-medium">
              Analysis failed for {jobStatus.errors.length} entry point
              {jobStatus.errors.length !== 1 ? 's' : ''}
            </span>
          </div>
          <ul className="mt-1.5 space-y-0.5 pl-6 text-[10px] text-danger/80">
            {jobStatus.errors.slice(0, 3).map((e, i) => (
              <li key={i}>
                {e.entry_point}: {e.error}
              </li>
            ))}
            {jobStatus.errors.length > 3 && (
              <li>...and {jobStatus.errors.length - 3} more</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
