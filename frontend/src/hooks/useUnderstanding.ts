/**
 * useUnderstanding Hook
 *
 * Manages the Deep Understanding lifecycle:
 * entry point discovery, analysis triggering with polling,
 * result browsing, and chain detail loading.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  EntryPoint,
  AnalysisSummary,
  AnalysisJobStatus,
  ChainDetail,
} from '../types/index.ts';
import * as api from '../services/api.ts';

const POLL_INTERVAL_MS = 3000;

export interface UseUnderstandingReturn {
  entryPoints: EntryPoint[];
  analyses: AnalysisSummary[];
  coveragePct: number;
  jobStatus: AnalysisJobStatus | null;
  selectedChain: ChainDetail | null;
  isLoadingEntryPoints: boolean;
  isLoadingResults: boolean;
  isLoadingChain: boolean;
  error: string | null;
  triggerAnalysis: () => Promise<void>;
  loadChainDetail: (analysisId: string) => Promise<void>;
  clearSelectedChain: () => void;
  refresh: () => Promise<void>;
}

export function useUnderstanding(
  projectId: string,
  asgStatus: string,
): UseUnderstandingReturn {
  const [entryPoints, setEntryPoints] = useState<EntryPoint[]>([]);
  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([]);
  const [coveragePct, setCoveragePct] = useState(0);
  const [jobStatus, setJobStatus] = useState<AnalysisJobStatus | null>(null);
  const [selectedChain, setSelectedChain] = useState<ChainDetail | null>(null);

  const [isLoadingEntryPoints, setIsLoadingEntryPoints] = useState(false);
  const [isLoadingResults, setIsLoadingResults] = useState(false);
  const [isLoadingChain, setIsLoadingChain] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const jobIdRef = useRef<string | null>(null);

  // Fetch entry points + results
  const fetchData = useCallback(async () => {
    if (asgStatus !== 'complete') return;

    setIsLoadingEntryPoints(true);
    setIsLoadingResults(true);
    setError(null);

    try {
      const [epRes, arRes] = await Promise.all([
        api.getEntryPoints(projectId).catch(() => ({ entry_points: [] as EntryPoint[], count: 0 })),
        api.getAnalysisResults(projectId).catch(() => ({ analyses: [] as AnalysisSummary[], count: 0, coverage_pct: 0 })),
      ]);
      setEntryPoints(epRes.entry_points);
      setAnalyses(arRes.analyses);
      setCoveragePct(arRes.coverage_pct);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load understanding data');
    } finally {
      setIsLoadingEntryPoints(false);
      setIsLoadingResults(false);
    }
  }, [projectId, asgStatus]);

  // Initial load
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Polling logic
  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (jobId: string) => {
      stopPolling();
      jobIdRef.current = jobId;

      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getAnalysisStatus(projectId, jobId);
          setJobStatus(status);

          if (status.status === 'completed' || status.status === 'failed') {
            stopPolling();
            // Refresh data on completion
            if (status.status === 'completed') {
              await fetchData();
            }
          }
        } catch {
          // Polling error — keep trying
        }
      }, POLL_INTERVAL_MS);
    },
    [projectId, stopPolling, fetchData],
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const triggerAnalysis = useCallback(async () => {
    setError(null);
    try {
      const res = await api.startAnalysis(projectId);
      const initialStatus: AnalysisJobStatus = {
        job_id: res.job_id,
        project_id: res.project_id,
        status: res.status as AnalysisJobStatus['status'],
        progress: { total: 0, completed: 0 },
        created_at: null,
        started_at: null,
        completed_at: null,
        retry_count: 0,
        errors: null,
      };
      setJobStatus(initialStatus);
      startPolling(res.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start analysis');
    }
  }, [projectId, startPolling]);

  const loadChainDetail = useCallback(
    async (analysisId: string) => {
      setIsLoadingChain(true);
      setError(null);
      try {
        const chain = await api.getChainDetail(projectId, analysisId);
        setSelectedChain(chain);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load chain detail');
      } finally {
        setIsLoadingChain(false);
      }
    },
    [projectId],
  );

  const clearSelectedChain = useCallback(() => {
    setSelectedChain(null);
  }, []);

  const refresh = useCallback(async () => {
    await fetchData();
  }, [fetchData]);

  return {
    entryPoints,
    analyses,
    coveragePct,
    jobStatus,
    selectedChain,
    isLoadingEntryPoints,
    isLoadingResults,
    isLoadingChain,
    error,
    triggerAnalysis,
    loadChainDetail,
    clearSelectedChain,
    refresh,
  };
}
