/**
 * useWikiData — central data-fetching hook for the Project Wiki.
 *
 * Manages lazy loading and caching per section.
 * Each load* function fetches only when data isn't already cached.
 */

import { useState, useCallback, useRef } from 'react';
import * as api from '../services/api.ts';
import type {
  ProjectAnalytics,
  GraphOverview,
  EntryPoint,
  AnalysisSummary,
  ChainDetail,
  MigrationPlan,
  MigrationPhaseOutput,
  MvpDetail,
  DiagramAvailability,
  DiagramResponse,
} from '../types/index.ts';

export interface WikiData {
  analytics: ProjectAnalytics | null;
  graphOverview: GraphOverview | null;
  entryPoints: EntryPoint[] | null;
  analysisResults: { analyses: AnalysisSummary[]; count: number; coverage_pct: number } | null;
  migrationPlan: MigrationPlan | null;
  loading: Record<string, boolean>;
  errors: Record<string, string | null>;

  loadOverview: () => Promise<void>;
  loadArchitecture: () => Promise<void>;
  loadUnderstanding: () => Promise<void>;
  loadMigration: () => Promise<void>;
  loadChainDetail: (analysisId: string) => Promise<ChainDetail | null>;
  loadMvpDetail: (mvpId: number) => Promise<MvpDetail | null>;
  loadPhaseOutput: (phaseNumber: number, mvpId?: number) => Promise<MigrationPhaseOutput | null>;
  loadDiagramAvailability: (mvpId: number) => Promise<DiagramAvailability[] | null>;
  loadDiagram: (mvpId: number, diagramType: string) => Promise<DiagramResponse | null>;
}

export function useWikiData(projectId: string): WikiData {
  const [analytics, setAnalytics] = useState<ProjectAnalytics | null>(null);
  const [graphOverview, setGraphOverview] = useState<GraphOverview | null>(null);
  const [entryPoints, setEntryPoints] = useState<EntryPoint[] | null>(null);
  const [analysisResults, setAnalysisResults] = useState<WikiData['analysisResults']>(null);
  const [migrationPlan, setMigrationPlan] = useState<MigrationPlan | null>(null);
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string | null>>({});

  // Caches for on-demand lookups
  const chainCache = useRef<Map<string, ChainDetail>>(new Map());
  const mvpCache = useRef<Map<number, MvpDetail>>(new Map());
  const phaseCache = useRef<Map<string, MigrationPhaseOutput>>(new Map());
  const diagAvailCache = useRef<Map<number, DiagramAvailability[]>>(new Map());
  const diagCache = useRef<Map<string, DiagramResponse>>(new Map());
  const loaded = useRef<Set<string>>(new Set());

  const setLoad = useCallback((key: string, val: boolean) => {
    setLoading(prev => ({ ...prev, [key]: val }));
  }, []);
  const setErr = useCallback((key: string, val: string | null) => {
    setErrors(prev => ({ ...prev, [key]: val }));
  }, []);

  const loadOverview = useCallback(async () => {
    if (loaded.current.has('overview')) return;
    setLoad('overview', true);
    setErr('overview', null);
    try {
      const data = await api.getProjectAnalytics(projectId);
      setAnalytics(data);
      loaded.current.add('overview');
    } catch (e: unknown) {
      setErr('overview', e instanceof Error ? e.message : 'Failed to load analytics');
    } finally {
      setLoad('overview', false);
    }
  }, [projectId, setLoad, setErr]);

  const loadArchitecture = useCallback(async () => {
    if (loaded.current.has('architecture')) return;
    setLoad('architecture', true);
    setErr('architecture', null);
    try {
      const [ep, graph] = await Promise.all([
        api.getEntryPoints(projectId),
        api.getGraphOverview(projectId),
      ]);
      setEntryPoints(ep.entry_points);
      setGraphOverview(graph);
      loaded.current.add('architecture');
    } catch (e: unknown) {
      setErr('architecture', e instanceof Error ? e.message : 'Failed to load architecture data');
    } finally {
      setLoad('architecture', false);
    }
  }, [projectId, setLoad, setErr]);

  const loadUnderstanding = useCallback(async () => {
    if (loaded.current.has('understanding')) return;
    setLoad('understanding', true);
    setErr('understanding', null);
    try {
      const data = await api.getAnalysisResults(projectId);
      setAnalysisResults(data);
      loaded.current.add('understanding');
    } catch (e: unknown) {
      setErr('understanding', e instanceof Error ? e.message : 'Failed to load analysis results');
    } finally {
      setLoad('understanding', false);
    }
  }, [projectId, setLoad, setErr]);

  const loadMigration = useCallback(async () => {
    if (loaded.current.has('migration')) return;
    setLoad('migration', true);
    setErr('migration', null);
    try {
      const plans = await api.listMigrationPlans(projectId);
      if (plans.length > 0) {
        // Prefer in_progress, then draft, then first
        const active = plans.find(p => p.status === 'in_progress')
          ?? plans.find(p => p.status === 'draft')
          ?? plans[0];
        const full = await api.getMigrationPlan(active.plan_id);
        setMigrationPlan(full);
      }
      loaded.current.add('migration');
    } catch (e: unknown) {
      setErr('migration', e instanceof Error ? e.message : 'Failed to load migration data');
    } finally {
      setLoad('migration', false);
    }
  }, [projectId, setLoad, setErr]);

  const loadChainDetail = useCallback(async (analysisId: string): Promise<ChainDetail | null> => {
    if (chainCache.current.has(analysisId)) return chainCache.current.get(analysisId)!;
    try {
      const detail = await api.getChainDetail(projectId, analysisId);
      chainCache.current.set(analysisId, detail);
      return detail;
    } catch {
      return null;
    }
  }, [projectId]);

  const loadMvpDetail = useCallback(async (mvpId: number): Promise<MvpDetail | null> => {
    if (!migrationPlan) return null;
    if (mvpCache.current.has(mvpId)) return mvpCache.current.get(mvpId)!;
    try {
      const detail = await api.getMvpDetail(migrationPlan.plan_id, mvpId);
      mvpCache.current.set(mvpId, detail);
      return detail;
    } catch {
      return null;
    }
  }, [migrationPlan]);

  const loadPhaseOutput = useCallback(async (phaseNumber: number, mvpId?: number): Promise<MigrationPhaseOutput | null> => {
    if (!migrationPlan) return null;
    const key = `${phaseNumber}-${mvpId ?? 'plan'}`;
    if (phaseCache.current.has(key)) return phaseCache.current.get(key)!;
    try {
      const output = await api.getMigrationPhaseOutput(migrationPlan.plan_id, phaseNumber, mvpId);
      phaseCache.current.set(key, output);
      return output;
    } catch {
      return null;
    }
  }, [migrationPlan]);

  const loadDiagramAvailability = useCallback(async (mvpId: number): Promise<DiagramAvailability[] | null> => {
    if (!migrationPlan) return null;
    if (diagAvailCache.current.has(mvpId)) return diagAvailCache.current.get(mvpId)!;
    try {
      const avail = await api.listMvpDiagrams(migrationPlan.plan_id, mvpId);
      diagAvailCache.current.set(mvpId, avail);
      return avail;
    } catch {
      return null;
    }
  }, [migrationPlan]);

  const loadDiagram = useCallback(async (mvpId: number, diagramType: string): Promise<DiagramResponse | null> => {
    if (!migrationPlan) return null;
    const key = `${mvpId}-${diagramType}`;
    if (diagCache.current.has(key)) return diagCache.current.get(key)!;
    try {
      const diagram = await api.getMvpDiagram(migrationPlan.plan_id, mvpId, diagramType);
      diagCache.current.set(key, diagram);
      return diagram;
    } catch {
      return null;
    }
  }, [migrationPlan]);

  return {
    analytics, graphOverview, entryPoints, analysisResults, migrationPlan,
    loading, errors,
    loadOverview, loadArchitecture, loadUnderstanding, loadMigration,
    loadChainDetail, loadMvpDetail, loadPhaseOutput, loadDiagramAvailability, loadDiagram,
  };
}
