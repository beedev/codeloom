/**
 * useSettings Hook
 *
 * Manages LLM model, reranker, and response-type settings.
 * Fetches current config on mount and provides mutators.
 */

import { useState, useEffect, useCallback } from 'react';
import * as api from '../services/api.ts';
import type { ModelProvider, RerankerConfig, RerankerOption } from '../types/index.ts';

export type ResponseType = 'detailed' | 'concise';

interface UseSettingsReturn {
  // Models
  providers: ModelProvider[];
  currentProvider: string;
  currentModel: string;
  isLoadingModels: boolean;
  updateModel: (provider: string, model: string) => Promise<void>;

  // Reranker
  rerankerConfig: RerankerConfig | null;
  rerankerOptions: RerankerOption[];
  isLoadingReranker: boolean;
  updateReranker: (model: string) => Promise<void>;
  disableReranker: () => Promise<void>;

  // Response type (local-only, sent with each chat request)
  responseType: ResponseType;
  setResponseType: (type: ResponseType) => void;
}

export function useSettings(): UseSettingsReturn {
  // Model state
  const [providers, setProviders] = useState<ModelProvider[]>([]);
  const [currentProvider, setCurrentProvider] = useState('');
  const [currentModel, setCurrentModel] = useState('');
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  // Reranker state
  const [rerankerConfig, setRerankerConfig] = useState<RerankerConfig | null>(null);
  const [rerankerOptions, setRerankerOptions] = useState<RerankerOption[]>([]);
  const [isLoadingReranker, setIsLoadingReranker] = useState(true);

  // Response type (persisted in localStorage)
  const [responseType, setResponseTypeState] = useState<ResponseType>(() => {
    return (localStorage.getItem('codeloom_response_type') as ResponseType) || 'detailed';
  });

  const setResponseType = useCallback((type: ResponseType) => {
    setResponseTypeState(type);
    localStorage.setItem('codeloom_response_type', type);
  }, []);

  // Fetch models on mount
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await api.getModels();
        if (cancelled) return;
        setProviders(data.providers);
        setCurrentProvider(data.current.provider);
        setCurrentModel(data.current.model);
      } catch {
        // Settings are non-critical -- silently degrade
      } finally {
        if (!cancelled) setIsLoadingModels(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, []);

  // Fetch reranker on mount
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await api.getReranker();
        if (cancelled) return;
        setRerankerConfig(data.config);
        setRerankerOptions(data.available_models);
      } catch {
        // Non-critical
      } finally {
        if (!cancelled) setIsLoadingReranker(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, []);

  // Mutators
  const updateModel = useCallback(async (provider: string, model: string) => {
    await api.setModel(provider, model);
    setCurrentProvider(provider);
    setCurrentModel(model);
  }, []);

  const updateReranker = useCallback(async (model: string) => {
    await api.setReranker(true, model);
    const data = await api.getReranker();
    setRerankerConfig(data.config);
  }, []);

  const disableReranker = useCallback(async () => {
    await api.setReranker(false, 'disabled');
    const data = await api.getReranker();
    setRerankerConfig(data.config);
  }, []);

  return {
    providers,
    currentProvider,
    currentModel,
    isLoadingModels,
    updateModel,
    rerankerConfig,
    rerankerOptions,
    isLoadingReranker,
    updateReranker,
    disableReranker,
    responseType,
    setResponseType,
  };
}
