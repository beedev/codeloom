/**
 * Settings page — editable LLM provider, model, and reranker configuration.
 *
 * Fetches available providers/models and current config on mount.
 * Changes are persisted to the backend immediately via POST endpoints.
 */

import { useState, useEffect, useMemo } from 'react';
import { Check, Loader2, RotateCcw } from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import * as api from '../services/api.ts';
import type { ModelsResponse, RerankerResponse } from '../services/api.ts';
import type { ModelProvider } from '../types/index.ts';

type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

export function Settings() {
  // ── Data from backend ─────────────────────────────────────────────
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [rerankerData, setRerankerData] = useState<RerankerResponse | null>(null);
  const [loading, setLoading] = useState(true);

  // ── LLM form state ────────────────────────────────────────────────
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [llmStatus, setLlmStatus] = useState<SaveStatus>('idle');
  const [llmError, setLlmError] = useState('');

  // ── Reranker form state ───────────────────────────────────────────
  const [rerankerEnabled, setRerankerEnabled] = useState(true);
  const [rerankerModel, setRerankerModel] = useState('');
  const [rerankerTopN, setRerankerTopN] = useState(10);
  const [rerankerStatus, setRerankerStatus] = useState<SaveStatus>('idle');
  const [rerankerError, setRerankerError] = useState('');

  // ── Load settings ─────────────────────────────────────────────────
  useEffect(() => {
    Promise.all([api.getModels(), api.getReranker()])
      .then(([m, r]) => {
        setModelsData(m);
        setSelectedProvider(m.current.provider);
        setSelectedModel(m.current.model);
        setRerankerData(r);
        setRerankerEnabled(r.config.enabled);
        setRerankerModel(r.config.model);
        setRerankerTopN(r.config.top_n);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  // ── Derived: models for the selected provider ─────────────────────
  const currentProviderObj: ModelProvider | undefined = useMemo(
    () => modelsData?.providers.find((p) => p.name === selectedProvider),
    [modelsData, selectedProvider],
  );

  // When provider changes, reset model to first available
  const handleProviderChange = (providerName: string) => {
    setSelectedProvider(providerName);
    setLlmStatus('idle');
    setLlmError('');
    const prov = modelsData?.providers.find((p) => p.name === providerName);
    if (prov && prov.models.length > 0) {
      setSelectedModel(prov.models[0].name);
    }
  };

  // ── LLM dirty check ──────────────────────────────────────────────
  const llmDirty =
    modelsData != null &&
    (selectedProvider !== modelsData.current.provider ||
      selectedModel !== modelsData.current.model);

  // ── Reranker dirty check ──────────────────────────────────────────
  const rerankerDirty =
    rerankerData != null &&
    (rerankerEnabled !== rerankerData.config.enabled ||
      rerankerModel !== rerankerData.config.model ||
      rerankerTopN !== rerankerData.config.top_n);

  // ── Save LLM ─────────────────────────────────────────────────────
  const saveLlm = async () => {
    setLlmStatus('saving');
    setLlmError('');
    try {
      await api.setModel(selectedProvider, selectedModel);
      // Refresh to get confirmed state
      const fresh = await api.getModels();
      setModelsData(fresh);
      setSelectedProvider(fresh.current.provider);
      setSelectedModel(fresh.current.model);
      setLlmStatus('saved');
      setTimeout(() => setLlmStatus('idle'), 2000);
    } catch (err) {
      setLlmError(err instanceof Error ? err.message : 'Failed to update LLM');
      setLlmStatus('error');
    }
  };

  // ── Save Reranker ─────────────────────────────────────────────────
  const saveReranker = async () => {
    setRerankerStatus('saving');
    setRerankerError('');
    try {
      await api.setReranker(rerankerEnabled, rerankerModel, rerankerTopN);
      const fresh = await api.getReranker();
      setRerankerData(fresh);
      setRerankerEnabled(fresh.config.enabled);
      setRerankerModel(fresh.config.model);
      setRerankerTopN(fresh.config.top_n);
      setRerankerStatus('saved');
      setTimeout(() => setRerankerStatus('idle'), 2000);
    } catch (err) {
      setRerankerError(err instanceof Error ? err.message : 'Failed to update reranker');
      setRerankerStatus('error');
    }
  };

  // ── Reset helpers ─────────────────────────────────────────────────
  const resetLlm = () => {
    if (modelsData) {
      setSelectedProvider(modelsData.current.provider);
      setSelectedModel(modelsData.current.model);
      setLlmStatus('idle');
      setLlmError('');
    }
  };

  const resetReranker = () => {
    if (rerankerData) {
      setRerankerEnabled(rerankerData.config.enabled);
      setRerankerModel(rerankerData.config.model);
      setRerankerTopN(rerankerData.config.top_n);
      setRerankerStatus('idle');
      setRerankerError('');
    }
  };

  return (
    <Layout>
      <div className="flex-1 overflow-y-auto p-6">
        <h1 className="text-lg font-semibold text-text">Settings</h1>
        <p className="mt-1 text-sm text-text-muted">
          CodeLoom configuration and preferences.
        </p>

        {loading ? (
          <div className="mt-8 flex items-center gap-2 text-sm text-text-dim">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-void-surface border-t-glow" />
            Loading configuration...
          </div>
        ) : (
          <div className="mt-6 grid gap-6 max-w-2xl">
            {/* ── LLM Provider ─────────────────────────────────── */}
            <section className="rounded-xl border border-void-surface bg-void-light/30 p-5">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-medium text-text">LLM Provider</h2>
                  <p className="mt-0.5 text-xs text-text-dim">
                    Select the AI model used for code chat and migration.
                  </p>
                </div>
                <StatusBadge status={llmStatus} />
              </div>

              {modelsData && (
                <div className="mt-4 grid gap-4 sm:grid-cols-2">
                  {/* Provider selector */}
                  <FieldGroup label="Provider">
                    <select
                      value={selectedProvider}
                      onChange={(e) => handleProviderChange(e.target.value)}
                      className="select-field"
                    >
                      {modelsData.providers.map((p) => (
                        <option key={p.name} value={p.name}>
                          {p.name}
                        </option>
                      ))}
                    </select>
                  </FieldGroup>

                  {/* Model selector */}
                  <FieldGroup label="Model">
                    <select
                      value={selectedModel}
                      onChange={(e) => {
                        setSelectedModel(e.target.value);
                        setLlmStatus('idle');
                        setLlmError('');
                      }}
                      className="select-field"
                    >
                      {currentProviderObj?.models.map((m) => (
                        <option key={m.name} value={m.name}>
                          {m.display_name}
                        </option>
                      ))}
                    </select>
                  </FieldGroup>
                </div>
              )}

              {llmError && (
                <p className="mt-3 text-xs text-danger">{llmError}</p>
              )}

              {/* Actions */}
              <div className="mt-4 flex items-center gap-2">
                <button
                  onClick={saveLlm}
                  disabled={!llmDirty || llmStatus === 'saving'}
                  className="btn-primary"
                >
                  {llmStatus === 'saving' ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    'Save'
                  )}
                </button>
                {llmDirty && (
                  <button onClick={resetLlm} className="btn-ghost" title="Reset">
                    <RotateCcw className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </section>

            {/* ── Reranker ─────────────────────────────────────── */}
            <section className="rounded-xl border border-void-surface bg-void-light/30 p-5">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-medium text-text">Reranker</h2>
                  <p className="mt-0.5 text-xs text-text-dim">
                    Refines search results by re-scoring retrieved code chunks.
                  </p>
                </div>
                <StatusBadge status={rerankerStatus} />
              </div>

              {rerankerData && (
                <div className="mt-4 grid gap-4">
                  {/* Enable toggle */}
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-muted">Enabled</span>
                    <button
                      onClick={() => {
                        setRerankerEnabled(!rerankerEnabled);
                        setRerankerStatus('idle');
                        setRerankerError('');
                      }}
                      className={`relative h-5 w-9 rounded-full transition-colors ${
                        rerankerEnabled ? 'bg-glow' : 'bg-void-surface'
                      }`}
                      role="switch"
                      aria-checked={rerankerEnabled}
                    >
                      <span
                        className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
                          rerankerEnabled ? 'translate-x-4' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>

                  <div className={`grid gap-4 sm:grid-cols-2 ${!rerankerEnabled ? 'opacity-40 pointer-events-none' : ''}`}>
                    {/* Model selector */}
                    <FieldGroup label="Model">
                      <select
                        value={rerankerModel}
                        onChange={(e) => {
                          setRerankerModel(e.target.value);
                          setRerankerStatus('idle');
                          setRerankerError('');
                        }}
                        className="select-field"
                      >
                        {rerankerData.available_models.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name}
                          </option>
                        ))}
                      </select>
                    </FieldGroup>

                    {/* Top N */}
                    <FieldGroup label="Top N Results">
                      <input
                        type="number"
                        min={1}
                        max={50}
                        value={rerankerTopN}
                        onChange={(e) => {
                          setRerankerTopN(Math.max(1, Math.min(50, parseInt(e.target.value) || 1)));
                          setRerankerStatus('idle');
                          setRerankerError('');
                        }}
                        className="select-field tabular-nums"
                      />
                    </FieldGroup>
                  </div>
                </div>
              )}

              {rerankerError && (
                <p className="mt-3 text-xs text-danger">{rerankerError}</p>
              )}

              {/* Actions */}
              <div className="mt-4 flex items-center gap-2">
                <button
                  onClick={saveReranker}
                  disabled={!rerankerDirty || rerankerStatus === 'saving'}
                  className="btn-primary"
                >
                  {rerankerStatus === 'saving' ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    'Save'
                  )}
                </button>
                {rerankerDirty && (
                  <button onClick={resetReranker} className="btn-ghost" title="Reset">
                    <RotateCcw className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </section>
          </div>
        )}
      </div>
    </Layout>
  );
}


// ── Sub-components ────────────────────────────────────────────────────

function FieldGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs text-text-dim">{label}</span>
      {children}
    </label>
  );
}

function StatusBadge({ status }: { status: SaveStatus }) {
  if (status === 'saved') {
    return (
      <span className="flex items-center gap-1 rounded-full bg-success/10 px-2 py-0.5 text-[10px] font-medium text-success">
        <Check className="h-3 w-3" /> Saved
      </span>
    );
  }
  if (status === 'error') {
    return (
      <span className="rounded-full bg-danger/10 px-2 py-0.5 text-[10px] font-medium text-danger">
        Error
      </span>
    );
  }
  return null;
}
