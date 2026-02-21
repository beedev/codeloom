/**
 * SettingsPanel Component
 *
 * Collapsible panel for LLM model, reranker, and response type settings.
 * Intended to sit in the chat page top bar area.
 */

import { useState } from 'react';
import { Loader2 } from 'lucide-react';
import { useSettings } from '../hooks/useSettings.ts';

export function SettingsPanel() {
  const {
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
  } = useSettings();

  const [isSavingModel, setIsSavingModel] = useState(false);
  const [isSavingReranker, setIsSavingReranker] = useState(false);

  // ---- Model change handler ----

  async function handleModelChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const value = e.target.value; // "provider::model"
    const [provider, model] = value.split('::');
    if (!provider || !model) return;

    setIsSavingModel(true);
    try {
      await updateModel(provider, model);
    } catch {
      // TODO: show error toast
    } finally {
      setIsSavingModel(false);
    }
  }

  // ---- Reranker change handler ----

  async function handleRerankerChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const value = e.target.value;
    setIsSavingReranker(true);
    try {
      if (value === 'disabled') {
        await disableReranker();
      } else {
        await updateReranker(value);
      }
    } catch {
      // TODO: show error toast
    } finally {
      setIsSavingReranker(false);
    }
  }

  const selectedModelValue = `${currentProvider}::${currentModel}`;
  const selectedRerankerValue = rerankerConfig?.enabled
    ? rerankerConfig.model
    : 'disabled';

  return (
    <div className="border-b border-void-surface bg-void/50 px-4 py-3">
      <div className="flex flex-wrap items-center gap-4">
        {/* LLM Model */}
        <div className="flex items-center gap-2">
          <label className="text-[10px] font-medium uppercase tracking-wider text-text-dim">
            Model
          </label>
          {isLoadingModels ? (
            <Loader2 className="h-3 w-3 animate-spin text-text-dim" />
          ) : (
            <select
              value={selectedModelValue}
              onChange={handleModelChange}
              disabled={isSavingModel}
              className="rounded border border-void-surface bg-void-light px-2 py-1 text-xs text-text-dim focus:border-glow/50 focus:outline-none disabled:opacity-50"
            >
              {providers.map((p) => (
                <optgroup key={p.name} label={p.name.charAt(0).toUpperCase() + p.name.slice(1)}>
                  {p.models.map((m) => (
                    <option key={`${p.name}::${m.name}`} value={`${p.name}::${m.name}`}>
                      {m.display_name}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          )}
          {isSavingModel && (
            <Loader2 className="h-3 w-3 animate-spin text-glow" />
          )}
        </div>

        {/* Reranker */}
        <div className="flex items-center gap-2">
          <label className="text-[10px] font-medium uppercase tracking-wider text-text-dim">
            Reranker
          </label>
          {isLoadingReranker ? (
            <Loader2 className="h-3 w-3 animate-spin text-text-dim" />
          ) : (
            <select
              value={selectedRerankerValue}
              onChange={handleRerankerChange}
              disabled={isSavingReranker}
              className="rounded border border-void-surface bg-void-light px-2 py-1 text-xs text-text-dim focus:border-glow/50 focus:outline-none disabled:opacity-50"
            >
              {rerankerOptions.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.name} {opt.description ? `- ${opt.description}` : ''}
                </option>
              ))}
            </select>
          )}
          {isSavingReranker && (
            <Loader2 className="h-3 w-3 animate-spin text-glow" />
          )}
        </div>

        {/* Response Type Toggle */}
        <div className="flex items-center gap-2">
          <label className="text-[10px] font-medium uppercase tracking-wider text-text-dim">
            Response
          </label>
          <div className="flex rounded border border-void-surface bg-void-light text-xs">
            <button
              onClick={() => setResponseType('detailed')}
              className={`px-2.5 py-1 transition-colors ${
                responseType === 'detailed'
                  ? 'bg-glow text-white'
                  : 'text-text-muted hover:text-text-dim'
              }`}
            >
              Detailed
            </button>
            <button
              onClick={() => setResponseType('concise')}
              className={`px-2.5 py-1 transition-colors ${
                responseType === 'concise'
                  ? 'bg-glow text-white'
                  : 'text-text-muted hover:text-text-dim'
              }`}
            >
              Concise
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
