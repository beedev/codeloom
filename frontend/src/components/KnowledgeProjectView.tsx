/**
 * KnowledgeProjectView
 *
 * Two-panel layout for knowledge projects (document collections + RAG chat).
 * Left panel (30%): document list + upload zone.
 * Right panel (70%): chat interface with query settings bar.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  Upload,
  FileText,
  File,
  Loader2,
  AlertCircle,
  Send,
  MessageSquare,
  ArrowLeft,
  CheckCircle2,
  XCircle,
  BookOpen,
  Settings2,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import * as api from '../services/api.ts';
import type { Project, CodeFile, ModelProvider } from '../types/index.ts';

// ---------------------------------------------------------------------------
// File type styling
// ---------------------------------------------------------------------------

const FILE_TYPE_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  pdf:  { bg: 'bg-red-500/20',   text: 'text-red-400',   label: 'PDF' },
  docx: { bg: 'bg-blue-500/20',  text: 'text-blue-400',  label: 'DOCX' },
  doc:  { bg: 'bg-blue-500/20',  text: 'text-blue-400',  label: 'DOC' },
  txt:  { bg: 'bg-gray-500/20',  text: 'text-gray-400',  label: 'TXT' },
  md:   { bg: 'bg-green-500/20', text: 'text-green-400', label: 'MD' },
  epub: { bg: 'bg-amber-500/20', text: 'text-amber-400', label: 'EPUB' },
  pptx: { bg: 'bg-orange-500/20', text: 'text-orange-400', label: 'PPTX' },
  ppt:  { bg: 'bg-orange-500/20', text: 'text-orange-400', label: 'PPT' },
};

function getFileExt(path: string): string {
  const parts = path.split('.');
  return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : '';
}

function getFileStyle(path: string) {
  const ext = getFileExt(path);
  return FILE_TYPE_STYLES[ext] ?? { bg: 'bg-void-surface/50', text: 'text-text-dim', label: ext.toUpperCase() || 'FILE' };
}

// ---------------------------------------------------------------------------
// Chat message type
// ---------------------------------------------------------------------------

interface ChatMsg {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{ filename: string; snippet: string; score: number }>;
}

// ---------------------------------------------------------------------------
// Query settings types
// ---------------------------------------------------------------------------

interface QuerySettings {
  model: string;
  reranker: string;  // '' = disabled, 'base' = default, 'xsmall', 'large', 'groq:scout', etc.
  responseFormat: 'default' | 'analytical' | 'detailed' | 'brief';
  topK: number;
}

const DEFAULT_QUERY_SETTINGS: QuerySettings = {
  model: '',  // empty = use server default
  reranker: 'base',  // default reranker
  responseFormat: 'default',
  topK: 6,
};

const RERANKER_OPTIONS = [
  { value: '', label: 'Disabled', description: 'No reranking' },
  { value: 'xsmall', label: 'XSmall', description: 'Fastest (~22s CPU)' },
  { value: 'base', label: 'Base', description: 'Balanced (~30s CPU)' },
  { value: 'large', label: 'Large', description: 'Best local (~60s CPU)' },
  { value: 'groq:scout', label: 'Llama 4 Scout', description: 'Fast (~300ms)' },
  { value: 'groq:maverick', label: 'Llama 4 Maverick', description: 'Better (~400ms)' },
  { value: 'groq:llama70b', label: 'Llama 3.3 70B', description: 'Highest quality (~600ms)' },
] as const;

const RESPONSE_FORMAT_OPTIONS = [
  { value: 'default', label: 'Default' },
  { value: 'analytical', label: 'Analytical' },
  { value: 'detailed', label: 'Detailed' },
  { value: 'brief', label: 'Brief' },
] as const;

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface KnowledgeProjectViewProps {
  projectId: string;
  project: Project;
}

export function KnowledgeProjectView({ projectId, project }: KnowledgeProjectViewProps) {
  // Document list
  const [documents, setDocuments] = useState<CodeFile[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(true);

  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Chat state
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Query settings state
  const [querySettings, setQuerySettings] = useState<QuerySettings>(DEFAULT_QUERY_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [providers, setProviders] = useState<ModelProvider[]>([]);
  const [currentModel, setCurrentModel] = useState('');
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  // Load documents
  const loadDocuments = useCallback(async () => {
    setIsLoadingDocs(true);
    try {
      const result = await api.getProjectFiles(projectId);
      const files = Array.isArray(result) ? result : (result?.files ?? []);
      setDocuments(files);
    } catch {
      // Silently handle -- project may have no files yet
      setDocuments([]);
    } finally {
      setIsLoadingDocs(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  // Load available models
  useEffect(() => {
    setIsLoadingModels(true);
    api.getModels()
      .then((data) => {
        setProviders(data.providers);
        setCurrentModel(data.current.model);
      })
      .catch(() => {
        // Non-admin or models endpoint unavailable -- leave empty
      })
      .finally(() => setIsLoadingModels(false));
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Clear transient upload feedback after 4s
  useEffect(() => {
    if (uploadSuccess) {
      const timer = setTimeout(() => setUploadSuccess(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [uploadSuccess]);

  // Upload handler
  const handleUpload = useCallback(async (file: File) => {
    setUploadError(null);
    setUploadSuccess(null);
    setIsUploading(true);
    try {
      await api.uploadDocument(projectId, file);
      setUploadSuccess(`Uploaded "${file.name}" successfully.`);
      loadDocuments();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }, [projectId, loadDocuments]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
    // Reset so same file can be re-selected
    e.target.value = '';
  }, [handleUpload]);

  // Drag-and-drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  }, [handleUpload]);

  // Chat send
  const handleSend = useCallback(async () => {
    const query = input.trim();
    if (!query || isSending) return;

    setChatError(null);
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: query }]);
    setIsSending(true);

    try {
      const options: api.ChatQueryOptions = {};
      if (querySettings.model) options.model = querySettings.model;
      options.reranker_enabled = querySettings.reranker !== '';
      if (querySettings.reranker) options.reranker_model = querySettings.reranker;
      if (querySettings.responseFormat !== 'default') {
        options.response_format = querySettings.responseFormat;
      }
      options.top_k = querySettings.topK;

      const result = await api.chatWithProject(projectId, query, true, options);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: result.response, sources: result.sources },
      ]);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : 'Chat request failed');
    } finally {
      setIsSending(false);
    }
  }, [input, isSending, projectId, querySettings]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  // Flatten all models from providers for the selector
  const allModels = providers.flatMap((p) =>
    p.models.map((m) => ({
      provider: p.name,
      name: m.name,
      display: m.display_name || m.name,
    })),
  );

  const hasCustomSettings =
    querySettings.model !== DEFAULT_QUERY_SETTINGS.model ||
    querySettings.reranker !== DEFAULT_QUERY_SETTINGS.reranker ||
    querySettings.responseFormat !== DEFAULT_QUERY_SETTINGS.responseFormat ||
    querySettings.topK !== DEFAULT_QUERY_SETTINGS.topK;

  const ACCEPTED_FORMATS = '.pdf,.docx,.doc,.txt,.md,.epub,.pptx,.ppt';

  return (
    <div className="flex h-full flex-col">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-void-surface px-4 py-3">
        <div className="flex items-center gap-3">
          <Link
            to="/"
            className="rounded p-1 text-text-muted hover:bg-void-surface hover:text-text"
            aria-label="Back to dashboard"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <BookOpen className="h-4 w-4 text-indigo-400" />
          <h1 className="text-sm font-medium text-text">{project.name}</h1>
          <span className="rounded border border-indigo-500/30 bg-indigo-500/20 px-2 py-0.5 text-[10px] font-medium text-indigo-400">
            Knowledge Base
          </span>
        </div>
      </div>

      {/* Two-panel layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* -- Left panel: Documents -- */}
        <div className="flex w-[30%] shrink-0 flex-col border-r border-void-surface">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-void-surface/50 px-4 py-3">
            <div className="flex items-center gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-text-dim">
                Documents
              </h2>
              <span className="rounded-full bg-void-surface/50 px-2 py-0.5 text-[10px] font-medium text-text-muted">
                {documents.length}
              </span>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="flex items-center gap-1.5 rounded-md bg-glow px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-glow-dim disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isUploading ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Upload className="h-3.5 w-3.5" />
              )}
              Upload
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_FORMATS}
              onChange={handleFileInputChange}
              className="hidden"
            />
          </div>

          {/* Upload feedback */}
          {uploadError && (
            <div className="mx-4 mt-3 flex items-center gap-2 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
              <XCircle className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate">{uploadError}</span>
            </div>
          )}
          {uploadSuccess && (
            <div className="mx-4 mt-3 flex items-center gap-2 rounded-lg border border-success/30 bg-success/10 px-3 py-2 text-xs text-success">
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate">{uploadSuccess}</span>
            </div>
          )}

          {/* Drag-and-drop zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`mx-4 mt-3 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-5 transition-colors ${
              isDragOver
                ? 'border-glow bg-glow/10'
                : 'border-void-surface/50 hover:border-void-surface hover:bg-void-light/30'
            }`}
          >
            <Upload className={`h-5 w-5 ${isDragOver ? 'text-glow' : 'text-text-dim'}`} />
            <p className="mt-2 text-xs text-text-muted">
              {isDragOver ? 'Drop file here' : 'Drag & drop or click to upload'}
            </p>
            <p className="mt-1 text-[10px] text-text-dim">
              PDF, DOCX, TXT, MD, EPUB, PPTX
            </p>
          </div>

          {/* Document list */}
          <div className="flex-1 overflow-y-auto px-4 py-3">
            {isLoadingDocs ? (
              <div className="flex items-center justify-center py-10">
                <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
              </div>
            ) : documents.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-10 text-center">
                <File className="h-8 w-8 text-void-surface" />
                <p className="mt-3 text-xs text-text-dim">No documents yet.</p>
                <p className="mt-1 text-[10px] text-text-dim">
                  Upload documents to build your knowledge base.
                </p>
              </div>
            ) : (
              <div className="space-y-1.5">
                {documents.map((doc) => {
                  const style = getFileStyle(doc.file_path);
                  const fileName = doc.file_path.split('/').pop() ?? doc.file_path;
                  return (
                    <div
                      key={doc.file_id}
                      className="flex items-center gap-3 rounded-lg border border-void-surface/50 bg-void-light/20 px-3 py-2.5 transition-colors hover:border-void-surface hover:bg-void-light/40"
                    >
                      <span className={`shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold uppercase ${style.bg} ${style.text}`}>
                        {style.label}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-xs font-medium text-text-muted">{fileName}</p>
                        {doc.created_at && (
                          <p className="mt-0.5 text-[10px] text-text-dim">
                            {new Date(doc.created_at).toLocaleDateString()}
                          </p>
                        )}
                      </div>
                      <FileText className="h-3.5 w-3.5 shrink-0 text-text-dim/50" />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* -- Right panel: Chat -- */}
        <div className="flex flex-1 flex-col">
          {/* Chat header */}
          <div className="flex items-center gap-2 border-b border-void-surface/50 px-4 py-3">
            <MessageSquare className="h-4 w-4 text-text-dim" />
            <h2 className="text-xs font-semibold uppercase tracking-wider text-text-dim">
              Chat
            </h2>
          </div>

          {/* Messages area */}
          <div className="flex-1 overflow-y-auto px-4 py-4">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center text-center">
                <MessageSquare className="h-10 w-10 text-void-surface" />
                <p className="mt-4 text-sm text-text-muted">
                  Ask questions about your documents
                </p>
                <p className="mt-1 text-xs text-text-dim">
                  Upload documents and start chatting to get AI-powered answers with source references.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div
                      className={`max-w-[80%] rounded-xl px-4 py-3 text-sm ${
                        msg.role === 'user'
                          ? 'bg-glow/20 text-text'
                          : 'border border-void-surface/50 bg-void-light/30 text-text-muted'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{msg.content}</p>

                      {/* Source references */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-3 space-y-1.5 border-t border-void-surface/30 pt-2">
                          <p className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
                            Sources
                          </p>
                          {msg.sources.map((src, srcIdx) => (
                            <div
                              key={srcIdx}
                              className="rounded-md bg-void-surface/30 px-2.5 py-2 text-[11px]"
                            >
                              <p className="font-medium text-text-muted">{src.filename}</p>
                              {src.snippet && (
                                <p className="mt-1 line-clamp-3 text-text-dim">{src.snippet}</p>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {/* Typing indicator */}
                {isSending && (
                  <div className="flex justify-start">
                    <div className="flex items-center gap-2 rounded-xl border border-void-surface/50 bg-void-light/30 px-4 py-3">
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-text-dim" />
                      <span className="text-xs text-text-dim">Thinking...</span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Chat error */}
          {chatError && (
            <div className="mx-4 mb-2 flex items-center gap-2 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
              <AlertCircle className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate">{chatError}</span>
            </div>
          )}

          {/* Query settings bar */}
          <div className="border-t border-void-surface/30 px-4 py-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`flex items-center gap-1.5 text-[11px] font-medium transition-colors ${
                hasCustomSettings ? 'text-glow' : 'text-text-dim hover:text-text-muted'
              }`}
            >
              <Settings2 className="h-3 w-3" />
              <span>Query Settings</span>
              {hasCustomSettings && (
                <span className="rounded bg-glow/20 px-1 py-0.5 text-[9px] text-glow">
                  Custom
                </span>
              )}
              {showSettings ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>

            {showSettings && (
              <div className="mt-2 grid grid-cols-2 gap-3 rounded-lg border border-void-surface bg-void-light/50 p-3">
                {/* Model selector */}
                <div>
                  <label className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-text-dim">
                    Model
                  </label>
                  {isLoadingModels ? (
                    <div className="flex h-8 items-center">
                      <Loader2 className="h-3 w-3 animate-spin text-text-dim" />
                    </div>
                  ) : (
                    <select
                      value={querySettings.model}
                      onChange={(e) =>
                        setQuerySettings((s) => ({ ...s, model: e.target.value }))
                      }
                      className="h-8 w-full rounded border border-void-surface bg-void-light px-2 text-xs text-text focus:border-glow/50 focus:outline-none"
                    >
                      <option value="">
                        Server default{currentModel ? ` (${currentModel})` : ''}
                      </option>
                      {allModels.map((m) => (
                        <option key={`${m.provider}/${m.name}`} value={m.name}>
                          {m.provider}: {m.display}
                        </option>
                      ))}
                    </select>
                  )}
                </div>

                {/* Response format */}
                <div>
                  <label className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-text-dim">
                    Response Format
                  </label>
                  <div className="flex h-8 gap-0.5 rounded border border-void-surface bg-void-surface/30 p-0.5">
                    {RESPONSE_FORMAT_OPTIONS.map((opt) => (
                      <button
                        key={opt.value}
                        onClick={() =>
                          setQuerySettings((s) => ({
                            ...s,
                            responseFormat: opt.value as QuerySettings['responseFormat'],
                          }))
                        }
                        className={`flex-1 rounded text-[10px] font-medium transition-all ${
                          querySettings.responseFormat === opt.value
                            ? 'bg-glow text-void shadow-sm'
                            : 'text-text-dim hover:text-text-muted'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Reranker selector */}
                <div>
                  <label className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-text-dim">
                    Reranker
                  </label>
                  <select
                    value={querySettings.reranker}
                    onChange={(e) =>
                      setQuerySettings((s) => ({ ...s, reranker: e.target.value }))
                    }
                    className="h-8 w-full rounded border border-void-surface bg-void-light px-2 text-xs text-text focus:border-glow/50 focus:outline-none"
                  >
                    {RERANKER_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label} — {opt.description}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Top-K slider */}
                <div>
                  <label className="mb-1 flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-text-dim">
                    <span>Results (Top-K)</span>
                    <span className="normal-case tracking-normal text-text-muted">
                      {querySettings.topK}
                    </span>
                  </label>
                  <input
                    type="range"
                    min={3}
                    max={10}
                    value={querySettings.topK}
                    onChange={(e) =>
                      setQuerySettings((s) => ({
                        ...s,
                        topK: Number(e.target.value),
                      }))
                    }
                    className="mt-1 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-void-surface accent-glow"
                  />
                  <div className="mt-0.5 flex justify-between text-[9px] text-text-dim">
                    <span>3</span>
                    <span>10</span>
                  </div>
                </div>

                {/* Reset */}
                {hasCustomSettings && (
                  <div className="col-span-2">
                    <button
                      onClick={() => setQuerySettings(DEFAULT_QUERY_SETTINGS)}
                      className="w-full rounded border border-void-surface py-1.5 text-[10px] text-text-dim transition-colors hover:border-text-dim hover:text-text-muted"
                    >
                      Reset to Defaults
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Input area */}
          <div className="border-t border-void-surface px-4 py-3">
            <div className="flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  documents.length === 0
                    ? 'Upload documents first to start chatting...'
                    : 'Ask a question about your documents...'
                }
                disabled={isSending}
                rows={1}
                className="flex-1 resize-none rounded-lg border border-void-surface bg-void-light px-4 py-2.5 text-sm text-text placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50 disabled:opacity-50"
                style={{ minHeight: '40px', maxHeight: '120px' }}
                onInput={(e) => {
                  const el = e.target as HTMLTextAreaElement;
                  el.style.height = 'auto';
                  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
                }}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isSending}
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-glow text-white transition-colors hover:bg-glow-dim disabled:cursor-not-allowed disabled:opacity-50"
                aria-label="Send message"
              >
                {isSending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
