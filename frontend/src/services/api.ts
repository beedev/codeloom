import type {
  Notebook,
  NotebookListResponse,
  Document,
  DocumentListResponse,
  ModelsResponse,
  ChatRequest,
  UploadResponse,
  ImageGenerateRequest,
  ImageGenerateResponse,
  ApiError,
  WebSearchResponse,
  WebScrapePreviewResponse,
  WebSourceAddResponse,
  StudioGalleryResponse,
  StudioGenerateRequest,
  StudioGenerateResponse,
  StudioGeneratorsResponse,
  GeneratedContent,
} from '../types';

import type { TokenMetricsResponse } from '../types/auth';

import type {
  Quiz,
  QuizPublicInfo,
  QuizResultsResponse,
  CreateQuizRequest,
  StartAttemptResponse,
  SubmitAnswerResponse,
  AttemptStatusResponse,
} from '../types/quiz';

const API_BASE = '/api';

// Generic fetch wrapper with error handling
async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  console.log(`[API] Fetching: ${url}`);

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    console.log(`[API] Response status: ${response.status} for ${url}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error(`[API] Error response:`, errorData);
      const error: ApiError = {
        error: errorData.error || 'Unknown error',
        message: errorData.message || `HTTP ${response.status}`,
        status: response.status,
      };
      throw error;
    }

    const data = await response.json();
    console.log(`[API] Success response for ${url}:`, data);
    return data;
  } catch (err) {
    console.error(`[API] Fetch error for ${url}:`, err);
    throw err;
  }
}

// ============================================
// Notebook API
// ============================================

interface BackendNotebookListResponse {
  success: boolean;
  notebooks: Array<{
    id: string;
    name: string;
    description?: string;
    document_count?: number;
  }>;
  count: number;
}

interface BackendCreateNotebookResponse {
  success: boolean;
  notebook: { id: string; name: string };
  message: string;
}

export async function getNotebooks(): Promise<NotebookListResponse> {
  const response = await fetchApi<BackendNotebookListResponse>('/notebooks');
  return {
    notebooks: response.notebooks.map(n => ({
      id: n.id,
      name: n.name,
      description: n.description,
      documentCount: n.document_count || 0,
    })),
  };
}

export async function createNotebook(
  name: string,
  description?: string
): Promise<Notebook> {
  const response = await fetchApi<BackendCreateNotebookResponse>('/notebooks', {
    method: 'POST',
    body: JSON.stringify({ name, description }),
  });
  return {
    id: response.notebook.id,
    name: response.notebook.name,
    description: description,
    documentCount: 0,
  };
}

export async function getNotebook(id: string): Promise<Notebook> {
  const response = await fetchApi<{ success: boolean; notebook: { id: string; name: string; description?: string; document_count?: number } }>(`/notebooks/${id}`);
  return {
    id: response.notebook.id,
    name: response.notebook.name,
    description: response.notebook.description,
    documentCount: response.notebook.document_count || 0,
  };
}

export async function updateNotebook(
  id: string,
  data: Partial<Notebook>
): Promise<Notebook> {
  const response = await fetchApi<{ success: boolean; notebook: { id: string; name: string; description?: string } }>(`/notebooks/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
  return {
    id: response.notebook.id,
    name: response.notebook.name,
    description: response.notebook.description,
    documentCount: 0,
  };
}

export async function deleteNotebook(id: string): Promise<void> {
  await fetchApi<{ success: boolean }>(`/notebooks/${id}`, {
    method: 'DELETE',
  });
}

// ============================================
// Document API
// ============================================

interface BackendDocumentListResponse {
  success: boolean;
  documents: Array<{
    source_id: string;
    file_name: string;  // Backend uses snake_case
    active?: boolean;
    file_type?: string;
    chunk_count?: number;
    // AI Transformation fields
    dense_summary?: string | null;
    key_insights?: string[] | null;
    transformation_status?: 'pending' | 'processing' | 'completed' | 'failed';
  }>;
  count: number;
}

export async function getDocuments(
  notebookId: string
): Promise<DocumentListResponse> {
  const response = await fetchApi<BackendDocumentListResponse>(`/notebooks/${notebookId}/documents`);
  return {
    documents: response.documents.map(d => ({
      source_id: d.source_id,
      filename: d.file_name,  // Map snake_case to camelCase
      active: d.active !== false, // default to true
      file_type: d.file_type,
      chunk_count: d.chunk_count,
      // AI Transformation fields
      dense_summary: d.dense_summary,
      key_insights: d.key_insights,
      transformation_status: d.transformation_status,
    })),
  };
}

export async function uploadDocument(
  notebookId: string,
  file: File
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('files', file);  // Backend expects 'files' (plural)
  formData.append('notebook_id', notebookId);

  console.log(`[API] Uploading file: ${file.name} to notebook: ${notebookId}`);

  const response = await fetch(`/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Upload failed',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

export async function deleteDocument(
  notebookId: string,
  sourceId: string
): Promise<void> {
  await fetchApi<void>(`/notebooks/${notebookId}/documents/${sourceId}`, {
    method: 'DELETE',
  });
}

interface BackendToggleDocumentResponse {
  success: boolean;
  document: {
    source_id: string;
    file_name: string;
    file_type?: string;
    chunk_count?: number;
    active: boolean;
  };
  message: string;
}

export async function toggleDocumentActive(
  notebookId: string,
  sourceId: string,
  active: boolean
): Promise<Document> {
  const response = await fetchApi<BackendToggleDocumentResponse>(
    `/notebooks/${notebookId}/documents/${sourceId}`,
    {
      method: 'PATCH',
      body: JSON.stringify({ active }),
    }
  );
  // Map backend snake_case to frontend camelCase
  return {
    source_id: response.document.source_id,
    filename: response.document.file_name,
    file_type: response.document.file_type,
    chunk_count: response.document.chunk_count,
    active: response.document.active,
  };
}

// ============================================
// Document Content API (for preview)
// ============================================

export interface DocumentContent {
  content: string;
  chunkCount: number;
  filename: string;
  fileType: string;
}

interface BackendDocumentContentResponse {
  success: boolean;
  content: string;
  chunk_count: number;
  filename: string;
  file_type: string;
}

export async function getDocumentContent(
  notebookId: string,
  sourceId: string
): Promise<DocumentContent> {
  const response = await fetchApi<BackendDocumentContentResponse>(
    `/notebooks/${notebookId}/documents/${sourceId}/content`
  );
  return {
    content: response.content,
    chunkCount: response.chunk_count,
    filename: response.filename,
    fileType: response.file_type,
  };
}

// ============================================
// Model API
// ============================================

interface BackendModel {
  name: string;
  display_name?: string;
  provider: string;
  type: string;
}

interface BackendModelsResponse {
  success: boolean;
  models: BackendModel[];
  count: number;
  default_model?: string;
  default_provider?: string;
}

export async function getModels(): Promise<ModelsResponse> {
  const response = await fetchApi<BackendModelsResponse>('/models');

  // Transform flat array to grouped format for frontend
  const grouped = new Map<string, BackendModel[]>();
  for (const model of response.models) {
    const provider = model.provider.toLowerCase();
    if (!grouped.has(provider)) {
      grouped.set(provider, []);
    }
    grouped.get(provider)!.push(model);
  }

  const models: ModelsResponse['models'] = [];
  for (const [provider, providerModels] of grouped) {
    models.push({
      provider: provider as 'ollama' | 'openai' | 'anthropic' | 'google',
      models: providerModels.map(m => ({
        name: m.name,
        displayName: m.display_name || m.name
      })),
    });
  }

  // Use default from config, or first available model
  const defaultModel = response.default_model || response.models[0]?.name || '';
  const defaultProvider = response.default_provider || response.models[0]?.provider.toLowerCase() || 'ollama';

  return {
    models,
    currentModel: defaultModel,
    currentProvider: defaultProvider as 'ollama' | 'openai' | 'anthropic' | 'google',
  };
}

export async function setModel(
  _model: string,
  _provider?: string
): Promise<{ success: boolean }> {
  // Model selection is handled client-side for now
  // The actual model is passed with each chat request
  return Promise.resolve({ success: true });
}

// ============================================
// Chat API with SSE Streaming
// ============================================

export interface SourceCitation {
  filename: string;
  page?: string | number;
  score?: number;
  snippet?: string;
}

export interface ChatMetadata {
  execution_time_ms?: number;
  timings?: Record<string, number>;
  node_count?: number;
  model?: string;
  retrieval_strategy?: string;
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onComplete: (fullResponse: string, sources?: SourceCitation[], metadata?: ChatMetadata) => void;
  onError: (error: string) => void;
  onImages?: (images: string[]) => void;
}

export async function sendChatMessage(
  request: ChatRequest,
  callbacks: StreamCallbacks
): Promise<void> {
  const response = await fetch('/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      ...request,
      stream: true,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    callbacks.onError(errorData.message || `HTTP ${response.status}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let fullResponse = '';
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);

          if (data === '[DONE]') {
            callbacks.onComplete(fullResponse);
            return;
          }

          try {
            const parsed = JSON.parse(data);

            if (parsed.token) {
              fullResponse += parsed.token;
              callbacks.onToken(parsed.token);
            }

            if (parsed.images && callbacks.onImages) {
              callbacks.onImages(parsed.images);
            }

            // Handle done signal with optional sources and metadata
            if (parsed.done) {
              const sources = parsed.sources || [];
              const metadata = parsed.metadata || undefined;
              callbacks.onComplete(fullResponse, sources, metadata);
              return;
            }

            if (parsed.error) {
              callbacks.onError(parsed.error);
              return;
            }
          } catch {
            // Plain text token
            fullResponse += data;
            callbacks.onToken(data);
          }
        }
      }
    }

    // Handle any remaining buffer
    if (buffer.startsWith('data: ')) {
      const data = buffer.slice(6);
      if (data !== '[DONE]') {
        fullResponse += data;
        callbacks.onToken(data);
      }
    }

    callbacks.onComplete(fullResponse);
  } catch (error) {
    callbacks.onError(error instanceof Error ? error.message : 'Stream error');
  }
}

// Non-streaming chat (for simple requests)
export async function sendChatMessageSync(
  request: ChatRequest
): Promise<{ response: string; images?: string[] }> {
  const response = await fetch('/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      ...request,
      stream: false,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Chat failed',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

// ============================================
// Image Generation API
// ============================================

export async function generateImage(
  request: ImageGenerateRequest
): Promise<ImageGenerateResponse> {
  const response = await fetch('/image/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Image generation failed',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

// ============================================
// Utility functions
// ============================================

export function clearChat(): Promise<{ success: boolean }> {
  // Note: Using non-API endpoint /clear (not /api/clear)
  return fetch('/clear', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  }).then(res => res.json());
}

export async function getSystemInfo(): Promise<{
  version: string;
  model: string;
  provider: string;
}> {
  return fetchApi('/info');
}

// ============================================
// Web Search API
// ============================================

export async function searchWeb(
  query: string,
  numResults: number = 5
): Promise<WebSearchResponse> {
  const response = await fetch('/api/web/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      num_results: numResults,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Web search failed',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

export async function previewWebUrl(
  url: string,
  maxChars: number = 500
): Promise<WebScrapePreviewResponse> {
  const response = await fetch('/api/web/scrape-preview', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      url,
      max_chars: maxChars,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Web preview failed',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

export async function addWebSources(
  notebookId: string,
  urls: string[],
  sourceName?: string
): Promise<WebSourceAddResponse> {
  const response = await fetch(`/api/notebooks/${notebookId}/web-sources`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      urls,
      source_name: sourceName,  // Pass search query for document naming
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to add web sources',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

// ============================================
// Content Studio API
// ============================================

export async function getStudioGallery(options?: {
  type?: string;
  notebookId?: string;
  limit?: number;
  offset?: number;
}): Promise<StudioGalleryResponse> {
  const params = new URLSearchParams();
  if (options?.type) params.set('type', options.type);
  if (options?.notebookId) params.set('notebook_id', options.notebookId);
  if (options?.limit) params.set('limit', options.limit.toString());
  if (options?.offset) params.set('offset', options.offset.toString());

  const queryString = params.toString();
  const url = `/api/studio/gallery${queryString ? `?${queryString}` : ''}`;

  const response = await fetch(url);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to fetch gallery',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }
  return response.json();
}

export async function generateStudioContent(
  request: StudioGenerateRequest
): Promise<StudioGenerateResponse> {
  const response = await fetch('/api/studio/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Content generation failed',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

export async function getStudioContent(
  contentId: string
): Promise<{ success: boolean; content: GeneratedContent }> {
  const response = await fetch(`/api/studio/content/${contentId}`);

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to fetch content',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

export async function deleteStudioContent(
  contentId: string
): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`/api/studio/content/${contentId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to delete content',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

export async function getStudioGenerators(): Promise<StudioGeneratorsResponse> {
  const response = await fetch('/api/studio/generators');

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to fetch generators',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

// ============================================
// Conversation History API
// ============================================

export interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface ConversationHistoryResponse {
  success: boolean;
  messages: ConversationMessage[];
  count: number;
}

export async function getConversationHistory(
  notebookId: string,
  options?: { limit?: number; offset?: number; userId?: string }
): Promise<ConversationHistoryResponse> {
  const params = new URLSearchParams();
  if (options?.limit) params.set('limit', options.limit.toString());
  if (options?.offset) params.set('offset', options.offset.toString());
  if (options?.userId) params.set('user_id', options.userId);

  const queryString = params.toString();
  const url = `/api/notebooks/${notebookId}/conversations${queryString ? `?${queryString}` : ''}`;

  const response = await fetch(url);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to fetch conversation history',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }
  return response.json();
}

// ============================================
// V2 Chat API (Multi-user with memory)
// ============================================

import type { ChatV2Request, ChatV2Response, ConversationHistoryItem } from '../types';

export interface V2ChatCallbacks {
  onContent: (content: string) => void;
  onSources: (sources: SourceCitation[]) => void;
  onComplete: (sessionId: string, metadata?: ChatMetadata) => void;
  onError: (error: string) => void;
}

/**
 * V2 Chat API with streaming response and conversation memory.
 * Multi-user safe with session tracking.
 */
export async function sendChatV2Message(
  request: ChatV2Request,
  callbacks: V2ChatCallbacks
): Promise<void> {
  const response = await fetch('/api/v2/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    callbacks.onError(errorData.error || `HTTP ${response.status}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const parsed = JSON.parse(data);

            if (parsed.type === 'content') {
              callbacks.onContent(parsed.content);
            } else if (parsed.type === 'sources') {
              callbacks.onSources(parsed.sources);
            } else if (parsed.type === 'done') {
              callbacks.onComplete(parsed.session_id, parsed.metadata);
              return;
            } else if (parsed.type === 'error') {
              callbacks.onError(parsed.error);
              return;
            }
          } catch {
            // Plain text content
            callbacks.onContent(data);
          }
        }
      }
    }
  } catch (error) {
    callbacks.onError(error instanceof Error ? error.message : 'Stream error');
  }
}

/**
 * V2 Chat API synchronous (non-streaming) version.
 */
export async function sendChatV2MessageSync(
  request: ChatV2Request
): Promise<ChatV2Response> {
  const response = await fetch('/api/v2/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Chat failed',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

/**
 * Get V2 chat conversation history.
 */
export async function getChatV2History(
  notebookId: string,
  userId: string,
  limit: number = 50
): Promise<{ success: boolean; history: ConversationHistoryItem[]; count: number }> {
  const params = new URLSearchParams({
    notebook_id: notebookId,
    user_id: userId,
    limit: limit.toString(),
  });

  const response = await fetch(`/api/v2/chat/history?${params}`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to fetch history',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

/**
 * Clear V2 chat conversation history.
 */
export async function clearChatV2History(
  notebookId: string,
  userId: string
): Promise<{ success: boolean; cleared: number }> {
  const response = await fetch('/api/v2/chat/history', {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      notebook_id: notebookId,
      user_id: userId,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to clear history',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}

// ============================================
// Quiz API
// ============================================

/**
 * Create a new quiz from notebook content (admin)
 */
export async function createQuiz(
  request: CreateQuizRequest
): Promise<{ quizId: string; link: string; title: string; questionSource?: string; includeCodeQuestions?: boolean }> {
  const response = await fetch('/api/quiz/create', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      notebook_id: request.notebookId,
      title: request.title,
      num_questions: request.numQuestions || 10,
      difficulty_mode: request.difficultyMode || 'adaptive',
      time_limit: request.timeLimit,
      llm_model: request.llmModel || null,
      question_source: request.questionSource || 'notebook_only',
      include_code_questions: request.includeCodeQuestions || false,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to create quiz',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  const data = await response.json();
  return {
    quizId: data.quiz_id,
    link: data.link,
    title: data.title,
    questionSource: data.question_source,
    includeCodeQuestions: data.include_code_questions,
  };
}

/**
 * Get available LLM models for quiz question generation
 */
export async function getQuizModels(): Promise<{ value: string; label: string }[]> {
  const response = await fetch('/api/models');

  if (!response.ok) {
    console.warn('Failed to fetch quiz models, using default');
    return [{ value: '', label: 'Default' }];
  }

  const data = await response.json();

  // Transform the models to the expected format
  const models: { value: string; label: string }[] = [
    { value: '', label: 'Default' }
  ];

  if (data.models && Array.isArray(data.models)) {
    for (const model of data.models) {
      const provider = (model.provider || '').toLowerCase();
      const name = model.name || '';
      const displayName = model.display_name || name;

      // Create value in "provider:model" format
      const value = `${provider}:${name}`;
      const label = `${displayName} (${model.provider || 'Unknown'})`;

      models.push({ value, label });
    }
  }

  return models;
}

/**
 * List all quizzes created by current user (admin)
 */
export async function listQuizzes(): Promise<Quiz[]> {
  const response = await fetch('/api/quiz/list');

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to list quizzes',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  const data = await response.json();
  // Map snake_case to camelCase
  return (data.quizzes || []).map((q: Record<string, unknown>) => ({
    id: q.id,
    notebookId: q.notebook_id,
    notebookName: q.notebook_name,
    title: q.title,
    numQuestions: q.num_questions,
    difficultyMode: q.difficulty_mode,
    timeLimitMinutes: q.time_limit,
    isActive: q.is_active !== false,
    attemptCount: q.attempt_count || 0,
    link: q.link,
    createdAt: q.created_at,
  }));
}

/**
 * Get quiz results and statistics (admin)
 */
export async function getQuizResults(quizId: string): Promise<QuizResultsResponse> {
  const response = await fetch(`/api/quiz/${quizId}/results`);

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to get quiz results',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  const data = await response.json();
  return {
    quiz: {
      id: data.quiz.id,
      notebookId: data.quiz.notebook_id,
      notebookName: data.quiz.notebook_name,
      title: data.quiz.title,
      numQuestions: data.quiz.num_questions,
      difficultyMode: data.quiz.difficulty_mode,
      timeLimitMinutes: data.quiz.time_limit,
      isActive: data.quiz.is_active,
      attemptCount: data.statistics.total_attempts,
      link: `/quiz/${data.quiz.id}`,
      createdAt: data.quiz.created_at,
    },
    statistics: {
      totalAttempts: data.statistics.total_attempts,
      completedAttempts: data.statistics.completed_attempts,
      avgScore: data.statistics.avg_score,
      avgPercentage: data.statistics.avg_percentage,
      passRate: data.statistics.pass_rate,
    },
    attempts: (data.attempts || []).map((a: Record<string, unknown>) => ({
      id: a.id,
      quizId: quizId,
      takerName: a.taker_name,
      score: a.score,
      total: a.total,
      percentage: a.percentage,
      passed: a.passed,
      startedAt: a.started_at,
      completedAt: a.completed_at,
    })),
  };
}

/**
 * Delete a quiz (admin)
 */
export async function deleteQuiz(quizId: string): Promise<void> {
  const response = await fetch(`/api/quiz/${quizId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to delete quiz',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }
}

/**
 * Get public quiz info (no auth required)
 */
export async function getQuizInfo(quizId: string): Promise<QuizPublicInfo> {
  const response = await fetch(`/api/quiz/${quizId}/info`);

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Quiz not found',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  const data = await response.json();
  return {
    quizId: data.quiz_id,
    title: data.title,
    numQuestions: data.num_questions,
    difficultyMode: data.difficulty_mode,
    timeLimit: data.time_limit,
    hasTimeLimit: data.has_time_limit,
    questionSource: data.question_source,
    includeCodeQuestions: data.include_code_questions,
  };
}

/**
 * Start a quiz attempt (no auth required)
 * If email is provided and there's an incomplete attempt, resumes it.
 */
export async function startQuizAttempt(
  quizId: string,
  takerName: string,
  takerEmail?: string
): Promise<StartAttemptResponse> {
  const response = await fetch(`/api/quiz/${quizId}/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      taker_name: takerName,
      taker_email: takerEmail || null,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to start quiz',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  const data = await response.json();
  return {
    attemptId: data.attempt_id,
    quizTitle: data.quiz_title,
    resumed: data.resumed || false,
    question: data.question ? {
      text: data.question.question,
      question: data.question.question,
      options: data.question.options,
      difficulty: data.difficulty || 'medium',
      correctAnswer: data.question.correct_answer,
      explanation: data.question.explanation,
      topic: data.question.topic,
      type: data.question.type || 'multiple_choice',
      code_snippet: data.question.code_snippet,
    } : null,
    questionNum: data.question_num,
    total: data.total,
    score: data.score,
    timeLimit: data.time_limit,
    difficulty: data.difficulty || 'medium',
  };
}

/**
 * Submit an answer to current question (no auth required)
 */
export async function submitQuizAnswer(
  attemptId: string,
  answer: 'A' | 'B' | 'C' | 'D'
): Promise<SubmitAnswerResponse> {
  const response = await fetch(`/api/quiz/attempt/${attemptId}/answer`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ answer }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to submit answer',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  const data = await response.json();
  return {
    correct: data.correct,
    explanation: data.explanation,
    correctAnswer: data.correct_answer,
    completed: data.completed,
    nextQuestion: data.next_question ? {
      question: data.next_question.question,
      options: data.next_question.options,
      questionNum: data.next_question.question_num,
      total: data.next_question.total,
      difficulty: data.next_question.difficulty,
      type: data.next_question.type || 'multiple_choice',
      code_snippet: data.next_question.code_snippet,
    } : undefined,
    results: data.results ? {
      score: data.results.score,
      total: data.results.total,
      percentage: data.results.percentage,
      passed: data.results.passed,
      answers: data.results.answers,
    } : undefined,
  };
}

/**
 * Get current status of an attempt (for resuming)
 */
export async function getAttemptStatus(attemptId: string): Promise<AttemptStatusResponse> {
  const response = await fetch(`/api/quiz/attempt/${attemptId}/status`);

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to get attempt status',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  const data = await response.json();
  return {
    completed: data.completed,
    quizTitle: data.quiz_title,
    takerName: data.taker_name,
    questionNum: data.question_num,
    total: data.total,
    score: data.score,
    currentQuestion: data.current_question ? {
      question: data.current_question.question,
      options: data.current_question.options,
      difficulty: data.current_question.difficulty,
    } : undefined,
    results: data.results,
  };
}

// ============================================
// Admin Token Metrics API
// ============================================

/**
 * Get token usage metrics for admin dashboard.
 * @param days Number of days to look back (default: 30)
 */
export async function getAdminTokenMetrics(days: number = 30): Promise<TokenMetricsResponse> {
  const response = await fetch(`/api/admin/metrics/tokens?days=${days}`, {
    credentials: 'include',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw {
      error: errorData.error || 'Failed to fetch token metrics',
      message: errorData.message || `HTTP ${response.status}`,
      status: response.status,
    };
  }

  return response.json();
}
