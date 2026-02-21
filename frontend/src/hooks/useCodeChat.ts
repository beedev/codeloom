/**
 * useCodeChat Hook
 *
 * SSE streaming chat for code RAG queries.
 * Uses fetch + ReadableStream to parse server-sent events
 * from POST /api/projects/{id}/chat/stream.
 *
 * SSE event format:
 *   data: {"type":"sources","sources":[...]}
 *   data: {"type":"content","content":"..."}
 *   data: {"type":"done"}
 *   data: {"type":"error","error":"..."}
 */

import { useState, useCallback, useRef } from 'react';
import type { ChatMessage, ChatSource } from '../types/index.ts';

export interface ImpactEntry {
  unit_name: string;
  file_path: string;
  direct: number;
  indirect: number;
  files_affected: number;
  impact_score: number;
  impact_level: string;
  dependents: { name: string; edge_type: string; depth: number }[];
}

interface UseCodeChatReturn {
  messages: ChatMessage[];
  sources: ChatSource[];
  impact: ImpactEntry[];
  isStreaming: boolean;
  error: string | null;
  sendMessage: (projectId: string, query: string, mode?: 'chat' | 'impact') => Promise<void>;
  clearMessages: () => void;
}

export function useCodeChat(): UseCodeChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<ChatSource[]>([]);
  const [impact, setImpact] = useState<ImpactEntry[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (projectId: string, query: string, mode: 'chat' | 'impact' = 'chat') => {
      // Abort any in-flight request
      if (abortRef.current) {
        abortRef.current.abort();
      }

      const controller = new AbortController();
      abortRef.current = controller;

      // Add user message immediately
      const userMessage: ChatMessage = { role: 'user', content: query };
      setMessages((prev) => [...prev, userMessage]);
      setSources([]);
      setImpact([]);
      setError(null);
      setIsStreaming(true);

      // Placeholder for assistant message -- will accumulate tokens
      let assistantContent = '';
      let streamSources: ChatSource[] = [];

      setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

      try {
        const response = await fetch(
          `/api/projects/${projectId}/chat/stream`,
          {
            method: 'POST',
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json',
              Accept: 'text/event-stream',
            },
            body: JSON.stringify({
              query,
              response_type: localStorage.getItem('codeloom_response_type') || 'detailed',
              mode,
            }),
            signal: controller.signal,
          },
        );

        if (!response.ok) {
          const body = await response.text();
          throw new Error(body || `HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6).trim();
            if (!data || data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data) as {
                type: string;
                content?: string;
                sources?: ChatSource[];
                impact?: ImpactEntry[];
                error?: string;
              };

              if (parsed.type === 'impact' && parsed.impact) {
                setImpact(parsed.impact);
              } else if (parsed.type === 'impact_status') {
                const statusMsg = (parsed as { message?: string }).message || 'Impact analysis unavailable.';
                assistantContent += `> **Impact Analysis**: ${statusMsg}\n\n`;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: assistantContent,
                  };
                  return updated;
                });
              } else if (parsed.type === 'sources' && parsed.sources) {
                streamSources = parsed.sources;
                setSources(parsed.sources);
              } else if (parsed.type === 'content' && parsed.content) {
                assistantContent += parsed.content;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: assistantContent,
                  };
                  return updated;
                });
              } else if (parsed.type === 'done') {
                // Final message -- attach sources
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: assistantContent,
                    sources: streamSources,
                  };
                  return updated;
                });
              } else if (parsed.type === 'error') {
                setError(parsed.error ?? 'Unknown stream error');
              }
            } catch {
              // Non-JSON line -- treat as raw content token
              assistantContent += data;
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role: 'assistant',
                  content: assistantContent,
                };
                return updated;
              });
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          // User cancelled -- not an error
          return;
        }
        const message = err instanceof Error ? err.message : 'Stream failed';
        setError(message);
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [],
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setSources([]);
    setImpact([]);
    setError(null);
  }, []);

  return {
    messages,
    sources,
    impact,
    isStreaming,
    error,
    sendMessage,
    clearMessages,
  };
}
