/**
 * CodeChatPage
 *
 * 60/40 split: Chat messages on left, Source References on right.
 * Top: Settings panel (collapsible)
 */

import { useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Settings } from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import { CodeChat } from '../components/CodeChat.tsx';
import { ImpactPanel } from '../components/ImpactPanel.tsx';
import { SourcePanel } from '../components/SourcePanel.tsx';
import { SettingsPanel } from '../components/SettingsPanel.tsx';
import { useCodeChat } from '../hooks/useCodeChat.ts';
import { useProjects } from '../hooks/useProjects.ts';

export function CodeChatPage() {
  const { id: projectId } = useParams<{ id: string }>();
  const { projects, isLoading: projectsLoading } = useProjects();
  const { messages, sources, impact, isStreaming, error, sendMessage, clearMessages } =
    useCodeChat();

  const [showSettings, setShowSettings] = useState(false);

  const handleSendMessage = useCallback(
    (query: string, mode?: 'chat' | 'impact') => {
      if (projectId) {
        sendMessage(projectId, query, mode);
      }
    },
    [projectId, sendMessage],
  );

  if (!projectId) {
    return (
      <Layout projects={projects} projectsLoading={projectsLoading}>
        <div className="flex h-full items-center justify-center text-text-muted">
          No project selected
        </div>
      </Layout>
    );
  }

  const project = projects.find((p) => p.project_id === projectId);

  return (
    <Layout projects={projects} projectsLoading={projectsLoading}>
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-void-surface px-4 py-3">
        <div className="flex items-center gap-3">
          <Link
            to={`/project/${projectId}`}
            className="rounded p-1 text-text-muted hover:bg-void-surface hover:text-text"
            aria-label="Back to project"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <h1 className="text-sm font-medium text-text">
            {project?.name ?? 'Project'} / Chat
          </h1>
        </div>
        <button
          onClick={() => setShowSettings((s) => !s)}
          className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors ${
            showSettings
              ? 'border-glow bg-glow/20 text-glow'
              : 'border-void-surface text-text-muted hover:bg-void-surface hover:text-text'
          }`}
        >
          <Settings className="h-3.5 w-3.5" />
          Settings
        </button>
      </div>

      {/* Settings panel (collapsible) */}
      {showSettings && <SettingsPanel />}

      {/* 60/40 split: Chat + Source References */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chat - 60% */}
        <div className="flex w-3/5 flex-col border-r border-void-surface">
          <CodeChat
            messages={messages}
            isStreaming={isStreaming}
            error={error}
            projectId={projectId}
            hideInlineSources
            onSendMessage={handleSendMessage}
            onClear={clearMessages}
          />
        </div>
        {/* Source References + Impact - 40% */}
        <div className="w-2/5 overflow-hidden flex flex-col">
          {impact.length > 0 && <ImpactPanel impact={impact} />}
          <div className="flex-1 overflow-hidden">
            <SourcePanel sources={sources} projectId={projectId} />
          </div>
        </div>
      </div>
    </Layout>
  );
}
