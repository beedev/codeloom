/**
 * ProjectList Component
 *
 * Sidebar list of projects with name, language badge, file count.
 * Click to navigate to project view.
 */

import { useNavigate, useParams } from 'react-router-dom';
import { FolderCode, Hash } from 'lucide-react';
import type { Project } from '../types/index.ts';

interface ProjectListProps {
  projects: Project[];
  isLoading: boolean;
}

/** Language-to-color mapping for badges */
const languageColors: Record<string, string> = {
  python: 'bg-yellow-600/30 text-yellow-300',
  javascript: 'bg-yellow-500/30 text-yellow-200',
  typescript: 'bg-blue-500/30 text-blue-300',
  java: 'bg-red-500/30 text-red-300',
  csharp: 'bg-purple-500/30 text-purple-300',
};

function getLanguageBadge(language: string | null): string {
  if (!language) return 'bg-gray-700/50 text-gray-400';
  return languageColors[language.toLowerCase()] ?? 'bg-gray-700/50 text-gray-400';
}

export function ProjectList({ projects, isLoading }: ProjectListProps) {
  const navigate = useNavigate();
  const { id: activeId } = useParams<{ id: string }>();

  if (isLoading) {
    return (
      <div className="space-y-2 px-1">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-16 animate-pulse rounded-md bg-gray-800"
          />
        ))}
      </div>
    );
  }

  if (projects.length === 0) {
    return (
      <p className="px-1 text-sm text-gray-500">No projects yet.</p>
    );
  }

  return (
    <div className="space-y-1">
      <p className="mb-2 px-1 text-xs font-medium uppercase tracking-wider text-gray-500">
        Projects
      </p>
      {projects.map((project) => {
        const isActive = project.project_id === activeId;
        return (
          <button
            key={project.project_id}
            onClick={() => navigate(`/project/${project.project_id}`)}
            className={`flex w-full items-start gap-2.5 rounded-md px-2.5 py-2 text-left transition-colors ${
              isActive
                ? 'bg-blue-600/20 text-white'
                : 'text-gray-300 hover:bg-gray-800 hover:text-white'
            }`}
          >
            <FolderCode className="mt-0.5 h-4 w-4 shrink-0 text-gray-500" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{project.name}</p>
              <div className="mt-1 flex items-center gap-2 text-xs text-gray-500">
                {project.primary_language && (
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${getLanguageBadge(
                      project.primary_language,
                    )}`}
                  >
                    {project.primary_language}
                  </span>
                )}
                <span className="flex items-center gap-0.5">
                  <Hash className="h-3 w-3" />
                  {project.file_count} files
                </span>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
