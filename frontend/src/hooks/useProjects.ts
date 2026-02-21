/**
 * useProjects Hook
 *
 * Manages the project list: fetch, create, delete, refresh.
 * Provides loading and error states for UI feedback.
 */

import { useState, useEffect, useCallback } from 'react';
import type { Project } from '../types/index.ts';
import * as api from '../services/api.ts';

interface UseProjectsReturn {
  projects: Project[];
  isLoading: boolean;
  error: string | null;
  createProject: (name: string, description?: string) => Promise<Project | null>;
  deleteProject: (projectId: string) => Promise<boolean>;
  refreshProjects: () => Promise<void>;
}

export function useProjects(): UseProjectsReturn {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshProjects = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.listProjects();
      setProjects(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load projects';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshProjects();
  }, [refreshProjects]);

  const createProject = useCallback(
    async (name: string, description?: string): Promise<Project | null> => {
      setError(null);
      try {
        const project = await api.createProject({ name, description });
        setProjects((prev) => [project, ...prev]);
        return project;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to create project';
        setError(message);
        return null;
      }
    },
    [],
  );

  const deleteProject = useCallback(
    async (projectId: string): Promise<boolean> => {
      setError(null);
      try {
        await api.deleteProject(projectId);
        setProjects((prev) => prev.filter((p) => p.project_id !== projectId));
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to delete project';
        setError(message);
        return false;
      }
    },
    [],
  );

  return {
    projects,
    isLoading,
    error,
    createProject,
    deleteProject,
    refreshProjects,
  };
}
