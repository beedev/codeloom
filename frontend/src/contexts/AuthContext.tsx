/**
 * AuthContext
 *
 * Provides authentication state to the entire application.
 * Checks /api/auth/me on mount to restore sessions from cookies.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
} from 'react';
import type { ReactNode } from 'react';
import type { User } from '../types/index.ts';
import * as api from '../services/api.ts';

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isEditor: boolean;
  isViewer: boolean;
  hasRole: (role: string) => boolean;
  login: (credentials: { username: string; password: string }) => Promise<boolean>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const checkAuth = useCallback(async () => {
    try {
      const currentUser = await api.getCurrentUser();
      setUser(currentUser);
    } catch {
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  const login = useCallback(
    async (credentials: { username: string; password: string }): Promise<boolean> => {
      try {
        const loggedInUser = await api.login(credentials);
        setUser(loggedInUser);
        return true;
      } catch {
        return false;
      }
    },
    [],
  );

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      // Ignore logout errors -- clear local state regardless
    }
    setUser(null);
  }, []);

  const roles = user?.roles ?? [];
  const hasRole = useCallback(
    (role: string) => roles.includes(role),
    [roles],
  );

  const value: AuthState = {
    user,
    isLoading,
    isAuthenticated: user !== null,
    isAdmin: roles.includes('admin'),
    isEditor: roles.includes('admin') || roles.includes('editor'),
    isViewer: user !== null,
    hasRole,
    login,
    logout,
    checkAuth,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
