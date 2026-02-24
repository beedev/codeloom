/**
 * App.tsx -- Root component and router for CodeLoom.
 *
 * Routes:
 *   /login            -> Login page (public)
 *   /                 -> Dashboard (protected)
 *   /project/:id      -> Project view (protected)
 *   /project/:id/chat -> Code chat (protected)
 */

import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext.tsx';
import { Login } from './pages/Login.tsx';
import { Dashboard } from './pages/Dashboard.tsx';
import { ProjectView } from './pages/ProjectView.tsx';
import { CodeChatPage } from './pages/CodeChatPage.tsx';
import { MigrationWizard } from './pages/MigrationWizard.tsx';
import { MigrationPlans } from './pages/MigrationPlans.tsx';
import { AnalyticsPage } from './pages/AnalyticsPage.tsx';
import { Settings } from './pages/Settings.tsx';
import { ProjectWikiPage } from './pages/ProjectWikiPage.tsx';

// ---------------------------------------------------------------------------
// ProtectedRoute -- redirects to /login if not authenticated
// ---------------------------------------------------------------------------

function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-void">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-void-surface border-t-glow" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}

// ---------------------------------------------------------------------------
// AppRoutes
// ---------------------------------------------------------------------------

function AppRoutes() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/login" element={<Login />} />

      {/* Protected */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/project/:id"
        element={
          <ProtectedRoute>
            <ProjectView />
          </ProtectedRoute>
        }
      />
      <Route
        path="/project/:id/chat"
        element={
          <ProtectedRoute>
            <CodeChatPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/project/:id/analytics"
        element={
          <ProtectedRoute>
            <AnalyticsPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/migrations"
        element={
          <ProtectedRoute>
            <MigrationPlans />
          </ProtectedRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ProtectedRoute>
            <Settings />
          </ProtectedRoute>
        }
      />
      <Route
        path="/project/:id/wiki"
        element={
          <ProtectedRoute>
            <ProjectWikiPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/migration/:planId"
        element={
          <ProtectedRoute>
            <MigrationWizard />
          </ProtectedRoute>
        }
      />

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
