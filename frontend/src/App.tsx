import { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './components/Layout';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/Chat';
import { ToastContainer } from './components/ui';
import { ProtectedRoute } from './components/ProtectedRoute';
import { AppProviders, useNotebook, useDocument, SQLChatProvider, useApp } from './contexts';
import { useNotebooks } from './hooks/useNotebooks';
import { useModels } from './hooks/useModels';
import { useToast } from './hooks/useToast';
import { AnalyticsPage, QueryPage, QuizTakePage, QuizCreatePage, QuizResultsPage, QuizPage, DocumentsManagementPage, DocumentsLandingPage } from './pages';
import { Login } from './pages/Login';
import { Admin } from './pages/Admin';
import { Profile } from './pages/Profile';
import { SQLChatPage } from './components/SQLChat';

function AppContent() {
  // Use App context for navigation and model state
  const { currentView, setModelsState } = useApp();

  // Use hooks for API logic
  const {
    notebooks,
    selectedNotebook,
    documents,
    isLoadingDocs,
    error: notebooksError,
    selectNotebook,
    updateNotebook,
    deleteNotebook,
    uploadDocument,
  } = useNotebooks();

  const {
    models,
    selectedModel,
    selectedProvider,
    isLoading: isLoadingModels,
    error: modelsError,
    selectModel,
  } = useModels();

  // Sync models state with AppContext for the Header
  useEffect(() => {
    setModelsState({
      models,
      selectedModel,
      selectedProvider,
      isLoadingModels,
      selectModel,
    });
  }, [models, selectedModel, selectedProvider, isLoadingModels, selectModel, setModelsState]);

  const { toasts, removeToast, success, error: showError } = useToast();

  // Get context setters to sync hook state with contexts
  const notebookContext = useNotebook();
  const documentContext = useDocument();

  // Sync notebooks hook state with context
  useEffect(() => {
    notebookContext.setNotebooks(notebooks);
  }, [notebooks, notebookContext]);

  useEffect(() => {
    notebookContext.selectNotebook(selectedNotebook);
  }, [selectedNotebook, notebookContext]);

  // Sync documents hook state with context
  useEffect(() => {
    documentContext.setDocuments(documents);
  }, [documents, documentContext]);

  useEffect(() => {
    documentContext.setLoading(isLoadingDocs);
  }, [isLoadingDocs, documentContext]);

  // Show errors from hooks as toasts
  useEffect(() => {
    if (notebooksError) {
      console.error('Notebooks API error:', notebooksError);
      showError(`Notebooks: ${notebooksError}`);
    }
  }, [notebooksError, showError]);

  useEffect(() => {
    if (modelsError) {
      console.error('Models API error:', modelsError);
      showError(`Models: ${modelsError}`);
    }
  }, [modelsError, showError]);

  // Handle file upload with toast notifications
  const handleFileUpload = async (file: File): Promise<boolean> => {
    const result = await uploadDocument(file);
    if (result) {
      success(`Uploaded: ${file.name}`);
    } else {
      showError(`Failed to upload: ${file.name}`);
    }
    return result;
  };


  // Handle copy to clipboard
  const handleCopy = (content: string) => {
    navigator.clipboard.writeText(content);
    success('Copied to clipboard');
  };

  // Render Analytics page
  if (currentView === 'analytics') {
    return (
      <MainLayout header={<Header />}>
        <AnalyticsPage notebookId={selectedNotebook?.id} />
        <ToastContainer toasts={toasts} onDismiss={removeToast} />
      </MainLayout>
    );
  }

  // Render SQL Chat page
  if (currentView === 'sql-chat') {
    return (
      <MainLayout header={<Header />}>
        <SQLChatPage />
        <ToastContainer toasts={toasts} onDismiss={removeToast} />
      </MainLayout>
    );
  }

  // Render Query API page
  if (currentView === 'query-api') {
    return (
      <MainLayout header={<Header />}>
        <QueryPage />
        <ToastContainer toasts={toasts} onDismiss={removeToast} />
      </MainLayout>
    );
  }

  // Render main Chat view
  return (
    <MainLayout
      header={<Header />}
      sidebar={
        <Sidebar
          // Notebook operations
          onSelectNotebook={selectNotebook}
          onDeleteNotebook={deleteNotebook}
          onUpdateNotebook={updateNotebook}
        />
      }
    >
      <ChatArea
        notebookId={selectedNotebook?.id}
        notebookName={selectedNotebook?.name}
        selectedModel={selectedModel}
        onCopy={handleCopy}
        onFileUpload={handleFileUpload}
      />

      {/* Toast notifications */}
      <ToastContainer toasts={toasts} onDismiss={removeToast} />
    </MainLayout>
  );
}

function AppRoutes() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/quiz/:quizId" element={<QuizTakePage />} />

      {/* Protected routes */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <SQLChatProvider>
              <AppContent />
            </SQLChatProvider>
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin"
        element={
          <ProtectedRoute requireAdmin>
            <Admin />
          </ProtectedRoute>
        }
      />
      <Route
        path="/profile"
        element={
          <ProtectedRoute>
            <Profile />
          </ProtectedRoute>
        }
      />
      <Route
        path="/documents"
        element={
          <ProtectedRoute>
            <DocumentsLandingPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/quizzes"
        element={
          <ProtectedRoute>
            <QuizPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/quizzes/create"
        element={
          <ProtectedRoute>
            <QuizCreatePage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/quizzes/:quizId/results"
        element={
          <ProtectedRoute>
            <QuizResultsPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/notebook/:notebookId/documents"
        element={
          <ProtectedRoute>
            <DocumentsManagementPage />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppProviders>
        <AppRoutes />
      </AppProviders>
    </BrowserRouter>
  );
}

export default App;
