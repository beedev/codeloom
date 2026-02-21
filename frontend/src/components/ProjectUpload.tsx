/**
 * ProjectUpload Component
 *
 * Three-tab ingestion UI: Upload Zip | Git Repository | Local Path.
 * Shows upload/ingestion progress and results for all source types.
 */

import { useState, useRef, useCallback } from 'react';
import type { DragEvent, ChangeEvent, FormEvent } from 'react';
import {
  Upload,
  GitBranch,
  FolderOpen,
  CheckCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import * as api from '../services/api.ts';
import type { IngestionResult } from '../types/index.ts';

interface ProjectUploadProps {
  projectId: string;
  onUploadComplete?: (result: IngestionResult) => void;
}

type SourceTab = 'zip' | 'git' | 'local';
type IngestionPhase = 'idle' | 'processing' | 'success' | 'error';

const TABS: { key: SourceTab; label: string; icon: typeof Upload }[] = [
  { key: 'zip', label: 'Upload Zip', icon: Upload },
  { key: 'git', label: 'Git Repository', icon: GitBranch },
  { key: 'local', label: 'Local Path', icon: FolderOpen },
];

export function ProjectUpload({
  projectId,
  onUploadComplete,
}: ProjectUploadProps) {
  const [activeTab, setActiveTab] = useState<SourceTab>('zip');
  const [phase, setPhase] = useState<IngestionPhase>('idle');
  const [result, setResult] = useState<IngestionResult | null>(null);
  const [errorMsg, setErrorMsg] = useState('');

  const resetState = useCallback(() => {
    setPhase('idle');
    setResult(null);
    setErrorMsg('');
  }, []);

  const handleTabChange = useCallback(
    (tab: SourceTab) => {
      setActiveTab(tab);
      resetState();
    },
    [resetState],
  );

  const handleSuccess = useCallback(
    (ingestionResult: IngestionResult) => {
      setResult(ingestionResult);
      setPhase('success');
      onUploadComplete?.(ingestionResult);
    },
    [onUploadComplete],
  );

  const handleError = useCallback((err: unknown) => {
    const message = err instanceof Error ? err.message : 'Ingestion failed';
    setErrorMsg(message);
    setPhase('error');
  }, []);

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 rounded-lg bg-void-light/50 p-1">
        {TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => handleTabChange(key)}
            disabled={phase === 'processing'}
            className={`flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-2 text-xs font-medium transition-colors ${
              activeTab === key
                ? 'bg-void-surface text-text'
                : 'text-text-muted hover:text-text-dim'
            } ${phase === 'processing' ? 'opacity-60' : ''}`}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'zip' && (
        <ZipUpload
          projectId={projectId}
          phase={phase}
          setPhase={setPhase}
          onSuccess={handleSuccess}
          onError={handleError}
        />
      )}
      {activeTab === 'git' && (
        <GitIngest
          projectId={projectId}
          phase={phase}
          setPhase={setPhase}
          onSuccess={handleSuccess}
          onError={handleError}
        />
      )}
      {activeTab === 'local' && (
        <LocalIngest
          projectId={projectId}
          phase={phase}
          setPhase={setPhase}
          onSuccess={handleSuccess}
          onError={handleError}
        />
      )}

      {/* Result / Error display */}
      <IngestionFeedback phase={phase} result={result} errorMsg={errorMsg} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Zip Upload (drag-and-drop)
// ---------------------------------------------------------------------------

interface TabProps {
  projectId: string;
  phase: IngestionPhase;
  setPhase: (p: IngestionPhase) => void;
  onSuccess: (r: IngestionResult) => void;
  onError: (e: unknown) => void;
}

function ZipUpload({ projectId, phase, setPhase, onSuccess, onError }: TabProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith('.zip')) {
        onError(new Error('Only .zip files are accepted.'));
        return;
      }
      setPhase('processing');
      try {
        const result = await api.uploadCodebase(projectId, file);
        onSuccess(result);
      } catch (err) {
        onError(err);
      }
    },
    [projectId, setPhase, onSuccess, onError],
  );

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile],
  );

  const handleFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) processFile(file);
    },
    [processFile],
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={() => setIsDragOver(false)}
      onClick={() => fileInputRef.current?.click()}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
        isDragOver
          ? 'border-glow bg-glow/5'
          : 'border-void-surface bg-void-light/50 hover:border-void-surface hover:bg-void-light'
      } ${phase === 'processing' ? 'pointer-events-none opacity-60' : ''}`}
      role="button"
      tabIndex={0}
      aria-label="Upload codebase zip file"
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          fileInputRef.current?.click();
        }
      }}
    >
      {phase === 'processing' ? (
        <Loader2 className="h-8 w-8 animate-spin text-glow" />
      ) : (
        <Upload className="h-8 w-8 text-text-muted" />
      )}
      <p className="mt-3 text-sm text-text-muted">
        {phase === 'processing'
          ? 'Uploading and processing...'
          : 'Drop a .zip file here or click to browse'}
      </p>
      <p className="mt-1 text-xs text-text-dim">Max 50 MB</p>
      <input
        ref={fileInputRef}
        type="file"
        accept=".zip"
        onChange={handleFileChange}
        className="hidden"
        aria-hidden="true"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Git Repository Ingestion
// ---------------------------------------------------------------------------

function GitIngest({ projectId, phase, setPhase, onSuccess, onError }: TabProps) {
  const [repoUrl, setRepoUrl] = useState('');
  const [branch, setBranch] = useState('main');

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!repoUrl.trim()) return;
      setPhase('processing');
      try {
        const result = await api.ingestFromGit(projectId, repoUrl.trim(), branch.trim() || 'main');
        onSuccess(result);
      } catch (err) {
        onError(err);
      }
    },
    [projectId, repoUrl, branch, setPhase, onSuccess, onError],
  );

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div>
        <label htmlFor="repo-url" className="mb-1 block text-xs font-medium text-text-muted">
          Repository URL
        </label>
        <input
          id="repo-url"
          type="text"
          value={repoUrl}
          onChange={(e) => setRepoUrl(e.target.value)}
          placeholder="https://github.com/user/repo.git"
          disabled={phase === 'processing'}
          className="w-full rounded-md border border-void-surface bg-void-light px-3 py-2 text-sm text-text placeholder-text-dim focus:border-glow/50 focus:outline-none disabled:opacity-60"
        />
      </div>
      <div>
        <label htmlFor="repo-branch" className="mb-1 block text-xs font-medium text-text-muted">
          Branch
        </label>
        <input
          id="repo-branch"
          type="text"
          value={branch}
          onChange={(e) => setBranch(e.target.value)}
          placeholder="main"
          disabled={phase === 'processing'}
          className="w-full rounded-md border border-void-surface bg-void-light px-3 py-2 text-sm text-text placeholder-text-dim focus:border-glow/50 focus:outline-none disabled:opacity-60"
        />
      </div>
      <button
        type="submit"
        disabled={phase === 'processing' || !repoUrl.trim()}
        className="flex w-full items-center justify-center gap-2 rounded-md bg-glow px-4 py-2 text-sm font-medium text-white hover:bg-glow-dim disabled:opacity-60"
      >
        {phase === 'processing' ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Cloning and ingesting...
          </>
        ) : (
          <>
            <GitBranch className="h-4 w-4" />
            Clone & Ingest
          </>
        )}
      </button>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Local Path Ingestion
// ---------------------------------------------------------------------------

function LocalIngest({ projectId, phase, setPhase, onSuccess, onError }: TabProps) {
  const [dirPath, setDirPath] = useState('');

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!dirPath.trim()) return;
      setPhase('processing');
      try {
        const result = await api.ingestFromLocal(projectId, dirPath.trim());
        onSuccess(result);
      } catch (err) {
        onError(err);
      }
    },
    [projectId, dirPath, setPhase, onSuccess, onError],
  );

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div>
        <label htmlFor="dir-path" className="mb-1 block text-xs font-medium text-text-muted">
          Directory Path
        </label>
        <input
          id="dir-path"
          type="text"
          value={dirPath}
          onChange={(e) => setDirPath(e.target.value)}
          placeholder="/Users/you/projects/my-app"
          disabled={phase === 'processing'}
          className="w-full rounded-md border border-void-surface bg-void-light px-3 py-2 text-sm text-text placeholder-text-dim focus:border-glow/50 focus:outline-none disabled:opacity-60"
        />
        <p className="mt-1 text-xs text-text-dim">
          Absolute path to a directory on the server
        </p>
      </div>
      <button
        type="submit"
        disabled={phase === 'processing' || !dirPath.trim()}
        className="flex w-full items-center justify-center gap-2 rounded-md bg-glow px-4 py-2 text-sm font-medium text-white hover:bg-glow-dim disabled:opacity-60"
      >
        {phase === 'processing' ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Ingesting...
          </>
        ) : (
          <>
            <FolderOpen className="h-4 w-4" />
            Ingest
          </>
        )}
      </button>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Shared result / error feedback
// ---------------------------------------------------------------------------

function IngestionFeedback({
  phase,
  result,
  errorMsg,
}: {
  phase: IngestionPhase;
  result: IngestionResult | null;
  errorMsg: string;
}) {
  if (phase === 'success' && result) {
    return (
      <div className="rounded-lg border border-success/30 bg-success/5 p-4">
        <div className="mb-2 flex items-center gap-2 text-success">
          <CheckCircle className="h-4 w-4" />
          <span className="text-sm font-medium">Ingestion complete</span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs text-text-muted">
          <div>Files processed: <span className="text-text">{result.files_processed}</span></div>
          <div>Files skipped: <span className="text-text">{result.files_skipped}</span></div>
          <div>Units extracted: <span className="text-text">{result.units_extracted}</span></div>
          <div>Chunks created: <span className="text-text">{result.chunks_created}</span></div>
          <div>Embeddings stored: <span className="text-text">{result.embeddings_stored}</span></div>
          <div>Time: <span className="text-text">{result.elapsed_seconds.toFixed(1)}s</span></div>
        </div>
        {result.errors.length > 0 && (
          <div className="mt-3 space-y-1">
            <p className="text-xs font-medium text-warning">Warnings:</p>
            {result.errors.map((err, i) => (
              <p key={i} className="text-xs text-warning/70">{err}</p>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (phase === 'error') {
    return (
      <div className="rounded-lg border border-danger/30 bg-danger/5 p-4">
        <div className="flex items-center gap-2 text-danger">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">{errorMsg}</span>
        </div>
      </div>
    );
  }

  return null;
}
