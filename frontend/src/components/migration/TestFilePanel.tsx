/**
 * TestFilePanel — grouped test file viewer for Phase 6 (Test Generation).
 *
 * Groups files by test_type (unit → integration → equivalence → sp_stub),
 * shows summary stats with colored badges, and provides file browsing
 * with syntax highlighting.
 */

import { useState, useMemo } from 'react';
import { Highlight, themes } from 'prism-react-renderer';
import {
  ChevronRight,
  ChevronDown,
  Download,
  FileCode,
  Play,
  FlaskConical,
  Layers,
  Equal,
  Database,
} from 'lucide-react';
import type { MigrationFile } from '../../types/index.ts';

// ── Test type config ────────────────────────────────────────────────

type TestType = 'unit' | 'integration' | 'equivalence' | 'sp_stub';

const TEST_TYPE_ORDER: TestType[] = ['unit', 'integration', 'equivalence', 'sp_stub'];

const TEST_TYPE_CONFIG: Record<TestType, {
  label: string;
  icon: typeof FlaskConical;
  color: string;
  bgColor: string;
  borderColor: string;
}> = {
  unit: {
    label: 'Unit Tests',
    icon: FlaskConical,
    color: 'text-green-400',
    bgColor: 'bg-green-500/10',
    borderColor: 'border-green-500/20',
  },
  integration: {
    label: 'Integration Tests',
    icon: Layers,
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/20',
  },
  equivalence: {
    label: 'Equivalence Tests',
    icon: Equal,
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/20',
  },
  sp_stub: {
    label: 'SP Stub Tests',
    icon: Database,
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/20',
  },
};

// ── Props ───────────────────────────────────────────────────────────

interface TestFilePanelProps {
  files: MigrationFile[];
  onDownloadAll: () => void;
  onDownloadFile: (filePath: string) => void;
}

// ── Component ───────────────────────────────────────────────────────

export function TestFilePanel({
  files,
  onDownloadAll,
  onDownloadFile,
}: TestFilePanelProps) {
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(
    () => new Set(TEST_TYPE_ORDER),
  );
  const [selectedFile, setSelectedFile] = useState<MigrationFile | null>(null);
  const [showRunModal, setShowRunModal] = useState(false);

  // Group files by test_type
  const groups = useMemo(() => {
    const grouped = new Map<string, MigrationFile[]>();
    for (const file of files) {
      const type = file.test_type ?? 'unit';
      const existing = grouped.get(type) ?? [];
      existing.push(file);
      grouped.set(type, existing);
    }
    return grouped;
  }, [files]);

  const toggleGroup = (type: string) => {
    setExpandedGroups(prev => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left sidebar: file groups */}
      <div className="w-72 shrink-0 overflow-y-auto border-r border-void-surface">
        {/* Summary header */}
        <div className="border-b border-void-surface px-4 py-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-medium text-text">Test Files</h3>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setShowRunModal(true)}
                className="flex items-center gap-1 rounded border border-void-surface px-2 py-1 text-[10px] text-text-muted hover:border-glow/30 hover:text-text"
                title="Run Locally"
              >
                <Play className="h-2.5 w-2.5" />
                Run Locally
              </button>
              <button
                onClick={onDownloadAll}
                className="flex items-center gap-1 rounded border border-void-surface px-2 py-1 text-[10px] text-text-muted hover:border-glow/30 hover:text-text"
                title="Download Tests"
              >
                <Download className="h-2.5 w-2.5" />
                Download
              </button>
            </div>
          </div>

          {/* Stats badges */}
          <div className="mt-2 flex flex-wrap gap-1.5">
            {TEST_TYPE_ORDER.map((type) => {
              const count = groups.get(type)?.length ?? 0;
              if (count === 0) return null;
              const cfg = TEST_TYPE_CONFIG[type];
              return (
                <span
                  key={type}
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${cfg.bgColor} ${cfg.color}`}
                >
                  {count} {cfg.label.replace(' Tests', '')}
                </span>
              );
            })}
          </div>
        </div>

        {/* File groups */}
        <div className="py-1">
          {TEST_TYPE_ORDER.map((type) => {
            const groupFiles = groups.get(type);
            if (!groupFiles || groupFiles.length === 0) return null;
            const cfg = TEST_TYPE_CONFIG[type];
            const Icon = cfg.icon;
            const isExpanded = expandedGroups.has(type);

            return (
              <div key={type}>
                {/* Group header */}
                <button
                  onClick={() => toggleGroup(type)}
                  className="flex w-full items-center gap-2 px-3 py-2 text-xs hover:bg-void-light/50"
                >
                  {isExpanded
                    ? <ChevronDown className="h-3 w-3 text-text-dim" />
                    : <ChevronRight className="h-3 w-3 text-text-dim" />
                  }
                  <Icon className={`h-3.5 w-3.5 ${cfg.color}`} />
                  <span className={`font-medium ${cfg.color}`}>{cfg.label}</span>
                  <span className={`ml-auto rounded-full px-1.5 py-0.5 text-[9px] ${cfg.bgColor} ${cfg.color}`}>
                    {groupFiles.length}
                  </span>
                </button>

                {/* File list */}
                {isExpanded && (
                  <div className="pb-1">
                    {groupFiles.map((file, idx) => {
                      const isSelected = selectedFile === file;
                      return (
                        <button
                          key={idx}
                          onClick={() => setSelectedFile(file)}
                          className={`flex w-full items-center gap-2 px-3 py-1.5 pl-9 text-xs ${
                            isSelected
                              ? 'bg-glow/10 text-glow'
                              : 'text-text-muted hover:bg-void-light/30 hover:text-text'
                          }`}
                        >
                          <FileCode className="h-3 w-3 shrink-0" />
                          <span className="truncate">{file.file_path.split('/').pop()}</span>
                          <button
                            onClick={(e) => { e.stopPropagation(); onDownloadFile(file.file_path); }}
                            className="ml-auto shrink-0 rounded p-0.5 text-text-dim/50 hover:text-text-muted"
                            title="Download"
                          >
                            <Download className="h-2.5 w-2.5" />
                          </button>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Right: code viewer */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {selectedFile ? (
          <>
            {/* File header */}
            <div className="flex items-center justify-between border-b border-void-surface px-4 py-2">
              <div className="flex items-center gap-2 text-xs">
                <FileCode className="h-3.5 w-3.5 text-text-dim" />
                <span className="font-mono text-text-muted">{selectedFile.file_path}</span>
                <span className="rounded bg-void-surface px-1.5 py-0.5 text-[10px] text-text-dim">
                  {selectedFile.language}
                </span>
                {selectedFile.test_type && (
                  <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
                    TEST_TYPE_CONFIG[selectedFile.test_type]?.bgColor ?? 'bg-void-surface'
                  } ${TEST_TYPE_CONFIG[selectedFile.test_type]?.color ?? 'text-text-dim'}`}>
                    {TEST_TYPE_CONFIG[selectedFile.test_type]?.label ?? selectedFile.test_type}
                  </span>
                )}
              </div>
              <button
                onClick={() => onDownloadFile(selectedFile.file_path)}
                className="flex items-center gap-1 rounded border border-void-surface px-2 py-1 text-[10px] text-text-dim hover:border-glow/30 hover:text-text-muted"
              >
                <Download className="h-2.5 w-2.5" />
                Download
              </button>
            </div>

            {/* Code */}
            <div className="flex-1 overflow-auto">
              <TestCodeView file={selectedFile} />
            </div>
          </>
        ) : (
          <div className="flex h-full items-center justify-center text-text-dim">
            <div className="text-center">
              <FlaskConical className="mx-auto h-8 w-8 opacity-30" />
              <p className="mt-3 text-sm">Select a test file to view</p>
              <p className="mt-1 text-xs">
                {files.length} test file{files.length !== 1 ? 's' : ''} generated across{' '}
                {groups.size} categor{groups.size !== 1 ? 'ies' : 'y'}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Run Locally modal */}
      {showRunModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="mx-4 max-w-lg rounded-xl border border-void-surface bg-void-light p-6 shadow-2xl">
            <h3 className="text-sm font-medium text-text">Run Tests Locally</h3>
            <p className="mt-2 text-xs text-text-dim">
              Download the test files and run them with your project's test framework:
            </p>
            <div className="mt-3 rounded-lg bg-void p-3 font-code text-xs text-text-muted">
              <p className="text-text-dim"># 1. Download the test ZIP</p>
              <p># 2. Extract into your project's test directory</p>
              <p># 3. Run with your framework:</p>
              <p className="mt-2 text-green-400">$ pytest tests/         # Python</p>
              <p className="text-green-400">$ mvn test              # Java/Maven</p>
              <p className="text-green-400">$ dotnet test           # C#/.NET</p>
              <p className="text-green-400">$ npm test              # Node.js</p>
            </div>
            <div className="mt-4 flex items-center justify-end gap-2">
              <button
                onClick={onDownloadAll}
                className="flex items-center gap-1.5 rounded-lg bg-glow px-4 py-2 text-xs font-medium text-white hover:bg-glow-dim"
              >
                <Download className="h-3 w-3" />
                Download Tests
              </button>
              <button
                onClick={() => setShowRunModal(false)}
                className="rounded-lg border border-void-surface px-4 py-2 text-xs text-text-muted hover:text-text"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Test code viewer ────────────────────────────────────────────────

function TestCodeView({ file }: { file: MigrationFile }) {
  const langMap: Record<string, string> = {
    python: 'python',
    java: 'java',
    javascript: 'javascript',
    typescript: 'typescript',
    csharp: 'csharp',
    go: 'go',
    rust: 'rust',
  };
  const lang = langMap[file.language] ?? file.language;

  return (
    <Highlight theme={themes.nightOwl} code={file.content} language={lang}>
      {({ style, tokens, getLineProps, getTokenProps }) => (
        <pre
          className="overflow-x-auto p-4 text-xs leading-relaxed"
          style={{ ...style, background: 'transparent' }}
        >
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })}>
              <span className="mr-4 inline-block w-8 text-right text-text-dim select-none">
                {i + 1}
              </span>
              {line.map((token, key) => (
                <span key={key} {...getTokenProps({ token })} />
              ))}
            </div>
          ))}
        </pre>
      )}
    </Highlight>
  );
}
