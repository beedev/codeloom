/**
 * DiagramsSection -- Project Wiki diagrams panel.
 *
 * Lets users pick an MVP, select a diagram type (structural / behavioral),
 * and renders the SVG output from the backend PlantUML pipeline.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Loader2,
  FileImage,
  Clock,
} from 'lucide-react';
import type {
  MigrationPlan,
  DiagramAvailability,
  DiagramResponse,
  DiagramType,
  FunctionalMvpSummary,
} from '../../types/index.ts';

/* ── Props ───────────────────────────────────────────────────────── */

interface Props {
  migrationPlan: MigrationPlan | null;
  loading?: boolean;
  onLoad: () => Promise<void>;
  onLoadAvailability: (mvpId: number) => Promise<DiagramAvailability[] | null>;
  onLoadDiagram: (mvpId: number, diagramType: string) => Promise<DiagramResponse | null>;
}

/* ── Diagram categories ──────────────────────────────────────────── */

const STRUCTURAL_TYPES: Array<{ type: DiagramType; label: string }> = [
  { type: 'class', label: 'Class' },
  { type: 'package', label: 'Package' },
  { type: 'component', label: 'Component' },
];

const BEHAVIORAL_TYPES: Array<{ type: DiagramType; label: string }> = [
  { type: 'sequence', label: 'Sequence' },
  { type: 'usecase', label: 'Use Case' },
  { type: 'activity', label: 'Activity' },
  { type: 'deployment', label: 'Deployment' },
];

/* ── MVP pill selector ───────────────────────────────────────────── */

function MvpPillBar({
  mvps,
  selectedId,
  onSelect,
}: {
  mvps: FunctionalMvpSummary[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {mvps.map((mvp) => {
        const active = selectedId === mvp.mvp_id;
        return (
          <button
            key={mvp.mvp_id}
            onClick={() => onSelect(mvp.mvp_id)}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
              active
                ? 'bg-glow/10 text-glow border-glow/20'
                : 'border-void-surface text-text-muted hover:border-void-surface hover:text-text'
            }`}
          >
            {mvp.name}
          </button>
        );
      })}
    </div>
  );
}

/* ── Diagram type tabs ───────────────────────────────────────────── */

function DiagramTypeTabs({
  availability,
  selectedType,
  onSelect,
}: {
  availability: DiagramAvailability[];
  selectedType: DiagramType | null;
  onSelect: (type: DiagramType) => void;
}) {
  const availableTypes = new Set(availability.map((a) => a.diagram_type));

  function renderGroup(
    label: string,
    types: Array<{ type: DiagramType; label: string }>,
  ) {
    return (
      <div className="space-y-1.5">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-text-dim">
          {label}
        </p>
        <div className="flex flex-wrap gap-1.5">
          {types.map(({ type, label: typeLabel }) => {
            const isAvailable = availableTypes.has(type);
            const active = selectedType === type;
            return (
              <button
                key={type}
                onClick={() => onSelect(type)}
                disabled={!isAvailable}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                  active
                    ? 'bg-glow/10 text-glow border-glow/20'
                    : isAvailable
                      ? 'border-void-surface text-text-muted hover:text-text'
                      : 'border-void-surface/50 text-text-dim/50 cursor-not-allowed'
                }`}
              >
                {typeLabel}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-6 flex-wrap">
      {renderGroup('Structural', STRUCTURAL_TYPES)}
      {renderGroup('Behavioral', BEHAVIORAL_TYPES)}
    </div>
  );
}

/* ── Diagram viewer ──────────────────────────────────────────────── */

function DiagramViewer({
  diagram,
  loading,
}: {
  diagram: DiagramResponse | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-16 text-text-muted">
        <Loader2 className="h-5 w-5 animate-spin" />
        <span className="text-xs">Generating diagram...</span>
      </div>
    );
  }

  if (!diagram) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-16 text-text-dim">
        <FileImage className="h-6 w-6" />
        <span className="text-xs">Select a diagram type to view.</span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-text">{diagram.title}</h3>

      {diagram.svg ? (
        <div className="bg-white rounded-lg p-4 overflow-auto max-h-[600px]">
          <div dangerouslySetInnerHTML={{ __html: diagram.svg }} />
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center gap-3 py-16 text-text-dim rounded-lg border border-void-surface">
          <FileImage className="h-6 w-6" />
          <span className="text-xs">No SVG available for this diagram.</span>
        </div>
      )}

      {diagram.generated_at && (
        <div className="flex items-center gap-1.5 text-[10px] text-text-dim">
          <Clock className="h-3 w-3" />
          <span>Generated {new Date(diagram.generated_at).toLocaleString()}</span>
        </div>
      )}
    </div>
  );
}

/* ── Main component ──────────────────────────────────────────────── */

export function DiagramsSection({
  migrationPlan,
  loading,
  onLoad,
  onLoadAvailability,
  onLoadDiagram,
}: Props) {
  const [selectedMvpId, setSelectedMvpId] = useState<number | null>(null);
  const [availability, setAvailability] = useState<DiagramAvailability[] | null>(null);
  const [loadingAvailability, setLoadingAvailability] = useState(false);
  const [selectedType, setSelectedType] = useState<DiagramType | null>(null);
  const [diagram, setDiagram] = useState<DiagramResponse | null>(null);
  const [loadingDiagram, setLoadingDiagram] = useState(false);

  useEffect(() => {
    if (migrationPlan === null) {
      void onLoad();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* Load availability when MVP changes */
  const handleSelectMvp = useCallback(
    async (mvpId: number) => {
      setSelectedMvpId(mvpId);
      setSelectedType(null);
      setDiagram(null);
      setAvailability(null);
      setLoadingAvailability(true);
      const avail = await onLoadAvailability(mvpId);
      setAvailability(avail);
      setLoadingAvailability(false);
    },
    [onLoadAvailability],
  );

  /* Load diagram when type changes */
  const handleSelectType = useCallback(
    async (type: DiagramType) => {
      if (selectedMvpId === null) return;
      setSelectedType(type);
      setDiagram(null);
      setLoadingDiagram(true);
      const result = await onLoadDiagram(selectedMvpId, type);
      setDiagram(result);
      setLoadingDiagram(false);
    },
    [selectedMvpId, onLoadDiagram],
  );

  /* Loading */
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-muted">
        <Loader2 className="h-6 w-6 animate-spin" />
        <span className="text-sm">Loading diagrams data...</span>
      </div>
    );
  }

  /* Empty state */
  if (!migrationPlan || migrationPlan.mvps.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-32 text-text-dim">
        <FileImage className="h-8 w-8" />
        <span className="text-sm">
          {migrationPlan
            ? 'No MVPs available for diagram generation.'
            : 'No migration plan available.'}
        </span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-lg font-semibold text-text">Diagrams</h2>

      {/* MVP selector */}
      <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
        <h3 className="text-sm font-semibold text-text mb-3">Select MVP</h3>
        <MvpPillBar
          mvps={migrationPlan.mvps}
          selectedId={selectedMvpId}
          onSelect={handleSelectMvp}
        />
      </div>

      {/* Diagram type tabs */}
      {selectedMvpId !== null && (
        <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
          <h3 className="text-sm font-semibold text-text mb-3">Diagram Type</h3>
          {loadingAvailability ? (
            <div className="flex items-center gap-2 py-4 text-text-muted">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-xs">Loading available diagrams...</span>
            </div>
          ) : availability ? (
            <DiagramTypeTabs
              availability={availability}
              selectedType={selectedType}
              onSelect={handleSelectType}
            />
          ) : (
            <p className="text-xs text-text-dim">No diagram types available for this MVP.</p>
          )}
        </div>
      )}

      {/* Diagram display */}
      {selectedMvpId !== null && selectedType !== null && (
        <div className="rounded-xl border border-void-surface bg-void-light/30 p-5">
          <DiagramViewer diagram={diagram} loading={loadingDiagram} />
        </div>
      )}
    </div>
  );
}
