/**
 * GraphViewer — Force-directed ASG visualization.
 *
 * Renders code units as nodes and relationships as edges using
 * react-force-graph-2d. Supports filtering by edge type, node search,
 * and click-to-inspect with full unit detail panel.
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Highlight, themes } from 'prism-react-renderer';
import {
  Loader2,
  AlertCircle,
  Search,
  X,
  GitBranch,
  ArrowUpRight,
  ArrowDownLeft,
  Maximize2,
  ZoomIn,
  ZoomOut,
  FileCode2,
  Hash,
  Braces,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import * as api from '../services/api.ts';
import type { GraphOverview, UnitDetail, EdgeUnit } from '../types/index.ts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

interface GraphNode {
  id: string;
  name: string;
  qualified_name?: string;
  unit_type: string;
  language?: string;
  file_id?: string;
  // d3-force adds these at runtime
  x?: number;
  y?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  edge_type: string;
}

interface Props {
  projectId: string;
  asgStatus: string;
}

// ---------------------------------------------------------------------------
// Color palette
// ---------------------------------------------------------------------------

const NODE_COLORS: Record<string, string> = {
  function: '#60a5fa',   // blue-400
  method: '#818cf8',     // indigo-400
  class: '#f59e0b',      // amber-500
  interface: '#a78bfa',  // violet-400
  type_alias: '#c084fc', // purple-400
  module: '#34d399',     // emerald-400
};

const EDGE_COLORS: Record<string, string> = {
  calls: '#475569',      // slate-600
  contains: '#334155',   // slate-700
  inherits: '#b45309',   // amber-700
  imports: '#065f46',    // emerald-800
  implements: '#7c3aed', // violet-600
  overrides: '#be185d',  // pink-700
};

const EDGE_LABELS: Record<string, string> = {
  calls: 'Calls',
  contains: 'Contains',
  inherits: 'Inherits',
  imports: 'Imports',
  implements: 'Implements',
  overrides: 'Overrides',
};

const EDGE_DESCRIPTIONS: Record<string, { outgoing: string; incoming: string }> = {
  calls: { outgoing: 'Calls', incoming: 'Called by' },
  contains: { outgoing: 'Contains', incoming: 'Contained in' },
  inherits: { outgoing: 'Extends', incoming: 'Extended by' },
  imports: { outgoing: 'Imports', incoming: 'Imported by' },
  implements: { outgoing: 'Implements', incoming: 'Implemented by' },
  overrides: { outgoing: 'Overrides', incoming: 'Overridden by' },
};

const LANG_MAP: Record<string, string> = {
  python: 'python',
  javascript: 'javascript',
  typescript: 'typescript',
  java: 'java',
  csharp: 'csharp',
};

function nodeColor(type: string): string {
  return NODE_COLORS[type] ?? '#6b7280';
}

function edgeColor(type: string): string {
  return EDGE_COLORS[type] ?? '#374151';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function GraphViewer({ projectId, asgStatus }: Props) {
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Data
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [overview, setOverview] = useState<GraphOverview | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [enabledEdgeTypes, setEnabledEdgeTypes] = useState<Set<string>>(
    new Set(['calls', 'contains', 'inherits', 'imports', 'implements', 'overrides']),
  );
  const [searchQuery, setSearchQuery] = useState('');

  // Selection — ref mirrors state so ForceGraph callbacks stay stable
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const selectedIdRef = useRef<string | null>(null);
  const [unitDetail, setUnitDetail] = useState<UnitDetail | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);

  // Dimensions
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  // Load graph data
  useEffect(() => {
    if (asgStatus !== 'complete') {
      setIsLoading(false);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    setError(null);

    Promise.all([
      api.getFullGraph(projectId),
      api.getGraphOverview(projectId),
    ])
      .then(([fullGraph, ov]) => {
        if (cancelled) return;
        setGraphData({
          nodes: fullGraph.nodes,
          links: fullGraph.links,
        });
        setOverview(ov);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load graph');
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => { cancelled = true; };
  }, [projectId, asgStatus]);

  // Filtered graph data
  const filteredData = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] };

    const filteredLinks = graphData.links.filter((l) =>
      enabledEdgeTypes.has(l.edge_type),
    );

    // Only include nodes that are connected by visible edges
    const visibleNodeIds = new Set<string>();
    for (const l of filteredLinks) {
      const srcId = typeof l.source === 'string' ? l.source : l.source.id;
      const tgtId = typeof l.target === 'string' ? l.target : l.target.id;
      visibleNodeIds.add(srcId);
      visibleNodeIds.add(tgtId);
    }

    let filteredNodes = graphData.nodes.filter((n) => visibleNodeIds.has(n.id));

    // Apply search filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchedIds = new Set(
        filteredNodes
          .filter((n) => n.name.toLowerCase().includes(q) || (n.qualified_name ?? '').toLowerCase().includes(q))
          .map((n) => n.id),
      );
      // Show matched nodes + their immediate neighbors
      const neighborIds = new Set(matchedIds);
      for (const l of filteredLinks) {
        const srcId = typeof l.source === 'string' ? l.source : l.source.id;
        const tgtId = typeof l.target === 'string' ? l.target : l.target.id;
        if (matchedIds.has(srcId)) neighborIds.add(tgtId);
        if (matchedIds.has(tgtId)) neighborIds.add(srcId);
      }
      filteredNodes = filteredNodes.filter((n) => neighborIds.has(n.id));
    }

    return { nodes: filteredNodes, links: filteredLinks };
  }, [graphData, enabledEdgeTypes, searchQuery]);

  // Synchronous node click — NO async, NO Promise returned to the library
  const handleNodeClick = useCallback(
    (node: any) => {
      selectedIdRef.current = node.id;
      setSelectedNode(node as GraphNode);
    },
    [],
  );

  // Fetch unit detail when selection changes (decoupled from click event)
  useEffect(() => {
    if (!selectedNode) {
      setUnitDetail(null);
      return;
    }
    let cancelled = false;
    setIsLoadingDetail(true);
    setUnitDetail(null);

    api.getUnitDetail(projectId, selectedNode.id)
      .then((detail) => { if (!cancelled) setUnitDetail(detail); })
      .catch((err) => { if (!cancelled) console.warn('Failed to load unit detail:', selectedNode.id, err); })
      .finally(() => { if (!cancelled) setIsLoadingDetail(false); });

    return () => { cancelled = true; };
  }, [projectId, selectedNode]);

  // Navigate to a related unit from the detail panel
  const handleNavigateToUnit = useCallback(
    (eu: EdgeUnit) => {
      const graphNode = graphData?.nodes.find((n) => n.id === eu.unit_id);
      const target: GraphNode = graphNode ?? {
        id: eu.unit_id,
        name: eu.name,
        qualified_name: eu.qualified_name,
        unit_type: eu.unit_type,
        language: eu.language,
      };
      selectedIdRef.current = target.id;
      setSelectedNode(target);
      // Center the graph on the target node
      if (graphRef.current && graphNode?.x != null && graphNode?.y != null) {
        graphRef.current.centerAt(graphNode.x, graphNode.y, 400);
        graphRef.current.zoom(3, 400);
      }
    },
    [graphData],
  );

  const closePanel = useCallback(() => {
    selectedIdRef.current = null;
    setSelectedNode(null);
    setUnitDetail(null);
  }, []);

  const toggleEdgeType = useCallback((type: string) => {
    setEnabledEdgeTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }, []);

  const handleZoomToFit = useCallback(() => {
    graphRef.current?.zoomToFit(400, 40);
  }, []);

  // Escape key closes panel
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && selectedNode) closePanel();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedNode, closePanel]);

  // ---- Not ready states ----

  if (asgStatus !== 'complete') {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-text-muted">
        <GitBranch className="h-10 w-10 text-text-dim" />
        <p className="text-sm">ASG not built for this project</p>
        <p className="text-xs text-text-dim">
          Status: <span className="text-text-muted">{asgStatus}</span>
        </p>
        <p className="mt-2 text-xs text-text-dim">
          Ingest code or use the Build ASG button to generate the graph.
        </p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-text-dim" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center gap-2 text-danger">
        <AlertCircle className="h-5 w-5" />
        <span className="text-sm">{error}</span>
      </div>
    );
  }

  const panelOpen = selectedNode != null;

  return (
    <div className="relative h-full" ref={containerRef}>
      {/* Toolbar overlay */}
      <div className="absolute left-3 top-3 z-10 flex flex-col gap-2">
        {/* Search */}
        <div className="flex items-center gap-1.5 rounded-lg bg-void-light/90 px-2.5 py-1.5 backdrop-blur-sm">
          <Search className="h-3.5 w-3.5 text-text-dim" />
          <input
            type="text"
            placeholder="Search units..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-40 bg-transparent text-xs text-text-dim placeholder-text-dim outline-none"
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery('')}>
              <X className="h-3 w-3 text-text-dim hover:text-text-muted" />
            </button>
          )}
        </div>

        {/* Edge type toggles */}
        <div className="flex flex-wrap gap-1">
          {Object.keys(EDGE_LABELS).map((type) => {
            const active = enabledEdgeTypes.has(type);
            const count = overview?.edge_types[type] ?? 0;
            return (
              <button
                key={type}
                onClick={() => toggleEdgeType(type)}
                className={`rounded px-2 py-0.5 text-[10px] font-medium transition ${
                  active
                    ? 'bg-void-surface text-text-dim'
                    : 'bg-void-light/60 text-text-dim/50 line-through'
                }`}
              >
                {EDGE_LABELS[type]} ({count})
              </button>
            );
          })}
        </div>
      </div>

      {/* Zoom controls */}
      <div className="absolute bottom-3 right-3 z-10 flex flex-col gap-1">
        <button
          onClick={handleZoomToFit}
          className="rounded-lg bg-void-light/90 p-1.5 text-text-muted hover:text-text backdrop-blur-sm"
          title="Zoom to fit"
        >
          <Maximize2 className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => graphRef.current?.zoom(graphRef.current.zoom() * 1.3, 300)}
          className="rounded-lg bg-void-light/90 p-1.5 text-text-muted hover:text-text backdrop-blur-sm"
          title="Zoom in"
        >
          <ZoomIn className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => graphRef.current?.zoom(graphRef.current.zoom() * 0.7, 300)}
          className="rounded-lg bg-void-light/90 p-1.5 text-text-muted hover:text-text backdrop-blur-sm"
          title="Zoom out"
        >
          <ZoomOut className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Stats overlay */}
      <div className="absolute bottom-3 left-3 z-10 rounded-lg bg-void-light/90 px-2.5 py-1 text-[10px] text-text-dim backdrop-blur-sm">
        {filteredData.nodes.length} nodes / {filteredData.links.length} edges
      </div>

      {/* Force graph — always fills full container */}
      <ForceGraph2D
        ref={graphRef}
        graphData={filteredData}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor="#0f172a"
        nodeLabel={(node: any) => `${node.unit_type}: ${node.qualified_name ?? node.name}`}
        nodeColor={(node: any) => {
          // Use ref so this doesn't force ForceGraph re-processing
          if (selectedIdRef.current === node.id) return '#ffffff';
          return nodeColor(node.unit_type);
        }}
        nodeRelSize={5}
        nodeVal={(node: any) => {
          if (node.unit_type === 'class') return 3;
          if (node.unit_type === 'interface') return 2;
          return 1;
        }}
        linkColor={(link: any) => edgeColor(link.edge_type)}
        linkWidth={0.5}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        onNodeClick={handleNodeClick}
        onEngineStop={() => graphRef.current?.zoomToFit(400, 60)}
        cooldownTicks={100}
        d3VelocityDecay={0.3}
      />

      {/* Detail panel — absolute overlay on right */}
      {panelOpen && (
        <div className="absolute right-0 top-0 bottom-0 z-20 w-[360px]">
          <DetailPanel
            node={selectedNode!}
            detail={unitDetail}
            isLoading={isLoadingDetail}
            onClose={closePanel}
            onNavigate={handleNavigateToUnit}
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// DetailPanel — Rich unit detail sidebar
// ---------------------------------------------------------------------------

function DetailPanel({
  node,
  detail,
  isLoading,
  onClose,
  onNavigate,
}: {
  node: GraphNode;
  detail: UnitDetail | null;
  isLoading: boolean;
  onClose: () => void;
  onNavigate: (eu: EdgeUnit) => void;
}) {
  const [showSource, setShowSource] = useState(false);

  // Reset source collapse when detail changes
  useEffect(() => {
    setShowSource(false);
  }, [detail?.unit_id]);

  return (
    <div className="flex h-full flex-col overflow-hidden border-l border-void-surface bg-void shadow-xl">
      {/* Header */}
      <div className="border-b border-void-surface px-4 py-3">
        <div className="flex items-start justify-between">
          <div className="min-w-0 flex-1">
            <span
              className="inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
              style={{
                backgroundColor: nodeColor(node.unit_type) + '20',
                color: nodeColor(node.unit_type),
              }}
            >
              {node.unit_type}
            </span>
            <h3 className="mt-1.5 truncate text-sm font-semibold text-text">
              {node.name}
            </h3>
            {(detail?.qualified_name ?? node.qualified_name) && (
              <p className="mt-0.5 break-all text-[10px] text-text-dim">
                {detail?.qualified_name ?? node.qualified_name}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="ml-2 shrink-0 rounded p-0.5 text-text-dim hover:bg-void-surface hover:text-text-muted"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Body — scrollable */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
          </div>
        ) : detail ? (
          <div className="space-y-0">
            {/* File location */}
            {detail.file_path && (
              <div className="border-b border-void-surface px-4 py-2.5">
                <div className="flex items-start gap-2">
                  <FileCode2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-text-dim" />
                  <div className="min-w-0">
                    <p className="break-all text-[11px] text-text-muted">{detail.file_path}</p>
                    {detail.start_line != null && detail.end_line != null && (
                      <p className="mt-0.5 text-[10px] text-text-dim">
                        Lines {detail.start_line}–{detail.end_line} ({detail.line_count} lines)
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Signature */}
            {detail.signature && (
              <div className="border-b border-void-surface px-4 py-2.5">
                <div className="flex items-start gap-2">
                  <Braces className="mt-0.5 h-3.5 w-3.5 shrink-0 text-text-dim" />
                  <pre className="min-w-0 whitespace-pre-wrap break-all text-[11px] leading-relaxed text-glow-bright font-mono">
                    {detail.signature}
                  </pre>
                </div>
              </div>
            )}

            {/* Docstring */}
            {detail.docstring && (
              <div className="border-b border-void-surface px-4 py-2.5">
                <p className="whitespace-pre-wrap text-[11px] leading-relaxed text-text-muted italic">
                  {detail.docstring.length > 300
                    ? detail.docstring.slice(0, 300) + '...'
                    : detail.docstring}
                </p>
              </div>
            )}

            {/* Source code (collapsible) */}
            {detail.source && (
              <div className="border-b border-void-surface">
                <button
                  onClick={() => setShowSource(!showSource)}
                  className="flex w-full items-center gap-2 px-4 py-2 text-[11px] font-medium text-text-muted hover:bg-void-surface/50 hover:text-text-dim"
                >
                  {showSource ? (
                    <ChevronDown className="h-3 w-3" />
                  ) : (
                    <ChevronRight className="h-3 w-3" />
                  )}
                  Source Code
                  <span className="ml-auto text-[10px] text-text-dim">
                    {detail.line_count ?? '?'} lines
                  </span>
                </button>
                {showSource && (
                  <div className="max-h-[300px] overflow-auto bg-[#011627]">
                    <Highlight
                      theme={themes.nightOwl}
                      code={detail.source}
                      language={LANG_MAP[detail.language ?? ''] ?? 'python'}
                    >
                      {({ tokens, getLineProps, getTokenProps }) => (
                        <pre className="px-3 py-2 text-[10px] leading-4" style={{ background: 'transparent' }}>
                          {tokens.map((line, idx) => {
                            const lineNum = (detail.start_line ?? 1) + idx;
                            return (
                              <div key={idx} {...getLineProps({ line })} style={undefined} className="flex">
                                <span className="inline-block w-8 shrink-0 select-none pr-2 text-right text-text-dim">
                                  {lineNum}
                                </span>
                                <span className="flex-1 whitespace-pre">
                                  {line.map((token, key) => (
                                    <span key={key} {...getTokenProps({ token })} />
                                  ))}
                                </span>
                              </div>
                            );
                          })}
                        </pre>
                      )}
                    </Highlight>
                  </div>
                )}
              </div>
            )}

            {/* Relationships */}
            <RelationshipSection
              title="Outgoing"
              icon={<ArrowUpRight className="h-3 w-3" />}
              edgeGroups={detail.edges.outgoing}
              onNavigate={onNavigate}
            />
            <RelationshipSection
              title="Incoming"
              icon={<ArrowDownLeft className="h-3 w-3" />}
              edgeGroups={detail.edges.incoming}
              onNavigate={onNavigate}
            />
          </div>
        ) : (
          /* Fallback: show basic node info when detail fetch fails */
          <div className="px-4 py-4">
            <p className="text-[11px] text-text-dim">
              Could not load full details for this unit.
            </p>
            {node.language && (
              <p className="mt-2 text-[11px] text-text-dim">
                Language: <span className="text-text-muted">{node.language}</span>
              </p>
            )}
          </div>
        )}
      </div>

      {/* Legend footer */}
      <div className="border-t border-void-surface px-4 py-2">
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[9px]">
          {Object.entries(NODE_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1">
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-text-dim">{type}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// RelationshipSection — grouped edges
// ---------------------------------------------------------------------------

function RelationshipSection({
  title,
  icon,
  edgeGroups,
  onNavigate,
}: {
  title: string;
  icon: React.ReactNode;
  edgeGroups: Record<string, EdgeUnit[]>;
  onNavigate: (eu: EdgeUnit) => void;
}) {
  const types = Object.keys(edgeGroups);
  if (types.length === 0) return null;

  const totalCount = types.reduce((sum, t) => sum + edgeGroups[t].length, 0);

  return (
    <div className="border-b border-void-surface px-4 py-2.5">
      <h4 className="mb-2 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-text-dim">
        {icon}
        {title} ({totalCount})
      </h4>
      <div className="space-y-2">
        {types.map((edgeType) => {
          const units = edgeGroups[edgeType];
          const desc = EDGE_DESCRIPTIONS[edgeType];
          const label = title === 'Outgoing'
            ? (desc?.outgoing ?? edgeType)
            : (desc?.incoming ?? edgeType);

          return (
            <EdgeTypeGroup
              key={edgeType}
              edgeType={edgeType}
              label={label}
              units={units}
              onNavigate={onNavigate}
            />
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// EdgeTypeGroup — collapsible list of related units for one edge type
// ---------------------------------------------------------------------------

function EdgeTypeGroup({
  edgeType,
  label,
  units,
  onNavigate,
}: {
  edgeType: string;
  label: string;
  units: EdgeUnit[];
  onNavigate: (eu: EdgeUnit) => void;
}) {
  const [expanded, setExpanded] = useState(true);
  const color = EDGE_COLORS[edgeType] ?? '#6b7280';

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-1.5 text-[10px] text-text-muted hover:text-text-dim"
      >
        {expanded ? (
          <ChevronDown className="h-2.5 w-2.5" />
        ) : (
          <ChevronRight className="h-2.5 w-2.5" />
        )}
        <span
          className="inline-block h-1.5 w-1.5 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span className="font-medium">{label}</span>
        <span className="text-text-dim">({units.length})</span>
      </button>
      {expanded && (
        <ul className="mt-1 ml-4 space-y-0.5">
          {units.map((eu) => (
            <li key={eu.unit_id}>
              <button
                onClick={() => onNavigate(eu)}
                className="group flex w-full items-center gap-1.5 rounded px-1.5 py-1 text-left text-[11px] hover:bg-void-surface/70"
              >
                <Hash className="h-2.5 w-2.5 shrink-0 text-text-dim group-hover:text-text-muted" />
                <span className="truncate text-text-muted group-hover:text-text">
                  {eu.name}
                </span>
                <span className="ml-auto shrink-0 text-[9px] text-text-dim">
                  {eu.unit_type}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
