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
  Eye,
  EyeOff,
  ArrowLeft,
  Layers,
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
// Constants
// ---------------------------------------------------------------------------

const NODE_COLORS: Record<string, string> = {
  function:         '#00bfff',  // neon sky blue
  method:           '#7b68ee',  // neon medium slate
  class:            '#ffab00',  // neon amber
  interface:        '#e040fb',  // neon purple-pink
  type_alias:       '#d500f9',  // neon violet
  module:           '#00e676',  // neon green
  constructor:      '#00e5ff',  // neon cyan
  property:         '#76ff03',  // neon lime
  stored_procedure: '#ff6d00',  // neon orange
  sql_function:     '#ff9100',  // neon deep orange
  step:             '#ff4081',  // neon pink
  paragraph:        '#40c4ff',  // neon light blue
  section:          '#448aff',  // neon blue
  program:          '#ff1744',  // neon red
  division:         '#ea80fc',  // neon light purple
  job:              '#ffea00',  // neon yellow
  proc_step:        '#ff5252',  // neon coral
  struts_action:    '#ff4081',  // neon rose
  struts_form:      '#ff80ab',  // neon light rose
  jsp_page:         '#1de9b6',  // neon teal
  record:           '#b2ff59',  // neon lime green
  struct:           '#e040fb',  // neon fuchsia
  enum:             '#18ffff',  // neon aqua
  copybook:         '#ffd740',  // neon gold
};

const EDGE_COLORS: Record<string, string> = {
  calls:      '#00e5ff',  // electric cyan
  contains:   '#ff00e5',  // hot magenta
  inherits:   '#ff9100',  // neon orange
  imports:    '#39ff14',  // neon green
  implements: '#ffea00',  // electric yellow
  overrides:  '#ff3d71',  // neon coral-red
  type_dep:   '#b388ff',  // bright violet
  calls_sp:   '#ffd180',  // warm neon peach
  data_flow:  '#18ffff',  // bright aqua
};

const EDGE_LABELS: Record<string, string> = {
  calls: 'Calls',
  contains: 'Contains',
  inherits: 'Inherits',
  imports: 'Imports',
  implements: 'Implements',
  overrides: 'Overrides',
  type_dep: 'Type Dep',
  calls_sp: 'Calls SP',
  data_flow: 'Data Flow',
};

const EDGE_DESCRIPTIONS: Record<string, { outgoing: string; incoming: string }> = {
  calls: { outgoing: 'Calls', incoming: 'Called by' },
  contains: { outgoing: 'Contains', incoming: 'Contained in' },
  inherits: { outgoing: 'Extends', incoming: 'Extended by' },
  imports: { outgoing: 'Imports', incoming: 'Imported by' },
  implements: { outgoing: 'Implements', incoming: 'Implemented by' },
  overrides: { outgoing: 'Overrides', incoming: 'Overridden by' },
  type_dep: { outgoing: 'Uses type', incoming: 'Type used by' },
  calls_sp: { outgoing: 'Calls SP', incoming: 'SP called by' },
  data_flow: { outgoing: 'Writes to', incoming: 'Reads from' },
};

const LANG_MAP: Record<string, string> = {
  python: 'python',
  javascript: 'javascript',
  typescript: 'typescript',
  java: 'java',
  csharp: 'csharp',
};

const ALL_EDGE_TYPES = Object.keys(EDGE_LABELS);

// Container unit types — shown at Level 1 (top-level view)
const CONTAINER_TYPES = new Set([
  'program', 'copybook', 'class', 'interface', 'module',
  'stored_procedure', 'sql_function', 'job',
  'struts_action', 'struts_form', 'jsp_page',
  'record', 'struct', 'enum',
]);

// Dimmed color for orphan (isolated) nodes
const ORPHAN_OPACITY = '50';  // hex alpha suffix for ~31% opacity

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getNodeColor(type: string): string {
  return NODE_COLORS[type] ?? '#6b7280';
}

function getEdgeColor(type: string): string {
  return EDGE_COLORS[type] ?? '#374151';
}

function getNodeVal(unitType: string, isDrilled: boolean): number {
  if (!isDrilled) {
    // Level 1: moderate size, let labels do the work
    if (unitType === 'program' || unitType === 'class') return 2;
    if (unitType === 'module' || unitType === 'interface' || unitType === 'job') return 1.5;
    return 1;
  }
  // Level 2
  if (unitType === 'class' || unitType === 'program') return 2;
  return 1;
}

function getLinkId(link: GraphLink): { src: string; tgt: string } {
  return {
    src: typeof link.source === 'string' ? link.source : link.source.id,
    tgt: typeof link.target === 'string' ? link.target : link.target.id,
  };
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
    new Set(ALL_EDGE_TYPES),
  );
  const [searchQuery, setSearchQuery] = useState('');
  const [showOrphans, setShowOrphans] = useState(false);

  // Selection — ref mirrors state so ForceGraph callbacks stay stable
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const selectedIdRef = useRef<string | null>(null);
  const [unitDetail, setUnitDetail] = useState<UnitDetail | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);

  // Drill-down: null = Level 1 (containers only), string = Level 2 (children of this node)
  const [drillTarget, setDrillTarget] = useState<GraphNode | null>(null);

  // Dimensions
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

  // ------ Effects ------

  // Use a callback ref to measure as soon as the container mounts.
  // This handles the case where the Graph tab is activated after initial page load.
  const measuredRef = useRef(false);
  const setContainerRef = useCallback((node: HTMLDivElement | null) => {
    containerRef.current = node;
    if (node && !measuredRef.current) {
      const rect = node.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setDimensions({ width: rect.width, height: rect.height });
        measuredRef.current = true;
      }
    }
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) {
        setDimensions({ width, height });
      }
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, [dimensions]); // re-run when dimensions first set (container mounted)

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
        setGraphData({ nodes: fullGraph.nodes, links: fullGraph.links });
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

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && selectedNode) closePanel();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedNode]);

  // Configure forces for good spread — only when data or drill changes
  useEffect(() => {
    const fg = graphRef.current;
    if (!fg || !graphData) return;
    fg.d3Force('charge')?.strength(-500).distanceMax(1500);
    fg.d3Force('link')?.distance(150).strength(0.05);
    fg.d3Force('center')?.strength(0.02);
    fg.d3ReheatSimulation();
  }, [graphData, drillTarget]);

  // Re-fit graph when panel opens/closes so nodes fill the available space
  const panelOpenRef = useRef(false);
  useEffect(() => {
    const isOpen = selectedNode != null;
    if (isOpen !== panelOpenRef.current) {
      panelOpenRef.current = isOpen;
      if (graphRef.current && dimensions && dimensions.width > 0) {
        setTimeout(() => graphRef.current?.zoomToFit(400, 60), 50);
      }
    }
  }, [selectedNode, dimensions]);

  // ------ Filtered graph data ------

  const { filteredData, orphanIds } = useMemo(() => {
    if (!graphData) return { filteredData: { nodes: [], links: [] }, orphanIds: new Set<string>() };

    let baseNodes = graphData.nodes;
    let baseLinks = graphData.links;

    // ── Drill-down filtering ──
    if (drillTarget) {
      // Level 2: show the drilled container + its direct children (paragraphs/methods)
      const childIds = new Set<string>();
      childIds.add(drillTarget.id);
      for (const l of graphData.links) {
        if (l.edge_type === 'contains') {
          const { src, tgt } = getLinkId(l);
          if (src === drillTarget.id) childIds.add(tgt);
        }
      }

      // Build a lookup of node id → node for type checking
      const nodeById = new Map(graphData.nodes.map((n) => [n.id, n]));

      // Also include OTHER container-level nodes that have edges to/from
      // the drilled program or its children (shows inter-program relationships)
      const expandedIds = new Set(childIds);
      for (const l of graphData.links) {
        if (l.edge_type === 'contains') continue;
        const { src, tgt } = getLinkId(l);
        if (childIds.has(src)) {
          const targetNode = nodeById.get(tgt);
          // Include the target if it's a container (other program/class)
          // or if it belongs to the drilled program (internal edge)
          if (targetNode && (CONTAINER_TYPES.has(targetNode.unit_type) || childIds.has(tgt))) {
            expandedIds.add(tgt);
          }
        }
        if (childIds.has(tgt)) {
          const sourceNode = nodeById.get(src);
          if (sourceNode && (CONTAINER_TYPES.has(sourceNode.unit_type) || childIds.has(src))) {
            expandedIds.add(src);
          }
        }
      }
      baseNodes = graphData.nodes.filter((n) => expandedIds.has(n.id));
      baseLinks = graphData.links.filter((l) => {
        const { src, tgt } = getLinkId(l);
        return expandedIds.has(src) && expandedIds.has(tgt);
      });
    } else {
      // Level 1: show only container-type nodes
      const containerIds = new Set(
        graphData.nodes.filter((n) => CONTAINER_TYPES.has(n.unit_type)).map((n) => n.id),
      );
      baseNodes = graphData.nodes.filter((n) => containerIds.has(n.id));
      // Only show edges between containers (exclude contains edges at this level)
      baseLinks = graphData.links.filter((l) => {
        if (l.edge_type === 'contains') return false;
        const { src, tgt } = getLinkId(l);
        return containerIds.has(src) && containerIds.has(tgt);
      });
    }

    const filteredLinks = baseLinks.filter((l) =>
      enabledEdgeTypes.has(l.edge_type),
    );

    // Nodes connected by at least one visible edge
    const connectedIds = new Set<string>();
    for (const l of filteredLinks) {
      const { src, tgt } = getLinkId(l);
      connectedIds.add(src);
      connectedIds.add(tgt);
    }

    // Identify orphan nodes (not connected by any visible edge)
    const orphans = new Set<string>();
    for (const n of baseNodes) {
      if (!connectedIds.has(n.id)) orphans.add(n.id);
    }

    // Include connected nodes + optionally orphans
    let filteredNodes = showOrphans
      ? baseNodes
      : baseNodes.filter((n) => connectedIds.has(n.id));

    // Search filter: matched nodes + immediate neighbors
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchedIds = new Set(
        filteredNodes
          .filter((n) =>
            n.name.toLowerCase().includes(q) ||
            (n.qualified_name ?? '').toLowerCase().includes(q),
          )
          .map((n) => n.id),
      );
      const neighborIds = new Set(matchedIds);
      for (const l of filteredLinks) {
        const { src, tgt } = getLinkId(l);
        if (matchedIds.has(src)) neighborIds.add(tgt);
        if (matchedIds.has(tgt)) neighborIds.add(src);
      }
      filteredNodes = filteredNodes.filter((n) => neighborIds.has(n.id));
    }

    return {
      filteredData: { nodes: filteredNodes, links: filteredLinks },
      orphanIds: orphans,
    };
  }, [graphData, enabledEdgeTypes, searchQuery, showOrphans, drillTarget]);

  // ------ Callbacks ------

  const selectNode = useCallback((node: any) => {
    if (!node) return;
    const gn = node as GraphNode;
    // At Level 1, clicking a container drills into it
    if (!drillTarget && CONTAINER_TYPES.has(gn.unit_type)) {
      console.log('[Graph] Drilling into:', gn.name, gn.unit_type);
      setDrillTarget(gn);
      setSelectedNode(null);
      selectedIdRef.current = null;
      setTimeout(() => graphRef.current?.zoomToFit(400, 40), 300);
      return;
    }
    console.log('[Graph] Selecting:', gn.name, gn.unit_type, 'drillTarget:', drillTarget?.name ?? 'null');
    selectedIdRef.current = node.id;
    setSelectedNode(gn);
  }, [drillTarget]);


  const handleDrillBack = useCallback(() => {
    setDrillTarget(null);
    setSelectedNode(null);
    selectedIdRef.current = null;
    setTimeout(() => graphRef.current?.zoomToFit(400, 40), 300);
  }, []);

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

  // ------ Node rendering callbacks ------

  /**
   * Node color: white if selected, dimmed if orphan, type color otherwise.
   *
   * IMPORTANT: We intentionally do NOT use nodeCanvasObject / nodeCanvasObjectMode.
   * force-graph v1.29.1 binds nodeCanvasObject to BOTH the display canvas AND
   * the shadow (hit-detection) canvas via `bindBoth`. Any custom drawing with
   * colors other than the node's __indexColor corrupts the shadow canvas and
   * breaks click/hover detection for nearby nodes. The library's default
   * rendering is the only reliable path for hit detection.
   */
  const getNodeDisplayColor = useCallback((node: any) => {
    if (selectedIdRef.current === node.id) return '#ffffff';
    const base = getNodeColor(node.unit_type);
    if (orphanIds.has(node.id)) return base + ORPHAN_OPACITY;
    return base;
  }, [orphanIds]);

  /**
   * Fallback click handler: when the library's shadow canvas misses a node,
   * onBackgroundClick fires. We manually check if any node is within click
   * radius using graph-space coordinates.
   */
  const handleBackgroundClick = useCallback((event: MouseEvent) => {
    if (!graphRef.current) return;
    const { x, y } = graphRef.current.screen2GraphCoords(event.offsetX, event.offsetY);

    let closest: GraphNode | null = null;
    let closestDist = Infinity;

    for (const node of filteredData.nodes) {
      const nx = (node as any).x;
      const ny = (node as any).y;
      if (nx == null || ny == null) continue;
      const dx = nx - x;
      const dy = ny - y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      // Match the library's default circle radius: nodeRelSize * sqrt(nodeVal)
      const r = 6 * Math.sqrt(getNodeVal(node.unit_type, false));
      if (dist <= r + 3 && dist < closestDist) {
        closest = node;
        closestDist = dist;
      }
    }

    if (closest) {
      selectNode(closest);
    } else {
      // Actual background click — deselect
      closePanel();
    }
  }, [filteredData, selectNode, closePanel]);

  // ------ Render: not-ready states ------

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
  const orphanCount = orphanIds.size;
  const PANEL_WIDTH = 360;
  const graphWidth = dimensions ? (panelOpen ? dimensions.width - PANEL_WIDTH : dimensions.width) : 0;
  const graphHeight = dimensions?.height ?? 0;

  // ------ Render: main ------

  return (
    <div className="relative h-full graph-viewer-root" ref={setContainerRef}>
      {/* Drill-down breadcrumb bar */}
      {drillTarget && (
        <div className="absolute left-0 right-0 top-0 z-30 flex items-center gap-2 border-b border-void-surface bg-void-light/95 px-4 py-2 backdrop-blur-sm">
          <button
            onClick={handleDrillBack}
            className="flex items-center gap-1.5 rounded-md bg-void-surface px-2.5 py-1 text-xs text-text-muted hover:bg-glow/20 hover:text-text"
          >
            <ArrowLeft className="h-3 w-3" />
            All Programs
          </button>
          <ChevronRight className="h-3 w-3 text-text-dim/40" />
          <span className="flex items-center gap-1.5 text-xs font-medium text-text">
            <Layers className="h-3 w-3 text-glow" />
            {drillTarget.name}
          </span>
          <span className="text-[10px] text-text-dim">
            ({filteredData.nodes.length} units, {filteredData.links.length} edges)
          </span>
        </div>
      )}

      {/* Toolbar */}
      <div className={`absolute left-3 z-10 flex flex-col gap-2 ${drillTarget ? 'top-12' : 'top-3'}`}>
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
          {ALL_EDGE_TYPES.map((type) => {
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

        {/* Orphan node toggle */}
        {orphanCount > 0 && (
          <button
            onClick={() => setShowOrphans(!showOrphans)}
            className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[10px] font-medium transition ${
              showOrphans
                ? 'bg-void-surface text-text-dim'
                : 'bg-void-light/60 text-text-dim/50'
            }`}
          >
            {showOrphans ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
            {orphanCount} isolated nodes
          </button>
        )}
      </div>

      {/* Zoom controls */}
      <div className={`absolute bottom-3 z-10 flex flex-col gap-1 ${panelOpen ? 'right-[375px]' : 'right-3'}`}>
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

      {/* Stats */}
      <div className="absolute bottom-3 left-3 z-10 rounded-lg bg-void-light/90 px-2.5 py-1 text-[10px] text-text-dim backdrop-blur-sm">
        {filteredData.nodes.length} nodes / {filteredData.links.length} edges
      </div>

      {/* Force graph
        * IMPORTANT: No nodeCanvasObject / nodeCanvasObjectMode.
        * force-graph v1.29.1 binds these to BOTH display and shadow canvases
        * (via bindBoth in force-graph.js). Any custom drawing on the shadow
        * canvas with non-__indexColor colors corrupts hit detection.
        * We use only the library's default rendering (nodeColor + nodeRelSize +
        * nodeVal) and add onBackgroundClick as a manual fallback for any nodes
        * the shadow canvas still misses.
        */}
      {dimensions && <ForceGraph2D
        ref={graphRef}
        graphData={filteredData}
        width={graphWidth}
        height={graphHeight}
        backgroundColor="#0f172a"
        nodeLabel={(node: any) => {
          const label = node.qualified_name ?? node.name;
          const prefix = node.language
            ? `${node.unit_type} [${node.language}]`
            : node.unit_type;
          return orphanIds.has(node.id)
            ? `${prefix}: ${label} (isolated — no visible edges)`
            : `${prefix}: ${label}`;
        }}
        nodeColor={getNodeDisplayColor}
        nodeRelSize={4}
        nodeVal={(node: any) => getNodeVal(node.unit_type, drillTarget != null)}
        nodeCanvasObjectMode={() => 'after'}
        nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
          const label = node.name ?? '';
          const fontSize = Math.max(10 / globalScale, 2);
          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
          ctx.fillText(label, node.x, node.y + 5 / globalScale);
        }}
        linkColor={(link: any) => getEdgeColor(link.edge_type)}
        linkLabel={(link: any) => EDGE_LABELS[link.edge_type] ?? link.edge_type}
        linkWidth={0.8}
        linkCurvature={0.2}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={0.9}
        onNodeClick={selectNode}
        onNodeDragEnd={selectNode}
        onBackgroundClick={handleBackgroundClick as any}
        onNodeHover={(node: any) => {
          const el = containerRef.current;
          if (el) el.style.cursor = node ? 'pointer' : 'default';
        }}
        onEngineStop={() => graphRef.current?.zoomToFit(400, 40)}
        cooldownTicks={200}
        d3VelocityDecay={0.3}
        d3AlphaDecay={0.02}
        warmupTicks={50}
      />}

      {/* Detail panel */}
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
// DetailPanel
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
                backgroundColor: getNodeColor(node.unit_type) + '20',
                color: getNodeColor(node.unit_type),
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

      {/* Body */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="h-5 w-5 animate-spin text-text-dim" />
          </div>
        ) : detail ? (
          <div className="space-y-0">
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

            {detail.docstring && (
              <div className="border-b border-void-surface px-4 py-2.5">
                <p className="whitespace-pre-wrap text-[11px] leading-relaxed text-text-muted italic">
                  {detail.docstring.length > 300
                    ? detail.docstring.slice(0, 300) + '...'
                    : detail.docstring}
                </p>
              </div>
            )}

            {detail.source && (
              <div className="border-b border-void-surface">
                <button
                  onClick={() => setShowSource(!showSource)}
                  className="flex w-full items-center gap-2 px-4 py-2 text-[11px] font-medium text-text-muted hover:bg-void-surface/50 hover:text-text-dim"
                >
                  {showSource ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
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
// RelationshipSection
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
// EdgeTypeGroup
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
        {expanded ? <ChevronDown className="h-2.5 w-2.5" /> : <ChevronRight className="h-2.5 w-2.5" />}
        <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
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
