/**
 * DiagramViewer — renders pre-rendered SVG from backend with zoom/pan controls.
 *
 * Receives SVG string from DiagramService and renders it with:
 * - Zoom in/out (CSS transform scale)
 * - Pan via overflow scroll
 * - Fullscreen toggle
 * - Download SVG button
 */

import { useState, useRef, useCallback } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Minimize2, Download, RotateCcw } from 'lucide-react';

interface DiagramViewerProps {
  svg: string;
  title: string;
  puml?: string;
  onDownload?: () => void;
}

const ZOOM_STEP = 0.2;
const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3.0;

export function DiagramViewer({ svg, title, onDownload }: DiagramViewerProps) {
  const [zoom, setZoom] = useState(1.0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleZoomIn = useCallback(() => {
    setZoom((z) => Math.min(z + ZOOM_STEP, MAX_ZOOM));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom((z) => Math.max(z - ZOOM_STEP, MIN_ZOOM));
  }, []);

  const handleReset = useCallback(() => {
    setZoom(1.0);
  }, []);

  const handleFullscreen = useCallback(() => {
    setIsFullscreen((f) => !f);
  }, []);

  const handleDownload = useCallback(() => {
    if (onDownload) {
      onDownload();
      return;
    }
    // Default: download SVG as file
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.replace(/[^a-zA-Z0-9]/g, '_')}.svg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [svg, title, onDownload]);

  const containerClass = isFullscreen
    ? 'fixed inset-0 z-50 bg-void flex flex-col'
    : 'flex flex-col';

  return (
    <div className={containerClass}>
      {/* Toolbar */}
      <div className="flex items-center gap-1.5 border-b border-void-surface/50 bg-void-light/30 px-3 py-1.5">
        <span className="mr-auto text-[11px] font-medium text-text-muted truncate">{title}</span>

        <button
          onClick={handleZoomOut}
          className="rounded p-1 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
          title="Zoom out"
        >
          <ZoomOut className="h-3.5 w-3.5" />
        </button>
        <span className="min-w-[3rem] text-center text-[10px] text-text-muted">
          {Math.round(zoom * 100)}%
        </span>
        <button
          onClick={handleZoomIn}
          className="rounded p-1 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
          title="Zoom in"
        >
          <ZoomIn className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={handleReset}
          className="rounded p-1 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
          title="Reset zoom"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>

        <div className="mx-1 h-3 w-px bg-void-surface" />

        <button
          onClick={handleDownload}
          className="rounded p-1 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
          title="Download SVG"
        >
          <Download className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={handleFullscreen}
          className="rounded p-1 text-text-dim hover:bg-void-surface hover:text-text transition-colors"
          title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
        >
          {isFullscreen
            ? <Minimize2 className="h-3.5 w-3.5" />
            : <Maximize2 className="h-3.5 w-3.5" />
          }
        </button>
      </div>

      {/* SVG viewport */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto bg-white rounded-b"
        style={{ minHeight: isFullscreen ? undefined : '400px', maxHeight: isFullscreen ? undefined : '70vh' }}
      >
        <div
          className="inline-block origin-top-left p-4 diagram-svg [&_svg]:max-w-none [&_svg]:h-auto"
          style={{ transform: `scale(${zoom})` }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </div>
    </div>
  );
}
