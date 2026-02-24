/**
 * WikiSidebar — left navigation for the Project Wiki.
 *
 * Shows 7 sections with icons and data-availability indicators.
 */

import {
  LayoutDashboard,
  Network,
  Brain,
  ArrowRightLeft,
  Boxes,
  FileImage,
  FileCode,
} from 'lucide-react';
import type { ProjectAnalytics } from '../../types/index.ts';

export type WikiSection =
  | 'overview'
  | 'architecture'
  | 'understanding'
  | 'migration'
  | 'mvp-catalog'
  | 'diagrams'
  | 'generated-code';

const SECTIONS: Array<{ id: WikiSection; label: string; icon: typeof LayoutDashboard }> = [
  { id: 'overview',       label: 'Overview',       icon: LayoutDashboard },
  { id: 'architecture',   label: 'Architecture',   icon: Network },
  { id: 'understanding',  label: 'Understanding',  icon: Brain },
  { id: 'migration',      label: 'Migration',      icon: ArrowRightLeft },
  { id: 'mvp-catalog',    label: 'MVP Catalog',    icon: Boxes },
  { id: 'diagrams',       label: 'Diagrams',       icon: FileImage },
  { id: 'generated-code', label: 'Generated Code', icon: FileCode },
];

function hasData(section: WikiSection, analytics: ProjectAnalytics | null): boolean {
  if (!analytics) return section === 'overview';
  switch (section) {
    case 'overview': return true;
    case 'architecture': return Object.keys(analytics.code_breakdown.edges_by_type).length > 0;
    case 'understanding': return analytics.understanding.analyses_count > 0;
    case 'migration': return analytics.migration.plan_count > 0;
    case 'mvp-catalog': return analytics.migration.active_plan != null;
    case 'diagrams': return analytics.migration.active_plan != null;
    case 'generated-code': return analytics.migration.active_plan != null;
    default: return false;
  }
}

interface WikiSidebarProps {
  activeSection: WikiSection;
  onSelectSection: (section: WikiSection) => void;
  analytics: ProjectAnalytics | null;
}

export function WikiSidebar({ activeSection, onSelectSection, analytics }: WikiSidebarProps) {
  return (
    <div className="w-60 shrink-0 overflow-y-auto border-r border-void-surface bg-void-light/30">
      <div className="p-3">
        <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-text-dim">
          Wiki Sections
        </p>
        <nav className="space-y-0.5">
          {SECTIONS.map(({ id, label, icon: Icon }) => {
            const active = activeSection === id;
            const available = hasData(id, analytics);
            return (
              <button
                key={id}
                onClick={() => onSelectSection(id)}
                className={`flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-xs transition-colors ${
                  active
                    ? 'bg-glow/10 text-glow'
                    : 'text-text-muted hover:bg-void-surface hover:text-text'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span className="flex-1 text-left">{label}</span>
                {available && (
                  <span className={`h-1.5 w-1.5 rounded-full ${active ? 'bg-glow' : 'bg-success'}`} />
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </div>
  );
}
