/**
 * ProjectWikiPage — aggregates all generated knowledge about a project.
 *
 * Left sidebar with 7 sections, main content area renders the active section.
 * Data is lazy-loaded per section via useWikiData hook.
 */

import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import { Layout } from '../components/Layout.tsx';
import { WikiSidebar } from '../components/wiki/WikiSidebar.tsx';
import type { WikiSection } from '../components/wiki/WikiSidebar.tsx';
import { useWikiData } from '../hooks/useWikiData.ts';
import { OverviewSection } from '../components/wiki/OverviewSection.tsx';
import { ArchitectureSection } from '../components/wiki/ArchitectureSection.tsx';
import { UnderstandingSection } from '../components/wiki/UnderstandingSection.tsx';
import { MigrationSection } from '../components/wiki/MigrationSection.tsx';
import { MvpCatalogSection } from '../components/wiki/MvpCatalogSection.tsx';
import { DiagramsSection } from '../components/wiki/DiagramsSection.tsx';
import { GeneratedCodeSection } from '../components/wiki/GeneratedCodeSection.tsx';

export function ProjectWikiPage() {
  const { id: projectId } = useParams<{ id: string }>();
  const [activeSection, setActiveSection] = useState<WikiSection>('overview');
  const wiki = useWikiData(projectId!);

  // Eager-load overview (analytics) on mount
  useEffect(() => {
    wiki.loadOverview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const projectName = wiki.analytics?.project.name ?? 'Project';

  return (
    <Layout>
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Breadcrumb */}
        <div className="flex items-center gap-1.5 border-b border-void-surface px-6 py-3 text-xs text-text-dim">
          <Link to="/" className="hover:text-text">Projects</Link>
          <ChevronRight className="h-3 w-3" />
          <Link to={`/project/${projectId}`} className="hover:text-text">{projectName}</Link>
          <ChevronRight className="h-3 w-3" />
          <span className="text-text">Wiki</span>
        </div>

        {/* Sidebar + Content */}
        <div className="flex flex-1 overflow-hidden">
          <WikiSidebar
            activeSection={activeSection}
            onSelectSection={setActiveSection}
            analytics={wiki.analytics}
          />

          <div className="flex-1 overflow-y-auto">
            {activeSection === 'overview' && (
              <OverviewSection
                analytics={wiki.analytics}
                loading={wiki.loading.overview}
                error={wiki.errors.overview}
                projectId={projectId!}
              />
            )}
            {activeSection === 'architecture' && (
              <ArchitectureSection
                projectId={projectId!}
                analytics={wiki.analytics}
                entryPoints={wiki.entryPoints}
                graphOverview={wiki.graphOverview}
                loading={wiki.loading.architecture}
                error={wiki.errors.architecture}
                onLoad={wiki.loadArchitecture}
              />
            )}
            {activeSection === 'understanding' && (
              <UnderstandingSection
                projectId={projectId!}
                analysisResults={wiki.analysisResults}
                loading={wiki.loading.understanding}
                error={wiki.errors.understanding}
                onLoad={wiki.loadUnderstanding}
                onLoadChain={wiki.loadChainDetail}
              />
            )}
            {activeSection === 'migration' && (
              <MigrationSection
                migrationPlan={wiki.migrationPlan}
                loading={wiki.loading.migration}
                error={wiki.errors.migration}
                onLoad={wiki.loadMigration}
                onLoadPhase={wiki.loadPhaseOutput}
              />
            )}
            {activeSection === 'mvp-catalog' && (
              <MvpCatalogSection
                migrationPlan={wiki.migrationPlan}
                loading={wiki.loading.migration}
                onLoad={wiki.loadMigration}
                onLoadDetail={wiki.loadMvpDetail}
              />
            )}
            {activeSection === 'diagrams' && (
              <DiagramsSection
                migrationPlan={wiki.migrationPlan}
                loading={wiki.loading.migration}
                onLoad={wiki.loadMigration}
                onLoadAvailability={wiki.loadDiagramAvailability}
                onLoadDiagram={wiki.loadDiagram}
              />
            )}
            {activeSection === 'generated-code' && (
              <GeneratedCodeSection
                migrationPlan={wiki.migrationPlan}
                loading={wiki.loading.migration}
                onLoad={wiki.loadMigration}
                onLoadPhase={wiki.loadPhaseOutput}
              />
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}
