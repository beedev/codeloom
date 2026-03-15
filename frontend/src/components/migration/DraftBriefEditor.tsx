/**
 * DraftBriefEditor — business context form for CLI-created draft plans.
 *
 * Shown when a plan was created via `/migrate` CLI with status='draft'
 * and orchestrator='claude_code_cli'. The user fills in business context
 * here, saves it, then runs `/migrate resume` in CLI to continue.
 */

import { useState } from 'react';
import { FileText, Loader2, Save, Terminal } from 'lucide-react';
import type { MigrationPlan } from '../../types/index.ts';
import * as api from '../../services/api.ts';

interface DraftBriefEditorProps {
  planId: string;
  plan: MigrationPlan;
  onSaved: () => void;
}

export function DraftBriefEditor({ planId, plan, onSaved }: DraftBriefEditorProps) {
  const brief = (plan.discovery_metadata?.migration_brief ?? {}) as Record<string, string>;

  const [deadCode, setDeadCode] = useState(brief.dead_code ?? '');
  const [processingVolumes, setProcessingVolumes] = useState(brief.processing_volumes ?? '');
  const [integrations, setIntegrations] = useState(brief.integrations ?? '');
  const [landmines, setLandmines] = useState(brief.landmines ?? '');
  const [compliance, setCompliance] = useState(brief.compliance ?? '');
  const [deploymentPlatform, setDeploymentPlatform] = useState(brief.deployment_platform ?? '');
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    setSaved(false);
    try {
      const data: Record<string, string> = {};
      if (deadCode.trim()) data.dead_code = deadCode.trim();
      if (processingVolumes.trim()) data.processing_volumes = processingVolumes.trim();
      if (integrations.trim()) data.integrations = integrations.trim();
      if (landmines.trim()) data.landmines = landmines.trim();
      if (compliance.trim()) data.compliance = compliance.trim();
      if (deploymentPlatform.trim()) data.deployment_platform = deploymentPlatform.trim();

      await api.updateMigrationBrief(planId, data);
      setSaved(true);
      onSaved();
    } catch (e) {
      console.error('Failed to save brief:', e);
    } finally {
      setIsSaving(false);
    }
  };

  const inputClass =
    'mt-1 w-full rounded border border-void-surface bg-void-light px-3 py-2 text-sm text-text placeholder:text-text-dim focus:border-glow/50 focus:outline-none';
  const labelClass = 'block text-xs font-medium text-text-muted';

  return (
    <div className="mx-auto max-w-xl space-y-6">
      {/* Header */}
      <div className="rounded-lg border border-glow/30 bg-glow/5 p-4">
        <div className="flex items-start gap-3">
          <Terminal className="mt-0.5 h-5 w-5 shrink-0 text-glow" />
          <div>
            <h3 className="text-sm font-semibold text-text">
              Plan created from Claude CLI
            </h3>
            <p className="mt-1 text-xs text-text-muted leading-relaxed">
              Fill in the business context below to help the migration engine make better
              architecture decisions. This information captures what the code analysis
              cannot detect — dead code, processing volumes, external integrations, and
              known issues.
            </p>
            <p className="mt-2 text-xs text-text-dim">
              After saving, run <code className="rounded bg-void-surface px-1.5 py-0.5 text-glow">/migrate resume</code> in
              Claude CLI to continue the migration.
            </p>
          </div>
        </div>
      </div>

      {/* Plan summary */}
      <div className="flex items-center gap-2 text-xs text-text-muted">
        <FileText className="h-3.5 w-3.5" />
        <span className="font-medium text-text">{plan.target_brief}</span>
      </div>

      {/* Form fields */}
      <div className="space-y-4">
        <div>
          <label className={labelClass}>Dead / unused programs to skip</label>
          <input
            type="text"
            value={deadCode}
            onChange={e => setDeadCode(e.target.value)}
            placeholder="e.g., CBOLD01C, TESTPGM — or leave blank"
            className={inputClass}
          />
        </div>

        <div>
          <label className={labelClass}>Processing volumes (batch job daily record counts)</label>
          <input
            type="text"
            value={processingVolumes}
            onChange={e => setProcessingVolumes(e.target.value)}
            placeholder="e.g., CBTRN01C: ~50K/day, 200K peak month-end — or unknown"
            className={inputClass}
          />
        </div>

        <div>
          <label className={labelClass}>External integrations (queues, file transfers, shared databases)</label>
          <textarea
            value={integrations}
            onChange={e => setIntegrations(e.target.value)}
            placeholder="e.g., MQ: ACCT.REQUEST.Q → Core Banking; Daily file from Payment Gateway"
            rows={3}
            className={inputClass}
          />
        </div>

        <div>
          <label className={labelClass}>Known landmines (undocumented rules, workarounds, gotchas)</label>
          <textarea
            value={landmines}
            onChange={e => setLandmines(e.target.value)}
            placeholder="e.g., CBACT04C treats status '77' as success — undocumented"
            rows={3}
            className={inputClass}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className={labelClass}>Compliance</label>
            <select
              value={compliance}
              onChange={e => setCompliance(e.target.value)}
              className={inputClass}
            >
              <option value="">None / Not sure</option>
              <option value="pci-dss">PCI-DSS</option>
              <option value="sox">SOX</option>
              <option value="gdpr">GDPR</option>
              <option value="hipaa">HIPAA</option>
              <option value="multiple">Multiple</option>
            </select>
          </div>
          <div>
            <label className={labelClass}>Deployment platform</label>
            <select
              value={deploymentPlatform}
              onChange={e => setDeploymentPlatform(e.target.value)}
              className={inputClass}
            >
              <option value="">Not decided</option>
              <option value="aws">AWS (ECS / Lambda)</option>
              <option value="kubernetes">Kubernetes</option>
              <option value="azure">Azure App Service</option>
              <option value="gcp">Google Cloud</option>
              <option value="on-prem">On-premises</option>
            </select>
          </div>
        </div>
      </div>

      {/* Save button */}
      <button
        onClick={handleSave}
        disabled={isSaving}
        className="flex w-full items-center justify-center gap-2 rounded-md bg-glow px-4 py-2.5 text-sm font-medium text-white hover:bg-glow-dim disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isSaving ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Saving...
          </>
        ) : saved ? (
          <>
            <Save className="h-4 w-4" />
            Saved — run /migrate resume to continue
          </>
        ) : (
          <>
            <Save className="h-4 w-4" />
            Save Business Context
          </>
        )}
      </button>
    </div>
  );
}
