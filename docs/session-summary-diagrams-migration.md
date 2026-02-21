# Session Summary — CodeLoom Diagram & Migration Improvements

**Date**: 2026-02-20
**Scope**: Per-MVP UML diagrams, migration prompt fixes, tech stack display, MVP disk persistence, UI collapsibles

---

## 1. Per-MVP UML Diagram Generation

**New backend module**: `codeloom/core/diagrams/` — 7 files

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports |
| `service.py` | Orchestrator — generates/caches diagrams, writes to disk |
| `queries.py` | MVP-scoped SQL queries against ASG (code_units, code_edges) |
| `structural.py` | Deterministic PlantUML generators (Class, Package, Component) |
| `behavioral.py` | LLM-generated PlantUML (Sequence, Use Case, Activity, Deployment) |
| `prompts.py` | LLM prompt templates for behavioral diagrams |
| `renderer.py` | PlantUML text to SVG via PlantUML server (deflate + base64 encoding) |

**API endpoints** (`codeloom/api/routes/diagrams.py`):
- `GET /api/migration/{plan_id}/mvps/{mvp_id}/diagrams` — list availability
- `GET /api/migration/{plan_id}/mvps/{mvp_id}/diagrams/{type}` — get/generate
- `POST /api/migration/{plan_id}/mvps/{mvp_id}/diagrams/{type}/refresh` — force regenerate

**Frontend components**:
- `MvpDiagramPanel.tsx` — Tab bar (7 diagram types), loading states, refresh button
- `DiagramViewer.tsx` — SVG renderer with zoom, pan, fullscreen, download
- Integrated into `MvpInfoBar.tsx` as collapsible "UML Diagrams" section

**DB**: `FunctionalMVP.diagrams` JSONB column added (caches behavioral diagram PlantUML + SVG).

**Rendering approach**: Backend uses PlantUML server (configurable via `PLANTUML_SERVER_URL` env var, defaults to public `https://www.plantuml.com/plantuml`). Frontend receives pre-rendered SVG — no WASM or CheerpJ needed.

**Caching**: Structural diagrams are generated fresh on each request (fast, deterministic from ASG). Behavioral diagrams are LLM-generated and cached on the `FunctionalMVP.diagrams` JSONB column. Cache is cleared when MVP unit membership changes (`update_mvp`, `merge_mvps`, `split_mvp`).

---

## 2. Class Diagram — Missing Links Fix

**Root cause**: `queries.py` only fetched `inherits`/`implements`/`overrides` edges. Most codebases have few or none of these. The abundant `calls` and `imports` edges were ignored entirely.

**Fix** (`queries.py`):
- Added query for `calls` + `imports` edges between MVP units (LIMIT 500)
- Built child-to-parent reverse map from `contains` edges
- Aggregated method-level calls to class-level `depends` relationships
- Recursive resolution handles nested classes (method -> inner class -> outer class)
- Deduplication ensures one arrow per class pair

**Fix** (`structural.py`):
- Added `depends` edge rendering as `-->` association arrows
- Existing arrow types preserved: `--|>` (inherits), `..|>` (implements), `..>` (overrides)

---

## 3. SQL File Type Preservation in Migration

**Root cause**: Migration LLM prompts had zero file-type guardrails. The LLM saw "target: Java" and converted everything (including .sql schema files, migrations, stored procedures) into Java classes.

**Fix** (`codeloom/core/migration/prompts.py`):

Added `### File Type Conversion Rules` section to both `phase_2_architecture()` and `phase_5_transform()`:
- **Source code files** (.py, .js, .ts, .java, .cs, etc.) — convert to target language
- **SQL files** (.sql) — keep as SQL, map to target data access patterns but do NOT rewrite SQL into target language classes
- **Config files** (.yaml, .json, .xml, .properties, etc.) — map to target config format
- **Build files** (pom.xml, package.json, Dockerfile, etc.) — map to target build system
- **Documentation** (.md, .txt, .rst) — preserve as-is, update references only

Updated Module Structure Mapping table to include Source Language / Target Language columns for explicit tracking.

---

## 4. Source/Target Tech Stack Display

**Backend** (`codeloom/core/migration/engine.py`):
- `get_plan_status()` now queries `Project` model and includes `source_stack` (`primary_language`, `languages[]`) in response
- `list_plans()` includes the same `source_stack` field

**Frontend**:
- **`MigrationWizard.tsx`** — Tech stack info bar below breadcrumb showing:
  - Source languages (neutral gray badges)
  - Arrow separator
  - Target languages (glow badges) + frameworks (nebula badges) + versions
  - Migration type badge on the right (e.g., "framework migration")
- **`MigrationPlans.tsx`** — Plan cards show `source languages -> target languages` with arrow separator instead of just target
- **`types/index.ts`** — Added `source_stack` field to `MigrationPlan` interface

---

## 5. MVP Feature Documents on Disk

**Backend** (`codeloom/core/migration/engine.py`):
- Added `_get_plan_dir()` static method (DRY refactor from `_write_phase_to_disk`)
- Added `_write_mvp_documents()` method that writes:
  - `summary.md` — MVP name, description, status, priority, metrics table, source files table, dependency list
  - `analysis.md` — Full deep analysis output (Functional Requirements Register)
- Wired into `analyze_mvp()` with fire-and-forget pattern (DB is source of truth)

**Backend** (`codeloom/core/diagrams/service.py`):
- Added `_write_diagram_to_disk()` method
- Writes `{diagram_type}.puml` and `{diagram_type}.svg` after behavioral diagram generation
- Uses same `_get_plan_dir()` for path resolution

**Output location**: `outputs/migrations/{plan-id-prefix}-{project-slug}/_plans/mvp-{id}/`
```
mvp-1/
  summary.md
  analysis.md
  diagrams/
    sequence.puml
    sequence.svg
    usecase.puml
    usecase.svg
    ...
```

---

## 6. Collapsible Source Files & Code Units

**Frontend** (`MvpInfoBar.tsx`):
- Source Files and Code Units converted from always-visible inline layout to collapsible toggle sections
- Same chevron toggle pattern as Architecture Mapping and UML Diagrams
- Each section has `max-h-64` scrollable container for long lists
- All sections start collapsed; state resets on MVP change

---

## Files Modified (Complete List)

| File | Changes |
|------|---------|
| `codeloom/core/db/models.py` | `diagrams` JSONB column on FunctionalMVP |
| `codeloom/core/diagrams/__init__.py` (new) | Public API exports |
| `codeloom/core/diagrams/service.py` (new) | DiagramService orchestrator + disk persistence |
| `codeloom/core/diagrams/queries.py` (new) | MVP-scoped ASG queries + call aggregation |
| `codeloom/core/diagrams/structural.py` (new) | Class, Package, Component PlantUML generators |
| `codeloom/core/diagrams/behavioral.py` (new) | LLM-generated Sequence, Use Case, Activity, Deployment |
| `codeloom/core/diagrams/prompts.py` (new) | Behavioral diagram prompt templates |
| `codeloom/core/diagrams/renderer.py` (new) | PlantUML server encoding + SVG fetch |
| `codeloom/core/migration/prompts.py` | File type conversion rules in Architecture + Transform prompts |
| `codeloom/core/migration/engine.py` | `source_stack` in API, `_get_plan_dir()`, `_write_mvp_documents()` |
| `codeloom/api/routes/diagrams.py` (new) | Diagram API endpoints |
| `codeloom/api/app.py` | Registered diagrams router |
| `codeloom/api/deps.py` | `get_diagram_service()` dependency |
| `frontend/src/components/migration/MvpDiagramPanel.tsx` (new) | Diagram tab selector |
| `frontend/src/components/migration/DiagramViewer.tsx` (new) | SVG viewer with zoom/pan/fullscreen |
| `frontend/src/components/migration/MvpInfoBar.tsx` | Diagrams section + collapsible files/units |
| `frontend/src/pages/MigrationWizard.tsx` | Tech stack info bar |
| `frontend/src/pages/MigrationPlans.tsx` | Source to target badges on plan cards |
| `frontend/src/types/index.ts` | `DiagramResponse`, `DiagramAvailability`, `source_stack` |
| `frontend/src/services/api.ts` | `getMvpDiagram()`, `listMvpDiagrams()`, `refreshMvpDiagram()` |

---

## Pending / Not Yet Verified Visually

- [ ] Class diagram dependency arrows rendering with real project data
- [ ] Behavioral diagrams (Sequence, Use Case, Activity, Deployment) — require LLM + PlantUML server
- [ ] File type preservation — needs a new migration run to verify SQL files are preserved
- [ ] MVP documents on disk — needs a deep analysis run to trigger disk write
- [ ] Source stack display — needs a project with detected languages to verify badges
- [ ] PlantUML server connectivity — defaults to public server, may need `PLANTUML_SERVER_URL` env var for offline use
