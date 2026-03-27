# Reverse Engineering Documentation — Feature Plan

## Do We Need It?

**Short answer: Yes, but 70% of the intelligence already exists. We need an orchestration layer + UI.**

### What Already Exists

| Capability | Source | Status |
|-----------|--------|--------|
| Entry point detection | Understanding Engine | Done |
| Business rule extraction | Understanding Engine (LLM) | Done |
| Data entity extraction | Understanding Engine (LLM) | Done |
| Integration detection | Understanding Engine (LLM) | Done |
| Call tree tracing | Understanding Engine + ASG | Done |
| 7 UML diagram types | DiagramService | Done |
| Cyclomatic complexity | ASG enricher + queries | Done |
| Dead code detection | ASG queries | Done |
| Transitive call chains | ASG queries | Done |
| Module dependency graph | ASG queries | Done |
| Class hierarchy | ASG queries | Done |
| Ground truth (framework/layer) | GroundTruth validator | Done |
| RAG search | Hybrid retrieval | Done |

### What's Missing (the 30%)

| Gap | What It Is |
|-----|-----------|
| **Orchestration** | No service that composes all the above into a structured document |
| **LLM synthesis** | No narrative generation for architecture overview, data flow, risk assessment |
| **Document storage** | No DB table for persisted reverse engineering docs |
| **Document viewer UI** | No frontend to browse generated chapters |
| **MCP tools** | No programmatic access for CLI skill |
| **CLI skill** | No `/reverse-engineer` command |

## Architecture

```
/reverse-engineer skill (Claude Code CLI)
        ↓ MCP tools
ReverseEngineeringService (orchestrator)
        ↓ composes
┌───────────────────────────────────────────────┐
│  Chapter Generators (one per chapter)          │
│                                                │
│  Ch1: Executive Summary     ← project_intel    │
│  Ch2: Architecture Overview ← module_deps + LLM│
│  Ch3: Entry Points          ← understanding    │
│  Ch4: Functional Reqs       ← business_rules   │
│  Ch5: Data Model            ← data_entities    │
│  Ch6: Call Trees            ← transitive_calls │
│  Ch7: Integrations          ← integrations     │
│  Ch8: Code Quality          ← complexity+dead   │
│  Ch9: Tech Stack            ← ground_truth     │
└───────────────────────────────────────────────┘
        ↓ stores
    reverse_engineering_docs (DB)
        ↓ serves
    API routes → Frontend (Wiki integration)
```

## 9 Chapters

| # | Chapter | Data Source | LLM? | Speed |
|---|---------|-----------|------|-------|
| 1 | Executive Summary | project_intel + unit counts + analysis counts | No | <1s |
| 2 | Architecture Overview | module_deps + class_hierarchy + ground_truth | Yes | 5-15s |
| 3 | Entry Points & API Surface | entry_points table | No | <1s |
| 4 | Functional Requirements | deep_analyses.business_rules | No | <1s |
| 5 | Data Model | deep_analyses.data_entities + type_dep edges | Yes | 5-15s |
| 6 | Call Trees & Control Flow | get_all_callees() per top entry points | Yes | 5-20s |
| 7 | External Integrations | deep_analyses.integrations | Minimal | 2-5s |
| 8 | Code Quality & Risk | complexity + dead_code + module_deps | No | <2s |
| 9 | Technology Stack | project languages + ground_truth patterns | No | <1s |

**Total generation time: ~30-60s** (6 chapters are instant, 3 need LLM)

## Implementation

### Backend

**New files:**
```
codeloom/core/reverse_engineering/
├── __init__.py
├── service.py       # ReverseEngineeringService (orchestrator)
├── chapters.py      # 9 chapter generator functions
└── prompts.py       # LLM prompts for chapters 2, 5, 6
```

**Modified files:**
- `codeloom/core/db/models.py` — add ReverseEngineeringDoc model
- `codeloom/api/app.py` — register new router
- `codeloom/api/deps.py` — add get_reverse_engineering_service dependency
- `codeloom/mcp/server.py` — add 3 new MCP tools
- Alembic migration for new table

### API Routes

```
POST /api/projects/{id}/reverse-engineer/generate     → {doc_id, status}
GET  /api/projects/{id}/reverse-engineer/status/{doc}  → {status, progress}
GET  /api/projects/{id}/reverse-engineer/docs          → list docs
GET  /api/projects/{id}/reverse-engineer/doc/{doc}     → full markdown
GET  /api/projects/{id}/reverse-engineer/doc/{doc}/chapter/{n} → single chapter
```

### MCP Tools (3 new)

- `codeloom_generate_reverse_doc(project_id, chapters?)`
- `codeloom_get_reverse_doc(doc_id)`
- `codeloom_list_reverse_docs(project_id)`

### CLI Skill

`~/.claude/commands/reverse-engineer.md`:
```
/reverse-engineer [project]    — Generate full documentation
/reverse-engineer status       — Check progress
/reverse-engineer view [ch#]   — View chapter
```

### Frontend — Wiki Integration

- Add "Reverse Engineering Report" section to Project Wiki page
- Chapter sidebar navigation (9 chapters)
- Markdown renderer (react-markdown + remark-gfm)
- "Generate" button (disabled until deep analysis complete)
- Progress indicator during generation
- "Download as Markdown" button
- Embedded SVG diagrams in architecture + call tree chapters

### DB Model

```python
class ReverseEngineeringDoc(Base):
    doc_id       UUID PK
    project_id   UUID FK → projects
    status       String (pending|generating|complete|failed)
    chapters     JSONB {1: "markdown", 2: "markdown", ...}
    chapter_titles JSONB ["Executive Summary", ...]
    progress     Integer (chapters completed)
    total_chapters Integer (default 9)
    error        Text (nullable)
    created_at   Timestamp
    updated_at   Timestamp
```

## Prerequisite

**Deep analysis must be complete before generating.** The service checks `deep_analysis_status` and returns an error if not run. The UI disables the button with a message: "Run Deep Analysis first."

This is intentional — we compose from existing results, not re-analyze.

## Implementation Order

1. DB model + migration (~30 min)
2. Chapter generators + service (~3 hours — the core work)
3. API routes (~1 hour)
4. MCP tools (~30 min)
5. CLI skill (~30 min)
6. Frontend Wiki integration (~2 hours)

**Total: ~8 hours**
