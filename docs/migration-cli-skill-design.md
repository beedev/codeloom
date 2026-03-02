# Migration CLI Skill Design — Deep Analysis & Architecture

> **Session**: Deep analysis of migration engine gaps + design for a Claude CLI skill
> **Date**: 2026-03-02
> **Status**: Design Phase — Ready for Review

---

## Part 1: Gap Analysis — Why Migration Is Underperforming

### 1.1 The Core Problem: Rich Analysis, Weak Utilization

CodeLoom has exceptional code intelligence infrastructure:
- **AST** → 12 languages, semantic enrichment, enricher.py adds params/return types/modifiers/fields
- **ASG** → 7 edge types (contains, imports, calls, inherits, implements, overrides, type_dep)
- **RAPTOR** → hierarchical semantic summaries, L1 per-file, L2 functional clusters
- **Deep Understanding** → tiered LLM analysis with call chain tracing, entry point detection, business rules
- **MVP Clustering** → RAPTOR-driven semantic clusters with cohesion/coupling metrics

The migration engine *knows* your codebase deeply. The gap is that this knowledge barely influences **what gets generated**.

### 1.2 Specific Gaps Found

#### Gap 1: Lane Infrastructure Exists But Doesn't Drive Generation

The 3 existing lanes (`StrutsToSpringBoot`, `StoredProcToORM`, `VBNetToDotnetCore`) have:
- `TransformRule` — pattern→template mappings
- `augment_prompt()` — injects lane context into LLM prompts
- `apply_transforms()` — deterministic transforms for matched units

**But**: The agentic loop (`phases.py:execute_phase_agentic`) calls `build_tools_for_phase()` which gives the LLM tools to *read* data, then lets the LLM figure out the transform. The lane's `apply_transforms()` is called separately in the non-agentic path, creating two disconnected code paths.

The lane's `augment_prompt()` IS called in the non-agentic path but the agentic path (`_build_agentic_task_prompt`) assembles its own prompt without lane augmentation. **This is the single biggest gap** — the agentic loop ignores lane intelligence.

#### Gap 2: Deep Understanding Results Not Fed Into Migration

`get_deep_analysis` tool exists and can fetch understanding narratives. But:
- It's a passive tool the LLM *can* call — it's not automatically injected
- The understanding engine's entry-point detection, call chain traces, and business rule extraction are never systematically used to seed migration prompts
- The `deep_analysis_jobs` table may not even have results when migration runs (it's a separate background job)

**Result**: The migration LLM has to re-derive business logic that CodeLoom already analyzed.

#### Gap 3: Ground Truth Validation Is Post-Hoc

`CodebaseGroundTruth` extracts verified facts (actual class names, endpoint paths, SP names, etc.) and can validate outputs. But it's used only for:
- Post-clustering validation (checking cluster sanity)
- Not used to validate Transform outputs before approval

There's no `validate_transform_against_ground_truth()` call in the transform or test phases.

#### Gap 4: No Feedback Loop in Transform

When `validate_syntax` finds errors, the agent can attempt retry but:
- The LLM doesn't get structured guidance on *what* to fix
- There's no lane-specific fix heuristic (e.g., "Struts ActionForm → Spring DTO requires these specific field annotations")
- After `max_turns` the engine gives up and marks phase as requiring human review with no structured diagnosis

#### Gap 5: Lane Coverage Is Thin

3 lanes for what is potentially dozens of migration paths. Notable missing lanes:
- Express.js → NestJS / FastAPI
- Django 1.x → Django 4.x / FastAPI
- Angular.js → Angular / React
- jQuery → React
- JDBC → JPA/Hibernate (distinct from StoredProc→ORM)
- Flask → FastAPI
- PHP → Python/Node
- EJB → Spring Boot

#### Gap 6: V2/V3 Pipeline — Transform Still Single-Shot LLM

Even in V3 (Design-before-Transform), the Design phase generates a spec and Transform implements it. But:
- The design spec is markdown text passed as previous output context
- No structured code-level spec (interfaces, method signatures, data models)
- The LLM generating Transform must re-parse the Design output
- No per-file transform plan — it's one LLM call per MVP, not per file

#### Gap 7: The Web UI Is a Bottleneck

The current workflow requires:
1. Log in to web UI
2. Upload project zip
3. Navigate to migration tab
4. Create plan
5. Execute phases one by one
6. Review SSE-streamed output
7. Approve/reject
8. Repeat for each MVP

This is cumbersome for a developer workflow. There's no way to:
- Script a migration run
- Integrate with CI/CD
- Run from your editor
- Leverage Claude's full reasoning capabilities during migration

---

## Part 2: The Claude CLI Skill Concept

### 2.1 The Big Insight

**Claude Code CLI IS an agentic migration engine.** Instead of CodeLoom embedding a mini-agent (10 tools, SSE loop), let Claude Code be the agent with:

- **All Claude Code tools** natively available (Read, Write, Grep, Glob, Bash, Agent, AskUserQuestion)
- **Context7 MCP** for framework docs (no HTTP API key needed, direct integration)
- **CodeLoom API** for project data (AST, ASG, deep analysis, existing migration infrastructure)
- **Human-in-the-loop** via `AskUserQuestion` — interactive, not just approval gates
- **Subagents** for parallel MVP processing
- **Full filesystem access** to write generated files directly

This makes migration a first-class Claude Code workflow, not a web app feature.

### 2.2 What the Skill Is

A Claude Code slash command — `/migrate` — that:

1. **Connects to a running CodeLoom instance** (or directly to its DB/files) to pull project intelligence
2. **Presents a rich terminal UI** — lane selection, MVP overview, progress, diffs
3. **Orchestrates migration lane by lane, MVP by MVP** with interactive approval gates
4. **Writes output files directly** using Claude Code's Write tool
5. **Leverages all available intelligence** — AST, ASG, deep understanding, ground truth

---

## Part 3: Architecture Design

### 3.1 Skill Structure

```
~/.claude/commands/
└── migrate.md          ← main skill (the /migrate command)

~/.claude/commands/migrate/
├── lane-struts.md      ← /migrate:lane-struts  — Struts→Spring Boot lane skill
├── lane-express.md     ← /migrate:lane-express — Express.js→NestJS lane skill
├── lane-django.md      ← /migrate:lane-django  — Django→FastAPI lane skill
├── lane-jquery.md      ← /migrate:lane-jquery  — jQuery→React lane skill
└── lane-dotnet.md      ← /migrate:lane-dotnet  — VB.NET/.NET Fx→.NET Core lane skill
```

Each lane skill is a specialized version of the base migrate skill with lane-specific:
- Transform rules baked into the prompt
- Framework docs lookups (via Context7)
- Quality gate logic
- Output file structure guidance

### 3.2 Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    /migrate  (Claude CLI Skill)                       │
│                                                                       │
│  Phase 1: Project Intelligence Pull                                   │
│  ───────────────────────────────────────────────────────────────────│
│  • API call → GET /api/projects/{id}                                 │
│  • API call → GET /api/projects/{id}/graph/stats                     │
│  • API call → GET /api/understanding/{id}/results                    │
│  • API call → GET /api/migration/{plan_id} (if existing plan)        │
│  • Stores: ast_data, asg_data, deep_analysis, entry_points           │
│                                                                       │
│  Phase 2: Lane Detection & Selection                                  │
│  ───────────────────────────────────────────────────────────────────│
│  • API call → GET /api/migration/lanes                                │
│  • Analyze project's detected languages + frameworks (from AST)      │
│  • Show AskUserQuestion: which lane? what target stack?              │
│  • Activate lane-specific subskill                                    │
│                                                                       │
│  Phase 3: Architecture & Discovery (with deep intelligence)           │
│  ───────────────────────────────────────────────────────────────────│
│  • API call → POST /api/migration/{plan_id}/phase/1/execute          │
│  • Inject: deep analysis narratives, entry points, ASG topology       │
│  • AskUserQuestion: approve architecture? modify constraints?         │
│                                                                       │
│  Phase 4: MVP Overview & Prioritization                               │
│  ───────────────────────────────────────────────────────────────────│
│  • API call → GET /api/migration/{plan_id}/mvps                      │
│  • Render: terminal table of MVPs (name, units, cohesion, readiness) │
│  • AskUserQuestion: which MVPs to migrate first? any to skip?        │
│                                                                       │
│  Phase 5: Lane-Based Transform (per MVP, parallel option)             │
│  ───────────────────────────────────────────────────────────────────│
│  • Spawn Agent per MVP (or sequential based on user choice)          │
│  • Each agent: pull source → lane rules → Context7 docs → generate  │
│  • Write output files directly to filesystem                          │
│  • Validate: syntax + ground truth + gate checks                     │
│  • AskUserQuestion: review diff? approve? retry?                     │
│                                                                       │
│  Phase 6: Post-Migration                                              │
│  ───────────────────────────────────────────────────────────────────│
│  • Show migration scorecard                                           │
│  • Commit options                                                     │
│  • Next lane recommendations                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow: How Intelligence Is Leveraged

```
CodeLoom DB                   Claude CLI Skill              Output
──────────                   ─────────────────              ──────

code_units (AST)    ──────► read_source_unit()  ──────►
  qualified_name              Grep for patterns
  unit_type                   Read source files
  metadata.params
  metadata.return_type
  metadata.modifiers          ► Lane transform rules
  metadata.fields               applied to unit type

code_edges (ASG)    ──────► analyze_dependencies()  ──────►
  inherits/implements         identify what depends
  calls                       on what (blast radius)
  type_dep
                              ► Guides ordering of
                                generated files

deep_analyses       ──────► inject_understanding()  ──────► Richer prompts
  narrative                   prefill migration prompt       with business
  business_rules              with already-analyzed          context baked in
  integration_points          business logic

entry_points        ──────► identify_start_here()   ──────► Prioritize
  qualified_name              migration entry points         high-value
  call_depth                  are highest priority           units first

RAPTOR L2 clusters  ──────► map_to_mvps()           ──────► Semantically
  semantic meaning            validate/refine                coherent MVPs
  embeddings                  MVP groupings

ground_truth        ──────► validate_output()        ──────► Catch
  actual_class_names          check generated code            hallucinations
  actual_endpoints            against verified facts          early
  actual_sp_names
```

---

## Part 4: The Skill Design — `/migrate`

### 4.1 Command Interface

```bash
# Basic usage
/migrate --project <project-id> --lane struts-to-springboot

# Target stack specification
/migrate --project <id> --lane struts-to-springboot --target-stack '{"view_layer":"rest","java_version":"17"}'

# Use existing migration plan
/migrate --project <id> --plan <plan-id> --resume

# Parallel MVP processing
/migrate --project <id> --lane struts-to-springboot --parallel

# Dry run (analysis only, no file output)
/migrate --project <id> --lane struts-to-springboot --dry-run

# Specific MVPs only
/migrate --project <id> --lane struts-to-springboot --mvps "user-auth,order-processing"

# Output directory
/migrate --project <id> --lane struts-to-springboot --output ./migrated/
```

### 4.2 Terminal UI Design

#### Project Intelligence Summary (Phase 1 Output)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 Project Intelligence: myapp (v1.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Languages:      Java (1,247 units) | XML Config (89 files)
  Frameworks:     Struts 2.3 (detected), Hibernate 4.x, MySQL 5.7
  ASG Edges:      4,892 (1,203 calls | 892 inherits | 2,797 type_dep)
  Entry Points:   23 detected (Struts Actions, Servlets)
  Deep Analysis:  ✅ Complete (47 chains analyzed)

  Top Entry Points by Call Depth:
  ├── UserLoginAction.execute()       depth=8  [AUTH]
  ├── OrderSubmitAction.execute()     depth=12 [ORDERS]
  ├── ProductSearchAction.execute()   depth=6  [CATALOG]
  └── +20 more

  Detected Lane:  Struts 2.x → Spring Boot (confidence: 0.94)
  Suggested MVP Order (by readiness):
    1. User Authentication    cohesion=0.82  coupling=0.14  [LOW coupling ✅]
    2. Product Catalog        cohesion=0.71  coupling=0.31
    3. Order Processing       cohesion=0.69  coupling=0.44  [HIGH coupling ⚠️]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### Lane Selector (Phase 2 — `AskUserQuestion`)
```
? Which migration lane?
  ┌─────────────────────────────────────────────────────┐
  │ ● Struts 2.x → Spring Boot REST (detected, 0.94)    │
  │   Angular 1.x → React (view layer only)             │
  │   Hibernate 4.x → Spring Data JPA                  │
  │   Custom lane (describe target)                     │
  └─────────────────────────────────────────────────────┘

? Target view layer?
  ● REST API + DTOs      (recommended for decoupled frontend)
    Thymeleaf templates  (keep server-side rendering)
    React SPA            (full frontend rewrite)
```

#### MVP Progress Dashboard (during Phase 5)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔄 Migration Progress — Struts 2.x → Spring Boot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MVP Status:
  ✅ User Authentication     12 files  [approved]   confidence: HIGH
  🔄 Product Catalog         8 files   [transform]  ████████░░ 80%
  ⏳ Order Processing        23 files  [pending]
  ⏳ Admin Dashboard         6 files   [pending]
  ⏳ Reporting               4 files   [pending]

  Gates (User Authentication):
  ✅ endpoint_parity       23/23 @RequestMapping found
  ✅ service_layer         All ActionSupport→@Service converted
  ✅ syntax_valid          All generated files parse cleanly
  ⚠️  unit_test            12 tests pending (run mvn test to verify)

  Files Generated: ./migrated/src/main/java/com/example/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### Diff View (per file, before approval)
```
? Review generated: UserController.java

  Source: UserLoginAction.java (Struts)          →  Target: UserController.java (Spring Boot)
  ──────────────────────────────                    ──────────────────────────────────────────
  @Action("/user/login")                         →  @RestController
  public class UserLoginAction extends            →  @RequestMapping("/api/user")
      ActionSupport {                             →  public class UserController {
                                                 →
    private String username;                     →    private final UserService userService;
    private String password;                     →
    private String email;                        →    @Autowired
                                                 →    public UserController(UserService svc) {
    @Override                                    →      this.userService = svc;
    public String execute() {                    →    }
      UserService us = new UserService();        →
      User u = us.login(username, password);     →    @PostMapping("/login")
      if (u != null) {                           →    public ResponseEntity<UserDTO> login(
        ActionContext.getContext()               →        @RequestBody LoginRequest req) {
          .getSession()                          →      User u = userService.login(
          .put("user", u);                       →          req.getUsername(), req.getPassword());
        return SUCCESS;                          →      if (u == null) return ResponseEntity
      }                                          →          .status(401).build();
      addActionError("Invalid credentials");     →      return ResponseEntity.ok(
      return ERROR;                              →          new UserDTO(u.getId(), u.getEmail()));
    }                                            →    }
  }                                              →  }

  Lane rules applied: action_to_controller, inject_service, session_to_jwt
  Confidence: HIGH (0.92)
  Ground truth: ✅ path /user/login → @RequestMapping verified

  [A]pprove  [R]etry with feedback  [E]dit manually  [S]kip
```

---

## Part 5: Lane Skill Architecture

### 5.1 Each Lane Skill Provides

A lane skill file (`~/.claude/commands/migrate/lane-struts.md`) contains:

1. **Lane Identity Block** — source/target frameworks, version ranges
2. **Transform Rules Table** — deterministic mappings (unit_type → target_pattern)
3. **Context7 Lookup Map** — which docs to pull for which unit types
4. **Prompt Templates** — phase-specific prompt overlays
5. **Gate Definitions** — what to verify before approval
6. **File Layout Guide** — expected output directory structure
7. **Known Gotchas** — common pitfalls that trip up LLMs

### 5.2 Transform Rules as Embedded Knowledge

Instead of just passing rules to an LLM, the lane skill embeds them as structured data the skill can apply deterministically:

```markdown
## Transform Rules

| Source Unit Type     | Target Pattern          | Template Key                | Confidence |
|---------------------|-------------------------|------------------------------|-----------|
| struts_action        | @RestController          | spring_rest_controller       | 0.90      |
| struts_action_form   | Record/DTO class         | spring_dto_record            | 0.95      |
| struts_service       | @Service                 | spring_service_class         | 0.88      |
| struts_dao           | @Repository + JpaRepo    | spring_jpa_repository        | 0.85      |
| struts_config_xml    | Spring Boot main class   | spring_boot_app_config       | 0.92      |
| struts_validator_xml | @Valid + ConstraintAnnot | spring_bean_validation       | 0.80      |
| tiles_config_xml     | [SKIP or ThymeleafConfig]| depends on view_layer target | 0.70      |
```

### 5.3 Context7 Doc Lookup Map

Each lane knows exactly which docs to fetch from Context7 for each transform type:

```markdown
## Documentation Lookups (Context7)

| Unit Type Being Transformed | Framework | Topic                              |
|----------------------------|-----------|------------------------------------|
| struts_action              | Spring Boot | @RestController, @RequestMapping  |
| struts_action_form         | Spring Boot | @RequestBody, DTO patterns         |
| struts_service             | Spring      | @Service, @Transactional           |
| struts_dao                 | Spring Data | JpaRepository, @Query              |
| hibernate_entity           | Spring Data | @Entity, @Id, JPA annotations      |
| struts_validator_xml       | Spring Boot | @Valid, Bean Validation            |
| struts_config_xml          | Spring Boot | @SpringBootApplication, auto-config|
```

### 5.4 Understanding Integration

Each lane skill knows how to use the deep understanding output:

```markdown
## Understanding Integration

When deep_analysis results are available for an MVP:

1. Extract `business_rules` field → inject as "BUSINESS RULES TO PRESERVE" section
2. Extract `integration_points` → inject as "EXTERNAL DEPENDENCIES" section
3. Extract `data_flow` → inject as "DATA FLOW" section to guide DTO design
4. Use `entry_point.call_depth` to identify which controllers are primary vs utility
5. Use `business_entities` to ensure DTO field completeness

These should prefill the transform prompt BEFORE the LLM reads source code,
so it understands the business context, not just the structural patterns.
```

### 5.5 Gate Logic as Executable Checks

Rather than asking the LLM to validate, gates run as actual code checks:

```markdown
## Quality Gates

### Gate: endpoint_parity
Check: Every `@Action` path in source has a corresponding `@RequestMapping` in output.
Method:
1. grep all struts-config.xml for <action path="..."> → collect paths
2. grep generated Java files for @RequestMapping("...") → collect paths
3. Compare: missing_paths = source_paths - target_paths
4. PASS if missing_paths is empty

### Gate: service_injection
Check: No `new ServiceClass()` in generated code (direct instantiation)
Method:
1. grep generated Java files for `new \w+Service\(\)` or `new \w+Dao\(\)`
2. PASS if no matches

### Gate: syntax_valid
Check: All generated Java files parse with tree-sitter
Method: Run validate_syntax on each generated file
```

---

## Part 6: How It Leverages the Full Stack

### 6.1 AST Data → Transform Precision

Instead of: "The LLM reads source and guesses the target structure"

With the skill:
```
1. Pull code_units WHERE mvp_id = X AND unit_type = 'struts_action'
2. For each unit, get:
   - metadata.parsed_params → exact method parameters with types
   - metadata.return_type → exact return types
   - metadata.modifiers → public/private/static/final
   - metadata.annotations → @Override, @Transactional etc.
   - metadata.fields → class field declarations with types
3. Use this to generate EXACT method signatures in target code
   (not guessed from source text)
```

This produces type-safe generated code with correct signatures, not hallucinated ones.

### 6.2 ASG Edges → Correct Dependency Injection

```
1. Pull code_edges WHERE (source_id IN mvp_units OR target_id IN mvp_units)
   AND edge_type IN ('calls', 'type_dep', 'inherits', 'implements')
2. Build dependency graph: which classes call which
3. Use this to:
   - Generate correct @Autowired constructor parameters
   - Identify circular dependencies early
   - Order file generation (dependencies before dependents)
   - Know which interfaces to create vs which classes to inject
```

### 6.3 Deep Understanding → Business Context Preservation

```
1. Pull deep_analyses WHERE project_id = X
   (ordered by relevance to current MVP's entry points)
2. Extract:
   - narrative → prepend to transform prompt as "BUSINESS LOGIC CONTEXT"
   - business_rules → list of rules to explicitly preserve in generated code
   - integration_points → external APIs/DBs/services to maintain contracts for
3. This means the LLM generates code that preserves business logic,
   not just structural patterns
```

### 6.4 Ground Truth → Hallucination Prevention

```
1. Before calling LLM for transform, extract verified facts:
   - all class names in scope
   - all method names in scope
   - all @Action paths
   - all table names (from Hibernate/JPA entities)
2. After generation, validate:
   - No invented class names that don't exist in source
   - All referenced tables exist in source entities
   - All @RequestMapping paths correspond to source @Action paths
   - No methods called that don't exist in source
```

### 6.5 RAPTOR Semantic Clusters → Better MVPs

```
1. RAPTOR L2 clusters have semantic embeddings (what this code does)
2. Use these to:
   - Verify MVP coherence (are all units in an MVP semantically related?)
   - Identify when an MVP should be split (mixed semantics = bad cohesion)
   - Order MVPs by semantic independence (migrate standalone modules first)
   - Name MVPs by their semantic theme (not just "Cluster 3")
```

---

## Part 7: Implementation Plan

### 7.1 What to Build

#### Tier 1: Base Skill (`/migrate`) — ~2-3 days

**File**: `~/.claude/commands/migrate.md`

Core workflow:
1. Parse arguments
2. Connect to CodeLoom API, pull intelligence
3. Detect/select lane
4. Show project summary TUI
5. Create/resume migration plan
6. Execute Architecture + Discovery phases
7. Show MVP table, get user prioritization
8. For each MVP: transform → validate → diff → approve
9. Show completion scorecard

**No new backend code needed** — purely Claude CLI skill using existing API endpoints.

#### Tier 2: Lane Skills — ~1 day each

**Files**: `~/.claude/commands/migrate/lane-{name}.md`

Start with the 3 existing lanes (struts, storedproc, vbnet), add:
- `lane-express.md` (Express.js → NestJS/FastAPI)
- `lane-django.md` (Django 1.x/2.x → Django 4.x or FastAPI)
- `lane-jquery.md` (jQuery/Backbone → React)

Each lane skill is ~200 lines of structured markdown knowledge.

#### Tier 3: Intelligence Connectors — ~1 day

A Python helper script (`tools/codeloom_cli.py`) that the skill uses via Bash:
- `codeloom_cli.py get-project-intel <project_id>` → pulls AST+ASG+understanding in one call
- `codeloom_cli.py validate-output <plan_id> <mvp_id> <output_dir>` → runs gate checks
- `codeloom_cli.py write-diff <source_file> <generated_file>` → pretty diff output

#### Tier 4: Backend Enhancements — ~2 days (optional, high value)

New API endpoint: `GET /api/projects/{id}/migration-intel`

Returns a single payload combining:
- ASG topology summary (top nodes by connectivity)
- Entry points with call depths
- Deep analysis results (narratives per entry point)
- Ground truth snapshot (verified entity/class/path names)
- RAPTOR cluster summaries with semantic labels

This reduces the number of API calls from ~8 to 1.

### 7.2 File Structure

```
~/.claude/commands/
├── migrate.md                    ← /migrate base skill

~/.claude/commands/migrate/
├── lane-struts.md               ← /migrate:lane-struts
├── lane-storedproc.md           ← /migrate:lane-storedproc
├── lane-vbnet.md                ← /migrate:lane-vbnet
├── lane-express.md              ← /migrate:lane-express (new)
├── lane-django.md               ← /migrate:lane-django (new)
└── lane-jquery.md               ← /migrate:lane-jquery (new)

codeloom/
├── tools/
│   └── codeloom_cli.py          ← CLI helper for intelligence pull
└── api/routes/
    └── migration_intel.py       ← New combined intel endpoint (optional)
```

### 7.3 Priority Order

```
Priority 1 (highest value, fastest impact):
  → /migrate base skill with Struts lane
  → Intelligence pull from existing API endpoints
  → Terminal diff view + AskUserQuestion approval gates
  → Direct file write via Claude Code's Write tool

Priority 2 (quality improvement):
  → Backend: /api/projects/{id}/migration-intel endpoint
  → Ground truth validation gates as executable checks
  → Context7 MCP integration for real-time framework docs

Priority 3 (coverage expansion):
  → Additional lane skills (express, django, jquery)
  → Parallel MVP processing via Agent spawning
  → Migration scorecard + progress persistence

Priority 4 (power features):
  → CI/CD integration mode (--headless flag)
  → Git branch-per-MVP mode
  → Rollback support
```

---

## Part 8: Why This Is Qualitatively Better

### Current Architecture (Web UI + Internal Agent)
```
User → React UI → FastAPI → MigrationEngine → MigrationAgent
                                               └─ 10 tools (HTTP calls back to CodeLoom)
                                               └─ SSE stream to browser
                                               └─ Human approval via button click
```

Problems:
- Agent calls CodeLoom API to get data that's already in the same process
- SSE streaming to browser has latency and disconnect issues
- Human approval is a web form, not a conversation
- Can't leverage Claude Code's own tool access
- Can't spawn subagents per MVP
- LLM context is limited to what 10 tools return

### Claude CLI Skill Architecture
```
Developer → /migrate skill → Claude Code (full capabilities)
                              ├─ Read/Write/Grep/Glob → source and output files
                              ├─ Bash → validation scripts, tree-sitter
                              ├─ Agent → spawn per-MVP subagents
                              ├─ Context7 MCP → real-time framework docs
                              ├─ AskUserQuestion → conversational approval
                              ├─ TodoWrite → progress tracking
                              └─ API calls → CodeLoom for project intelligence
```

Benefits:
- Claude Code IS the agent — no wrapper, full context, full tools
- Lane intelligence is in the skill prompt, not a database lookup
- Framework docs via Context7 MCP (free, real-time, high quality)
- Parallel MVP processing via Agent spawning
- Conversational approval (user can give feedback, not just approve/reject)
- Files written directly to filesystem
- No SSE streaming issues
- Can be scripted and integrated with CI/CD

---

## Part 9: Key Questions for Review

Before building, decisions needed:

1. **CodeLoom Server Requirement**: Should the skill require a running CodeLoom instance (localhost:9005), or should it work with direct DB access? Direct DB access would be more powerful but more complex.

2. **File Output Strategy**: Write directly to a `--output` directory, or integrate with the existing `_write_phase_to_disk` mechanism in the engine? Direct write is simpler and more transparent.

3. **Lane Skill Distribution**: Should lane skills live in `~/.claude/commands/` (user-global) or in the CodeLoom repo itself (project-specific)? Project-specific would allow versioning lane knowledge alongside the codebase.

4. **Approval Granularity**: Approve per-file? Per-MVP? At the end? Per-file is safest but slowest. Recommend: per-MVP with file-level review on request.

5. **Backend Enhancement Priority**: Is the new `/api/projects/{id}/migration-intel` endpoint worth 2 days? It reduces complexity significantly. Recommend: yes, build it as Tier 2.

6. **Existing Plan Resumption**: The skill should detect if a migration plan already exists for a project and offer to resume it. This requires the API connection.

---

## Appendix: Available Data via API

For reference, what's available without backend changes:

| Data | Endpoint | Skill Use |
|------|----------|-----------|
| Project + languages | `GET /api/projects/{id}` | Lane detection |
| Code units (paginated) | `GET /api/projects/{id}/units` | Source reading |
| ASG edges | `GET /api/projects/{id}/graph/edges` | Dependency analysis |
| Entry points | `GET /api/understanding/{id}/entry-points` | Migration priority |
| Deep analysis results | `GET /api/understanding/{id}/results` | Business context |
| Migration lanes | `GET /api/migration/lanes` | Lane selection |
| Migration plan status | `GET /api/migration/{plan_id}` | Progress tracking |
| MVP list | `GET /api/migration/{plan_id}/mvps` | MVP overview |
| Phase output | `GET /api/migration/{plan_id}/phase/{n}` | Context carry-forward |
| Source + migrated diff | `GET /api/migration/{plan_id}/phase/{n}/diff-context` | Diff view |
| File download | `GET /api/migration/{plan_id}/phase/{n}/download` | Pull generated files |

---

*Design document ready for review.*
*Estimated implementation: 5-8 days for Tier 1 + 2 (base skill + 3 lane skills + intel endpoint)*
