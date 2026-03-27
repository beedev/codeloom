# ASG Enhancement Plan
Inspired by CodeGraphContext. 5 features, ordered by dependency + impact.

## Feature 1: Cyclomatic Complexity (Foundation — others depend on this)

**What**: Count branching nodes per function/method via tree-sitter. Store on `code_units.cyclomatic_complexity`.

**Backend changes**:
- `codeloom/core/ast_parser/enricher.py` — add `_compute_complexity(tree, node)` that counts: `if`, `elif`, `else`, `for`, `while`, `except`, `case`/`when`, `&&`, `||`, ternary. Score = 1 + branch_count.
- `codeloom/core/db/models.py` — add `cyclomatic_complexity = Column(Integer, nullable=True)` to `CodeUnit`
- Alembic migration: `ALTER TABLE code_units ADD COLUMN cyclomatic_complexity INTEGER`
- `codeloom/api/routes/projects.py` — add `GET /api/projects/{id}/complexity?sort=desc&limit=20` returning top-N complex units

**Frontend changes**:
- New `ComplexityPanel.tsx` in project view — sortable table: unit name, file, complexity score, color-coded (green <10, yellow 10-20, red >20)

**Effort**: Small — enricher already runs on all files during ingestion. ~2 hours.

---

## Feature 2: Dead Code Detection

**What**: Find functions/methods with zero incoming `calls` edges (excluding test files, entry points, decorators).

**Backend changes**:
- `codeloom/core/asg_builder/analyzer.py` (**NEW**) — `find_dead_code(project_id, exclude_patterns=["test_*", "main", "__init__"])`:
  ```sql
  SELECT cu.id, cu.name, cu.file_path
  FROM code_units cu
  WHERE cu.project_id = :pid
    AND cu.unit_type IN ('function', 'method')
    AND cu.name NOT LIKE 'test_%'
    AND cu.id NOT IN (
      SELECT DISTINCT target_unit_id FROM code_edges
      WHERE edge_type = 'calls' AND project_id = :pid
    )
  ```
- `codeloom/api/routes/projects.py` — add `GET /api/projects/{id}/dead-code?exclude=test_*,__init__`

**Frontend changes**:
- New `DeadCodePanel.tsx` — list of potentially dead functions with file link, option to exclude by pattern

**Effort**: Small — pure SQL query on existing edges. ~1.5 hours.

---

## Feature 3: Transitive Call Chains

**What**: Given a function, find ALL callers (or callees) recursively up to depth N. Critical for impact analysis.

**Backend changes**:
- `codeloom/core/asg_builder/analyzer.py` — add two functions:
  - `find_all_callers(unit_id, max_depth=5)` — recursive CTE on `code_edges WHERE edge_type='calls'`
  - `find_all_callees(unit_id, max_depth=5)` — same, reverse direction
  - `find_call_chain(from_unit_id, to_unit_id, max_depth=10)` — BFS/DFS pathfinding
  ```sql
  WITH RECURSIVE callers AS (
    SELECT source_unit_id, target_unit_id, 1 as depth
    FROM code_edges WHERE edge_type = 'calls' AND target_unit_id = :uid
    UNION ALL
    SELECT e.source_unit_id, e.target_unit_id, c.depth + 1
    FROM code_edges e JOIN callers c ON e.target_unit_id = c.source_unit_id
    WHERE e.edge_type = 'calls' AND c.depth < :max_depth
  )
  SELECT DISTINCT source_unit_id, depth FROM callers
  ```
- `codeloom/api/routes/graph.py` — add:
  - `GET /api/projects/{id}/graph/all-callers/{unit_id}?depth=5`
  - `GET /api/projects/{id}/graph/all-callees/{unit_id}?depth=5`
  - `GET /api/projects/{id}/graph/call-chain?from={uid1}&to={uid2}&depth=10`
- Existing `GET /api/projects/{id}/graph/callers/{unit_id}` stays (1-hop)

**Frontend changes**:
- Enhance GraphViewer to support multi-hop visualization
- Add depth slider control
- Highlight the chain path in a different color

**Effort**: Medium — recursive CTE is straightforward but frontend graph rendering for deep chains needs care. ~4 hours.

---

## Feature 4: Decorator/Annotation Search

**What**: Find all functions with a specific decorator (`@app.route`, `@Override`, `@Transactional`, etc.).

**Backend changes**:
- Enricher already extracts `modifiers` into CodeUnit metadata. Just needs a query:
  ```sql
  SELECT cu.* FROM code_units cu
  WHERE cu.project_id = :pid
    AND cu.metadata->>'modifiers' LIKE '%@' || :decorator || '%'
  ```
- `codeloom/api/routes/projects.py` — add `GET /api/projects/{id}/units/by-decorator?name=Override`
- Also support: `GET /api/projects/{id}/units/by-argument?type=String` (search `parsed_params` metadata)

**Frontend changes**:
- Search filter dropdown in code browser: "Filter by decorator" with autocomplete from known decorators

**Effort**: Small — metadata already exists, just query + UI. ~1.5 hours.

---

## Feature 5: Module Dependency Graph

**What**: Aggregate `imports` edges from unit-level to file/directory-level. Shows which modules depend on which.

**Backend changes**:
- `codeloom/core/asg_builder/analyzer.py` — add `get_module_dependencies(project_id, level='file'|'directory')`:
  ```sql
  -- File-level: aggregate imports edges by source_file → target_file
  SELECT
    source_file.file_path as source_module,
    target_file.file_path as target_module,
    COUNT(*) as import_count
  FROM code_edges e
  JOIN code_units source_unit ON e.source_unit_id = source_unit.id
  JOIN code_files source_file ON source_unit.file_id = source_file.id
  JOIN code_units target_unit ON e.target_unit_id = target_unit.id
  JOIN code_files target_file ON target_unit.file_id = target_file.id
  WHERE e.edge_type = 'imports' AND e.project_id = :pid
    AND source_file.id != target_file.id
  GROUP BY source_file.file_path, target_file.file_path
  ```
  - Directory-level: extract first N path segments, group by directory
- `codeloom/api/routes/graph.py` — add `GET /api/projects/{id}/graph/module-deps?level=file&prefix=src/`

**Frontend changes**:
- New visualization mode in GraphViewer: "Module Dependencies" tab
- Nodes = files or directories, edges = import count (thickness = weight)
- Useful for migration MVP clustering (shows natural module boundaries)

**Effort**: Medium — query is straightforward but the directory-level aggregation and new graph view need work. ~3 hours.

---

## Implementation Order

```
Feature 1: Complexity  ──→  Feature 2: Dead Code  ──→  Feature 4: Decorators
    (foundation)              (uses complexity        (quick win)
                               for prioritization)
                                      │
Feature 3: Transitive Calls ──────────┘
    (independent)            (dead code + chains = impact analysis)

Feature 5: Module Deps
    (independent, enhances migration MVP clustering)
```

**Total estimate**: ~12 hours across 5 features.

**Priority for migration use case**: 3 (transitive calls) > 2 (dead code) > 5 (module deps) > 1 (complexity) > 4 (decorators)

**Priority for general code intelligence**: 1 (complexity) > 2 (dead code) > 3 (transitive calls) > 4 (decorators) > 5 (module deps)
