# Lessons Learned

## 2026-03-16: Migration Accuracy Hallucination (Previous Session)

**What happened**: /migrate compare scored COACTUPC migration at 89% when actual paragraph-level accuracy was 18%. Overall score reported as 89/100 when deep comparison revealed 62/100.

**Root cause**: Scored migration accuracy by reading only Java target files, not pulling and comparing against COBOL source. A 97-line AccountService "looked reasonable" without knowing the source was 4,000 lines with 48 paragraphs.

**Rule**: NEVER score migration accuracy without pulling actual source code via codeloom_get_source_unit for EVERY program. The compare workflow MUST be source-first: pull source paragraphs → find Java equivalent → score. Not target-first.

**Specific anti-patterns to avoid**:
1. Don't score before comparison agents return results
2. Don't write accuracy reports based on target-only review
3. Don't count cosmetic refactoring (ControllerAdvice, Lombok removal) as substantive fixes
4. Don't report a score without evidence for every program's paragraph-level comparison
5. Don't rush to "complete" the compare workflow — thoroughness > speed

**How to apply**: When running /migrate compare:
- Pull ALL source units for each program BEFORE scoring
- Wait for ALL comparison agents to complete BEFORE writing reports
- Score = (matched_paragraphs / total_paragraphs) — not a gut feeling
- If you haven't read the source, the score is UNKNOWN, not "probably 90%"

---

## 2026-03-16: Subagent Path Discipline (This Session)

**What happened**: MVP 4 agent wrote 49 Java files to `backend/src/main/java/...` instead of `src/main/java/...`. This created a complete shadow project that Gradle never saw. Java compiled "UP-TO-DATE" (nothing new in the real src/), masking the problem until the comparison agent found 0% accuracy.

**Root cause**: The subagent prompt said "Write files to `/Users/bharath/Desktop/codeloom/migration-output/carddemo/`" but the agent interpreted this as needing a `backend/` subdirectory (common convention for mono-repos). The lack of explicit path examples in the prompt left room for interpretation.

**Rule**: ALWAYS include explicit, full file paths with examples in subagent prompts. Never say "write to the project" — say "write to `/exact/path/src/main/java/com/carddemo/service/TransactionService.java`".

**Detection**: The comparison agent caught this (0% score = "files don't exist"). Without source-first comparison, this would have been invisible.

**How to apply**:
- In subagent prompts, include 2-3 full path examples for generated files
- After subagent completes, verify files exist in expected locations with `find` or `ls`
- If Gradle says "UP-TO-DATE" after a big generation, something is wrong — files weren't written to the source set

---

## 2026-03-16: Entity-Schema Alignment (This Session)

**What happened**: JPA entities used surrogate `@Id Long id` + `@GeneratedValue(IDENTITY)` but Flyway SQL used business keys as primary keys (`account_id BIGINT PRIMARY KEY`). Hibernate validation failed with "missing column [id] in table [accounts]".

**Root cause**: Two separate subagents generated entities and SQL independently. The entity agent followed "surrogate key best practice" while the SQL agent followed "COBOL VSAM key → DB primary key". Neither consulted the other.

**Rule**: When generating entities AND schema in parallel agents, share the PK strategy explicitly. Better: use the original data model diagrams as the single source of truth.

**The CardDemo fix**: Business keys as PKs (matching COBOL VSAM keys) — `account_id`, `customer_id`, `card_number` for xref. Only Transaction and Card keep surrogate IDs (no stable COBOL key or encrypted field).

**How to apply**:
- Always check original data model diagrams BEFORE generating entities
- Share PK strategy in BOTH entity and SQL prompts
- After generation, test with `./gradlew bootRun` not just `compileJava` — compile doesn't catch schema mismatches

---

## 2026-03-16: Financial Calculation Precision (This Session)

**What happened**: COBOL `COMPUTE WS-MONTHLY-INT = (TRAN-CAT-BAL * DIS-INT-RATE) / 1200` was migrated as `balance * rate / 365`. Monthly interest vs daily interest — $150/month vs $4.93/day for $10K at 18%.

**Root cause**: The subagent inferred "daily interest" from context instead of reading the actual COBOL formula. COBOL divides by 1200 (= 100 * 12 = percent-to-decimal * annual-to-monthly). This is a domain-specific constant that must be preserved exactly.

**Rule**: For ANY financial calculation in COBOL, the EXACT formula must be copied from source. Never interpret or "modernize" financial math — copy the arithmetic literally, then convert syntax.

**How to apply**:
- Pull source for EVERY batch program with COMPUTE statements
- Copy the formula into comments above the Java equivalent
- Verify: same divisor, same rounding mode, same scale
- Write precision regression tests comparing COBOL output vs Java output

---

## 2026-03-16: Missing Intermediate Data Stores (This Session)

**What happened**: COBOL CBTRN02C updates a TCATBAL (Transaction Category Balance) file — a per-account/type/category ledger. This was completely missing from the Java migration. CBACT04C reads this file as input for interest calculation, so the entire interest pipeline was broken.

**Root cause**: The TCATBAL file is an intermediate data store between two batch programs. It's not a "main" entity like Account or Transaction. The entity generation agent didn't create it because it wasn't in the primary copybooks.

**Rule**: Map the COMPLETE batch pipeline data flow BEFORE generating code. Every file that one batch program writes and another reads is a required entity.

**How to apply**:
- During init, trace batch program chains: CBTRN01C → CBTRN02C → CBACT04C
- Identify ALL intermediate files (not just main entities)
- Create entities for every inter-program data store
- Include these in the Foundation MVP

---

## 2026-03-16: Accuracy Report Format Must Match UI Parser (This Session)

**What happened**: Wrote MIGRATION_ACCURACY.md with a different markdown structure than the AccuracyPanel.tsx parser expects. Report showed in DB but programs/bugs didn't render.

**Root cause**: The parser expects `### MVP N: Name` → `#### PROGRAM → \`target\`` → `| Paragraph | Status | Note |` table → `**Bugs:**` section. The first report used `### MVP 1: Auth & Users` headers with flat tables — no `####` program headers.

**Rule**: Always check the UI parser's expected format BEFORE writing the report. The markdown is a data contract, not freeform text.

**Additional bug found**: `---` horizontal rules were parsed as bug items because `bl.startsWith('-')` matches `---`. Fixed to `bl.startsWith('- ')`.

**How to apply**:
- Read AccuracyPanel.tsx `parseAccuracyMarkdown()` function first
- Use exact format: `####`, `→`, backtick-wrapped paths, status tokens (✅ ⚠️ ❌ 📝)
- Test with a small report before writing the full one
- Don't use `---` horizontal rules inside program blocks

---

## 2026-03-16: Seed Data Must Be Verified E2E (This Session)

**What happened**: The bcrypt hash in V2__seed_reference_data.sql didn't actually hash to "admin123". Login returned "Wrong Password" despite seed data being present.

**Root cause**: Used a generic bcrypt hash from the internet instead of generating one. Also, the UserType enum was stored as 'ADMIN' (Java enum name) but the COBOL value was 'A' — needed to match the JPA `@Enumerated(STRING)` strategy.

**Rule**: After seeding, always test login E2E. Generate bcrypt hashes programmatically, don't copy-paste.

---

## 2026-03-16: Port Conflicts in Multi-App Dev (This Session)

**What happened**: Migrated CardDemo frontend defaulted to port 3000, same as CodeLoom's frontend. API proxy sent requests to CodeLoom's FastAPI backend (port 9005) instead of Spring Boot (port 8080).

**Rule**: When migrating apps that coexist with CodeLoom, use different ports. CodeLoom owns 3000+9005. Migrated apps should use 3001+ and 8080+.
