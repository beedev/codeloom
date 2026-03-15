# Migration Brief Template

Template for the `MIGRATION_BRIEF.md` file generated during `/migrate init` Step 2.
Captures human decisions, business context, and tribal knowledge that tooling cannot auto-detect.
Fed into the architecture phase prompt as additional context.

---

## 1. Migration Decisions (REQUIRED)

```yaml
project_name: ""
source_platform: ""          # e.g., IBM z/OS, AS/400, Windows Server, Linux
source_languages: []          # e.g., [COBOL, PL/I, JCL, copybooks]
source_middleware: []          # e.g., [CICS, IMS, MQ, DB2]

target_stack:
  language: ""                # e.g., Java 21, Python 3.12, C# .NET 8
  framework: ""               # e.g., Spring Boot 3.x, FastAPI, ASP.NET Core
  database: ""                # e.g., PostgreSQL 16, Aurora, SQL Server
  batch_framework: ""         # e.g., Spring Batch, Python streaming, Airflow
  messaging: ""               # e.g., RabbitMQ, Kafka, SQS, REST webhooks
  ui_strategy: ""             # e.g., React SPA, API-only (no UI), server-rendered

migration_strategy: ""        # rewrite | convert | hybrid
```

## 2. Business Context (REQUIRED)

### Active vs Dead Code
Programs/modules that are dead code and should be skipped entirely.

### Processing Volumes
Approximate records/day and peak volumes for batch jobs. SLA deadlines if any.

### Compliance & Regulatory
PCI-DSS, SOX, GDPR, or other compliance requirements that constrain the target architecture.

## 3. Integration Boundaries (REQUIRED)

### Messaging / Queues
Queue names, direction (in/out), external system on the other end, protocol.

### File Transfers
Dataset/file names, direction, external system, frequency, format (fixed-width, CSV, etc.).

### Database Connections
External databases accessed (read-only lookups, shared tables with other applications).

## 4. Known Landmines (RECOMMENDED)

### Undocumented Business Rules
Rules that exist in code but aren't documented anywhere. Critical edge cases.

### Known Bugs / Workarounds
Behaviors that are technically bugs but are relied upon. Do not "fix" during migration.

### Runtime vs Code Discrepancies
Code paths that exist but aren't used in production, or behaviors that differ from what code suggests.

## 5. Shared Asset Inventory (RECOMMENDED)

Copybooks, includes, shared modules that define data contracts across programs.
Which programs use them, what domain they cover.

## 6. MVP Sequencing Preferences (OPTIONAL)

Override suggestions for migration order. Which programs must migrate together,
which can be skipped, which should go first/last.

## 7. Target Environment Constraints (OPTIONAL)

```yaml
deployment:
  platform: ""                # e.g., AWS ECS, Kubernetes, Azure App Service
  ci_cd: ""                   # e.g., GitHub Actions, Jenkins
constraints:
  max_memory_per_service: ""  # e.g., 512MB, 2GB
  auth_mechanism: ""          # e.g., OAuth2, API keys, mTLS
  network_restrictions: ""    # e.g., no internet from batch tier
```
