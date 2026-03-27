# CardDemo MVP Migration Plan

## Mainframe COBOL/CICS to Java 21 + Spring Boot 3

**Total source files**: 61 (30 programs + 30+ copybooks)
**Total MVPs**: 7 (MVP 0–6)
**Every file assigned. No orphans.**

---

## MVP 0: Foundation

**Priority**: 0
**Description**: All shared copybooks (fan-in >= 3), all non-shared copybooks, and utility programs. Produces JPA entities, value objects, shared DTOs, enums, configuration, date utilities, validation framework, and the Spring Boot application skeleton. This MVP is the bedrock — every subsequent MVP depends on it.

**Source files**:
```
# High fan-in copybooks (shared data contracts)
COCOM01Y.cpy    (17 importers) — Common COMMAREA
COTTL01Y.cpy    (17) — Title/header layout
CSDAT01Y.cpy    (17) — Date data area
CSMSG01Y.cpy    (17) — Message area
CSUSR01Y.cpy    (12) — User security record
CVACT03Y.cpy    (11) — Account xref record
CVACT01Y.cpy    (10) — Account record
CVTRA05Y.cpy    (10) — Transaction record
CVACT02Y.cpy    (7)  — Card record
CVCUS01Y.cpy    (7)  — Customer record
CVCRD01Y.cpy    (5)  — Credit card record
CSMSG02Y.cpy    (4)  — Extended message area
CSSTRPFY.cpy    (4)  — String/proof utility

# Non-shared copybooks (lower fan-in, still foundation)
COADM02Y.cpy
CODATECN.cpy
COMEN02Y.cpy
COSTM01.cpy
CSLKPCDY.cpy
CSSETATY.cpy
CSUTLDPY.cpy
CSUTLDWY.cpy
CUSTREC.cpy
CVEXPORT.cpy
CVTRA01Y.cpy
CVTRA02Y.cpy
CVTRA03Y.cpy
CVTRA04Y.cpy
CVTRA06Y.cpy
CVTRA07Y.cpy
UNUSED1Y.cpy

# Utility programs
CSUTLDTC.cbl    — Date conversion utility
COBSWAIT.cbl    — Wait subroutine
```

**Target path**: `migration-output/carddemo/src/main/java/com/carddemo/`

**Depends on**: none

**Why this order**: Every program (batch and online) depends on these copybook layouts for record definitions, COMMAREA structure, and shared constants. The entity layer, DTOs, and utilities must exist before any business logic can be migrated.

**Functional Spec**:

Purpose: Establish the complete data model, shared infrastructure, and Spring Boot application skeleton that all subsequent MVPs build upon.

Key transforms:
- COCOM01Y (COMMAREA) → `CommArea.java` decomposed into purpose-specific DTOs (`AccountDto`, `CardDto`, `TransactionDto`, `UserDto`, `PagedResponse`, `ApiResponse`)
- CVACT01Y (Account record) → `Account.java` JPA entity with `@Entity`, `BigDecimal` for monetary fields, audit columns
- CVACT02Y (Card record) → `Card.java` entity with `@Convert(converter = EncryptedStringConverter.class)` for PCI-DSS
- CVACT03Y (Xref record) → `CrossReference.java` entity (card_number → account_id lookup)
- CVCUS01Y (Customer record) → `Customer.java` entity with GDPR-flagged PII fields
- CVCRD01Y (Credit card record) → `CreditCard.java` entity
- CVTRA05Y (Transaction record) → `Transaction.java` entity, partitioned by date
- CSUSR01Y (User security record) → `User.java` entity with bcrypt password hash
- CSDAT01Y (Date data area) → `LocalDate`/`LocalDateTime` native Java types
- CSMSG01Y/02Y (Message areas) → `ApiResponse<T>` wrapper class
- COTTL01Y (Title/header) → Removed (replaced by API metadata)
- CSSTRPFY (String/proof utility) → `StringProofUtil.java`
- CSUTLDTC (Date utility program) → `DateConversionUtil.java` (Julian, Lilian, century windowing)
- COBSWAIT (Wait subroutine) → `Thread.sleep()` / scheduled task delay (minimal)
- CVTRA01Y–07Y (Transaction variants) → Fields merged into `Transaction.java` entity + `TransactionType` enum
- COSTM01 (Statement layout) → `StatementDto.java`
- CUSTREC (Customer record variant) → Merged into `Customer.java`
- CVEXPORT (Export layout) → `DataExportDto.java`
- CODATECN (Date conversion data) → Constants in `DateConversionUtil.java`
- CSLKPCDY (Lookup by card) → Query method signature on `CrossReferenceRepository`
- CSSETATY (Set attribute) → Screen attribute logic removed (no BMS)
- CSUTLDPY/CSUTLDWY (Date utility working storage) → Private fields in `DateConversionUtil`
- COADM02Y (Admin menu layout) → Removed (REST routing replaces menus)
- COMEN02Y (Main menu layout) → Removed (REST routing replaces menus)
- UNUSED1Y → Skipped (unused)

Renamed exports table:

| Old Symbol | New Symbol | Old Import Path | New Import Path |
|---|---|---|---|
| COCOM01Y COMMAREA | CommArea (+ decomposed DTOs) | `COPY COCOM01Y` | `com.carddemo.dto.*` |
| CVACT01Y ACCOUNT-RECORD | Account entity | `COPY CVACT01Y` | `com.carddemo.entity.Account` |
| CVACT02Y CARD-RECORD | Card entity | `COPY CVACT02Y` | `com.carddemo.entity.Card` |
| CVACT03Y CARD-XREF-RECORD | CrossReference entity | `COPY CVACT03Y` | `com.carddemo.entity.CrossReference` |
| CVCUS01Y CUSTOMER-RECORD | Customer entity | `COPY CVCUS01Y` | `com.carddemo.entity.Customer` |
| CVCRD01Y CREDIT-CARD-RECORD | CreditCard entity | `COPY CVCRD01Y` | `com.carddemo.entity.CreditCard` |
| CVTRA05Y TRAN-RECORD | Transaction entity | `COPY CVTRA05Y` | `com.carddemo.entity.Transaction` |
| CSUSR01Y SEC-USER-DATA | User entity | `COPY CSUSR01Y` | `com.carddemo.entity.User` |
| CSDAT01Y DATE-DATA | LocalDate / LocalDateTime | `COPY CSDAT01Y` | `java.time.LocalDate` |
| CSMSG01Y MESSAGE-AREA | ApiResponse<T> | `COPY CSMSG01Y` | `com.carddemo.dto.ApiResponse` |
| CSMSG02Y EXT-MSG-AREA | ApiResponse<T> (merged) | `COPY CSMSG02Y` | `com.carddemo.dto.ApiResponse` |
| COTTL01Y TITLE-LINE | Removed | `COPY COTTL01Y` | N/A (API metadata) |
| CSSTRPFY STRING-PROOF | StringProofUtil | `COPY CSSTRPFY` | `com.carddemo.util.StringProofUtil` |
| CSUTLDTC (program) | DateConversionUtil | `CALL 'CSUTLDTC'` | `com.carddemo.util.DateConversionUtil` |
| COBSWAIT (program) | Removed / Thread.sleep | `CALL 'COBSWAIT'` | `Thread.sleep()` or removed |

Acceptance criteria:
- All JPA entities compile and have correct field types (BigDecimal for money, String for IDs)
- `Account`, `Card`, `Transaction`, `Customer`, `User`, `CrossReference`, `CreditCard` entities pass schema validation
- Flyway migration `V1__create_schema.sql` creates all tables with correct constraints and indexes
- `DateConversionUtil` passes tests for: Julian→Gregorian, Gregorian→Julian, Lilian day number, leap year handling, century windowing
- `StringProofUtil` matches COBOL INSPECT/STRING behavior for test cases
- Spring Boot application starts with empty database
- All repositories have basic CRUD integration tests via Testcontainers
- `EncryptedStringConverter` encrypts/decrypts card numbers correctly (AES-256-GCM)
- Audit columns (`createdAt`, `updatedAt`, `createdBy`, `updatedBy`) auto-populated via JPA auditing
- SYMBOLS.md written with all entity/DTO/util mappings

---

## MVP 1: Authentication and User Management

**Priority**: 1
**Description**: Sign-on, admin menu, and full user CRUD (list, add, update, delete). Establishes Spring Security with JWT, RBAC, and the admin workflow.

**Source files**:
```
COSGN00C.cbl    — Sign-on screen (user authentication via USRSEC file)
COMEN01C.cbl    — Main menu navigation
COADM01C.cbl    — Admin menu navigation
COUSR00C.cbl    — User list (paginated browse of USRSEC file)
COUSR01C.cbl    — User add
COUSR02C.cbl    — User update
COUSR03C.cbl    — User delete
```

**Target path**: `migration-output/carddemo/src/main/java/com/carddemo/`
- `controller/AuthController.java`, `controller/UserController.java`
- `service/AuthService.java`, `service/UserService.java`
- `security/SecurityConfig.java`, `security/JwtTokenProvider.java`
- `config/SecurityConfig.java`

**Depends on**: MVP 0 (Foundation)

**Why this order**: Authentication is the gateway to all other functionality. Every online program checks user credentials via the USRSEC file. Spring Security must be configured before any protected endpoints exist. User management is tightly coupled to auth (same VSAM file, same domain).

**Functional Spec**:

Purpose: Replace CICS sign-on and user administration with Spring Security JWT authentication and a REST user management API.

Key transforms:
- COSGN00C `EXEC CICS READ FILE('USRSEC')` → `AuthService.authenticate()` → `UserRepository.findByUserId()` + BCrypt password verify → JWT token
- COSGN00C `EXEC CICS RETURN TRANSID('CM01')` → Return JWT in `POST /api/auth/login` response
- COMEN01C / COADM01C (menu navigation) → Removed; replaced by REST endpoint routing + role-based access
- COUSR00C `EXEC CICS STARTBR/READNEXT FILE('USRSEC')` → `GET /api/users?page=0&size=20` → `UserRepository.findAll(Pageable)`
- COUSR01C `EXEC CICS WRITE FILE('USRSEC')` → `POST /api/users` → `UserService.createUser()` → `UserRepository.save()`
- COUSR02C `EXEC CICS READ/REWRITE FILE('USRSEC')` → `PUT /api/users/{id}` → `UserService.updateUser()`
- COUSR03C `EXEC CICS DELETE FILE('USRSEC')` → `DELETE /api/users/{id}` → `UserService.deleteUser()`
- PF-key navigation → HTTP methods + URL routing
- COMMAREA user fields → `SignOnRequest`, `UserDto`, `UserCreateRequest`

Renamed exports table:

| Old Symbol | New Symbol | Old Import Path | New Import Path |
|---|---|---|---|
| COSGN00C (program) | AuthController + AuthService | `EXEC CICS XCTL PROGRAM('COSGN00C')` | `com.carddemo.controller.AuthController` |
| COMEN01C (program) | Removed (routing) | `EXEC CICS XCTL PROGRAM('COMEN01C')` | N/A |
| COADM01C (program) | Removed (routing) | `EXEC CICS XCTL PROGRAM('COADM01C')` | N/A |
| COUSR00C (program) | UserController.listUsers() | `EXEC CICS XCTL PROGRAM('COUSR00C')` | `com.carddemo.controller.UserController` |
| COUSR01C (program) | UserController.createUser() | `EXEC CICS XCTL PROGRAM('COUSR01C')` | `com.carddemo.controller.UserController` |
| COUSR02C (program) | UserController.updateUser() | `EXEC CICS XCTL PROGRAM('COUSR02C')` | `com.carddemo.controller.UserController` |
| COUSR03C (program) | UserController.deleteUser() | `EXEC CICS XCTL PROGRAM('COUSR03C')` | `com.carddemo.controller.UserController` |

Acceptance criteria:
- `POST /api/auth/login` with valid credentials returns JWT token
- `POST /api/auth/login` with invalid credentials returns 401
- JWT token contains user ID, roles, and expiry
- `GET /api/users` returns paginated user list (requires ADMIN role)
- `POST /api/users` creates new user with bcrypt-hashed password
- `PUT /api/users/{id}` updates user fields
- `DELETE /api/users/{id}` soft-deletes user (SOX: no hard delete of security records)
- Role-based access: only ADMIN users can access `/api/users/*` and `/api/admin/*`
- All auth actions logged in audit trail (SOX)
- Password validation: minimum 8 chars, complexity rules

---

## MVP 2: Account Management

**Priority**: 2
**Description**: Account view, account update with full field validation, and card cross-reference lookup. Core account domain.

**Source files**:
```
COACTVWC.cbl    — Account view (read-only account display with card xref)
COACTUPC.cbl    — Account update (CICS BMS maps, full field validation: SSN, phone, state, zip, FICO)
```

**Target path**: `migration-output/carddemo/src/main/java/com/carddemo/`
- `controller/AccountController.java`
- `service/AccountService.java`
- `validation/SsnValidator.java`, `PhoneValidator.java`, `StateCodeValidator.java`, `ZipCodeValidator.java`, `FicoScoreValidator.java`

**Depends on**: MVP 0 (Foundation), MVP 1 (Auth — protected endpoints)

**Why this order**: Accounts are the central domain object. Cards, transactions, and billing all reference accounts. Account update (COACTUPC) is one of the most complex online programs with extensive field validation that must be implemented as reusable validators.

**Functional Spec**:

Purpose: Replace CICS account view and update screens with REST endpoints, implementing all field validation as Jakarta Bean Validation constraints.

Key transforms:
- COACTVWC `EXEC CICS READ FILE('ACCTDAT')` → `GET /api/accounts/{id}` → `AccountService.getAccount()` → `AccountRepository.findById()`
- COACTVWC xref lookup → `CrossReferenceRepository.findByAccountId()` to include associated cards
- COACTUPC `EXEC CICS READ FILE('ACCTDAT') UPDATE` + `EXEC CICS REWRITE` → `PUT /api/accounts/{id}` → `@Transactional AccountService.updateAccount()`
- COACTUPC inline SSN validation (9 digits, not all zeros) → `@ValidSsn` custom constraint + `SsnValidator`
- COACTUPC inline phone validation → `@ValidPhone` + `PhoneValidator`
- COACTUPC inline state code validation (50 states + DC) → `@ValidStateCode` + `StateCodeValidator`
- COACTUPC inline zip validation (5 or 9 digits) → `@ValidZipCode` + `ZipCodeValidator`
- COACTUPC inline FICO validation (300–850 range) → `@ValidFicoScore` + `FicoScoreValidator`
- COACTUPC `EVALUATE TRUE / WHEN` field checks → Jakarta `@Valid` on `AccountUpdateRequest` DTO
- COACTUPC error messages to BMS map → `FieldError` list in `ApiResponse<T>`

Renamed exports table:

| Old Symbol | New Symbol | Old Import Path | New Import Path |
|---|---|---|---|
| COACTVWC (program) | AccountController.getAccount() | `EXEC CICS XCTL PROGRAM('COACTVWC')` | `com.carddemo.controller.AccountController` |
| COACTUPC (program) | AccountController.updateAccount() | `EXEC CICS XCTL PROGRAM('COACTUPC')` | `com.carddemo.controller.AccountController` |
| COACTUPC SSN-VALIDATE paragraph | SsnValidator.isValid() | inline in COACTUPC | `com.carddemo.validation.SsnValidator` |
| COACTUPC PHONE-VALIDATE paragraph | PhoneValidator.isValid() | inline in COACTUPC | `com.carddemo.validation.PhoneValidator` |
| COACTUPC STATE-VALIDATE paragraph | StateCodeValidator.isValid() | inline in COACTUPC | `com.carddemo.validation.StateCodeValidator` |

Acceptance criteria:
- `GET /api/accounts/{id}` returns account with associated card numbers
- `PUT /api/accounts/{id}` validates all fields before update
- Invalid SSN returns 400 with field-level error message
- Invalid phone/state/zip/FICO each return 400 with specific error
- Account update writes audit trail (old value → new value) for SOX
- BigDecimal precision matches COBOL `PIC S9(n)V9(m)` for balance, credit limit
- Concurrent updates handled via optimistic locking (`@Version`)

---

## MVP 3: Card Management

**Priority**: 3
**Description**: Credit card list (paginated with filtering), card select/view, and card update. Card-specific domain.

**Source files**:
```
COCRDLIC.cbl    — Credit card list (paginated browse with filtering)
COCRDSLC.cbl    — Credit card select/view
COCRDUPC.cbl    — Credit card update (edit card fields, status, expiry)
```

**Target path**: `migration-output/carddemo/src/main/java/com/carddemo/`
- `controller/CardController.java`
- `service/CardService.java`

**Depends on**: MVP 0 (Foundation — Card entity, encryption), MVP 1 (Auth), MVP 2 (Account — cards reference accounts)

**Why this order**: Cards depend on accounts (via xref). Card list uses the same pagination pattern established in MVP 1 (user list). Card data requires PCI-DSS encryption from Foundation.

**Functional Spec**:

Purpose: Replace CICS card browse, select, and update screens with REST endpoints, ensuring PCI-DSS compliant card number handling throughout.

Key transforms:
- COCRDLIC `EXEC CICS STARTBR FILE('CARDDAT')` + `READNEXT` loop with PF7/PF8 → `GET /api/cards?page=0&size=20&accountId=&status=` → `CardRepository.findByFilters(Specification, Pageable)`
- COCRDLIC filter logic (by account, by status) → Spring Data JPA `Specification<Card>` dynamic queries
- COCRDSLC `EXEC CICS READ FILE('CARDDAT') KEY IS card-number` → `GET /api/cards/{cardNumber}` → `CardService.getCard()` (returns masked number, full details)
- COCRDUPC `EXEC CICS READ FILE('CARDDAT') UPDATE` + `REWRITE` → `PUT /api/cards/{cardNumber}` → `@Transactional CardService.updateCard()`
- COCRDUPC status change validation → `CardStatus` enum with valid transition rules
- COCRDUPC expiry date update → `LocalDate` with future-date validation
- All card number display → Masked via `PciDataMasker` (show last 4 only)

Renamed exports table:

| Old Symbol | New Symbol | Old Import Path | New Import Path |
|---|---|---|---|
| COCRDLIC (program) | CardController.listCards() | `EXEC CICS XCTL PROGRAM('COCRDLIC')` | `com.carddemo.controller.CardController` |
| COCRDSLC (program) | CardController.getCard() | `EXEC CICS XCTL PROGRAM('COCRDSLC')` | `com.carddemo.controller.CardController` |
| COCRDUPC (program) | CardController.updateCard() | `EXEC CICS XCTL PROGRAM('COCRDUPC')` | `com.carddemo.controller.CardController` |

Acceptance criteria:
- `GET /api/cards` returns paginated card list with masked card numbers (never full PAN in response)
- Filtering by account ID and card status works correctly
- `GET /api/cards/{cardNumber}` returns full card details (still masked number) with account reference
- `PUT /api/cards/{cardNumber}` updates card status and expiry
- Invalid status transitions rejected (e.g., CLOSED → ACTIVE without reactivation)
- Card number encrypted at rest in database (verified via raw SQL query showing ciphertext)
- All card data access logged in audit trail (PCI-DSS)
- No card numbers appear in application logs

---

## MVP 4: Transaction Online (View, Add, Billing, Reports)

**Priority**: 4
**Description**: Transaction list, transaction detail view, transaction add, billing/statement view, and report submission. All online transaction-related programs grouped together.

**Source files**:
```
COTRN00C.cbl    — Transaction list (paginated browse with PF7/PF8)
COTRN01C.cbl    — Transaction detail view
COTRN02C.cbl    — Transaction add (validates input, writes to tran file, reads xref/acct)
COBIL00C.cbl    — Billing/statement view (browse transactions, read acct via xref)
CORPT00C.cbl    — Report submission (submits batch jobs to internal reader/TDQ)
```

**Target path**: `migration-output/carddemo/src/main/java/com/carddemo/`
- `controller/TransactionController.java`, `controller/ReportController.java`
- `service/TransactionService.java`, `service/BillingService.java`, `service/ReportService.java`

**Depends on**: MVP 0 (Foundation — Transaction entity), MVP 1 (Auth), MVP 2 (Account — transactions reference accounts), MVP 3 (Card — transactions reference cards via xref)

**Why this order**: Transactions depend on accounts and cards (xref lookup). COBIL00C reads account+transaction data established in prior MVPs. CORPT00C submits batch jobs that will be built in MVP 5, but the submission endpoint belongs here with the online transaction domain.

**Functional Spec**:

Purpose: Replace CICS transaction browse, add, billing view, and report submission screens with REST endpoints.

Key transforms:
- COTRN00C `STARTBR/READNEXT` paginated browse → `GET /api/transactions?page=0&size=20&accountId=&dateFrom=&dateTo=`
- COTRN01C `READ FILE('TRANDAT')` → `GET /api/transactions/{id}`
- COTRN02C `WRITE FILE('TRANDAT')` + xref/acct validation → `POST /api/transactions` → `@Transactional TransactionService.addTransaction()` (validates account exists via xref, validates card active, writes transaction)
- COBIL00C browse transactions by account → `GET /api/accounts/{id}/billing?page=&period=` → `BillingService.getBilling()` (reads xref → account → transactions for period)
- CORPT00C `EXEC CICS WRITEQ TDQ` / `WRITE internal reader` → `POST /api/reports/submit` → `ReportService.submitReport()` → `JobLauncher.run(reportJob)` → return job execution ID
- CORPT00C report status check → `GET /api/reports/{jobId}/status` → Spring Batch `JobExplorer.getJobExecution()`
- PF7/PF8 paging → `page` + `size` query parameters

Renamed exports table:

| Old Symbol | New Symbol | Old Import Path | New Import Path |
|---|---|---|---|
| COTRN00C (program) | TransactionController.listTransactions() | `EXEC CICS XCTL PROGRAM('COTRN00C')` | `com.carddemo.controller.TransactionController` |
| COTRN01C (program) | TransactionController.getTransaction() | `EXEC CICS XCTL PROGRAM('COTRN01C')` | `com.carddemo.controller.TransactionController` |
| COTRN02C (program) | TransactionController.addTransaction() | `EXEC CICS XCTL PROGRAM('COTRN02C')` | `com.carddemo.controller.TransactionController` |
| COBIL00C (program) | AccountController.getBilling() | `EXEC CICS XCTL PROGRAM('COBIL00C')` | `com.carddemo.controller.AccountController` |
| CORPT00C (program) | ReportController.submitReport() | `EXEC CICS XCTL PROGRAM('CORPT00C')` | `com.carddemo.controller.ReportController` |

Acceptance criteria:
- `GET /api/transactions` returns paginated transactions with filtering by account, date range
- `GET /api/transactions/{id}` returns full transaction detail including category description
- `POST /api/transactions` validates: account exists, card active, amount > 0, writes transaction with status PENDING
- `GET /api/accounts/{id}/billing` returns transactions for billing period with account summary
- `POST /api/reports/submit` launches Spring Batch job and returns job execution ID
- `GET /api/reports/{jobId}/status` returns STARTED/COMPLETED/FAILED
- Transaction amounts use BigDecimal with correct precision
- All transaction writes logged in audit trail (SOX)

---

## MVP 5: Batch — Data Readers and Export

**Priority**: 5
**Description**: Batch file reader programs (account, card, xref, customer) and the full data export. These are the simpler batch programs that read data and produce output files.

**Source files**:
```
CBACT01C.cbl    — Account file reader/writer (file I/O, outputs fixed/array/VB formats)
CBACT02C.cbl    — Card file reader
CBACT03C.cbl    — Xref file reader
CBCUS01C.cbl    — Customer file reader
CBEXPORT.cbl    — Full data export (customers, accounts, xrefs, transactions, cards → export files)
```

**Target path**: `migration-output/carddemo/src/main/java/com/carddemo/batch/`
- `config/AccountFileJobConfig.java`, `config/DataExportJobConfig.java`
- `reader/AccountItemReader.java`, `reader/CardItemReader.java`, `reader/CustomerItemReader.java`
- `writer/DataExportWriter.java`

**Depends on**: MVP 0 (Foundation — all entities and repositories)

**Why this order**: These batch programs are data-oriented (read + write) with minimal business logic. They establish the Spring Batch infrastructure (job repository, chunk processing, flat file writers) that the more complex transaction batch programs in MVP 6 will reuse. CBEXPORT depends on all entity types from Foundation.

**Functional Spec**:

Purpose: Replace COBOL batch file reader programs and the data export program with Spring Batch jobs using chunk-oriented processing.

Key transforms:
- CBACT01C `OPEN INPUT / READ ... AT END / CLOSE` sequential pattern → `AccountFileJob` with `RepositoryItemReader<Account>` → `FlatFileItemWriter` (fixed-format, matching original output layout)
- CBACT01C multiple output formats (fixed, array, VB) → Multi-step job with different `FlatFileItemWriter` configurations per step
- CBACT02C `READ card file` loop → Step within `AccountFileJob` using `CardItemReader`
- CBACT03C `READ xref file` loop → Step within `AccountFileJob` using `CrossReferenceItemReader`
- CBCUS01C `READ customer file` loop → Step within `DataExportJob` using `CustomerItemReader`
- CBEXPORT multi-file output → `DataExportJob` with 5 steps (customers, accounts, xrefs, transactions, cards), each step: `RepositoryItemReader` → `FlatFileItemWriter`
- All batch: `OPEN/CLOSE` → Spring Batch managed lifecycle
- All batch: `PERFORM UNTIL EOF` → Chunk-oriented reading (chunk size 500)

Renamed exports table:

| Old Symbol | New Symbol | Old Import Path | New Import Path |
|---|---|---|---|
| CBACT01C (program) | AccountFileJob | JCL `EXEC PGM=CBACT01C` | `com.carddemo.batch.config.AccountFileJobConfig` |
| CBACT02C (program) | AccountFileJob (card step) | JCL `EXEC PGM=CBACT02C` | `com.carddemo.batch.config.AccountFileJobConfig` |
| CBACT03C (program) | AccountFileJob (xref step) | JCL `EXEC PGM=CBACT03C` | `com.carddemo.batch.config.AccountFileJobConfig` |
| CBCUS01C (program) | DataExportJob (customer step) | JCL `EXEC PGM=CBCUS01C` | `com.carddemo.batch.config.DataExportJobConfig` |
| CBEXPORT (program) | DataExportJob | JCL `EXEC PGM=CBEXPORT` | `com.carddemo.batch.config.DataExportJobConfig` |

Acceptance criteria:
- `AccountFileJob` reads all accounts from DB and writes fixed-format flat file matching COBOL output layout
- Card and xref steps produce equivalent output files
- `DataExportJob` exports all 5 entity types to separate flat files
- Chunk processing: 500 records per commit, restartable from last committed chunk
- Export file row counts match database record counts
- Output field formats match COBOL PIC clauses (right-justified numerics, left-justified alpha, zero-padded)
- Jobs can be triggered via `POST /api/reports/submit` (from MVP 4) or Kubernetes CronJob

---

## MVP 6: Batch — Transaction Processing and Reporting

**Priority**: 6
**Description**: Transaction validation, transaction posting, interest/fee calculation, statement generation, and transaction reporting. The complex batch programs with business logic, cross-file lookups, and financial calculations.

**Source files**:
```
CBTRN01C.cbl    — Daily transaction validation (reads daily tran, lookups xref/acct/card, writes validated)
CBTRN02C.cbl    — Transaction posting (validates, posts to tran file, updates acct/tcatbal, writes rejects)
CBTRN03C.cbl    — Transaction report (reads tran, looks up xref/type/category, writes report with totals)
CBACT04C.cbl    — Interest/fee calculator (reads tcatbal, xref, discgrp, acct, tran; computes interest+fees, writes tx)
CBSTM03A.cbl   — Statement generator (HTML output, reads xref→cust→acct→tran, calls CBSTM03B)
CBSTM03B.cbl   — Statement subroutine (indexed file I/O for tran/xref/cust/acct)
```

**Target path**: `migration-output/carddemo/src/main/java/com/carddemo/batch/`
- `config/TransactionValidationJobConfig.java`, `config/TransactionPostingJobConfig.java`
- `config/InterestCalculationJobConfig.java`, `config/StatementGenerationJobConfig.java`
- `config/TransactionReportJobConfig.java`
- `processor/TransactionValidationProcessor.java`, `processor/TransactionPostingProcessor.java`
- `processor/InterestCalculationProcessor.java`, `processor/StatementProcessor.java`
- `writer/StatementHtmlWriter.java`, `writer/ReportWriter.java`
- `service/InterestCalculationService.java`

**Depends on**: MVP 0 (Foundation — entities), MVP 2 (Account — balance updates), MVP 5 (Batch infrastructure — readers, writers)

**Why this order**: These are the most complex batch programs with the highest business logic density. They depend on the batch infrastructure established in MVP 5 (readers, writers, job configurations) and require all entity types to be stable. CBSTM03A+B must stay together (caller + subroutine). CBACT04C belongs here because it performs financial calculations (interest/fees) that update transaction and account balance data.

**Functional Spec**:

Purpose: Replace the core batch transaction processing pipeline — validation, posting, interest calculation, statement generation, and reporting — with Spring Batch jobs.

Key transforms:
- CBTRN01C sequential read + multi-file lookup → `TransactionValidationJob`: `FlatFileItemReader` (or `JdbcCursorItemReader`) → `TransactionValidationProcessor` (lookups xref, account, card via repositories; validates amount, date, card status) → `RepositoryItemWriter` (valid) + `FlatFileItemWriter` (rejects)
- CBTRN02C post + update balances → `TransactionPostingJob`: read validated → `TransactionPostingProcessor` (posts transaction, updates `Account.balance`, updates category balance) → `RepositoryItemWriter` with `@Transactional` per chunk
- CBACT04C interest/fee computation → `InterestCalculationJob`: read category balances → `InterestCalculationProcessor` (look up discount group rates, compute daily interest, compute fees per category rules) → write interest/fee transactions via `RepositoryItemWriter`
- CBACT04C `COMPUTE ... ROUNDED` → `BigDecimal.multiply().setScale(2, RoundingMode.HALF_UP)`
- CBSTM03A+B → `StatementGenerationJob`: partitioned by account range → `StatementProcessor` (xref → customer → account → transactions for period → HTML template) → `StatementHtmlWriter` (generates HTML files)
- CBSTM03B `CALL` subroutine → Merged into `StatementProcessor` (private methods for file I/O replaced by repository calls)
- CBTRN03C → `TransactionReportJob`: read transactions → enrich with xref/type/category lookups → `ReportWriter` (generates report with category subtotals, grand totals, record counts)

Renamed exports table:

| Old Symbol | New Symbol | Old Import Path | New Import Path |
|---|---|---|---|
| CBTRN01C (program) | TransactionValidationJob | JCL `EXEC PGM=CBTRN01C` | `com.carddemo.batch.config.TransactionValidationJobConfig` |
| CBTRN02C (program) | TransactionPostingJob | JCL `EXEC PGM=CBTRN02C` | `com.carddemo.batch.config.TransactionPostingJobConfig` |
| CBACT04C (program) | InterestCalculationJob | JCL `EXEC PGM=CBACT04C` | `com.carddemo.batch.config.InterestCalculationJobConfig` |
| CBSTM03A (program) | StatementGenerationJob | JCL `EXEC PGM=CBSTM03A` | `com.carddemo.batch.config.StatementGenerationJobConfig` |
| CBSTM03B (subroutine) | StatementProcessor (merged) | `CALL 'CBSTM03B'` | `com.carddemo.batch.processor.StatementProcessor` |
| CBTRN03C (program) | TransactionReportJob | JCL `EXEC PGM=CBTRN03C` | `com.carddemo.batch.config.TransactionReportJobConfig` |

Acceptance criteria:
- `TransactionValidationJob`: processes 10,000 test transactions, correctly accepts valid and rejects invalid (missing account, inactive card, zero amount)
- `TransactionPostingJob`: posts validated transactions and updates account balances; balance after posting matches COBOL output for identical input
- `InterestCalculationJob`: computes interest for 1,000 accounts; BigDecimal results match COBOL COMPUTE ROUNDED output to the penny (zero drift over 1000 records)
- `StatementGenerationJob`: produces HTML statements matching original format; customer name, account summary, transaction details, period totals all correct
- `TransactionReportJob`: produces report with correct category subtotals and grand total; record counts match input
- All batch jobs: restartable from last committed chunk, job execution metadata persisted
- All financial calculations: precision regression test comparing Java vs COBOL output for identical datasets
- Batch job pipeline can run in sequence: validation → posting → interest calc → statement gen → report
- Jobs triggerable from REST endpoint (MVP 4) and schedulable via Kubernetes CronJob

---

## File Assignment Verification

### All 61 files assigned:

| MVP | Programs | Copybooks | Total |
|---|---|---|---|
| MVP 0: Foundation | 2 (CSUTLDTC, COBSWAIT) | 30 (all copybooks) | 32 |
| MVP 1: Auth & Users | 7 (COSGN00C, COMEN01C, COADM01C, COUSR00C-03C) | 0 | 7 |
| MVP 2: Accounts | 2 (COACTVWC, COACTUPC) | 0 | 2 |
| MVP 3: Cards | 3 (COCRDLIC, COCRDSLC, COCRDUPC) | 0 | 3 |
| MVP 4: Transaction Online | 5 (COTRN00C, COTRN01C, COTRN02C, COBIL00C, CORPT00C) | 0 | 5 |
| MVP 5: Batch Readers/Export | 5 (CBACT01C, CBACT02C, CBACT03C, CBCUS01C, CBEXPORT) | 0 | 5 |
| MVP 6: Batch Tran Processing | 6 (CBTRN01C, CBTRN02C, CBTRN03C, CBACT04C, CBSTM03A, CBSTM03B) | 0 | 6 |
| **TOTAL** | **30** | **30** | **60** |

**Note**: The input states "61 COBOL files (30 .cbl programs + 30 .cpy copybooks + possibly 1 more)". All 30 programs and 30 copybooks are accounted for above (60 files). If a 61st file exists, it should be identified and assigned to the appropriate MVP based on its content.

---

## Dependency Graph

```
MVP 0: Foundation
    │
    ├──→ MVP 1: Auth & Users
    │       │
    │       ├──→ MVP 2: Accounts
    │       │       │
    │       │       ├──→ MVP 3: Cards
    │       │       │       │
    │       │       │       └──→ MVP 4: Transaction Online
    │       │       │
    │       │       └──→ MVP 6: Batch Tran Processing
    │       │
    │       └──→ MVP 4: Transaction Online
    │
    └──→ MVP 5: Batch Readers/Export
            │
            └──→ MVP 6: Batch Tran Processing
```

## Execution Timeline (Estimated)

| MVP | Duration | Parallel? | Notes |
|---|---|---|---|
| MVP 0 | 2 weeks | No | Must complete first |
| MVP 1 | 1.5 weeks | No | Needed for all protected endpoints |
| MVP 2 | 1 week | After MVP 1 | Core domain |
| MVP 3 | 1 week | After MVP 2 | Can overlap with MVP 5 start |
| MVP 5 | 1.5 weeks | After MVP 0 | **Parallel with MVPs 2-3** (batch has no auth dependency for logic) |
| MVP 4 | 1.5 weeks | After MVP 3 | Online transaction features |
| MVP 6 | 2.5 weeks | After MVP 5 + MVP 2 | Most complex batch logic |
| **Total** | **~8 weeks** | | With parallelization of batch track |

**Critical path**: MVP 0 → MVP 1 → MVP 2 → MVP 3 → MVP 4 (online track) = 7 weeks
**Parallel track**: MVP 0 → MVP 5 → MVP 6 (batch track) = 6 weeks
**With parallelization**: ~8 weeks total (batch track runs alongside online track)
