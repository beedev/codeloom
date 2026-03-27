# CardDemo Migration Architecture Document

## Mainframe COBOL/CICS to Java 21 + Spring Boot 3

**Project**: CardDemo
**Source**: 61 COBOL files (30 programs, 30+ copybooks), 601 code units
**Target**: Java 21, Spring Boot 3.x, Spring Data JPA, PostgreSQL
**Strategy**: Big Bang
**Deployment**: Kubernetes
**Compliance**: PCI-DSS, SOX, GDPR
**Volume**: ~1M transactions/day

---

## 1. Source Summary

### 1.1 Framework Patterns

**CICS Online (pseudo-conversational)**:
The 17 online programs (CO*) follow CICS pseudo-conversational pattern: each program receives control via `EXEC CICS RECEIVE MAP`, processes the input, updates VSAM files, builds an output map, then returns control with `EXEC CICS RETURN TRANSID`. State is carried between interactions through a COMMAREA (`COCOM01Y.cpy`) passed on each `RETURN`. PF-key navigation (PF3=back, PF7/PF8=page up/down) drives screen flow.

**Batch (sequential file processing)**:
The 11 batch programs (CB*) follow standard COBOL batch patterns: `OPEN INPUT/OUTPUT`, `READ ... AT END`, `PERFORM UNTIL EOF`, `WRITE`, `CLOSE`. Files are VSAM KSDS (keyed), ESDS (sequential), or flat fixed-format. Programs chain via JCL job steps sharing intermediate files.

**Shared data contracts**:
Copybooks define record layouts shared across programs. `COCOM01Y` (COMMAREA) is the universal data-passing mechanism for online programs. Domain-specific copybooks (`CVACT01Y`, `CVTRA05Y`, etc.) define file record layouts used by both batch and online.

### 1.2 Data Patterns

| VSAM File / Logical Entity | Copybook(s) | Key | Usage |
|---|---|---|---|
| Account master | CVACT01Y | Account ID | R/W by batch + online |
| Account cross-reference | CVACT03Y | Card number → Account ID | Lookup bridge |
| Card record | CVACT02Y | Card number | R/W online, R batch |
| Customer master | CVCUS01Y | Customer ID | R/W batch + online |
| Credit card record | CVCRD01Y | Card number | R/W online |
| Transaction file | CVTRA05Y + CVTRA01Y-07Y | Tran ID / composite | R/W batch, R online |
| User security | CSUSR01Y | User ID | Auth, user CRUD |
| Daily transaction input | (inline layouts) | Sequential | Batch validation input |
| Transaction category/balance | (inline layouts) | Category code | Interest/fee calc |

**File I/O patterns**: VSAM `READ` with key, `START`/`READNEXT` for browse, `REWRITE` for update, `WRITE` for insert. Batch programs use sequential `READ ... AT END` loops.

### 1.3 Screen Flow (CICS BMS)

```
COSGN00C (Sign-on)
    │
    ├── COMEN01C (Main Menu)
    │       ├── COACTVWC (Account View)
    │       ├── COACTUPC (Account Update)
    │       ├── COCRDLIC (Card List) → COCRDSLC (Card Select) → COCRDUPC (Card Update)
    │       ├── COBIL00C (Billing/Statement View)
    │       ├── COTRN00C (Transaction List) → COTRN01C (Tran Detail)
    │       │                               → COTRN02C (Tran Add)
    │       └── CORPT00C (Report Submission)
    │
    └── COADM01C (Admin Menu)
            ├── COUSR00C (User List)
            ├── COUSR01C (User Add)
            ├── COUSR02C (User Update)
            └── COUSR03C (User Delete)
```

Navigation is PF-key driven. PF3 returns to prior screen. PF7/PF8 pages through browse lists. Each screen is a BMS map with fields mapped to COMMAREA sections.

---

## 2. Target Architecture

### 2.1 Layered Spring Boot Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway / Ingress                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ REST         │  │ REST         │  │ REST         │          │
│  │ Controllers  │  │ Controllers  │  │ Controllers  │          │
│  │ (Account)    │  │ (Card)       │  │ (User)       │  ...     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                   │
│  ┌──────▼──────────────────▼──────────────────▼───────────────┐ │
│  │                    Service Layer                             │ │
│  │  AccountService, CardService, TransactionService,           │ │
│  │  UserService, AuthService, ReportService                    │ │
│  └──────┬──────────────────┬──────────────────┬───────────────┘ │
│         │                  │                  │                   │
│  ┌──────▼──────────────────▼──────────────────▼───────────────┐ │
│  │                  Repository Layer (JPA)                      │ │
│  │  AccountRepository, CardRepository, TransactionRepository,  │ │
│  │  CustomerRepository, UserRepository                         │ │
│  └──────┬──────────────────┬──────────────────┬───────────────┘ │
│         │                  │                  │                   │
│  ┌──────▼──────────────────▼──────────────────▼───────────────┐ │
│  │                   PostgreSQL + pgvector                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Spring Batch (Batch Processing)                 │ │
│  │  AccountBatchJob, TransactionValidationJob,                 │ │
│  │  TransactionPostingJob, InterestCalcJob,                    │ │
│  │  StatementGenerationJob, ExportJob, ReportJob               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Cross-Cutting: Security (Spring Security + JWT), Audit,        │
│  Encryption (PCI-DSS), Logging, Exception Handling               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Runtime | Java 21 (LTS) | Virtual threads for high concurrency |
| Framework | Spring Boot 3.3.x | Production-grade, enterprise standard |
| Web | Spring Web MVC | REST controllers replacing BMS screens |
| Persistence | Spring Data JPA + Hibernate 6 | Replaces VSAM keyed access |
| Database | PostgreSQL 17 | VSAM replacement, ACID, pgcrypto for encryption |
| Batch | Spring Batch 5.x | Direct replacement for COBOL batch programs |
| Security | Spring Security 6 + JWT | Replaces CICS sign-on + RACF |
| Validation | Jakarta Bean Validation | Replaces inline COBOL field validation |
| Migration | Flyway | Schema versioning |
| Build | Gradle 8 (Kotlin DSL) | Modern build tooling |
| Containerization | Docker + Kubernetes | Per requirements |
| Observability | Micrometer + Prometheus + Grafana | Replaces CICS SMF/log records |

### 2.3 Package Structure

```
com.carddemo
├── config/                    # Spring configuration classes
│   ├── SecurityConfig.java
│   ├── BatchConfig.java
│   ├── EncryptionConfig.java
│   └── AuditConfig.java
├── controller/                # REST controllers (replace BMS maps)
│   ├── AuthController.java
│   ├── AccountController.java
│   ├── CardController.java
│   ├── TransactionController.java
│   ├── UserController.java
│   └── ReportController.java
├── service/                   # Business logic (replace COBOL paragraphs)
│   ├── AuthService.java
│   ├── AccountService.java
│   ├── CardService.java
│   ├── TransactionService.java
│   ├── CustomerService.java
│   ├── UserService.java
│   ├── BillingService.java
│   ├── InterestCalculationService.java
│   └── ReportService.java
├── repository/                # JPA repositories (replace VSAM I/O)
│   ├── AccountRepository.java
│   ├── CardRepository.java
│   ├── TransactionRepository.java
│   ├── CustomerRepository.java
│   ├── UserRepository.java
│   └── CrossReferenceRepository.java
├── entity/                    # JPA entities (replace copybook record layouts)
│   ├── Account.java
│   ├── Card.java
│   ├── Transaction.java
│   ├── Customer.java
│   ├── User.java
│   ├── CrossReference.java
│   ├── TransactionCategory.java
│   └── DiscountGroup.java
├── dto/                       # DTOs (replace COMMAREA sections)
│   ├── CommArea.java
│   ├── AccountDto.java
│   ├── CardDto.java
│   ├── TransactionDto.java
│   ├── CustomerDto.java
│   ├── UserDto.java
│   ├── PagedResponse.java
│   └── request/
│       ├── SignOnRequest.java
│       ├── AccountUpdateRequest.java
│       ├── CardUpdateRequest.java
│       ├── TransactionAddRequest.java
│       └── UserCreateRequest.java
├── batch/                     # Spring Batch jobs (replace CB* programs)
│   ├── job/
│   │   ├── AccountFileJob.java
│   │   ├── TransactionValidationJob.java
│   │   ├── TransactionPostingJob.java
│   │   ├── InterestCalculationJob.java
│   │   ├── StatementGenerationJob.java
│   │   ├── DataExportJob.java
│   │   └── TransactionReportJob.java
│   ├── reader/
│   │   ├── AccountItemReader.java
│   │   ├── CardItemReader.java
│   │   ├── CustomerItemReader.java
│   │   └── TransactionItemReader.java
│   ├── processor/
│   │   ├── TransactionValidationProcessor.java
│   │   ├── TransactionPostingProcessor.java
│   │   ├── InterestCalculationProcessor.java
│   │   └── StatementProcessor.java
│   └── writer/
│       ├── TransactionItemWriter.java
│       ├── StatementHtmlWriter.java
│       ├── DataExportWriter.java
│       └── ReportWriter.java
├── validation/                # Field validators (replace inline COBOL checks)
│   ├── SsnValidator.java
│   ├── PhoneValidator.java
│   ├── StateCodeValidator.java
│   ├── ZipCodeValidator.java
│   └── FicoScoreValidator.java
├── util/                      # Utilities (replace utility programs)
│   ├── DateConversionUtil.java
│   └── StringProofUtil.java
├── exception/                 # Exception handling
│   ├── CardDemoException.java
│   ├── AccountNotFoundException.java
│   ├── InvalidTransactionException.java
│   └── GlobalExceptionHandler.java
├── audit/                     # SOX audit trail
│   ├── AuditEvent.java
│   ├── AuditRepository.java
│   └── AuditAspect.java
└── security/                  # PCI-DSS encryption, GDPR
    ├── CardDataEncryptor.java
    ├── PciDataMasker.java
    └── GdprDataHandler.java
```

---

## 3. Data Model Mapping

### 3.1 VSAM Files to JPA Entities

| VSAM File | Copybook | JPA Entity | Primary Key | Notes |
|---|---|---|---|---|
| Account master | CVACT01Y | `Account` | `accountId (Long)` | All numeric fields → BigDecimal for money |
| Account xref | CVACT03Y | `CrossReference` | `cardNumber (String)` → `accountId` | Lookup table, may become FK on Card |
| Card record | CVACT02Y | `Card` | `cardNumber (String)` | PCI: number encrypted at rest |
| Customer master | CVCUS01Y | `Customer` | `customerId (Long)` | GDPR: PII fields flagged |
| Credit card detail | CVCRD01Y | `CreditCard` | `cardNumber (String)` | May merge with Card entity |
| Transaction file | CVTRA05Y | `Transaction` | `transactionId (Long)` | Partitioned by date for volume |
| User security | CSUSR01Y | `User` | `userId (String)` | Password → bcrypt hash |
| Tran category | (inline) | `TransactionCategory` | `categoryCode` | Reference data |
| Discount group | (inline) | `DiscountGroup` | `groupCode` | Interest rate tiers |

### 3.2 COMMAREA to DTOs

The COMMAREA (`COCOM01Y`) is a monolithic data-passing structure used by all online programs. In Java, this decomposes into purpose-specific DTOs:

| COMMAREA Section | Java DTO | Usage |
|---|---|---|
| Header fields (program-name, tran-id) | Removed | Handled by Spring routing |
| User/auth fields | `SignOnRequest` / JWT claims | Authentication context |
| Account data fields | `AccountDto` | Account view/update |
| Card data fields | `CardDto` | Card list/select/update |
| Transaction data fields | `TransactionDto` | Transaction list/add |
| User admin fields | `UserDto` | User CRUD |
| Message area (CSMSG01Y/02Y) | `ApiResponse<T>` wrapper | Standardized error/success messages |
| Title/header (COTTL01Y) | Removed | Replaced by API metadata |
| Date area (CSDAT01Y) | `LocalDate` / `LocalDateTime` | Native Java date handling |
| Pagination fields | `PagedResponse<T>` | Spring Data `Page<T>` |

### 3.3 Entity Design Highlights

```java
@Entity
@Table(name = "accounts")
public class Account {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long accountId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "customer_id")
    private Customer customer;

    @Column(precision = 15, scale = 2)
    private BigDecimal balance;

    @Column(precision = 15, scale = 2)
    private BigDecimal creditLimit;

    @Enumerated(EnumType.STRING)
    private AccountStatus status;

    // Audit fields (SOX)
    @CreatedDate private LocalDateTime createdAt;
    @LastModifiedDate private LocalDateTime updatedAt;
    @CreatedBy private String createdBy;
    @LastModifiedBy private String updatedBy;
}

@Entity
@Table(name = "cards")
public class Card {
    @Id
    private Long id;

    @Convert(converter = EncryptedStringConverter.class)  // PCI-DSS
    @Column(name = "card_number")
    private String cardNumber;

    @Column(name = "card_number_masked")  // Last 4 only
    private String cardNumberMasked;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "account_id")
    private Account account;

    private LocalDate expiryDate;

    @Enumerated(EnumType.STRING)
    private CardStatus status;
}
```

---

## 4. Key Transforms

### 4.1 CICS Commands to Spring

| COBOL/CICS Pattern | Java/Spring Equivalent |
|---|---|
| `EXEC CICS RECEIVE MAP` | `@PostMapping` / `@PutMapping` with `@RequestBody` |
| `EXEC CICS SEND MAP` | Return `ResponseEntity<T>` |
| `EXEC CICS RETURN TRANSID` | Stateless — no equivalent needed (REST is inherently stateless) |
| `EXEC CICS READ FILE` | `repository.findById(key)` |
| `EXEC CICS READ FILE ... UPDATE` | `@Transactional` + `repository.findById()` + modify + auto-flush |
| `EXEC CICS REWRITE` | JPA dirty checking within `@Transactional` |
| `EXEC CICS WRITE FILE` | `repository.save(entity)` |
| `EXEC CICS DELETE FILE` | `repository.deleteById(key)` |
| `EXEC CICS STARTBR / READNEXT / ENDBR` | `repository.findAllBy...()` with `Pageable` |
| `EXEC CICS LINK PROGRAM` | Direct method call on injected `@Service` |
| `EXEC CICS XCTL` | Controller redirect / service delegation |
| `EXEC CICS HANDLE CONDITION` | `try/catch` + `@ControllerAdvice` |
| `EXEC CICS WRITEQ TDQ` | Spring `ApplicationEvent` or message queue |
| `EXEC CICS START` (batch submission) | Spring Batch `JobLauncher.run()` |
| COMMAREA passing | Method parameters / DTOs |
| `EVALUATE TRUE` | `switch` expression (Java 21) |
| `PERFORM paragraph` | Private method call |
| `PERFORM ... VARYING` | `for` / `IntStream.range()` |
| `INSPECT ... TALLYING` | `String.chars().filter()` |
| `STRING ... DELIMITED BY` | `String.join()` / `StringBuilder` |
| `COMPUTE` with `ROUNDED` | `BigDecimal` with `RoundingMode.HALF_UP` |

### 4.2 VSAM to JPA

| VSAM Operation | JPA Equivalent |
|---|---|
| `OPEN INPUT file` | Repository auto-managed by Spring |
| `OPEN OUTPUT file` | Spring Batch `ItemWriter` |
| `READ file INTO record KEY IS key` | `repository.findById(key)` / `repository.findByCardNumber(num)` |
| `START file KEY >= key` + `READNEXT` | `repository.findByAccountIdGreaterThanEqual(key, pageable)` |
| `WRITE record FROM area` | `repository.save(entity)` |
| `REWRITE record FROM area` | Modify entity in `@Transactional`, auto-flushed |
| `DELETE file RECORD` | `repository.delete(entity)` |
| Sequential file read (batch) | Spring Batch `RepositoryItemReader` or `JdbcCursorItemReader` |
| Fixed-format file write (batch) | Spring Batch `FlatFileItemWriter` (for export compatibility) |

### 4.3 BMS Maps to REST Endpoints

| BMS Screen (Program) | REST Endpoint | HTTP Method |
|---|---|---|
| Sign-on (COSGN00C) | `POST /api/auth/login` | POST |
| Main menu (COMEN01C) | `GET /api/menu` | GET |
| Admin menu (COADM01C) | `GET /api/admin/menu` | GET |
| Account view (COACTVWC) | `GET /api/accounts/{id}` | GET |
| Account update (COACTUPC) | `PUT /api/accounts/{id}` | PUT |
| Card list (COCRDLIC) | `GET /api/cards?page=&size=&filter=` | GET |
| Card select (COCRDSLC) | `GET /api/cards/{cardNumber}` | GET |
| Card update (COCRDUPC) | `PUT /api/cards/{cardNumber}` | PUT |
| Billing view (COBIL00C) | `GET /api/accounts/{id}/billing` | GET |
| Transaction list (COTRN00C) | `GET /api/transactions?page=&size=` | GET |
| Transaction detail (COTRN01C) | `GET /api/transactions/{id}` | GET |
| Transaction add (COTRN02C) | `POST /api/transactions` | POST |
| Report submit (CORPT00C) | `POST /api/reports/submit` | POST |
| User list (COUSR00C) | `GET /api/users?page=&size=` | GET |
| User add (COUSR01C) | `POST /api/users` | POST |
| User update (COUSR02C) | `PUT /api/users/{id}` | PUT |
| User delete (COUSR03C) | `DELETE /api/users/{id}` | DELETE |

### 4.4 Batch Program to Spring Batch Job Mapping

| COBOL Batch | Spring Batch Job | Pattern |
|---|---|---|
| CBACT01C (Account reader) | `AccountFileJob` | Chunk-oriented: DB read → flat file write |
| CBACT02C (Card reader) | Part of `AccountFileJob` step | Additional step in same job |
| CBACT03C (Xref reader) | Part of `AccountFileJob` step | Additional step in same job |
| CBCUS01C (Customer reader) | Part of `DataExportJob` | Reader step |
| CBTRN01C (Tran validation) | `TransactionValidationJob` | Chunk: read daily → validate → write valid/reject |
| CBTRN02C (Tran posting) | `TransactionPostingJob` | Chunk: read validated → post → update balances |
| CBACT04C (Interest/fee calc) | `InterestCalculationJob` | Chunk: read accounts → compute → write transactions |
| CBTRN03C (Tran report) | `TransactionReportJob` | Chunk: read trans → enrich → write report |
| CBSTM03A+B (Statement gen) | `StatementGenerationJob` | Chunk: xref → cust+acct+tran → HTML output |
| CBEXPORT (Data export) | `DataExportJob` | Multi-step: each entity type is a step |

---

## 5. Dependency Injection Strategy

### 5.1 Spring Stereotype Annotations

| COBOL Pattern | Spring Annotation | Java Class |
|---|---|---|
| CICS program (screen handler) | `@RestController` | Controllers in `controller/` |
| Business logic paragraphs | `@Service` | Services in `service/` |
| VSAM file I/O paragraphs | `@Repository` | Repositories in `repository/` |
| Copybook record layouts | `@Entity` | Entities in `entity/` |
| COMMAREA sections | Plain POJO / `record` | DTOs in `dto/` |
| Utility subroutines | `@Component` or static utility | Utils in `util/` |
| Batch programs | `@Configuration` + `@Bean` | Batch configs in `batch/` |

### 5.2 Injection Flow

```
@RestController AccountController
    └── @Autowired AccountService
            ├── @Autowired AccountRepository (extends JpaRepository<Account, Long>)
            ├── @Autowired CrossReferenceRepository
            ├── @Autowired CustomerRepository
            └── @Autowired CardDataEncryptor
```

All injection is constructor-based (no field injection). Services are the boundary for `@Transactional`. Repositories are Spring Data interfaces — no implementation classes needed.

### 5.3 Cross-Cutting via AOP

```java
@Aspect @Component
public class AuditAspect {
    // Intercepts all @Service methods for SOX audit trail
    @AfterReturning("@within(org.springframework.stereotype.Service)")
    public void auditServiceCall(JoinPoint jp) { ... }
}
```

---

## 6. Target File Structure

```
migration-output/carddemo/
├── build.gradle.kts
├── settings.gradle.kts
├── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── secret.yaml
├── src/
│   ├── main/
│   │   ├── java/com/carddemo/
│   │   │   ├── CardDemoApplication.java
│   │   │   ├── config/
│   │   │   │   ├── SecurityConfig.java
│   │   │   │   ├── BatchConfig.java
│   │   │   │   ├── EncryptionConfig.java
│   │   │   │   ├── AuditConfig.java
│   │   │   │   └── JpaConfig.java
│   │   │   ├── controller/
│   │   │   │   ├── AuthController.java
│   │   │   │   ├── AccountController.java
│   │   │   │   ├── CardController.java
│   │   │   │   ├── TransactionController.java
│   │   │   │   ├── UserController.java
│   │   │   │   └── ReportController.java
│   │   │   ├── service/
│   │   │   │   ├── AuthService.java
│   │   │   │   ├── AccountService.java
│   │   │   │   ├── CardService.java
│   │   │   │   ├── TransactionService.java
│   │   │   │   ├── CustomerService.java
│   │   │   │   ├── UserService.java
│   │   │   │   ├── BillingService.java
│   │   │   │   ├── InterestCalculationService.java
│   │   │   │   └── ReportService.java
│   │   │   ├── repository/
│   │   │   │   ├── AccountRepository.java
│   │   │   │   ├── CardRepository.java
│   │   │   │   ├── CreditCardRepository.java
│   │   │   │   ├── TransactionRepository.java
│   │   │   │   ├── CustomerRepository.java
│   │   │   │   ├── UserRepository.java
│   │   │   │   ├── CrossReferenceRepository.java
│   │   │   │   ├── TransactionCategoryRepository.java
│   │   │   │   ├── DiscountGroupRepository.java
│   │   │   │   └── AuditRepository.java
│   │   │   ├── entity/
│   │   │   │   ├── Account.java
│   │   │   │   ├── Card.java
│   │   │   │   ├── CreditCard.java
│   │   │   │   ├── Transaction.java
│   │   │   │   ├── Customer.java
│   │   │   │   ├── User.java
│   │   │   │   ├── CrossReference.java
│   │   │   │   ├── TransactionCategory.java
│   │   │   │   ├── DiscountGroup.java
│   │   │   │   ├── AuditEvent.java
│   │   │   │   └── enums/
│   │   │   │       ├── AccountStatus.java
│   │   │   │       ├── CardStatus.java
│   │   │   │       └── TransactionType.java
│   │   │   ├── dto/
│   │   │   │   ├── AccountDto.java
│   │   │   │   ├── CardDto.java
│   │   │   │   ├── TransactionDto.java
│   │   │   │   ├── CustomerDto.java
│   │   │   │   ├── UserDto.java
│   │   │   │   ├── PagedResponse.java
│   │   │   │   ├── ApiResponse.java
│   │   │   │   └── request/
│   │   │   │       ├── SignOnRequest.java
│   │   │   │       ├── AccountUpdateRequest.java
│   │   │   │       ├── CardUpdateRequest.java
│   │   │   │       ├── TransactionAddRequest.java
│   │   │   │       ├── UserCreateRequest.java
│   │   │   │       └── ReportSubmitRequest.java
│   │   │   ├── batch/
│   │   │   │   ├── config/
│   │   │   │   │   ├── AccountFileJobConfig.java
│   │   │   │   │   ├── TransactionValidationJobConfig.java
│   │   │   │   │   ├── TransactionPostingJobConfig.java
│   │   │   │   │   ├── InterestCalculationJobConfig.java
│   │   │   │   │   ├── StatementGenerationJobConfig.java
│   │   │   │   │   ├── DataExportJobConfig.java
│   │   │   │   │   └── TransactionReportJobConfig.java
│   │   │   │   ├── reader/
│   │   │   │   ├── processor/
│   │   │   │   └── writer/
│   │   │   ├── validation/
│   │   │   │   ├── SsnValidator.java
│   │   │   │   ├── PhoneValidator.java
│   │   │   │   ├── StateCodeValidator.java
│   │   │   │   ├── ZipCodeValidator.java
│   │   │   │   └── FicoScoreValidator.java
│   │   │   ├── util/
│   │   │   │   ├── DateConversionUtil.java
│   │   │   │   └── StringProofUtil.java
│   │   │   ├── exception/
│   │   │   │   ├── CardDemoException.java
│   │   │   │   ├── AccountNotFoundException.java
│   │   │   │   ├── InvalidTransactionException.java
│   │   │   │   └── GlobalExceptionHandler.java
│   │   │   ├── audit/
│   │   │   │   └── AuditAspect.java
│   │   │   └── security/
│   │   │       ├── CardDataEncryptor.java
│   │   │       ├── PciDataMasker.java
│   │   │       ├── GdprDataHandler.java
│   │   │       └── JwtTokenProvider.java
│   │   └── resources/
│   │       ├── application.yaml
│   │       ├── application-dev.yaml
│   │       ├── application-prod.yaml
│   │       └── db/migration/
│   │           ├── V1__create_schema.sql
│   │           ├── V2__seed_reference_data.sql
│   │           └── V3__create_audit_tables.sql
│   └── test/
│       └── java/com/carddemo/
│           ├── service/
│           ├── controller/
│           ├── batch/
│           └── repository/
└── SYMBOLS.md
```

---

## 7. Volume-Aware Design (1M transactions/day)

### 7.1 Transaction Throughput

- 1M tx/day = ~12 tx/sec average, ~50 tx/sec peak (4x multiplier)
- This is moderate volume — Spring Boot handles this without exotic architecture
- Java 21 virtual threads eliminate thread pool bottlenecks for I/O-bound operations

### 7.2 Batch Processing Strategy

**Spring Batch with chunk-oriented processing**:

```java
@Bean
public Step transactionPostingStep() {
    return new StepBuilder("transactionPosting", jobRepository)
        .<TransactionInput, Transaction>chunk(500, transactionManager)  // 500 per chunk
        .reader(validatedTransactionReader())
        .processor(transactionPostingProcessor())
        .writer(transactionWriter())
        .faultTolerant()
        .retryLimit(3)
        .retry(OptimisticLockingFailureException.class)
        .build();
}
```

**Key batch design decisions**:
- **Chunk size**: 500 (tuned for PostgreSQL batch insert performance)
- **Partitioning**: `StatementGenerationJob` partitions by account range for parallelism
- **Streaming**: `JdbcCursorItemReader` for large result sets (no full-load into memory)
- **Scheduling**: Spring Scheduler / Kubernetes CronJob for nightly batch runs
- **Restart**: Spring Batch `ExecutionContext` enables restart from last committed chunk

### 7.3 Database Optimization

- **Transaction table partitioning**: Range partition by `transaction_date` (monthly)
- **Indexes**: Composite index on `(account_id, transaction_date)` for statement generation
- **Connection pool**: HikariCP with 20 connections (sufficient for ~50 concurrent queries)
- **Read replicas**: Optional for report queries (transaction list, billing view)

### 7.4 API Performance

- **Pagination**: All list endpoints use Spring Data `Pageable` (default page size 20)
- **Caching**: Spring Cache on reference data (transaction categories, discount groups)
- **Virtual threads**: `spring.threads.virtual.enabled=true` for effortless concurrency

---

## 8. Compliance Notes

### 8.1 PCI-DSS (Payment Card Industry)

| Requirement | Implementation |
|---|---|
| Encrypt card data at rest | `@Convert(converter = EncryptedStringConverter.class)` using AES-256-GCM via pgcrypto or application-level |
| Mask card numbers in display | `PciDataMasker`: store last-4 in `card_number_masked`, never return full PAN in API responses |
| Secure key management | Kubernetes Secrets + external vault (HashiCorp Vault recommended) |
| Access logging | `AuditAspect` logs all card data access with user, timestamp, action |
| Network segmentation | Kubernetes NetworkPolicy isolating card-processing pods |
| No card data in logs | Log filter strips card number patterns (`\\d{13,19}`) from all log output |

### 8.2 SOX (Sarbanes-Oxley)

| Requirement | Implementation |
|---|---|
| Audit trail for financial data | `AuditEvent` entity: who, what, when, old value, new value |
| Immutable audit records | `INSERT`-only audit table, no `UPDATE`/`DELETE` permissions |
| Segregation of duties | RBAC: admin vs. operator vs. viewer roles via Spring Security |
| Change control | All account balance changes logged with before/after values |
| Batch job auditing | Spring Batch `JobExecution` metadata retained for 7 years |

### 8.3 GDPR (General Data Protection Regulation)

| Requirement | Implementation |
|---|---|
| Right to erasure | `GdprDataHandler.anonymize(customerId)`: replaces PII with anonymized tokens |
| Data portability | `GET /api/customers/{id}/export` returns JSON of all customer data |
| Consent tracking | `Customer.consentDate` + `Customer.consentVersion` fields |
| Data minimization | API responses exclude unnecessary PII; DTOs are purpose-specific |
| Breach notification | Audit log anomaly detection triggers alert pipeline |

---

## 9. Risks and Mitigations

### 9.1 High Risk

| Risk | Impact | Mitigation |
|---|---|---|
| **CICS pseudo-conversational to stateless REST** | COBOL programs maintain state across interactions via COMMAREA. REST is stateless. Multi-step workflows (e.g., account update with confirmation) may lose context. | All state lives in the database. Multi-step flows use idempotent PUT operations. No server-side session state. Frontend holds transient UI state. |
| **VSAM keyed access semantics** | VSAM `START`/`READNEXT` browse with exact/generic key positioning has no direct JPA equivalent. Programs rely on key ordering (EBCDIC collation) for range scans. | JPA `findBy...GreaterThanEqual()` with `Sort` and `Pageable`. Verify collation differences between EBCDIC and PostgreSQL `C` locale. Write integration tests comparing sort order for edge cases. |
| **Decimal precision drift** | COBOL `PIC S9(n)V9(m)` has fixed decimal places. Java `double` or incorrect `BigDecimal` scale will cause penny drift in financial calculations (interest, fees, balances). | All monetary fields use `BigDecimal(precision=15, scale=2)`. All arithmetic uses `RoundingMode.HALF_UP` (matching COBOL `ROUNDED`). Write precision regression tests comparing COBOL and Java output for 1000+ sample calculations. |

### 9.2 Medium Risk

| Risk | Impact | Mitigation |
|---|---|---|
| **Report job submission (CORPT00C)** | Original submits JCL to CICS internal reader / TDQ. No equivalent mechanism in Java. | Replace with Spring Batch `JobLauncher.run()` triggered by REST endpoint. Report status via `GET /api/reports/{jobId}/status`. |
| **CBSTM03A+B subroutine coupling** | CBSTM03B is called by CBSTM03A via `CALL` with shared WORKING-STORAGE. Tight coupling through shared data areas. | Merge into single `StatementGenerationJob` with `StatementProcessor` class. The subroutine's indexed file I/O becomes repository calls within the same service. |
| **Batch file chaining via JCL** | Multiple batch programs share intermediate files via JCL DD statements. | Replace with Spring Batch multi-step jobs. Intermediate data stays in database (no file handoff). `TransactionValidationJob` output is `TransactionPostingJob` input via shared `Transaction` entity with status column. |
| **Date handling (CSUTLDTC)** | COBOL date utility handles Julian dates, century windowing, Lilian dates. Java `LocalDate` handles none of this natively. | `DateConversionUtil` must implement: Julian↔Gregorian, Lilian day number, COBOL date format strings (YYYYMMDD, YYMMDD, YYDDD). Write exhaustive tests for date boundary cases (leap years, century rollover). |

### 9.3 Lower Risk

| Risk | Impact | Mitigation |
|---|---|---|
| **EBCDIC to UTF-8** | Data migration may corrupt special characters in customer names/addresses. | Data migration script with explicit EBCDIC→UTF-8 conversion. Validate with sample data before cutover. |
| **PF-key navigation paradigm** | Users accustomed to PF-key workflow will need retraining. | Not a code risk — UI/UX concern. Document the screen-to-endpoint mapping for frontend team. |
| **Concurrent batch and online** | COBOL batch and CICS online can conflict on VSAM file locks. PostgreSQL MVCC eliminates this, but transaction isolation must be configured correctly. | Use `READ_COMMITTED` isolation (PostgreSQL default). Batch jobs use `@Transactional(propagation = REQUIRES_NEW)` per chunk. |

---

## Appendix A: Migration Validation Strategy

For each migrated program, validation follows this order:

1. **Unit tests**: Service-layer tests covering every paragraph's business logic
2. **Integration tests**: Repository tests with real PostgreSQL (Testcontainers)
3. **API tests**: Controller tests with MockMvc validating request/response contracts
4. **Batch tests**: Spring Batch Test with `JobLauncherTestUtils`
5. **Regression tests**: Compare Java output against COBOL output for identical input data sets
6. **Precision tests**: Financial calculation comparison (COBOL vs Java) for 1000+ records

## Appendix B: Data Migration Strategy

1. **Extract**: Unload VSAM files to flat files (utility or COBOL extract program)
2. **Transform**: Python/Spark ETL: EBCDIC→UTF-8, COBOL packed decimal→numeric, date format normalization
3. **Load**: `COPY` into PostgreSQL staging tables, validate counts/checksums
4. **Verify**: Row counts, checksum comparison, sample record spot-checks
5. **Cutover**: Swap DNS/ingress to Java application, keep mainframe in read-only standby for 72h
