# Reverse Engineering Document Template
Reference: Carddemo_modernization_architecture_report_itr_2.docx

## Sections (14 chapters matching the reference template)

### A01: System Architecture Overview
- Processing pattern classification (Batch, Interactive/CICS, MQ async, DB2, File-based)
- Logical layer diagram (Presentation → Business Logic → Data Access → Storage)
- Technology indicators (EXEC CICS, MQ APIs, embedded SQL, file I/O patterns)
- Program → processing pattern mapping table
- Known control & lifecycle conventions (paragraph bands, file open→read→close)
- Risks and migration considerations

### A02: Component Architecture
- Component tree (PERFORM hierarchy mapped to components)
- Inter-program call graph (observed edges, coupling classification)
- Dispatcher module relationships
- Coupling drivers and technical risks
- Known mappings between business data and components

### A03: Data Architecture
- Physical data model (FD-level file records and key fields)
- Field hierarchies and in-memory record structures
- COBOL → target type mapping rules (PIC X→VARCHAR, PIC 9→DECIMAL, COMP-3, REDEFINES, 88-levels, OCCURS)
- REDEFINES semantics and handling
- Packed decimal (COMP-3) handling
- Data quality / constraints from 88-levels

### A04: Integration Architecture
- COMMAREA-based CICS API contracts
- CICS synchronous transfers (XCTL/LINK)
- CALL-based subroutine invocations
- MQ async messaging
- Shared files / VSAM / batch file access
- Mechanism → modernization pattern mapping table

### A05: Technology Stack Profile
- CICS Transaction Server
- IBM MQ
- VSAM KSDS
- Batch file / Sequential I/O
- BMS (Basic Mapping Support)
- DB2 (embedded SQL)
- LE Runtime / System abend handler

### A06: Database Architecture & SQL
- CRUD matrix (Table × Program × Operation)
- Cursor patterns and host variables
- SQL return handling (SQLCODE/SQLSTATE)
- Transaction boundary semantics

### A07: Processing Flow & Control
- Batch processing flows (per-program flow description)
- CICS pseudo-conversational flows
- Control break patterns
- Screen navigation (PF-key flows)
- Program chaining and transfer of control

### A08: Error Handling & Recovery
- Abend patterns (9999-ABEND-PROGRAM, CEE3ABD)
- IO-status normalization (9910-DISPLAY-IO-STATUS)
- CICS RESP/RESP2 handling
- MQ condition/reason code handling
- Recovery and restart semantics

### A10: Performance & Scalability
- Throughput characteristics
- Batch window analysis
- Concurrency patterns (file sharing, DB2 locking)
- Resource utilization indicators

### A11: Technical Debt Assessment
- Complexity hotspots (cyclomatic complexity top N)
- Dead code inventory
- Coupling risks (high fan-in/fan-out modules)
- Hardcoded values and magic numbers
- Code duplication indicators

### A12: Target State Architecture
- Recommended target patterns (per source pattern)
- Service decomposition proposal
- Data migration strategy
- Integration modernization paths

### A13: Non-Functional Requirements
- Security requirements (from auth patterns, PCI data)
- Compliance requirements (audit trail patterns, SOX indicators)
- Availability requirements (from error handling patterns)
- Data integrity requirements (from transaction patterns)

### A14: Architecture Risks & Gaps
- Migration risks ranked by severity
- Unknown areas requiring further analysis
- Dependencies on external systems
- Data quality risks
