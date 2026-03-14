/**
 * Custom Prism grammars for COBOL and JCL syntax highlighting.
 *
 * Registers languages on the shared Prism instance exported by prism-react-renderer
 * so that <Highlight language="cobol"> works out of the box.
 */
import { Prism } from 'prism-react-renderer';

// ── COBOL grammar ───────────────────────────────────────────────────

(Prism.languages as Record<string, unknown>).cobol = {
  comment: [
    // Fixed-format after cols 1-6 stripped: col 1 = indicator (* or /)
    { pattern: /^[*/].*$/m, greedy: true },
    // Inline *> comments (COBOL 2002+)
    { pattern: /\*>.*$/m, greedy: true },
  ],
  string: [
    { pattern: /"(?:[^"\r\n]|"")*"/, greedy: true },
    { pattern: /'(?:[^'\r\n]|'')*'/, greedy: true },
  ],
  // Division / Section / Paragraph headers
  'keyword division': {
    pattern:
      /\b(?:IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION\b/i,
    greedy: true,
  },
  'keyword section': {
    pattern:
      /\b(?:WORKING-STORAGE|LINKAGE|FILE|LOCAL-STORAGE|SCREEN|REPORT|CONFIGURATION|INPUT-OUTPUT|FILE-CONTROL)\s+SECTION\b/i,
    greedy: true,
  },
  keyword: {
    pattern:
      /\b(?:ACCEPT|ADD|ALTER|CALL|CANCEL|CLOSE|COMPUTE|CONTINUE|COPY|DELETE|DISPLAY|DIVIDE|ELSE|END-CALL|END-COMPUTE|END-EVALUATE|END-IF|END-MULTIPLY|END-PERFORM|END-READ|END-RETURN|END-REWRITE|END-SEARCH|END-START|END-STRING|END-SUBTRACT|END-UNSTRING|END-WRITE|EVALUATE|EXEC|EXIT|FD|GO|GOBACK|IF|INITIALIZE|INSPECT|MERGE|MOVE|MULTIPLY|NEXT|NOT|OPEN|PERFORM|READ|RELEASE|RETURN|REWRITE|SEARCH|SET|SORT|START|STOP|STRING|SUBTRACT|UNSTRING|WRITE|WHEN|THRU|THROUGH|UNTIL|VARYING|GIVING|USING|BY|INTO|FROM|TO|AT|END|ON|SIZE|ERROR|OVERFLOW|INVALID|KEY|ASCENDING|DESCENDING|CORRESPONDING|ROUNDED|REMAINDER|TALLYING|REPLACING|LEADING|TRAILING|ALL|REFERENCE|CONTENT|VALUE|OCCURS|DEPENDING|INDEXED|TIMES|WITH|TEST|BEFORE|AFTER|ALSO|TRUE|FALSE|OTHER|ZEROS|ZEROES|ZERO|SPACES|SPACE|HIGH-VALUES?|LOW-VALUES?|QUOTES?|UPON|ADVANCING|PAGE|LINE|LINES|RETURNING|OMITTED)\b/i,
    greedy: true,
  },
  // Data types and level numbers
  'class-name': {
    pattern:
      /\b(?:PIC|PICTURE)\s+(?:X|A|9|S9|V9|Z|B|\$|,|\.|\+|-|CR|DB|\*|0)+(?:\(\d+\))?(?:(?:V|P)(?:X|A|9|S9)+(?:\(\d+\))?)?/i,
    greedy: true,
  },
  'level-number': {
    pattern: /^\s*(?:0[1-9]|[1-4]\d|49|66|77|88)\b/m,
    alias: 'number',
  },
  // COBOL special registers / figurative constants
  builtin: {
    pattern:
      /\b(?:RETURN-CODE|SORT-RETURN|TALLY|WHEN-COMPILED|DEBUG-ITEM|LINAGE-COUNTER|ADDRESS\s+OF|LENGTH\s+OF)\b/i,
  },
  // File status / condition names
  'file-status': {
    pattern: /\b(?:FILE\s+STATUS|SELECT|ASSIGN|ORGANIZATION|ACCESS|RECORD\s+KEY|ALTERNATE\s+RECORD\s+KEY|RELATIVE\s+KEY|STATUS)\b/i,
    alias: 'keyword',
  },
  // Numeric literals (including COMP-3 etc.)
  number: {
    pattern: /\b(?:COMP(?:-[1-5])?|BINARY|PACKED-DECIMAL|COMPUTATIONAL(?:-[1-5])?|DISPLAY)\b|\b\d+(?:\.\d+)?\b/i,
  },
  // Data names (hyphenated identifiers)
  variable: {
    pattern: /\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b/i,
    alias: 'property',
  },
  // Paragraph names (standalone identifiers followed by period on next significant token)
  operator: /[=<>]+/,
  punctuation: /[().,:]/,
};

// ── JCL grammar ─────────────────────────────────────────────────────

(Prism.languages as Record<string, unknown>).jcl = {
  comment: [
    // JCL comment: //* in columns 1-3
    { pattern: /^\/\/\*.*$/m, greedy: true },
  ],
  // JCL statement labels (//NAME)
  'job-label': {
    pattern: /^\/\/[A-Z@#$][\w@#$]*/im,
    alias: 'function',
  },
  // JCL keywords (after //)
  keyword: {
    pattern:
      /\b(?:JOB|EXEC|DD|PROC|PEND|SET|IF|THEN|ELSE|ENDIF|INCLUDE|JCLLIB|OUTPUT|COMMAND|CNTL|ENDCNTL|XMIT|DEFINE|REPRO|DELETE|PRINT|LISTCAT|SORT|MERGE|OPTION|OUTREC|OUTFIL|INREC|INCLUDE|OMIT|SUM|FIELDS|COND|GDG|CLUSTER|DATA|INDEX|DSNAME|DSN|DISP|UNIT|VOL|SPACE|DCB|SYSOUT|SYSIN|DUMMY|MEMBER|PGM|PARM)\b/i,
    greedy: true,
  },
  // Symbolic parameters (&NAME)
  symbol: {
    pattern: /&[A-Z@#$][\w@#$]*/i,
    alias: 'variable',
  },
  // Dataset names (quoted or unqualified)
  string: [
    { pattern: /'(?:[^'\r\n]|'')*'/, greedy: true },
  ],
  number: /\b\d+\b/,
  operator: /[=(),]/,
  punctuation: /[.]/,
};

// Aliases so both "cobol" and "cbl" work, "jcl" and "job"
(Prism.languages as Record<string, unknown>).cbl = Prism.languages.cobol;
(Prism.languages as Record<string, unknown>).cob = Prism.languages.cobol;
(Prism.languages as Record<string, unknown>).job = Prism.languages.jcl;
