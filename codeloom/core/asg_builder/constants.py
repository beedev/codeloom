"""ASG Builder constants — shared regex patterns and exclusion sets.

Extracted from builder.py to enable domain-module imports without
circular dependencies.
"""

import re

# ── Regex patterns for structural edge detection ─────────────────

# Extract base classes from class signatures (Python/JS/TS fallback)
PYTHON_BASES_RE = re.compile(r"class\s+\w+\s*\(([^)]+)\)\s*:")
JS_EXTENDS_RE = re.compile(r"class\s+\w+\s+extends\s+(\w+)")

# Extract call targets from source
CALL_RE = re.compile(r"(?<!\w)(\w+)\s*\(")
# Qualified calls: obj.method() or Package.Class.method()
QUALIFIED_CALL_RE = re.compile(r"(?:\w+\.)+(\w+)\s*\(")

# ── SP call detection patterns ───────────────────────────────────

# Java: CallableStatement / prepareCall("{ call usp_Name(...) }")
JAVA_SP_CALL_RE = re.compile(
    r"""(?:prepareCall|callproc)\s*\(\s*["']\{?\s*call\s+(\w+)""",
    re.IGNORECASE,
)
# Java: @Procedure(name = "usp_Name") or @Procedure("usp_Name")
JAVA_PROC_ANNOTATION_RE = re.compile(
    r"""@Procedure\s*\(\s*(?:name\s*=\s*)?["'](\w+)["']""",
    re.IGNORECASE,
)
# C#: SqlCommand(..., "usp_Name") + CommandType.StoredProcedure
CSHARP_SP_RE = re.compile(
    r"""CommandText\s*=\s*["'](\w+)["']|new\s+SqlCommand\s*\(\s*["'](\w+)["']""",
    re.IGNORECASE,
)
# Python: cursor.callproc("usp_Name")
PYTHON_CALLPROC_RE = re.compile(
    r"""callproc\s*\(\s*["'](\w+)["']""",
    re.IGNORECASE,
)
# Generic: EXEC[UTE] usp_Name in string literals
EXEC_IN_STRING_RE = re.compile(
    r"""["'].*?\bEXEC(?:UTE)?\s+(\w+).*?["']""",
    re.IGNORECASE,
)

# ── COBOL call detection patterns ────────────────────────────────

# PERFORM paragraph-name / PERFORM section-name [THRU ...]
# COBOL names can contain hyphens: 1000-INIT, PROCESS-DATA
COBOL_PERFORM_RE = re.compile(r"\bPERFORM\s+([\w-]+)", re.IGNORECASE)

# PERFORM X THRU Y — captures the THRU end-target (second call edge)
COBOL_PERFORM_THRU_RE = re.compile(
    r"\bPERFORM\s+[\w-]+\s+THRU\s+([\w-]+)", re.IGNORECASE
)

# GO TO paragraph-name (unconditional jump — resolves to a call edge)
# GO TO DEPENDING ON uses multiple targets: GO TO p1 p2 p3 DEPENDING ON var
COBOL_GOTO_RE = re.compile(r"\bGO\s+TO\s+([\w-]+)", re.IGNORECASE)

# CALL 'program-name' USING ... (external program invocation)
COBOL_CALL_RE = re.compile(r'\bCALL\s+["\']?([\w-]+)["\']?', re.IGNORECASE)

# ── COBOL embedded block detection ───────────────────────────────

# EXEC SQL ... END-EXEC and EXEC CICS ... END-EXEC
# Used to tag paragraphs that contain DB or CICS calls (metadata enrichment)
COBOL_EXEC_SQL_RE = re.compile(r"\bEXEC\s+SQL\b", re.IGNORECASE)
COBOL_EXEC_CICS_RE = re.compile(r"\bEXEC\s+CICS\b", re.IGNORECASE)

# ── PL/1 call detection patterns ─────────────────────────────────

# CALL procedure_name[(args)];
# PL/1 names are alphanumeric + underscore (no hyphens)
PL1_CALL_RE = re.compile(r"\bCALL\s+(\w+)\s*(?:[(;]|\s)", re.IGNORECASE)

# GO TO label; (unconditional jump)
PL1_GOTO_RE = re.compile(r"\bGO\s+TO\s+(\w+)\s*;", re.IGNORECASE)

# ── Exclusion sets ───────────────────────────────────────────────

# Common builtins/keywords to exclude from call detection
BUILTINS = frozenset({
    # Python builtins
    "print", "len", "range", "int", "str", "float", "bool", "list", "dict",
    "set", "tuple", "type", "isinstance", "issubclass", "getattr", "setattr",
    "hasattr", "super", "property", "classmethod", "staticmethod", "enumerate",
    "zip", "map", "filter", "sorted", "reversed", "any", "all", "min", "max",
    "sum", "abs", "round", "open", "format", "repr", "hash", "id", "input",
    "vars", "dir", "callable", "iter", "next", "slice",
    # JS/TS builtins
    "console", "log", "error", "warn", "setTimeout", "setInterval",
    "clearTimeout", "clearInterval", "parseInt", "parseFloat",
    "Array", "Object", "String", "Number", "Boolean", "Date", "Math",
    "JSON", "Promise", "Error", "RegExp", "Map", "Set", "WeakMap", "WeakSet",
    "Symbol", "Proxy", "Reflect", "fetch", "require",
    # C# builtins / common framework types
    "Console", "WriteLine", "Write", "ReadLine", "ToString", "GetType",
    "Equals", "GetHashCode", "ReferenceEquals",
    "Task", "Func", "Action", "Predicate", "Delegate",
    "List", "Dictionary", "HashSet", "Queue", "Stack",
    "var", "nameof", "typeof", "sizeof", "default",
    "Dispose", "ConfigureAwait", "GetAwaiter", "GetResult",
    # Java builtins / common framework types
    "System", "out", "println", "equals", "hashCode", "getClass",
    "toString", "valueOf", "compareTo", "iterator",
    # Common patterns that look like calls but aren't meaningful edges
    "self", "this", "cls", "return", "raise", "throw", "new", "delete",
    "if", "for", "while", "switch", "catch", "try", "finally",
    # COBOL reserved words that commonly follow PERFORM (not paragraph names)
    "VARYING", "UNTIL", "TIMES", "THROUGH", "THRU", "TEST", "AFTER",
    "BEFORE", "INLINE", "WITH", "NO",
    # PL/1 built-in functions (not user-defined procedures)
    "SUBSTR", "LENGTH", "INDEX", "TRIM", "VERIFY", "TRANSLATE",
    "FIXED", "FLOAT", "CHAR", "BIT", "COMPLEX", "REAL", "IMAG",
    "ABS", "SIGN", "MOD", "MAX", "MIN", "SUM", "PROD",
    # VB.NET builtins
    "MsgBox", "InputBox", "CStr", "CInt", "CLng", "CDbl", "CSng", "CBool",
    "CByte", "CChar", "CDate", "CDec", "CObj", "CShort", "CType",
    "DirectCast", "TryCast", "IsNothing", "IsNumeric",
    "MyBase", "MyClass", "Me",
    "Len", "Mid", "Left", "Right", "Trim", "UCase", "LCase",
    "Val", "Asc", "Chr",
})

# Common framework / primitive types to exclude from type_dep resolution.
# These never correspond to user-defined code units in a project.
PRIMITIVE_TYPES = frozenset({
    # Java primitives + wrappers
    "String", "Integer", "Long", "Double", "Float", "Boolean", "Byte",
    "Short", "Character", "Number", "Object", "Class", "Void",
    # Java collections / standard lib
    "List", "Set", "Map", "Collection", "Queue", "Deque",
    "ArrayList", "LinkedList", "HashMap", "TreeMap", "LinkedHashMap",
    "HashSet", "TreeSet", "LinkedHashSet", "ConcurrentHashMap",
    "Iterator", "Iterable", "Comparable", "Serializable", "Cloneable",
    "Optional", "Stream", "Collectors",
    "CompletableFuture", "Future", "Callable", "Runnable",
    "Supplier", "Consumer", "Function", "Predicate", "BiFunction",
    "BiConsumer", "BiPredicate", "UnaryOperator", "BinaryOperator",
    # C# / .NET primitives + standard lib
    "IList", "ISet", "IMap", "IDictionary", "IEnumerable", "IEnumerator",
    "ICollection", "IReadOnlyList", "IReadOnlyDictionary", "IReadOnlyCollection",
    "IDisposable", "IComparable", "IEquatable", "IFormattable", "ICloneable",
    "Task", "ValueTask", "Action", "Func", "Predicate",
    "CancellationToken", "CancellationTokenSource",
    "StringBuilder", "StringComparer",
    "DateTime", "DateTimeOffset", "TimeSpan", "Guid",
    "Nullable", "Lazy", "Tuple", "KeyValuePair",
    "Exception", "ArgumentException", "InvalidOperationException",
    "NotImplementedException", "NullReferenceException",
    "ILogger", "IConfiguration", "IServiceProvider", "IOptions",
    # Python standard types
    "Any", "Dict", "Tuple", "Type", "Callable", "Generator",
    "AsyncGenerator", "Awaitable", "Coroutine", "Protocol",
    "ClassVar", "Final", "Literal", "TypeVar", "Generic",
    "Union", "Sequence", "Mapping", "MutableMapping", "Iterable",
    "AbstractSet", "MutableSet", "MutableSequence",
    # TypeScript/JS standard types
    "Array", "Record", "Partial", "Required", "Readonly", "Pick",
    "Omit", "Exclude", "Extract", "NonNullable", "ReturnType",
    "InstanceType", "Parameters", "ConstructorParameters",
    "Promise", "Date", "RegExp", "Error", "TypeError",
    "Uint8Array", "Int32Array", "Float64Array", "ArrayBuffer",
    "ReadonlyArray", "PropertyKey", "Symbol",
})
