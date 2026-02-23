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
