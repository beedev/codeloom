using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System.Text;
using System.Text.Json;

/// <summary>
/// Lightweight C# semantic analyzer for CodeLoom.
///
/// Reads a single .cs file and outputs enrichment JSON to stdout.
/// Enrichments include fully qualified type names, nullable annotations,
/// generic constraints, and override chains.
///
/// Usage: dotnet run -- &lt;path-to-cs-file&gt;
///    or: dotnet roslyn-analyzer.dll &lt;path-to-cs-file&gt;
/// </summary>

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: roslyn-analyzer <file.cs>");
    Environment.Exit(1);
}

try
{
    var filePath = args[0];
    var sourceText = File.ReadAllText(filePath);
    var tree = CSharpSyntaxTree.ParseText(sourceText, path: filePath);
    var root = tree.GetCompilationUnitRoot();

    // Build a minimal compilation for semantic analysis
    var compilation = CSharpCompilation.Create("Analysis")
        .AddReferences(MetadataReference.CreateFromFile(typeof(object).Assembly.Location))
        .AddSyntaxTrees(tree);

    var model = compilation.GetSemanticModel(tree);

    var enrichments = new List<object>();

    // Collect using directives for type resolution
    var usingMap = new Dictionary<string, string>();
    foreach (var usingDir in root.Usings)
    {
        var ns = usingDir.Name?.ToString() ?? "";
        if (!string.IsNullOrEmpty(ns))
        {
            // Store namespace for resolution hints
            usingMap[ns.Split('.').Last()] = ns;
        }
    }

    // Walk all type declarations
    foreach (var typeDecl in root.DescendantNodes().OfType<TypeDeclarationSyntax>())
    {
        var typeQN = GetQualifiedName(typeDecl);

        // Process methods
        foreach (var method in typeDecl.Members.OfType<MethodDeclarationSyntax>())
        {
            enrichments.Add(AnalyzeMethod(method, typeQN, model));
        }

        // Process constructors
        foreach (var ctor in typeDecl.Members.OfType<ConstructorDeclarationSyntax>())
        {
            enrichments.Add(AnalyzeConstructor(ctor, typeQN, model));
        }
    }

    var result = new
    {
        file = Path.GetFileName(filePath),
        enrichments
    };

    var options = new JsonSerializerOptions { WriteIndented = false };
    Console.WriteLine(JsonSerializer.Serialize(result, options));
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    Environment.Exit(1);
}

static object AnalyzeMethod(MethodDeclarationSyntax method, string parentQN, SemanticModel model)
{
    var methodName = method.Identifier.Text;
    var qualifiedName = $"{parentQN}.{methodName}";

    // Resolved parameters
    var resolvedParams = method.ParameterList.Parameters.Select(p =>
    {
        var typeName = ResolveTypeName(p.Type, model);
        var isNullable = p.Type is NullableTypeSyntax;
        var defaultValue = p.Default?.Value.ToString();

        var param = new Dictionary<string, object?>
        {
            ["name"] = p.Identifier.Text,
            ["type"] = typeName,
            ["nullable"] = isNullable
        };
        if (defaultValue != null)
            param["default"] = defaultValue;

        return param;
    }).ToList();

    // Return type
    var returnType = ResolveTypeName(method.ReturnType, model);

    // Modifiers
    var modifiers = method.Modifiers.Select(m => m.Text).ToList();

    // Generic type parameters with constraints
    var generics = new List<string>();
    if (method.TypeParameterList != null)
    {
        foreach (var tp in method.TypeParameterList.Parameters)
        {
            var constraint = method.ConstraintClauses
                .FirstOrDefault(c => c.Name.ToString() == tp.Identifier.Text);
            if (constraint != null)
                generics.Add($"{tp.Identifier.Text} : {constraint.Constraints}");
            else
                generics.Add(tp.Identifier.Text);
        }
    }

    // Nullable annotations
    var nullableAnnotations = new List<string>();
    if (method.ReturnType is NullableTypeSyntax)
        nullableAnnotations.Add("return:nullable");
    foreach (var p in method.ParameterList.Parameters)
    {
        if (p.Type is NullableTypeSyntax)
            nullableAnnotations.Add($"{p.Identifier.Text}:nullable");
    }

    // Check if override
    bool isOverride = modifiers.Contains("override");

    var result = new Dictionary<string, object>
    {
        ["qualified_name"] = qualifiedName,
        ["resolved_params"] = resolvedParams,
        ["resolved_return_type"] = returnType
    };

    if (generics.Count > 0)
        result["generic_type_params"] = generics;

    if (nullableAnnotations.Count > 0)
        result["nullable_annotations"] = nullableAnnotations;

    if (isOverride)
        result["is_override"] = true;

    return result;
}

static object AnalyzeConstructor(ConstructorDeclarationSyntax ctor, string parentQN, SemanticModel model)
{
    var qualifiedName = $"{parentQN}.{ctor.Identifier.Text}";

    var resolvedParams = ctor.ParameterList.Parameters.Select(p =>
    {
        var typeName = ResolveTypeName(p.Type, model);
        var defaultValue = p.Default?.Value.ToString();

        var param = new Dictionary<string, object?>
        {
            ["name"] = p.Identifier.Text,
            ["type"] = typeName
        };
        if (defaultValue != null)
            param["default"] = defaultValue;

        return param;
    }).ToList();

    return new Dictionary<string, object>
    {
        ["qualified_name"] = qualifiedName,
        ["resolved_params"] = resolvedParams,
        ["resolved_return_type"] = "void"
    };
}

static string GetQualifiedName(TypeDeclarationSyntax typeDecl)
{
    var parts = new List<string> { typeDecl.Identifier.Text };

    SyntaxNode? current = typeDecl.Parent;
    while (current != null)
    {
        if (current is TypeDeclarationSyntax parentType)
            parts.Insert(0, parentType.Identifier.Text);
        else if (current is NamespaceDeclarationSyntax ns)
            parts.Insert(0, ns.Name.ToString());
        else if (current is FileScopedNamespaceDeclarationSyntax fsns)
            parts.Insert(0, fsns.Name.ToString());

        current = current.Parent;
    }

    return string.Join(".", parts);
}

static string ResolveTypeName(TypeSyntax? typeSyntax, SemanticModel model)
{
    if (typeSyntax == null) return "void";

    // Try semantic model first for fully qualified names
    try
    {
        var typeInfo = model.GetTypeInfo(typeSyntax);
        if (typeInfo.Type != null && typeInfo.Type.Kind != SymbolKind.ErrorType)
        {
            return typeInfo.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                .Replace("global::", "");
        }
    }
    catch
    {
        // Fall through to syntax-based resolution
    }

    // Fallback: return the syntax text
    return typeSyntax.ToString();
}
