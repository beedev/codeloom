package cli;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.resolution.declarations.ResolvedReferenceTypeDeclaration;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Lightweight Java semantic analyzer for CodeLoom.
 *
 * Reads a single .java file and outputs enrichment JSON to stdout.
 * Enrichments include fully qualified type names (via import resolution),
 * generic type parameters, checked exceptions, and interface implementations.
 *
 * Usage: java -jar javaparser-cli.jar <path-to-java-file>
 */
public class Analyzer {

    // Simple import map: short name -> fully qualified name
    private final Map<String, String> importMap = new HashMap<>();
    private String packageName = "";

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java -jar javaparser-cli.jar <file.java>");
            System.exit(1);
        }

        try {
            Analyzer analyzer = new Analyzer();
            String json = analyzer.analyze(new File(args[0]));
            System.out.println(json);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    public String analyze(File file) throws Exception {
        CompilationUnit cu = StaticJavaParser.parse(file);

        // Build import map for type resolution
        packageName = cu.getPackageDeclaration()
                .map(pd -> pd.getNameAsString())
                .orElse("");

        for (ImportDeclaration imp : cu.getImports()) {
            if (!imp.isAsterisk()) {
                String fqn = imp.getNameAsString();
                String shortName = fqn.substring(fqn.lastIndexOf('.') + 1);
                importMap.put(shortName, fqn);
            }
        }

        List<String> enrichments = new ArrayList<>();

        // Process all type declarations (classes, interfaces, enums)
        for (TypeDeclaration<?> type : cu.getTypes()) {
            processType(type, "", enrichments);
        }

        return "{\"file\":" + jsonStr(file.getName()) + ","
                + "\"enrichments\":[" + String.join(",", enrichments) + "]}";
    }

    private void processType(TypeDeclaration<?> type, String parentQN, List<String> enrichments) {
        String qn = buildQualifiedName(type.getNameAsString(), parentQN);

        // Process methods
        for (MethodDeclaration method : type.getMethods()) {
            enrichments.add(analyzeMethod(method, qn));
        }

        // Process constructors
        for (ConstructorDeclaration ctor : type.getConstructors()) {
            enrichments.add(analyzeConstructor(ctor, qn));
        }

        // Collect interface implementations for classes
        if (type instanceof ClassOrInterfaceDeclaration) {
            ClassOrInterfaceDeclaration cid = (ClassOrInterfaceDeclaration) type;

            // Process nested types
            for (BodyDeclaration<?> member : cid.getMembers()) {
                if (member instanceof TypeDeclaration) {
                    processType((TypeDeclaration<?>) member, qn, enrichments);
                }
            }
        }
    }

    private String analyzeMethod(MethodDeclaration method, String parentQN) {
        String methodQN = parentQN + "." + method.getNameAsString();

        // Resolved parameters
        List<String> params = method.getParameters().stream()
                .map(p -> "{\"name\":" + jsonStr(p.getNameAsString())
                        + ",\"type\":" + jsonStr(resolveType(p.getType())) + "}")
                .collect(Collectors.toList());

        // Resolved return type
        String returnType = resolveType(method.getType());

        // Thrown exceptions
        List<String> thrown = method.getThrownExceptions().stream()
                .map(t -> jsonStr(resolveTypeName(t.asString())))
                .collect(Collectors.toList());

        // Generic type parameters
        List<String> generics = method.getTypeParameters().stream()
                .map(tp -> jsonStr(tp.toString()))
                .collect(Collectors.toList());

        // Interface implementations (from parent class)
        List<String> interfaces = getImplementedInterfaces(method);

        StringBuilder sb = new StringBuilder();
        sb.append("{\"qualified_name\":").append(jsonStr(methodQN));
        sb.append(",\"resolved_params\":[").append(String.join(",", params)).append("]");
        sb.append(",\"resolved_return_type\":").append(jsonStr(returnType));

        if (!thrown.isEmpty()) {
            sb.append(",\"thrown_exceptions\":[").append(String.join(",", thrown)).append("]");
        }
        if (!interfaces.isEmpty()) {
            sb.append(",\"implements_interfaces\":[").append(String.join(",", interfaces)).append("]");
        }
        if (!generics.isEmpty()) {
            sb.append(",\"generic_type_params\":[").append(String.join(",", generics)).append("]");
        }

        sb.append("}");
        return sb.toString();
    }

    private String analyzeConstructor(ConstructorDeclaration ctor, String parentQN) {
        String ctorQN = parentQN + "." + ctor.getNameAsString();

        List<String> params = ctor.getParameters().stream()
                .map(p -> "{\"name\":" + jsonStr(p.getNameAsString())
                        + ",\"type\":" + jsonStr(resolveType(p.getType())) + "}")
                .collect(Collectors.toList());

        List<String> thrown = ctor.getThrownExceptions().stream()
                .map(t -> jsonStr(resolveTypeName(t.asString())))
                .collect(Collectors.toList());

        StringBuilder sb = new StringBuilder();
        sb.append("{\"qualified_name\":").append(jsonStr(ctorQN));
        sb.append(",\"resolved_params\":[").append(String.join(",", params)).append("]");
        sb.append(",\"resolved_return_type\":").append(jsonStr("void"));

        if (!thrown.isEmpty()) {
            sb.append(",\"thrown_exceptions\":[").append(String.join(",", thrown)).append("]");
        }

        sb.append("}");
        return sb.toString();
    }

    /**
     * Resolve a Type to its fully qualified name string.
     * Handles generics like List<String> -> java.util.List<java.lang.String>.
     */
    private String resolveType(Type type) {
        if (type.isVoidType()) return "void";
        if (type.isPrimitiveType()) return type.asString();
        if (type.isArrayType()) {
            return resolveType(type.asArrayType().getComponentType()) + "[]";
        }

        if (type.isClassOrInterfaceType()) {
            ClassOrInterfaceType cit = type.asClassOrInterfaceType();
            String baseName = resolveTypeName(cit.getNameAsString());

            if (cit.getTypeArguments().isPresent()) {
                String typeArgs = cit.getTypeArguments().get().stream()
                        .map(this::resolveType)
                        .collect(Collectors.joining(", "));
                return baseName + "<" + typeArgs + ">";
            }
            return baseName;
        }

        // Wildcard, type variable, etc. — return as-is
        return type.asString();
    }

    /**
     * Resolve a short type name to fully qualified using the import map.
     * Falls back to java.lang.* for standard types, or package-qualified name.
     */
    private String resolveTypeName(String shortName) {
        // Already qualified
        if (shortName.contains(".")) return shortName;

        // Check import map
        if (importMap.containsKey(shortName)) return importMap.get(shortName);

        // java.lang types (common subset)
        if (isJavaLangType(shortName)) return "java.lang." + shortName;

        // Same package
        if (!packageName.isEmpty()) return packageName + "." + shortName;

        return shortName;
    }

    private boolean isJavaLangType(String name) {
        switch (name) {
            case "String": case "Object": case "Integer": case "Long":
            case "Double": case "Float": case "Boolean": case "Byte":
            case "Short": case "Character": case "Void": case "Number":
            case "Math": case "System": case "Class": case "Thread":
            case "Runnable": case "Comparable": case "Iterable":
            case "Throwable": case "Exception": case "RuntimeException":
            case "Error": case "Override": case "Deprecated":
            case "SuppressWarnings": case "StringBuilder": case "StringBuffer":
                return true;
            default:
                return false;
        }
    }

    /**
     * Get interfaces implemented by the parent class of a method.
     */
    private List<String> getImplementedInterfaces(MethodDeclaration method) {
        if (method.getParentNode().isEmpty()) return Collections.emptyList();
        if (!(method.getParentNode().get() instanceof ClassOrInterfaceDeclaration)) {
            return Collections.emptyList();
        }

        ClassOrInterfaceDeclaration parent = (ClassOrInterfaceDeclaration) method.getParentNode().get();
        if (parent.isInterface()) return Collections.emptyList();

        return parent.getImplementedTypes().stream()
                .map(t -> resolveTypeName(t.getNameAsString()))
                .collect(Collectors.toList());
    }

    private String buildQualifiedName(String name, String parentQN) {
        if (!parentQN.isEmpty()) return parentQN + "." + name;
        if (!packageName.isEmpty()) return packageName + "." + name;
        return name;
    }

    /** Escape a string for JSON output. */
    private static String jsonStr(String value) {
        if (value == null) return "null";
        return "\"" + value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
                + "\"";
    }
}
