"""MVP-scoped ASG queries for structural diagram generation.

Reuses query patterns from migration/context_builder.py, scoped to MVP
unit_ids and file_ids for deterministic diagram data extraction.
"""

import logging
from typing import Dict, List
from uuid import UUID

from sqlalchemy import text

logger = logging.getLogger(__name__)


def get_mvp_class_data(db, project_id: str, unit_ids: List[str]) -> Dict:
    """Get class/interface hierarchy data for a Class Diagram.

    Returns:
        {
            "classes": [{unit_id, name, qualified_name, unit_type, language,
                         file_path, signature, metadata, members: [...]}],
            "edges": [{source, target, edge_type}]  (inherits, implements, contains)
        }
    """
    if not unit_ids:
        return {"classes": [], "edges": []}

    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    uids = [UUID(u) if isinstance(u, str) else u for u in unit_ids]

    with db.get_session() as session:
        # Get classes, interfaces, and their members
        rows = session.execute(text("""
            SELECT
                cu.unit_id,
                cu.name,
                cu.qualified_name,
                cu.unit_type,
                cu.language,
                cu.signature,
                cu.metadata,
                cf.file_path
            FROM code_units cu
            JOIN code_files cf ON cu.file_id = cf.file_id
            WHERE cu.unit_id = ANY(:uids)
            ORDER BY cu.qualified_name
        """), {"uids": uids})
        units = [dict(r._mapping) for r in rows.fetchall()]

        # Get structural edges: inherits, implements, contains within MVP
        edges = session.execute(text("""
            SELECT
                su.qualified_name AS source,
                tu.qualified_name AS target,
                ce.edge_type
            FROM code_edges ce
            JOIN code_units su ON ce.source_unit_id = su.unit_id
            JOIN code_units tu ON ce.target_unit_id = tu.unit_id
            WHERE ce.project_id = :pid
              AND ce.edge_type IN ('inherits', 'implements', 'contains', 'overrides')
              AND (ce.source_unit_id = ANY(:uids) OR ce.target_unit_id = ANY(:uids))
            ORDER BY ce.edge_type, su.qualified_name
        """), {"pid": pid, "uids": uids})
        edge_list = [dict(r._mapping) for r in edges.fetchall()]

        # Get calls/imports edges to derive class-level dependencies
        call_edges = session.execute(text("""
            SELECT
                su.qualified_name AS source,
                tu.qualified_name AS target
            FROM code_edges ce
            JOIN code_units su ON ce.source_unit_id = su.unit_id
            JOIN code_units tu ON ce.target_unit_id = tu.unit_id
            WHERE ce.project_id = :pid
              AND ce.edge_type IN ('calls', 'imports')
              AND ce.source_unit_id = ANY(:uids)
              AND ce.target_unit_id = ANY(:uids)
            ORDER BY su.qualified_name
            LIMIT 500
        """), {"pid": pid, "uids": uids})
        call_list = [dict(r._mapping) for r in call_edges.fetchall()]

    # Group members under their parent classes via 'contains' edges
    parent_map = {}
    structural_edges = []
    for e in edge_list:
        if e["edge_type"] == "contains":
            parent_map.setdefault(e["source"], []).append(e["target"])
        else:
            structural_edges.append(e)

    # Build child→parent reverse map for lifting calls to class level
    child_to_parent = {}
    for parent_qn, children in parent_map.items():
        for child_qn in children:
            child_to_parent[child_qn] = parent_qn

    def _resolve_to_class(qname: str) -> str:
        """Walk up the contains chain to the top-level class."""
        visited = set()
        current = qname
        while current in child_to_parent and current not in visited:
            visited.add(current)
            current = child_to_parent[current]
        return current

    # Build class list with members
    qname_to_unit = {u["qualified_name"]: u for u in units}
    classes = []
    for unit in units:
        qn = unit["qualified_name"]
        member_qnames = parent_map.get(qn, [])
        members = [qname_to_unit[m] for m in member_qnames if m in qname_to_unit]
        unit["members"] = members
        # Only include top-level classes/interfaces, not methods that are contained
        is_member = any(qn in children for children in parent_map.values())
        if not is_member:
            classes.append(unit)

    # Aggregate calls/imports to class-level "depends" edges
    class_qnames = {c["qualified_name"] for c in classes}
    class_deps: set[tuple[str, str]] = set()
    for call in call_list:
        src_class = _resolve_to_class(call["source"])
        tgt_class = _resolve_to_class(call["target"])
        if (src_class != tgt_class
                and src_class in class_qnames
                and tgt_class in class_qnames):
            class_deps.add((src_class, tgt_class))

    for src, tgt in sorted(class_deps):
        structural_edges.append({"source": src, "target": tgt, "edge_type": "depends"})

    return {"classes": classes, "edges": structural_edges}


def get_mvp_package_data(db, project_id: str, file_ids: List[str], unit_ids: List[str]) -> Dict:
    """Get package/directory grouping data for a Package Diagram.

    Returns:
        {
            "packages": [{directory, files: [{file_path, language, name}]}],
            "imports": [{source_dir, target_dir, count}]
        }
    """
    if not file_ids and not unit_ids:
        return {"packages": [], "imports": []}

    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    fids = [UUID(f) if isinstance(f, str) else f for f in file_ids] if file_ids else []
    uids = [UUID(u) if isinstance(u, str) else u for u in unit_ids] if unit_ids else []

    with db.get_session() as session:
        # Get files in this MVP
        if fids:
            file_rows = session.execute(text("""
                SELECT file_id, file_path, language
                FROM code_files
                WHERE file_id = ANY(:fids)
                ORDER BY file_path
            """), {"fids": fids})
        else:
            file_rows = session.execute(text("""
                SELECT DISTINCT cf.file_id, cf.file_path, cf.language
                FROM code_files cf
                JOIN code_units cu ON cu.file_id = cf.file_id
                WHERE cu.unit_id = ANY(:uids)
                ORDER BY cf.file_path
            """), {"uids": uids})
        files = [dict(r._mapping) for r in file_rows.fetchall()]

        # Get import edges between MVP files
        all_fids = [f["file_id"] for f in files]
        if all_fids:
            import_rows = session.execute(text("""
                SELECT
                    sf.file_path AS source_path,
                    tf.file_path AS target_path,
                    COUNT(*) AS count
                FROM code_edges ce
                JOIN code_units su ON ce.source_unit_id = su.unit_id
                JOIN code_units tu ON ce.target_unit_id = tu.unit_id
                JOIN code_files sf ON su.file_id = sf.file_id
                JOIN code_files tf ON tu.file_id = tf.file_id
                WHERE ce.project_id = :pid
                  AND ce.edge_type = 'imports'
                  AND su.file_id = ANY(:fids)
                GROUP BY sf.file_path, tf.file_path
                ORDER BY count DESC
            """), {"pid": pid, "fids": all_fids})
            imports_raw = [dict(r._mapping) for r in import_rows.fetchall()]
        else:
            imports_raw = []

    # Group files by directory
    dir_map: Dict[str, list] = {}
    for f in files:
        parts = f["file_path"].rsplit("/", 1)
        directory = parts[0] if len(parts) > 1 else "."
        filename = parts[1] if len(parts) > 1 else parts[0]
        dir_map.setdefault(directory, []).append({
            "file_path": f["file_path"],
            "language": f["language"],
            "name": filename,
        })

    packages = [{"directory": d, "files": flist} for d, flist in sorted(dir_map.items())]

    # Aggregate imports by directory
    def _dir_of(path: str) -> str:
        parts = path.rsplit("/", 1)
        return parts[0] if len(parts) > 1 else "."

    dir_imports: Dict[tuple, int] = {}
    for imp in imports_raw:
        sd = _dir_of(imp["source_path"])
        td = _dir_of(imp["target_path"])
        if sd != td:
            key = (sd, td)
            dir_imports[key] = dir_imports.get(key, 0) + imp["count"]

    imports = [
        {"source_dir": k[0], "target_dir": k[1], "count": v}
        for k, v in sorted(dir_imports.items(), key=lambda x: -x[1])
    ]

    return {"packages": packages, "imports": imports}


def get_mvp_component_data(db, project_id: str, unit_ids: List[str], file_ids: List[str]) -> Dict:
    """Get component/service-level data for a Component Diagram.

    Classifies units by stereotype (controller, service, repository, entity)
    based on naming conventions and metadata.

    Returns:
        {
            "components": [{name, qualified_name, stereotype, unit_type, file_path}],
            "connectors": [{source, target, edge_type}]
        }
    """
    if not unit_ids:
        return {"components": [], "connectors": []}

    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    uids = [UUID(u) if isinstance(u, str) else u for u in unit_ids]

    with db.get_session() as session:
        rows = session.execute(text("""
            SELECT
                cu.unit_id,
                cu.name,
                cu.qualified_name,
                cu.unit_type,
                cu.language,
                cu.metadata,
                cf.file_path
            FROM code_units cu
            JOIN code_files cf ON cu.file_id = cf.file_id
            WHERE cu.unit_id = ANY(:uids)
              AND cu.unit_type IN ('class', 'interface', 'module', 'struct')
            ORDER BY cu.qualified_name
        """), {"uids": uids})
        units = [dict(r._mapping) for r in rows.fetchall()]

        # Get call + import edges between these components
        edges = session.execute(text("""
            SELECT
                su.qualified_name AS source,
                tu.qualified_name AS target,
                ce.edge_type
            FROM code_edges ce
            JOIN code_units su ON ce.source_unit_id = su.unit_id
            JOIN code_units tu ON ce.target_unit_id = tu.unit_id
            WHERE ce.project_id = :pid
              AND ce.edge_type IN ('calls', 'imports')
              AND ce.source_unit_id = ANY(:uids)
              AND ce.target_unit_id = ANY(:uids)
              AND su.unit_type IN ('class', 'interface', 'module', 'struct')
              AND tu.unit_type IN ('class', 'interface', 'module', 'struct')
            ORDER BY ce.edge_type, su.qualified_name
            LIMIT 200
        """), {"pid": pid, "uids": uids})
        connectors = [dict(r._mapping) for r in edges.fetchall()]

    # Classify stereotypes
    components = []
    for u in units:
        name_lower = u["name"].lower()
        qname_lower = u["qualified_name"].lower()
        path_lower = (u.get("file_path") or "").lower()

        stereotype = _classify_stereotype(name_lower, qname_lower, path_lower, u.get("metadata"))
        components.append({
            "name": u["name"],
            "qualified_name": u["qualified_name"],
            "stereotype": stereotype,
            "unit_type": u["unit_type"],
            "file_path": u.get("file_path", ""),
        })

    return {"components": components, "connectors": connectors}


def get_detected_infrastructure(db, project_id: str, unit_ids: List[str]) -> str:
    """Detect actual infrastructure from import statements and annotations in the ASG.

    Scans code_units for known framework, database, and service imports to ground
    deployment diagrams in reality rather than LLM imagination.

    Returns:
        Formatted string of detected infrastructure components.
    """
    if not unit_ids:
        return ""

    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    uids = [UUID(u) if isinstance(u, str) else u for u in unit_ids]

    # Known infrastructure patterns to detect
    _INFRA_PATTERNS = {
        # Databases
        "postgresql": {"category": "database", "label": "PostgreSQL", "port": "5432"},
        "psycopg": {"category": "database", "label": "PostgreSQL", "port": "5432"},
        "mysql": {"category": "database", "label": "MySQL", "port": "3306"},
        "mongodb": {"category": "database", "label": "MongoDB", "port": "27017"},
        "mongoose": {"category": "database", "label": "MongoDB", "port": "27017"},
        "sqlalchemy": {"category": "database", "label": "PostgreSQL/SQL DB", "port": "5432"},
        "hibernate": {"category": "database", "label": "SQL Database (JPA)", "port": "5432"},
        "typeorm": {"category": "database", "label": "SQL Database (TypeORM)", "port": "5432"},
        "prisma": {"category": "database", "label": "SQL Database (Prisma)", "port": "5432"},
        "entity.framework": {"category": "database", "label": "SQL Server (EF)", "port": "1433"},
        "sequelize": {"category": "database", "label": "SQL Database (Sequelize)", "port": "5432"},
        # Caching
        "redis": {"category": "cache", "label": "Redis", "port": "6379"},
        "memcached": {"category": "cache", "label": "Memcached", "port": "11211"},
        # Message queues
        "kafka": {"category": "message_queue", "label": "Apache Kafka", "port": "9092"},
        "rabbitmq": {"category": "message_queue", "label": "RabbitMQ", "port": "5672"},
        "amqp": {"category": "message_queue", "label": "AMQP Broker", "port": "5672"},
        "celery": {"category": "message_queue", "label": "Celery Worker", "port": ""},
        "bullmq": {"category": "message_queue", "label": "BullMQ (Redis)", "port": "6379"},
        # Search
        "elasticsearch": {"category": "search", "label": "Elasticsearch", "port": "9200"},
        "opensearch": {"category": "search", "label": "OpenSearch", "port": "9200"},
        # Frameworks
        "spring": {"category": "framework", "label": "Spring Boot", "port": "8080"},
        "fastapi": {"category": "framework", "label": "FastAPI", "port": "8000"},
        "flask": {"category": "framework", "label": "Flask", "port": "5000"},
        "django": {"category": "framework", "label": "Django", "port": "8000"},
        "express": {"category": "framework", "label": "Express.js", "port": "3000"},
        "nestjs": {"category": "framework", "label": "NestJS", "port": "3000"},
        "asp.net": {"category": "framework", "label": "ASP.NET", "port": "5000"},
        "kestrel": {"category": "framework", "label": "ASP.NET (Kestrel)", "port": "5000"},
        # External APIs
        "httpx": {"category": "external", "label": "HTTP Client (outbound)", "port": ""},
        "requests": {"category": "external", "label": "HTTP Client (outbound)", "port": ""},
        "axios": {"category": "external", "label": "HTTP Client (outbound)", "port": ""},
        "grpc": {"category": "external", "label": "gRPC", "port": ""},
        # Storage
        "s3": {"category": "storage", "label": "AWS S3", "port": ""},
        "azure.blob": {"category": "storage", "label": "Azure Blob Storage", "port": ""},
        "minio": {"category": "storage", "label": "MinIO / S3", "port": "9000"},
    }

    with db.get_session() as session:
        # Get import targets from code_edges
        rows = session.execute(text("""
            SELECT DISTINCT tu.qualified_name, tu.name
            FROM code_edges ce
            JOIN code_units su ON ce.source_unit_id = su.unit_id
            JOIN code_units tu ON ce.target_unit_id = tu.unit_id
            WHERE ce.project_id = :pid
              AND ce.edge_type = 'imports'
              AND ce.source_unit_id = ANY(:uids)
            UNION
            SELECT DISTINCT cu.qualified_name, cu.name
            FROM code_units cu
            WHERE cu.unit_id = ANY(:uids)
              AND cu.source IS NOT NULL
        """), {"pid": pid, "uids": uids})
        all_names = [
            (r.qualified_name or "").lower() + " " + (r.name or "").lower()
            for r in rows.fetchall()
        ]

    # Detect infrastructure
    detected: Dict[str, Dict] = {}
    combined_text = " ".join(all_names)
    for pattern, info in _INFRA_PATTERNS.items():
        if pattern in combined_text:
            key = info["label"]
            if key not in detected:
                detected[key] = info

    if not detected:
        return ""

    # Format as structured text
    lines = []
    by_category = {}
    for label, info in detected.items():
        by_category.setdefault(info["category"], []).append(info)

    category_labels = {
        "framework": "Application Frameworks",
        "database": "Databases",
        "cache": "Caching",
        "message_queue": "Message Queues",
        "search": "Search Engines",
        "external": "External Connections",
        "storage": "Object Storage",
    }

    for cat in ["framework", "database", "cache", "message_queue", "search", "storage", "external"]:
        items = by_category.get(cat, [])
        if items:
            lines.append(f"**{category_labels.get(cat, cat.title())}**:")
            for item in items:
                port_str = f" (port {item['port']})" if item['port'] else ""
                lines.append(f"  - {item['label']}{port_str}")

    return "\n".join(lines)


def _classify_stereotype(name: str, qname: str, path: str, metadata: dict | None) -> str:
    """Classify a unit into an architectural stereotype."""
    combined = f"{name} {qname} {path}"

    if any(k in combined for k in ("controller", "handler", "endpoint", "route", "view", "api")):
        return "controller"
    if any(k in combined for k in ("service", "manager", "facade", "orchestrat")):
        return "service"
    if any(k in combined for k in ("repository", "repo", "dao", "store", "gateway", "adapter")):
        return "repository"
    if any(k in combined for k in ("entity", "model", "dto", "schema", "domain")):
        return "entity"
    if any(k in combined for k in ("util", "helper", "common", "shared", "lib")):
        return "utility"
    if any(k in combined for k in ("config", "setting", "constant")):
        return "config"
    if any(k in combined for k in ("middleware", "filter", "interceptor", "guard")):
        return "middleware"
    if any(k in combined for k in ("test", "spec", "fixture")):
        return "test"

    # Check metadata decorators/annotations
    if metadata:
        modifiers = metadata.get("modifiers", [])
        decorators = metadata.get("decorators", [])
        all_annotations = " ".join(modifiers + decorators).lower()
        if any(k in all_annotations for k in ("@controller", "@restcontroller", "@api")):
            return "controller"
        if any(k in all_annotations for k in ("@service", "@component")):
            return "service"
        if any(k in all_annotations for k in ("@repository", "@dao")):
            return "repository"
        if any(k in all_annotations for k in ("@entity", "@table", "@model")):
            return "entity"

    return "component"
