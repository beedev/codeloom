"""PlantUML text -> SVG rendering with dual-mode support.

Primary:  Local JAR via `java -jar plantuml.jar -tsvg -pipe` (no size limits).
Fallback: Public PlantUML HTTP server with deflate + custom base64 URL encoding.

The local JAR is preferred because large diagrams (41KB+ PUML source) exceed URL
length limits on the HTTP server, causing 400/509 errors.  The JAR renders via
stdin/stdout pipe with no such constraint.

JAR location resolution order:
  1. PLANTUML_JAR_PATH env var (explicit override)
  2. tools/plantuml/plantuml.jar (repo-local, downloaded by `./dev.sh setup-tools`)
"""

import logging
import os
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JAR availability detection (cached per-process, same pattern as java_bridge)
# ---------------------------------------------------------------------------

_DEFAULT_JAR_PATH = Path(__file__).parents[3] / "tools" / "plantuml" / "plantuml.jar"

_jar_availability_checked = False
_jar_available = False
_resolved_jar_path: Optional[Path] = None
_graphviz_available: Optional[bool] = None


def _resolve_jar_path() -> Optional[Path]:
    """Return the PlantUML JAR path if it exists, else None."""
    env_path = os.environ.get("PLANTUML_JAR_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.is_file() else None
    return _DEFAULT_JAR_PATH if _DEFAULT_JAR_PATH.is_file() else None


def _check_jar_available() -> bool:
    """Check once whether local JAR rendering is possible."""
    global _jar_availability_checked, _jar_available, _resolved_jar_path

    if _jar_availability_checked:
        return _jar_available

    _jar_availability_checked = True

    _resolved_jar_path = _resolve_jar_path()
    if _resolved_jar_path is None:
        logger.info(
            "PlantUML JAR not found — using HTTP fallback "
            "(run `./dev.sh setup-tools` for local rendering)"
        )
        _jar_available = False
        return False

    if shutil.which("java") is None:
        logger.info(
            "Java not in PATH — PlantUML JAR present but unusable, using HTTP fallback"
        )
        _jar_available = False
        return False

    logger.info("PlantUML local JAR available at %s", _resolved_jar_path)
    _jar_available = True
    return True


def _has_graphviz() -> bool:
    """Check once whether Graphviz (dot) is available on PATH."""
    global _graphviz_available
    if _graphviz_available is None:
        _graphviz_available = shutil.which("dot") is not None
        if not _graphviz_available:
            logger.info(
                "Graphviz (dot) not found — PlantUML will use built-in Smetana layout engine"
            )
    return _graphviz_available


def _ensure_smetana(puml: str) -> str:
    """Inject '!pragma layout smetana' when Graphviz is unavailable.

    The Smetana engine is PlantUML's built-in Java layout engine that works
    without external Graphviz.  Only injected for JAR rendering when dot is
    missing — the HTTP server has its own Graphviz installation.

    Activity and use-case diagrams use PlantUML's own layout engine (not
    Graphviz), so the Smetana pragma is skipped for them — it can cause
    rendering errors on those diagram types.
    """
    if _has_graphviz():
        return puml
    # Activity and use-case diagrams don't use Graphviz; Smetana breaks them
    if "\nstart\n" in puml or "\nleft to right direction\n" in puml:
        return puml
    pragma = "!pragma layout smetana"
    if pragma in puml:
        return puml
    # Insert right after the first @startuml line
    return puml.replace("@startuml", f"@startuml\n{pragma}", 1)


# ---------------------------------------------------------------------------
# Local JAR rendering
# ---------------------------------------------------------------------------


def _render_via_jar(puml: str) -> Optional[str]:
    """Render PlantUML source to SVG via local JAR (stdin -> stdout pipe).

    Returns SVG string on success, None on failure (caller should fall back).
    """
    assert _resolved_jar_path is not None

    # Use Smetana layout when Graphviz isn't installed
    puml = _ensure_smetana(puml)

    cmd = [
        "java",
        "-Djava.awt.headless=true",
        "-jar",
        str(_resolved_jar_path),
        "-tsvg",
        "-pipe",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=puml.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        logger.warning("PlantUML JAR timed out after 60s — falling back to HTTP")
        return None
    except OSError as e:
        logger.warning("PlantUML JAR execution failed: %s — falling back to HTTP", e)
        return None

    stdout = result.stdout.decode("utf-8", errors="replace")

    # PlantUML writes SVG to stdout even for syntax errors (the SVG shows the error).
    # Accept any output that looks like SVG.
    if stdout.strip().startswith("<") and "<svg" in stdout[:500]:
        if result.returncode != 0:
            logger.debug(
                "PlantUML JAR returned exit code %d but produced SVG — using it",
                result.returncode,
            )
        return stdout

    # Real failure — log stderr and signal fallback
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    logger.warning(
        "PlantUML JAR produced no SVG (exit=%d, stderr=%s) — falling back to HTTP",
        result.returncode,
        stderr[:300] if stderr else "(empty)",
    )
    return None


# ---------------------------------------------------------------------------
# HTTP server rendering (original approach — fallback)
# ---------------------------------------------------------------------------

_DEFAULT_SERVER = "https://www.plantuml.com/plantuml"

_PLANTUML_ALPHABET = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
)


def _encode6bit(b: int) -> str:
    """Encode a 6-bit value to PlantUML's custom base64 character."""
    return _PLANTUML_ALPHABET[b & 0x3F]


def _encode3bytes(b1: int, b2: int, b3: int) -> str:
    """Encode 3 bytes into 4 PlantUML base64 characters."""
    c1 = b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 = b3 & 0x3F
    return _encode6bit(c1) + _encode6bit(c2) + _encode6bit(c3) + _encode6bit(c4)


def _plantuml_encode(text: str) -> str:
    """Encode PlantUML text using deflate + custom base64 for URL embedding."""
    data = zlib.compress(text.encode("utf-8"))[2:-4]  # raw deflate

    result = []
    for i in range(0, len(data), 3):
        if i + 2 < len(data):
            result.append(_encode3bytes(data[i], data[i + 1], data[i + 2]))
        elif i + 1 < len(data):
            result.append(_encode3bytes(data[i], data[i + 1], 0))
        else:
            result.append(_encode3bytes(data[i], 0, 0))

    return "".join(result)


def _render_via_http(puml: str, server_url: Optional[str] = None) -> str:
    """Render PlantUML source to SVG via HTTP server (GET with encoded URL).

    Raises RuntimeError on failure.
    """
    server = server_url or os.environ.get("PLANTUML_SERVER_URL", _DEFAULT_SERVER)
    server = server.rstrip("/")

    encoded = _plantuml_encode(puml)
    url = f"{server}/svg/{encoded}"

    logger.debug("Rendering PlantUML via HTTP %s (encoded len=%d)", server, len(encoded))

    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)

        body = response.text
        if body.strip().startswith("<") and "<svg" in body[:500]:
            if response.status_code != 200:
                logger.warning(
                    "PlantUML server returned %d but with SVG content — using it",
                    response.status_code,
                )
            return body

        if response.status_code != 200:
            raise RuntimeError(
                f"PlantUML server returned {response.status_code} with non-SVG body"
            )

        raise RuntimeError(f"PlantUML server returned unexpected content: {body[:200]}")
    except httpx.RequestError as e:
        raise RuntimeError(f"PlantUML server request failed: {e}") from e


# ---------------------------------------------------------------------------
# Public API (signature unchanged)
# ---------------------------------------------------------------------------


def render_puml_to_svg(puml: str, server_url: Optional[str] = None) -> str:
    """Render PlantUML text to SVG.

    Tries local JAR first (no size limits), falls back to HTTP server.

    Args:
        puml: PlantUML source text (including @startuml/@enduml).
        server_url: PlantUML server base URL for HTTP fallback. Defaults to
                    PLANTUML_SERVER_URL env var or the public PlantUML server.

    Returns:
        SVG string.

    Raises:
        RuntimeError: If both JAR and HTTP rendering fail.
    """
    # Try local JAR first
    if _check_jar_available():
        svg = _render_via_jar(puml)
        if svg is not None:
            logger.debug("Rendered via local JAR (%d chars SVG)", len(svg))
            return svg
        # JAR failed for this specific render — fall through to HTTP

    # HTTP fallback
    return _render_via_http(puml, server_url)
