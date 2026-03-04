"""Dataset flow edge detector for JCL batch jobs.

Produces `data_flow` edges between JCL steps that share a dataset:
  step_A (writer) → step_B (reader) with edge_metadata describing the DSN.

JCL execution is sequential within a job — dataset flow edges model the
dependency between steps that produce and consume the same datasets.

Producer detection: DISP contains 'NEW' or 'PASS' (step writes the dataset)
Consumer detection: DISP is 'SHR', 'OLD', or '(OLD,...)' without PASS
Temporary datasets (&&name) are included — they flow within the same job.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List

from .context import EdgeContext

logger = logging.getLogger(__name__)

# DISP value patterns
_DISP_NEW_RE = re.compile(r"\bNEW\b", re.IGNORECASE)
_DISP_PASS_RE = re.compile(r"\bPASS\b", re.IGNORECASE)
_DISP_READ_RE = re.compile(r"^(?:SHR|OLD)$", re.IGNORECASE)


def _is_producer(disp: str | None) -> bool:
    """Return True if DISP indicates this step writes/creates the dataset."""
    if not disp:
        return False
    # NEW: step creates the dataset
    # PASS: step keeps the dataset for the next step
    return bool(_DISP_NEW_RE.search(disp)) or bool(_DISP_PASS_RE.search(disp))


def _is_consumer(disp: str | None) -> bool:
    """Return True if DISP indicates this step reads an existing dataset."""
    if not disp:
        return False
    # SHR or OLD (without NEW) = reading an existing dataset
    if _DISP_PASS_RE.search(disp):
        return False  # PASS is producer behaviour
    return bool(_DISP_READ_RE.match(disp.strip("() ")))


def detect_dataset_flow(ctx: EdgeContext) -> List[dict]:
    """Detect data_flow edges between JCL steps that share a dataset.

    Algorithm:
    1. Group step units by job file (JCL execution is per-file sequential).
    2. For each step, read metadata["dd_statements"] for DD allocation data.
    3. Track producers: {dsn → step_unit} (DISP=NEW or PASS).
    4. For consumer steps (DISP=SHR or OLD), if the DSN has a producer
       in the same job → emit a data_flow edge from producer to consumer.

    Edge metadata:
      {"dsn": "PROD.CUST.MASTER", "ddname_producer": "OUTFILE",
       "ddname_consumer": "INFILE", "producer_disp": "NEW,CATLG"}
    """
    edges: List[dict] = []

    # Group JCL step units by file (= same JCL job)
    steps_by_file: Dict[str, list] = defaultdict(list)
    for u in ctx.units:
        if u.unit_type == "step" and u.language == "jcl":
            steps_by_file[str(u.file_id)].append(u)

    for file_id, steps in steps_by_file.items():
        # Map DSN (normalised upper) → (step_unit, ddname, disp)
        producers: Dict[str, tuple] = {}

        for step in steps:
            dd_stmts = (step.unit_metadata or {}).get("dd_statements", [])
            for dd in dd_stmts:
                dsn = dd.get("dsn")
                if not dsn:
                    continue
                dsn_upper = dsn.upper().lstrip("&")  # normalise temp (&&) datasets
                disp = dd.get("disp", "")
                ddname = dd.get("ddname", "")

                if _is_producer(disp):
                    producers[dsn_upper] = (step, ddname, disp)

        # Second pass: find consumers and emit edges
        for step in steps:
            dd_stmts = (step.unit_metadata or {}).get("dd_statements", [])
            for dd in dd_stmts:
                dsn = dd.get("dsn")
                if not dsn:
                    continue
                dsn_upper = dsn.upper().lstrip("&")
                disp = dd.get("disp", "")
                ddname = dd.get("ddname", "")

                if not _is_consumer(disp):
                    continue

                producer_info = producers.get(dsn_upper)
                if not producer_info:
                    continue
                prod_step, prod_ddname, prod_disp = producer_info

                # Avoid self-loop (same step as both producer and consumer)
                if prod_step.unit_id == step.unit_id:
                    continue

                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": prod_step.unit_id,
                    "target_unit_id": step.unit_id,
                    "edge_type": "data_flow",
                    "edge_metadata": {
                        "dsn": dsn_upper,
                        "ddname_producer": prod_ddname,
                        "ddname_consumer": ddname,
                        "producer_disp": prod_disp,
                    },
                })

    return edges
