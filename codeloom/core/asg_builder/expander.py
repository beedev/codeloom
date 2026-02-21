"""ASG-Expanded Retrieval — enriches retrieval results with graph neighbors.

After hybrid BM25+vector retrieval returns initial chunks, this expander
finds ASG-related code units (callers, callees, imports) and adds them
to the result set with decayed scores, before final reranking.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from llama_index.core.schema import NodeWithScore, TextNode

from .queries import get_callers, get_callees, get_dependencies, get_dependents
from ..db import DatabaseManager

logger = logging.getLogger(__name__)


class ASGExpander:
    """Expand retrieval results using ASG neighbor traversal.

    Given initial retrieval results, looks up each chunk's unit_id
    in the ASG graph, fetches neighbors (callers + callees + imports),
    and adds their corresponding TextNodes with decayed scores.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def expand(
        self,
        results: List[NodeWithScore],
        project_id: str,
        cached_nodes: List[TextNode],
        max_expansion: int = 12,
        score_decay: float = 0.7,
        intent: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """Expand retrieval results with ASG neighbors.

        Args:
            results: Initial retrieval results (NodeWithScore list)
            project_id: UUID string of the project
            cached_nodes: All cached TextNodes for the project
            max_expansion: Maximum total results after expansion
            score_decay: Score multiplier for neighbor nodes (0.0-1.0)
            intent: Optional query intent to adjust expansion behavior.
                'flow' → deeper expansion (depth=2, decay=0.5) to
                         trace complete execution paths.
                'data_lifecycle' → follows data flow edges with
                         moderate depth.
                None → default behavior (depth=1, standard decay).

        Returns:
            Expanded list of NodeWithScore, sorted by score descending
        """
        if not results:
            return results

        # Adjust expansion parameters based on intent
        neighbor_depth = 1
        if intent == "impact":
            neighbor_depth = 3
            score_decay = min(score_decay, 0.4)
            max_expansion = max(max_expansion, 24)
        elif intent == "flow":
            neighbor_depth = 2
            score_decay = min(score_decay, 0.5)
            max_expansion = max(max_expansion, 20)
        elif intent == "data_lifecycle":
            neighbor_depth = 2
            score_decay = min(score_decay, 0.6)

        # Build a lookup from unit_id -> TextNode for fast neighbor resolution
        node_by_unit_id: Dict[str, TextNode] = {}
        for node in cached_nodes:
            uid = (node.metadata or {}).get("unit_id")
            if uid:
                node_by_unit_id[uid] = node

        # Track which nodes are already in results (by node id)
        seen_node_ids: Set[str] = set()
        for nws in results:
            seen_node_ids.add(nws.node.node_id)

        # Collect neighbor nodes
        neighbor_nodes: List[NodeWithScore] = []

        for nws in results:
            meta = nws.node.metadata or {}
            unit_id = meta.get("unit_id")
            if not unit_id:
                continue

            # Get ASG neighbors based on intent
            neighbors = []
            try:
                if intent == "impact":
                    neighbors.extend(get_dependents(self._db, project_id, unit_id, depth=neighbor_depth))
                else:
                    neighbors.extend(get_callers(self._db, project_id, unit_id, depth=neighbor_depth))
                    neighbors.extend(get_callees(self._db, project_id, unit_id, depth=neighbor_depth))
            except Exception as e:
                logger.debug(f"ASG neighbor lookup failed for unit {unit_id}: {e}")
                continue

            # Resolve neighbor unit_ids to TextNodes
            for neighbor in neighbors:
                neighbor_uid = neighbor["unit_id"]
                neighbor_node = node_by_unit_id.get(neighbor_uid)

                if not neighbor_node:
                    continue
                if neighbor_node.node_id in seen_node_ids:
                    continue

                seen_node_ids.add(neighbor_node.node_id)
                decayed_score = nws.score * score_decay if nws.score else score_decay

                neighbor_nodes.append(
                    NodeWithScore(
                        node=neighbor_node,
                        score=decayed_score,
                    )
                )

                # Respect expansion budget
                if len(results) + len(neighbor_nodes) >= max_expansion:
                    break

            if len(results) + len(neighbor_nodes) >= max_expansion:
                break

        # Merge original + neighbors and sort by score
        merged = list(results) + neighbor_nodes
        merged.sort(key=lambda nws: nws.score or 0.0, reverse=True)

        logger.debug(
            f"ASG expansion: {len(results)} initial + {len(neighbor_nodes)} neighbors "
            f"= {len(merged)} total"
        )

        return merged
