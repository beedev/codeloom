"""Query logging service for token usage tracking and cost calculation."""

import logging
from datetime import datetime
from typing import Dict, Optional, List
from uuid import uuid4

logger = logging.getLogger(__name__)


# Model pricing database (per 1M tokens)
MODEL_PRICING = {
    # OpenAI (per 1M tokens)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.00},

    # Anthropic (per 1M tokens)
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},

    # Google Gemini (per 1M tokens)
    "gemini-2.0-flash-exp": {"input": 0.00, "output": 0.00},  # Free tier
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},

    # Ollama (local - no cost)
    "llama3.1": {"input": 0.00, "output": 0.00},
    "llama3.1:latest": {"input": 0.00, "output": 0.00},
    "llama3.1:70b": {"input": 0.00, "output": 0.00},
    "deepseek-r1:32b": {"input": 0.00, "output": 0.00},
    "deepseek-r1": {"input": 0.00, "output": 0.00},

    # Groq (per 1M tokens - very competitive pricing)
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.20, "output": 0.60},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},  # Groq GPT OSS 120B 128k
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama3-70b-8192": {"input": 0.59, "output": 0.79},
    "llama3-8b-8192": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},

    # OpenAI newer models (GPT-4.1 series)
    "gpt-4.1": {"input": 2.00, "output": 8.00},  # Full model
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},

}


class QueryLogger:
    """
    Query logging service for tracking token usage and calculating costs.

    Provides in-memory logging with optional database persistence.
    Tracks token usage, response times, and estimated costs per query.
    """

    def __init__(self, db_manager=None):
        """
        Initialize query logger.

        Args:
            db_manager: Optional database manager for persistent storage
        """
        self.db = db_manager
        self._in_memory_logs: List[Dict] = []
        logger.info("QueryLogger initialized")

    def log_query(
        self,
        notebook_id: Optional[str],
        user_id: str,
        query_text: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time_ms: int
    ) -> str:
        """
        Log a query with token usage and timing information.

        Args:
            notebook_id: Notebook identifier
            user_id: User identifier
            query_text: The query text
            model_name: LLM model used
            prompt_tokens: Input token count
            completion_tokens: Output token count
            response_time_ms: Response time in milliseconds

        Returns:
            Log ID (UUID)
        """
        log_id = str(uuid4())
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = self.estimate_cost(model_name, prompt_tokens, completion_tokens)

        log_entry = {
            "log_id": log_id,
            "notebook_id": notebook_id,
            "user_id": user_id,
            "query_text": query_text,
            "model_name": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "response_time_ms": response_time_ms,
            "estimated_cost": estimated_cost,
            "timestamp": datetime.utcnow()
        }

        # Store in memory
        self._in_memory_logs.append(log_entry)

        # Store in database if available
        if self.db:
            try:
                with self.db.get_session() as session:
                    from ..db.models import QueryLog

                    query_log = QueryLog(
                        log_id=log_id,
                        notebook_id=notebook_id,
                        user_id=user_id,
                        query_text=query_text,
                        model_name=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        response_time_ms=response_time_ms,
                        timestamp=log_entry["timestamp"]
                    )
                    session.add(query_log)
                logger.debug(f"Query logged to database: {log_id}")
            except Exception as e:
                logger.error(f"Failed to log query to database: {e}")

        logger.info(
            f"Query logged | Model: {model_name} | "
            f"Tokens: {total_tokens} ({prompt_tokens} in, {completion_tokens} out) | "
            f"Time: {response_time_ms}ms | Cost: ${estimated_cost:.4f}"
        )

        return log_id

    def estimate_cost(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate estimated cost based on token usage and model pricing.

        Args:
            model_name: LLM model name
            prompt_tokens: Input token count
            completion_tokens: Output token count

        Returns:
            Estimated cost in USD
        """
        # Normalize model name
        model_key = model_name.lower().strip()

        # Check if model exists in pricing database
        if model_key not in MODEL_PRICING:
            # Try to find partial match
            for key in MODEL_PRICING.keys():
                if key in model_key or model_key in key:
                    model_key = key
                    break
            else:
                # Unknown model, assume free (local) or return 0
                logger.warning(f"Unknown model '{model_name}', assuming free/local")
                return 0.0

        pricing = MODEL_PRICING[model_key]

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return total_cost

    def get_usage_stats(
        self,
        notebook_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get usage statistics with optional filtering.

        Args:
            notebook_id: Filter by notebook ID
            user_id: Filter by user ID
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            Dictionary with usage statistics:
            - total_queries: Total number of queries
            - total_tokens: Total tokens used
            - total_cost: Total estimated cost
            - avg_response_time: Average response time in ms
            - queries_by_model: Breakdown by model
        """
        # Filter logs based on criteria
        filtered_logs = self._in_memory_logs

        if notebook_id:
            filtered_logs = [log for log in filtered_logs if log["notebook_id"] == notebook_id]

        if user_id:
            filtered_logs = [log for log in filtered_logs if log["user_id"] == user_id]

        if start_date:
            filtered_logs = [log for log in filtered_logs if log["timestamp"] >= start_date]

        if end_date:
            filtered_logs = [log for log in filtered_logs if log["timestamp"] <= end_date]

        # Calculate statistics
        if not filtered_logs:
            return {
                "total_queries": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_response_time": 0.0,
                "queries_by_model": {}
            }

        total_queries = len(filtered_logs)
        total_tokens = sum(log["total_tokens"] for log in filtered_logs)
        total_cost = sum(log["estimated_cost"] for log in filtered_logs)
        avg_response_time = sum(log["response_time_ms"] for log in filtered_logs) / total_queries

        # Breakdown by model
        queries_by_model = {}
        for log in filtered_logs:
            model = log["model_name"]
            if model not in queries_by_model:
                queries_by_model[model] = {
                    "count": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            queries_by_model[model]["count"] += 1
            queries_by_model[model]["tokens"] += log["total_tokens"]
            queries_by_model[model]["cost"] += log["estimated_cost"]

        return {
            "total_queries": total_queries,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_response_time": avg_response_time,
            "queries_by_model": queries_by_model
        }

    def get_recent_logs(self, limit: int = 50) -> List[Dict]:
        """
        Get most recent query logs.

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of log entries sorted by timestamp (newest first)
        """
        sorted_logs = sorted(
            self._in_memory_logs,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        return sorted_logs[:limit]

    def clear_logs(self) -> None:
        """Clear all in-memory logs."""
        self._in_memory_logs.clear()
        logger.info("Query logs cleared")

    def get_model_pricing(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get pricing information for a specific model.

        Args:
            model_name: Model name

        Returns:
            Dictionary with 'input' and 'output' pricing per 1M tokens,
            or None if model not found
        """
        model_key = model_name.lower().strip()
        return MODEL_PRICING.get(model_key)

    def list_supported_models(self) -> List[str]:
        """
        Get list of all models with pricing information.

        Returns:
            List of model names
        """
        return list(MODEL_PRICING.keys())

    def get_admin_metrics(self, days: int = 30) -> Dict:
        """
        Get aggregated metrics for admin dashboard.

        Aggregates token usage, costs, and query counts by model, user, and day
        from the database (with in-memory fallback).

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            Dictionary with:
            - summary: Total tokens, cost, queries, avg_response_time
            - by_model: List of metrics grouped by model
            - by_user: List of metrics grouped by user
            - by_day: List of metrics grouped by day
        """
        from datetime import timedelta
        from sqlalchemy import func, cast, Date

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Default empty response
        empty_response = {
            "summary": {
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_queries": 0,
                "avg_response_time": 0.0
            },
            "by_model": [],
            "by_user": [],
            "by_day": []
        }

        # Try database query first
        if self.db:
            try:
                with self.db.get_session() as session:
                    from ..db.models import QueryLog, User

                    # Base query for time range
                    base_query = session.query(QueryLog).filter(
                        QueryLog.timestamp >= cutoff_date
                    )

                    # Get total count
                    total_queries = base_query.count()
                    if total_queries == 0:
                        return empty_response

                    # Summary aggregation
                    summary_result = session.query(
                        func.sum(QueryLog.total_tokens).label('total_tokens'),
                        func.sum(QueryLog.prompt_tokens).label('prompt_tokens'),
                        func.sum(QueryLog.completion_tokens).label('completion_tokens'),
                        func.avg(QueryLog.response_time_ms).label('avg_response_time')
                    ).filter(QueryLog.timestamp >= cutoff_date).first()

                    # Calculate total cost from token usage
                    total_prompt = summary_result.prompt_tokens or 0
                    total_completion = summary_result.completion_tokens or 0

                    # Get model breakdown for accurate cost calculation
                    model_results = session.query(
                        QueryLog.model_name,
                        func.count(QueryLog.log_id).label('query_count'),
                        func.sum(QueryLog.total_tokens).label('total_tokens'),
                        func.sum(QueryLog.prompt_tokens).label('prompt_tokens'),
                        func.sum(QueryLog.completion_tokens).label('completion_tokens')
                    ).filter(
                        QueryLog.timestamp >= cutoff_date
                    ).group_by(QueryLog.model_name).all()

                    by_model = []
                    total_cost = 0.0
                    for row in model_results:
                        model_name = row.model_name or "unknown"
                        prompt = row.prompt_tokens or 0
                        completion = row.completion_tokens or 0
                        cost = self.estimate_cost(model_name, prompt, completion)
                        total_cost += cost
                        by_model.append({
                            "model": model_name,
                            "tokens": row.total_tokens or 0,
                            "cost": round(cost, 4),
                            "queries": row.query_count
                        })

                    # Sort by tokens descending
                    by_model.sort(key=lambda x: x["tokens"], reverse=True)

                    # User aggregation with username lookup and accurate cost per model
                    user_model_results = session.query(
                        QueryLog.user_id,
                        User.username,
                        QueryLog.model_name,
                        func.count(QueryLog.log_id).label('query_count'),
                        func.sum(QueryLog.total_tokens).label('total_tokens'),
                        func.sum(QueryLog.prompt_tokens).label('prompt_tokens'),
                        func.sum(QueryLog.completion_tokens).label('completion_tokens')
                    ).join(
                        User, QueryLog.user_id == User.user_id, isouter=True
                    ).filter(
                        QueryLog.timestamp >= cutoff_date
                    ).group_by(QueryLog.user_id, User.username, QueryLog.model_name).all()

                    # Aggregate by user with accurate per-model cost calculation
                    user_agg = {}
                    for row in user_model_results:
                        user_id = str(row.user_id)
                        username = row.username or user_id[:8]
                        model_name = row.model_name or "unknown"
                        prompt = row.prompt_tokens or 0
                        completion = row.completion_tokens or 0
                        cost = self.estimate_cost(model_name, prompt, completion)

                        if user_id not in user_agg:
                            user_agg[user_id] = {
                                "username": username,
                                "tokens": 0,
                                "cost": 0.0,
                                "queries": 0
                            }
                        user_agg[user_id]["tokens"] += row.total_tokens or 0
                        user_agg[user_id]["cost"] += cost
                        user_agg[user_id]["queries"] += row.query_count

                    by_user = [
                        {
                            "user_id": uid,
                            "username": data["username"],
                            "tokens": data["tokens"],
                            "cost": round(data["cost"], 4),
                            "queries": data["queries"]
                        }
                        for uid, data in user_agg.items()
                    ]

                    # Sort by tokens descending
                    by_user.sort(key=lambda x: x["tokens"], reverse=True)

                    # Daily aggregation
                    day_results = session.query(
                        cast(QueryLog.timestamp, Date).label('date'),
                        func.count(QueryLog.log_id).label('query_count'),
                        func.sum(QueryLog.total_tokens).label('total_tokens')
                    ).filter(
                        QueryLog.timestamp >= cutoff_date
                    ).group_by(
                        cast(QueryLog.timestamp, Date)
                    ).order_by(
                        cast(QueryLog.timestamp, Date)
                    ).all()

                    by_day = []
                    for row in day_results:
                        by_day.append({
                            "date": row.date.isoformat() if row.date else None,
                            "tokens": row.total_tokens or 0,
                            "queries": row.query_count
                        })

                    return {
                        "summary": {
                            "total_tokens": summary_result.total_tokens or 0,
                            "total_cost": round(total_cost, 4),
                            "total_queries": total_queries,
                            "avg_response_time": round(summary_result.avg_response_time or 0, 2)
                        },
                        "by_model": by_model,
                        "by_user": by_user,
                        "by_day": by_day
                    }

            except Exception as e:
                logger.error(f"Failed to get admin metrics from database: {e}")
                # Fall through to in-memory fallback

        # In-memory fallback
        filtered_logs = [
            log for log in self._in_memory_logs
            if log["timestamp"] >= cutoff_date
        ]

        if not filtered_logs:
            return empty_response

        # Summary
        total_tokens = sum(log["total_tokens"] for log in filtered_logs)
        total_cost = sum(log["estimated_cost"] for log in filtered_logs)
        avg_response_time = sum(log["response_time_ms"] for log in filtered_logs) / len(filtered_logs)

        # By model
        model_agg = {}
        for log in filtered_logs:
            model = log["model_name"]
            if model not in model_agg:
                model_agg[model] = {"tokens": 0, "cost": 0.0, "queries": 0}
            model_agg[model]["tokens"] += log["total_tokens"]
            model_agg[model]["cost"] += log["estimated_cost"]
            model_agg[model]["queries"] += 1

        by_model = [
            {"model": m, "tokens": d["tokens"], "cost": round(d["cost"], 4), "queries": d["queries"]}
            for m, d in model_agg.items()
        ]
        by_model.sort(key=lambda x: x["tokens"], reverse=True)

        # By user (with username lookup from database if available)
        user_agg = {}
        for log in filtered_logs:
            user_id = log["user_id"]
            if user_id not in user_agg:
                user_agg[user_id] = {"tokens": 0, "cost": 0.0, "queries": 0}
            user_agg[user_id]["tokens"] += log["total_tokens"]
            user_agg[user_id]["cost"] += log["estimated_cost"]
            user_agg[user_id]["queries"] += 1

        # Lookup usernames from database if available
        usernames = {}
        if self.db:
            try:
                with self.db.get_session() as session:
                    from ..db.models import User
                    from uuid import UUID as UUID_type
                    for user_id in user_agg.keys():
                        try:
                            user = session.query(User).filter(
                                User.user_id == UUID_type(user_id)
                            ).first()
                            if user:
                                usernames[user_id] = user.username
                        except (ValueError, Exception):
                            pass
            except Exception:
                pass

        by_user = [
            {
                "user_id": u,
                "username": usernames.get(u, u[:8] if len(u) > 8 else u),
                "tokens": d["tokens"],
                "cost": round(d["cost"], 4),
                "queries": d["queries"]
            }
            for u, d in user_agg.items()
        ]
        by_user.sort(key=lambda x: x["tokens"], reverse=True)

        # By day
        day_agg = {}
        for log in filtered_logs:
            day = log["timestamp"].date().isoformat()
            if day not in day_agg:
                day_agg[day] = {"tokens": 0, "queries": 0}
            day_agg[day]["tokens"] += log["total_tokens"]
            day_agg[day]["queries"] += 1

        by_day = [
            {"date": d, "tokens": data["tokens"], "queries": data["queries"]}
            for d, data in sorted(day_agg.items())
        ]

        return {
            "summary": {
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 4),
                "total_queries": len(filtered_logs),
                "avg_response_time": round(avg_response_time, 2)
            },
            "by_model": by_model,
            "by_user": by_user,
            "by_day": by_day
        }
