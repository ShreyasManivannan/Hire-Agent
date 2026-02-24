"""
RAG Retriever — Semantic search over ChromaDB interview question bank.
Filters by domain and difficulty, returns top-k relevant Q&A pairs.
Supports adaptive difficulty and deduplication of already-asked questions.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def retrieve_questions(
    query: str,
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
    top_k: int = 5,
    exclude_ids: Optional[List[str]] = None,
    candidate_score: Optional[float] = None,
) -> List[dict]:
    """
    Retrieve relevant interview questions from the knowledge base.

    Uses the same sentence-transformer embeddings that were used to index
    the documents, ensuring consistent high-quality semantic matching.

    Args:
        query: Search query (skill, topic, or question type)
        domain: Filter by domain (python, dsa, system_design, etc.)
        difficulty: Filter by difficulty (easy, medium, hard)
        top_k: Number of results to return
        exclude_ids: Question IDs to exclude (already asked)
        candidate_score: Current running score (0-1) for adaptive difficulty

    Returns:
        List of question documents with metadata and relevance scores
    """
    from .knowledge_base import get_collection

    collection = get_collection()
    if collection is None:
        logger.warning("Knowledge base not initialised — falling back to static list")
        return _fallback_questions(domain, difficulty)

    # Adaptive difficulty: if candidate is doing well, bump difficulty up
    if candidate_score is not None and difficulty == "adaptive":
        if candidate_score >= 0.75:
            difficulty = "hard"
        elif candidate_score >= 0.45:
            difficulty = "medium"
        else:
            difficulty = "easy"

    # Build ChromaDB where-filter
    where_filter = _build_where_filter(domain, difficulty)

    try:
        # Request extra results so that after excluding already-asked IDs
        # we still have enough left
        extra = len(exclude_ids) if exclude_ids else 0
        query_kwargs = {
            "query_texts": [query],
            "n_results": min(top_k + extra + 2, 20),  # cap to avoid huge queries
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        questions = _parse_results(results, exclude_ids, top_k)

        # If we didn't get enough results with strict filters, relax and retry
        if len(questions) < top_k and (domain or difficulty):
            # Retry with domain-only filter
            if domain and difficulty:
                relaxed_filter = {"domain": domain}
                query_kwargs["where"] = relaxed_filter
                query_kwargs["n_results"] = min(top_k + extra + 2, 20)
                results = collection.query(**query_kwargs)
                extra_qs = _parse_results(results, exclude_ids, top_k - len(questions))
                seen_ids = {q["id"] for q in questions}
                for q in extra_qs:
                    if q["id"] not in seen_ids:
                        questions.append(q)
                    if len(questions) >= top_k:
                        break

            # Still short? Remove all filters
            if len(questions) < top_k:
                query_kwargs.pop("where", None)
                query_kwargs["n_results"] = min(top_k + extra + 2, 20)
                results = collection.query(**query_kwargs)
                extra_qs = _parse_results(results, exclude_ids, top_k - len(questions))
                seen_ids = {q["id"] for q in questions}
                for q in extra_qs:
                    if q["id"] not in seen_ids:
                        questions.append(q)
                    if len(questions) >= top_k:
                        break

        logger.info(
            f"Retrieved {len(questions)} questions for query='{query[:50]}...' "
            f"domain={domain} difficulty={difficulty}"
        )
        return questions

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return _fallback_questions(domain, difficulty)


def retrieve_answer_context(question: str, domain: Optional[str] = None, top_k: int = 3) -> List[dict]:
    """
    Retrieve similar questions + answer keys to provide context for scoring.
    Used by the response_analyzer to compare candidate answers against
    expected answer patterns from the knowledge base.
    """
    from .knowledge_base import get_collection

    collection = get_collection()
    if collection is None:
        return []

    try:
        query_kwargs = {"query_texts": [question], "n_results": top_k}
        if domain:
            query_kwargs["where"] = {"domain": domain}

        results = collection.query(**query_kwargs)

        contexts = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                contexts.append({
                    "question": meta.get("question", ""),
                    "answer_key": meta.get("answer_key", ""),
                    "domain": meta.get("domain", ""),
                    "difficulty": meta.get("difficulty", ""),
                })
        return contexts

    except Exception as e:
        logger.error(f"Answer context retrieval failed: {e}")
        return []


# ---------- Internal helpers ----------

def _build_where_filter(domain: Optional[str], difficulty: Optional[str]) -> dict:
    """Build ChromaDB where-clause from domain and difficulty."""
    conditions = []
    if domain:
        conditions.append({"domain": domain})
    if difficulty and difficulty != "adaptive":
        conditions.append({"difficulty": difficulty})

    if len(conditions) > 1:
        return {"$and": conditions}
    elif len(conditions) == 1:
        return conditions[0]
    return {}


def _parse_results(results, exclude_ids: Optional[List[str]], limit: int) -> List[dict]:
    """Parse ChromaDB query results into question dicts."""
    questions = []
    if not results or not results.get("documents"):
        return questions

    for i, doc in enumerate(results["documents"][0]):
        qid = results["ids"][0][i] if results.get("ids") else f"q_{i}"

        if exclude_ids and qid in exclude_ids:
            continue

        metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
        distance = results["distances"][0][i] if results.get("distances") else 0

        # Convert distance to relevance score (ChromaDB returns L2; lower = better)
        relevance = max(0, 1.0 - distance / 2.0)

        questions.append({
            "id": qid,
            "document": doc,
            "question": metadata.get("question", doc),
            "answer_key": metadata.get("answer_key", ""),
            "domain": metadata.get("domain", "general"),
            "difficulty": metadata.get("difficulty", "medium"),
            "tags": metadata.get("tags", ""),
            "distance": distance,
            "relevance": round(relevance, 3),
        })

        if len(questions) >= limit:
            break

    return questions


def _fallback_questions(
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> List[dict]:
    """Fallback questions when ChromaDB is not available."""
    from .knowledge_base import KNOWLEDGE_BASE

    questions = []
    for i, item in enumerate(KNOWLEDGE_BASE):
        if domain and item["domain"] != domain:
            continue
        if difficulty and difficulty != "adaptive" and item["difficulty"] != difficulty:
            continue
        questions.append({
            "id": f"q_{i}",
            "document": item["question"],
            "question": item["question"],
            "answer_key": item.get("answer_key", ""),
            "domain": item["domain"],
            "difficulty": item["difficulty"],
            "tags": ", ".join(item.get("tags", [])),
            "distance": 0,
            "relevance": 1.0,
        })
        if len(questions) >= 5:
            break

    return questions
