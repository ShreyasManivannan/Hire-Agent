"""
Response Analyzer Agent — Scores answers against RAG answer keys,
detects AI-generated content via stylometric analysis, and performs
sentiment/confidence analysis. ZERO static data — everything driven
by RAG knowledge base and LLM.
"""

import re
import math
import logging
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
def analyze_response(
    question: str,
    answer: str,
    domain: str = "general",
    difficulty: str = "medium",
    session_id: Optional[str] = None,
) -> dict:
    """
    Full response analysis pipeline:
      1. Retrieve answer keys from ChromaDB (RAG)
      2. Check answer correctness against expected key points
      3. Score with LLM using retrieved context
      4. Run AI detection (multi-signal stylometric analysis)
      5. Sentiment and confidence analysis

    Returns:
        dict with score, feedback, correctness, ai_probability, sentiment, etc.
    """
    if not answer or len(answer.strip()) < 5:
        return {
            "score": 0,
            "score_percent": 0,
            "feedback": "No meaningful answer provided.",
            "ai_probability": 0.0,
            "ai_signals": [],
            "sentiment": "neutral",
            "sentiment_score": 0.5,
            "confidence": 0.0,
            "is_dont_know": _is_dont_know(answer),
            "strengths": [],
            "weaknesses": ["No answer provided"],
            "word_count": 0,
            "correctness_score": 0,
            "correctness_details": {},
        }

    # Step 1: Retrieve answer context from RAG
    answer_context = _retrieve_answer_context(question, domain)

    # Step 2: Correctness check against answer keys
    correctness = _check_correctness(answer, answer_context)

    # Step 3: LLM scoring with RAG context
    score_result = _score_answer(question, answer, domain, difficulty, answer_context, correctness)

    # Step 4: AI detection
    ai_detection = detect_ai_content(answer)

    # Step 5: Sentiment + confidence
    sentiment = _analyze_sentiment(answer)
    is_dont_know = _is_dont_know(answer)

    raw_score = score_result.get("score", 5)
    # If clearly AI-generated, penalize score slightly
    if ai_detection["probability"] > 0.75:
        raw_score = max(0, raw_score - 1.5)
        score_result.setdefault("weaknesses", []).append("High AI-generation probability detected")

    return {
        "score": round(raw_score, 1),
        "score_percent": round((raw_score / 10) * 100),
        "feedback": score_result.get("feedback", ""),
        "strengths": score_result.get("strengths", []),
        "weaknesses": score_result.get("weaknesses", []),
        "ai_probability": ai_detection["probability"],
        "ai_signals": ai_detection["signals"],
        "ai_analysis": ai_detection.get("analysis", {}),
        "sentiment": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "confidence": sentiment.get("confidence", 0.5),
        "is_dont_know": is_dont_know,
        "word_count": len(answer.split()),
        "correctness_score": correctness["score"],
        "correctness_details": correctness,
        "rag_context_used": len(answer_context) > 0,
    }


# ──────────────────────────────────────────────────────────────────────
# Step 1: Retrieve RAG answer context
# ──────────────────────────────────────────────────────────────────────
def _retrieve_answer_context(question: str, domain: str) -> list:
    """Pull similar questions + answer keys from ChromaDB."""
    try:
        from rag.retriever import retrieve_answer_context
        contexts = retrieve_answer_context(question, domain=domain, top_k=3)
        return contexts
    except Exception as e:
        logger.warning(f"RAG answer context retrieval failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────
# Step 2: Correctness check (answer vs answer key keywords)
# ──────────────────────────────────────────────────────────────────────
def _check_correctness(answer: str, answer_context: list) -> dict:
    """
    Compare candidate's answer against knowledge base answer keys.
    Returns a correctness score (0-1) and matched/missed key points.
    """
    if not answer_context:
        return {"score": 0.5, "matched_points": [], "missed_points": [], "note": "No answer key available"}

    answer_lower = answer.lower()
    all_matched = []
    all_missed = []
    best_answer_key = ""

    for ctx in answer_context:
        key = ctx.get("answer_key", "")
        if not key:
            continue
        best_answer_key = key

        # Extract key concepts from answer_key (words > 3 chars)
        key_points = [w.strip(".,;:") for w in key.lower().split() if len(w) > 3]
        # Also extract noun phrases (consecutive capitalized or important terms)
        important = [kp for kp in key_points if kp not in {
            "that", "this", "with", "from", "they", "when", "where",
            "have", "each", "than", "also", "which", "both"
        }]

        matched = [kp for kp in important if kp in answer_lower]
        missed = [kp for kp in important if kp not in answer_lower][:5]

        all_matched.extend(matched)
        all_missed.extend(missed)

    # De-duplicate
    all_matched = list(set(all_matched))
    all_missed = list(set(all_missed) - set(all_matched))

    # Correctness score: ratio of matched/total expected points
    total = len(all_matched) + len(all_missed)
    if total == 0:
        correctness_score = 0.5
    else:
        correctness_score = round(len(all_matched) / total, 2)

    return {
        "score": correctness_score,
        "matched_points": all_matched[:8],
        "missed_points": all_missed[:5],
        "answer_key_preview": best_answer_key[:200] if best_answer_key else "",
    }


# ──────────────────────────────────────────────────────────────────────
# Step 3: LLM scoring with RAG context
# ──────────────────────────────────────────────────────────────────────
def _score_answer(
    question: str,
    answer: str,
    domain: str,
    difficulty: str,
    answer_context: list,
    correctness: dict,
) -> dict:
    """Score the answer using LLM with injected RAG answer-key context."""
    from llm.inference import generate_response, parse_json_response

    # Build reference context from RAG answer keys
    rag_context = ""
    if answer_context:
        rag_lines = []
        for ctx in answer_context[:2]:
            if ctx.get("answer_key"):
                rag_lines.append(f"  - Expected: {ctx['answer_key']}")
        if rag_lines:
            rag_context = "\n**Reference Answer Key Points (from knowledge base)**:\n" + "\n".join(rag_lines)

    correctness_note = (
        f"\n**Correctness Check**: {round(correctness['score'] * 100)}% key points covered. "
        f"Matched: {', '.join(correctness.get('matched_points', [])[:5]) or 'none'}. "
        f"Missed: {', '.join(correctness.get('missed_points', [])[:3]) or 'none'}."
        if correctness.get("answer_key_preview") else ""
    )

    prompt = f"""Evaluate this technical interview answer on a scale of 0-10.

**Question**: {question}
**Domain**: {domain} | **Difficulty**: {difficulty}
**Candidate's Answer**: {answer}
{rag_context}{correctness_note}

Scoring criteria:
- 0-2: Completely wrong, irrelevant, or refuses to answer
- 3-4: Shows basic awareness but significant gaps or errors
- 5-6: Adequate understanding, some gaps
- 7-8: Strong answer with good depth and accuracy
- 9-10: Expert-level with examples, nuances, and practical insight

Also check: Is this answer suspiciously perfect or AI-generated?

Respond ONLY in valid JSON:
{{
    "score": <number 0-10>,
    "feedback": "<brief constructive feedback>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>"],
    "is_correct": <true/false>,
    "correctness_note": "<what was right or wrong about the technical content>"
}}"""

    response = generate_response(
        prompt=prompt,
        system_prompt="You are a fair, expert technical interviewer. Evaluate answers objectively. Always respond in valid JSON.",
        temperature=0.25,
        json_output=True,
    )

    result = parse_json_response(response)

    # Fallback: heuristic scoring if LLM fails
    if "score" not in result:
        word_count = len(answer.split())
        correctness_boost = correctness["score"] * 3  # up to +3 for correctness
        base = min(6, max(2, word_count // 15))
        result = {
            "score": round(min(9, base + correctness_boost), 1),
            "feedback": "Answer scored using correctness analysis (LLM unavailable).",
            "strengths": [f"Covered key points: {', '.join(correctness.get('matched_points', [])[:3])}"] if correctness.get("matched_points") else [],
            "weaknesses": [f"Missing: {', '.join(correctness.get('missed_points', [])[:3])}"] if correctness.get("missed_points") else ["Needs more depth"],
            "is_correct": correctness["score"] > 0.5,
            "correctness_note": f"{round(correctness['score'] * 100)}% key points covered",
        }

    return result


# ──────────────────────────────────────────────────────────────────────
# Step 4: AI detection — multi-signal stylometric analysis
# ──────────────────────────────────────────────────────────────────────
def detect_ai_content(text: str) -> dict:
    """
    Detect AI-generated content using 7 stylometric signals:
      1. Burstiness — sentence length variation
      2. Sentence starter uniformity
      3. Formal/transitional marker density
      4. Filler word absence
      5. Bigram repetition (AI repeats phrase structures)
      6. Perfect formatting (no typos, no contractions)
      7. Hedge word / uncertainty ratio (AI is very confident)

    Each signal contributes a weighted score. Returns probability (0-1).
    """
    if not text or len(text) < 20:
        return {"probability": 0.0, "signals": [], "analysis": {}}

    signals = []
    weighted_scores = []  # (score, weight) tuples
    text_lower = text.lower()
    words = text.split()
    word_count = len(words)

    # ── Signal 1: Burstiness (sentence length variation) ──
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        burstiness = math.sqrt(variance) / max(mean_len, 1)

        if burstiness < 0.25:
            signals.append("Very uniform sentence lengths (low burstiness)")
            weighted_scores.append((0.8, 2.0))
        elif burstiness < 0.45:
            weighted_scores.append((0.45, 2.0))
        else:
            weighted_scores.append((0.1, 2.0))

    # ── Signal 2: Sentence starter repetition ──
    if len(sentences) >= 4:
        starters = [s.split()[0].lower() if s.split() else "" for s in sentences]
        counter = Counter(starters)
        top_ratio = counter.most_common(1)[0][1] / len(starters)
        if top_ratio > 0.45:
            signals.append("Repetitive sentence starters")
            weighted_scores.append((0.65, 1.5))
        else:
            weighted_scores.append((0.2, 1.5))

    # ── Signal 3: Formal/transitional markers ──
    formal_markers = [
        r'\bfurthermore\b', r'\bmoreover\b', r'\bin conclusion\b',
        r'\bconsequently\b', r'\bnevertheless\b', r'\bthus\b',
        r'\btherefore\b', r'\bit is important to note\b',
        r'\bit should be noted\b', r'\bin summary\b', r'\bto summarize\b',
        r'\badditionally\b', r'\bsubsequently\b', r'\bin essence\b',
        r'\bin terms of\b', r'\bhere are\b',
    ]
    formal_count = sum(1 for m in formal_markers if re.search(m, text_lower))
    if formal_count >= 4:
        signals.append(f"High formal marker density ({formal_count} found)")
        weighted_scores.append((0.9, 2.5))
    elif formal_count >= 2:
        weighted_scores.append((0.55, 2.5))
    else:
        weighted_scores.append((0.1, 2.5))

    # ── Signal 4: Filler word absence ──
    fillers = [r'\bum\b', r'\buh\b', r'\byou know\b', r'\bi mean\b',
               r'\bkind of\b', r'\bsort of\b', r'\bbasically\b']
    filler_count = sum(1 for f in fillers if re.search(f, text_lower))
    if filler_count == 0 and word_count > 40:
        signals.append("No natural speech fillers in long answer")
        weighted_scores.append((0.55, 1.0))
    elif filler_count >= 2:
        weighted_scores.append((0.1, 1.0))
    else:
        weighted_scores.append((0.3, 1.0))

    # ── Signal 5: Bigram phrase repetition (AI likes repeating structures) ──
    if word_count >= 30:
        word_lower = [w.lower().strip(".,!?;:") for w in words]
        bigrams = [f"{word_lower[i]}_{word_lower[i+1]}" for i in range(len(word_lower)-1)]
        bigram_counter = Counter(bigrams)
        repeated = sum(1 for _, cnt in bigram_counter.items() if cnt > 2)
        if repeated >= 3:
            signals.append("Repeated phrase structures (high bigram repetition)")
            weighted_scores.append((0.65, 1.5))
        else:
            weighted_scores.append((0.15, 1.5))

    # ── Signal 6: Perfect formatting / no contractions ──
    human_patterns = [
        r"\bi\b",          # lowercase 'i' (human typing habit)
        r"don't|doesn't|isn't|can't|won't|wouldn't|shouldn't|didn't",
        r"\.{2,}",         # ellipsis or multiple dots
        r"\b(hmm|yeah|nope|yep|ok|okay)\b",
    ]
    human_hits = sum(1 for p in human_patterns if re.search(p, text, re.IGNORECASE))
    if human_hits == 0 and word_count > 40:
        signals.append("Unusually perfect formatting, no contractions or informal language")
        weighted_scores.append((0.6, 1.5))
    else:
        weighted_scores.append((0.1, 1.5))

    # ── Signal 7: Hedge/uncertainty ratio ──
    hedges = [r'\bmaybe\b', r'\bi think\b', r'\bprobably\b', r'\bnot sure\b',
              r'\bi believe\b', r'\bi guess\b', r'\bperhaps\b', r'\bseems\b']
    hedge_count = sum(1 for h in hedges if re.search(h, text_lower))
    hedge_ratio = hedge_count / max(word_count / 10, 1)
    if hedge_ratio < 0.1 and word_count > 50:
        # AI almost never hedges in long answers
        weighted_scores.append((0.5, 1.0))
    else:
        weighted_scores.append((0.15, 1.0))

    # ── Final weighted probability ──
    if not weighted_scores:
        probability = 0.3
    else:
        total_weight = sum(w for _, w in weighted_scores)
        probability = sum(s * w for s, w in weighted_scores) / total_weight

    probability = round(min(1.0, max(0.0, probability)), 3)

    # Label
    if probability >= 0.7:
        label = "LIKELY AI"
    elif probability >= 0.45:
        label = "POSSIBLY AI"
    else:
        label = "LIKELY HUMAN"

    return {
        "probability": probability,
        "label": label,
        "signals": signals,
        "analysis": {
            "sentence_count": len(sentences),
            "word_count": word_count,
            "formal_marker_count": formal_count,
            "filler_count": filler_count,
            "hedge_count": hedge_count,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Step 5: Sentiment + confidence
# ──────────────────────────────────────────────────────────────────────
def _analyze_sentiment(text: str) -> dict:
    """Analyze sentiment and confidence from answer text."""
    text_lower = text.lower()

    positive = ['understand', 'experience', 'implemented', 'built', 'created',
                'solved', 'improved', 'optimized', 'achieved', 'successfully',
                'used', 'worked on', 'familiar with']
    negative = ['not sure', 'maybe', 'i guess', "don't know", 'unclear',
                'confused', 'difficult', "haven't", 'never used']
    confident = ['definitely', 'certainly', 'absolutely', 'clearly',
                 'obviously', 'precisely', 'exactly', 'specifically', 'always']

    pos = sum(1 for w in positive if w in text_lower)
    neg = sum(1 for w in negative if w in text_lower)
    conf = sum(1 for w in confident if w in text_lower)

    if pos > neg * 1.5:
        label, score = "positive", min(1.0, 0.5 + pos * 0.08)
    elif neg > pos:
        label, score = "negative", max(0.0, 0.5 - neg * 0.08)
    else:
        label, score = "neutral", 0.5

    confidence = min(1.0, 0.3 + conf * 0.12 + min(len(text.split()) / 200, 0.3))

    return {"label": label, "score": round(score, 2), "confidence": round(confidence, 2)}


def _is_dont_know(text: str) -> bool:
    if not text:
        return True
    patterns = [
        r"i\s*don'?t\s*know", r"no\s*idea", r"not\s*sure",
        r"i\s*can'?t\s*answer", r"\bskip\b", r"\bpass\b",
        r"i\s*haven'?t\s*(?:learned|studied|worked)", r"not\s*familiar",
    ]
    txt = text.lower().strip()
    if len(txt.split()) <= 4:
        return any(re.search(p, txt) for p in patterns)
    return False
