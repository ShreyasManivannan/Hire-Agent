"""
Sentiment Analysis — Transformer-based sentiment and confidence analysis.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_sentiment_pipeline = None


def _load_pipeline():
    """Load sentiment analysis pipeline."""
    global _sentiment_pipeline
    try:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        logger.info("Sentiment pipeline loaded")
    except Exception as e:
        logger.warning(f"Failed to load transformer pipeline: {e}")


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of text using transformer model with fallback.

    Returns:
        dict with label, score, confidence
    """
    if not text or len(text.strip()) < 5:
        return {"label": "neutral", "score": 0.5, "confidence": 0.0}

    # Try transformer pipeline
    global _sentiment_pipeline
    try:
        if _sentiment_pipeline is None:
            _load_pipeline()

        if _sentiment_pipeline is not None:
            result = _sentiment_pipeline(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]

            if label == "negative":
                score = 1 - score

            return {
                "label": "positive" if score > 0.6 else ("negative" if score < 0.4 else "neutral"),
                "score": round(score, 3),
                "confidence": round(result["score"], 3),
                "model": "distilbert",
            }
    except Exception as e:
        logger.warning(f"Transformer sentiment failed: {e}")

    # Fallback: keyword-based
    return _keyword_sentiment(text)


def _keyword_sentiment(text: str) -> dict:
    """Keyword-based sentiment fallback."""
    text_lower = text.lower()

    positive = ['good', 'great', 'excellent', 'understand', 'experience',
                'built', 'created', 'solved', 'improved', 'love', 'enjoy',
                'confident', 'success', 'achieve', 'strong']
    negative = ['bad', 'difficult', 'struggle', 'fail', 'don\'t know',
                'confused', 'hard', 'problem', 'issue', 'wrong',
                'never', 'not sure', 'unfortunately']

    pos = sum(1 for w in positive if w in text_lower)
    neg = sum(1 for w in negative if w in text_lower)

    total = pos + neg + 1
    score = (pos + 0.5) / total

    if score > 0.6:
        label = "positive"
    elif score < 0.4:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": round(score, 3),
        "confidence": round(min(1.0, (pos + neg) * 0.15), 3),
        "model": "keyword",
    }


def analyze_answer_confidence(text: str) -> dict:
    """
    Determine how confident the candidate sounds in their answer.
    """
    text_lower = text.lower()

    # Strong confidence indicators
    strong = ['definitely', 'certainly', 'absolutely', 'clearly',
              'i know', 'exactly', 'precisely', 'without a doubt']
    # Moderate confidence
    moderate = ['i think', 'i believe', 'probably', 'usually',
                'in my experience', 'generally', 'mostly']
    # Low confidence
    low = ['maybe', 'i guess', 'not sure', 'i think so',
           'perhaps', 'might be', 'could be', 'don\'t know']

    s_count = sum(1 for w in strong if w in text_lower)
    m_count = sum(1 for w in moderate if w in text_lower)
    l_count = sum(1 for w in low if w in text_lower)

    if s_count > l_count:
        level = "high"
        score = min(1.0, 0.7 + s_count * 0.1)
    elif l_count > s_count:
        level = "low"
        score = max(0.0, 0.3 - l_count * 0.05)
    else:
        level = "moderate"
        score = 0.5

    return {
        "confidence_level": level,
        "confidence_score": round(score, 2),
        "indicators": {
            "strong": s_count,
            "moderate": m_count,
            "low": l_count,
        }
    }
