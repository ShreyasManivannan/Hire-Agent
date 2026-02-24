"""
Report Generator Agent — Produces structured evaluation reports
with scores, analysis, and hiring recommendations.
"""

import logging
import json
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def generate_report(session_data: dict) -> dict:
    """
    Generate a comprehensive interview evaluation report.

    Args:
        session_data: Complete interview session state

    Returns:
        Structured report dictionary
    """
    candidate_name = session_data.get("candidate_name", "Candidate")
    experience = session_data.get("experience_years", 0)
    skill_scores = session_data.get("skill_scores", {})
    answers = session_data.get("answers", [])
    plan = session_data.get("plan", {})
    face_analysis = session_data.get("face_analysis_history", [])
    voice_analysis = session_data.get("voice_analysis_history", [])

    # Calculate aggregate scores
    total_score = 0
    total_questions = len(answers)
    ai_probabilities = []
    sentiments = []
    strengths = []
    weaknesses = []
    topic_scores = {}

    for answer_data in answers:
        analysis = answer_data.get("analysis", {})
        score = analysis.get("score", 5)
        total_score += score
        ai_probabilities.append(analysis.get("ai_probability", 0.3))
        sentiments.append(analysis.get("sentiment", "neutral"))

        for s in analysis.get("strengths", []):
            if s not in strengths:
                strengths.append(s)
        for w in analysis.get("weaknesses", []):
            if w not in weaknesses:
                weaknesses.append(w)

        topic = answer_data.get("domain", "general")
        if topic not in topic_scores:
            topic_scores[topic] = []
        topic_scores[topic].append(score)

    # Average scores
    avg_score = round(total_score / max(total_questions, 1), 1)
    avg_ai_prob = round(sum(ai_probabilities) / max(len(ai_probabilities), 1), 2)

    # Face analysis summary
    face_summary = _summarize_face_analysis(face_analysis)

    # Voice analysis summary
    voice_summary = _summarize_voice_analysis(voice_analysis)

    # Topic-level scores
    topic_report = {}
    for topic, scores in topic_scores.items():
        topic_avg = round(sum(scores) / len(scores), 1)
        topic_report[topic] = {
            "average_score": topic_avg,
            "questions_answered": len(scores),
            "rating": _score_to_rating(topic_avg),
        }

    # Use skill_scores from planner
    for topic, score in skill_scores.items():
        if topic not in topic_report:
            topic_report[topic] = {
                "average_score": round(score * 10, 1),
                "questions_answered": 0,
                "rating": _score_to_rating(score * 10),
            }

    # Strong and weak areas
    strong_areas = [t for t, r in topic_report.items() if r["average_score"] >= 7]
    weak_areas = [t for t, r in topic_report.items() if r["average_score"] < 5]

    # Skipped topics
    skipped = plan.get("skipped_topics", [])

    # Hiring recommendation
    recommendation = _generate_recommendation(avg_score, avg_ai_prob, total_questions)

    # Overall confidence
    confidence = _calculate_confidence(total_questions, avg_ai_prob, face_summary)

    report = {
        "candidate_name": candidate_name,
        "interview_date": datetime.now().isoformat(),
        "experience_years": experience,
        "total_questions": total_questions,
        "total_duration_minutes": session_data.get("duration_minutes", 0),

        # Scores
        "overall_score": avg_score,
        "overall_rating": _score_to_rating(avg_score),
        "topic_scores": topic_report,

        # Strong/Weak areas
        "strong_areas": strong_areas,
        "weak_areas": weak_areas,
        "skipped_topics": [s["topic"] for s in skipped],

        # AI Detection
        "ai_generated_probability": avg_ai_prob,
        "ai_detection_label": _ai_prob_label(avg_ai_prob),

        # Behavioral Analysis
        "face_analysis": face_summary,
        "voice_analysis": voice_summary,
        "sentiment_distribution": _count_sentiments(sentiments),

        # Recommendations
        "hire_recommendation": recommendation["decision"],
        "recommendation_reasoning": recommendation["reasoning"],
        "confidence_level": confidence,

        # Details
        "strengths": strengths[:10],
        "weaknesses": weaknesses[:10],
        "detailed_answers": [
            {
                "question": a.get("question", ""),
                "answer": a.get("answer", "")[:200],
                "score": a.get("analysis", {}).get("score", 0),
                "domain": a.get("domain", "general"),
            }
            for a in answers
        ],
    }

    logger.info(f"Report generated for {candidate_name}: Score {avg_score}/10, Recommendation: {recommendation['decision']}")
    return report


def _summarize_face_analysis(face_data: list) -> dict:
    """Summarize face detection history."""
    if not face_data:
        return {"status": "no_data", "face_detected_ratio": 0, "suspicion_level": "unknown"}

    detected = sum(1 for f in face_data if f.get("face_detected", False))
    total = len(face_data)
    ratio = detected / max(total, 1)

    if ratio >= 0.9:
        suspicion = "low"
    elif ratio >= 0.7:
        suspicion = "medium"
    else:
        suspicion = "high"

    return {
        "status": "analyzed",
        "face_detected_ratio": round(ratio, 2),
        "total_frames_analyzed": total,
        "suspicion_level": suspicion,
    }


def _summarize_voice_analysis(voice_data: list) -> dict:
    """Summarize voice analysis history."""
    if not voice_data:
        return {"status": "no_data", "avg_confidence": 0}

    confidences = [v.get("confidence", 0.5) for v in voice_data]
    avg_conf = sum(confidences) / len(confidences)

    human_count = sum(1 for v in voice_data if v.get("is_human_likely", True))
    human_ratio = human_count / len(voice_data)

    return {
        "status": "analyzed",
        "avg_confidence": round(avg_conf, 2),
        "human_speech_ratio": round(human_ratio, 2),
        "total_segments": len(voice_data),
    }


def _score_to_rating(score: float) -> str:
    """Convert numeric score to rating label."""
    if score >= 9:
        return "Exceptional"
    elif score >= 7:
        return "Strong"
    elif score >= 5:
        return "Adequate"
    elif score >= 3:
        return "Below Average"
    else:
        return "Poor"


def _ai_prob_label(prob: float) -> str:
    """Convert AI probability to label."""
    if prob >= 0.8:
        return "Highly Likely AI-Generated"
    elif prob >= 0.6:
        return "Likely AI-Assisted"
    elif prob >= 0.4:
        return "Possibly AI-Assisted"
    elif prob >= 0.2:
        return "Likely Human"
    else:
        return "Human-Generated"


def _generate_recommendation(avg_score: float, ai_prob: float, total_questions: int) -> dict:
    """Generate hiring recommendation based on scores."""
    if total_questions == 0:
        return {
            "decision": "Insufficient Data",
            "reasoning": "Interview was too short to make a recommendation."
        }

    if ai_prob >= 0.7:
        return {
            "decision": "Not Recommended",
            "reasoning": f"High probability ({ai_prob:.0%}) of AI-generated answers. Integrity concern."
        }

    if avg_score >= 8:
        decision = "Strongly Recommended"
        reasoning = f"Excellent performance (Score: {avg_score}/10). Demonstrated strong expertise."
    elif avg_score >= 6:
        decision = "Recommended"
        reasoning = f"Good performance (Score: {avg_score}/10). Shows solid understanding."
    elif avg_score >= 4:
        decision = "Consider with Reservations"
        reasoning = f"Average performance (Score: {avg_score}/10). Has potential but gaps exist."
    else:
        decision = "Not Recommended"
        reasoning = f"Below average performance (Score: {avg_score}/10). Significant knowledge gaps."

    return {"decision": decision, "reasoning": reasoning}


def _calculate_confidence(total_questions: int, ai_prob: float, face_summary: dict) -> float:
    """Calculate confidence in the assessment."""
    base = 0.5

    # More questions = more confidence
    question_bonus = min(0.3, total_questions * 0.05)

    # Low AI probability = more confidence
    ai_bonus = 0.1 * (1 - ai_prob)

    # Face detection data = more confidence
    face_bonus = 0.1 if face_summary.get("status") == "analyzed" else 0

    return round(min(1.0, base + question_bonus + ai_bonus + face_bonus), 2)


def _count_sentiments(sentiments: list) -> dict:
    """Count sentiment distribution."""
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for s in sentiments:
        if s in counts:
            counts[s] += 1
    return counts
