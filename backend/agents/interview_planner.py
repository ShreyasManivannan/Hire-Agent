"""
Interview Planner Agent — Selects topic priority, defines difficulty curve,
handles topic skipping, and manages time allocation.
"""

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# Domain display names
DOMAIN_NAMES = {
    "programming": "Programming",
    "frontend": "Frontend Development",
    "backend": "Backend Development",
    "cloud": "Cloud & DevOps",
    "devops": "DevOps & CI/CD",
    "database": "Databases",
    "ai_ml": "AI/ML",
    "dsa": "Data Structures & Algorithms",
    "system_design": "System Design",
    "servicenow": "ServiceNow",
    "behavioral": "Behavioral",
    "general": "General",
}


def create_interview_plan(
    candidate_profile: dict,
    duration_minutes: int = 10,
    custom_topics: Optional[list] = None,
    difficulty_preference: str = "adaptive",
) -> dict:
    """
    Create an interview plan based on candidate profile.

    Args:
        candidate_profile: Output from resume analyzer
        duration_minutes: Total interview duration
        custom_topics: User-selected topics (overrides auto-selection)
        difficulty_preference: 'easy', 'medium', 'hard', or 'adaptive'

    Returns:
        Interview plan with topic schedule and configuration
    """
    domains = candidate_profile.get("domains", [])
    skills = candidate_profile.get("skills", [])
    experience = candidate_profile.get("experience_years", 0)

    # Determine topics
    if custom_topics and len(custom_topics) > 0:
        topics = custom_topics
    else:
        # Auto-select based on resume
        topics = list(set(domains))
        if "dsa" not in topics:
            topics.append("dsa")
        if "behavioral" not in topics:
            topics.append("behavioral")

    # Filter to available domain mappings
    valid_topics = [t for t in topics if t in DOMAIN_NAMES or t in _reverse_domain_map()]
    if not valid_topics:
        valid_topics = ["programming", "dsa", "behavioral"]

    # Calculate questions per topic based on duration
    # Roughly 2 minutes per question
    total_questions = max(2, duration_minutes // 2)
    questions_per_topic = max(1, total_questions // len(valid_topics))
    remaining = total_questions - (questions_per_topic * len(valid_topics))

    # Build topic schedule
    topic_schedule = []
    for i, topic in enumerate(valid_topics):
        extra = 1 if i < remaining else 0
        num_questions = questions_per_topic + extra

        # Determine difficulty curve
        if difficulty_preference == "adaptive":
            if experience >= 5:
                difficulties = _generate_difficulty_curve(num_questions, start="medium")
            elif experience >= 2:
                difficulties = _generate_difficulty_curve(num_questions, start="easy")
            else:
                difficulties = _generate_difficulty_curve(num_questions, start="easy")
        else:
            difficulties = [difficulty_preference] * num_questions

        topic_schedule.append({
            "topic": topic,
            "display_name": DOMAIN_NAMES.get(topic, topic.title()),
            "num_questions": num_questions,
            "difficulties": difficulties,
            "related_skills": [s for s in skills if _skill_domain(s) == topic],
            "status": "pending",
            "current_question": 0,
        })

    plan = {
        "candidate_name": candidate_profile.get("name", "Candidate"),
        "total_duration_minutes": duration_minutes,
        "total_questions": total_questions,
        "topics": topic_schedule,
        "current_topic_index": 0,
        "difficulty_preference": difficulty_preference,
        "skill_scores": {topic["topic"]: 0.5 for topic in topic_schedule},
        "skipped_topics": [],
        "asked_question_ids": [],
    }

    logger.info(
        f"Interview plan created: {total_questions} questions, "
        f"{len(valid_topics)} topics, {duration_minutes} min"
    )
    return plan


def get_next_topic(plan: dict) -> Optional[dict]:
    """Get the next topic to ask about, skipping depleted/skipped topics."""
    for i, topic in enumerate(plan["topics"]):
        if topic["status"] == "pending" and topic["current_question"] < topic["num_questions"]:
            plan["current_topic_index"] = i
            return topic
    return None


def get_current_difficulty(plan: dict) -> str:
    """Get the current difficulty level based on plan state."""
    topic_idx = plan.get("current_topic_index", 0)
    if topic_idx < len(plan["topics"]):
        topic = plan["topics"][topic_idx]
        q_idx = topic.get("current_question", 0)
        difficulties = topic.get("difficulties", ["medium"])
        if q_idx < len(difficulties):
            return difficulties[q_idx]
    return "medium"


def skip_topic(plan: dict, topic_name: str, reason: str = "candidate_unknown") -> dict:
    """
    Skip a topic when candidate doesn't know the subject.
    Reduces difficulty and moves to next topic.
    """
    for topic in plan["topics"]:
        if topic["topic"] == topic_name:
            topic["status"] = "skipped"
            plan["skipped_topics"].append({
                "topic": topic_name,
                "reason": reason,
            })
            plan["skill_scores"][topic_name] = 0.1
            logger.info(f"Topic skipped: {topic_name} - {reason}")
            break

    return plan


def update_skill_score(plan: dict, topic_name: str, score: float) -> dict:
    """Update the running skill score for a topic."""
    if topic_name in plan["skill_scores"]:
        current = plan["skill_scores"][topic_name]
        # Exponential moving average
        plan["skill_scores"][topic_name] = 0.6 * score + 0.4 * current
    return plan


def advance_question(plan: dict) -> dict:
    """Move to the next question in the current topic."""
    topic_idx = plan.get("current_topic_index", 0)
    if topic_idx < len(plan["topics"]):
        plan["topics"][topic_idx]["current_question"] += 1
        # Check if topic is complete
        topic = plan["topics"][topic_idx]
        if topic["current_question"] >= topic["num_questions"]:
            topic["status"] = "completed"
    return plan


def is_interview_complete(plan: dict) -> bool:
    """Check if all topics have been completed or skipped."""
    return all(
        topic["status"] in ("completed", "skipped")
        for topic in plan["topics"]
    )


def _generate_difficulty_curve(num_questions: int, start: str = "easy") -> list:
    """Generate a progressive difficulty curve."""
    if num_questions == 1:
        return [start]

    levels = ["easy", "medium", "hard"]
    start_idx = levels.index(start) if start in levels else 0

    curve = []
    for i in range(num_questions):
        progress = i / max(1, num_questions - 1)
        level_idx = min(start_idx + int(progress * (2 - start_idx)), 2)
        curve.append(levels[level_idx])

    return curve


def _skill_domain(skill: str) -> str:
    """Get the domain for a skill."""
    from .resume_analyzer import SKILL_DOMAINS
    return SKILL_DOMAINS.get(skill, "general")


def _reverse_domain_map() -> dict:
    """Build reverse mapping from domain names to keys."""
    return {v.lower(): k for k, v in DOMAIN_NAMES.items()}
