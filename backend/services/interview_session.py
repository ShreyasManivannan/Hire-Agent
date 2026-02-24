"""
Interview Session Service — Manages interview state, tracks progress,
and coordinates between agents.
"""

import logging
import uuid
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# In-memory session store
_sessions = {}


class InterviewSession:
    """Manages the state of a single interview session."""

    def __init__(
        self,
        candidate_profile: dict,
        duration_minutes: int = 10,
        custom_topics: Optional[list] = None,
        difficulty: str = "adaptive",
    ):
        self.session_id = str(uuid.uuid4())
        self.candidate_profile = candidate_profile
        self.duration_minutes = duration_minutes
        self.status = "initialized"  # initialized, active, paused, completed, stopped
        self.start_time = None
        self.end_time = None

        # Create interview plan
        from agents.interview_planner import create_interview_plan
        self.plan = create_interview_plan(
            candidate_profile=candidate_profile,
            duration_minutes=duration_minutes,
            custom_topics=custom_topics,
            difficulty_preference=difficulty,
        )

        # Tracking
        self.answers = []
        self.current_question = None
        self.current_question_index = 0
        self.face_analysis_history = []
        self.voice_analysis_history = []
        self.transcript = []

        # Store globally
        _sessions[self.session_id] = self

        logger.info(f"Session created: {self.session_id}")

    def start(self):
        """Start the interview session."""
        self.status = "active"
        self.start_time = time.time()
        logger.info(f"Session started: {self.session_id}")

    def get_next_question(self) -> Optional[dict]:
        """Get the next interview question."""
        from agents.interview_planner import get_next_topic, get_current_difficulty
        from agents.question_generator import generate_question

        if self.is_time_up() or self.is_complete():
            return None

        topic = get_next_topic(self.plan)
        if topic is None:
            return None

        difficulty = get_current_difficulty(self.plan)

        # Get related skill from the topic
        skills = topic.get("related_skills", [])
        skill = skills[0] if skills else topic["topic"]

        # Generate question via RAG
        question = generate_question(
            skill=skill,
            domain=topic["topic"],
            difficulty=difficulty,
            experience_years=self.candidate_profile.get("experience_years", 0),
            asked_ids=self.plan.get("asked_question_ids", []),
        )

        # Track the question
        self.current_question = {
            "question_index": self.current_question_index,
            "question": question.get("question", ""),
            "domain": topic["topic"],
            "difficulty": difficulty,
            "topic_display": topic["display_name"],
            "timestamp": time.time(),
            **question,
        }

        if question.get("source_id"):
            self.plan["asked_question_ids"].append(question["source_id"])

        self.current_question_index += 1

        return self.current_question

    def submit_answer(self, answer_text: str, audio_data: Optional[str] = None) -> dict:
        """Submit an answer and get analysis."""
        from agents.response_analyzer import analyze_response
        from agents.interview_planner import (
            advance_question, skip_topic, update_skill_score
        )

        if not self.current_question:
            return {"error": "No active question"}

        # Analyze the response
        analysis = analyze_response(
            question=self.current_question.get("question", ""),
            answer=answer_text,
            domain=self.current_question.get("domain", "general"),
            difficulty=self.current_question.get("difficulty", "medium"),
        )

        # Voice analysis if audio provided
        voice_result = None
        if audio_data:
            from analysis.voice_analyzer import transcribe_audio, analyze_voice
            transcription = transcribe_audio(audio_data)
            if transcription.get("success"):
                voice_result = analyze_voice(
                    transcription["text"],
                    transcription.get("segments", [])
                )
                self.voice_analysis_history.append(voice_result)

        # Store answer
        answer_record = {
            "question_index": self.current_question.get("question_index", 0),
            "question": self.current_question.get("question", ""),
            "answer": answer_text,
            "domain": self.current_question.get("domain", "general"),
            "difficulty": self.current_question.get("difficulty", "medium"),
            "analysis": analysis,
            "voice_analysis": voice_result,
            "timestamp": time.time(),
        }
        self.answers.append(answer_record)

        # Update transcript
        self.transcript.append({
            "type": "question",
            "text": self.current_question.get("question", ""),
            "timestamp": self.current_question.get("timestamp", time.time()),
        })
        self.transcript.append({
            "type": "answer",
            "text": answer_text,
            "timestamp": time.time(),
        })

        # Handle "don't know" — skip topic
        if analysis.get("is_dont_know"):
            skip_topic(
                self.plan,
                self.current_question.get("domain", "general"),
                reason="candidate said don't know"
            )
        else:
            # Update skill score
            score_normalized = analysis.get("score", 5) / 10.0
            update_skill_score(
                self.plan,
                self.current_question.get("domain", "general"),
                score_normalized
            )
            # Advance to next question
            advance_question(self.plan)

        return {
            "analysis": analysis,
            "voice_analysis": voice_result,
            "is_dont_know": analysis.get("is_dont_know", False),
            "questions_remaining": self._questions_remaining(),
            "time_remaining_seconds": self._time_remaining(),
        }

    def add_face_analysis(self, face_result: dict):
        """Add a face analysis result to history."""
        self.face_analysis_history.append(face_result)

    def stop(self) -> dict:
        """Stop the interview and generate report."""
        self.status = "completed" if self.is_complete() else "stopped"
        self.end_time = time.time()

        from agents.report_generator import generate_report

        session_data = {
            "candidate_name": self.candidate_profile.get("name", "Candidate"),
            "experience_years": self.candidate_profile.get("experience_years", 0),
            "skill_scores": self.plan.get("skill_scores", {}),
            "answers": self.answers,
            "plan": self.plan,
            "face_analysis_history": self.face_analysis_history,
            "voice_analysis_history": self.voice_analysis_history,
            "duration_minutes": round((self.end_time - (self.start_time or self.end_time)) / 60, 1),
        }

        report = generate_report(session_data)
        self.report = report

        logger.info(f"Session {self.session_id} ended: {self.status}")
        return report

    def is_time_up(self) -> bool:
        """Check if interview time is up."""
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        return elapsed >= self.duration_minutes * 60

    def is_complete(self) -> bool:
        """Check if interview is complete."""
        from agents.interview_planner import is_interview_complete
        return is_interview_complete(self.plan)

    def _questions_remaining(self) -> int:
        """Count remaining questions."""
        total = sum(t["num_questions"] for t in self.plan["topics"])
        asked = sum(t.get("current_question", 0) for t in self.plan["topics"])
        skipped_q = sum(
            t["num_questions"] for t in self.plan["topics"]
            if t["status"] == "skipped"
        )
        return max(0, total - asked - skipped_q)

    def _time_remaining(self) -> float:
        """Get remaining time in seconds."""
        if self.start_time is None:
            return self.duration_minutes * 60
        elapsed = time.time() - self.start_time
        return max(0, self.duration_minutes * 60 - elapsed)

    def get_state(self) -> dict:
        """Get current session state for frontend."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "candidate_name": self.candidate_profile.get("name", "Candidate"),
            "current_question": self.current_question,
            "questions_answered": len(self.answers),
            "questions_remaining": self._questions_remaining(),
            "time_remaining_seconds": self._time_remaining(),
            "skill_scores": self.plan.get("skill_scores", {}),
            "current_topic": (
                self.plan["topics"][self.plan.get("current_topic_index", 0)]["display_name"]
                if self.plan["topics"] else "General"
            ),
            "face_status": (
                self.face_analysis_history[-1] if self.face_analysis_history else None
            ),
        }


def get_session(session_id: str) -> Optional[InterviewSession]:
    """Get a session by ID."""
    return _sessions.get(session_id)
