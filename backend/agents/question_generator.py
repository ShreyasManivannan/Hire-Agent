"""
Question Generator Agent — Generates UNIQUE, FRESH interview questions using
a two-stage pipeline:
  Stage 1: RAG retrieves candidate's resume highlights + knowledge base topics
           to build rich context (NOT to repeat those questions)
  Stage 2: LLM generates a BRAND NEW question using that context as inspiration,
           adapting to experience level, previous answers, and conversation history

Key design: The LLM is explicitly told NOT to repeat or copy retrieved questions.
It uses them only as topic seeds. Temperature is tuned for variety.
"""

import logging
import json
import hashlib
from typing import Optional, List

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Experience → difficulty mapping
# ──────────────────────────────────────────────────────────────────────
def _exp_to_difficulty(years: float, override: str = "adaptive") -> str:
    """Map experience years to difficulty level."""
    if override != "adaptive":
        return override
    if years >= 6:
        return "hard"
    elif years >= 3:
        return "medium"
    else:
        return "easy"


def _exp_to_level(years: float) -> str:
    if years >= 7:
        return "senior engineer (7+ years)"
    elif years >= 4:
        return "mid-level engineer (4-7 years)"
    elif years >= 1:
        return "junior engineer (1-3 years)"
    else:
        return "entry-level / fresher"


# ──────────────────────────────────────────────────────────────────────
# Core question generation — fully dynamic via LLM
# ──────────────────────────────────────────────────────────────────────
def generate_question(
    skill: str,
    domain: str,
    difficulty: str = "adaptive",
    experience_years: float = 0,
    asked_ids: Optional[List[str]] = None,
    asked_questions: Optional[List[str]] = None,
    context: str = "",
    resume_profile: Optional[dict] = None,
    candidate_score: Optional[float] = None,
    session_id: Optional[str] = None,
    conversation_history: Optional[List[dict]] = None,
) -> dict:
    """
    Generate a unique, never-repeated interview question.

    The flow is:
      1. Resolve effective difficulty from experience + candidate score
      2. Query ChromaDB resume collection for candidate-specific context
      3. Query ChromaDB knowledge base for topic breadth (used as inspiration only)
      4. Build a rich prompt with previous questions listed to AVOID repeating them
      5. LLM generates a completely new question at a higher temperature for variety
      6. Return question + follow-up + expected topics

    Args:
        skill:              Primary skill to test (e.g. "React", "system design")
        domain:             Domain category (python, dsa, system_design, etc.)
        difficulty:         "easy"|"medium"|"hard"|"adaptive"
        experience_years:   Candidate's years of experience from resume
        asked_ids:          ChromaDB doc IDs already used (for exclusion)
        asked_questions:    Full text of previously asked questions (to avoid duplicates)
        resume_profile:     Full parsed resume dict for personalisation
        candidate_score:    Running average score 0-1 (for adaptive difficulty)
        session_id:         Session ID for resume ChromaDB collection
        conversation_history: List of {question, answer} dicts for this session

    Returns:
        dict with question, difficulty, domain, follow_up, expected_topics, source
    """
    if asked_ids is None:
        asked_ids = []
    if asked_questions is None:
        asked_questions = []
    if conversation_history is None:
        conversation_history = []

    resume_skills = []
    projects = []
    if resume_profile:
        resume_skills = resume_profile.get("skills", [])
        projects = resume_profile.get("projects", [])

    # ── Step 1: Resolve difficulty ──
    effective_difficulty = _exp_to_difficulty(experience_years, difficulty)

    # Adaptive: if candidate is scoring high, bump difficulty
    if candidate_score is not None and difficulty == "adaptive":
        if candidate_score >= 0.75:
            effective_difficulty = "hard"
        elif candidate_score >= 0.4:
            effective_difficulty = "medium"
        else:
            effective_difficulty = "easy"

    level_label = _exp_to_level(experience_years)

    # ── Step 2: Query resume RAG collection for personalised context ──
    resume_context_docs = []
    if session_id and resume_profile:
        try:
            from rag.knowledge_base import query_resume_collection
            resume_context_docs = query_resume_collection(
                query=f"{skill} {domain} interview",
                session_id=session_id,
                top_k=3,
                domain_filter=domain if domain not in ("general", "behavioral") else None,
            )
        except Exception as e:
            logger.warning(f"Resume RAG query failed: {e}")

    # ── Step 3: Query knowledge base for topic breadth ──
    kb_topics = []
    try:
        from rag.retriever import retrieve_questions
        query = f"{skill} {domain} {effective_difficulty} technical question"
        kb_results = retrieve_questions(
            query=query,
            domain=domain,
            difficulty=effective_difficulty,
            top_k=4,
            exclude_ids=asked_ids,
            candidate_score=candidate_score,
        )
        kb_topics = [r.get("question", "") for r in kb_results if r.get("question")]
    except Exception as e:
        logger.warning(f"Knowledge base query failed: {e}")

    # ── Step 4: Build dynamic prompt ──

    # Format previous questions to explicitly avoid
    avoid_block = ""
    all_previous = list(set(asked_questions + kb_topics[:2]))
    if all_previous:
        avoid_list = "\n".join(f"  - {q[:120]}" for q in all_previous[:8])
        avoid_block = f"\n**QUESTIONS ALREADY ASKED — DO NOT REPEAT OR REPHRASE THESE:**\n{avoid_list}"

    # Format resume highlights
    resume_block = ""
    if resume_skills:
        resume_block = f"\n**Candidate's Tech Stack**: {', '.join(resume_skills[:12])}"
    if projects:
        resume_block += f"\n**Projects**: {'; '.join(projects[:3])}"

    # Format conversation so far (for follow-up intelligence)
    conv_block = ""
    if conversation_history:
        recent = conversation_history[-3:]  # last 3 exchanges
        lines = []
        for ex in recent:
            lines.append(f"  Q: {ex.get('question', '')[:100]}")
            ans = ex.get('answer', '')
            if ans:
                lines.append(f"  A: {ans[:120]}...")
        conv_block = "\n**Recent Interview Exchanges**:\n" + "\n".join(lines)

    # KB topics as inspiration (not to copy)
    inspiration = ""
    if kb_topics:
        inspiration = (
            "\n**Topic areas from knowledge base** (use as INSPIRATION for your new question "
            "— do NOT repeat or reword these):\n" +
            "\n".join(f"  • {t[:100]}" for t in kb_topics[:3])
        )

    prompt = f"""You are a seasoned technical interviewer with 15 years of experience.
Generate a COMPLETELY NEW, UNIQUE interview question for this candidate.

**Skill Focus**: {skill}
**Domain**: {domain}
**Difficulty**: {effective_difficulty.upper()}
**Candidate Level**: {level_label}
**Additional Context**: {context or "None"}{resume_block}{conv_block}{inspiration}{avoid_block}

**STRICT RULES:**
1. Generate a NEW question — do NOT copy, repeat, or rephrase any question listed above
2. Make it SPECIFIC to the candidate's tech stack and experience level
3. For {effective_difficulty} difficulty: {"ask about edge cases, design trade-offs, or system-level thinking" if effective_difficulty == "hard" else "test practical understanding with real-world scenarios" if effective_difficulty == "medium" else "test foundational concepts with simple practical examples"}
4. For {experience_years} years of experience, the question complexity must match
5. Generate a different ANGLE on {skill} than any previous question
6. Use creative scenario framing, real-world problems, or code snippet analysis
7. The follow_up should probe DEEPER into the same concept, not ask something new

Respond ONLY with valid JSON:
{{
    "question": "<your unique, creative question here>",
    "difficulty": "{effective_difficulty}",
    "domain": "{domain}",
    "follow_up": "<a deeper follow-up probing the same concept>",
    "expected_topics": ["<key concept 1>", "<key concept 2>", "<key concept 3>"],
    "angle": "<brief note on what makes this question unique from typical {skill} questions>"
}}"""

    # ── Step 5: Generate via LLM (high temp for variety) ──
    try:
        from llm.inference import generate_response, parse_json_response

        response = generate_response(
            prompt=prompt,
            system_prompt=(
                "You are an elite technical interviewer. You generate insightful, creative interview "
                "questions that reveal true depth of understanding. You NEVER repeat questions. "
                "Always respond with valid JSON only."
            ),
            temperature=0.92,   # High temperature = diverse, creative questions
            json_output=True,
        )
        result = parse_json_response(response)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        result = {}

    # ── Step 6: Fallback if LLM fails ──
    if "question" not in result:
        if kb_topics:
            # Use a KB topic but frame it differently
            base = kb_topics[0]
            result = {
                "question": f"In a production system using {skill}, "
                            f"how would you handle: {base.replace('Explain ', '').replace('What is ', '')}?",
                "difficulty": effective_difficulty,
                "domain": domain,
                "follow_up": f"What are the trade-offs in your approach and how would you measure success?",
                "expected_topics": [skill, domain],
                "angle": "Production-focused framing",
                "source": "rag_fallback",
            }
        else:
            result = {
                "question": (
                    f"You're designing a new feature in a {skill} codebase at scale. "
                    f"Walk me through your approach to [{'architecture decisions and trade-offs' if effective_difficulty=='hard' else 'implementation and testing' if effective_difficulty=='medium' else 'core concepts involved'}]."
                ),
                "difficulty": effective_difficulty,
                "domain": domain,
                "follow_up": "What would change if the scale was 10x larger?",
                "expected_topics": [skill],
                "angle": "Scenario-based",
                "source": "hard_fallback",
            }

    # Tag metadata
    result["skill"] = skill
    result["experience_years"] = experience_years
    result["rag_kb_hits"] = len(kb_topics)
    result["resume_context_hits"] = len(resume_context_docs)

    # Generate stable ID for deduplication
    q_hash = hashlib.md5(result["question"].encode()).hexdigest()[:12]
    result["question_id"] = f"gen_{q_hash}"

    logger.info(
        f"Generated question: skill={skill}, domain={domain}, "
        f"diff={effective_difficulty}, exp={experience_years}yrs, "
        f"kb_hits={len(kb_topics)}, resume_hits={len(resume_context_docs)}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────
# Batch generation from resume
# ──────────────────────────────────────────────────────────────────────
def generate_from_resume(
    resume_profile: dict,
    domains: Optional[List[str]] = None,
    num_questions: int = 5,
    difficulty: str = "adaptive",
    session_id: Optional[str] = None,
) -> List[dict]:
    """
    Generate a diverse set of interview questions from a resume profile.
    Uses round-robin across domains and skills to ensure variety.
    Tracks asked_questions text to prevent semantic duplicates.
    """
    skills = resume_profile.get("skills", [])
    skill_graph = resume_profile.get("skill_graph", {})
    experience = resume_profile.get("experience_years", 0)

    available_domains = list(skill_graph.keys()) if skill_graph else ["python"]
    if domains:
        available_domains = [d for d in domains if d in available_domains] or available_domains

    questions = []
    asked_ids = []
    asked_questions = []
    conversation_history = []

    for i in range(num_questions):
        domain = available_domains[i % len(available_domains)]
        domain_skills = skill_graph.get(domain, skills[:3])

        # Rotate through skills in this domain to ensure variety
        skill_idx = (i // len(available_domains)) % max(len(domain_skills), 1)
        primary_skill = domain_skills[skill_idx] if domain_skills else "software engineering"

        q = generate_question(
            skill=primary_skill,
            domain=domain,
            difficulty=difficulty,
            experience_years=experience,
            asked_ids=asked_ids,
            asked_questions=asked_questions,
            resume_profile=resume_profile,
            session_id=session_id,
            conversation_history=conversation_history,
        )

        qid = q.get("question_id", f"q_{i}")
        asked_ids.append(qid)
        asked_questions.append(q.get("question", ""))
        questions.append(q)

    logger.info(f"Batch generated {len(questions)} questions ({len(available_domains)} domains, {experience}yrs)")
    return questions


# ──────────────────────────────────────────────────────────────────────
# Follow-up generation (context-aware)
# ──────────────────────────────────────────────────────────────────────
def generate_follow_up(
    original_question: str,
    candidate_answer: str,
    domain: str,
    difficulty: str = "medium",
    ai_probability: float = 0.0,
) -> dict:
    """
    Generate a targeted follow-up that:
    - Probes deeper into the candidate's specific answer
    - If AI-generation is suspected, asks for a concrete personal scenario
    - Tests practical application of the concept
    """
    from llm.inference import generate_response, parse_json_response

    ai_note = ""
    if ai_probability > 0.6:
        ai_note = (
            "\nNOTE: The answer shows signs of AI generation. "
            "Generate a follow-up that requires a PERSONAL, SPECIFIC scenario "
            "the candidate must describe from their own experience. "
            "Ask 'Tell me about a specific time when YOU...' type questions."
        )

    prompt = f"""Generate a targeted follow-up question based on this interview exchange.

**Original Question**: {original_question}
**Candidate's Answer**: {candidate_answer[:500]}
**Domain**: {domain}
**Difficulty**: {difficulty}{ai_note}

The follow-up must:
1. Build DIRECTLY on what the candidate said (reference specific points they made)
2. Test DEEPER understanding of the same concept
3. Ask about a real-world edge case or failure mode
4. NOT introduce a completely new topic

Respond in JSON:
{{
    "question": "<follow-up question that directly references their answer>",
    "purpose": "deeper_understanding|practical_application|consistency_check|personal_experience",
    "difficulty": "{difficulty}",
    "targets": "<specific part of their answer you are probing>"
}}"""

    response = generate_response(
        prompt=prompt,
        system_prompt="You are a critical technical interviewer generating penetrating follow-up questions. Return valid JSON only.",
        temperature=0.7,
        json_output=True,
    )

    result = parse_json_response(response)
    if "question" not in result:
        # Smart fallback based on answer content
        result = {
            "question": (
                "You mentioned several key points — can you walk me through a "
                "specific situation from your own projects where you encountered "
                f"a challenge with {domain} and how you resolved it?"
            ),
            "purpose": "personal_experience",
            "difficulty": difficulty,
            "targets": "personal project experience",
        }

    return result
