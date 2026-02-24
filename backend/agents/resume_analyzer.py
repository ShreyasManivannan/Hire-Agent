"""
Resume Analyzer Agent — Extracts structured profile from uploaded PDF resume.
Uses PyMuPDF for text extraction and pattern matching for skill/experience parsing.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Skill taxonomy mapping skills to domains
SKILL_DOMAINS = {
    "python": "programming",
    "java": "programming",
    "javascript": "programming",
    "typescript": "programming",
    "c++": "programming",
    "c#": "programming",
    "go": "programming",
    "rust": "programming",
    "ruby": "programming",
    "php": "programming",
    "swift": "programming",
    "kotlin": "programming",
    "react": "frontend",
    "angular": "frontend",
    "vue": "frontend",
    "html": "frontend",
    "css": "frontend",
    "next.js": "frontend",
    "nextjs": "frontend",
    "tailwind": "frontend",
    "node.js": "backend",
    "nodejs": "backend",
    "express": "backend",
    "django": "backend",
    "flask": "backend",
    "fastapi": "backend",
    "spring": "backend",
    "spring boot": "backend",
    ".net": "backend",
    "graphql": "backend",
    "rest api": "backend",
    "aws": "cloud",
    "azure": "cloud",
    "gcp": "cloud",
    "docker": "devops",
    "kubernetes": "devops",
    "jenkins": "devops",
    "terraform": "devops",
    "ci/cd": "devops",
    "github actions": "devops",
    "sql": "database",
    "mysql": "database",
    "postgresql": "database",
    "mongodb": "database",
    "redis": "database",
    "elasticsearch": "database",
    "machine learning": "ai_ml",
    "deep learning": "ai_ml",
    "tensorflow": "ai_ml",
    "pytorch": "ai_ml",
    "nlp": "ai_ml",
    "computer vision": "ai_ml",
    "llm": "ai_ml",
    "transformers": "ai_ml",
    "rag": "ai_ml",
    "data structures": "dsa",
    "algorithms": "dsa",
    "system design": "system_design",
    "microservices": "system_design",
    "servicenow": "servicenow",
    "glide record": "servicenow",
    "itil": "servicenow",
}


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except ImportError:
        logger.error("PyMuPDF not installed. Install with: pip install PyMuPDF")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""


def extract_skills(text: str) -> list:
    """Extract technical skills from resume text."""
    text_lower = text.lower()
    found_skills = []

    for skill in SKILL_DOMAINS:
        # Use word boundary matching for short skills
        if len(skill) <= 3:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        else:
            if skill in text_lower:
                found_skills.append(skill)

    return list(set(found_skills))


def extract_experience_years(text: str) -> float:
    """Extract years of experience from resume text."""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience\s*:?\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:professional|work|industry)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1))

    # Estimate from date ranges
    year_ranges = re.findall(r'(20\d{2})\s*[-–]\s*(20\d{2}|present|current)', text.lower())
    if year_ranges:
        total_years = 0
        for start, end in year_ranges:
            start_year = int(start)
            end_year = 2026 if end in ('present', 'current') else int(end)
            total_years += end_year - start_year
        return min(total_years, 30)  # Cap at 30 years

    return 0


def extract_education(text: str) -> list:
    """Extract education details from resume text."""
    education = []
    degree_patterns = [
        r"((?:bachelor|master|phd|b\.?tech|m\.?tech|b\.?e|m\.?e|b\.?sc|m\.?sc|b\.?s|m\.?s|mba|bca|mca)[\w\s.]*?)(?:\n|,|;|\||from)",
        r"((?:bachelor|master|phd|b\.?tech|m\.?tech|mba|bca|mca)[\w\s.]*)",
    ]

    for pattern in degree_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip().rstrip(',;|')
            if len(cleaned) > 3 and cleaned not in education:
                education.append(cleaned)

    return education[:5]  # Max 5 entries


def extract_projects(text: str) -> list:
    """Extract project names/descriptions from resume text."""
    projects = []

    # Look for project headers
    project_section = re.search(
        r'(?:projects?|portfolio|work samples?)[\s:]*\n(.*?)(?=\n(?:education|experience|skills|certifications?)|$)',
        text, re.IGNORECASE | re.DOTALL
    )

    if project_section:
        lines = project_section.group(1).strip().split('\n')
        for line in lines:
            line = line.strip().strip('•-*▪◦')
            if len(line) > 10 and len(line) < 200:
                projects.append(line.strip())

    return projects[:10]  # Max 10 projects


def analyze_resume(pdf_bytes: bytes) -> dict:
    """
    Full resume analysis pipeline.
    Returns structured candidate profile.
    """
    text = extract_text_from_pdf(pdf_bytes)

    if not text:
        return {
            "error": "Could not extract text from PDF",
            "name": "Unknown",
            "skills": [],
            "domains": [],
            "experience_years": 0,
            "education": [],
            "projects": [],
            "skill_graph": {},
        }

    skills = extract_skills(text)
    experience = extract_experience_years(text)
    education = extract_education(text)
    projects = extract_projects(text)

    # Build skill → domain mapping
    skill_graph = {}
    domains = set()
    for skill in skills:
        domain = SKILL_DOMAINS.get(skill, "general")
        domains.add(domain)
        if domain not in skill_graph:
            skill_graph[domain] = []
        skill_graph[domain].append(skill)

    # Extract name (first line heuristic)
    name = "Candidate"
    lines = text.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        if 5 < len(line) < 50 and not re.search(r'@|http|phone|address|\d{5,}', line.lower()):
            name = line
            break

    profile = {
        "name": name,
        "skills": skills,
        "domains": list(domains),
        "experience_years": experience,
        "education": education,
        "projects": projects,
        "skill_graph": skill_graph,
        "raw_text_length": len(text),
    }

    logger.info(f"Resume analyzed: {name}, {len(skills)} skills, {experience} years experience")
    return profile
