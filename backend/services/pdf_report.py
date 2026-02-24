"""
PDF Report Generator — Creates downloadable PDF reports from interview data.
Uses fpdf2 for PDF generation with styled formatting.
"""

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def generate_pdf_report(report: dict, output_dir: str = "./reports") -> str:
    """
    Generate a styled PDF report from interview data.

    Args:
        report: Report dictionary from report_generator agent
        output_dir: Directory to save PDF

    Returns:
        Path to generated PDF file
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.error("fpdf2 not installed")
        return ""

    os.makedirs(output_dir, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 30, 100)
    pdf.cell(0, 15, "Interview Evaluation Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Divider
    pdf.set_draw_color(100, 100, 200)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # Candidate Info
    _add_section_header(pdf, "Candidate Information")
    _add_info_row(pdf, "Name", report.get("candidate_name", "N/A"))
    _add_info_row(pdf, "Experience", f"{report.get('experience_years', 0)} years")
    _add_info_row(pdf, "Interview Date", _format_date(report.get("interview_date", "")))
    _add_info_row(pdf, "Duration", f"{report.get('total_duration_minutes', 0)} minutes")
    _add_info_row(pdf, "Questions Asked", str(report.get("total_questions", 0)))
    pdf.ln(5)

    # Overall Score
    _add_section_header(pdf, "Overall Assessment")
    score = report.get("overall_score", 0)
    rating = report.get("overall_rating", "N/A")
    _add_score_box(pdf, "Overall Score", f"{score}/10", rating)
    pdf.ln(3)

    # Hiring Recommendation
    recommendation = report.get("hire_recommendation", "N/A")
    confidence = report.get("confidence_level", 0)
    _add_info_row(pdf, "Recommendation", recommendation)
    _add_info_row(pdf, "Confidence", f"{confidence:.0%}")
    reasoning = report.get("recommendation_reasoning", "")
    if reasoning:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, reasoning)
    pdf.ln(5)

    # AI Detection
    _add_section_header(pdf, "AI Detection Analysis")
    ai_prob = report.get("ai_generated_probability", 0)
    ai_label = report.get("ai_detection_label", "N/A")
    _add_info_row(pdf, "AI Probability", f"{ai_prob:.0%}")
    _add_info_row(pdf, "Assessment", ai_label)
    pdf.ln(5)

    # Topic Scores
    topic_scores = report.get("topic_scores", {})
    if topic_scores:
        _add_section_header(pdf, "Topic-wise Performance")
        for topic, data in topic_scores.items():
            _add_topic_score(pdf, topic.replace("_", " ").title(), data)
        pdf.ln(5)

    # Strong Areas
    strong = report.get("strong_areas", [])
    if strong:
        _add_section_header(pdf, "Strong Areas")
        for area in strong:
            _add_bullet(pdf, area.replace("_", " ").title())
        pdf.ln(3)

    # Weak Areas
    weak = report.get("weak_areas", [])
    if weak:
        _add_section_header(pdf, "Areas for Improvement")
        for area in weak:
            _add_bullet(pdf, area.replace("_", " ").title())
        pdf.ln(3)

    # Skipped Topics
    skipped = report.get("skipped_topics", [])
    if skipped:
        _add_section_header(pdf, "Skipped Topics")
        for topic in skipped:
            _add_bullet(pdf, topic.replace("_", " ").title())
        pdf.ln(3)

    # Behavioral Analysis
    _add_section_header(pdf, "Behavioral Analysis")
    face = report.get("face_analysis", {})
    voice = report.get("voice_analysis", {})

    if face.get("status") == "analyzed":
        _add_info_row(pdf, "Face Detection", f"{face.get('face_detected_ratio', 0):.0%} frames")
        _add_info_row(pdf, "Suspicion Level", face.get("suspicion_level", "N/A").title())

    if voice.get("status") == "analyzed":
        _add_info_row(pdf, "Human Speech", f"{voice.get('human_speech_ratio', 0):.0%}")
        _add_info_row(pdf, "Voice Confidence", f"{voice.get('avg_confidence', 0):.0%}")

    sentiment = report.get("sentiment_distribution", {})
    if sentiment:
        total_s = sum(sentiment.values())
        if total_s > 0:
            _add_info_row(
                pdf, "Sentiment",
                f"Positive: {sentiment.get('positive', 0)}, "
                f"Neutral: {sentiment.get('neutral', 0)}, "
                f"Negative: {sentiment.get('negative', 0)}"
            )
    pdf.ln(5)

    # Detailed Q&A (if space)
    detailed = report.get("detailed_answers", [])
    if detailed:
        pdf.add_page()
        _add_section_header(pdf, "Detailed Question & Answer Log")
        for i, qa in enumerate(detailed, 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 30, 100)
            pdf.multi_cell(0, 5, f"Q{i}. [{qa.get('domain', 'general').title()}] {qa.get('question', 'N/A')}")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 50, 50)
            answer_preview = qa.get("answer", "No answer")[:200]
            pdf.multi_cell(0, 5, f"A: {answer_preview}")
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, f"Score: {qa.get('score', 0)}/10", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # Footer
    pdf.set_y(-30)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Generated by HireAgent AI Interview System | {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    # Save
    filename = f"interview_report_{report.get('candidate_name', 'candidate').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(output_dir, filename)
    pdf.output(filepath)

    logger.info(f"PDF report generated: {filepath}")
    return filepath


def _add_section_header(pdf, title: str):
    """Add a styled section header."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(40, 40, 120)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(200, 200, 220)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(3)


def _add_info_row(pdf, label: str, value: str):
    """Add a label-value row."""
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(60, 6, f"{label}:", new_x="END")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")


def _add_score_box(pdf, label: str, score: str, rating: str):
    """Add a highlighted score display."""
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 100, 30)
    pdf.cell(60, 12, score, new_x="END")
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 12, f"({rating})", new_x="LMARGIN", new_y="NEXT")


def _add_topic_score(pdf, topic: str, data: dict):
    """Add topic score row."""
    score = data.get("average_score", 0)
    rating = data.get("rating", "N/A")
    questions = data.get("questions_answered", 0)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(70, 6, f"  {topic}", new_x="END")
    pdf.cell(30, 6, f"{score}/10", new_x="END")
    pdf.cell(30, 6, rating, new_x="END")
    pdf.cell(0, 6, f"({questions} Q)", new_x="LMARGIN", new_y="NEXT")


def _add_bullet(pdf, text: str):
    """Add a bullet point."""
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(10, 5, chr(8226), new_x="END")
    pdf.cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")


def _format_date(date_str: str) -> str:
    """Format ISO date string."""
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except (ValueError, TypeError):
        return datetime.now().strftime("%B %d, %Y at %I:%M %p")
