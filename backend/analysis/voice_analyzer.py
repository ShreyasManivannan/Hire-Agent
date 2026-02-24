"""
Voice Analyzer — Speech-to-text with Whisper and AI vs Human speech detection.
Analyzes speaking patterns, pace, pauses, and filler words.
"""

import logging
import base64
import tempfile
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_whisper_model = None


def _load_whisper(model_size: str = "base"):
    """Load Whisper model for speech-to-text."""
    global _whisper_model
    try:
        import whisper
        _whisper_model = whisper.load_model(model_size)
        logger.info(f"Whisper model loaded: {model_size}")
    except ImportError:
        logger.warning("Whisper not installed. Using fallback transcription.")
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")


def transcribe_audio(audio_data: str, format: str = "webm") -> dict:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_data: Base64-encoded audio data
        format: Audio format (webm, wav, mp3)

    Returns:
        dict with text, language, segments
    """
    global _whisper_model

    try:
        # Decode audio
        audio_bytes = base64.b64decode(audio_data)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            if _whisper_model is None:
                _load_whisper()

            if _whisper_model is not None:
                result = _whisper_model.transcribe(temp_path)
                return {
                    "text": result["text"].strip(),
                    "language": result.get("language", "en"),
                    "segments": [
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"],
                        }
                        for seg in result.get("segments", [])
                    ],
                    "success": True,
                }
            else:
                return {
                    "text": "",
                    "language": "en",
                    "segments": [],
                    "success": False,
                    "error": "Whisper model not available",
                }
        finally:
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {
            "text": "",
            "language": "en",
            "segments": [],
            "success": False,
            "error": str(e),
        }


def analyze_voice(transcribed_text: str, segments: list = None) -> dict:
    """
    Analyze voice/speech patterns to detect AI vs human speech.

    Looks at:
    - Speaking pace (words per minute)
    - Pause patterns
    - Filler word frequency
    - Sentence flow naturalness
    """
    if not transcribed_text or len(transcribed_text.strip()) < 5:
        return {
            "is_human_likely": True,
            "confidence": 0.0,
            "speaking_pace": 0,
            "filler_count": 0,
            "pause_ratio": 0,
            "analysis": "Insufficient speech data",
        }

    words = transcribed_text.split()
    word_count = len(words)

    # Speaking pace analysis
    duration_seconds = 0
    if segments and len(segments) > 0:
        duration_seconds = segments[-1]["end"] - segments[0]["start"]

    wpm = (word_count / max(duration_seconds, 1)) * 60 if duration_seconds > 0 else 0

    # Filler word detection (indicates human speech)
    filler_patterns = [
        r'\bum+\b', r'\buh+\b', r'\blike\b', r'\byou know\b',
        r'\bi mean\b', r'\bso+\b', r'\bwell\b', r'\bactually\b',
        r'\bbasically\b', r'\bhonestly\b', r'\bkind of\b',
        r'\bsort of\b', r'\bright\b', r'\bokay\b',
    ]
    filler_count = sum(
        len(re.findall(pattern, transcribed_text.lower()))
        for pattern in filler_patterns
    )

    # Pause analysis from segments
    pause_ratio = 0
    long_pauses = 0
    if segments and len(segments) > 1:
        total_pause = 0
        for i in range(1, len(segments)):
            gap = segments[i]["start"] - segments[i-1]["end"]
            total_pause += max(0, gap)
            if gap > 2.0:
                long_pauses += 1
        pause_ratio = total_pause / max(duration_seconds, 1)

    # Human speech indicators
    human_score = 0
    signals = []

    # Fillers are strong human signal
    if filler_count >= 3:
        human_score += 0.3
        signals.append(f"Natural filler words detected ({filler_count})")
    elif filler_count >= 1:
        human_score += 0.15

    # Natural pace (120-180 WPM for human speech)
    if 100 < wpm < 200:
        human_score += 0.2
        signals.append("Natural speaking pace")
    elif wpm > 200:
        human_score += 0.05  # Very fast could be reading
        signals.append("Very fast pace - possibly reading")

    # Pauses are natural
    if 0.1 < pause_ratio < 0.4:
        human_score += 0.2
        signals.append("Natural pause patterns")
    elif pause_ratio > 0.4:
        human_score += 0.1
        signals.append("Long pauses - possibly thinking or looking up answers")

    # Sentence variations
    sentences = re.split(r'[.!?]+', transcribed_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    if len(sentences) >= 2:
        lengths = [len(s.split()) for s in sentences]
        if max(lengths) - min(lengths) > 5:
            human_score += 0.15
            signals.append("Varied sentence structure")

    # Self-corrections (human trait)
    corrections = re.findall(r'\b(I mean|sorry|actually|wait|no)\b', transcribed_text.lower())
    if corrections:
        human_score += 0.15
        signals.append("Self-corrections detected")

    is_human = human_score >= 0.35
    confidence = min(1.0, human_score + 0.3)

    return {
        "is_human_likely": is_human,
        "confidence": round(confidence, 2),
        "human_score": round(human_score, 2),
        "speaking_pace_wpm": round(wpm, 0),
        "filler_count": filler_count,
        "pause_ratio": round(pause_ratio, 2),
        "long_pauses": long_pauses,
        "signals": signals,
        "analysis": "Human speech patterns detected" if is_human else "Possibly reading or AI-assisted",
    }
