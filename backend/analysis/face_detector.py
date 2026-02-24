"""
Face Detector — OpenCV-based face detection, gaze tracking, and expression analysis.
Provides accurate proctoring scores for the live interview.
Uses Haar cascades with multi-scale detection and temporal smoothing.
"""

import logging
import base64
import math
import numpy as np
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)

# Lazy-loaded cascade classifiers
_face_cascade = None
_eye_cascade = None
_smile_cascade = None
_profile_cascade = None

# Temporal state for smoothing scores across frames
_history = deque(maxlen=30)  # last 30 frames (~3 seconds at 10 fps)


def _load_cascades():
    """Load Haar cascade classifiers for face, eye, smile, and profile detection."""
    global _face_cascade, _eye_cascade, _smile_cascade, _profile_cascade
    try:
        import cv2
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        _eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        _smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        _profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        logger.info("Face detection cascades loaded (face, eye, smile, profile)")
    except Exception as e:
        logger.error(f"Failed to load cascades: {e}")


def analyze_frame(frame_data: str) -> dict:
    """
    Analyze a video frame for proctoring.

    Computes:
      - face_detected, face_count
      - gaze_direction (center / looking_left / looking_right / looking_up / looking_down)
      - attention_score (0-100): how focused the candidate is
      - expression_label (neutral / focused / engaged / distracted)
      - suspicion_level (low / medium / high) with explanatory reason
      - proctoring_score (0-100): overall integrity metric

    Args:
        frame_data: Base64-encoded image data (JPEG/PNG)
    """
    global _face_cascade, _eye_cascade

    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed")
        return _empty_result("OpenCV not available")

    try:
        if _face_cascade is None:
            _load_cascades()

        # --- Decode image ---
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return _empty_result("Failed to decode image")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Histogram equalisation for better detection under varying lighting
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape

        # --- Face detection (multi-scale) ---
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6,
            minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            # Try profile face as fallback
            if _profile_cascade is not None:
                faces = _profile_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60),
                )
            if len(faces) == 0:
                result = {
                    "face_detected": False,
                    "face_count": 0,
                    "gaze_direction": "not_detected",
                    "expression_label": "unknown",
                    "attention_score": 0,
                    "suspicion_level": "high",
                    "suspicion_reason": "No face detected — candidate may be absent or looking away",
                    "face_size_ratio": 0,
                    "eyes_detected": 0,
                    "proctoring_score": 15,
                }
                _history.append(result)
                return _smooth(result)

        # --- Multiple faces ---
        if len(faces) > 1:
            result = {
                "face_detected": True,
                "face_count": len(faces),
                "gaze_direction": "multiple_faces",
                "expression_label": "unknown",
                "attention_score": 20,
                "suspicion_level": "high",
                "suspicion_reason": f"Multiple faces detected ({len(faces)}) — possible external help",
                "face_size_ratio": 0,
                "eyes_detected": 0,
                "proctoring_score": 10,
            }
            _history.append(result)
            return _smooth(result)

        # --- Single face analysis ---
        x, y, fw, fh = faces[0]
        face_roi_gray = gray[y : y + fh, x : x + fw]
        face_roi_color = frame[y : y + fh, x : x + fw]
        face_size_ratio = (fw * fh) / (w * h)

        # Eye detection inside face ROI
        eyes = _eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(20, 20),
        )

        # --- Gaze estimation ---
        face_cx = x + fw // 2
        face_cy = y + fh // 2
        norm_x = face_cx / w  # 0‥1
        norm_y = face_cy / h

        if norm_x < 0.30:
            gaze = "looking_left"
        elif norm_x > 0.70:
            gaze = "looking_right"
        elif norm_y < 0.25:
            gaze = "looking_up"
        elif norm_y > 0.75:
            gaze = "looking_down"
        else:
            gaze = "center"

        # Refine gaze with eye positions if two eyes found
        if len(eyes) >= 2:
            eys = sorted(eyes, key=lambda e: e[0])
            left_eye_cx = eys[0][0] + eys[0][2] // 2
            right_eye_cx = eys[1][0] + eys[1][2] // 2
            eye_mid = (left_eye_cx + right_eye_cx) / 2
            eye_ratio = eye_mid / fw
            if eye_ratio < 0.35:
                gaze = "looking_left"
            elif eye_ratio > 0.65:
                gaze = "looking_right"
            else:
                gaze = "center"

        # --- Expression estimation ---
        expression = _estimate_expression(face_roi_gray, face_roi_color)

        # --- Attention score (0‥100) ---
        attention = _compute_attention(gaze, len(eyes), face_size_ratio, expression)

        # --- Suspicion ---
        suspicion, suspicion_reason = _compute_suspicion(
            gaze, len(eyes), face_size_ratio, attention
        )

        # --- Proctoring score (0‥100) ---
        proctoring = _compute_proctoring_score(
            attention, suspicion, face_size_ratio, gaze, len(eyes),
        )

        result = {
            "face_detected": True,
            "face_count": 1,
            "gaze_direction": gaze,
            "expression_label": expression,
            "attention_score": attention,
            "suspicion_level": suspicion,
            "suspicion_reason": suspicion_reason,
            "face_size_ratio": round(face_size_ratio, 4),
            "eyes_detected": len(eyes),
            "proctoring_score": proctoring,
        }
        _history.append(result)
        return _smooth(result)

    except Exception as e:
        logger.error(f"Frame analysis failed: {e}")
        return _empty_result(str(e))


# ---------- Scoring helpers ----------

def _compute_attention(gaze: str, eyes_count: int, face_ratio: float, expression: str) -> int:
    """Compute 0-100 attention score."""
    score = 50  # baseline

    # Gaze
    if gaze == "center":
        score += 30
    elif gaze in ("looking_left", "looking_right"):
        score -= 15
    elif gaze in ("looking_up", "looking_down"):
        score -= 10

    # Eyes visible
    if eyes_count >= 2:
        score += 15
    elif eyes_count == 1:
        score += 5
    else:
        score -= 10

    # Face distance
    if face_ratio > 0.05:
        score += 5
    elif face_ratio < 0.02:
        score -= 10

    # Expression bonus
    if expression in ("focused", "engaged"):
        score += 5

    return max(0, min(100, score))


def _compute_suspicion(gaze, eyes_count, face_ratio, attention):
    """Determine suspicion level and reason."""
    if attention >= 70:
        return "low", "Candidate appears attentive"
    elif attention >= 45:
        reasons = []
        if gaze in ("looking_left", "looking_right"):
            reasons.append(f"Candidate {gaze.replace('_', ' ')}")
        if eyes_count == 0:
            reasons.append("Eyes not clearly visible")
        if face_ratio < 0.02:
            reasons.append("Face too far from camera")
        return "medium", "; ".join(reasons) if reasons else "Moderate attention"
    else:
        return "high", "Low attention — candidate may be distracted or looking elsewhere"


def _compute_proctoring_score(attention, suspicion, face_ratio, gaze, eyes_count) -> int:
    """
    Compute an overall proctoring integrity score (0-100).
    Combines attention, gaze, and historical consistency.
    """
    base = attention

    # Bonus for good posture
    if gaze == "center" and eyes_count >= 2 and face_ratio >= 0.04:
        base += 10

    # Penalty for suspicious signals
    if suspicion == "high":
        base -= 20
    elif suspicion == "medium":
        base -= 5

    return max(0, min(100, base))


def _estimate_expression(face_gray, face_color) -> str:
    """Improved expression estimation with smile detection and region analysis."""
    try:
        import cv2

        fh, fw = face_gray.shape

        # Try smile detection
        if _smile_cascade is not None:
            mouth_region = face_gray[int(fh * 0.55) :, :]
            smiles = _smile_cascade.detectMultiScale(
                mouth_region, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25)
            )
            if len(smiles) > 0:
                return "engaged"

        # Forehead / brow region variance (thinking / focused)
        brow_region = face_gray[int(fh * 0.15) : int(fh * 0.35), int(fw * 0.2) : int(fw * 0.8)]
        brow_std = float(np.std(brow_region))

        # Mouth region variance
        mouth_region = face_gray[int(fh * 0.6) : int(fh * 0.9), int(fw * 0.25) : int(fw * 0.75)]
        mouth_std = float(np.std(mouth_region))

        # Laplacian for overall texture / sharpness (correlates with expression)
        laplacian_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())

        if mouth_std > 55:
            return "engaged"
        elif brow_std > 48 or laplacian_var > 500:
            return "focused"
        else:
            return "neutral"

    except Exception:
        return "neutral"


def _smooth(current: dict) -> dict:
    """Smooth numeric scores using exponential moving average over history."""
    if len(_history) < 2:
        return current

    alpha = 0.4  # weight of the current frame
    for key in ("attention_score", "proctoring_score"):
        past_vals = [h.get(key, current.get(key, 50)) for h in _history]
        avg = sum(past_vals) / len(past_vals)
        current[key] = int(alpha * current[key] + (1 - alpha) * avg)

    return current


def _empty_result(reason: str = "Unknown") -> dict:
    """Return empty face analysis result."""
    return {
        "face_detected": False,
        "face_count": 0,
        "gaze_direction": "not_detected",
        "expression_label": "unknown",
        "attention_score": 0,
        "suspicion_level": "unknown",
        "suspicion_reason": reason,
        "face_size_ratio": 0,
        "eyes_detected": 0,
        "proctoring_score": 0,
    }
