"""
HireAgent Backend — FastAPI application with REST + WebSocket endpoints.
Orchestrates the multi-agent interview system.
"""

import os
import sys
import logging
import asyncio
import json
import base64
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="HireAgent - AI Interview System",
    description="Multi-agent RAG-based AI interview platform with adaptive questioning, face monitoring, and AI detection.",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ────────────────────────────────────────────────

class InterviewConfig(BaseModel):
    duration_minutes: int = 10
    topics: Optional[list] = None
    difficulty: str = "adaptive"


class AnswerSubmission(BaseModel):
    session_id: str
    answer_text: str
    audio_data: Optional[str] = None


class FrameAnalysis(BaseModel):
    session_id: str
    frame_data: str  # Base64-encoded image


# ─── Startup ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Initialize RAG knowledge base and MCP server on startup."""
    logger.info("Starting HireAgent backend...")

    # Initialize knowledge base
    try:
        from rag.knowledge_base import init_knowledge_base
        init_knowledge_base()
        logger.info("Knowledge base initialized")
    except Exception as e:
        logger.warning(f"Knowledge base init failed: {e}")

    # Initialize MCP server
    try:
        from mcp.server import get_mcp_server
        mcp = get_mcp_server()
        logger.info(f"MCP Server initialized with {len(mcp.list_tools())} tools")
    except Exception as e:
        logger.warning(f"MCP server init failed: {e}")

    logger.info("HireAgent backend ready!")


# ─── API Endpoints ───────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "HireAgent API is running", "version": "1.0.0"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "hire-agent"}


@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a PDF resume and extract structured candidate profile.
    Uses Resume Analyzer Agent with PyMuPDF extraction.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        pdf_bytes = await file.read()

        from agents.resume_analyzer import analyze_resume
        profile = analyze_resume(pdf_bytes)

        if profile.get("error"):
            raise HTTPException(status_code=422, detail=profile["error"])

        return {
            "success": True,
            "profile": profile,
            "message": f"Resume parsed: {len(profile.get('skills', []))} skills detected"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start-interview")
async def start_interview(
    config: InterviewConfig,
    session_id: Optional[str] = None,
):
    """
    Start a new interview session.
    Requires resume to be uploaded first (pass profile via session or re-upload).
    """
    from services.interview_session import get_session

    if session_id:
        session = get_session(session_id)
        if session:
            session.start()
            first_question = session.get_next_question()
            return {
                "success": True,
                "session_id": session.session_id,
                "first_question": first_question,
                "state": session.get_state(),
            }

    raise HTTPException(
        status_code=400,
        detail="No active session. Upload resume first."
    )


@app.post("/api/create-session")
async def create_session(config: InterviewConfig):
    """
    Create a session from a previously uploaded profile.
    This is called after resume upload with the config settings.
    """
    # For demo: create session with sample profile if no real upload
    profile = getattr(app.state, '_last_profile', None)
    if not profile:
        profile = {
            "name": "Demo Candidate",
            "skills": ["python", "javascript", "react"],
            "domains": ["programming", "frontend"],
            "experience_years": 2,
            "education": [],
            "projects": [],
            "skill_graph": {"programming": ["python", "javascript"], "frontend": ["react"]},
        }

    from services.interview_session import InterviewSession

    session = InterviewSession(
        candidate_profile=profile,
        duration_minutes=config.duration_minutes,
        custom_topics=config.topics,
        difficulty=config.difficulty,
    )

    return {
        "success": True,
        "session_id": session.session_id,
        "plan": {
            "total_questions": session.plan["total_questions"],
            "topics": [
                {"topic": t["topic"], "display_name": t["display_name"], "questions": t["num_questions"]}
                for t in session.plan["topics"]
            ],
            "duration_minutes": config.duration_minutes,
        },
    }


@app.post("/api/upload-and-create")
async def upload_and_create(
    file: UploadFile = File(...),
    duration_minutes: int = 10,
    difficulty: str = "adaptive",
):
    """Combined endpoint: upload resume + create session in one call."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        pdf_bytes = await file.read()

        from agents.resume_analyzer import analyze_resume
        profile = analyze_resume(pdf_bytes)

        if profile.get("error"):
            raise HTTPException(status_code=422, detail=profile["error"])

        # Store for later use
        app.state._last_profile = profile

        from services.interview_session import InterviewSession

        session = InterviewSession(
            candidate_profile=profile,
            duration_minutes=duration_minutes,
            difficulty=difficulty,
        )

        return {
            "success": True,
            "session_id": session.session_id,
            "profile": profile,
            "plan": {
                "total_questions": session.plan["total_questions"],
                "topics": [
                    {"topic": t["topic"], "display_name": t["display_name"], "questions": t["num_questions"]}
                    for t in session.plan["topics"]
                ],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and create failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start/{session_id}")
async def start_session(session_id: str):
    """Start the interview and get the first question."""
    from services.interview_session import get_session

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.start()
    first_question = session.get_next_question()

    return {
        "success": True,
        "question": first_question,
        "state": session.get_state(),
    }


@app.post("/api/submit-answer")
async def submit_answer(submission: AnswerSubmission):
    """
    Submit an answer and get analysis + next question.
    Handles adaptive topic skipping if candidate doesn't know.
    """
    from services.interview_session import get_session

    session = get_session(submission.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "active":
        raise HTTPException(status_code=400, detail=f"Session is {session.status}")

    # Analyze the answer
    result = session.submit_answer(
        answer_text=submission.answer_text,
        audio_data=submission.audio_data,
    )

    # Check if interview should end
    if session.is_time_up() or session.is_complete():
        report = session.stop()
        return {
            "success": True,
            "analysis": result["analysis"],
            "interview_complete": True,
            "report": report,
            "state": session.get_state(),
        }

    # Get next question
    next_question = session.get_next_question()

    return {
        "success": True,
        "analysis": result,
        "next_question": next_question,
        "interview_complete": next_question is None,
        "state": session.get_state(),
    }


@app.post("/api/analyze-frame")
async def analyze_frame(data: FrameAnalysis):
    """Analyze a webcam frame for face detection and cheating indicators."""
    from services.interview_session import get_session
    from analysis.face_detector import analyze_frame as detect_face

    session = get_session(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = detect_face(data.frame_data)
    session.add_face_analysis(result)

    return {"success": True, "analysis": result}


@app.post("/api/stop-interview/{session_id}")
async def stop_interview(session_id: str):
    """Stop interview early and generate partial report."""
    from services.interview_session import get_session

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    report = session.stop()

    return {
        "success": True,
        "report": report,
        "state": session.get_state(),
    }


@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    """Get the interview report as JSON."""
    from services.interview_session import get_session

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not hasattr(session, 'report'):
        raise HTTPException(status_code=400, detail="Interview not yet completed")

    return {"success": True, "report": session.report}


@app.get("/api/report/{session_id}/pdf")
async def download_report_pdf(session_id: str):
    """Download the interview report as PDF."""
    from services.interview_session import get_session
    from services.pdf_report import generate_pdf_report

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not hasattr(session, 'report'):
        raise HTTPException(status_code=400, detail="Interview not yet completed")

    pdf_path = generate_pdf_report(session.report)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="PDF generation failed")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(pdf_path),
    )


@app.get("/api/session/{session_id}")
async def get_session_state(session_id: str):
    """Get current session state."""
    from services.interview_session import get_session

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"success": True, "state": session.get_state()}


@app.get("/api/mcp/tools")
async def list_mcp_tools():
    """List all registered MCP tools."""
    from mcp.server import get_mcp_server
    mcp = get_mcp_server()
    return {"tools": mcp.list_tools()}


# ─── WebSocket for Real-Time Interview ───────────────────────────────

@app.websocket("/ws/interview/{session_id}")
async def websocket_interview(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time interview communication.
    Handles: questions, answers, face frames, timer updates.
    """
    from services.interview_session import get_session

    session = get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "get_question":
                question = session.get_next_question()
                await websocket.send_json({
                    "type": "question",
                    "data": question,
                    "state": session.get_state(),
                })

            elif msg_type == "submit_answer":
                result = session.submit_answer(
                    answer_text=data.get("answer", ""),
                    audio_data=data.get("audio"),
                )

                if session.is_time_up() or session.is_complete():
                    report = session.stop()
                    await websocket.send_json({
                        "type": "interview_complete",
                        "report": report,
                        "state": session.get_state(),
                    })
                    break
                else:
                    next_q = session.get_next_question()
                    await websocket.send_json({
                        "type": "answer_result",
                        "analysis": result,
                        "next_question": next_q,
                        "state": session.get_state(),
                    })

            elif msg_type == "analyze_frame":
                from analysis.face_detector import analyze_frame as detect_face
                result = detect_face(data.get("frame", ""))
                session.add_face_analysis(result)
                await websocket.send_json({
                    "type": "face_analysis",
                    "data": result,
                })

            elif msg_type == "stop_interview":
                report = session.stop()
                await websocket.send_json({
                    "type": "interview_complete",
                    "report": report,
                    "state": session.get_state(),
                })
                break

            elif msg_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "state": session.get_state(),
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        if session.status == "active":
            session.stop()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if session.status == "active":
            session.stop()


# ─── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
