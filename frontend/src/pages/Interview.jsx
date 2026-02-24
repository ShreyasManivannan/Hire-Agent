import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';

export default function Interview() {
    const navigate = useNavigate();
    const { sessionId } = useParams();
    const videoRef = useRef(null);
    const mcpLogRef = useRef(null);

    // Load config
    const configRef = useRef(null);
    if (!configRef.current) {
        try { configRef.current = JSON.parse(sessionStorage.getItem('interviewConfig') || '{}'); } catch { configRef.current = {}; }
    }
    const config = configRef.current;
    const useBackend = config.useBackend !== false;
    const topicsConfig = config.domains || ['python', 'dsa', 'system_design'];

    const [timerSeconds, setTimerSeconds] = useState((config.duration || 5) * 60);
    const totalSeconds = (config.duration || 5) * 60;
    const [questionNumber, setQuestionNumber] = useState(0);
    const [topicIndex, setTopicIndex] = useState(0);
    const [currentTopic, setCurrentTopic] = useState(topicsConfig[0] || 'python');
    const [questionText, setQuestionText] = useState('Loading question from RAG...');
    const [displayedText, setDisplayedText] = useState('Loading question from RAG...');
    const [isTyping, setIsTyping] = useState(false);
    const [answer, setAnswer] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [questionDifficulty, setQuestionDifficulty] = useState(config.difficulty || 'medium');

    const [responses, setResponses] = useState([]);
    const [evalHistory, setEvalHistory] = useState([]);
    const [topicScores, setTopicScores] = useState(() => {
        const obj = {}; topicsConfig.forEach(t => obj[t] = []); return obj;
    });

    const [attention, setAttention] = useState(0);
    const [sentiment, setSentiment] = useState(50);
    const [confidence, setConfidence] = useState(0);
    const [aiProb, setAiProb] = useState(0);
    const [aiLabel, setAiLabel] = useState('Low');

    const [mcpLogs, setMcpLogs] = useState([]);
    const [videoStream, setVideoStream] = useState(null);
    const [audioEnabled, setAudioEnabled] = useState(false);

    const [toastVisible, setToastVisible] = useState(false);
    const [toastScore, setToastScore] = useState(0);
    const [toastDetail, setToastDetail] = useState('');

    const timerRef = useRef(null);
    const attentionRef = useRef(null);
    const endedRef = useRef(false);
    const currentQuestionRef = useRef(null);

    // Type question animation
    useEffect(() => {
        if (!questionText) return;
        setDisplayedText('');
        setIsTyping(true);
        let i = 0;
        const text = questionText;
        const iv = setInterval(() => {
            if (i < text.length) {
                setDisplayedText(prev => prev + text[i]);
                i++;
            } else {
                clearInterval(iv);
                setIsTyping(false);
            }
        }, 18);
        return () => clearInterval(iv);
    }, [questionText]);

    function addMCPLog(msg, status = 'ok') {
        setMcpLogs(prev => [...prev, { msg, status }]);
        setTimeout(() => {
            if (mcpLogRef.current) mcpLogRef.current.scrollTop = mcpLogRef.current.scrollHeight;
        }, 50);
    }

    function showToast(score, detail) {
        setToastScore(score);
        setToastDetail(detail);
        setToastVisible(true);
        setTimeout(() => setToastVisible(false), 3500);
    }

    // ── Start session & fetch first question from backend ──
    useEffect(() => {
        async function init() {
            if (useBackend) {
                addMCPLog('interview_controller → starting session...', 'pending');
                try {
                    const res = await axios.post(`/api/start/${sessionId}`);
                    if (res.data.success && res.data.question) {
                        const q = res.data.question;
                        currentQuestionRef.current = q;
                        setQuestionNumber(1);
                        setCurrentTopic(q.domain || topicsConfig[0]);
                        setQuestionText(q.question || q.text);
                        setQuestionDifficulty(q.difficulty || config.difficulty);
                        addMCPLog('question_generator → RAG retrieved (' + (q.rag_retrieved_count || '?') + ' hits)', 'ok');
                    }
                } catch (err) {
                    addMCPLog('Backend start failed — using fallback', 'ok');
                    // Fallback first question
                    setQuestionNumber(1);
                    setQuestionText('Explain a concept you have deep experience with and how you applied it in a real project.');
                }
            } else {
                setQuestionNumber(1);
                setQuestionText('Explain a concept you have deep experience with and how you applied it in a real project.');
                addMCPLog('question_generator → client-side mode', 'ok');
            }
        }
        init();
    }, []);

    // Timer
    useEffect(() => {
        timerRef.current = setInterval(() => {
            setTimerSeconds(prev => {
                if (prev <= 1) { clearInterval(timerRef.current); endInterview(); return 0; }
                return prev - 1;
            });
        }, 1000);
        return () => clearInterval(timerRef.current);
    }, []);

    // Attention simulation
    useEffect(() => {
        attentionRef.current = setInterval(() => {
            setAttention(Math.round((0.55 + Math.random() * 0.45) * 100));
        }, 3000);
        return () => clearInterval(attentionRef.current);
    }, []);

    // Start video
    useEffect(() => {
        if (config.enableVideo) toggleVideo();
        return () => { if (videoStream) videoStream.getTracks().forEach(t => t.stop()); };
    }, []);

    async function toggleVideo() {
        if (videoStream) {
            videoStream.getTracks().forEach(t => t.stop());
            setVideoStream(null);
            if (videoRef.current) videoRef.current.srcObject = null;
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            setVideoStream(stream);
            if (videoRef.current) videoRef.current.srcObject = stream;
        } catch (e) { console.warn('Camera denied'); }
    }

    function toggleAudio() { setAudioEnabled(prev => !prev); }

    // ── Submit answer to backend ──
    async function submitAnswer() {
        if (!answer.trim() || answer.trim().length < 5 || submitting) return;
        setSubmitting(true);
        addMCPLog('answer_evaluator → sending to LLM...', 'pending');
        addMCPLog('ai_detector → analyzing...', 'pending');

        if (useBackend) {
            try {
                const res = await axios.post('/api/submit-answer', {
                    session_id: sessionId,
                    answer_text: answer,
                });

                if (res.data.success) {
                    const analysis = res.data.analysis || {};
                    const score = analysis.score ?? analysis.evaluation?.score ?? 0.5;
                    const ai = analysis.ai_probability ?? analysis.ai_detection?.probability ?? 0.2;
                    const sent = analysis.sentiment ?? 0.5;
                    const feedback = analysis.feedback || analysis.evaluation?.feedback || '';

                    handleResult(score, ai, sent, feedback);

                    // Next question from backend
                    if (res.data.interview_complete) {
                        // Store report and navigate
                        if (res.data.report) {
                            sessionStorage.setItem('interviewReport', JSON.stringify(res.data.report));
                        }
                        endInterview();
                    } else if (res.data.next_question) {
                        const nq = res.data.next_question;
                        currentQuestionRef.current = nq;
                        setQuestionNumber(prev => prev + 1);
                        setCurrentTopic(nq.domain || currentTopic);
                        setQuestionText(nq.question || nq.text);
                        setQuestionDifficulty(nq.difficulty || questionDifficulty);
                        addMCPLog('question_generator → RAG: ' + (nq.domain || 'next'), 'ok');
                    }
                }
            } catch (err) {
                console.error('Submit failed:', err);
                // Fallback: client-side scoring
                handleResult(0.5 + Math.random() * 0.3, 0.2 + Math.random() * 0.3, 0.5, 'Answer received (backend unavailable)');
                setQuestionNumber(prev => prev + 1);
                setQuestionText('Tell me about a challenging technical problem you solved recently.');
            }
        } else {
            // Client-side fallback scoring
            const score = Math.min(1, Math.max(0.1, answer.split(/\s+/).length / 150 + Math.random() * 0.2));
            const ai = Math.random() * 0.4;
            handleResult(score, ai, 0.5, score > 0.6 ? 'Good depth' : 'Could use more detail');
            setQuestionNumber(prev => prev + 1);
            setQuestionText('Tell me about a challenging technical problem you solved recently.');
        }

        setAnswer('');
        setSubmitting(false);
    }

    function handleResult(score, ai, sent, feedback) {
        const pctScore = Math.round(score * 100);
        const pctAi = Math.round(ai * 100);

        setResponses(prev => [...prev, {
            question: questionText, answer, topic: currentTopic,
            score, aiProb: ai, sentiment: sent, timestamp: Date.now(),
        }]);
        setTopicScores(prev => {
            const copy = { ...prev };
            copy[currentTopic] = [...(copy[currentTopic] || []), score];
            return copy;
        });
        setConfidence(pctScore);
        setSentiment(Math.round(sent * 100));
        setAiProb(pctAi);
        setAiLabel(ai > 0.6 ? 'HIGH' : ai > 0.4 ? 'MED' : 'LOW');

        setEvalHistory(prev => [{
            topic: currentTopic, num: questionNumber, q: questionText,
            a: answer, score: pctScore, ai: pctAi,
        }, ...prev]);

        const detail = ai > 0.6 ? '⚠️ High AI probability detected' :
            score > 0.7 ? '✓ Strong answer! Good depth.' :
                score > 0.45 ? '↗ Decent answer. More detail would help.' :
                    '↓ Answer needs more depth and specifics.';
        showToast(score, detail);
        addMCPLog(`answer_evaluator → score: ${pctScore}%`, 'ok');
        addMCPLog(`ai_detector → ${pctAi}% AI prob`, 'ok');
    }

    function skipQuestion() {
        if (answer.trim()) { submitAnswer(); return; }
        // Just request next question
        setQuestionNumber(prev => prev + 1);
        setQuestionText('Explain a concept from your domain that you find most interesting.');
        addMCPLog('question_generator → skipped', 'ok');
    }

    async function endInterview() {
        if (endedRef.current) return;
        endedRef.current = true;
        clearInterval(timerRef.current);
        clearInterval(attentionRef.current);
        if (videoStream) videoStream.getTracks().forEach(t => t.stop());

        addMCPLog('report_generator → generating...', 'pending');

        if (useBackend) {
            try {
                const res = await axios.post(`/api/stop-interview/${sessionId}`);
                if (res.data.report) {
                    sessionStorage.setItem('interviewReport', JSON.stringify(res.data.report));
                }
            } catch (err) {
                console.error('Stop failed, using local data');
            }
        }

        // Always store local results as fallback
        sessionStorage.setItem('interviewResults', JSON.stringify({
            responses, topicScores,
            candidateName: config.candidateName || 'Candidate',
            sessionId, startTime: config.startTime, domains: topicsConfig,
        }));

        addMCPLog('report_generator → complete', 'ok');
        navigate('/report/' + sessionId);
    }

    // Timer display
    const mins = Math.floor(timerSeconds / 60);
    const secs = timerSeconds % 60;
    const timerStr = mins + ':' + secs.toString().padStart(2, '0');
    const timerPct = timerSeconds / totalSeconds;
    const circumference = 2 * Math.PI * 35;
    const timerOffset = circumference * (1 - timerPct);
    const timerColor = timerPct > 0.5 ? 'var(--accent)' : timerPct > 0.25 ? '#ffa500' : 'var(--accent2)';
    const wordCount = answer.trim() ? answer.trim().split(/\s+/).length : 0;

    return (
        <div className="page page-interview">
            <div className="interview-layout">
                {/* Main column */}
                <div>
                    {/* Status bar */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20, flexWrap: 'wrap', gap: 12 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                            <div className="timer-ring-wrap">
                                <svg className="ring" viewBox="0 0 80 80">
                                    <circle className="ring-bg" cx="40" cy="40" r="35" />
                                    <circle className="ring-fill" cx="40" cy="40" r="35" style={{ strokeDashoffset: timerOffset, stroke: timerColor }} />
                                </svg>
                                <div className="timer-text">{timerStr}</div>
                            </div>
                            <div>
                                <div style={{ fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: '1.1rem' }}>
                                    {config.candidateName || 'Candidate'}
                                </div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text2)' }}>{sessionId}</div>
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: 10 }}>
                            <button className="btn btn-secondary btn-sm" onClick={skipQuestion}>Skip ⤳</button>
                            <button className="btn btn-danger btn-sm" onClick={endInterview}>End Interview</button>
                        </div>
                    </div>

                    {/* Question card */}
                    <div className="q-card">
                        <div className="q-header">
                            <div className="q-meta">
                                <span className="q-num">Q{questionNumber}</span>
                                <span className={`topic-badge ${currentTopic.replace(' ', '_')}`}>{currentTopic.replace('_', ' ')}</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                {submitting && <span className="spinner" />}
                                <span style={{ fontSize: '0.75rem', color: 'var(--text2)', fontFamily: "'DM Mono',monospace" }}>{questionDifficulty}</span>
                            </div>
                        </div>
                        <div className="q-body">
                            <div className={`q-text${isTyping ? ' typing' : ''}`}>{displayedText || 'Loading question from RAG pipeline...'}</div>
                            <div className="answer-area">
                                <textarea
                                    value={answer}
                                    onChange={e => setAnswer(e.target.value)}
                                    placeholder="Type your answer here... Be detailed and explain your reasoning."
                                    onKeyDown={e => { if (e.ctrlKey && e.key === 'Enter') submitAnswer(); }}
                                />
                                <div className="answer-meta">
                                    <div className="word-count">{wordCount} word{wordCount !== 1 ? 's' : ''}</div>
                                    <div className="answer-actions">
                                        <button className="btn btn-secondary btn-sm" onClick={() => setAnswer('')}>Clear</button>
                                        <button className="btn btn-success btn-sm" onClick={submitAnswer} disabled={submitting}>
                                            {submitting ? <span className="spinner" /> : 'Submit ↵'}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Eval history */}
                    <div style={{ marginTop: 20 }}>
                        {evalHistory.map((ev, i) => (
                            <div className="eval-card" key={i} style={{ borderLeft: `3px solid ${ev.score > 70 ? 'var(--accent3)' : ev.score > 45 ? '#ffa500' : 'var(--accent2)'}` }}>
                                <div className="eval-topic">{ev.topic.toUpperCase()} — Q{ev.num}</div>
                                <div className="eval-q">{ev.q}</div>
                                <div className="eval-a">{ev.a.substring(0, 200)}{ev.a.length > 200 ? '...' : ''}</div>
                                <div className="eval-pills">
                                    <span className="eval-pill score">Score: {ev.score}%</span>
                                    <span className="eval-pill ai">AI: {ev.ai}%</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Sidebar */}
                <div className="sidebar">
                    <div className="sidebar-card">
                        <h4>📷 Live Monitor</h4>
                        <div className="video-wrap">
                            <video ref={videoRef} autoPlay muted playsInline />
                            <div className="video-overlay">
                                <span className="v-badge live">● REC</span>
                                <span className="v-badge">Attn: {attention}%</span>
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: 8 }}>
                            <button className="btn btn-secondary btn-sm" style={{ flex: 1 }} onClick={toggleVideo}>📷 {videoStream ? 'Disable' : 'Enable'}</button>
                            <button className="btn btn-secondary btn-sm" style={{ flex: 1 }} onClick={toggleAudio}>🎤 {audioEnabled ? 'Disable' : 'Enable'}</button>
                        </div>
                    </div>

                    <div className="sidebar-card">
                        <h4>📈 Real-time Metrics</h4>
                        <div className="metric-row">
                            <span className="metric-label">Attention</span>
                            <div className="metric-bar-wrap"><div className="metric-bar green" style={{ width: attention + '%' }} /></div>
                            <span className="metric-val">{attention}%</span>
                        </div>
                        <div className="metric-row">
                            <span className="metric-label">Sentiment</span>
                            <div className="metric-bar-wrap"><div className="metric-bar blue" style={{ width: sentiment + '%' }} /></div>
                            <span className="metric-val">{sentiment}%</span>
                        </div>
                        <div className="metric-row">
                            <span className="metric-label">Confidence</span>
                            <div className="metric-bar-wrap"><div className="metric-bar blue" style={{ width: confidence + '%' }} /></div>
                            <span className="metric-val">{confidence}%</span>
                        </div>
                        <div style={{ marginTop: 12 }}>
                            <div style={{ fontSize: '0.78rem', color: 'var(--text2)', marginBottom: 8 }}>AI Detection</div>
                            <div className="gauge-wrap">
                                <div className="gauge">
                                    <div className="gauge-fill" style={{ width: aiProb + '%', background: aiProb > 60 ? 'var(--accent2)' : aiProb > 40 ? '#ffa500' : 'var(--accent3)' }} />
                                </div>
                                <span className={`gauge-label ${aiProb > 60 ? 'danger' : aiProb > 40 ? 'warn' : 'safe'}`}>{aiLabel}</span>
                            </div>
                        </div>
                    </div>

                    <div className="sidebar-card">
                        <h4>📚 Topics</h4>
                        <div className="topic-list">
                            {topicsConfig.map((t, i) => {
                                const scores = topicScores[t] || [];
                                const avg = scores.length ? Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 100) : null;
                                const dotClass = i < topicIndex ? 'done' : i === topicIndex ? 'active' : 'pending';
                                return (
                                    <div className="topic-item" key={t}>
                                        <div className={`topic-dot ${dotClass}`} />
                                        <span className="topic-name">{t.replace('_', ' ')}</span>
                                        <span className="topic-score">{avg !== null ? avg + '%' : '—'}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    <div className="sidebar-card">
                        <h4>⚙ MCP Tool Calls</h4>
                        <div className="mcp-call-log" ref={mcpLogRef}>
                            {mcpLogs.map((l, i) => (
                                <div className="mcp-log-item" key={i}>
                                    <div className={`mcp-log-dot ${l.status}`} />
                                    {l.msg}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Toast */}
            <div className={`feedback-toast${toastVisible ? ' show' : ''}`}>
                <div className={`toast-score ${toastScore > 0.7 ? 'high' : toastScore > 0.45 ? 'mid' : 'low'}`}>
                    {Math.round(toastScore * 100)}%
                </div>
                <div className="toast-label">Answer Score</div>
                <div className="toast-detail">{toastDetail}</div>
            </div>
        </div>
    );
}
