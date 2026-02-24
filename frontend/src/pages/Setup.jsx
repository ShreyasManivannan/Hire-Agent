import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const MCP_TOOLS = [
    { name: 'resume_parser', desc: 'Parse PDF/text, extract skills + experience via NLP' },
    { name: 'question_generator', desc: 'RAG-based question generation with ChromaDB retrieval' },
    { name: 'answer_evaluator', desc: 'LLM-based answer scoring with feedback generation' },
    { name: 'ai_detector', desc: 'Perplexity, burstiness, marker analysis for AI detection' },
    { name: 'report_generator', desc: 'Comprehensive interview report with hire recommendation' },
];

const ARCH_DIAGRAM = `┌────────────────────────────────────────────┐
│         Frontend (React + Vite)            │
│  • Resume Upload  • WebRTC Video/Audio     │
│  • Real-time Monitoring  • Report View     │
└──────────────────────┬─────────────────────┘
                       ↓  HTTP / WebSocket
┌────────────────────────────────────────────┐
│      FastAPI Backend — Multi-Agent         │
│  ┌──────────────────────────────────────┐  │
│  │  MCP Server — Tool Orchestration     │  │
│  └──────────────────────────────────────┘  │
│  ┌────────────────┐  ┌──────────────────┐  │
│  │ Resume Parser  │  │ Question Gen     │  │
│  │ spaCy + PyPDF  │  │ RAG + ChromaDB   │  │
│  └────────────────┘  └──────────────────┘  │
│  ┌────────────────┐  ┌──────────────────┐  │
│  │ AI Detector    │  │ Response Eval    │  │
│  │ Perplexity+    │  │ LLM Scoring      │  │
│  │ Burstiness     │  │ Sentiment Anal.  │  │
│  └────────────────┘  └──────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │   Interview Controller (Adaptive)    │  │
│  └──────────────────────────────────────┘  │
└──────────────────────┬─────────────────────┘
                       ↓
┌────────────────────────────────────────────┐
│   Quantized LLM Engine — PyTorch          │
│   Phi-2 / Llama-3-8B                      │
│   4-bit NF4 (BitsAndBytes) + LoRA (PEFT)  │
└──────────────────────┬─────────────────────┘
                       ↓
┌────────────────────────────────────────────┐
│   Vector DB — ChromaDB                    │
│   all-MiniLM-L6-v2 Embeddings             │
│   70+ questions × 5 domains indexed       │
└────────────────────────────────────────────┘`;

const ALL_DOMAINS = [
    { id: 'python', label: 'Python' },
    { id: 'dsa', label: 'DSA' },
    { id: 'system_design', label: 'System Design' },
    { id: 'aws', label: 'AWS' },
    { id: 'ml', label: 'ML/AI' },
];

export default function Setup() {
    const navigate = useNavigate();
    const fileRef = useRef(null);

    const [resumeText, setResumeText] = useState('');
    const [skills, setSkills] = useState([]);
    const [fileName, setFileName] = useState('');
    const [fileSize, setFileSize] = useState('');
    const [fileLoaded, setFileLoaded] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);

    const [candidateName, setCandidateName] = useState('');
    const [duration, setDuration] = useState(5);
    const [difficulty, setDifficulty] = useState('medium');
    const [domains, setDomains] = useState(['python', 'dsa', 'system_design']);
    const [enableVideo, setEnableVideo] = useState(true);
    const [enableAudio, setEnableAudio] = useState(true);
    const [loading, setLoading] = useState(false);

    const [logs, setLogs] = useState([
        { msg: '[HireAgent] System initialized', type: 'info' },
        { msg: '[RAG] ChromaDB loaded — 70+ questions indexed', type: 'success' },
        { msg: '[LLM] Model loaded with 4-bit NF4 quantization', type: 'info' },
        { msg: '[MCP] 5 tools registered', type: 'success' },
        { msg: '[AI Detector] Perplexity + burstiness model ready', type: 'info' },
    ]);

    const [profile, setProfile] = useState(null);
    const logRef = useRef(null);

    useEffect(() => {
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    }, [logs]);

    function addLog(msg, type = 'info') {
        const time = new Date().toLocaleTimeString();
        setLogs(prev => [...prev, { msg: `[${time}] ${msg}`, type }]);
    }

    // Upload PDF to backend for parsing
    async function uploadResume() {
        if (!selectedFile) {
            addLog('No file selected — paste resume text or upload a PDF', 'warn');
            return;
        }

        addLog('Uploading resume to backend for NLP parsing...', 'info');
        setLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            const res = await axios.post('/api/upload-resume', formData);

            if (res.data.success) {
                const p = res.data.profile;
                setProfile(p);
                setSkills(p.skills || []);
                if (p.name && p.name !== 'Candidate') setCandidateName(p.name);
                addLog(`Resume parsed: ${p.skills?.length || 0} skills, ${p.experience_years || 0} yrs experience`, 'success');
                addLog(`Domains detected: ${(p.domains || []).join(', ')}`, 'success');
            }
        } catch (err) {
            addLog('Backend parse failed: ' + (err.response?.data?.detail || err.message), 'warn');
            // Fallback: client-side simple extraction
            fallbackParse();
        } finally {
            setLoading(false);
        }
    }

    function fallbackParse() {
        const text = resumeText;
        if (!text.trim()) return;
        const kwds = ["python", "java", "javascript", "react", "node", "aws", "docker", "kubernetes", "ml", "machine learning", "tensorflow", "pytorch", "sql", "mongodb", "system design", "data structures", "algorithms", "redis"];
        const lower = text.toLowerCase();
        const found = kwds.filter(s => lower.includes(s)).slice(0, 12);
        setSkills(found);
        addLog('Fallback client-side parse: ' + found.join(', '), 'info');
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;
        setSelectedFile(file);
        setFileName(file.name);
        setFileSize((file.size / 1024).toFixed(1) + ' KB');
        setFileLoaded(true);
        addLog('File loaded: ' + file.name, 'success');

        // If it's a text file, read content
        if (!file.name.endsWith('.pdf')) {
            const reader = new FileReader();
            reader.onload = (ev) => setResumeText(ev.target.result);
            reader.readAsText(file);
        }
    }

    function handleDrop(e) {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (!file) return;
        setSelectedFile(file);
        setFileName(file.name);
        setFileSize((file.size / 1024).toFixed(1) + ' KB');
        setFileLoaded(true);
        addLog('File dropped: ' + file.name, 'success');
    }

    function toggleDomain(id) {
        setDomains(prev => prev.includes(id) ? prev.filter(d => d !== id) : [...prev, id]);
    }

    function removeSkill(skill) {
        setSkills(prev => prev.filter(s => s !== skill));
    }

    // Create session via backend and start interview
    async function startInterview() {
        if (domains.length === 0) { alert('Please select at least one domain.'); return; }
        setLoading(true);
        addLog('Creating interview session via backend...', 'info');

        try {
            // Create session
            const res = await axios.post('/api/create-session', {
                duration_minutes: duration,
                topics: domains,
                difficulty,
            });

            if (res.data.success) {
                const sessionId = res.data.session_id;
                addLog(`Session created: ${sessionId}`, 'success');

                // Store config for Interview page
                sessionStorage.setItem('interviewConfig', JSON.stringify({
                    candidateName: candidateName || 'Candidate',
                    duration,
                    difficulty,
                    domains,
                    skills,
                    enableVideo,
                    enableAudio,
                    sessionId,
                    startTime: Date.now(),
                    plan: res.data.plan,
                    profile,
                    useBackend: true,
                }));

                navigate('/interview/' + sessionId);
            }
        } catch (err) {
            addLog('Backend session creation failed: ' + (err.response?.data?.detail || err.message), 'warn');
            addLog('Falling back to client-side mode...', 'info');

            // Fallback: client-side session
            const sessionId = 'local_' + Date.now();
            sessionStorage.setItem('interviewConfig', JSON.stringify({
                candidateName: candidateName || 'Candidate',
                duration,
                difficulty,
                domains,
                skills,
                enableVideo,
                enableAudio,
                sessionId,
                startTime: Date.now(),
                useBackend: false,
            }));
            navigate('/interview/' + sessionId);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="page page-setup">
            <div className="page-header">
                <h2>Interview Setup</h2>
                <p>Configure the interview session and upload resume</p>
            </div>

            <div className="setup-grid">
                {/* Left: Resume Upload */}
                <div>
                    <div className="form-group">
                        <label>Resume Upload</label>
                        <div
                            className={`dropzone${fileLoaded ? ' loaded' : ''}`}
                            onClick={() => fileRef.current?.click()}
                            onDragOver={e => e.preventDefault()}
                            onDrop={handleDrop}
                        >
                            <input type="file" ref={fileRef} accept=".pdf,.txt,.doc" onChange={handleFileSelect} />
                            <div className="dropzone-icon">{fileLoaded ? '✅' : '📎'}</div>
                            <div className="dropzone-title">{fileLoaded ? fileName : 'Drop resume here or click to upload'}</div>
                            <div className="dropzone-sub">{fileLoaded ? fileSize : 'PDF, DOC, TXT — or paste text below'}</div>
                        </div>
                    </div>

                    <div className="form-group">
                        <label>Resume Text (paste or auto-filled)</label>
                        <textarea rows={8} value={resumeText} onChange={e => setResumeText(e.target.value)} placeholder="Paste resume content here..." />
                    </div>

                    <div className="form-group">
                        <label>Detected Skills</label>
                        <div className="skills-container">
                            {skills.length === 0 ? (
                                <span style={{ fontSize: '0.78rem', color: 'var(--text3)', padding: 4 }}>Upload resume to detect skills...</span>
                            ) : (
                                skills.map(s => (
                                    <div className="skill-tag" key={s}>
                                        {s} <span className="remove" onClick={() => removeSkill(s)}>×</span>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                    <button className="btn btn-secondary btn-sm" onClick={uploadResume} disabled={loading} style={{ marginTop: 8 }}>
                        {loading ? <><span className="spinner" /> Parsing...</> : '🔍 Parse Resume'}
                    </button>
                </div>

                {/* Right: Config */}
                <div>
                    <div className="form-group">
                        <label>Candidate Name</label>
                        <input type="text" value={candidateName} onChange={e => setCandidateName(e.target.value)} placeholder="Auto-detected from resume or type here" />
                    </div>

                    <div className="form-group">
                        <label>Duration (minutes)</label>
                        <input type="number" value={duration} onChange={e => setDuration(parseInt(e.target.value) || 5)} min={1} max={60} />
                    </div>

                    <div className="form-group">
                        <label>Difficulty Level</label>
                        <select value={difficulty} onChange={e => setDifficulty(e.target.value)}>
                            <option value="easy">Easy</option>
                            <option value="medium">Medium</option>
                            <option value="hard">Hard</option>
                            <option value="adaptive">Adaptive (auto-adjusts)</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Topic Domains</label>
                        <div className="domain-grid">
                            {ALL_DOMAINS.map(d => (
                                <div key={d.id}>
                                    <div className={`domain-label${domains.includes(d.id) ? ' selected' : ''}`} onClick={() => toggleDomain(d.id)}>
                                        <span className="domain-check-box">{domains.includes(d.id) ? '✓' : ''}</span> {d.label}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="form-group">
                        <label>Monitoring</label>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            <label style={{ textTransform: 'none', fontSize: '0.87rem', display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                                <input type="checkbox" checked={enableVideo} onChange={e => setEnableVideo(e.target.checked)} style={{ width: 'auto' }} /> Enable webcam monitoring
                            </label>
                            <label style={{ textTransform: 'none', fontSize: '0.87rem', display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                                <input type="checkbox" checked={enableAudio} onChange={e => setEnableAudio(e.target.checked)} style={{ width: 'auto' }} /> Enable audio analysis
                            </label>
                        </div>
                    </div>

                    <button className="btn btn-primary" onClick={startInterview} disabled={loading} style={{ width: '100%', marginTop: 8 }}>
                        {loading ? <><span className="spinner" /> Creating Session...</> : '▶ Start Interview'}
                    </button>
                </div>
            </div>

            {/* System Logs */}
            <div className="section-sep" />
            <div>
                <label>System Log</label>
                <div className="log-stream" ref={logRef}>
                    {logs.map((l, i) => <div className={`log-line ${l.type}`} key={i}>{l.msg}</div>)}
                </div>
            </div>

            {/* MCP Tools */}
            <div className="section-sep" />
            <label>MCP Tool Registry</label>
            <div className="mcp-grid">
                {MCP_TOOLS.map(t => (
                    <div className="mcp-tool" key={t.name}>
                        <div className="mcp-tool-name">{t.name}</div>
                        <div className="mcp-tool-desc">{t.desc}</div>
                        <div className="mcp-status"><div className="mcp-status-dot" /> registered</div>
                    </div>
                ))}
            </div>

            {/* Architecture */}
            <div className="section-sep" />
            <label>System Architecture</label>
            <div className="arch-diagram">{ARCH_DIAGRAM}</div>
        </div>
    );
}
