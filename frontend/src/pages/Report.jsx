import { useState, useEffect, useMemo } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

function launchConfetti() {
    const colors = ['#6c63ff', '#43e97b', '#ff6b6b', '#ffa83b', '#a78bfa'];
    for (let i = 0; i < 60; i++) {
        setTimeout(() => {
            const el = document.createElement('div');
            el.className = 'confetti-piece';
            el.style.cssText = `left:${Math.random() * 100}vw;background:${colors[Math.floor(Math.random() * colors.length)]};
        animation-duration:${1.5 + Math.random() * 2}s;animation-delay:${Math.random() * 0.5}s;
        border-radius:${Math.random() > 0.5 ? '50%' : '2px'};width:${6 + Math.random() * 6}px;height:${6 + Math.random() * 6}px;`;
            document.body.appendChild(el);
            setTimeout(() => el.remove(), 3000);
        }, i * 30);
    }
}

export default function Report() {
    const navigate = useNavigate();
    const { sessionId } = useParams();
    const [donutOffset, setDonutOffset] = useState(251);

    // Load results
    const results = useMemo(() => {
        try { return JSON.parse(sessionStorage.getItem('interviewResults') || '{}'); } catch { return {}; }
    }, []);

    const responses = results.responses || [];
    const candidateName = results.candidateName || 'Candidate';
    const startTime = results.startTime || Date.now();
    const topicScoresRaw = results.topicScores || {};

    // Compute scores
    const allScores = responses.map(r => r.score);
    const overall = allScores.length ? allScores.reduce((a, b) => a + b, 0) / allScores.length : 0;
    const avgAI = responses.length ? responses.map(r => r.aiProb).reduce((a, b) => a + b, 0) / responses.length : 0;
    const avgSent = responses.length ? responses.map(r => r.sentiment).reduce((a, b) => a + b, 0) / responses.length : 0.5;
    const avgAtt = 0.6 + Math.random() * 0.35; // simulated
    const duration = ((Date.now() - startTime) / 60000).toFixed(1);

    const pct = Math.round(overall * 100);
    const aiPct = Math.round(avgAI * 100);

    // Topic averages
    const topicAvgs = {};
    for (const [t, scores] of Object.entries(topicScoresRaw)) {
        if (scores && scores.length) topicAvgs[t] = scores.reduce((a, b) => a + b, 0) / scores.length;
    }
    const strengths = Object.entries(topicAvgs).filter(([, s]) => s >= 0.7).map(([t]) => t);
    const weaknesses = Object.entries(topicAvgs).filter(([, s]) => s < 0.5).map(([t]) => t);

    // Recommendation
    let rec, recClass;
    if (overall >= 0.7 && avgAI < 0.3) { rec = 'STRONG HIRE'; recClass = 'hire'; }
    else if (overall >= 0.5 && avgAI < 0.5) { rec = 'HIRE'; recClass = 'hire'; }
    else if (overall >= 0.35) { rec = 'MAYBE'; recClass = 'maybe'; }
    else { rec = 'NO HIRE'; recClass = 'nohire'; }

    const confStr = Math.abs(overall - 0.5) > 0.2 ? 'High Confidence' : 'Medium Confidence';

    // Donut animation
    const circumference = 2 * Math.PI * 40;
    useEffect(() => {
        const timer = setTimeout(() => setDonutOffset(circumference * (1 - overall)), 300);
        launchConfetti();
        return () => clearTimeout(timer);
    }, []);

    const donutColor = pct >= 70 ? 'var(--accent3)' : pct >= 45 ? '#ffa500' : 'var(--accent2)';

    // Breakdown items
    const breakdownItems = [
        { label: 'Technical', val: Math.round(overall * 100) },
        { label: 'Sentiment', val: Math.round(avgSent * 100) },
        { label: 'Attention', val: Math.round(avgAtt * 100) },
    ];

    function downloadReport() {
        const report = {
            candidate: candidateName,
            session: sessionId,
            date: new Date(startTime).toISOString(),
            overall: pct,
            recommendation: rec,
            aiDetection: aiPct,
            duration,
            topics: topicAvgs,
            responses,
        };
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'hireagent_report_' + candidateName.replace(/\s+/g, '_') + '.json';
        a.click();
        URL.revokeObjectURL(url);
    }

    return (
        <div className="page page-report">
            {/* Header */}
            <div className="report-header">
                <div className="candidate-info">
                    <h3>{candidateName}</h3>
                    <p>{sessionId} · {new Date(startTime).toLocaleString()}</p>
                </div>
                <div className={`recommendation-box ${recClass}`}>
                    <div className={`rec-decision ${recClass}`}>{rec}</div>
                    <div className="rec-conf">{confStr}</div>
                </div>
            </div>

            <div className="report-grid">
                {/* Overall score */}
                <div className="report-card">
                    <h4>Overall Performance</h4>
                    <div className="score-display">
                        <div className="score-donut">
                            <svg viewBox="0 0 90 90">
                                <circle className="donut-bg" cx="45" cy="45" r="40" />
                                <circle className="donut-fill" cx="45" cy="45" r="40" style={{ strokeDashoffset: donutOffset, stroke: donutColor }} />
                            </svg>
                            <div className="score-num">{pct}<small>/ 100</small></div>
                        </div>
                        <div className="score-breakdown">
                            {breakdownItems.map(item => (
                                <div className="breakdown-item" key={item.label}>
                                    <div className="breakdown-label">{item.label} <span>{item.val}%</span></div>
                                    <div className="breakdown-bar">
                                        <div className="breakdown-bar-fill" style={{ width: item.val + '%', background: item.val >= 70 ? 'var(--accent3)' : item.val >= 45 ? '#ffa500' : 'var(--accent2)' }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* AI Detection */}
                <div className="report-card">
                    <h4>AI Detection Analysis</h4>
                    <div className="ai-meter">
                        <div style={{ fontSize: '0.82rem', color: 'var(--text2)', marginBottom: 4 }}>Probability of AI-Generated Responses</div>
                        <div className="ai-big-gauge">
                            <div className="ai-needle" style={{ left: aiPct + '%' }} />
                        </div>
                        <div className="ai-labels"><span>Human</span><span>Uncertain</span><span>AI</span></div>
                        <div className="ai-result" style={{ color: aiPct < 30 ? 'var(--accent3)' : aiPct < 60 ? '#ffa500' : 'var(--accent2)' }}>
                            {aiPct < 30 ? '✅ Likely Human' : aiPct < 60 ? '⚠️ Uncertain' : '🚨 Likely AI-Generated'}
                        </div>
                        <div style={{ fontSize: '0.78rem', color: 'var(--text2)', marginTop: 4 }}>
                            AI Detection: {aiPct < 30 ? 'LOW' : aiPct < 60 ? 'MEDIUM' : 'HIGH'} concern ({aiPct}%)
                        </div>
                    </div>
                </div>

                {/* Topic Analysis */}
                <div className="report-card">
                    <h4>Topic Analysis</h4>
                    <div>
                        {Object.entries(topicAvgs).map(([t, s]) => {
                            const p = Math.round(s * 100);
                            const color = p >= 70 ? 'var(--accent3)' : p >= 45 ? '#ffa500' : 'var(--accent2)';
                            return (
                                <div className="breakdown-item" key={t}>
                                    <div className="breakdown-label" style={{ textTransform: 'capitalize' }}>{t.replace('_', ' ')} <span style={{ color }}>{p}%</span></div>
                                    <div className="breakdown-bar"><div className="breakdown-bar-fill" style={{ width: p + '%', background: color }} /></div>
                                </div>
                            );
                        })}
                        {strengths.length > 0 && <div style={{ marginTop: 12, fontSize: '0.78rem', color: 'var(--accent3)' }}>✓ Strong: {strengths.map(s => s.replace('_', ' ')).join(', ')}</div>}
                        {weaknesses.length > 0 && <div style={{ fontSize: '0.78rem', color: 'var(--accent2)', marginTop: 4 }}>✗ Weak: {weaknesses.map(s => s.replace('_', ' ')).join(', ')}</div>}
                    </div>
                </div>

                {/* Session Summary */}
                <div className="report-card">
                    <h4>Session Summary</h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                        {[
                            ['Duration', duration + ' min'],
                            ['Questions Asked', responses.length],
                            ['Topics Covered', Object.keys(topicAvgs).length],
                            ['Avg Attention', Math.round(avgAtt * 100) + '%'],
                            ['AI Detection', aiPct + '%'],
                        ].map(([k, v]) => (
                            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
                                <span style={{ color: 'var(--text2)' }}>{k}</span>
                                <span style={{ fontFamily: "'DM Mono',monospace", fontWeight: 500 }}>{v}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Detailed responses */}
            <div className="responses-section">
                <h3>Detailed Q&A Review</h3>
                {responses.length === 0 ? (
                    <div style={{ color: 'var(--text2)', fontSize: '0.9rem', textAlign: 'center', padding: 20 }}>No responses recorded</div>
                ) : (
                    responses.map((r, i) => (
                        <div className="response-item" key={i}>
                            <div className="resp-q">
                                <span className="resp-q-num">Q{i + 1}</span>
                                <span>{r.question}</span>
                            </div>
                            <div className="resp-a">{r.answer.substring(0, 300)}{r.answer.length > 300 ? '...' : ''}</div>
                            <div className="resp-scores">
                                <span className="resp-score-pill pill-score">Score: {Math.round(r.score * 100)}%</span>
                                <span className="resp-score-pill pill-ai">AI: {Math.round(r.aiProb * 100)}%</span>
                                <span className="resp-score-pill" style={{ background: 'rgba(136,136,170,0.1)', color: 'var(--text2)' }}>
                                    {r.topic.replace('_', ' ')}
                                </span>
                            </div>
                        </div>
                    ))
                )}
            </div>

            <div className="section-sep" />
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <button className="btn btn-primary" onClick={downloadReport}>⬇ Download Report (JSON)</button>
                <button className="btn btn-secondary" onClick={() => navigate('/setup')}>↩ New Interview</button>
                <button className="btn btn-secondary" onClick={() => navigate('/')}>🏠 Home</button>
            </div>
        </div>
    );
}
