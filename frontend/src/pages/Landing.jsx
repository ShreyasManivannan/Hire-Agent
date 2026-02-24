import { useNavigate } from 'react-router-dom';

const FEATURES = [
    { icon: '📄', title: 'Resume Parser', desc: 'NLP-powered extraction of skills, experience, and projects using spaCy' },
    { icon: '🧠', title: 'RAG Questions', desc: 'ChromaDB vector search with curated question bank across 5 domains' },
    { icon: '⚡', title: '4-bit Quantized LLM', desc: 'BitsAndBytes NF4 quantization with LoRA fine-tuning support' },
    { icon: '🕵️', title: 'AI Detection', desc: 'Perplexity + burstiness analysis to detect AI-generated answers' },
    { icon: '🔀', title: 'Adaptive Flow', desc: 'Dynamic topic switching based on performance and time limits' },
    { icon: '📊', title: 'MCP Protocol', desc: 'Model Context Protocol orchestration with structured tool outputs' },
];

export default function Landing() {
    const navigate = useNavigate();

    return (
        <div className="page page-landing">
            <div className="hero-badge">✦ <span>AI-Powered</span> Technical Interview System</div>
            <h1 className="hero-title">Interview smarter,<br /><em>hire better</em></h1>
            <p className="hero-sub">Multi-agent AI system with RAG-based question generation, real-time AI detection, adaptive interviewing, and comprehensive analytics.</p>
            <div className="hero-actions">
                <button className="btn btn-primary" onClick={() => navigate('/setup')}>▶ Start Interview</button>
                <button className="btn btn-secondary" onClick={() => navigate('/setup')}>View Architecture</button>
            </div>
            <div className="features-grid">
                {FEATURES.map((f, i) => (
                    <div className="feature-card" key={i}>
                        <div className="feature-icon">{f.icon}</div>
                        <div className="feature-title">{f.title}</div>
                        <div className="feature-desc">{f.desc}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
