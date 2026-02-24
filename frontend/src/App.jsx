import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import Landing from './pages/Landing';
import Setup from './pages/Setup';
import Interview from './pages/Interview';
import Report from './pages/Report';
import './index.css';

function Nav() {
  const location = useLocation();
  const isInterview = location.pathname.startsWith('/interview');

  return (
    <nav>
      <Link to="/" className="logo">
        <div className="logo-dot"></div>
        HireAgent
      </Link>
      <div className="nav-status">
        {isInterview && <span className="status-pill session">Session Active</span>}
        <span className="status-pill online">System Online</span>
      </div>
    </nav>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/interview/:sessionId" element={<Interview />} />
        <Route path="/report/:sessionId" element={<Report />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
