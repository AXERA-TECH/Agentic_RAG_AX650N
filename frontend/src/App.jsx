import { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';
import KBPanel from './components/KBPanel';
import SettingsPanel from './components/SettingsPanel';
import { api } from './api';
import './App.css';

const PANELS = { chat: 'Chat', kb: 'Knowledge Base', settings: 'Settings' };

export default function App() {
  const [panel, setPanel] = useState('chat');
  const [connected, setConnected] = useState(false);
  const [sessionId, setSessionId] = useState(localStorage.getItem('agr_session_id') || '');
  const [chatKey, setChatKey] = useState(0);
  const [clearKey, setClearKey] = useState(0);

  useEffect(() => {
    fetch('/health').then(() => setConnected(true)).catch(() => setConnected(false));
  }, []);

  // Eagerly create/restore a session on mount so ChatPanel never sees a
  // sessionId transition during send — that transition would trigger the
  // useEffect([sessionId]) cleanup which clears messages and aborts streams.
  useEffect(() => {
    if (!sessionId) {
      fetch('/api/v1/session', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_id: 'web_ui' }) })
        .then(r => r.json())
        .then(({ session_id }) => {
          setSessionId(session_id);
          localStorage.setItem('agr_session_id', session_id);
        })
        .catch(() => {
          const sid = 'web_' + Date.now().toString(36);
          setSessionId(sid);
        });
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const ensureSession = useCallback(async () => {
    if (sessionId) return sessionId;
    try {
      const res = await fetch('/api/v1/session', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_id: 'web_ui' }) });
      const { session_id } = await res.json();
      setSessionId(session_id);
      localStorage.setItem('agr_session_id', session_id);
      return session_id;
    } catch { const sid = 'web_' + Date.now().toString(36); setSessionId(sid); return sid; }
  }, [sessionId]);

  return (
    <div className="app">
      <Sidebar panel={panel} onNavigate={setPanel} connected={connected} sessionId={sessionId} onSessionChange={setSessionId} />
      <main className="main">
        <header className="topbar">
          <span className="topbar-title">{PANELS[panel] || panel}</span>
          <span className="topbar-spacer" />
          <button className="icon-btn" onClick={() => { setSessionId(''); localStorage.removeItem('agr_session_id'); setChatKey(k => k + 1); setPanel('chat'); }} title="New Chat">＋</button>
          <button className="icon-btn" onClick={async () => {
            if (sessionId) await api.clearSessionMessages(sessionId);
            setClearKey(k => k + 1);
            setPanel('chat');
          }} title="Clear Messages">🗑</button>
        </header>
        <div className="content">
          {panel === 'chat' && <ChatPanel key={`${chatKey}-${clearKey}`} sessionId={sessionId} ensureSession={ensureSession} setSessionId={setSessionId} />}
          {panel === 'kb' && <KBPanel />}
          {panel === 'settings' && <SettingsPanel />}
        </div>
      </main>
    </div>
  );
}
