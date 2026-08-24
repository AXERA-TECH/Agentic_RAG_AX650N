import { useState, useEffect, useCallback } from 'react';
import { api } from '../api';

const NAV_ITEMS = [
  { id: 'chat', icon: '💬', label: 'Chat' },
];

export default function Sidebar({ panel, onNavigate, connected, sessionId, onSessionChange }) {
  const [collapsed, setCollapsed] = useState(localStorage.getItem('agr_sidebar_collapsed') === 'true');
  const [recentChats, setRecentChats] = useState([]);

  // Load recent chats from backend (cross-browser persistent)
  const loadSessions = useCallback(async () => {
    try {
      const data = await api.listSessions();
      setRecentChats((data.sessions || []).map(s => ({
        sessionId: s.id,
        title: s.title || s.id.slice(0, 8),
        time: s.updated_at || s.created_at,
      })));
    } catch {}
  }, []);

  useEffect(() => { loadSessions(); }, [loadSessions]);

  // Refresh when ChatPanel signals a new message was sent
  useEffect(() => {
    const refresh = () => { loadSessions(); };
    window.addEventListener('recent-chats-updated', refresh);
    return () => window.removeEventListener('recent-chats-updated', refresh);
  }, [loadSessions]);

  const toggle = () => {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem('agr_sidebar_collapsed', next);
  };

  const openRecent = useCallback(async (chat) => {
    if (!chat.sessionId) return;
    onSessionChange(chat.sessionId);
    localStorage.setItem('agr_session_id', chat.sessionId);
    onNavigate('chat');
  }, [onNavigate, onSessionChange]);

  const deleteRecent = async (e, idx) => {
    e.stopPropagation();
    const chat = recentChats[idx];
    if (!chat?.sessionId) return;
    try {
      await api.deleteSession(chat.sessionId);
      loadSessions();
    } catch {}
  };

  const clearAllRecent = async () => {
    if (!confirm('删除全部最近对话？此操作不可恢复。')) return;
    try {
      await api.deleteAllSessions();
      setRecentChats([]);
    } catch {}
  };

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-brand">
        <h1>Agentic RAG</h1>
        <button className="sidebar-toggle" onClick={toggle}>{collapsed ? '▶' : '◀'}</button>
      </div>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map(item => (
          <div key={item.id} className={`nav-item ${panel === item.id ? 'active' : ''}`} onClick={() => onNavigate(item.id)}>
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </div>
        ))}
        {!collapsed && (
          <div className="sidebar-recent">
            <div className="sidebar-recent-title">
              <span>最近对话</span>
              {recentChats.length > 0 && (
                <span className="recent-clear-all" onClick={clearAllRecent} title="清除全部">🗑</span>
              )}
            </div>
            <div className="sidebar-recent-list">
              {recentChats.slice(0, 10).map((c, i) => (
                <div key={c.sessionId || i} className="sidebar-recent-item" onClick={() => openRecent(c)} title={c.title}>
                  <span className="sidebar-recent-item-title">{c.title}</span>
                  <span className="recent-delete" onClick={e => deleteRecent(e, i)}>×</span>
                </div>
              ))}
              {!recentChats.length && <div className="sidebar-recent-empty">—</div>}
            </div>
          </div>
        )}
        <div className="nav-spacer" />
        <div className={`nav-item ${panel === 'kb' ? 'active' : ''}`} onClick={() => onNavigate('kb')}>
          <span className="nav-icon">📚</span><span className="nav-label">Knowledge</span>
        </div>
        <div className={`nav-item ${panel === 'settings' ? 'active' : ''}`} onClick={() => onNavigate('settings')}>
          <span className="nav-icon">⚙</span><span className="nav-label">Settings</span>
        </div>
      </nav>
      <div className="sidebar-footer">
        <span className={`status-dot ${connected ? 'online' : 'offline'}`} />
        <span>{connected ? 'Connected' : 'Disconnected'}</span>
      </div>
    </aside>
  );
}
