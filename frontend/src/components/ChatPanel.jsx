import { useState, useRef, useEffect, useCallback } from 'react';
import ChatInput from './ChatInput';
import { api, streamChat } from '../api';

function renderMarkdown(text) {
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^[\-*] (.+)$/gm, '<li>$1</li>')
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    .replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>')
    .replace(/^---$/gm, '<hr>')
    .replace(/\n\n/g, '</p><p>');
  html = '<p>' + html + '</p>';
  html = html.replace(/<\/li>\s*<li>/g, '</li><li>');
  html = html.replace(/(<li>.*?<\/li>)/gs, m => `<ul>${m}</ul>`);
  html = html.replace(/<\/ul>\s*<ul>/g, '');
  html = html.replace(/<p><\/p>/g, '');
  return html;
}

function stripMd(t) {
  return t.replace(/<[^>]*>/g, '').replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\*([^*]+)\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1').replace(/^#{1,6}\s+/gm, '').replace(/^[-*]\s+/gm, '').replace(/^>\s+/gm, '')
    .replace(/\n{2,}/g, '. ').replace(/\n/g, ' ').trim();
}

function MessageBubble({ role, content, isStreaming }) {
  return (
    <div className={`msg ${role}`}>
      <div className="msg-avatar">{role === 'user' ? '👤' : '⚡'}</div>
      <div className="msg-body">
        <div className={`msg-content${isStreaming ? ' streaming-cursor' : ''}`}
             dangerouslySetInnerHTML={{ __html: content ? renderMarkdown(content) : '' }} />
        <div className="msg-time">{new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
      </div>
    </div>
  );
}

export default function ChatPanel({ sessionId, ensureSession, setSessionId }) {
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [welcome, setWelcome] = useState(true);
  const [loading, setLoading] = useState(false);
  const messagesEnd = useRef(null);
  const abortRef = useRef(null);
  const assistIdxRef = useRef(-1); // stable ref for streaming updates
  const sendingRef = useRef(false); // guard: don't clear while send is in progress
  const voiceActiveRef = useRef(false);

  // ── Session isolation: clear immediately on session change ──
  useEffect(() => {
    // Don't clear if a send operation is in flight — the session change
    // was triggered by ensureSession(), not by an external "New Chat" action.
    if (sendingRef.current) return;

    // Cancel any in-flight stream
    if (abortRef.current) abortRef.current.abort();
    setStreaming(false);
    setMessages([]);
    setWelcome(true);
    setLoading(false);
    assistIdxRef.current = -1;

    if (!sessionId) return;

    setLoading(true);
    api.getMessages(sessionId).then(data => {
      if (voiceActiveRef.current) return;
      const msgs = (data.messages || []).filter(m => m.role !== 'system');
      if (msgs.length) {
        setMessages(msgs.map(m => ({ role: m.role, content: m.content })));
        setWelcome(false);
        assistIdxRef.current = msgs.length;
      }
    }).catch(() => {
      setMessages([]);
      setWelcome(true);
    }).finally(() => setLoading(false));
  }, [sessionId]);

  useEffect(() => {
    const el = messagesEnd.current;
    if (el) {
      const parent = el.parentElement;
      const nearBottom = parent.scrollHeight - parent.scrollTop - parent.clientHeight < 80;
      if (nearBottom) el.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const saveRecent = (sid, query) => {
    const title = query.slice(0, 40) + (query.length > 40 ? '…' : '');
    // Update session title on server (cross-browser persistent)
    api.updateSessionTitle(sid, title).catch(() => {});
    // Notify Sidebar to refresh from server
    window.dispatchEvent(new CustomEvent('recent-chats-updated'));
  };

  // ── Send: uses functional setState to avoid stale closure ──
  const send = useCallback(async (text, images = []) => {
    if (voiceActiveRef.current || (!text.trim() && !images.length)) return;
    console.log('[ChatPanel] send called:', text.slice(0, 50));
    sendingRef.current = true;
    setWelcome(false);
    const sid = await ensureSession();
    console.log('[ChatPanel] session:', sid?.slice(0, 12));
    // Save to recent immediately — so "New Chat" preserves the conversation
    // even before the assistant replies.
    saveRecent(sid, text);
    const displayText = text || (images.length ? `[Attached: ${images.length} image(s)]` : '');

    // Batch user + assistant placeholder in one functional update
    setMessages(prev => {
      assistIdxRef.current = prev.length + 1;
      console.log('[ChatPanel] adding messages, count:', prev.length, 'assistIdx:', assistIdxRef.current);
      return [...prev,
        { role: 'user', content: displayText },
        { role: 'assistant', content: '' },
      ];
    });

    setStreaming(true);
    abortRef.current = new AbortController();
    let fullContent = '';
    try {
      await streamChat(text, sid, images, ({ type, data }) => {
        if (type === 'text_delta') {
          fullContent += data.content || data.delta || '';
          const idx = assistIdxRef.current;
          if (!fullContent.trim()) console.log('[ChatPanel] first text_delta, idx:', idx);
          setMessages(prev => {
            const next = [...prev];
            if (idx < next.length) next[idx] = { ...next[idx], content: fullContent };
            return next;
          });
        } else if (type === 'tool_call_start') {
          setMessages(prev => {
            const next = [...prev];
            const idx = assistIdxRef.current;
            if (idx >= 0 && idx <= next.length) {
              next.splice(idx, 0, { role: 'tool', content: `🔧 ${data.tool || ''}` });
              assistIdxRef.current = idx + 1;
            } else {
              next.push({ role: 'tool', content: `🔧 ${data.tool || ''}` });
            }
            return next;
          });
        } else if (type === 'tool_call_result') {
          if (data.error) setMessages(prev => [...prev, { role: 'tool', content: `❌ ${data.error}` }]);
        } else if (type === 'error') {
          setMessages(prev => [...prev, { role: 'error', content: `⚠️ ${data.error || 'Stream error'}` }]);
        } else if (type === 'done') {
          const clean = data.final_answer || '';
          if (data.diagnostics) {
            console.info('[ChatPanel] execution diagnostics:', data.diagnostics);
          }
          if (clean) {
            const idx = assistIdxRef.current;
            setMessages(prev => {
              const next = [...prev];
              if (idx < next.length) next[idx] = { ...next[idx], content: clean };
              return next;
            });
          }
          saveRecent(sid, text);
        }
      }, abortRef.current.signal);
    } catch (err) {
      if (err.name !== 'AbortError')
        setMessages(prev => [...prev, { role: 'error', content: `Error: ${err.message}` }]);
    }
    setStreaming(false);
    abortRef.current = null;
    sendingRef.current = false;
  }, [ensureSession]);

  const stop = () => { if (abortRef.current) abortRef.current.abort(); };

  // ── Voice handlers: progressive UI updates (transcript → answer → audio) ──
  useEffect(() => {
    let voiceUserIdx = -1;
    let voiceMsgIdx = -1;  // track assistant message index during voice session

    const onStart = () => {
      voiceActiveRef.current = true;
      setWelcome(false);
      setMessages(prev => {
        voiceUserIdx = prev.length;
        voiceMsgIdx = prev.length + 1;  // +1 for user msg we're about to add
        return [...prev,
          { role: 'user', content: '🎤 ...' },
          { role: 'assistant', content: '' },
        ];
      });
    };

    const onTranscript = (e) => {
      const t = e.detail.text || '';
      setMessages(prev => {
        const next = [...prev];
        if (voiceUserIdx >= 0 && voiceUserIdx < next.length) {
          next[voiceUserIdx] = { ...next[voiceUserIdx], content: `🎤 ${t}` };
        }
        return next;
      });
    };

    const onDelta = (e) => {
      setMessages(prev => {
        const next = [...prev];
        if (voiceMsgIdx >= 0 && voiceMsgIdx < next.length) {
          next[voiceMsgIdx] = { ...next[voiceMsgIdx], content: next[voiceMsgIdx].content + (e.detail.content || '') };
        }
        return next;
      });
    };

    const onTool = (e) => {
      setMessages(prev => {
        const next = [...prev];
        if (voiceMsgIdx >= 0 && voiceMsgIdx <= next.length) {
          next.splice(voiceMsgIdx, 0, { role: 'tool', content: `🔧 ${e.detail.tool || ''}` });
          voiceMsgIdx += 1;
        } else {
          next.push({ role: 'tool', content: `🔧 ${e.detail.tool || ''}` });
        }
        return next;
      });
    };

    const onAudio = (e) => {
      const { audio, format } = e.detail;
      if (!audio) return;
      try {
        const binary = atob(audio);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const blob = new Blob([bytes], { type: format === 'mp3' ? 'audio/mpeg' : 'audio/wav' });
        new Audio(URL.createObjectURL(blob)).play().catch(() => {});
      } catch {}
    };

    const applyFinalAnswer = (finalAnswer) => {
      if (!finalAnswer || voiceMsgIdx < 0) return;
      setMessages(prev => {
        const next = [...prev];
        if (voiceMsgIdx < next.length) {
          next[voiceMsgIdx] = { ...next[voiceMsgIdx], content: finalAnswer };
        }
        return next;
      });
    };

    const onFinal = (e) => {
      applyFinalAnswer(e.detail.answer || '');
    };

    const onDone = (e) => {
      applyFinalAnswer(e.detail.answer || '');
      voiceActiveRef.current = false;
      voiceUserIdx = -1;
      voiceMsgIdx = -1;
    };

    const onError = (e) => {
      setMessages(prev => [...prev, { role: 'error', content: `⚠️ ${e.detail.error || 'Voice failed'}` }]);
      voiceActiveRef.current = false;
      voiceUserIdx = -1;
      voiceMsgIdx = -1;
    };

    const events = {
      'voice-start': onStart, 'voice-transcript': onTranscript,
      'voice-delta': onDelta, 'voice-tool': onTool,
      'voice-final': onFinal, 'voice-audio': onAudio,
      'voice-done': onDone, 'voice-error': onError,
    };
    Object.entries(events).forEach(([name, fn]) => window.addEventListener(name, fn));
    return () => Object.entries(events).forEach(([name, fn]) => window.removeEventListener(name, fn));
  }, []);

  return (
    <div className={`chat-panel${welcome && !loading ? ' welcome-active' : ''}`}>
      <div className="chat-messages">
        {loading && (
          <div className="welcome-block"><div className="welcome-title" style={{fontSize:'1rem',color:'var(--muted-foreground)'}}>Loading…</div></div>
        )}
        {!loading && welcome && (
          <div className="welcome-block">
            <div className="welcome-title">有什么可以帮你的？</div>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === 'tool' || m.role === 'error' ? (
            <div key={i} className={`msg-tool ${m.role}`}>{m.content}</div>
          ) : (
            <MessageBubble key={i} role={m.role} content={m.content} isStreaming={streaming && i === assistIdxRef.current} />
          )
        )}
        <div ref={messagesEnd} />
      </div>
      <ChatInput onSend={send} streaming={streaming} onStop={stop} sessionId={sessionId} ensureSession={ensureSession} />
    </div>
  );
}
