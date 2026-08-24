const API_BASE = localStorage.getItem('agr_api_base') || '';

async function request(path, opts = {}) {
  const url = `${API_BASE}${path}`;
  const config = { headers: {}, ...opts };
  if (config.body && typeof config.body === 'object' && !(config.body instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json';
    config.body = JSON.stringify(config.body);
  }
  const res = await fetch(url, config);
  if (!res.ok) {
    const errBody = await res.json().catch(() => null);
    throw new Error(errBody?.detail || errBody?.error || `HTTP ${res.status}`);
  }
  return res;
}

export const api = {
  health: () => request('/health').then(r => r.json()),
  readiness: () => request('/ready').then(r => r.json()),

  // Chat
  createSession: (userId = 'web_ui') =>
    request('/api/v1/session', { method: 'POST', body: { user_id: userId } }).then(r => r.json()),
  getSession: (id) => request(`/api/v1/session/${id}`).then(r => r.json()),
  getMessages: (id) => request(`/api/v1/session/${id}/messages`).then(r => r.json()),
  listSessions: (userId = 'default') => request(`/api/v1/sessions?user_id=${userId}`).then(r => r.json()),
  updateSessionTitle: (sessionId, title) =>
    request(`/api/v1/session/${sessionId}?title=${encodeURIComponent(title)}`, { method: 'PUT' }).then(r => r.json()),
  deleteSession: (sessionId) =>
    request(`/api/v1/session/${sessionId}`, { method: 'DELETE' }).then(r => r.json()),
  clearSessionMessages: (sessionId) =>
    request(`/api/v1/session/${sessionId}/messages`, { method: 'DELETE' }).then(r => r.json()),
  deleteAllSessions: () =>
    request('/api/v1/sessions', { method: 'DELETE' }).then(r => r.json()),

  // RAG
  search: (query, topK = 3, mode = 'hybrid', modality) =>
    request('/api/v1/rag/search', {
      method: 'POST',
      body: { query, top_k: topK, mode, ...(modality && modality !== 'all' ? { modality_filter: [modality] } : {}) },
    }).then(r => r.json()),
  stats: () => request('/api/v1/rag/stats').then(r => r.json()),
  listDocuments: () => request('/api/v1/rag/documents').then(r => r.json()),
  getDocument: (docId) => request(`/api/v1/rag/document/${docId}`).then(r => r.json()),
  deleteDocument: (docId) => request(`/api/v1/rag/document/${docId}`, { method: 'DELETE' }).then(r => r.json()),
  clearKB: () => request('/api/v1/rag/clear', { method: 'POST' }).then(r => r.json()),
  uploadFile: (file, source, ingestMode, mmMethod, chunkSize = 800, chunkOverlap = 150, enableKG = true) => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('source', source);
    fd.append('ingest_mode', ingestMode);
    fd.append('mm_method', mmMethod);
    fd.append('chunk_size', chunkSize);
    fd.append('chunk_overlap', chunkOverlap);
    fd.append('enable_kg', enableKG);
    return request('/api/v1/rag/upload', { method: 'POST', body: fd }).then(r => r.json());
  },
  serveFile: (path) => `${API_BASE}/api/v1/rag/file/${path.split('/').map(encodeURIComponent).join('/')}`,

  // Voice (streaming SSE)
  voiceChatStream: async (audioBlob, sessionId, tts, onEvent, signal) => {
    const fd = new FormData();
    fd.append('audio', audioBlob, 'recording.webm');
    fd.append('session_id', sessionId || '');
    fd.append('tts', tts !== false);
    const res = await fetch(`${API_BASE}/api/v1/chat/voice`, { method: 'POST', body: fd, signal });
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${res.status}`); }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', currentEvent = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ') && currentEvent) {
          try {
            const data = JSON.parse(line.slice(6));
            onEvent({ type: currentEvent, data });
          } catch {}
          currentEvent = '';
        }
      }
    }
  },

  // Settings
  getSettings: () => request('/api/v1/settings').then(r => r.json()),
  saveSettings: (sections) => request('/api/v1/settings', { method: 'PUT', body: { sections } }).then(r => r.json()),
  testConnection: (payload) => request('/api/v1/settings/test', { method: 'POST', body: payload }).then(r => r.json()),

  // MCP
  getMcpConfig: () => request('/api/v1/mcp/servers').then(r => r.json()),
  saveMcpConfig: (servers) => request('/api/v1/mcp/servers', { method: 'PUT', body: { servers } }).then(r => r.json()),
};

export async function streamChat(message, sessionId, images, onEvent, signal) {
  const res = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId, stream: true, images }),
    signal,
  });

  // Check HTTP status before processing stream
  if (!res.ok) {
    const errBody = await res.json().catch(() => null);
    throw new Error(errBody?.detail || errBody?.error || `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let receivedDone = false;

  try {
    while (!receivedDone) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const s = line.slice(6).trim();
        if (!s || s === '[DONE]') {
          receivedDone = true;
          break;
        }
        try {
          const parsed = JSON.parse(s);
          const type = parsed.event || parsed.event_type;
          let data = parsed.data || parsed;
          if (typeof data === 'string') {
            try { data = JSON.parse(data); } catch { /* keep raw string if unparseable */ }
          }
          onEvent({ type, data });
          // Stop reading when we receive the done event
          if (type === 'done') {
            receivedDone = true;
            break;
          }
        } catch { /* skip malformed */ }
      }
    }
  } finally {
    // Always release the reader to clean up resources
    reader.releaseLock();
  }
}
