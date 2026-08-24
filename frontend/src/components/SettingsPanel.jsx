import { useState, useEffect, useCallback } from 'react';
import { api } from '../api';

const TABS = [
  { key: 'llm', label: '模型 API', icon: '✦' },
  { key: 'embedding', label: 'Embedding', icon: '⊡' },
  { key: 'milvus', label: 'Milvus', icon: '▥' },
  { key: 'voice', label: '语音', icon: '♪' },
  { key: 'ocr', label: 'OCR', icon: '☷' },
  { key: 'gateway', label: 'Gateway', icon: '⇄' },
  { key: 'mcp', label: 'MCP', icon: '⌘' },
  { key: 'connection', label: '后端连接', icon: '⌁' },
];

const BLANK_LLM = { name: 'openai', api_base: '', api_key: '', model: '', vision_model: '', max_tokens: 4096, temperature: 0.7, has_api_key: false, api_key_preview: '' };
const BLANK_EMB = { provider: '', model: '', dim: 1536, api_base: '', api_key: '', batch_size: 100, has_api_key: false, api_key_preview: '' };
const BLANK_MILVUS = { host: 'localhost', port: 19530, dim: 1536 };
const BLANK_API = { host: '0.0.0.0', port: 8000 };
const BLANK_VOICE = { stt_provider: 'sensevoice', stt_model: '', stt_api_base: '', stt_language: 'auto', tts_provider: 'qwen', tts_model: '', tts_api_base: '', tts_language: 'Chinese', tts_voice: '', tts_speed: 1.0 };
const BLANK_OCR = { enabled: true, api_base: '', model: '', max_pages: 50 };
const BLANK_GATEWAY = { enabled: false, response_mode: 'sync', max_reply_length: 2000, wechat_work: { enabled: false, corp_id: '', token: '', encoding_aes_key: '', agent_id: '', secret: '' }, dingtalk: { enabled: false, app_key: '', app_secret: '' }, qqbot: { enabled: false, app_id: '', app_secret: '', sandbox: true } };

function Field({ label, help, children }) {
  return (
    <div className="set-field">
      <label>{label}</label>
      {children}
      {help && <small>{help}</small>}
    </div>
  );
}

function SecretInput({ value, onChange, placeholder, hasKey, preview }) {
  const [show, setShow] = useState(false);
  return (
    <div className="secret-wrap">
      <input
        type={show ? 'text' : 'password'}
        value={value}
        onChange={onChange}
        placeholder={hasKey ? `已保存 ${preview} · 留空保持不变` : placeholder}
        autoComplete="new-password"
      />
      <button type="button" onClick={() => setShow(v => !v)} title={show ? '隐藏' : '显示'}>
        {show ? '◉' : '◎'}
      </button>
    </div>
  );
}

function Toast({ text, type, onDone }) {
  useEffect(() => {
    const t = setTimeout(onDone, 2800);
    return () => clearTimeout(t);
  }, [onDone]);
  return <div className={`set-toast ${type}`}>{text}</div>;
}

export default function SettingsPanel() {
  const [tab, setTab] = useState('llm');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState(null);

  // All settings from server
  const [llm, setLlm] = useState(BLANK_LLM);
  const [providers, setProviders] = useState([]);
  const [defaultProvider, setDefaultProvider] = useState('');
  const [embedding, setEmbedding] = useState(BLANK_EMB);
  const [milvus, setMilvus] = useState(BLANK_MILVUS);
  const [apiServer, setApiServer] = useState(BLANK_API);
  const [voice, setVoice] = useState(BLANK_VOICE);
  const [ocr, setOcr] = useState(BLANK_OCR);
  const [gateway, setGateway] = useState(BLANK_GATEWAY);
  const [mcpServers, setMcpServers] = useState({});

  const [apiBase, setApiBase] = useState(localStorage.getItem('agr_api_base') || '');

  // MCP modal
  const [mcpModal, setMcpModal] = useState(null);

  const toastMsg = useCallback((type, text) => {
    setToast({ type, text, key: Date.now() });
  }, []);

  useEffect(() => {
    api.getSettings()
      .then(data => {
        setDefaultProvider(data.default_provider || '');
        const provs = data.llm_providers || [];
        setProviders(provs);
        const cur = provs.find(p => p.name === data.default_provider) || provs[0] || BLANK_LLM;
        setLlm({ ...cur, api_key: '' });
        setEmbedding({ ...BLANK_EMB, ...data.embedding, api_key: '' });
        setMilvus({ ...BLANK_MILVUS, ...data.milvus });
        setApiServer({ ...BLANK_API, ...data.api });
        setVoice({ ...BLANK_VOICE, ...data.voice });
        setOcr({ ...BLANK_OCR, ...data.ocr });
        setGateway(data.gateway || BLANK_GATEWAY);
        setMcpServers(data.mcp_servers || {});
      })
      .catch(() => toastMsg('error', '无法加载设置'))
      .finally(() => setLoading(false));
  }, [toastMsg]);

  const selectProvider = (name) => {
    const p = providers.find(x => x.name === name);
    if (p) setLlm({ ...p, api_key: '' });
  };

  const saveSection = async (section, data) => {
    setSaving(true);
    try {
      await api.saveSettings([{ section, data }]);
      toastMsg('success', '已保存');
    } catch (e) {
      toastMsg('error', e.message);
    } finally {
      setSaving(false);
    }
  };

  const testLlm = async () => {
    try {
      const res = await api.testConnection({ api_base: llm.api_base, api_key: llm.api_key || null, model: llm.model, provider_type: llm.name === 'claude' ? 'claude' : 'openai' });
      toastMsg('success', `连接成功 · ${res.detected_model || llm.model}`);
    } catch (e) {
      toastMsg('error', e.message);
    }
  };

  // ── MCP helpers ──
  const mcpList = Object.entries(mcpServers);

  const openMcp = (name) => {
    const s = mcpServers[name] || { command: '', args: [], env: {}, disabled: false };
    setMcpModal({
      name, isNew: !name,
      command: s.command,
      args: Array.isArray(s.args) ? s.args.join(' ') : (s.args || ''),
      env: s.env ? Object.entries(s.env).map(([k, v]) => `${k}=${v}`).join('\n') : '',
      disabled: !!s.disabled,
    });
  };

  const saveMcpLocal = () => {
    if (!mcpModal.name) return;
    const env = {};
    mcpModal.env.split('\n').filter(Boolean).forEach(line => {
      const i = line.indexOf('=');
      if (i > 0) env[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    });
    setMcpServers(prev => ({
      ...prev,
      [mcpModal.name]: { command: mcpModal.command, args: mcpModal.args.split(/\s+/).filter(Boolean), env, disabled: mcpModal.disabled },
    }));
    setMcpModal(null);
  };

  const deleteMcp = () => {
    if (!confirm(`Delete ${mcpModal.name}?`)) return;
    setMcpServers(prev => {
      const n = { ...prev };
      delete n[mcpModal.name];
      return n;
    });
    setMcpModal(null);
  };

  const saveMcpServer = async () => {
    setSaving(true);
    try {
      await api.saveMcpConfig(mcpServers);
      toastMsg('success', 'MCP 已保存 · 重启服务生效');
    } catch (e) {
      toastMsg('error', e.message);
    } finally {
      setSaving(false);
    }
  };

  const saveConnection = () => {
    localStorage.setItem('agr_api_base', apiBase.trim());
    toastMsg('success', '已保存 · 刷新页面生效');
  };

  if (loading) return <div className="set-loading">加载设置...</div>;

  return (
    <div className="set-page">
      {/* Tab bar */}
      <nav className="set-tabs">
        {TABS.map(t => (
          <button key={t.key} className={`set-tab ${tab === t.key ? 'active' : ''}`} onClick={() => setTab(t.key)}>
            <span>{t.icon}</span> {t.label}
          </button>
        ))}
      </nav>

      <div className="set-body">
        {/* ── LLM Provider ── */}
        {tab === 'llm' && (
          <section className="set-card">
            <div className="set-card-hd"><h2>✦ 模型 API</h2><small>配置 LLM 服务商、地址与模型</small></div>
            <div className="set-card-bd">
              <div className="set-row-tags">
                {providers.map(p => (
                  <button key={p.name} className={`set-chip ${llm.name === p.name ? 'active' : ''}`} onClick={() => selectProvider(p.name)}>
                    {p.name} {p.name === defaultProvider ? '(默认)' : ''}
                  </button>
                ))}
              </div>
              <div className="set-grid">
                <Field label="配置名称">
                  <select value={llm.name} onChange={e => selectProvider(e.target.value)}>
                    {providers.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
                  </select>
                </Field>
                <Field label="API Base URL">
                  <input value={llm.api_base} onChange={e => setLlm({ ...llm, api_base: e.target.value })} placeholder="https://api.openai.com/v1" />
                </Field>
                <Field label="API Key" help={llm.has_api_key ? '已存在密钥，留空保持不变' : ''}>
                  <SecretInput value={llm.api_key || ''} onChange={e => setLlm({ ...llm, api_key: e.target.value })} placeholder="sk-..." hasKey={llm.has_api_key} preview={llm.api_key_preview} />
                </Field>
                <Field label="Chat Model">
                  <input value={llm.model} onChange={e => setLlm({ ...llm, model: e.target.value })} placeholder="gpt-4o" />
                </Field>
                <Field label="Vision Model">
                  <input value={llm.vision_model} onChange={e => setLlm({ ...llm, vision_model: e.target.value })} placeholder="留空使用 Chat Model" />
                </Field>
                <Field label="Max Tokens">
                  <input type="number" value={llm.max_tokens} onChange={e => setLlm({ ...llm, max_tokens: Number(e.target.value) })} />
                </Field>
                <Field label={`Temperature ${Number(llm.temperature).toFixed(1)}`}>
                  <input type="range" min="0" max="2" step="0.1" value={llm.temperature} onChange={e => setLlm({ ...llm, temperature: e.target.value })} />
                </Field>
              </div>
              <div className="set-actions">
                <button className="btn-secondary" onClick={testLlm}>测试连接</button>
                <button className="btn-primary" onClick={() => saveSection('llm_provider', llm)} disabled={saving}>{saving ? '保存中...' : '保存'}</button>
              </div>
            </div>
          </section>
        )}

        {/* ── Embedding ── */}
        {tab === 'embedding' && (
          <section className="set-card">
            <div className="set-card-hd"><h2>⊡ Embedding 模型</h2><small>向量嵌入服务的端点与模型</small></div>
            <div className="set-card-bd">
              <div className="set-grid">
                <Field label="Provider">
                  <input value={embedding.provider} onChange={e => setEmbedding({ ...embedding, provider: e.target.value })} placeholder="openai / local" />
                </Field>
                <Field label="Model">
                  <input value={embedding.model} onChange={e => setEmbedding({ ...embedding, model: e.target.value })} placeholder="text-embedding-3-small" />
                </Field>
                <Field label="Dimension">
                  <input type="number" value={embedding.dim} onChange={e => setEmbedding({ ...embedding, dim: Number(e.target.value) })} />
                </Field>
                <Field label="Batch Size">
                  <input type="number" value={embedding.batch_size} onChange={e => setEmbedding({ ...embedding, batch_size: Number(e.target.value) })} />
                </Field>
                <Field label="API Base URL">
                  <input value={embedding.api_base} onChange={e => setEmbedding({ ...embedding, api_base: e.target.value })} placeholder="留空复用 LLM Provider" />
                </Field>
                <Field label="API Key" help={embedding.has_api_key ? '已存在密钥' : ''}>
                  <SecretInput value={embedding.api_key || ''} onChange={e => setEmbedding({ ...embedding, api_key: e.target.value })} placeholder="留空复用 LLM Provider" hasKey={embedding.has_api_key} preview={embedding.api_key_preview} />
                </Field>
              </div>
              <div className="set-actions">
                <button className="btn-primary" onClick={() => saveSection('embedding', embedding)} disabled={saving}>{saving ? '保存中...' : '保存'}</button>
              </div>
            </div>
          </section>
        )}

        {/* ── Milvus ── */}
        {tab === 'milvus' && (
          <section className="set-card">
            <div className="set-card-hd"><h2>▥ Milvus 向量数据库</h2></div>
            <div className="set-card-bd">
              <div className="set-grid set-grid-3">
                <Field label="Host"><input value={milvus.host} onChange={e => setMilvus({ ...milvus, host: e.target.value })} /></Field>
                <Field label="Port"><input type="number" value={milvus.port} onChange={e => setMilvus({ ...milvus, port: Number(e.target.value) })} /></Field>
                <Field label="Dimension"><input type="number" value={milvus.dim} onChange={e => setMilvus({ ...milvus, dim: Number(e.target.value) })} /></Field>
              </div>
              <div className="set-actions">
                <button className="btn-primary" onClick={() => saveSection('milvus', milvus)} disabled={saving}>{saving ? '保存中...' : '保存'}</button>
              </div>
            </div>
          </section>
        )}

        {/* ── Voice ── */}
        {tab === 'voice' && (
          <section className="set-card">
            <div className="set-card-hd"><h2>♪ 语音 (STT / TTS)</h2></div>
            <div className="set-card-bd">
              <h3 className="set-subtitle">语音识别 (STT)</h3>
              <div className="set-grid">
                <Field label="Provider"><select value={voice.stt_provider} onChange={e => setVoice({ ...voice, stt_provider: e.target.value })}><option value="sensevoice">SenseVoice</option><option value="whisper">Whisper</option><option value="openai">OpenAI</option></select></Field>
                <Field label="Model"><input value={voice.stt_model} onChange={e => setVoice({ ...voice, stt_model: e.target.value })} /></Field>
                <Field label="API Base"><input value={voice.stt_api_base} onChange={e => setVoice({ ...voice, stt_api_base: e.target.value })} /></Field>
                <Field label="Language"><input value={voice.stt_language} onChange={e => setVoice({ ...voice, stt_language: e.target.value })} /></Field>
              </div>
              <h3 className="set-subtitle">语音合成 (TTS)</h3>
              <div className="set-grid">
                <Field label="Provider"><select value={voice.tts_provider} onChange={e => setVoice({ ...voice, tts_provider: e.target.value })}><option value="qwen">Qwen</option><option value="kokoro">Kokoro</option><option value="edge">Edge</option><option value="openai">OpenAI</option></select></Field>
                <Field label="Model"><input value={voice.tts_model} onChange={e => setVoice({ ...voice, tts_model: e.target.value })} /></Field>
                <Field label="API Base"><input value={voice.tts_api_base} onChange={e => setVoice({ ...voice, tts_api_base: e.target.value })} /></Field>
                <Field label="Language"><input value={voice.tts_language} onChange={e => setVoice({ ...voice, tts_language: e.target.value })} /></Field>
                <Field label="Voice"><input value={voice.tts_voice} onChange={e => setVoice({ ...voice, tts_voice: e.target.value })} /></Field>
                <Field label="Speed"><input type="number" step="0.1" value={voice.tts_speed} onChange={e => setVoice({ ...voice, tts_speed: Number(e.target.value) })} /></Field>
              </div>
              <div className="set-actions">
                <button className="btn-primary" onClick={() => saveSection('voice', voice)} disabled={saving}>{saving ? '保存中...' : '保存'}</button>
              </div>
            </div>
          </section>
        )}

        {/* ── OCR ── */}
        {tab === 'ocr' && (
          <section className="set-card">
            <div className="set-card-hd"><h2>☷ OCR</h2><small>PaddleOCR-VL 配置</small></div>
            <div className="set-card-bd">
              <div className="set-grid">
                <Field label="Enabled">
                  <label className="set-check">
                    <input type="checkbox" checked={ocr.enabled} onChange={e => setOcr({ ...ocr, enabled: e.target.checked })} />
                    <i />
                    <span>{ocr.enabled ? '已启用' : '已停用'}</span>
                  </label>
                </Field>
                <Field label="API Base"><input value={ocr.api_base} onChange={e => setOcr({ ...ocr, api_base: e.target.value })} /></Field>
                <Field label="Model"><input value={ocr.model} onChange={e => setOcr({ ...ocr, model: e.target.value })} /></Field>
                <Field label="Max Pages"><input type="number" value={ocr.max_pages} onChange={e => setOcr({ ...ocr, max_pages: Number(e.target.value) })} /></Field>
              </div>
              <div className="set-actions">
                <button className="btn-primary" onClick={() => saveSection('ocr', ocr)} disabled={saving}>{saving ? '保存中...' : '保存'}</button>
              </div>
            </div>
          </section>
        )}

        {/* ── Gateway ── */}
        {tab === 'gateway' && (
          <section className="set-card">
            <div className="set-card-hd"><h2>⇄ Gateway 消息网关</h2></div>
            <div className="set-card-bd">
              <div className="set-grid set-grid-3">
                <Field label="Gateway">
                  <label className="set-check">
                    <input type="checkbox" checked={gateway.enabled} onChange={e => setGateway({ ...gateway, enabled: e.target.checked })} />
                    <i /><span>{gateway.enabled ? '已启用' : '已停用'}</span>
                  </label>
                </Field>
                <Field label="响应模式"><select value={gateway.response_mode} onChange={e => setGateway({ ...gateway, response_mode: e.target.value })}><option value="sync">Sync</option><option value="async">Async</option></select></Field>
                <Field label="最大回复长度"><input type="number" value={gateway.max_reply_length} onChange={e => setGateway({ ...gateway, max_reply_length: Number(e.target.value) })} /></Field>
              </div>

              <h3 className="set-subtitle">企业微信</h3>
              <div className="set-grid">
                <Field label="启用">
                  <label className="set-check">
                    <input type="checkbox" checked={gateway.wechat_work?.enabled || false} onChange={e => setGateway({ ...gateway, wechat_work: { ...gateway.wechat_work, enabled: e.target.checked } })} />
                    <i /><span>{gateway.wechat_work?.enabled ? '已启用' : '已停用'}</span>
                  </label>
                </Field>
                <Field label="Corp ID"><input value={gateway.wechat_work?.corp_id || ''} onChange={e => setGateway({ ...gateway, wechat_work: { ...gateway.wechat_work, corp_id: e.target.value } })} /></Field>
                <Field label="Token"><input value={gateway.wechat_work?.token || ''} onChange={e => setGateway({ ...gateway, wechat_work: { ...gateway.wechat_work, token: e.target.value } })} /></Field>
                <Field label="Encoding AES Key"><input value={gateway.wechat_work?.encoding_aes_key || ''} onChange={e => setGateway({ ...gateway, wechat_work: { ...gateway.wechat_work, encoding_aes_key: e.target.value } })} /></Field>
                <Field label="Agent ID"><input value={gateway.wechat_work?.agent_id || ''} onChange={e => setGateway({ ...gateway, wechat_work: { ...gateway.wechat_work, agent_id: e.target.value } })} /></Field>
                <Field label="Secret"><input type="password" value={gateway.wechat_work?.secret || ''} onChange={e => setGateway({ ...gateway, wechat_work: { ...gateway.wechat_work, secret: e.target.value } })} autoComplete="new-password" /></Field>
              </div>

              <h3 className="set-subtitle">钉钉</h3>
              <div className="set-grid set-grid-3">
                <Field label="启用">
                  <label className="set-check">
                    <input type="checkbox" checked={gateway.dingtalk?.enabled || false} onChange={e => setGateway({ ...gateway, dingtalk: { ...gateway.dingtalk, enabled: e.target.checked } })} />
                    <i /><span>{gateway.dingtalk?.enabled ? '已启用' : '已停用'}</span>
                  </label>
                </Field>
                <Field label="App Key"><input value={gateway.dingtalk?.app_key || ''} onChange={e => setGateway({ ...gateway, dingtalk: { ...gateway.dingtalk, app_key: e.target.value } })} /></Field>
                <Field label="App Secret"><input type="password" value={gateway.dingtalk?.app_secret || ''} onChange={e => setGateway({ ...gateway, dingtalk: { ...gateway.dingtalk, app_secret: e.target.value } })} autoComplete="new-password" /></Field>
              </div>

              <h3 className="set-subtitle">QQ Bot</h3>
              <div className="set-grid">
                <Field label="启用">
                  <label className="set-check">
                    <input type="checkbox" checked={gateway.qqbot?.enabled || false} onChange={e => setGateway({ ...gateway, qqbot: { ...gateway.qqbot, enabled: e.target.checked } })} />
                    <i /><span>{gateway.qqbot?.enabled ? '已启用' : '已停用'}</span>
                  </label>
                </Field>
                <Field label="App ID"><input value={gateway.qqbot?.app_id || ''} onChange={e => setGateway({ ...gateway, qqbot: { ...gateway.qqbot, app_id: e.target.value } })} /></Field>
                <Field label="App Secret"><input type="password" value={gateway.qqbot?.app_secret || ''} onChange={e => setGateway({ ...gateway, qqbot: { ...gateway.qqbot, app_secret: e.target.value } })} autoComplete="new-password" /></Field>
                <Field label="沙箱模式">
                  <label className="set-check">
                    <input type="checkbox" checked={gateway.qqbot?.sandbox ?? true} onChange={e => setGateway({ ...gateway, qqbot: { ...gateway.qqbot, sandbox: e.target.checked } })} />
                    <i /><span>{gateway.qqbot?.sandbox ? '沙箱' : '正式'}</span>
                  </label>
                </Field>
              </div>

              <div className="set-actions">
                <button className="btn-primary" onClick={() => saveSection('gateway', gateway)} disabled={saving}>{saving ? '保存中...' : '保存'}</button>
              </div>
            </div>
          </section>
        )}

        {/* ── MCP ── */}
        {tab === 'mcp' && (
          <section className="set-card">
            <div className="set-card-hd">
              <h2>⌘ MCP Servers</h2>
              <button className="btn-secondary btn-sm" onClick={() => openMcp('')}>＋ 添加</button>
            </div>
            <div className="set-card-bd">
              {mcpList.length === 0 && <div className="set-empty">尚未配置 MCP Server</div>}
              <div className="set-mcp-list">
                {mcpList.map(([name, s]) => (
                  <button key={name} className="set-mcp-row" onClick={() => openMcp(name)}>
                    <span className="set-mcp-icon">⌘</span>
                    <span className="set-mcp-info"><strong>{name}</strong><small>{[s.command, ...(Array.isArray(s.args) ? s.args : [])].join(' ')}</small></span>
                    <span className={`set-badge ${s.disabled ? 'off' : 'on'}`}>{s.disabled ? '停用' : '启用'}</span>
                    <span className="set-mcp-arrow">›</span>
                  </button>
                ))}
              </div>
              <div className="set-actions">
                <button className="btn-primary" onClick={saveMcpServer} disabled={saving}>{saving ? '保存中...' : '保存 MCP 配置'}</button>
              </div>
            </div>
          </section>
        )}

        {/* ── Connection ── */}
        {tab === 'connection' && (
          <section className="set-card">
            <div className="set-card-hd"><h2>⌁ 后端连接</h2><small>前端连接 Agentic RAG 服务的地址</small></div>
            <div className="set-card-bd">
              <Field label="Web API Base URL" help="修改后需刷新页面">
                <input value={apiBase} onChange={e => setApiBase(e.target.value)} placeholder="留空自动使用当前地址" />
              </Field>
              <div className="set-grid set-grid-2">
                <Field label="API Host"><input value={apiServer.host} onChange={e => setApiServer({ ...apiServer, host: e.target.value })} /></Field>
                <Field label="API Port"><input type="number" value={apiServer.port} onChange={e => setApiServer({ ...apiServer, port: Number(e.target.value) })} /></Field>
              </div>
              <div className="set-actions">
                <button className="btn-secondary" onClick={saveConnection}>保存前端地址</button>
                <button className="btn-primary" onClick={() => saveSection('api', apiServer)} disabled={saving}>{saving ? '保存中...' : '保存服务端地址'}</button>
              </div>
            </div>
          </section>
        )}

        {/* ── MCP Modal ── */}
        {mcpModal && (
          <div className="set-overlay" onClick={e => e.target === e.currentTarget && setMcpModal(null)}>
            <div className="set-modal">
              <div className="set-modal-hd">
                <span>{mcpModal.isNew ? '新建 MCP Server' : `编辑: ${mcpModal.name}`}</span>
                <button onClick={() => setMcpModal(null)}>✕</button>
              </div>
              <div className="set-modal-bd">
                <Field label="Name"><input value={mcpModal.name} onChange={e => setMcpModal({ ...mcpModal, name: e.target.value })} disabled={!mcpModal.isNew} /></Field>
                <Field label="Command"><input value={mcpModal.command} onChange={e => setMcpModal({ ...mcpModal, command: e.target.value })} /></Field>
                <Field label="Args (空格分隔)"><input value={mcpModal.args} onChange={e => setMcpModal({ ...mcpModal, args: e.target.value })} /></Field>
                <Field label="Env (KEY=VALUE)"><textarea rows={3} value={mcpModal.env} onChange={e => setMcpModal({ ...mcpModal, env: e.target.value })} /></Field>
                <label className="set-check"><input type="checkbox" checked={mcpModal.disabled} onChange={e => setMcpModal({ ...mcpModal, disabled: e.target.checked })} /><i /><span>停用</span></label>
              </div>
              <div className="set-modal-ft">
                {!mcpModal.isNew && <button className="btn-danger set-btn-inline" onClick={deleteMcp}>删除</button>}
                <span style={{ flex: 1 }} />
                <button className="btn-secondary" onClick={() => setMcpModal(null)}>取消</button>
                <button className="btn-primary" onClick={saveMcpLocal}>保存</button>
              </div>
            </div>
          </div>
        )}
      </div>

      {toast && <Toast key={toast.key} text={toast.text} type={toast.type} onDone={() => setToast(null)} />}
    </div>
  );
}
