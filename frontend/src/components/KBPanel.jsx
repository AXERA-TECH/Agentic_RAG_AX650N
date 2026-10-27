import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../api';

export default function KBPanel() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [docs, setDocs] = useState(0);
  const [chunks, setChunks] = useState(0);
  const [modality, setModality] = useState('all');
  const [topK, setTopK] = useState(3);
  const [threshold, setThreshold] = useState(0.35);
  const [mode, setMode] = useState('hybrid');
  const [ingestMode, setIngestMode] = useState('multimodal');
  const [mmMethod, setMmMethod] = useState('pure');
  const [chunkSize, setChunkSize] = useState(800);
  const [chunkOverlap, setChunkOverlap] = useState(150);
  const [enableKG, setEnableKG] = useState(false);
  const [preview, setPreview] = useState(null);
  const [previewTab, setPreviewTab] = useState('results');
  const [indexedFiles, setIndexedFiles] = useState([]);
  const [fileFilter, setFileFilter] = useState('');
  const [uploadFiles, setUploadFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [ingestSource, setIngestSource] = useState('upload');
  const [ingestText, setIngestText] = useState('');
  const [textSource, setTextSource] = useState('web_ui');
  const [ingestTab, setIngestTab] = useState('upload');
  const fileInputRef = useRef(null);
  const dropRef = useRef(null);

  const refreshStats = useCallback(async () => {
    try { const s = await api.stats(); setDocs(s.documents || 0); setChunks(s.chunks || 0); } catch {}
  }, []);

  useEffect(() => { refreshStats(); }, [refreshStats]);

  // Load indexed files from backend (cross-browser persistent)
  useEffect(() => {
    api.listDocuments().then(data => {
      const docs = (data.documents || []).map(d => ({
        name: d.name, size: d.size, type: d.type,
        time: d.time, docId: d.doc_id,
      }));
      if (docs.length) setIndexedFiles(docs);
    }).catch(() => {});
  }, []);

  const search = async () => {
    if (!query.trim()) return;
    try {
      const data = await api.search(query, topK, mode, modality);
      setResults(data.results || []);
      setPreviewTab('results');
    } catch (e) { alert('Search failed: ' + e.message); }
  };

  const uploadFile = async (file) => {
    try {
      const data = await api.uploadFile(file, ingestSource, ingestMode, mmMethod, chunkSize, chunkOverlap, enableKG);
      // Reload from server to get authoritative file list
      const docs = await api.listDocuments().then(r => (r.documents || []).map(d => ({
        name: d.name, size: d.size, type: d.type, time: d.time, docId: d.doc_id,
      }))).catch(() => null);
      if (docs) setIndexedFiles(docs);
      refreshStats();
      return true;
    } catch (e) { alert('Upload failed: ' + e.message); return false; }
  };

  const uploadAll = async () => {
    setUploading(true);
    let ok = 0, fail = 0;
    for (const f of uploadFiles) { (await uploadFile(f)) ? ok++ : fail++; }
    setUploadFiles([]);
    setUploading(false);
    if (ok) alert(`${ok} uploaded${fail ? ', ' + fail + ' failed' : ''}`);
  };

  const clearKB = async () => {
    if (!confirm('Delete ALL documents?')) return;
    await api.clearKB();
    setIndexedFiles([]);
    refreshStats(); setResults([]);
  };

  const deleteDoc = async (f, e) => {
    e?.stopPropagation();
    if (f.docId) await api.deleteDocument(f.docId).catch(() => {});
    // Reload from server
    const docs = await api.listDocuments().then(r => (r.documents || []).map(d => ({
      name: d.name, size: d.size, type: d.type, time: d.time, docId: d.doc_id,
    }))).catch(() => null);
    if (docs) setIndexedFiles(docs);
    refreshStats();
  };

  const previewFile = async (f) => {
    if (!f.docId) return;
    try {
      const data = await api.getDocument(f.docId);
      setPreview({ name: f.name, items: data.items || [] });
      setPreviewTab('preview');
    } catch { setPreview(null); }
  };

  const ingestTextContent = async () => {
    if (!ingestText.trim()) return;
    try {
      const response = await fetch('/api/v1/rag/ingest', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: ingestText,
          source: textSource,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
        }),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }
      // Reload from server
      const docs = await api.listDocuments().then(r => (r.documents || []).map(d => ({
        name: d.name, size: d.size, type: d.type, time: d.time, docId: d.doc_id,
      }))).catch(() => null);
      if (docs) setIndexedFiles(docs);
      setIngestText('');
      refreshStats();
    } catch (e) { alert('Ingest failed: ' + e.message); }
  };

  const handleDrop = (e) => { e.preventDefault(); [...e.dataTransfer.files].forEach(f => setUploadFiles(prev => [...prev, f])); };

  const filteredFiles = fileFilter ? indexedFiles.filter(f => f.name.toLowerCase().includes(fileFilter.toLowerCase())) : indexedFiles;

  return (
    <div className="kb-layout">
      {/* Left Sidebar: Sources + Ingest */}
      <div className="kb-sidebar">
        <div className="kb-section-label">📂 Sources</div>
        <div className="kb-doc-search"><input placeholder="Filter sources…" value={fileFilter} onChange={e => setFileFilter(e.target.value)} /></div>
        <div className="kb-doc-list">
          {filteredFiles.map((f, i) => (
            <div key={i} className="kb-doc-item" onClick={() => previewFile(f)}>
              <span className="kb-doc-icon">{['jpg','jpeg','png','gif','webp'].includes(f.name?.split('.').pop()?.toLowerCase()) ? '🖼' : f.name?.endsWith('.mp4') ? '🎬' : '📝'}</span>
              <div className="kb-doc-info"><div className="kb-doc-name">{f.name}</div><div className="kb-doc-meta">{f.type} · {(f.size||0) > 999 ? ((f.size||0)/1024).toFixed(1)+'KB' : (f.size||0)+'B'}</div></div>
              <span className="kb-doc-remove" onClick={e => deleteDoc(f, e)}>×</span>
            </div>
          ))}
          {!filteredFiles.length && <div className="kb-empty-state">No files indexed yet</div>}
        </div>

        <div className="ingest-section">
          <div className="ingest-tabs">
            <button className={`ingest-tab ${ingestTab === 'text' ? 'active' : ''}`} onClick={() => setIngestTab('text')}>📝 Text</button>
            <button className={`ingest-tab ${ingestTab === 'upload' ? 'active' : ''}`} onClick={() => setIngestTab('upload')}>📤 Upload</button>
          </div>
          {ingestTab === 'text' ? (
            <div className="ingest-panel">
              <textarea className="ingest-textarea" rows={6} value={ingestText} onChange={e => setIngestText(e.target.value)} placeholder="Paste text here…" style={{width:'100%',resize:'vertical',fontSize:13}} />
              <div style={{display:'flex',gap:8,marginTop:8}}>
                <input value={textSource} onChange={e => setTextSource(e.target.value)} placeholder="Source" style={{flex:1}} />
                <button className="btn-primary" onClick={ingestTextContent}>Ingest</button>
              </div>
            </div>
          ) : (
            <div className="ingest-panel">
              <div className="drop-zone" ref={dropRef} onClick={() => fileInputRef.current?.click()} onDragOver={e => e.preventDefault()} onDrop={handleDrop}>
                <div>📤 Drop files or click to browse</div>
                <div style={{fontSize:11,color:'var(--text-tertiary)',marginTop:4}}>TXT · MD · PDF · JPG · PNG · MP4 · MP3…</div>
              </div>
              <input ref={fileInputRef} type="file" hidden multiple accept=".txt,.md,.markdown,.pdf,.json,.yaml,.yml,.csv,.py,.html,.jpg,.jpeg,.png,.gif,.webp,.bmp,.svg,.mp4,.avi,.mov,.mkv,.webm,.mp3,.wav,.m4a,.ogg,.flac" onChange={e => { [...e.target.files].forEach(f => setUploadFiles(prev => [...prev, f])); e.target.value = ''; }} />
              {uploadFiles.length > 0 && (
                <div className="file-queue" style={{marginTop:8}}>
                  {uploadFiles.map((f, i) => (
                    <div key={i} className="file-queue-item">
                      <span>{f.name} ({(f.size > 999 ? (f.size/1024).toFixed(1)+'KB' : f.size+'B')})</span>
                      <span style={{cursor:'pointer',color:'var(--red)'}} onClick={() => setUploadFiles(prev => prev.filter((_, j) => j !== i))}>×</span>
                    </div>
                  ))}
                </div>
              )}
              <div style={{display:'flex',gap:8,marginTop:8}}>
                <input value={ingestSource} onChange={e => setIngestSource(e.target.value)} placeholder="Source" style={{flex:1}} />
                <button className="btn-primary" onClick={uploadAll} disabled={uploading || !uploadFiles.length}>{uploading ? 'Uploading…' : 'Upload All'}</button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main: Search + Results + Preview */}
      <div className="kb-main">
        <div className="kb-search-area">
          <input placeholder="Search knowledge base…" value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && search()} style={{flex:1}} />
          <button className="btn-primary" onClick={search}>Search</button>
        </div>
        <div className="kb-view-tabs">
          <button className={`kb-view-tab ${previewTab === 'results' ? 'active' : ''}`} onClick={() => setPreviewTab('results')}>🔍 Results</button>
          <button className={`kb-view-tab ${previewTab === 'preview' ? 'active' : ''}`} onClick={() => setPreviewTab('preview')}>📄 Preview</button>
        </div>
        {previewTab === 'results' ? (
          <div className="kb-results">
            {results.length > 0 && <div style={{fontSize:12,color:'var(--text-tertiary)',marginBottom:8}}>{results.length} results · {mode} · top-{topK}</div>}
            {results.filter(r => !threshold || (r.score || 0) >= threshold).map((r, i) => (
              <div key={i} className="kb-result-item" onClick={() => {
                const imgPath = r.image_path || '';
                const vidPath = r.video_path || '';
                const audPath = r.audio_path || '';
                setPreview({ name: r.document || 'Result', items: [{ type: r.content_type, text: r.content, image_path: imgPath, video_path: vidPath, audio_path: audPath }] });
                setPreviewTab('preview');
              }}>
                <div className="kb-result-header">
                  <span className={`kb-result-type ${r.content_type}`}>{r.content_type}</span>
                  <span className="kb-result-score">{(r.score || 0).toFixed(3)}</span>
                  <span style={{fontSize:11,color:'var(--text-tertiary)',marginLeft:'auto'}}>{r.document || ''}</span>
                </div>
                <div className="kb-result-text">{(r.content || '').slice(0, 300)}</div>
              </div>
            ))}
            {!results.length && <div className="kb-empty"><div>🔍</div><div>Search across text, images, video, audio</div></div>}
          </div>
        ) : (
          <div className="kb-preview-body">
            {preview ? (
              <>
                <h4 style={{marginBottom:12}}>{preview.name}</h4>
                {(preview.items || []).map((item, i) => {
                    const imgSrc = item.type === 'image' && item.image_path ? api.serveFile(item.image_path) : null;
                    const vidSrc = item.type === 'video' && item.video_path ? api.serveFile(item.video_path) : null;
                    const audSrc = item.type === 'audio' && item.audio_path ? api.serveFile(item.audio_path) : null;
                    return (
                      <div key={i} style={{marginBottom:12, borderBottom:'1px solid var(--border-subtle)', paddingBottom:12}}>
                        {imgSrc ? <img src={imgSrc} style={{maxWidth:'100%',maxHeight:360,borderRadius:6}} alt="" /> :
                         vidSrc ? <video controls style={{maxWidth:'100%',maxHeight:360}} src={vidSrc} /> :
                         audSrc ? <audio controls src={audSrc} /> :
                         <pre style={{whiteSpace:'pre-wrap',fontSize:12,color:'var(--text-secondary)'}}>{item.text?.slice(0, 500)}</pre>}
                      </div>
                    );
                  })}
              </>
            ) : <div className="kb-empty"><div>📄</div><div>Click a result to preview</div></div>}
          </div>
        )}
      </div>

      {/* Right: Stats + Settings */}
      <div className="kb-settings-sidebar">
        <div className="kb-stats">
          <div className="kb-stat"><div className="kb-stat-value">{docs}</div><div className="kb-stat-label">Docs</div></div>
          <div className="kb-stat"><div className="kb-stat-value">{chunks}</div><div className="kb-stat-label">Chunks</div></div>
        </div>
        <div className="kb-setting-group"><button className="btn-danger" onClick={clearKB}>🗑 Clear All</button></div>
        <div className="kb-setting-group">
          <div className="kb-setting-group-title">📥 Ingest Mode</div>
          <select value={ingestMode} onChange={e => setIngestMode(e.target.value)}><option value="multimodal">🎨 Multimodal</option><option value="text">📝 Text Only</option></select>
        </div>
        {ingestMode !== 'text' && (
          <div className="kb-setting-group">
            <div className="kb-setting-group-title">🔄 Embed Method</div>
            <select value={mmMethod} onChange={e => setMmMethod(e.target.value)}><option value="pure">🖼️ Pure Multimodal</option><option value="caption">📝 Caption → Text</option><option value="both">🔀 Both</option></select>
          </div>
        )}
        <div className="kb-setting-group">
          <div className="kb-setting-group-title">✂️ Chunking</div>
          <div className="kb-setting-row"><label>Chunk Size</label><input type="number" value={chunkSize} onChange={e => { const next = +e.target.value; setChunkSize(next); setChunkOverlap(prev => Math.min(prev, Math.max(0, next - 1))); }} min={128} max={4096} step={64} /></div>
          <div className="kb-setting-row"><label>Overlap</label><input type="number" value={chunkOverlap} onChange={e => setChunkOverlap(+e.target.value)} min={0} max={Math.max(0, chunkSize - 1)} step={16} /></div>
        </div>
        <div className="kb-setting-group">
          <div className="kb-setting-group-title">🕸️ Knowledge Graph</div>
          <div className="kb-setting-row">
            <label>Enable KG</label>
            <input type="checkbox" checked={enableKG} onChange={e => setEnableKG(e.target.checked)} />
          </div>
        </div>
        <div className="kb-setting-group">
          <div className="kb-setting-group-title">📋 Modality Filter</div>
          <select value={modality} onChange={e => setModality(e.target.value)}><option value="all">All</option><option value="text">📝 Text</option><option value="image">🖼️ Image</option><option value="video">🎬 Video</option><option value="audio">🎵 Audio</option></select>
        </div>
        <div className="kb-setting-group">
          <div className="kb-setting-group-title">🔍 Retrieval</div>
          <div className="kb-setting-row"><label>Top-K</label><input type="number" value={topK} onChange={e => setTopK(+e.target.value)} min={1} max={50} /></div>
          <div className="kb-setting-row"><label>Threshold</label><input type="number" value={threshold} onChange={e => setThreshold(+e.target.value)} min={0} max={1} step={0.05} /></div>
          <div className="kb-setting-row"><label>Mode</label><select value={mode} onChange={e => setMode(e.target.value)}><option value="hybrid">Hybrid</option><option value="naive">Vector</option></select></div>
        </div>
      </div>
    </div>
  );
}

function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
