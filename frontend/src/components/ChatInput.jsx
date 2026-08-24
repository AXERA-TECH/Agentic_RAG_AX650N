import { useState, useRef } from 'react';
import { api } from '../api';

export default function ChatInput({ onSend, streaming, onStop, ensureSession }) {
  const [text, setText] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileRef = useRef(null);

  const send = () => {
    if (processing || (!text.trim() && !attachments.length)) return;
    onSend(text, attachments.map(f => f.dataUri).filter(Boolean));
    setText(''); setAttachments([]);
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const attachFiles = (files) => {
    for (const f of files) {
      const reader = new FileReader();
      reader.onload = () => setAttachments(prev => [...prev, { name: f.name, dataUri: reader.result }]);
      reader.readAsDataURL(f);
    }
  };

  // ── Voice: record → SSE stream (transcript → answer → audio) ──

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus' : 'audio/webm';
      const mr = new MediaRecorder(stream, { mimeType: mime });
      audioChunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = () => { stream.getTracks().forEach(t => t.stop()); processVoice(); };
      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch (e) { alert('Microphone denied: ' + e.message); }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') mediaRecorderRef.current.stop();
    setRecording(false);
  };

  const processVoice = async () => {
    if (!audioChunksRef.current.length) return;
    setProcessing(true);
    const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

    try {
      const sid = await ensureSession?.() || '';
      const abort = new AbortController();

      // Dispatch events to ChatPanel via window for progressive UI updates
      let transcriptText = '';
      let answerText = '';

      window.dispatchEvent(new CustomEvent('voice-start'));

      await api.voiceChatStream(blob, sid, true, ({ type, data }) => {
        switch (type) {
          case 'transcript':
            transcriptText = data.text || '';
            window.dispatchEvent(new CustomEvent('voice-transcript', { detail: { text: transcriptText } }));
            break;
          case 'text_delta':
            answerText += data.content || '';
            window.dispatchEvent(new CustomEvent('voice-delta', { detail: { content: data.content || '' } }));
            break;
          case 'tool_call_start':
            window.dispatchEvent(new CustomEvent('voice-tool', { detail: { tool: data.tool || '' } }));
            break;
          case 'final_answer':
            answerText = data.final_answer || answerText;
            window.dispatchEvent(new CustomEvent('voice-final', {
              detail: { answer: answerText },
            }));
            break;
          case 'audio':
            if (data.audio) {
              window.dispatchEvent(new CustomEvent('voice-audio', { detail: { audio: data.audio, format: data.format || 'wav' } }));
            }
            break;
          case 'done':
            window.dispatchEvent(new CustomEvent('voice-done', {
              detail: {
                answer: data.final_answer || answerText,
                transcript: transcriptText,
              },
            }));
            break;
          case 'error':
            window.dispatchEvent(new CustomEvent('voice-error', { detail: { error: data.error || 'Voice error' } }));
            break;
        }
      }, abort.signal);

    } catch (e) {
      if (e.name !== 'AbortError') {
        window.dispatchEvent(new CustomEvent('voice-error', { detail: { error: e.message } }));
      }
    } finally {
      setProcessing(false);
    }
  };

  const toggleMic = () => {
    if (recording) stopRecording();
    else if (!processing) startRecording();
  };

  return (
    <div className="chat-input-area">
      {recording && (
        <div className="recording-bar">
          <span className="recording-pulse" />
          <span>Recording... click 🎤 again to stop</span>
        </div>
      )}
      {processing && (
        <div className="recording-bar processing">
          <span className="loading-dots"><span>.</span><span>.</span><span>.</span></span>
          <span>Processing voice...</span>
        </div>
      )}
      {attachments.length > 0 && (
        <div className="attach-chips-row">
          {attachments.map((f, i) => (
            <span key={i} className="attach-chip">{f.name} <span className="remove-chip" onClick={() => setAttachments(prev => prev.filter((_, j) => j !== i))}>×</span></span>
          ))}
        </div>
      )}
      <div className="chat-input-row">
        <button className="input-icon-btn" onClick={() => fileRef.current?.click()} title="Attach">📎</button>
        <input ref={fileRef} type="file" hidden multiple accept="image/*" onChange={e => { attachFiles(e.target.files); e.target.value = ''; }} />
        <textarea rows="1" value={text} onChange={e => setText(e.target.value)}
                  onKeyDown={handleKey} placeholder="How can I help you today?" />
        <button className={`input-icon-btn ${recording ? 'recording' : ''}`}
                onClick={toggleMic} disabled={processing}
                title={recording ? 'Stop' : 'Voice'}>
          {processing ? '⏳' : recording ? '🔴' : '🎤'}
        </button>
        {streaming
          ? <button className="send-btn stop" onClick={onStop}>■</button>
          : <button className="send-btn" onClick={send} disabled={processing || (!text.trim() && !attachments.length)}>↑</button>}
      </div>
    </div>
  );
}
