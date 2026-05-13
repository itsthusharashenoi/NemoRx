import { useCallback, useEffect, useRef, useState } from 'react';
import { LANGUAGES, GEMINI_LANGUAGES } from './languages.js';
import { VEXYL_SAMPLE_RATE, downsample, floatToInt16PCM } from './vexylAudio.js';
import { fetchSttHealth, transcribeWithGemini } from './geminiClient.js';
import './App.css';

const WS_URL = import.meta.env.VITE_VEXYL_WS_URL || 'ws://127.0.0.1:8091';

/** Online live: send audio slices to Gemini this often (ms). */
const GEMINI_LIVE_SLICE_MS = 5000;

function httpBaseFromWs(wsUrl) {
  try {
    const normalized = wsUrl.replace(/^ws:\/\//i, 'http://').replace(/^wss:\/\//i, 'https://');
    const u = new URL(normalized);
    return `${u.protocol}//${u.host}`;
  } catch {
    return 'http://127.0.0.1:8091';
  }
}

const HTTP_BASE = import.meta.env.VITE_VEXYL_HTTP_URL || httpBaseFromWs(WS_URL);
const STT_API_KEY = import.meta.env.VITE_VEXYL_STT_API_KEY || '';

function downloadBlob(blob, filename) {
  const a = document.createElement('a');
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function pickRecorderMime() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
  ];
  for (const t of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return '';
}

function timestampForFile() {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
}

export default function App() {
  const wsRef = useRef(null);
  const sessionIdRef = useRef(0);
  const isRecordingRef = useRef(false);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const transcriptBufferRef = useRef([]);
  const stopResolverRef = useRef(null);
  const sttProviderRef = useRef('local');

  const [serverState, setServerState] = useState('disconnected');
  const [modelName, setModelName] = useState('');
  const [lang, setLang] = useState('hi-IN');
  const [recording, setRecording] = useState(false);
  const [lines, setLines] = useState([]);
  const [logLines, setLogLines] = useState([]);
  const [mode, setMode] = useState('live'); // 'live' | 'conversation'
  const [conversationProcessing, setConversationProcessing] = useState(false);
  const [conversationResult, setConversationResult] = useState(null);
  const [sttProvider, setSttProvider] = useState('local'); // 'local' | 'gemini'
  const [healthSnapshot, setHealthSnapshot] = useState(null);

  useEffect(() => {
    sttProviderRef.current = sttProvider;
  }, [sttProvider]);

  const prevSttProviderRef = useRef('local');
  useEffect(() => {
    if (sttProvider === 'local' && lang === 'auto') {
      setLang('hi-IN');
    }
    if (sttProvider === 'gemini' && prevSttProviderRef.current === 'local') {
      setLang('auto');
    }
    prevSttProviderRef.current = sttProvider;
  }, [sttProvider, lang]);

  const langOptions = sttProvider === 'gemini' ? GEMINI_LANGUAGES : LANGUAGES;

  const log = useCallback((msg) => {
    const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
    setLogLines((prev) => [...prev.slice(-40), line]);
    console.log(msg);
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function pollHealth() {
      try {
        const h = await fetchSttHealth(HTTP_BASE);
        if (!cancelled) setHealthSnapshot(h);
      } catch {
        if (!cancelled) setHealthSnapshot(null);
      }
    }
    pollHealth();
    const intervalMs = sttProvider === 'gemini' ? 3000 : 8000;
    const id = setInterval(pollHealth, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [HTTP_BASE, sttProvider]);

  useEffect(() => {
    let ws;
    let closed = false;

    function connect() {
      setServerState('connecting');
      log(`Connecting to ${WS_URL}`);
      ws = new WebSocket(WS_URL);

      ws.onmessage = (e) => {
        let msg;
        try {
          msg = JSON.parse(e.data);
        } catch {
          return;
        }

        if (msg.type === 'ready') {
          setServerState('ready');
          setModelName(msg.model || 'vexyl-stt');
          log('Server ready');
        } else if (msg.type === 'final' && msg.text) {
          transcriptBufferRef.current.push(msg.text.trim());
          setLines((prev) => [...prev, { text: msg.text.trim(), lang: msg.lang }]);
        } else if (msg.type === 'stopped') {
          log('Session stopped');
          stopResolverRef.current?.();
          stopResolverRef.current = null;
        } else if (msg.type === 'error') {
          log(`Server error: ${msg.message || JSON.stringify(msg)}`);
        }
      };

      ws.onopen = () => {
        if (closed) return;
        log('WebSocket open');
      };

      ws.onerror = () => {
        if (!closed) log('WebSocket error');
      };

      ws.onclose = () => {
        if (closed) return;
        setServerState('disconnected');
        setModelName('');
        log('WebSocket closed');
      };
    }

    connect();
    wsRef.current = ws;

    return () => {
      closed = true;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
      wsRef.current = null;
    };
  }, [log]);

  const waitForStopped = useCallback(
    () =>
      new Promise((resolve) => {
        const t = setTimeout(() => {
          stopResolverRef.current = null;
          resolve();
        }, 1500);
        stopResolverRef.current = () => {
          clearTimeout(t);
          resolve();
        };
      }),
    []
  );

  /** STT HTTP server is reachable (GET /health succeeded with status ok). */
  const geminiHttpOk = healthSnapshot?.status === 'ok';
  /** Server process has GEMINI_API_KEY / GOOGLE_API_KEY loaded. */
  const geminiConfigured = healthSnapshot?.gemini_online === true;
  const localReady = serverState === 'ready';

  const stopConversationRecording = useCallback(async () => {
    isRecordingRef.current = false;
    setRecording(false);

    const stream = mediaStreamRef.current;
    const rec = mediaRecorderRef.current;

    mediaStreamRef.current = null;
    mediaRecorderRef.current = null;

    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }

    let audioBlob = null;
    let ext = 'webm';
    if (rec && rec.state !== 'inactive') {
      await new Promise((resolve) => {
        rec.onstop = resolve;
        rec.stop();
      });
      const mime = rec.mimeType || 'audio/webm';
      if (mime.includes('mp4')) ext = 'm4a';
      audioBlob = new Blob(recordedChunksRef.current, { type: mime });
    }
    recordedChunksRef.current = [];

    const ts = timestampForFile();

    if (!audioBlob || audioBlob.size === 0) {
      log('No audio captured');
      return;
    }

    downloadBlob(audioBlob, `conversation-raw-${ts}.${ext}`);

    setConversationProcessing(true);
    setConversationResult(null);

    const useGemini = sttProviderRef.current === 'gemini';
    log(
      useGemini
        ? 'Uploading to Gemini (conversation document)…'
        : 'Uploading for Doc / Patient / Voice segmentation…'
    );

    try {
      let data;
      if (useGemini) {
        data = await transcribeWithGemini(HTTP_BASE, audioBlob, `session.${ext}`, {
          languageCode: lang,
          promptType: 'conversation_doc',
          sttApiKey: STT_API_KEY,
        });
      } else {
        const form = new FormData();
        form.append('file', audioBlob, `session.${ext}`);
        form.append('language_code', lang);
        const headers = {};
        if (STT_API_KEY) headers['X-API-Key'] = STT_API_KEY;
        const res = await fetch(`${HTTP_BASE}/conversation/transcribe`, {
          method: 'POST',
          headers,
          body: form,
        });
        data = await res.json().catch(() => ({}));
        if (!res.ok) {
          log(`Conversation API error: ${res.status} ${data.error || JSON.stringify(data)}`);
          setConversationProcessing(false);
          return;
        }
      }

      setConversationResult(data);
      log(`Document ready (${(data.segments || []).length} segments)`);

      const doc = (data.document || '').trim();
      if (doc) {
        downloadBlob(
          new Blob([doc + '\n'], { type: 'text/plain;charset=utf-8' }),
          `conversation-doc-${ts}.txt`
        );
      }
    } catch (err) {
      log(`Request failed: ${err.message}`);
    } finally {
      setConversationProcessing(false);
    }
  }, [lang, log]);

  const stopLiveGeminiRecording = useCallback(async () => {
    isRecordingRef.current = false;
    setRecording(false);

    const stream = mediaStreamRef.current;
    const rec = mediaRecorderRef.current;

    mediaStreamRef.current = null;
    mediaRecorderRef.current = null;

    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }

    let audioBlob = null;
    let ext = 'webm';
    if (rec && rec.state !== 'inactive') {
      await new Promise((resolve) => {
        rec.onstop = resolve;
        rec.stop();
      });
      const mime = rec.mimeType || 'audio/webm';
      if (mime.includes('mp4')) ext = 'm4a';
      audioBlob = new Blob(recordedChunksRef.current, { type: mime });
    }
    recordedChunksRef.current = [];

    const parts = transcriptBufferRef.current.filter(Boolean);
    const fullText = parts.join(' ').trim();
    const ts = timestampForFile();

    if (fullText) {
      const txtBlob = new Blob([fullText + '\n'], { type: 'text/plain;charset=utf-8' });
      downloadBlob(txtBlob, `transcript-gemini-${ts}.txt`);
    }

    if (audioBlob && audioBlob.size > 0) {
      downloadBlob(audioBlob, `recording-gemini-${ts}.${ext}`);
    }

    if (!fullText && (!audioBlob || audioBlob.size === 0)) {
      log('Nothing to save (no audio / no transcript)');
    } else {
      const saved = [fullText && 'transcript', audioBlob?.size && 'recording'].filter(Boolean);
      log(`Saved (Gemini): ${saved.join(' + ')}`);
    }

    transcriptBufferRef.current = [];
  }, [log]);

  const stopLiveRecording = useCallback(async () => {
    if (sttProviderRef.current === 'gemini') {
      await stopLiveGeminiRecording();
      return;
    }

    isRecordingRef.current = false;
    setRecording(false);

    const proc = processorRef.current;
    const ctx = audioContextRef.current;
    const stream = mediaStreamRef.current;
    const rec = mediaRecorderRef.current;

    processorRef.current = null;
    audioContextRef.current = null;
    mediaStreamRef.current = null;
    mediaRecorderRef.current = null;

    if (proc) {
      proc.disconnect();
    }
    if (ctx) {
      await ctx.close().catch(() => {});
    }
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }

    let audioBlob = null;
    let ext = 'webm';
    if (rec && rec.state !== 'inactive') {
      await new Promise((resolve) => {
        rec.onstop = resolve;
        rec.stop();
      });
      const mime = rec.mimeType || 'audio/webm';
      if (mime.includes('mp4')) ext = 'm4a';
      audioBlob = new Blob(recordedChunksRef.current, { type: mime });
    }
    recordedChunksRef.current = [];

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }));
    }

    await waitForStopped();

    const parts = transcriptBufferRef.current.filter(Boolean);
    const fullText = parts.join(' ').trim();
    const ts = timestampForFile();

    if (fullText) {
      const txtBlob = new Blob([fullText + '\n'], { type: 'text/plain;charset=utf-8' });
      downloadBlob(txtBlob, `transcript-${ts}.txt`);
    }

    if (audioBlob && audioBlob.size > 0) {
      downloadBlob(audioBlob, `recording-${ts}.${ext}`);
    }

    if (!fullText && (!audioBlob || audioBlob.size === 0)) {
      log('Nothing to save (no audio / no transcript)');
    } else {
      const saved = [fullText && 'transcript', audioBlob?.size && 'recording'].filter(Boolean);
      log(`Saved: ${saved.join(' + ')}`);
    }

    transcriptBufferRef.current = [];
  }, [log, waitForStopped, stopLiveGeminiRecording]);

  const stopRecordingInternal = useCallback(async () => {
    if (mode === 'conversation') {
      await stopConversationRecording();
    } else {
      await stopLiveRecording();
    }
  }, [mode, stopConversationRecording, stopLiveRecording]);

  const startConversationRecording = useCallback(async () => {
    const ok = sttProvider === 'gemini' ? geminiHttpOk : localReady;
    if (!ok) {
      log(
        sttProvider === 'gemini'
          ? `Cannot reach STT server at ${HTTP_BASE} (GET /health failed). Start the Python server and set VITE_VEXYL_HTTP_URL if needed.`
          : 'Server not ready — start VEXYL-STT locally'
      );
      return;
    }

    if (sttProvider === 'gemini' && !geminiConfigured) {
      log(
        'Warning: /health reports Gemini is not configured (no GEMINI_API_KEY on server). Recording is allowed; upload may return 503 until you set the key and restart the server.'
      );
    }

    let mediaStream;
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (err) {
      log(`Microphone: ${err.message}`);
      return;
    }

    setConversationResult(null);
    transcriptBufferRef.current = [];
    setLines([]);

    isRecordingRef.current = true;
    recordedChunksRef.current = [];
    const mime = pickRecorderMime();
    const mediaRecorder = mime
      ? new MediaRecorder(mediaStream, { mimeType: mime })
      : new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = (ev) => {
      if (ev.data.size > 0) recordedChunksRef.current.push(ev.data);
    };
    mediaRecorder.start(250);

    mediaStreamRef.current = mediaStream;
    mediaRecorderRef.current = mediaRecorder;

    setRecording(true);
    log(
      `Conversation capture (${lang}) — one continuous take; stop to generate Doc/Patient document`
    );
  }, [lang, log, sttProvider, geminiHttpOk, geminiConfigured, localReady]);

  const startLiveGeminiRecording = useCallback(async () => {
    if (!geminiHttpOk) {
      log(
        `Cannot reach STT server at ${HTTP_BASE} (GET /health). Start vexyl_stt_server.py or fix VITE_VEXYL_HTTP_URL.`
      );
      return;
    }
    if (!geminiConfigured) {
      log(
        'Warning: Gemini API key not loaded on server — live slices may fail with 503 until GEMINI_API_KEY is set and the server restarted.'
      );
    }

    let mediaStream;
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (err) {
      log(`Microphone: ${err.message}`);
      return;
    }

    transcriptBufferRef.current = [];
    setLines([]);

    isRecordingRef.current = true;
    recordedChunksRef.current = [];
    const mime = pickRecorderMime();
    const mediaRecorder = mime
      ? new MediaRecorder(mediaStream, { mimeType: mime })
      : new MediaRecorder(mediaStream);

    mediaRecorder.ondataavailable = async (ev) => {
      if (!isRecordingRef.current || ev.data.size < 1200) return;
      const ext = ev.type?.includes('mp4') ? 'm4a' : 'webm';
      try {
        const data = await transcribeWithGemini(HTTP_BASE, ev.data, `live-${Date.now()}.${ext}`, {
          languageCode: lang,
          promptType: 'verbatim',
          sttApiKey: STT_API_KEY,
        });
        const t = (data.transcript || '').trim();
        if (t) {
          transcriptBufferRef.current.push(t);
          setLines((prev) => [...prev, { text: t, lang: `${lang} · Gemini` }]);
        }
      } catch (err) {
        log(`Gemini live chunk: ${err.message}`);
      }
    };

    mediaRecorder.start(GEMINI_LIVE_SLICE_MS);

    mediaStreamRef.current = mediaStream;
    mediaRecorderRef.current = mediaRecorder;

    setRecording(true);
    log(`Live (Gemini ${healthSnapshot?.gemini_model || '2.5 Flash'}) — ~${GEMINI_LIVE_SLICE_MS / 1000}s slices`);
  }, [lang, log, geminiHttpOk, geminiConfigured, healthSnapshot?.gemini_model]);

  const startLiveRecording = useCallback(async () => {
    if (sttProvider === 'gemini') {
      await startLiveGeminiRecording();
      return;
    }

    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || serverState !== 'ready') {
      log('Server not ready — start VEXYL-STT on the URL in .env (default ws://127.0.0.1:8091)');
      return;
    }

    let mediaStream;
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (err) {
      log(`Microphone: ${err.message}`);
      return;
    }

    transcriptBufferRef.current = [];
    setLines([]);

    const audioContext = new AudioContext();
    const nativeRate = audioContext.sampleRate;
    const source = audioContext.createMediaStreamSource(mediaStream);

    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    isRecordingRef.current = true;

    processor.onaudioprocess = (e) => {
      if (!isRecordingRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }
      const float32 = e.inputBuffer.getChannelData(0);
      const resampled = downsample(float32, nativeRate, VEXYL_SAMPLE_RATE);
      const int16 = floatToInt16PCM(resampled);
      wsRef.current.send(int16.buffer);
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    sessionIdRef.current += 1;
    const sid = `ui_${sessionIdRef.current}`;
    ws.send(JSON.stringify({ type: 'start', lang, session_id: sid }));

    recordedChunksRef.current = [];
    const mime = pickRecorderMime();
    const mediaRecorder = mime
      ? new MediaRecorder(mediaStream, { mimeType: mime })
      : new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = (ev) => {
      if (ev.data.size > 0) recordedChunksRef.current.push(ev.data);
    };
    mediaRecorder.start(250);

    mediaStreamRef.current = mediaStream;
    audioContextRef.current = audioContext;
    processorRef.current = processor;
    mediaRecorderRef.current = mediaRecorder;

    setRecording(true);
    log(`Live streaming (${lang})`);
  }, [lang, log, serverState, sttProvider, startLiveGeminiRecording]);

  const startRecording = useCallback(async () => {
    if (mode === 'conversation') {
      await startConversationRecording();
    } else {
      await startLiveRecording();
    }
  }, [mode, startConversationRecording, startLiveRecording]);

  const toggleMic = useCallback(() => {
    if (recording) {
      stopRecordingInternal();
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopRecordingInternal]);

  const micReady = sttProvider === 'gemini' ? geminiHttpOk : localReady;
  const micDisabled = !micReady || conversationProcessing;

  const statusDot =
    recording || conversationProcessing
      ? recording
        ? 'rec'
        : 'warn'
      : micReady && sttProvider === 'gemini' && !geminiConfigured
        ? 'warn'
        : micReady
          ? 'ok'
          : serverState === 'connecting'
            ? 'warn'
            : '';

  let statusText = '';
  if (conversationProcessing) {
    statusText =
      sttProvider === 'gemini' ? 'Gemini is writing the document…' : 'Building Doc / Patient document…';
  } else if (recording) {
    if (mode === 'conversation') {
      statusText = 'Recording full conversation…';
    } else {
      statusText = sttProvider === 'gemini' ? 'Live (Gemini slices)…' : 'Transcribing…';
    }
  } else if (sttProvider === 'gemini') {
    if (!healthSnapshot) {
      statusText = `Checking STT server (${HTTP_BASE})…`;
    } else if (!geminiHttpOk) {
      statusText = `Cannot reach STT server — open ${HTTP_BASE}/health in a browser or fix VITE_VEXYL_HTTP_URL`;
    } else if (!geminiConfigured) {
      statusText =
        'Server up — Gemini key missing: set GEMINI_API_KEY (or GOOGLE_API_KEY) in vexyl-stt/.env.secrets, restart server';
    } else {
      statusText = `Gemini online — ${healthSnapshot?.gemini_model || 'gemini-2.5-flash'}`;
    }
  } else if (localReady) {
    statusText = `Local ready${modelName ? ` — ${modelName}` : ''}`;
  } else if (serverState === 'connecting') {
    statusText = 'Connecting to local STT…';
  } else {
    statusText = 'Local STT disconnected — run VEXYL-STT';
  }

  return (
    <div className="app">
      <h1>NemoRx</h1>
      <p className="subtitle">
        <strong>Local</strong> uses the Indic Conformer model (pick one language).{' '}
        <strong>Online (Gemini)</strong> can <strong>auto-detect any language</strong> (and code-switching) or use a
        locale as a soft hint; audio is sent to Google Gemini 2.5 Flash through your HTTP server (API key on server
        only). Live + conversation document modes work for both.
      </p>

      <div className="panel panel-wide">
        <div className="status-row">
          <span className={`dot ${statusDot}`} aria-hidden />
          <span>{statusText}</span>
        </div>

        <div className="mode-row">
          <span className="mode-label">Engine</span>
          <div className="mode-toggle" role="group" aria-label="Speech engine">
            <button
              type="button"
              className={sttProvider === 'local' ? 'active' : ''}
              onClick={() => !recording && !conversationProcessing && setSttProvider('local')}
              disabled={recording || conversationProcessing}
            >
              Local
            </button>
            <button
              type="button"
              className={sttProvider === 'gemini' ? 'active' : ''}
              onClick={() => !recording && !conversationProcessing && setSttProvider('gemini')}
              disabled={recording || conversationProcessing}
            >
              Online · Gemini 2.5 Flash
            </button>
          </div>
        </div>

        <div className="mode-row">
          <span className="mode-label">Mode</span>
          <div className="mode-toggle" role="group" aria-label="Transcription mode">
            <button
              type="button"
              className={mode === 'live' ? 'active' : ''}
              onClick={() => !recording && !conversationProcessing && setMode('live')}
              disabled={recording || conversationProcessing}
            >
              Live
            </button>
            <button
              type="button"
              className={mode === 'conversation' ? 'active' : ''}
              onClick={() => !recording && !conversationProcessing && setMode('conversation')}
              disabled={recording || conversationProcessing}
            >
              Conversation document
            </button>
          </div>
        </div>

        <div className="lang-row">
          <label htmlFor="lang">{sttProvider === 'gemini' ? 'Language (Gemini)' : 'Language'}</label>
          <select
            id="lang"
            value={lang}
            onChange={(e) => setLang(e.target.value)}
            disabled={recording || conversationProcessing}
          >
            {langOptions.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label} ({l.code})
              </option>
            ))}
          </select>
        </div>

        <div className="mic-wrap">
          <button
            type="button"
            className={`mic-btn ${recording ? 'active' : ''}`}
            onClick={toggleMic}
            disabled={micDisabled}
            aria-pressed={recording}
            title={
              recording
                ? 'Stop'
                : mode === 'conversation'
                  ? 'Record full conversation'
                  : sttProvider === 'gemini'
                    ? 'Start live (Gemini slices)'
                    : 'Start live transcription'
            }
          >
            {recording ? (
              <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z" />
              </svg>
            )}
          </button>
          <span className="mic-label">
            {recording ? 'Stop' : mode === 'conversation' ? 'Record' : 'Start'}
          </span>
        </div>

        {mode === 'live' ? (
          <div className="transcript-wrap">
            <label>Live transcript</label>
            <div className="transcript-box">
              {lines.length === 0 ? (
                <span className="placeholder">
                  {sttProvider === 'gemini'
                    ? `Chunks appear every ~${GEMINI_LIVE_SLICE_MS / 1000}s (not word‑real‑time).`
                    : 'Segments appear as you speak…'}
                </span>
              ) : (
                lines.map((l, i) => (
                  <div key={`${i}-${l.text.slice(0, 12)}`} className="chunk">
                    {l.text}
                  </div>
                ))
              )}
            </div>
          </div>
        ) : (
          <div className="transcript-wrap">
            <label>Conversation document</label>
            <div className="transcript-box document-box">
              {!conversationResult?.document?.trim() && conversationResult?.note ? (
                <span className="placeholder">{conversationResult.note}</span>
              ) : !conversationResult?.document?.trim() ? (
                <span className="placeholder">
                  Record the full visit in one take. With <strong>Local</strong>, the server segments audio and
                  clusters speakers. With <strong>Gemini</strong>, the model labels Doc / Patient / Voice roles from
                  context.
                </span>
              ) : (
                <pre className="document-pre">{conversationResult.document}</pre>
              )}
            </div>
            {conversationResult?.segments?.length > 0 && (
              <p className="segment-meta">
                {conversationResult.segments.length} segment
                {conversationResult.segments.length === 1 ? '' : 's'}
                {conversationResult.audio_duration_sec != null
                  ? ` · ${conversationResult.audio_duration_sec}s audio`
                  : ''}
              </p>
            )}
          </div>
        )}

        <div className="log">
          {logLines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      </div>

      <p className="hint">
        WebSocket: <code style={{ color: '#9aa3af' }}>{WS_URL}</code>
        {' · '}
        HTTP: <code style={{ color: '#9aa3af' }}>{HTTP_BASE}</code>
        <br />
        Local conversation: <code style={{ color: '#9aa3af' }}>POST /conversation/transcribe</code>
        {' · '}
        Gemini: <code style={{ color: '#9aa3af' }}>POST /online/gemini/transcribe</code> (set{' '}
        <code style={{ color: '#9aa3af' }}>GEMINI_API_KEY</code> in <code style={{ color: '#9aa3af' }}>vexyl-stt/.env</code>
        ).
      </p>
    </div>
  );
}
