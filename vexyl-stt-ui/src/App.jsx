import { useCallback, useEffect, useRef, useState } from 'react';
import { LANGUAGES } from './languages.js';
import { VEXYL_SAMPLE_RATE, downsample, floatToInt16PCM } from './vexylAudio.js';
import './App.css';

const WS_URL = import.meta.env.VITE_VEXYL_WS_URL || 'ws://127.0.0.1:8091';

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

  const [serverState, setServerState] = useState('disconnected');
  const [modelName, setModelName] = useState('');
  const [lang, setLang] = useState('hi-IN');
  const [recording, setRecording] = useState(false);
  const [lines, setLines] = useState([]);
  const [logLines, setLogLines] = useState([]);

  const log = useCallback((msg) => {
    const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
    setLogLines((prev) => [...prev.slice(-40), line]);
    console.log(msg);
  }, []);

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

  const stopRecordingInternal = useCallback(async () => {
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
  }, [log, waitForStopped]);

  const startRecording = useCallback(async () => {
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
    log(`Recording (${lang})`);
  }, [lang, log, serverState]);

  const toggleMic = useCallback(() => {
    if (recording) {
      stopRecordingInternal();
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopRecordingInternal]);

  const statusDot =
    recording ? 'rec' : serverState === 'ready' ? 'ok' : serverState === 'connecting' ? 'warn' : '';
  const statusText = recording
    ? 'Transcribing…'
    : serverState === 'ready'
      ? `Connected${modelName ? ` — ${modelName}` : ''}`
      : serverState === 'connecting'
        ? 'Connecting…'
        : 'Disconnected — run VEXYL-STT locally';

  return (
    <div className="app">
      <h1>VEXYL-STT</h1>
      <p className="subtitle">
        Offline Indian-language speech-to-text. Run the VEXYL-STT server locally, choose a language,
        then use the button to start and stop. Transcript and microphone recording download when you
        stop.
      </p>

      <div className="panel">
        <div className="status-row">
          <span className={`dot ${statusDot}`} aria-hidden />
          <span>{statusText}</span>
        </div>

        <div className="lang-row">
          <label htmlFor="lang">Language</label>
          <select
            id="lang"
            value={lang}
            onChange={(e) => setLang(e.target.value)}
            disabled={recording}
          >
            {LANGUAGES.map((l) => (
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
            disabled={serverState !== 'ready'}
            aria-pressed={recording}
            title={recording ? 'Stop and save' : 'Start transcription'}
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
          <span className="mic-label">{recording ? 'Stop' : 'Start'}</span>
        </div>

        <div className="transcript-wrap">
          <label>Live transcript</label>
          <div className="transcript-box">
            {lines.length === 0 ? (
              <span className="placeholder">Segments appear as you speak…</span>
            ) : (
              lines.map((l, i) => (
                <div key={`${i}-${l.text.slice(0, 12)}`} className="chunk">
                  {l.text}
                </div>
              ))
            )}
          </div>
        </div>

        <div className="log">
          {logLines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      </div>

      <p className="hint">
        Server: <code style={{ color: '#9aa3af' }}>{WS_URL}</code> — set{' '}
        <code style={{ color: '#9aa3af' }}>VITE_VEXYL_WS_URL</code> if needed. See VEXYL-STT README
        for <code style={{ color: '#9aa3af' }}>setup.sh</code> / Docker.
      </p>
    </div>
  );
}
