import { useCallback, useEffect, useRef, useState } from 'react';
import { LANGUAGES } from './languages.js';
import { VEXYL_SAMPLE_RATE, downsample, floatToInt16PCM } from './vexylAudio.js';
import './App.css';

const WS_URL = import.meta.env.VITE_VEXYL_WS_URL || 'ws://127.0.0.1:8091';
const PIPELINE_URL =
  import.meta.env.VITE_WHISPER_PIPELINE_URL || 'http://127.0.0.1:8092';

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

  const [engineMode, setEngineMode] = useState('vexyl');
  const [serverState, setServerState] = useState('disconnected');
  const [pipelineState, setPipelineState] = useState('disconnected');
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
    if (engineMode !== 'vexyl') {
      return undefined;
    }
    let ws;
    let closed = false;

    function connect() {
      setServerState('connecting');
      log(`VEXYL: connecting to ${WS_URL}`);
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
          log('VEXYL server ready');
        } else if (msg.type === 'final' && msg.text) {
          transcriptBufferRef.current.push(msg.text.trim());
          setLines((prev) => [...prev, { text: msg.text.trim(), lang: msg.lang }]);
        } else if (msg.type === 'stopped') {
          log('VEXYL session stopped');
          stopResolverRef.current?.();
          stopResolverRef.current = null;
        } else if (msg.type === 'error') {
          log(`VEXYL error: ${msg.message || JSON.stringify(msg)}`);
        }
      };

      ws.onopen = () => {
        if (closed) return;
        log('VEXYL WebSocket open');
      };

      ws.onerror = () => {
        if (!closed) log('VEXYL WebSocket error');
      };

      ws.onclose = () => {
        if (closed) return;
        setServerState('disconnected');
        setModelName('');
        log('VEXYL WebSocket closed');
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
  }, [engineMode, log]);

  useEffect(() => {
    if (engineMode !== 'whisper') {
      setPipelineState('idle');
      return undefined;
    }
    let cancelled = false;

    async function ping() {
      try {
        const r = await fetch(`${PIPELINE_URL}/health`);
        if (!r.ok) throw new Error(String(r.status));
        const j = await r.json();
        if (cancelled) return;
        setPipelineState(j.model_loaded === false ? 'warming' : 'ready');
      } catch {
        if (!cancelled) setPipelineState('disconnected');
      }
    }

    setPipelineState('connecting');
    ping();
    const id = setInterval(ping, 4000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [engineMode]);

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

  const stopMediaOnly = useCallback(async () => {
    isRecordingRef.current = false;

    const proc = processorRef.current;
    const ctx = audioContextRef.current;
    const stream = mediaStreamRef.current;
    const rec = mediaRecorderRef.current;

    processorRef.current = null;
    audioContextRef.current = null;
    mediaStreamRef.current = null;
    mediaRecorderRef.current = null;

    if (proc) proc.disconnect();
    if (ctx) await ctx.close().catch(() => {});
    if (stream) stream.getTracks().forEach((t) => t.stop());

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
    return { audioBlob, ext };
  }, []);

  const finalizeWhisper = useCallback(
    async (audioBlob, ext) => {
      const ts = timestampForFile();
      if (!audioBlob || audioBlob.size === 0) {
        log('Nothing to send (no audio)');
        return;
      }
      log('Uploading to Whisper pipeline…');
      const fd = new FormData();
      const name = ext === 'm4a' ? `recording.${ext}` : 'recording.webm';
      fd.append('file', audioBlob, name);
      let res;
      try {
        res = await fetch(`${PIPELINE_URL}/transcribe`, {
          method: 'POST',
          body: fd,
        });
      } catch (e) {
        log(`Pipeline request failed: ${e.message}`);
        return;
      }
      if (!res.ok) {
        const t = await res.text();
        log(`Pipeline error ${res.status}: ${t.slice(0, 200)}`);
        return;
      }
      const data = await res.json();
      setPipelineState('ready');
      const segs = data.segments || [];
      setLines(
        segs.map((s) => ({
          text: s.text,
          lang: `${s.language} · ${s.language_label || s.language}`,
          start: s.start,
          end: s.end,
          prob: s.language_probability,
        }))
      );

      const fullText = (data.full_text || '').trim();
      if (fullText) {
        downloadBlob(
          new Blob([fullText + '\n'], { type: 'text/plain;charset=utf-8' }),
          `transcript-${ts}.txt`
        );
      }
      const jsonBlob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json',
      });
      downloadBlob(jsonBlob, `transcript-${ts}.json`);
      downloadBlob(audioBlob, `recording-${ts}.${ext}`);
      log(
        `Saved: recording + JSON + ${fullText ? 'txt' : 'no text'} (${segs.length} segments, primary ${data.primary_language || '?'})`
      );
    },
    [log]
  );

  const stopRecordingInternal = useCallback(async () => {
    setRecording(false);

    if (engineMode === 'whisper') {
      const { audioBlob, ext } = await stopMediaOnly();
      await finalizeWhisper(audioBlob, ext);
      return;
    }

    const { audioBlob, ext } = await stopMediaOnly();

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
  }, [engineMode, finalizeWhisper, log, stopMediaOnly, waitForStopped]);

  const startRecording = useCallback(async () => {
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

    if (engineMode === 'whisper') {
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
      isRecordingRef.current = true;
      setRecording(true);
      log('Recording (Whisper — transcribe on stop, languages per segment)');
      return;
    }

    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || serverState !== 'ready') {
      log('VEXYL not ready — start server on ws://127.0.0.1:8091');
      mediaStream.getTracks().forEach((t) => t.stop());
      return;
    }

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
    log(`Recording VEXYL (${lang})`);
  }, [engineMode, lang, log, serverState]);

  const toggleMic = useCallback(() => {
    if (recording) {
      stopRecordingInternal();
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopRecordingInternal]);

  const vexylReady = engineMode === 'vexyl' && serverState === 'ready';
  const whisperOk =
    engineMode === 'whisper' &&
    (pipelineState === 'ready' || pipelineState === 'warming');
  const canStart =
    !recording && (engineMode === 'vexyl' ? vexylReady : whisperOk);

  let statusDot = '';
  let statusText = '';
  if (recording) {
    statusDot = 'rec';
    statusText =
      engineMode === 'vexyl'
        ? 'Live (VEXYL)…'
        : 'Recording… (Whisper runs when you stop)';
  } else if (engineMode === 'vexyl') {
    statusDot = serverState === 'ready' ? 'ok' : serverState === 'connecting' ? 'warn' : '';
    statusText =
      serverState === 'ready'
        ? `VEXYL: ${modelName || 'connected'}`
        : serverState === 'connecting'
          ? 'VEXYL: connecting…'
          : 'VEXYL: disconnected';
  } else {
    statusDot = pipelineState === 'ready' ? 'ok' : pipelineState === 'warming' ? 'warn' : '';
    statusText =
      pipelineState === 'ready'
        ? 'Whisper pipeline: model ready'
        : pipelineState === 'warming'
          ? 'Whisper pipeline: up (model may load on first use)'
          : pipelineState === 'connecting'
            ? 'Whisper pipeline: checking…'
            : `Whisper pipeline: unreachable (${PIPELINE_URL})`;
  }

  return (
    <div className="app">
      <h1>Local STT</h1>
      <p className="subtitle">
        <strong>VEXYL</strong> for dedicated Indic locales (streaming).{' '}
        <strong>Multilingual</strong> uses faster-whisper locally: auto language, per-segment guesses
        — English, Indian languages, and typical code-mixing (Hinglish, Manglish, etc.) in one pass.
      </p>

      <div className="panel">
        <div className="engine-row">
          <span className="engine-label">Engine</span>
          <div className="engine-toggle">
            <button
              type="button"
              className={engineMode === 'vexyl' ? 'active' : ''}
              onClick={() => setEngineMode('vexyl')}
              disabled={recording}
            >
              VEXYL (Indic)
            </button>
            <button
              type="button"
              className={engineMode === 'whisper' ? 'active' : ''}
              onClick={() => setEngineMode('whisper')}
              disabled={recording}
            >
              Multilingual (Whisper)
            </button>
          </div>
        </div>

        <div className="status-row">
          <span className={`dot ${statusDot}`} aria-hidden />
          <span>{statusText}</span>
        </div>

        {engineMode === 'vexyl' && (
          <div className="lang-row">
            <label htmlFor="lang">VEXYL language</label>
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
        )}

        <div className="mic-wrap">
          <button
            type="button"
            className={`mic-btn ${recording ? 'active' : ''}`}
            onClick={toggleMic}
            disabled={!recording && !canStart}
            aria-pressed={recording}
            title={recording ? 'Stop and save' : 'Start'}
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
          <label>{engineMode === 'vexyl' ? 'Live transcript' : 'Result (after stop)'}</label>
          <div className="transcript-box">
            {lines.length === 0 ? (
              <span className="placeholder">
                {engineMode === 'vexyl'
                  ? 'Segments appear as you speak…'
                  : 'Stop recording to transcribe. Segments show guessed language per slice.'}
              </span>
            ) : (
              lines.map((l, i) => (
                <div key={`${i}-${l.text?.slice(0, 8)}`} className="chunk">
                  {l.lang && <span className="lang-tag">{l.lang}</span>}
                  <span className="chunk-text">{l.text}</span>
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
        VEXYL WS: <code>{WS_URL}</code>
        <br />
        Whisper API: <code>{PIPELINE_URL}</code> — run <code>./whisper-pipeline/run.sh</code>, install{' '}
        <code>ffmpeg</code>. Optional: <code>WHISPER_MODEL_SIZE=medium</code> for accuracy (slower on
        CPU).
      </p>
    </div>
  );
}
