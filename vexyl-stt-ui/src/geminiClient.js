/**
 * Online STT via the NemoRx / VEXYL HTTP proxy (Gemini API key stays on the server).
 */

export async function fetchSttHealth(httpBase) {
  const r = await fetch(`${httpBase.replace(/\/$/, '')}/health`);
  if (!r.ok) {
    throw new Error(`health HTTP ${r.status}`);
  }
  return r.json();
}

/**
 * @param {string} httpBase - e.g. http://127.0.0.1:8091
 * @param {Blob} blob
 * @param {string} fileName - filename with extension for MIME guess on server
 * @param {{ languageCode: string, promptType: 'verbatim' | 'conversation_doc', sttApiKey?: string }} opts
 */
export async function transcribeWithGemini(httpBase, blob, fileName, opts) {
  const base = httpBase.replace(/\/$/, '');
  const form = new FormData();
  form.append('file', blob, fileName);
  form.append('language_code', opts.languageCode);
  form.append('prompt_type', opts.promptType);
  const headers = {};
  if (opts.sttApiKey) {
    headers['X-API-Key'] = opts.sttApiKey;
  }
  const r = await fetch(`${base}/online/gemini/transcribe`, {
    method: 'POST',
    headers,
    body: form,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(data.error || `Gemini proxy HTTP ${r.status}`);
  }
  return data;
}
