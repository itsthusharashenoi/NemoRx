/** Indic-focused list for local VEXYL-STT (model requires a known tag). */
export const LANGUAGES = [
  { code: 'hi-IN', label: 'Hindi' },
  { code: 'ml-IN', label: 'Malayalam' },
  { code: 'ta-IN', label: 'Tamil' },
  { code: 'te-IN', label: 'Telugu' },
  { code: 'kn-IN', label: 'Kannada' },
  { code: 'bn-IN', label: 'Bengali' },
  { code: 'gu-IN', label: 'Gujarati' },
  { code: 'mr-IN', label: 'Marathi' },
  { code: 'pa-IN', label: 'Punjabi' },
  { code: 'or-IN', label: 'Odia' },
  { code: 'as-IN', label: 'Assamese' },
  { code: 'ur-IN', label: 'Urdu' },
  { code: 'sa-IN', label: 'Sanskrit' },
  { code: 'ne-IN', label: 'Nepali' },
];

/** Gemini online: auto-detect any language, or bias toward one locale. */
export const GEMINI_LANGUAGES = [
  { code: 'auto', label: 'Auto-detect (any language)' },
  ...LANGUAGES,
];
