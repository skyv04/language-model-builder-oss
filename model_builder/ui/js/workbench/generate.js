// Text generation helpers shared by Sampling and Chat: NDJSON streaming and stored decoder settings.
import { fmt } from "../dom.js";

export const PRESETS = {
  predictable: { label: "Predictable", temperature: 0.3, top_k: 10, top_p: 0.9, min_p: 0.1 },
  balanced: { label: "Balanced", temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05 },
  creative: { label: "Creative", temperature: 1.2, top_k: 100, top_p: 0.98, min_p: 0.02 },
};
export const PRESET_ORDER = ["predictable", "balanced", "creative"];

const DEFAULTS = { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, max_tokens: 120, seed: 42, seed_locked: true };

function readStore(key, fallback) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "null");
    return value && typeof value === "object" ? value : fallback;
  } catch {
    return fallback;
  }
}

function writeStore(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage full or unavailable: keep the in-memory copy */
  }
}

// Decoder settings live in localStorage so they survive page changes and restarts.
export function loadSampler(key = "lmb.sampler", defaults = {}) {
  const merged = { ...DEFAULTS, ...defaults, ...readStore(key, {}) };
  for (const name of Object.keys(DEFAULTS)) {
    if (typeof merged[name] !== typeof DEFAULTS[name]) merged[name] = defaults[name] ?? DEFAULTS[name];
  }
  return merged;
}

export function saveSampler(settings, key = "lmb.sampler") {
  writeStore(key, settings);
}

export function presetOf(settings) {
  return PRESET_ORDER.find((name) => {
    const preset = PRESETS[name];
    return ["temperature", "top_k", "top_p", "min_p"].every((field) => Math.abs(preset[field] - settings[field]) < 1e-6);
  }) || null;
}

export function randomSeed() {
  return 1 + Math.floor(Math.random() * 99999);
}

// Request body "settings"; an unlocked seed is left to the engine (it reports the one it used).
export function requestSettings(settings, { seed } = {}) {
  return {
    temperature: settings.temperature,
    top_k: settings.top_k,
    top_p: settings.top_p,
    min_p: settings.min_p,
    max_tokens: settings.max_tokens,
    seed: seed !== undefined ? seed : settings.seed_locked ? settings.seed : null,
  };
}

export function samplerSummary(settings, seed = settings.seed) {
  const parts = [`T ${settings.temperature.toFixed(2)}`, settings.top_k ? `k ${settings.top_k}` : "k off",
    `p ${settings.top_p.toFixed(2)}`, `min-p ${settings.min_p.toFixed(2)}`];
  if (seed != null) parts.push(`seed ${seed}`);
  return parts.join(" · ");
}

export function stopLabel(end, { chat = false } = {}) {
  if (!end) return "Stopped";
  if (end.stop_reason === "length") return "Hit the token limit";
  if (end.stop_reason === "context") return "Filled the context window";
  if (end.stop_reason === "stopped") return "Stopped";
  return chat ? "Ended cleanly" : "Reached an end-of-text token";
}

export function speedLine(end) {
  if (!end) return "";
  const parts = [`${fmt.int(end.tokens || 0)} tok`];
  if (end.tok_per_s) parts.push(`${fmt.int(Math.round(end.tok_per_s))} tok/s`);
  if (end.ttft != null) parts.push(`${end.ttft.toFixed(1)}s to first`);
  return parts.join(" · ");
}

// Streams POST /api/projects/current/generate. Resolves with {start, tokens, end, aborted}.
// An `{"type":"error"}` line from the engine rejects with that message (partial output on error.partial).
export async function generate(ctx, body, { onStart, onToken, signal } = {}) {
  const result = { start: null, tokens: [], end: null, aborted: false };
  let failure = null;
  try {
    await ctx.api.stream("/api/projects/current/generate", body, (event) => {
      if (event.type === "start") {
        result.start = event;
        if (onStart) onStart(event);
      } else if (event.type === "token") {
        result.tokens.push(event.token);
        if (onToken) onToken(event.token, result);
      } else if (event.type === "end") {
        result.end = event;
      } else if (event.type === "error") {
        failure = new Error(event.error || "Generation failed.");
      }
    }, signal);
  } catch (error) {
    if (error.name !== "AbortError" && !(signal && signal.aborted)) throw error;
    result.aborted = true;
  }
  if (failure) {
    failure.partial = result;
    throw failure;
  }
  return result;
}

// Stages that have a checkpoint to sample from, best (latest in the pipeline) first.
export function sampleableStages(project) {
  return ["dpo", "sft", "pretrain"].filter((stage) => project?.stages?.[stage]?.checkpoint_step != null);
}

export { readStore, writeStore };
