// Decoder settings panel (presets, temperature, top-k, top-p, min-p, max tokens) used by Sampling and Chat.
import { h } from "../dom.js";
import * as ui from "../components.js";
import { PRESETS, PRESET_ORDER, presetOf } from "./generate.js";

function rangeRow({ label, min, max, step, format, help, ends, onInput }) {
  const value = h("span", { class: "wb-range-value" });
  const input = ui.slider({ min, max, step, value: min, label, onInput });
  const helpLine = help ? h("div", { class: "wb-range-help" }) : null;
  const row = h("div", { class: "wb-range-row" },
    h("div", { class: "wb-range-top" }, h("span", { class: "wb-range-label" }, label), value),
    input,
    ends ? h("div", { class: "wb-range-ends" }, h("span", {}, ends[0]), h("span", {}, ends[1])) : null,
    helpLine);
  row.set = (next) => {
    input.setValue(next);
    value.textContent = format(next);
    if (helpLine) helpLine.textContent = help(next);
  };
  row.input = input;
  return row;
}

export function sideHead(label, ...extra) {
  return h("div", { class: "wb-side-head" }, h("span", { class: "section-label" }, label), ...extra);
}

// Returns an element with .sync() (re-read `settings`) and .setDisabled(bool). Mutates `settings` in place.
export function decodingPanel(settings, { onChange, maxTokens = [8, 512, 8] } = {}) {
  const mode = h("span", { class: "wb-side-mode" });
  const changed = () => {
    sync();
    if (onChange) onChange(settings);
  };
  const presets = ui.segmented(PRESET_ORDER.map((name) => ({ value: name, label: PRESETS[name].label })), presetOf(settings), (name) => {
    const { label, ...values } = PRESETS[name];
    Object.assign(settings, values);
    changed();
  }, { className: "wb-presets" });
  const field = (name) => (value) => {
    settings[name] = value;
    changed();
  };
  const rows = {
    temperature: rangeRow({ label: "Temperature", min: 0, max: 2, step: 0.05, format: (v) => v.toFixed(2), ends: ["predictable", "wild"], onInput: field("temperature") }),
    top_k: rangeRow({
      label: "Top-k", min: 0, max: 200, step: 1, format: (v) => (v ? String(v) : "off"), onInput: field("top_k"),
      help: (v) => (v ? `Only the ${v} likeliest tokens stay in the running.` : "Off: every token in the vocabulary stays in the running."),
    }),
    top_p: rangeRow({
      label: "Top-p", min: 0.05, max: 1, step: 0.01, format: (v) => v.toFixed(2), onInput: field("top_p"),
      help: (v) => (v >= 1 ? "Off: nothing is cut by total probability." : `Keeps the top tokens until they cover ${v.toFixed(2)}.`),
    }),
    min_p: rangeRow({
      label: "Min-p", min: 0, max: 0.5, step: 0.01, format: (v) => v.toFixed(2), onInput: field("min_p"),
      help: (v) => (v ? `Cuts tokens under ${v.toFixed(2)} × the favourite.` : "Off: long-shot tokens are not filtered."),
    }),
    max_tokens: rangeRow({
      label: "Max tokens", min: maxTokens[0], max: maxTokens[1], step: maxTokens[2], format: (v) => String(v), onInput: field("max_tokens"),
      help: (v) => `Each run writes at most ${v} new tokens.`,
    }),
  };
  function sync() {
    const preset = presetOf(settings);
    presets.set(preset);
    mode.textContent = preset ? PRESETS[preset].label.toLowerCase() : "custom";
    for (const [name, row] of Object.entries(rows)) row.set(settings[name]);
  }
  const element = h("div", { class: "wb-decoding" },
    h("section", { class: "wb-side-section" }, sideHead("Decoding", mode), presets, rows.temperature),
    h("section", { class: "wb-side-section" }, sideHead("Fine-tune"), rows.top_k, rows.top_p, rows.min_p, rows.max_tokens));
  element.sync = sync;
  element.setDisabled = (disabled) => {
    for (const row of Object.values(rows)) row.input.disabled = disabled;
    presets.querySelectorAll("button").forEach((button) => {
      button.disabled = disabled;
    });
  };
  sync();
  return element;
}
