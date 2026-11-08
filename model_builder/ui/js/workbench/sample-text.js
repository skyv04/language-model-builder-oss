// Views for the Sampling page: token text, next-token candidates, legend, recent samples, compare columns.
import { h, clear, tokenLabel, fmt } from "../dom.js";
import { icon } from "../icons.js";
import * as ui from "../components.js";
import { PRESETS, PRESET_ORDER, samplerSummary } from "./generate.js";

export const POPOVER_TITLE = "Options at this point";

// Prompt as plain text, generated tokens as chips shaded by probability (or plain text when chips are off).
export function sampleText({ chips = true, compact = false, onPick } = {}) {
  const element = h("div", { class: ["wb-out", compact ? "compact" : ""] });
  const caret = h("span", { class: "wb-caret" });
  let showChips = chips;

  function tokenNode(token, index) {
    if (showChips) {
      return ui.tokenChip(token, { popoverTitle: POPOVER_TITLE, onPick: onPick ? (alternative) => onPick(index, alternative, token) : null });
    }
    if (/^<\|[a-z_]+\|>$/.test(token.text)) return h("span", { class: "special-chip" }, token.text);
    return h("span", { class: "wb-out-text" }, token.text);
  }

  element.setRun = (run, { streaming = false } = {}) => {
    clear(element);
    element.classList.toggle("chips", showChips);
    if (!run) return;
    const prompt = (run.promptTokens || []).map((token) => token.text).join("");
    if (prompt) element.append(h("span", { class: "wb-out-prompt" }, ui.specialText(prompt)));
    run.tokens.forEach((token, index) => element.append(tokenNode(token, index)));
    if (streaming) element.append(caret);
  };
  element.addToken = (token, index) => {
    const node = tokenNode(token, index);
    if (caret.isConnected) caret.before(node);
    else element.append(node);
  };
  element.setStreaming = (streaming) => {
    if (streaming && !caret.isConnected) element.append(caret);
    if (!streaming) caret.remove();
  };
  element.setChips = (next) => {
    showChips = next;
  };
  return element;
}

export function legend() {
  return h("div", { class: "wb-legend" },
    h("span", { class: "wb-legend-swatches" }, [0.15, 0.45, 1].map((p) => h("span", { style: { "--p": p } }))),
    h("span", {}, "Brighter chips were likelier picks under the current decoder settings. Click any chip to see the runners-up."));
}

export function nextTokenRow({ onPick }) {
  const list = h("div", { class: "wb-next-list" });
  const element = h("div", { class: "wb-next" },
    h("div", { class: "wb-next-head" },
      h("span", { class: "panel-label" }, "Next token"),
      h("span", { class: "wb-next-caption" }, "The checkpoint's top candidates for what comes next. Pick one to write it in, then Continue.")),
    list);
  element.set = (peek) => {
    clear(list);
    element.hidden = !peek;
    if (!peek) return;
    if (peek.state === "loading") {
      list.append(h("span", { class: "wb-next-wait" }, h("span", { class: "spinner" }), "Asking the model…"));
      return;
    }
    if (peek.state === "error") {
      list.append(h("span", { class: "wb-next-wait" }, icon("alert", 13), peek.error));
      return;
    }
    const best = Math.max(...peek.alternatives.map((alternative) => alternative.prob), 1e-9);
    for (const alternative of peek.alternatives) {
      list.append(h("button", { class: "wb-next-chip", title: "Write this token next", onclick: () => onPick(alternative, peek) },
        h("span", { class: "wb-next-text" }, tokenLabel(alternative.text)),
        h("span", { class: "wb-next-pct" }, ui.formatProb(alternative.prob)),
        h("span", { class: "wb-next-bar" }, h("span", { style: { width: `${(alternative.prob / best) * 100}%` } }))));
    }
  };
  element.set(null);
  return element;
}

export function recentList({ onOpen, onClear }) {
  const count = h("span", { class: "wb-count" });
  const list = h("div", { class: "wb-recent-list" });
  const clearButton = h("button", { class: "btn ghost small wb-recent-clear", onclick: () => onClear() }, "Clear");
  const element = h("section", { class: "wb-side-section" },
    h("div", { class: "wb-side-head" }, h("span", { class: "section-label" }, "Recent samples"), count, h("span", { class: "spacer" }), clearButton),
    list);
  element.set = (samples, activeId) => {
    count.textContent = String(samples.length);
    count.hidden = !samples.length;
    clearButton.hidden = !samples.length;
    clear(list);
    if (!samples.length) {
      list.append(h("p", { class: "wb-side-note" }, "Each sample you generate is kept here, so you can come back to it."));
      return;
    }
    for (const sample of samples) {
      const prompt = (sample.promptTokens || []).map((token) => token.text).join("").replace(/<\|[a-z_]+\|>/g, " ").trim();
      const text = sample.tokens.map((token) => token.text).join("").replace(/<\|[a-z_]+\|>/g, " ").trim();
      list.append(h("button", { class: ["wb-recent", sample.id === activeId ? "active" : ""], onclick: () => onOpen(sample) },
        h("span", { class: "wb-recent-text" }, prompt ? h("span", { class: "wb-recent-prompt" }, `${prompt} `) : null, text || "…"),
        h("span", { class: "wb-recent-meta" }, `${sample.stageLabel || ""}${sample.stageLabel ? " · " : ""}${samplerSummary(sample.settings, sample.seed)}`)));
    }
  };
  return element;
}

// Three side-by-side columns (one per preset) for the compare mode.
export function compareColumns({ onClose }) {
  const columns = new Map();
  const grid = h("div", { class: "wb-compare-grid" });
  for (const name of PRESET_ORDER) {
    const text = sampleText({ chips: true, compact: true });
    const status = h("span", { class: "wb-compare-status" });
    const settingsLine = h("div", { class: "wb-compare-settings" });
    grid.append(h("div", { class: "panel-dark wb-compare-col" },
      h("div", { class: "wb-compare-head" }, h("span", { class: "panel-label" }, PRESETS[name].label), h("span", { class: "spacer" }), status),
      settingsLine, text));
    columns.set(name, { text, status, settingsLine });
  }
  const caption = h("span", { class: "wb-compare-caption" });
  const element = h("section", { class: "wb-compare", hidden: true },
    h("div", { class: "wb-compare-top" }, h("span", { class: "section-label" }, "Compare"), caption, h("span", { class: "spacer" }),
      h("button", { class: "btn ghost small", onclick: () => onClose() }, icon("x", 13), "Close")),
    grid);
  element.columns = columns;
  element.setCaption = (text) => {
    caption.textContent = text;
  };
  element.setStatus = (name, text) => {
    columns.get(name).status.textContent = text;
  };
  element.setSettings = (name, settings, seed) => {
    columns.get(name).settingsLine.textContent = samplerSummary(settings, seed);
  };
  return element;
}

export function tokensLabel(count) {
  return `${fmt.int(count)} ${count === 1 ? "token" : "tokens"}`;
}
