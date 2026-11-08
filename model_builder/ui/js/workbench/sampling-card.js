// The dark Sampling card: header (stage + checkpoint), prompt editor or token output, next-token picks, footer.
import { h, fmt } from "../dom.js";
import { icon } from "../icons.js";
import * as ui from "../components.js";
import { STAGE_SHORT } from "../pages/shared.js";
import { sampleText, legend, nextTokenRow, tokensLabel } from "./sample-text.js";

export const EXAMPLES = [
  { label: "Story", text: "Once upon a time, a little fox found a shiny key under" },
  { label: "Partial sentence", text: "The best part of a rainy afternoon is" },
  { label: "Question", text: "Why do cats like to sleep in boxes?" },
  { label: "Q&A", text: "Q: What do bees make?\nA:" },
  { label: "Chat-like", text: "User: Can you tell me a story about a brave mouse?\nAssistant:" },
];

function iconButton(name, title, onclick, size = 14) {
  return h("button", { class: "wb-icon-btn", title, "aria-label": title, onclick }, icon(name, size));
}

// `actions`: onStage, onToggleChips, onEdit, onPromptInput, onGenerate, onStop, onContinue, onRetry,
// onPickAlternative(index, alternative, token), onPickNext(alternative, peek), onSeed(n), onShuffle, onLock.
export function samplingCard(actions) {
  const stageHost = h("span", { class: "wb-sampler-stages" });
  const step = h("span", { class: "panel-label wb-sampler-step" });
  const chipsButton = iconButton("search", "Show each token as a chip", () => actions.onToggleChips());
  const editButton = iconButton("pencil", "Edit the prompt", () => actions.onEdit(), 13);
  const header = h("div", { class: "wb-sampler-head" },
    h("span", { class: "panel-label" }, "Sampling"), h("span", { class: "spacer" }), stageHost, step, chipsButton, editButton);
  const note = h("div", { class: "panel-note wb-sampler-note", hidden: true });

  const textarea = h("textarea", {
    class: "wb-prompt", rows: 5, spellcheck: false, "aria-label": "Prompt",
    placeholder: "Type the opening of something…",
  });
  textarea.addEventListener("input", () => {
    actions.onPromptInput(textarea.value);
    lead.hidden = Boolean(textarea.value);
  });
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      actions.onGenerate();
    }
  });
  const lead = h("p", { class: "wb-try-lead" },
    "Write the start of something. The model picks up exactly where your words stop and keeps going, one token at a time.");
  const tryRow = h("div", { class: "wb-try-row" }, h("span", { class: "panel-label" }, "Try"),
    EXAMPLES.map((example) => h("button", {
      class: "chip",
      title: example.text,
      onclick: () => {
        textarea.value = example.text;
        lead.hidden = true;
        actions.onPromptInput(example.text);
        textarea.focus();
      },
    }, example.label)));
  const editor = h("div", { class: "wb-editor" }, textarea, lead, tryRow);

  const output = sampleText({ chips: true, onPick: (...args) => actions.onPickAlternative(...args) });
  const meta = h("div", { class: "wb-out-meta" });
  const legendRow = legend();
  const outputWrap = h("div", { class: "wb-output" }, output, meta);
  const error = h("div", { class: "wb-dark-error", hidden: true });
  const body = h("div", { class: "wb-sampler-body" }, editor, outputWrap, error);
  const next = nextTokenRow({ onPick: (...args) => actions.onPickNext(...args) });

  const generateButton = h("button", { class: "btn primary wb-sampler-btn", onclick: () => (generateButton.dataset.mode === "stop" ? actions.onStop() : actions.onGenerate()) });
  const continueButton = h("button", { class: "btn wb-sampler-btn", title: "Keep writing from the end of the text", onclick: () => actions.onContinue() },
    icon("play", 12), "Continue");
  const retryButton = h("button", { class: "btn wb-sampler-btn", title: "Same prompt, fresh seed", onclick: () => actions.onRetry() },
    icon("refresh", 13), "Retry");
  const count = h("span", { class: "wb-sampler-count" });
  const seedInput = h("input", { class: "wb-seed-input", type: "text", inputmode: "numeric", "aria-label": "Seed", spellcheck: false });
  seedInput.addEventListener("change", () => {
    const value = Number.parseInt(seedInput.value.trim(), 10);
    if (Number.isFinite(value) && value >= 0) actions.onSeed(value);
    else seedInput.value = seedInput.dataset.value || "";
  });
  seedInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") seedInput.blur();
  });
  const seedBox = h("label", { class: "wb-seed", title: "Seed: the dice roll behind every pick. The same seed, prompt and settings give the same text." },
    icon("dice", 13), seedInput);
  const shuffleButton = iconButton("shuffle", "Roll a new seed", () => actions.onShuffle());
  const lockButton = h("button", { class: "wb-icon-btn wb-lock", onclick: () => actions.onLock() });
  const footer = h("div", { class: "wb-sampler-foot" },
    generateButton, continueButton, retryButton, h("span", { class: "spacer" }), count, seedBox, shuffleButton, lockButton);

  const element = h("section", { class: "panel-dark wb-sampler" }, header, note, body, next, footer);

  let editing = true;
  const card = {
    element, textarea, output, next,
    setStages(stages, current, checkpointStep) {
      stageHost.replaceChildren();
      if (stages.length > 1) {
        stageHost.append(ui.segmented(stages.map((stage) => ({ value: stage, label: STAGE_SHORT[stage] })), current,
          (stage) => actions.onStage(stage), { className: "wb-stage-switch" }));
      }
      step.textContent = checkpointStep != null ? `Checkpoint step ${fmt.int(checkpointStep)}` : "";
    },
    setMode({ editing: nextEditing, chips, hasRun }) {
      editing = nextEditing;
      editor.hidden = !editing;
      outputWrap.hidden = editing;
      next.classList.toggle("muted", editing);
      editButton.classList.toggle("active", editing);
      editButton.disabled = !hasRun && editing;
      chipsButton.classList.toggle("active", chips);
      lead.hidden = Boolean(textarea.value);
      if (chips && !editing) outputWrap.append(legendRow);
      else legendRow.remove();
    },
    setPrompt(text) {
      if (textarea.value !== text) textarea.value = text;
      lead.hidden = Boolean(text);
    },
    setBusy(kind, { hasRun, canGenerate = true } = {}) {
      const streaming = Boolean(kind);
      generateButton.dataset.mode = streaming ? "stop" : "go";
      generateButton.replaceChildren(streaming ? icon("stop", 12) : icon("play", 12), streaming ? "Stop" : "Generate");
      generateButton.classList.toggle("stop", streaming);
      generateButton.disabled = !streaming && !canGenerate;
      generateButton.title = streaming ? "Stop writing" : "Write a continuation of the prompt (Ctrl+Enter)";
      continueButton.disabled = streaming || !hasRun || !canGenerate;
      retryButton.disabled = streaming || !hasRun || !canGenerate;
      textarea.disabled = streaming;
    },
    setCount(total, detail) {
      count.textContent = total ? tokensLabel(total) : "";
      count.title = detail || "";
    },
    setMeta(text) {
      meta.textContent = text || "";
      meta.hidden = !text;
    },
    setSeed(seed, locked) {
      const text = seed != null ? String(seed) : "";
      seedInput.dataset.value = text;
      if (document.activeElement !== seedInput) seedInput.value = text;
      seedInput.placeholder = locked ? "" : "random";
      lockButton.classList.toggle("active", locked);
      lockButton.replaceChildren(icon(locked ? "lock" : "unlock", 13));
      lockButton.title = locked
        ? "Seed locked: every run uses this seed, so you can change one setting and compare"
        : "Seed unlocked: every run rolls a new seed (the last one used is shown)";
      lockButton.setAttribute("aria-label", lockButton.title);
      seedBox.classList.toggle("unlocked", !locked);
    },
    setNote(text) {
      note.textContent = text || "";
      note.hidden = !text;
    },
    setError(message) {
      error.replaceChildren();
      error.hidden = !message;
      if (message) error.append(icon("alert", 14), h("span", {}, message));
    },
    isEditing: () => editing,
  };
  return card;
}
