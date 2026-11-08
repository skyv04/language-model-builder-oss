// Views for the Chat page: raw-transcript instrument, settings card, composer, X-ray legend and the share dialogs.
import { h, clear, fmt } from "../dom.js";
import { icon } from "../icons.js";
import * as ui from "../components.js";
import { STAGES, STAGE_SHORT, STAGE_TITLE } from "../pages/shared.js";
import { decodingPanel, sideHead } from "./sampler.js";

// The engine's default system turn (model_builder/data.py); only used to preview the transcript before the first reply.
export const DEFAULT_SYSTEM = "You are a friendly assistant. Give clear, short answers.";

const TRAILING_STOP = /(?:<\|end\|>|<\|endoftext\|>)+$/;

// The chat format the engine trains and samples with: every turn is "<|role|>\n" + text + "<|end|>\n".
export function templateText(system, messages, { open = true } = {}) {
  const turn = (role, content) => `<|${role}|>\n${String(content || "").trim()}<|end|>\n`;
  let text = turn("system", system && system.trim() ? system : DEFAULT_SYSTEM);
  for (const message of messages) text += turn(message.role, message.content);
  return open ? `${text}<|assistant|>\n` : text;
}

export const rawText = (tokens) => tokens.map((token) => token.text).join("");

// What a reader should see: the reply without the stop token the model used to end its turn.
export const visibleText = (tokens) => rawText(tokens).replace(TRAILING_STOP, "").trim();

// Turns (user + assistant, excluding the open reply) that made it into the prompt the engine actually used.
export function turnsSeen(promptText) {
  const opened = (String(promptText || "").match(/<\|(?:user|assistant)\|>/g) || []).length;
  return Math.max(0, opened - 1);
}

export function samplerReadout(settings) {
  const topK = settings.top_k ? `${settings.top_k} top-k` : "top-k off";
  return `${Number(settings.temperature).toFixed(2)} temp · ${topK} · ${settings.max_tokens} max`;
}

export function xrayLegend() {
  return h("div", { class: "ct-legend" },
    h("span", { class: "ct-legend-swatches" }, [0.15, 0.45, 1].map((p) => h("span", { style: { "--p": p } }))),
    h("span", {}, "X-ray: every chip is one token the model chose. Stronger blue means it was surer of that pick; click a chip to see what else it weighed up."));
}

/* ---------- transcript: the raw string the model reads ---------- */

export function transcriptPanel({ onSampler }) {
  const count = h("span", { class: "ct-tr-count" });
  const copyButton = h("button", { class: "ct-tr-copy", title: "Copy the raw transcript" }, "Copy");
  const body = h("div", { class: "ct-tr-body", tabindex: 0, "aria-label": "Raw transcript" });
  const caption = h("div", { class: "ct-tr-caption" });
  const usage = h("span", { class: "ct-tr-usage" });
  const meter = ui.meter(0);
  const dropped = h("div", { class: "ct-tr-dropped", hidden: true });
  const sampler = h("button", { class: "ct-tr-sampler", title: "Change how the model samples", onclick: () => onSampler && onSampler() });
  let text = "";
  copyButton.addEventListener("click", async () => {
    try {
      await ui.copyText(text);
      ui.toast("Transcript copied");
    } catch (error) {
      ui.toast(`Couldn't copy: ${error.message}`, "error");
    }
  });
  const element = h("section", { class: "panel-dark ct-transcript" },
    h("div", { class: "ct-tr-head" }, h("span", { class: "panel-label" }, "Transcript"), count, h("span", { class: "spacer" }), copyButton),
    h("p", { class: "ct-tr-note" },
      "Exactly what the model reads. Special tokens mark where each turn starts and ends; the reply is whatever it writes after the open ",
      h("span", { class: "special-chip role" }, "<|assistant|>"), "."),
    body,
    h("div", { class: "ct-tr-foot" },
      h("div", { class: "ct-tr-row" }, h("span", { class: "panel-label" }, "Context"), h("span", { class: "spacer" }), usage),
      meter,
      caption,
      dropped,
      h("div", { class: "ct-tr-row" }, h("span", { class: "panel-label" }, "Sampler"), h("span", { class: "spacer" }), sampler)));

  // `used` is null before the engine has counted anything (preview of the next prompt).
  element.set = ({ text: next, streaming = false, used = null, limit = null, droppedTurns = 0, preview = false }) => {
    text = next;
    const stick = body.scrollHeight - body.scrollTop - body.clientHeight < 40;
    clear(body).append(ui.specialText(next));
    if (streaming) body.append(h("span", { class: "wb-caret" }));
    if (stick || streaming) body.scrollTop = body.scrollHeight;
    count.textContent = used != null ? `${fmt.int(used)} tok` : "";
    usage.textContent = `${used != null ? fmt.int(used) : "—"} / ${limit ? fmt.int(limit) : "—"} tok`;
    const fraction = used != null && limit ? used / limit : 0;
    meter.set(fraction);
    meter.classList.toggle("ct-meter-full", fraction > 0.85);
    caption.textContent = preview
      ? "Preview of the next prompt. The engine counts its tokens when the model replies."
      : fraction >= 1
        ? "Full: the reply ran into the end of the window. Next time, the oldest turns are dropped to make room."
        : fraction > 0.85
          ? "Nearly full: once the window is full, the oldest turns are dropped to make room."
          : "Prompt plus reply, out of the most tokens this model can look at once.";
    dropped.hidden = !droppedTurns;
    dropped.textContent = droppedTurns
      ? `The ${droppedTurns === 1 ? "oldest turn" : `${droppedTurns} oldest turns`} no longer fit, so the model never saw ${droppedTurns === 1 ? "it" : "them"}.`
      : "";
  };
  element.setSampler = (settings) => {
    sampler.textContent = samplerReadout(settings);
  };
  return element;
}

/* ---------- settings: system prompt, decoder and seed ---------- */

export function settingsCard(settings, { system, onSystem, onChange, onSeed, onShuffle, onLock }) {
  const textarea = h("textarea", { class: "text-input ct-system-input", rows: 3, spellcheck: "false", placeholder: DEFAULT_SYSTEM, "aria-label": "System prompt" });
  textarea.value = system;
  const resetButton = h("button", { class: "btn ghost small ct-system-reset", title: "Clear it and use the default" }, "Use default");
  const sync = () => {
    resetButton.disabled = !textarea.value.trim();
  };
  textarea.addEventListener("input", () => {
    sync();
    onSystem(textarea.value);
  });
  resetButton.addEventListener("click", () => {
    textarea.value = "";
    sync();
    onSystem("");
  });
  sync();

  const decoding = decodingPanel(settings, { onChange });
  const seedInput = h("input", { class: "ct-seed-input", type: "text", inputmode: "numeric", spellcheck: "false", "aria-label": "Seed" });
  seedInput.addEventListener("change", () => {
    const value = Number.parseInt(seedInput.value.trim(), 10);
    if (Number.isFinite(value) && value >= 0) onSeed(value);
    else seedInput.value = seedInput.dataset.value || "";
  });
  seedInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") seedInput.blur();
  });
  const seedBox = h("label", { class: "ct-seed" }, icon("dice", 13), seedInput);
  const shuffleButton = h("button", { class: "btn small icon-only", title: "Roll a new seed and lock it", "aria-label": "Roll a new seed", onclick: () => onShuffle() }, icon("shuffle", 13));
  const lockButton = h("button", { class: "btn small icon-only ct-lock", onclick: () => onLock() });
  const seedMode = h("span", { class: "wb-side-mode" });
  const seedNote = h("p", { class: "wb-side-note" });

  const element = h("aside", { class: "ct-settings" },
    h("section", { class: "wb-side-section" },
      sideHead("System prompt", h("span", { class: "spacer" }), resetButton),
      textarea,
      h("p", { class: "wb-side-note" }, "Sent ahead of every conversation as a hidden first turn. Leave it empty to use the default shown in grey. Small models only follow it if fine-tuning taught them to.")),
    decoding,
    h("section", { class: "wb-side-section" },
      sideHead("Seed", seedMode),
      h("div", { class: "ct-seed-row" }, seedBox, shuffleButton, lockButton),
      seedNote));

  element.sync = () => decoding.sync();
  element.focusSystem = () => {
    textarea.focus();
    textarea.scrollIntoView({ block: "nearest" });
  };
  element.setSeed = (seed, locked) => {
    const text = seed != null ? String(seed) : "";
    seedInput.dataset.value = text;
    if (document.activeElement !== seedInput) seedInput.value = text;
    seedInput.placeholder = locked ? "" : "random";
    seedBox.classList.toggle("unlocked", !locked);
    lockButton.classList.toggle("active", locked);
    lockButton.replaceChildren(icon(locked ? "lock" : "unlock", 13));
    lockButton.title = locked ? "Seed locked: click to roll a fresh seed for every reply" : "Seed unlocked: click to reuse this seed for every reply";
    lockButton.setAttribute("aria-label", lockButton.title);
    seedMode.textContent = locked ? "locked" : "new every reply";
    seedNote.textContent = locked
      ? "Every reply uses this seed, so the same conversation and settings give the same answer."
      : "Every reply rolls its own seed; the last one used is shown. Lock it to make replies repeatable.";
  };
  return element;
}

/* ---------- composer ---------- */

export function composer({ onSend, onStop }) {
  const input = h("textarea", { class: "ct-input", rows: 1, placeholder: "Message your model…", "aria-label": "Message" });
  const button = h("button", { class: "ct-send" });
  const hint = h("div", { class: "ct-hint" }, "Enter to send · Shift+Enter for a new line");
  let busy = false;
  const grow = () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 168)}px`;
  };
  const submit = () => {
    if (busy || !input.value.trim()) return;
    if (onSend(input.value)) {
      input.value = "";
      grow();
      sync();
    }
  };
  const sync = () => {
    button.disabled = !busy && !input.value.trim();
  };
  input.addEventListener("input", () => {
    grow();
    sync();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit();
    }
  });
  button.addEventListener("click", () => (busy ? onStop() : submit()));
  const element = h("div", { class: "ct-composer" }, h("div", { class: "ct-input-wrap" }, input, button), hint);
  element.setBusy = (next) => {
    busy = next;
    button.classList.toggle("stop", busy);
    button.replaceChildren(icon(busy ? "stop" : "arrow-up", busy ? 13 : 16));
    button.title = busy ? "Stop the reply" : "Send (Enter)";
    button.setAttribute("aria-label", busy ? "Stop" : "Send");
    hint.textContent = busy ? "The model is writing… you can type your next message meanwhile." : "Enter to send · Shift+Enter for a new line";
    sync();
  };
  element.focus = () => input.focus();
  element.setBusy(false);
  return element;
}

/* ---------- share: pick a checkpoint, then show where the .zip went ---------- */

export async function chooseExport(project, preferred) {
  const available = STAGES.filter((stage) => project.stages?.[stage]?.checkpoint_step != null);
  if (!available.length) return null;
  let choice = available.includes(preferred) ? preferred : available[available.length - 1];
  const detail = h("div", { class: "ct-share-detail" });
  const describe = () => {
    detail.textContent = `${STAGE_TITLE[choice]} checkpoint saved at step ${fmt.int(project.stages[choice].checkpoint_step)}`;
  };
  const picker = ui.segmented(STAGES.map((stage) => ({
    value: stage,
    label: STAGE_SHORT[stage],
    disabled: !available.includes(stage),
    title: available.includes(stage) ? STAGE_TITLE[stage] : `No ${STAGE_TITLE[stage].toLowerCase()} checkpoint yet`,
  })), choice, (value) => {
    choice = value;
    describe();
  });
  describe();
  const go = await ui.modal({
    title: "Share model",
    body: h("div", { class: "ct-share" },
      h("p", {}, "Pack a checkpoint into a single .zip you can keep, back up or hand to someone else."),
      h("div", { class: "ct-share-row" }, h("span", { class: "section-label" }, "Checkpoint"), picker),
      detail,
      h("ul", { class: "ct-share-list" },
        h("li", {}, h("code", {}, "model.safetensors"), " the trained weights"),
        h("li", {}, h("code", {}, "config.json"), " the model's shape and chat format"),
        h("li", {}, h("code", {}, "tokenizer.json"), " how text is cut into tokens"),
        h("li", {}, h("code", {}, "README.md"), " what it is and how to load it")),
      h("p", { class: "hint" }, "The file goes to your Downloads folder.")),
    actions: [{ label: "Cancel" }, { label: "Export .zip", primary: true }],
  });
  return go ? choice : null;
}

export function showExported(result, stage) {
  const copyButton = h("button", { class: "btn small" }, "Copy path");
  copyButton.addEventListener("click", async () => {
    try {
      await ui.copyText(result.path);
      copyButton.textContent = "Copied";
    } catch (error) {
      ui.toast(`Couldn't copy: ${error.message}`, "error");
    }
  });
  return ui.modal({
    title: "Model exported",
    body: h("div", { class: "ct-share" },
      h("p", {}, `The ${STAGE_TITLE[stage].toLowerCase()} checkpoint is packed and saved here:`),
      h("div", { class: "ct-path" }, h("code", {}, result.path), copyButton),
      result.folder ? h("p", { class: "hint" }, `Folder: ${result.folder}`) : null),
    actions: [{ label: "Done", primary: true }],
  });
}
