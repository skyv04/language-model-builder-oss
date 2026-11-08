// Project setup: model blueprint, tokenizer, shape presets and sliders, compute device.
import { h, fmt, clear } from "../dom.js";
import { icon } from "../icons.js";
import * as ui from "../components.js";
import { noProjectState, errorBanner, watchProject, sectionHead } from "./shared.js";
import { blueprintCard, parameterCount, vocabOf, GPT2_VOCAB } from "../workbench/blueprint.js";

const PRESETS = [
  { value: "pocket", label: "Pocket", text: "The smallest useful shape. Trains in minutes, which makes it ideal for a first walk through every stage." },
  { value: "starter", label: "Starter", text: "A balanced first model: light enough for this machine, large enough to write short, simple stories." },
  { value: "large", label: "Large", text: "Wider, deeper and with a longer context. Noticeably better text, at several times the training time." },
  { value: "custom", label: "Custom", text: "Your own shape. Drag the sliders below; the blueprint and estimates follow along." },
];
const VOCABS = [4000, 10000, 16000];

function headOptions(width, [low, high]) {
  const options = [];
  for (let heads = low; heads <= high; heads += 1) if (width % heads === 0 && (width / heads) % 2 === 0) options.push(heads);
  return options.length ? options : [1];
}

function nearest(values, target) {
  return values.reduce((best, value) => (Math.abs(value - target) < Math.abs(best - target) ? value : best), values[0]);
}

function shapeRow(label, hint, { min, max, step = 1, format = (value) => fmt.int(value), onInput }) {
  const value = h("span", { class: "wb-shape-value" });
  const input = ui.slider({ min, max, step, value: min, label, onInput: (next) => onInput(next) });
  const row = h("div", { class: "wb-shape-row" },
    h("div", { class: "wb-shape-top" }, h("span", { class: "wb-shape-label" }, label), value), input, h("div", { class: "wb-shape-hint" }, hint));
  row.set = (next, shown = next) => {
    input.setValue(next);
    value.textContent = format(shown);
  };
  row.input = input;
  return row;
}

export default {
  id: "project-setup",
  title: "Project setup",
  async render(root, ctx) {
    const body = h("div", { class: "wb-setup" });
    root.append(body);
    let project = null;
    let job = null;
    let built = null;
    let draft = null;
    let queued = null;
    let timer = null;
    let inflight = 0;
    let disposed = false;
    let calibProgress = null;
    let calibError = null;
    let lastBpe = 10000;

    const view = () => ({ config: draft?.config || project.config, tokenizer: draft?.tokenizer || project.tokenizer });

    async function flush() {
      clearTimeout(timer);
      timer = null;
      if (!queued) return;
      const patch = queued;
      queued = null;
      inflight += 1;
      let next = null;
      try {
        next = await ctx.api.patch("/api/projects/current", patch);
        built?.setError(null);
      } catch (error) {
        built?.setError(error.message);
      }
      inflight -= 1;
      if (disposed) return;
      if (next) project = next;
      if (!queued && !inflight) draft = null;
      render();
    }

    function edit(change, wait = 400) {
      const current = view();
      draft = { config: { ...current.config, ...(change.config || {}) }, tokenizer: change.tokenizer || current.tokenizer };
      queued = { ...(queued || {}), ...(change.config ? { config: change.patchConfig || change.config } : {}), ...(change.tokenizer ? { tokenizer: change.tokenizer } : {}) };
      clearTimeout(timer);
      timer = setTimeout(flush, wait);
      render();
    }

    function setShape(key, value) {
      const config = { ...view().config, [key]: value };
      if (key === "width") config.heads = nearest(headOptions(config.width, project.limits.heads), config.heads);
      const { context, width, blocks, heads } = config;
      const match = Object.entries(project.presets).find(([, preset]) => preset.context === context && preset.width === width && preset.blocks === blocks && preset.heads === heads);
      edit({ config: { context, width, blocks, heads, preset: match ? match[0] : "custom" }, patchConfig: { context, width, blocks, heads } });
    }

    function setPreset(preset) {
      if (preset === "custom") {
        edit({ config: { preset: "custom" }, patchConfig: { ...view().config, preset: "custom" } }, 0);
        return;
      }
      edit({ config: { ...project.presets[preset], preset }, patchConfig: { preset } }, 0);
    }

    function setTokenizer(tokenizer) {
      if (tokenizer.kind === "bpe") lastBpe = tokenizer.vocab_size;
      edit({ tokenizer: { ...view().tokenizer, ...tokenizer, vocab_size: tokenizer.kind === "gpt2" ? GPT2_VOCAB : tokenizer.vocab_size } }, 0);
    }

    async function calibrate() {
      calibError = null;
      try {
        const result = await ctx.api.post("/api/jobs", { kind: "calibrate" });
        job = result.job;
        calibProgress = 0;
      } catch (error) {
        calibError = error.message;
      }
      render();
    }

    async function resetPretraining() {
      const ok = await ui.confirm({
        title: "Reset pre-training?",
        message: "This deletes every checkpoint of this project (pre-training, fine-tuning and preference tuning) together with their loss history and samples. The shape and tokenizer become editable again.",
        confirm: "Reset",
        danger: true,
      });
      if (!ok) return;
      try {
        await ctx.api.post("/api/projects/current/reset", { stage: "pretrain" });
        await ctx.refresh();
      } catch (error) {
        ui.toast(error.message, "error");
      }
    }

    function tokenizerCard(kind, title, text, extra) {
      const vocab = h("span", { class: "wb-tok-vocab" });
      const titleText = h("span", { class: "wb-tok-name" }, title);
      const card = h("div", { class: ["wb-tok-card", kind === "bpe" ? "recommended" : ""], role: "radio", tabindex: 0 },
        kind === "bpe" ? h("span", { class: "wb-tok-pill" }, "Recommended") : null,
        h("span", { class: "wb-tok-radio" }, icon("check", 11)),
        h("div", { class: "wb-tok-main" },
          h("div", { class: "wb-tok-title" }, titleText, h("span", { class: "spacer" }), vocab),
          h("div", { class: "wb-tok-text" }, text), extra));
      const pick = () => {
        if (card.classList.contains("disabled") || view().tokenizer.kind === kind) return;
        setTokenizer(kind === "gpt2" ? { kind } : { kind, vocab_size: lastBpe });
      };
      card.addEventListener("click", pick);
      card.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          pick();
        }
      });
      card.vocab = vocab;
      card.titleText = titleText;
      return card;
    }

    function build() {
      const lockText = h("span", { class: "wb-lock-text" });
      const lock = h("div", { class: "banner wb-lock", hidden: true }, icon("lock", 15), lockText,
        h("button", { class: "wb-link", onclick: resetPretraining }, "Reset pre-training…"));
      const blueprint = blueprintCard({ onCalibrate: calibrate });
      const patchError = h("div", { class: "banner error", hidden: true });

      const vocabSwitch = ui.segmented(VOCABS.map((value) => ({ value, label: `${value / 1000}K` })), 10000,
        (value) => setTokenizer({ kind: "bpe", vocab_size: value }), { className: "wb-vocab-switch" });
      const bpeCard = tokenizerCard("bpe", "Fast BPE", "Byte-level BPE learned from your own pre-training text in a few seconds. A small vocabulary keeps the embedding table light, leaving more of the model for learning.", vocabSwitch);
      const gptCard = tokenizerCard("gpt2", "GPT-2 BPE", "The ready-made GPT-2 vocabulary plus four chat tokens. Familiar splits, but its large embedding table can outweigh the rest of a small model.");
      gptCard.vocab.textContent = `${fmt.int(GPT2_VOCAB)} vocab`;
      const tokenizerNote = h("div", { class: "hint wb-tok-note" });

      const deviceHost = h("div", { class: "wb-device-host" });
      const deviceNote = h("div", { class: "hint" });

      const presetSelect = h("select", { class: "text-input wb-preset", "aria-label": "Shape preset", onchange: () => setPreset(presetSelect.value) },
        PRESETS.map((preset) => h("option", { value: preset.value }, preset.label)));
      const presetText = h("p", { class: "wb-preset-text" });
      const shapeStats = h("p", { class: "wb-shape-stats" });
      const limits = project.limits;
      const context = shapeRow("Context window", "How many tokens the model can look back over when it predicts the next one.", { min: limits.context[0], max: limits.context[1], step: 16, format: (value) => `${fmt.int(value)} tokens`, onInput: (value) => setShape("context", value) });
      const width = shapeRow("Width", "How many numbers describe each token as it flows through the model.", { min: limits.width[0], max: limits.width[1], step: 32, onInput: (value) => setShape("width", value) });
      const blocks = shapeRow("Transformer blocks", "Blocks are stacked; each one refines what the previous block produced.", { min: limits.blocks[0], max: limits.blocks[1], onInput: (value) => setShape("blocks", value) });
      let heads = [];
      const headsRow = shapeRow("Attention heads", "Separate attention patterns per block. The width is split evenly between them.", { min: 0, max: 1, onInput: (index) => setShape("heads", heads[index]) });

      const left = h("div", { class: "wb-col" },
        sectionHead("Tokenizer"),
        h("p", { class: "hint wb-section-hint" }, "How text is cut into tokens. The vocabulary size sets both the embedding rows and the output scores."),
        h("div", { class: "wb-tok-list" }, bpeCard, gptCard), tokenizerNote,
        sectionHead("Compute device"), deviceHost, deviceNote);
      const right = h("div", { class: "wb-col" },
        sectionHead("Shape", h("span", { class: "spacer" }), presetSelect),
        presetText, shapeStats, h("div", { class: "wb-shape-rows" }, context, width, blocks, headsRow));
      const element = h("div", { class: "wb-stack" }, lock, blueprint, patchError, h("div", { class: "columns wb-setup-cols" }, left, right));

      let deviceKey = "";
      function update() {
        const { config, tokenizer } = view();
        const state = ctx.state() || {};
        const devices = state.devices || [];
        const setting = state.settings?.device || "auto";
        const auto = project.estimates.auto_device;
        const active = setting === "auto" ? auto : setting;
        const running = Boolean(job && job.kind !== "calibrate");
        const frozen = project.locked || (job && job.kind === "pretrain");
        blueprint.update({ project, config, tokenizer, devices, active, job, progress: calibProgress, error: calibError });

        lock.hidden = !frozen;
        lockText.textContent = project.locked
          ? "Pre-training has saved checkpoints for this shape, so the shape and tokenizer are fixed."
          : "Pre-training is starting with this shape. Pause it before changing anything.";
        lock.querySelector(".wb-link").hidden = !project.locked || running;

        const bpe = tokenizer.kind === "bpe";
        if (bpe) lastBpe = tokenizer.vocab_size;
        bpeCard.classList.toggle("selected", bpe);
        gptCard.classList.toggle("selected", !bpe);
        bpeCard.setAttribute("aria-checked", String(bpe));
        gptCard.setAttribute("aria-checked", String(!bpe));
        const bpeSize = bpe ? tokenizer.vocab_size : lastBpe;
        bpeCard.vocab.textContent = `${fmt.int(bpeSize)} vocab`;
        bpeCard.titleText.textContent = `Fast ${Math.round(bpeSize / 1000)}K BPE`;
        vocabSwitch.set(bpeSize);
        vocabSwitch.hidden = !bpe;
        for (const card of [bpeCard, gptCard]) card.classList.toggle("disabled", Boolean(frozen));
        vocabSwitch.querySelectorAll("button").forEach((button) => {
          button.disabled = Boolean(frozen);
        });
        tokenizerNote.textContent = bpe
          ? tokenizer.trained && !draft ? `Trained on this project's corpus · ${tokenizer.label}` : "Trained automatically on your corpus when pre-training starts."
          : "Ready to use, nothing to train.";

        const key = `${devices.map((device) => device.id).join(",")}`;
        if (key !== deviceKey) {
          deviceKey = key;
          const gpus = devices.filter((device) => device.kind === "gpu").length;
          const options = [{ value: "auto", label: "Auto" }, ...devices.map((device) => ({ value: device.id, label: device.kind === "gpu" ? (gpus > 1 ? device.label : "GPU") : "CPU", title: `${device.name} · ${device.detail || device.label}` }))];
          clear(deviceHost).append(ui.segmented(options, setting, async (value) => {
            await ctx.saveSettings({ device: value });
            if (!disposed) render();
          }, { className: "wb-device-switch" }));
        }
        deviceHost.firstChild.set(setting);
        const chosen = devices.find((device) => device.id === active);
        const name = chosen ? `${chosen.name} (${chosen.detail || chosen.label})` : active;
        deviceNote.textContent = (setting === "auto" ? `Auto picks the fastest device for this shape: ${name}.` : `Training runs on ${name}.`) + (running ? " A change applies from the next run." : "");

        presetSelect.value = config.preset;
        presetSelect.disabled = Boolean(frozen);
        presetText.textContent = PRESETS.find((preset) => preset.value === config.preset)?.text || "";
        const vocab = vocabOf(tokenizer);
        const parameters = parameterCount(config, vocab);
        const starter = project.presets.starter;
        const work = (parameters * config.context) / (parameterCount(starter, vocab) * starter.context);
        const tokName = bpe ? `the ${Math.round(tokenizer.vocab_size / 1000)}K BPE vocabulary` : "the GPT-2 vocabulary";
        clear(shapeStats).append(`With ${tokName} this shape holds about `, h("b", {}, `${fmt.compact(parameters)} parameters`),
          ", and every training step costs ", h("b", {}, `${work >= 10 ? Math.round(work) : work.toFixed(work < 1 ? 2 : 1)}×`), " a Starter step.");

        heads = headOptions(config.width, project.limits.heads);
        headsRow.input.max = Math.max(1, heads.length - 1);
        headsRow.input.disabled = Boolean(frozen) || heads.length < 2;
        context.set(config.context);
        width.set(config.width);
        blocks.set(config.blocks);
        headsRow.set(Math.max(0, heads.indexOf(config.heads)), config.heads);
        for (const row of [context, width, blocks]) row.input.disabled = Boolean(frozen);
      }
      function setError(message) {
        patchError.textContent = message || "";
        patchError.hidden = !message;
      }
      return { element, update, setError };
    }

    function render() {
      if (!project || !built) return;
      built.update();
    }

    function apply(nextProject, nextJob, error) {
      job = nextJob;
      if (error) {
        built = null;
        body.replaceChildren(errorBanner(error));
        return;
      }
      if (!nextProject) {
        project = null;
        built = null;
        draft = null;
        body.replaceChildren(noProjectState(ctx, { message: "Create a project to design its shape, choose a tokenizer and measure how fast this machine trains it." }));
        return;
      }
      if (project && nextProject.id !== project.id) {
        draft = null;
        built = null;
      }
      project = nextProject;
      if (!job || job.kind !== "calibrate") calibProgress = null;
      if (!built) {
        built = build();
        body.replaceChildren(built.element);
      }
      render();
    }

    const refresh = watchProject(ctx, apply);
    ctx.on("job", (message) => {
      job = message.job;
      if (job.kind === "calibrate" && calibProgress === null) calibProgress = 0;
      render();
    });
    ctx.on("progress", (message) => {
      if (message.phase !== "calibrate" && !(job && job.kind === "calibrate")) return;
      calibProgress = message.progress;
      render();
    });
    ctx.on("calibration", () => refresh());
    ctx.on("job_end", (message) => {
      if (message.job?.kind !== "calibrate") return;
      job = null;
      calibProgress = null;
      if (message.result === "error") calibError = message.error || "Calibration failed.";
      refresh();
    });

    await refresh();
    return () => {
      disposed = true;
      if (queued) flush();
      clearTimeout(timer);
    };
  },
};
