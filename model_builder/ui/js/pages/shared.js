// Helpers shared by the workbench pages (project loading, run cards, recipe controls, timelines).
import { h, fmt, clear, clamp } from "../dom.js";
import { icon } from "../icons.js";
import * as ui from "../components.js";

export const STAGES = ["pretrain", "sft", "dpo"];
export const STAGE_TITLE = { pretrain: "Pre-training", sft: "Fine-tuning", dpo: "Preference tuning" };
export const STAGE_SHORT = { pretrain: "Base", sft: "SFT", dpo: "DPO" };
export const STAGE_PAGE = { pretrain: "pretraining", sft: "sft", dpo: "dpo" };
const RUNNING_LABEL = { pretrain: "Pre-training", sft: "Fine-tuning", dpo: "Tuning" };
const JOB_TITLE = { pretrain: "pre-training", sft: "fine-tuning", dpo: "preference tuning", calibrate: "speed calibration" };

/* ---------- project loading ---------- */

export async function loadProject(ctx) {
  try {
    return await ctx.api.get("/api/projects/current");
  } catch (error) {
    if (error.status === 409) return null;
    throw error;
  }
}

export async function newProject(ctx, { then = "project-setup" } = {}) {
  const name = await ui.prompt({
    title: "New project",
    message: "Name your model. Each project keeps its own shape, data and checkpoints.",
    value: "My first model",
    confirm: "Create project",
  });
  if (!name) return null;
  try {
    const result = await ctx.api.post("/api/projects", { name });
    await ctx.refresh();
    if (then) ctx.navigate(then);
    return result.project;
  } catch (error) {
    ui.toast(error.message, "error");
    return null;
  }
}

export function noProjectState(ctx, { title = "No project open", message } = {}) {
  return h("div", { class: "empty-state wb-empty" },
    h("span", { class: "wb-empty-icon" }, icon("box", 26)),
    h("h3", {}, title),
    h("p", {}, message || "Create a project to choose a model shape, pick training data and start a run."),
    h("div", { class: "row", style: { justifyContent: "center", marginTop: "6px" } },
      h("button", { class: "btn primary", onclick: () => newProject(ctx) }, icon("plus", 14), "New project"),
      (ctx.state()?.projects || []).length
        ? h("button", { class: "btn", onclick: (event) => openProjectMenu(ctx, event.currentTarget) }, icon("folder-open", 14), "Open project…")
        : null,
    ),
  );
}

export function openProjectMenu(ctx, anchor) {
  const state = ctx.state() || {};
  const items = (state.projects || []).map((project) => ({
    label: project.name,
    detail: project.status ? project.status.label : "",
    icon: project.id === state.current ? "check" : "box",
    action: async () => {
      try {
        await ctx.api.post("/api/projects/select", { id: project.id });
        await ctx.refresh();
      } catch (error) {
        ui.toast(error.message, "error");
      }
    },
  }));
  if (!items.length) items.push({ label: "No projects yet", disabled: true });
  items.push("-", { label: "New project…", icon: "plus", action: () => newProject(ctx) });
  ui.menu(anchor, items);
}

export function errorBanner(error) {
  return h("div", { class: "banner error" }, icon("alert", 15), h("span", {}, error.message || String(error)));
}

/* ---------- formatting helpers ---------- */

export function tint(hex, alpha) {
  const match = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!match) return `rgba(10, 132, 255, ${alpha})`;
  const value = parseInt(match[1], 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

export function plural(count, word, many = `${word}s`) {
  return `${fmt.int(count)} ${count === 1 ? word : many}`;
}

export function stepsLabel(steps) {
  return steps >= 1000 ? `${fmt.compact(steps)} steps` : plural(steps, "step");
}

export function activeDevice(ctx, project) {
  const setting = ctx.state()?.settings?.device || "auto";
  if (setting !== "auto") return setting;
  return project?.estimates?.auto_device || "cpu";
}

export function deviceLabel(ctx, id) {
  const device = (ctx.state()?.devices || []).find((entry) => entry.id === id);
  return device ? device.label : id === "cpu" ? "CPU" : id;
}

// Seconds per optimisation step for a stage: measured pace if the stage has run, else the estimate.
export function paceSeconds(ctx, project, stage) {
  const info = project.stages[stage];
  const recipe = project.recipes[stage] || {};
  const tokens = (recipe.batch_size || 1) * project.config.context;
  if (info && info.tok_per_s) return tokens / info.tok_per_s;
  if (info && info.step > 0 && info.elapsed > 0) return info.elapsed / info.step;
  const estimate = project.estimates?.step_seconds?.[activeDevice(ctx, project)];
  if (!estimate) return null;
  const base = project.recipes.pretrain.batch_size || 1;
  return estimate * ((recipe.batch_size || base) / base) * (stage === "pretrain" ? 1 : 0.5);
}

export function sourcesLabel(project, dataset) {
  const label = dataset?.size_label || "";
  const files = /(\d+)\s*\.(txt|md|jsonl|json|csv|parquet)/i.exec(label);
  if (files) return plural(Number(files[1]), "file");
  const documents = project?.data?.corpus_documents ?? dataset?.status?.documents ?? dataset?.documents;
  if (documents) return `${fmt.compact(documents)} ${documents === 1 ? "doc" : "docs"}`;
  return label.split("·").pop().trim() || "—";
}

/* ---------- job / stage status ---------- */

export function jobFor(ctx) {
  return ctx.state()?.job || null;
}

export function busyElsewhere(job, stage) {
  if (!job) return null;
  if (job.kind === stage) return null;
  return `The engine is busy with ${JOB_TITLE[job.kind] || job.kind}. Pause it there before starting this run.`;
}

export function stagePill(stage, project, job) {
  const info = project?.stages?.[stage];
  if (job && job.kind === stage) {
    return { label: job.state === "stopping" ? "Pausing" : RUNNING_LABEL[stage], kind: "running", note: job.state === "stopping" ? "Saving a checkpoint…" : job.phase_label || "" };
  }
  if (!info) return { label: "Ready", kind: "", note: "" };
  if (info.state === "complete") return { label: "Run complete", kind: "done", note: `Finished at step ${fmt.int(info.step)}` };
  if (info.state === "paused") return { label: "Paused", kind: "paused", note: `Checkpoint saved at step ${fmt.int(info.checkpoint_step ?? info.step)}` };
  return { label: "Ready", kind: "", note: info.ready_reason || "" };
}

export function pageHeader(ctx, { stage, onReset, extra = null }) {
  const pillHost = h("span", { class: "wb-pill-host" });
  const note = h("span", { class: "wb-header-note" });
  const reset = h("button", { class: "btn small", title: "Delete this stage's checkpoints and history" }, icon("reset", 13), "Reset");
  const element = h("div", { class: "page-header wb-header" }, pillHost, note, h("div", { class: "spacer" }), extra, reset);
  let current = null;
  reset.addEventListener("click", async () => {
    if (!current) return;
    const later = stage === "pretrain" ? " Fine-tuning and preference tuning are reset too, and the shape and tokenizer unlock." : stage === "sft" ? " Preference tuning is reset too." : "";
    const ok = await ui.confirm({
      title: `Reset ${STAGE_TITLE[stage].toLowerCase()}?`,
      message: `This deletes every ${STAGE_TITLE[stage].toLowerCase()} checkpoint, the loss history and the saved samples.${later}`,
      confirm: "Reset",
      danger: true,
    });
    if (!ok) return;
    try {
      await ctx.api.post("/api/projects/current/reset", { stage });
      await ctx.refresh();
      if (onReset) onReset();
    } catch (error) {
      ui.toast(error.message, "error");
    }
  });
  element.update = (project, job) => {
    current = project;
    const pill = stagePill(stage, project, job);
    clear(pillHost).append(ui.statusPill(pill.label, pill.kind));
    const busy = busyElsewhere(job, stage);
    note.textContent = busy && !(job && job.kind === stage) ? busy : pill.note;
    const info = project?.stages?.[stage];
    reset.hidden = !project || Boolean(job && job.kind === stage);
    reset.disabled = !info || (info.checkpoint_step == null && !(stage === "pretrain" && project.locked));
  };
  return element;
}

/* ---------- light controls ---------- */

export function sectionHead(label, ...rest) {
  return h("div", { class: "section-head wb-section-head" }, h("span", { class: "section-label" }, label), ...rest);
}

export function lrSlider({ value, min = 1e-5, max = 3e-3, onChange }) {
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  const toPosition = (number) => Math.round(((Math.log10(clamp(number, min, max)) - lo) / (hi - lo)) * 1000);
  const fromPosition = (position) => {
    const raw = 10 ** (lo + (position / 1000) * (hi - lo));
    const exponent = Math.floor(Math.log10(raw));
    return Number((Math.round((raw / 10 ** exponent) * 10) / 10 * 10 ** exponent).toPrecision(2));
  };
  const label = h("span", { class: "setting-value" }, fmt.lr(value));
  const input = ui.slider({
    min: 0, max: 1000, step: 1, value: toPosition(value), label: "Learning rate",
    onInput: (position) => {
      label.textContent = fmt.lr(fromPosition(position));
    },
    onChange: (position) => onChange && onChange(fromPosition(position)),
  });
  const element = h("span", { class: "wb-lr" }, input, label);
  element.set = (next) => {
    input.setValue(toPosition(next));
    label.textContent = fmt.lr(next);
  };
  element.setDisabled = (flag) => {
    input.disabled = flag;
  };
  return element;
}

export function withValue(values, value) {
  return [...new Set([...values, value])].filter((entry) => Number.isFinite(entry)).sort((a, b) => a - b);
}

// Batch size, learning rate and warmup rows; `patch(values)` sends a partial recipe.
export function recipeRows({ recipe, patch, lr = {}, warmups = [0, 10, 20, 50, 100, 200, 500, 1000], batches = [1, 2, 4, 8, 16, 32, 64] }) {
  const batch = ui.stepper({ values: withValue(batches, recipe.batch_size), value: recipe.batch_size, onChange: (value) => patch({ batch_size: value }) });
  const rate = lrSlider({ value: recipe.learning_rate, ...lr, onChange: (value) => patch({ learning_rate: value }) });
  const rows = [
    ui.settingRow("Batch size", "Sequences read in every optimisation step", batch),
    ui.settingRow("Learning rate", "How far each step moves the weights", rate),
  ];
  let warm = null;
  if (recipe.warmup !== undefined) {
    warm = ui.stepper({ values: withValue(warmups, recipe.warmup), value: recipe.warmup, format: (value) => fmt.int(value), onChange: (value) => patch({ warmup: value }) });
    rows.push(ui.settingRow("Warmup", "Steps spent ramping the learning rate up", warm));
  }
  const element = h("div", { class: "wb-recipe" }, rows);
  element.update = (next, disabled) => {
    batch.set(next.batch_size);
    rate.set(next.learning_rate);
    if (warm) warm.set(next.warmup);
    batch.setDisabled(disabled);
    rate.setDisabled(disabled);
    if (warm) warm.setDisabled(disabled);
  };
  return element;
}

/* ---------- page wiring ---------- */

// Loads the current project now and after every state refresh / SSE reconnect; `apply(project, job, error)`.
export function watchProject(ctx, apply) {
  let ticket = 0;
  let disposed = false;
  let dropped = false;
  async function refresh() {
    const mine = ++ticket;
    let project = null;
    let error = null;
    try {
      project = await loadProject(ctx);
    } catch (caught) {
      error = caught;
    }
    if (disposed || mine !== ticket) return;
    apply(project, ctx.state()?.job || null, error);
  }
  ctx.onState(() => refresh());
  ctx.on("connection", (message) => {
    if (!message.connected) dropped = true;
    else if (dropped) {
      dropped = false;
      ctx.refresh();
    }
  });
  ctx.addCleanup(() => {
    disposed = true;
  });
  return refresh;
}

let datasetCache = null;

// Dataset catalog (both kinds) keyed by id; cached for a short while so pages can label data cheaply.
export async function datasetMap(ctx, { fresh = false } = {}) {
  if (!fresh && datasetCache && Date.now() - datasetCache.time < 15000) return datasetCache.map;
  try {
    const { datasets } = await ctx.api.get("/api/datasets");
    datasetCache = { time: Date.now(), map: new Map(datasets.map((info) => [info.id, info])) };
  } catch {
    datasetCache = { time: Date.now(), map: datasetCache?.map || new Map() };
  }
  return datasetCache.map;
}

export function forgetDatasets() {
  datasetCache = null;
}

// Parses "<|system|>\n…<|end|>\n<|user|>\n…" into messages (used for samples and previews).
export function parseTemplate(text) {
  const messages = [];
  const pattern = /<\|(system|user|assistant)\|>\n?([\s\S]*?)(?:<\|end\|>|$)/g;
  let match;
  while ((match = pattern.exec(text || ""))) {
    if (match[2].trim() || match[1] !== "assistant") messages.push({ role: match[1], content: match[2].trim() });
  }
  return messages;
}

export function chatTemplate(system, messages, { open = true } = {}) {
  let text = system ? `<|system|>\n${system}<|end|>\n` : "";
  for (const message of messages) text += `<|${message.role}|>\n${message.content}<|end|>\n`;
  if (open) text += "<|assistant|>\n";
  return text;
}
