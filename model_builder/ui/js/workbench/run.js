// Dark "run" card (loss chart, live stats, start/pause, progress) shared by pre-training, SFT and DPO.
import { h, fmt, clear } from "../dom.js";
import { icon } from "../icons.js";
import * as ui from "../components.js";
import { busyElsewhere, paceSeconds } from "../pages/shared.js";

function pushPoint(points, point) {
  if (!Number.isFinite(point[1])) return;
  while (points.length && points[points.length - 1][0] >= point[0]) points.pop();
  points.push(point);
}

export function runCard(ctx, options) {
  const { stage, title, caption, startLabel, resumeLabel, startNote, footer, pipeline, sigmaText, chartHeight = 260, emptyText } = options;
  const chart = ui.lossChart({ height: chartHeight, emptyText });
  const pipelineHost = h("div", { class: "wb-pipeline-host", style: { minHeight: `${chartHeight}px` }, hidden: true });
  const nameLabel = h("span", { class: "panel-label" });
  const stats = {
    step: ui.stat("Step"),
    loss: ui.stat("Loss", "—", "train"),
    val: ui.stat("Val", "—", "val"),
    lr: ui.stat("LR"),
    speed: ui.stat("Tok/s"),
    time: ui.stat("Time"),
  };
  const button = h("button", { class: "btn primary wb-run-btn" });
  const buttonNote = h("div", { class: "panel-note wb-run-note" });
  const meter = ui.meter(0);
  const progressText = h("div", { class: "panel-note wb-progress-text" });
  const sigma = h("div", { class: "wb-run-sigma" });
  const errorLine = h("div", { class: "wb-run-error", hidden: true });
  const footerHost = h("div", { class: "wb-run-footer" });
  const legend = h("div", { class: "legend" },
    h("span", {}, h("i", { style: { background: "var(--train)" } }), "train"),
    h("span", {}, h("i", { style: { background: "var(--val)" } }), "val"));

  const element = h("section", { class: "panel-dark wb-run" },
    h("div", { class: "panel-section wb-run-chart" },
      h("div", { class: "panel-header wb-run-title" }, h("span", { class: "panel-label" }, title), h("span", { class: "spacer" }), nameLabel),
      h("div", { class: "wb-chart-head" },
        h("span", { class: "panel-label" }, "Loss curve"),
        h("span", { class: "panel-note" }, caption),
        h("span", { class: "spacer" }),
        legend),
      chart,
      pipelineHost),
    h("div", { class: "panel-section wb-run-stats" },
      h("div", { class: "wb-run-row" },
        h("div", { class: "stats" }, Object.values(stats)),
        h("div", { class: "wb-run-action" }, button, buttonNote)),
      meter, progressText, sigma, errorLine),
    h("div", { class: "panel-section wb-run-foot" }, footerHost));

  let project = null;
  let job = null;
  let metrics = { train: [], val: [] };
  let live = null;
  let phase = null;
  let pending = false;
  let loadedFor = null;
  let drawTimer = null;
  let lastDraw = 0;

  const mine = () => Boolean(job && job.kind === stage);

  function draw() {
    drawTimer = null;
    lastDraw = Date.now();
    const target = live?.target ?? project?.stages?.[stage]?.target ?? 0;
    chart.update(metrics.train, metrics.val, target || 0);
    const node = pipeline && project && !metrics.train.length && !mine() ? pipeline(project) : null;
    pipelineHost.hidden = !node;
    chart.hidden = Boolean(node);
    legend.hidden = Boolean(node);
    clear(pipelineHost);
    if (node) pipelineHost.append(node);
  }

  function scheduleDraw() {
    if (drawTimer) return;
    drawTimer = setTimeout(draw, Math.max(0, 400 - (Date.now() - lastDraw)));
  }

  function pace() {
    if (live && live.tok_per_s && project) {
      const recipe = project.recipes[stage] || {};
      return ((recipe.batch_size || 1) * project.config.context) / live.tok_per_s;
    }
    return project ? paceSeconds(ctx, project, stage) : null;
  }

  function renderStats() {
    const info = project?.stages?.[stage] || {};
    const source = live || info;
    const lastVal = metrics.val.length ? metrics.val[metrics.val.length - 1][1] : info.val_loss;
    stats.step.set(Number.isFinite(source.step) && (source.step > 0 || mine()) ? fmt.int(source.step) : "—");
    stats.loss.set(fmt.loss(source.loss));
    stats.val.set(fmt.loss(lastVal));
    stats.lr.set(fmt.lr(source.lr));
    stats.speed.set(source.tok_per_s ? fmt.compact(source.tok_per_s) : "—");
    stats.time.set(source.elapsed ? fmt.duration(source.elapsed) : "—");
  }

  function renderButton() {
    const info = project?.stages?.[stage];
    clear(button);
    errorLine.hidden = !errorLine.textContent;
    if (!project) {
      button.disabled = true;
      button.append(icon("play", 13), startLabel);
      buttonNote.textContent = "Create a project first";
      return;
    }
    if (mine()) {
      const stopping = job.state === "stopping";
      button.disabled = pending || stopping;
      button.append(stopping ? h("span", { class: "spinner" }) : icon("pause", 13), stopping ? "Pausing…" : "Pause");
      buttonNote.textContent = stopping ? "Saving a checkpoint before stopping" : "Pausing saves a checkpoint you can resume from";
      return;
    }
    const resumable = info && info.checkpoint_step != null;
    button.append(icon("play", 13), resumable ? resumeLabel : startLabel);
    const busy = busyElsewhere(job, stage);
    button.disabled = true;
    if (busy) buttonNote.textContent = busy;
    else if (!info || info.state === "empty") buttonNote.textContent = info?.ready_reason || "Not ready yet";
    else if (info.state === "complete") buttonNote.textContent = "Step target reached. Raise it below to keep going.";
    else {
      button.disabled = pending;
      buttonNote.textContent = resumable ? `from the step ${fmt.int(info.checkpoint_step)} checkpoint` : (startNote ? startNote(project) : "from random weights");
    }
  }

  function renderProgress() {
    const info = project?.stages?.[stage] || {};
    meter.classList.toggle("wb-phase", Boolean(mine() && phase));
    if (mine() && phase) {
      meter.set(phase.progress || 0);
      const percent = Number.isFinite(phase.progress) ? ` · ${Math.round(phase.progress * 100)}%` : "";
      progressText.textContent = `${phase.label || "Preparing…"}${percent}`;
      clear(sigma);
      return;
    }
    const target = live && live.target !== undefined ? live.target : info.target;
    const step = live?.step ?? info.step ?? 0;
    if (target) {
      meter.set(step / target);
      const seconds = pace();
      let text = `${fmt.int(step)} / ${fmt.int(target)} steps`;
      if (step >= target) text += " · step target reached";
      else if (seconds) text += ` · about ${fmt.duration((target - step) * seconds)} left at this pace`;
      progressText.textContent = text;
    } else {
      meter.set(mine() ? 1 : 0);
      progressText.textContent = `${fmt.int(step)} steps · keeps going until you pause it`;
    }
    const tokens = live?.tokens_seen ?? info.tokens_seen ?? 0;
    clear(sigma);
    if (tokens > 0 && project) sigma.append(ui.sigmaLine(sigmaText ? sigmaText(project, tokens) : `≈${fmt.compact(tokens)} training tokens so far`));
  }

  function renderFooter() {
    clear(footerHost);
    if (!project || !footer) return;
    for (const [label, value, attrs] of footer(project)) {
      footerHost.append(h("div", { class: "wb-foot-item", ...(attrs || {}) }, h("span", { class: "panel-label" }, label), h("div", { class: "wb-foot-value" }, value)));
    }
  }

  function renderAll() {
    nameLabel.textContent = project ? project.name : "";
    renderStats();
    renderButton();
    renderProgress();
    renderFooter();
    scheduleDraw();
  }

  async function loadMetrics() {
    if (!project) return;
    try {
      const data = await ctx.api.get(`/api/projects/current/runs/${stage}/metrics`);
      metrics = { train: data.train || [], val: data.val || [] };
    } catch (error) {
      if (error.status !== 409) ctx.toast(error.message, "error");
      metrics = { train: [], val: [] };
    }
    renderStats();
    draw();
  }

  button.addEventListener("click", async () => {
    if (pending || !project) return;
    pending = true;
    errorLine.textContent = "";
    renderButton();
    try {
      if (mine()) await ctx.api.post("/api/jobs/stop", {});
      else await ctx.api.post("/api/jobs", { kind: stage });
    } catch (error) {
      errorLine.textContent = error.message;
    } finally {
      pending = false;
      await ctx.refresh();
      renderButton();
    }
  });

  element.update = (nextProject, nextJob) => {
    const projectChanged = !project || !nextProject || project.id !== nextProject.id;
    project = nextProject;
    job = nextJob || null;
    if (!mine()) {
      live = null;
      phase = null;
    } else if (job.phase && job.phase !== "train") {
      phase = { label: job.phase_label, progress: job.progress };
    }
    if (projectChanged) {
      metrics = { train: [], val: [] };
      loadedFor = null;
    }
    const info = project?.stages?.[stage];
    const key = project ? `${project.id}:${info?.checkpoint_step}:${info?.state === "running" ? "run" : "idle"}` : null;
    if (key && key !== loadedFor && !(mine() && loadedFor === `job:${job.id}`)) {
      loadedFor = key;
      loadMetrics();
    }
    renderAll();
  };
  element.reload = () => {
    loadedFor = null;
    if (project) loadMetrics();
  };
  element.onJob = (nextJob) => {
    job = nextJob;
    if (mine()) {
      phase = job.phase && job.phase !== "train" ? { label: job.phase_label, progress: job.progress } : null;
      if (job.phase === "train" && job.state === "running" && loadedFor !== `job:${job.id}`) {
        loadedFor = `job:${job.id}`;
        loadMetrics();
      }
    }
    renderAll();
  };
  element.onProgress = (event) => {
    if (event.stage !== stage || !mine()) return;
    if (event.phase && event.phase !== "train") {
      phase = { label: phase?.label || job?.phase_label, progress: event.progress };
      renderProgress();
      return;
    }
    phase = null;
    live = { ...(live || {}), ...event };
    if (Number.isFinite(event.loss)) pushPoint(metrics.train, [event.step, event.loss]);
    renderStats();
    renderProgress();
    scheduleDraw();
  };
  element.onEval = (event) => {
    if (event.stage !== stage) return;
    pushPoint(metrics.val, [event.step, event.val_loss]);
    renderStats();
    scheduleDraw();
  };
  element.onJobEnd = (event) => {
    if (event.job?.kind !== stage) return;
    if (event.result === "error") errorLine.textContent = event.error || "The run stopped with an error.";
    job = null;
    phase = null;
    renderAll();
  };
  element.destroy = () => {
    clearTimeout(drawTimer);
    chart.destroy();
  };
  return element;
}
