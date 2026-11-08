// Run length control and sample timeline used under the run cards.
import { h, fmt, clear, clamp } from "../dom.js";
import { icon } from "../icons.js";
import * as ui from "../components.js";
import { sectionHead, paceSeconds, stepsLabel } from "../pages/shared.js";

// Pre-training uses the log slider + "Until I stop" toggle; SFT/DPO use a compact up/down spinner.
export function runLength(ctx, { stage, variant = "slider", values = [], ticks = [], min = 100, max = 50000, help, sigmaText, patch }) {
  const big = h("span", { class: "wb-big" });
  const eta = h("span", { class: "hint" });
  const helpLine = h("p", { class: "hint wb-help" });
  const sigma = h("div", { class: "wb-light-sigma" });
  let project = null;
  let steps = 0;
  let mode = "steps";
  let disabled = false;

  const modeToggle = variant === "slider"
    ? ui.segmented([{ value: "steps", label: "Step target" }, { value: "until_stop", label: "Until I stop" }], "steps", (value) => {
      mode = value;
      render();
      patch({ mode: value });
    })
    : null;
  const slider = variant === "slider"
    ? ui.logSlider({
      min, max, value: 2000, ticks, format: fmt.compact,
      onInput: (value) => {
        steps = value;
        render();
      },
      onChange: (value) => patch({ steps: value }),
    })
    : null;
  const up = h("button", { class: "wb-spin-btn", "aria-label": "More steps", onclick: () => spin(1) }, icon("chevron-down", 12));
  const down = h("button", { class: "wb-spin-btn", "aria-label": "Fewer steps", onclick: () => spin(-1) }, icon("chevron-down", 12));
  const spinner = variant === "spin" ? h("div", { class: "wb-spin" }, up, down) : null;

  function spin(delta) {
    if (disabled) return;
    const list = [...new Set([...values, steps])].sort((a, b) => a - b);
    const index = clamp(list.indexOf(steps) + delta, 0, list.length - 1);
    if (list[index] === steps) return;
    steps = list[index];
    render();
    patch({ steps });
  }

  const element = h("div", { class: "wb-block wb-run-length" },
    sectionHead("Run length", h("span", { class: "spacer" }), modeToggle),
    h("div", { class: "wb-big-row" }, big, eta, h("span", { class: "spacer" }), spinner),
    slider,
    helpLine,
    sigma);

  function render() {
    if (!project) return;
    const info = project.stages[stage] || {};
    const done = info.checkpoint_step || 0;
    const until = mode === "until_stop";
    big.textContent = until ? "Until you pause" : stepsLabel(steps);
    const seconds = paceSeconds(ctx, project, stage);
    const remaining = Math.max(0, steps - done);
    if (until) eta.textContent = seconds ? `roughly ${fmt.int(Math.round(3600 / seconds))} steps an hour at the last pace` : "";
    else if (done >= steps && done > 0) eta.textContent = `reached · the latest checkpoint is step ${fmt.int(done)}`;
    else eta.textContent = seconds ? `about ${fmt.duration(remaining * seconds)} ${done ? "to go " : ""}at the last pace` : "calibrate in Project setup for a time estimate";
    helpLine.textContent = until
      ? "The run keeps going until you press Pause. A checkpoint is written regularly, and one more when you stop."
      : help;
    if (slider) {
      slider.hidden = until;
      if (slider.value !== steps) slider.set(steps);
      slider.setDisabled(disabled);
    }
    if (spinner) {
      up.disabled = disabled || steps >= Math.max(...values, steps);
      down.disabled = disabled || steps <= Math.min(...values, steps);
    }
    clear(sigma);
    if (sigmaText && !until) {
      const text = sigmaText(project, steps);
      if (text) sigma.append(ui.sigmaLine(text));
    }
  }

  element.update = (nextProject, isDisabled) => {
    project = nextProject;
    const recipe = project.recipes[stage] || {};
    steps = recipe.steps || steps;
    mode = recipe.mode || "steps";
    disabled = isDisabled;
    if (modeToggle) {
      modeToggle.set(mode);
      modeToggle.querySelectorAll("button").forEach((button) => {
        button.disabled = isDisabled;
      });
    }
    render();
  };
  return element;
}

function chatParts(prompt) {
  const user = /<\|user\|>\n?([\s\S]*?)<\|end\|>(?![\s\S]*<\|user\|>)/.exec(prompt || "");
  return user ? user[1].trim() : null;
}

export function sampleTimeline(ctx, { stage, caption, chat = false }) {
  let samples = [];
  let index = -1;
  let project = null;
  const counter = h("span", { class: "wb-pager-count" });
  const prev = h("button", { class: "btn ghost small icon-only", "aria-label": "Earlier sample", onclick: () => select(index - 1) }, icon("chevron-left", 14));
  const next = h("button", { class: "btn ghost small icon-only", "aria-label": "Later sample", onclick: () => select(index + 1) }, icon("chevron-right", 14));
  const track = h("div", { class: "wb-track" });
  const head = h("div", { class: "wb-sample-head" });
  const body = h("div", { class: "wb-sample-body" });
  const card = h("div", { class: "card wb-timeline-card" }, track, head, body);
  const element = h("div", { class: "wb-block wb-timeline" },
    sectionHead("Sample timeline", h("span", { class: "hint" }, caption), h("span", { class: "spacer" }), h("span", { class: "wb-pager" }, prev, counter, next)),
    card);

  function select(next) {
    if (!samples.length) return;
    index = clamp(next, 0, samples.length - 1);
    render();
  }

  function render() {
    clear(track);
    clear(head);
    clear(body);
    prev.disabled = index <= 0;
    next.disabled = index >= samples.length - 1;
    counter.textContent = samples.length ? `${index + 1} of ${samples.length}` : "";
    if (!samples.length) {
      const every = project?.recipes?.[stage]?.sample_every;
      body.append(h("div", { class: "wb-sample-empty" }, icon("clock", 15),
        every ? `No samples yet. The model writes a short sample every ${fmt.int(every)} steps while it trains.` : "No samples yet. The model writes a short sample at each checkpoint while it trains."));
      return;
    }
    const first = samples[0].step;
    const last = samples[samples.length - 1].step;
    track.append(h("div", { class: "wb-track-line" }));
    samples.forEach((sample, position) => {
      const ratio = last > first ? (sample.step - first) / (last - first) : 1;
      track.append(h("button", {
        class: ["wb-dot", position === index ? "active" : "", position === samples.length - 1 ? "latest" : ""],
        style: { left: `${ratio * 100}%` },
        title: `Step ${fmt.int(sample.step)}`,
        onclick: () => select(position),
      }));
    });
    const sample = samples[index];
    head.append(h("strong", {}, `Step ${fmt.int(sample.step)}`), index === samples.length - 1 ? h("span", { class: "wb-badge" }, "Latest") : null);
    if (chat) {
      const question = chatParts(sample.prompt);
      body.append(h("div", { class: "wb-sample-chat" },
        question ? h("div", { class: "wb-sample-turn" }, h("span", { class: "wb-role" }, "User"), h("div", {}, question)) : null,
        h("div", { class: "wb-sample-turn reply" }, h("span", { class: "wb-role" }, "Assistant"), h("div", {}, sample.text.replace(/<\|end\|>\s*$/, "")))));
    } else {
      body.append(h("div", { class: "wb-sample-text" }, h("span", { class: "wb-sample-prompt" }, sample.prompt), sample.text.replace(/<\|endoftext\|>[\s\S]*$/, "")));
    }
  }

  async function load() {
    try {
      const data = await ctx.api.get(`/api/projects/current/runs/${stage}/samples`);
      samples = data.samples || [];
    } catch (error) {
      samples = [];
    }
    index = samples.length - 1;
    render();
  }

  let loadedKey = null;
  element.update = (nextProject) => {
    project = nextProject;
    const info = project?.stages?.[stage];
    const key = project ? `${project.id}:${info?.checkpoint_step ?? "none"}` : null;
    if (key !== loadedKey && !(info?.state === "running" && loadedKey && loadedKey.startsWith(`${project.id}:`))) {
      loadedKey = key;
      if (project) load();
      else {
        samples = [];
        render();
      }
    }
  };
  element.onSample = (event) => {
    if (event.stage !== stage) return;
    const atLatest = index === samples.length - 1;
    samples = samples.filter((sample) => sample.step < event.step);
    samples.push({ step: event.step, prompt: event.prompt, text: event.text, time: Date.now() / 1000 });
    if (atLatest) index = samples.length - 1;
    render();
  };
  element.reload = () => {
    loadedKey = null;
    element.update(project);
  };
  render();
  return element;
}
