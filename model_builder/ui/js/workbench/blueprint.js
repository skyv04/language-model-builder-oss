// "Model blueprint" card for project setup: the shape diagram, size and speed estimates, calibration.
import { h, svg, fmt, clear } from "../dom.js";
import { icon } from "../icons.js";

export const GPT2_VOCAB = 50261;

export function vocabOf(tokenizer) {
  return tokenizer.kind === "gpt2" ? GPT2_VOCAB : tokenizer.vocab_size;
}

// Same count as the engine: tied token embeddings, learned positions, 12d²+13d per block, final LayerNorm.
export function parameterCount(config, vocab) {
  const d = config.width;
  return vocab * d + config.context * d + config.blocks * (12 * d * d + 13 * d) + 2 * d;
}

const DESCRIBE = {
  context: (c) => `The model reads up to ${fmt.int(c.context)} tokens at once, and every position gets its own learned position vector.`,
  embedding: (c, vocab) => `Each of the ${fmt.int(vocab)} vocabulary entries owns a row of ${fmt.int(c.width)} learned numbers. Looking up a token's row is the first step.`,
  blocks: (c) => `The vectors pass through ${c.blocks} block${c.blocks === 1 ? "" : "s"} in turn. Attention gathers context from earlier tokens, then the MLP reworks it.`,
  block: (c, vocab, index) => `Block ${index + 1} of ${c.blocks}: ${c.heads} attention head${c.heads === 1 ? "" : "s"} look back over the context, then an MLP ${fmt.int(c.width * 4)} numbers wide processes each token.`,
  output: (c, vocab) => `The last vector is compared with all ${fmt.int(vocab)} rows, giving one score per possible next token. Softmax turns the scores into probabilities.`,
};
const IDLE_TEXT = "Point at a stage to see what it does.";

function arrow() {
  return h("div", { class: "bp-arrow", "aria-hidden": "true" },
    svg("svg", { width: 18, height: 10, viewBox: "0 0 18 10" }, svg("path", { d: "M1 5h15M12 1.5 15.5 5 12 8.5" })));
}

const ART_HEIGHT = 116;

function contextArt(context) {
  const count = Math.max(3, Math.min(8, Math.round(Math.log2(context)) - 3));
  const size = 17;
  const gap = 3;
  const height = count * (size + gap) - gap;
  const root = svg("svg", { width: size, height, viewBox: `0 0 ${size} ${height}` });
  for (let index = 0; index < count; index += 1) {
    root.append(svg("rect", { x: 0.5, y: index * (size + gap) + 0.5, width: size - 1, height: size - 1, rx: 3, class: "bp-slot" }));
  }
  return root;
}

let gradientId = 0;

function columnArt(side, kind) {
  const bar = 24;
  const width = side ? bar + 20 : bar;
  const id = `bp-grad-${(gradientId += 1)}`;
  const root = svg("svg", { width, height: ART_HEIGHT, viewBox: `0 0 ${width} ${ART_HEIGHT}` },
    svg("defs", {}, svg("linearGradient", { id, x1: 0, x2: 1, y1: 0, y2: 0 },
      svg("stop", { offset: "0", "stop-color": "#a9c8ff", "stop-opacity": "0.95" }),
      svg("stop", { offset: "1", "stop-color": "#5f86d6", "stop-opacity": "0.85" }))));
  root.append(svg("rect", { x: 0.5, y: 0.5, width: bar - 1, height: ART_HEIGHT - 1, rx: 7, fill: `url(#${id})`, class: `bp-bar ${kind}` }));
  if (side) {
    const cx = bar + 12;
    root.append(svg("line", { x1: bar + 5, x2: bar + 5, y1: 2, y2: ART_HEIGHT - 2, class: "bp-side-line" }));
    root.append(svg("text", { x: cx, y: ART_HEIGHT / 2, class: "bp-side", "text-anchor": "middle", "dominant-baseline": "central", transform: `rotate(90 ${cx} ${ART_HEIGHT / 2})` }, side));
  }
  return root;
}

// One column per block: attention cells (one row per head, capped) above MLP cells.
function blocksArt(blocks, heads, onBlock) {
  const cell = blocks > 13 ? 11 : blocks > 9 ? 14 : 19;
  const gap = 2;
  const split = 6;
  const attnRows = Math.max(1, Math.min(8, heads));
  const mlpRows = 4;
  const rowHeight = (ART_HEIGHT - split - (attnRows + mlpRows - 2) * gap) / (attnRows + mlpRows);
  const width = blocks * (cell + gap) - gap;
  const root = svg("svg", { width, height: ART_HEIGHT, viewBox: `0 0 ${width} ${ART_HEIGHT}` });
  for (let block = 0; block < blocks; block += 1) {
    const x = block * (cell + gap);
    const group = svg("g", { class: "bp-block" });
    group.append(svg("rect", { x: x - 1, y: 0, width: cell + 2, height: ART_HEIGHT, class: "bp-block-hit" }));
    for (let row = 0; row < attnRows; row += 1) {
      group.append(svg("rect", { x, y: row * (rowHeight + gap), width: cell, height: rowHeight, rx: 1.5, class: "bp-attn" }));
    }
    const top = attnRows * (rowHeight + gap) - gap + split;
    for (let row = 0; row < mlpRows; row += 1) {
      group.append(svg("rect", { x, y: top + row * (rowHeight + gap), width: cell, height: rowHeight, rx: 1.5, class: "bp-mlp" }));
    }
    group.addEventListener("mouseenter", () => onBlock(block));
    root.append(group);
  }
  return root;
}

// "≈2–5h" style range for a training-time estimate.
function rangeLabel(low, high) {
  const span = (lo, hi, unit) => (lo === hi ? `≈${hi}${unit}` : `≈${lo}–${hi}${unit}`);
  if (high < 90) return span(Math.max(1, Math.round(low)), Math.round(high), "s");
  if (high < 90 * 60) return span(Math.max(1, Math.round(low / 60)), Math.round(high / 60), "m");
  if (high < 48 * 3600) return span(Math.max(1, Math.round(low / 3600)), Math.round(high / 3600), "h");
  return span(Math.max(1, Math.round(low / 86400)), Math.round(high / 86400), "d");
}

function deviceShort(device) {
  return device.kind === "gpu" ? "GPU" : "CPU";
}

/**
 * update({ project, config, tokenizer, devices, active, job, progress, error })
 *   config/tokenizer may be a not-yet-saved draft; estimates are then scaled from the saved project.
 */
export function blueprintCard({ onCalibrate }) {
  const nameLabel = h("span", { class: "panel-label" });
  const diagram = h("div", { class: "bp-diagram" });
  const legend = h("div", { class: "bp-legend" });
  const describe = h("div", { class: "bp-describe" }, IDLE_TEXT);
  const params = h("span", { class: "bp-params" });
  const facts = h("div", { class: "bp-facts" });
  const devicesLine = h("div", { class: "bp-devices" });
  const warning = h("div", { class: "bp-warning", hidden: true });
  const errorLine = h("div", { class: "wb-run-error", hidden: true });
  const element = h("section", { class: "panel-dark wb-blueprint" },
    h("div", { class: "panel-section bp-top" },
      h("div", { class: "panel-header" }, h("span", { class: "panel-label" }, "Model blueprint"), h("span", { class: "spacer" }), nameLabel),
      diagram, legend, describe),
    h("div", { class: "panel-section bp-foot" },
      h("div", { class: "bp-foot-row" }, h("div", { class: "bp-params-wrap" }, params, h("span", { class: "bp-params-label" }, "parameters")), h("span", { class: "spacer" }), facts),
      devicesLine, warning, errorLine));

  let current = null;
  let hovered = null;

  function setHover(key, block) {
    hovered = key;
    diagram.classList.toggle("hovering", Boolean(key));
    diagram.querySelectorAll(".bp-stage").forEach((stage) => stage.classList.toggle("hot", stage.dataset.stage === key));
    const focus = key === "blocks" && block !== undefined;
    diagram.querySelector('[data-stage="blocks"]')?.classList.toggle("block-focus", focus);
    diagram.querySelectorAll(".bp-block").forEach((group, index) => group.classList.toggle("hot", focus && block === index));
    if (!key || !current) {
      describe.textContent = IDLE_TEXT;
      describe.classList.remove("active");
      return;
    }
    const { config, vocab } = current;
    describe.textContent = key === "blocks" && block !== undefined ? DESCRIBE.block(config, vocab, block) : DESCRIBE[key](config, vocab);
    describe.classList.add("active");
  }
  diagram.addEventListener("mouseleave", () => setHover(null));

  function stage(key, art, value, label) {
    const node = h("div", { class: "bp-stage", dataset: { stage: key } },
      h("div", { class: "bp-art" }, art),
      h("div", { class: "bp-num" }, value),
      h("div", { class: "bp-label" }, label));
    node.addEventListener("mouseenter", () => setHover(key));
    return node;
  }

  function renderDiagram(config, vocab) {
    const key = `${config.context}:${config.width}:${config.blocks}:${config.heads}:${vocab}`;
    if (diagram.dataset.key === key) return;
    diagram.dataset.key = key;
    clear(diagram).append(
      stage("context", contextArt(config.context), fmt.int(config.context), "tokens of context"),
      arrow(),
      stage("embedding", columnArt(`width ${config.width}`, "embed"), fmt.int(vocab), "embedding rows"),
      arrow(),
      stage("blocks", blocksArt(config.blocks, config.heads, (index) => setHover("blocks", index)), fmt.int(config.blocks), `transformer block${config.blocks === 1 ? "" : "s"}`),
      arrow(),
      stage("output", columnArt(null, "out"), fmt.int(vocab), "output scores"));
    clear(legend).append(
      h("span", {}, h("i", { class: "attn" }), `attention · ${config.heads} head${config.heads === 1 ? "" : "s"} of ${config.width / config.heads}`),
      h("span", {}, h("i", { class: "mlp" }), "MLP"));
    if (hovered) setHover(hovered);
  }

  function fact(label, ...values) {
    return h("div", { class: "bp-fact" }, h("span", { class: "panel-label" }, label), ...values);
  }

  element.update = ({ project, config, tokenizer, devices = [], active, job, progress, error }) => {
    const vocab = vocabOf(tokenizer);
    const parameters = parameterCount(config, vocab);
    current = { config, vocab };
    nameLabel.textContent = project.name;
    renderDiagram(config, vocab);
    params.textContent = fmt.compact(parameters);

    const estimates = project.estimates;
    const saved = parameterCount(project.config, vocabOf(project.tokenizer));
    const scale = (parameters / saved) * (config.context / project.config.context);
    const draft = Math.abs(scale - 1) > 1e-9;
    const steps = project.recipes.pretrain.steps || 2000;
    const calibrating = Boolean(job && job.kind === "calibrate");
    const busy = Boolean(job && job.kind !== "calibrate");
    const rows = devices.map((device) => {
      const seconds = estimates.step_seconds?.[device.id];
      return { device, seconds: Number.isFinite(seconds) ? seconds * scale : null };
    });
    const anyMeasured = rows.some((row) => row.seconds);
    const measured = estimates.calibrated && !draft && anyMeasured;
    const [low, high] = measured ? [0.8, 1.3] : [0.6, 1.8];
    const timeOf = (row) => (row?.seconds ? rangeLabel(row.seconds * steps * low, row.seconds * steps * high) : "—");
    const speedOf = (row) => (row?.seconds ? `${row.seconds < 1 ? row.seconds.toFixed(2) : row.seconds.toFixed(1)} s/step` : "not measured");
    const mine = rows.find((row) => row.device.id === active);

    const button = h("button", {
      class: ["btn small bp-calibrate", anyMeasured || calibrating ? "" : "primary"],
      disabled: calibrating || busy,
      title: busy ? "The engine is busy with a training run." : "Time a few training steps on every device",
      onclick: onCalibrate,
    });
    if (calibrating) button.append(h("span", { class: "spinner" }), Number.isFinite(progress) ? `Measuring ${Math.round(progress * 100)}%` : "Measuring…");
    else button.append(icon(anyMeasured ? "refresh" : "gauge", 12), anyMeasured ? "Run again" : "Calibrate");
    let badge = null;
    if (measured) badge = h("span", { class: "bp-badge ok", title: estimates.calibrated_at ? `Measured ${new Date(estimates.calibrated_at * 1000).toLocaleString()}` : "" }, icon("check", 11), "Calibrated");
    else if (anyMeasured) badge = h("span", { class: "bp-badge" }, "Draft estimate");
    const timeValue = mine?.seconds ? timeOf(mine) : anyMeasured ? "—" : "Not measured yet";
    clear(facts).append(
      fact("Weights on disk", h("div", { class: "bp-value" }, fmt.bytes(draft ? parameters * 4 : estimates.weights_bytes ?? parameters * 4))),
      fact("Standard training time", h("div", { class: "bp-value-row" }, h("span", { class: ["bp-value", mine?.seconds ? "" : "dim"] }, timeValue), badge, button)));
    clear(devicesLine);
    if (anyMeasured) {
      devicesLine.append(
        h("span", { class: "panel-label" }, "Training speed"),
        ...rows.map((row) => h("span", { class: ["bp-dev", row.device.id === active ? "active" : ""], title: row.device.detail || row.device.label },
          h("b", {}, deviceShort(row.device)), ` ${row.device.name} · ${speedOf(row)} · ${timeOf(row)}`)),
        h("span", { class: "bp-dev-note" }, `for ${fmt.int(steps)} steps`));
    } else {
      const names = devices.map((device) => `${deviceShort(device)} (${device.name})`);
      const list = names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}` : names[0] || "this machine";
      devicesLine.append(
        h("span", { class: "panel-label" }, "Training speed"),
        h("span", { class: "bp-dev-note" }, calibrating
          ? "Timing a few training steps on each device…"
          : `Calibrate to estimate training time: it times a few training steps on the ${list}.`));
    }

    const memory = draft ? null : estimates.memory_warning;
    warning.hidden = !memory;
    clear(warning);
    if (memory) warning.append(icon("alert", 14), memory);
    errorLine.textContent = error || "";
    errorLine.hidden = !error;
  };
  return element;
}
