// Chapter 7: How models learn. The loss as a score, gradients and backpropagation, learning rates and
// schedules, AdamW and clipping, a tiny model trained live in the page, and how to read a training run.
import { h, clear, svg, tokenLabel } from "../dom.js";
import { icon } from "../icons.js";
import { chapterShell, finishChapter, prose, callout, figure, lifecycle, control, note, fmt } from "../book/kit.js";
import { surface, axes, linear, logScale, ticks, pathFrom } from "../book/plot.js";
import { learningRateAt, makeVocab, encode, TinyCharModel, mulberry32, TOY_TEXT, TOY_VALIDATION, TOY_TINY } from "../book/learning.js";

const percentText = (p) => {
  if (p >= 0.1) return `${(p * 100).toFixed(0)}%`;
  if (p >= 0.01) return `${(p * 100).toFixed(1)}%`;
  return `${(p * 100).toPrecision(2)}%`;
};

/* ---------- Figure 1: the surprise meter ---------- */

const VOCABS = [
  { value: "4096", label: "4K", size: 4096 },
  { value: "10000", label: "10K", size: 10000 },
  { value: "16000", label: "16K", size: 16000 },
  { value: "50261", label: "GPT-2", size: 50261, title: "GPT-2's 50,257 tokens plus the workbench's four chat tokens" },
];

function surpriseFigure(ctx, life) {
  const fig = figure({
    number: 1,
    title: "Surprise meter",
    info: "loss = −ln p",
    caption: "The loss for a single prediction depends only on the probability the model gave to the token that really came next. Certainty costs nothing, a coin-flip costs 0.69, and a confident mistake costs a lot. The dashed line is where every fresh model starts: an even guess across the whole vocabulary.",
  });
  const state = { logP: Math.log10(0.2), vocab: 4096 };
  const stats = {
    p: ctx.ui.stat("Probability", "—"),
    loss: ctx.ui.stat("Loss", "—", "train"),
    ppl: ctx.ui.stat("Perplexity", "—"),
    fresh: ctx.ui.stat("Fresh model", "—", "val"),
  };
  const slider = ctx.ui.slider({
    min: -5, max: 0, step: 0.01, value: state.logP, label: "Probability given to the right token",
    onInput: (value) => {
      state.logP = value;
      update();
    },
  });
  const probControl = control("Probability the model gave to the right token", slider);
  const vocabSwitch = ctx.ui.segmented(VOCABS.map(({ value, label, title }) => ({ value, label, title })), String(state.vocab), (value) => {
    state.vocab = Number(value);
    update();
  });
  const readout = h("p", { class: "bk-learn-readout" });

  const plot = surface({
    height: 200,
    life,
    label: "Loss as a function of the probability of the right token",
    draw(root, width, height) {
      const area = { left: 40, right: width - 16, top: 14, bottom: height - 30 };
      const x = logScale(1e-5, 1, area.left, area.right);
      const y = linear(0, 12, area.bottom, area.top);
      axes(root, {
        x, y, area,
        xTicks: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        yTicks: [0, 3, 6, 9, 12],
        xFormat: (value) => `${Number((value * 100).toPrecision(1))}%`,
        xLabel: "probability of the right token",
        yLabel: "loss",
      });
      const fresh = Math.log(state.vocab);
      root.append(
        svg("line", { class: "bk-learn-ref", x1: area.left, x2: area.right, y1: y(fresh), y2: y(fresh) }),
        svg("text", { class: "bk-learn-ref-label", x: area.right - 2, y: y(fresh) - 6, "text-anchor": "end" }, `fresh model · ln ${fmt.int(state.vocab)} ≈ ${fresh.toFixed(2)}`),
      );
      const points = [];
      for (let index = 0; index <= 120; index += 1) {
        const p = 10 ** (-5 + (5 * index) / 120);
        points.push([x(p), y(-Math.log(p))]);
      }
      root.append(svg("path", { class: "bk-learn-curve", d: pathFrom(points) }));
      const p = 10 ** state.logP;
      const px = x(p);
      const py = y(-Math.log(p));
      root.append(
        svg("line", { class: "bk-learn-drop", x1: px, x2: px, y1: py, y2: area.bottom }),
        svg("circle", { class: "bk-learn-marker", cx: px, cy: py, r: 5.5 }),
        svg("text", { class: "bk-learn-marker-label", x: px + (px > area.right - 90 ? -10 : 10), y: py - 8, "text-anchor": px > area.right - 90 ? "end" : "start" }, `loss ${(-Math.log(p)).toFixed(2)}`),
      );
    },
  });

  function update() {
    const p = 10 ** state.logP;
    const loss = -Math.log(p);
    const even = 1 / state.vocab;
    probControl.setValue(percentText(p));
    stats.p.set(percentText(p));
    stats.loss.set(loss.toFixed(2));
    stats.ppl.set(1 / p >= 100 ? fmt.int(1 / p) : (1 / p).toFixed(1));
    stats.fresh.set(Math.log(state.vocab).toFixed(2));
    const ratio = p / even;
    clear(readout);
    readout.append(
      `A loss of ${loss.toFixed(2)} is as surprised as picking blindly among ${1 / p >= 100 ? fmt.int(1 / p) : (1 / p).toFixed(1)} equally likely tokens. `,
      ratio >= 1
        ? `That is ${ratio >= 100 ? fmt.int(ratio) : ratio.toFixed(1)} times better than the fresh model, which gives every one of its ${fmt.int(state.vocab)} tokens the same ${percentText(even)}.`
        : h("b", {}, `That is worse than the fresh model's even guess of ${percentText(even)}: a confident mistake costs more than knowing nothing.`),
    );
    plot.redraw();
  }

  fig.body.append(
    h("div", { class: "bk-toolbar bk-learn-toolbar" }, h("span", { class: "panel-label" }, "Vocabulary"), vocabSwitch),
    h("div", { class: "bk-learn-surprise" }, probControl),
    h("div", { class: "bk-stats bk-learn-stats" }, Object.values(stats)),
    plot.element,
    readout,
  );
  update();
  return fig.element;
}

/* ---------- Figure 2: inputs and targets ---------- */

const SHIFT_TOKENS = ["<|endoftext|>", "Once", " upon", " a", " time", ",", " a", " little", " fox", " found", " a", " shiny", " red", " ball", "."];

function shiftFigure(ctx, life) {
  const count = SHIFT_TOKENS.length - 1;
  const fig = figure({
    number: 2,
    title: "Inputs and targets",
    info: `${SHIFT_TOKENS.length} tokens · ${count} graded predictions`,
    tone: "light",
    caption: "The targets are the inputs moved one place to the left, so ordinary text grades itself. Point at a column to see what the model may read at that position and what it is graded on. All the predictions come out of a single forward pass.",
  });
  let selected = 7;
  const chip = (text) => h("span", { class: ["token", "light", /^<\|[a-z_]+\|>$/.test(text) ? "special" : ""] }, tokenLabel(text));
  const columns = [];
  const row = h("div", { class: "bk-learn-pairs" },
    h("div", { class: "bk-learn-pair bk-learn-legend", "aria-hidden": "true" },
      h("span", { class: "bk-label" }, "Input"),
      h("span", { class: "bk-learn-arrow" }),
      h("span", { class: "bk-label" }, "Target"),
    ),
  );
  for (let index = 0; index < count; index += 1) {
    const input = chip(SHIFT_TOKENS[index]);
    const target = chip(SHIFT_TOKENS[index + 1]);
    const button = h("button", {
      class: "bk-learn-pair",
      type: "button",
      "aria-label": `Prediction ${index + 1}: after “${SHIFT_TOKENS.slice(1, index + 1).join("")}” comes “${SHIFT_TOKENS[index + 1].trim()}”`,
      onclick: () => {
        stopPlaying();
        select(index);
      },
    }, input, h("span", { class: "bk-learn-arrow" }, "↓"), target);
    button.addEventListener("pointerenter", () => {
      if (!playing) select(index);
    });
    columns.push({ button, input, target });
    row.append(button);
  }
  const readout = h("p", { class: "bk-learn-readout light" });
  const timer = life.timer();
  let playing = false;
  const playButton = h("button", { class: "btn small", type: "button" });
  const syncPlay = () => {
    clear(playButton);
    playButton.append(icon(playing ? "pause" : "play", 13), playing ? "Pause" : "Step through");
  };
  const tick = () => {
    select((selected + 1) % count);
    timer.set(tick, 1100);
  };
  function stopPlaying() {
    playing = false;
    timer.clear();
    syncPlay();
  }
  playButton.addEventListener("click", () => {
    if (playing) stopPlaying();
    else {
      playing = true;
      syncPlay();
      tick();
    }
  });

  function select(index) {
    selected = index;
    columns.forEach((column, position) => {
      column.button.classList.toggle("on", position === index);
      column.input.classList.toggle("bk-dim", position > index);
      column.target.classList.toggle("bk-dim", position !== index);
      column.target.classList.toggle("bk-hot", position === index);
    });
    clear(readout);
    const seen = SHIFT_TOKENS.slice(0, index + 1).join("");
    readout.append(
      h("b", {}, `Prediction ${index + 1} of ${count}. `),
      "The model may read ",
      h("span", { class: "bk-chip-inline" }, seen),
      " and is graded on the probability it gave to ",
      h("span", { class: "bk-chip-inline orange" }, tokenLabel(SHIFT_TOKENS[index + 1])),
      index === 0 ? ". The start-of-text marker alone has to predict how a text begins." : ". The tokens to its right are hidden by the causal mask.",
    );
  }

  fig.body.append(
    h("div", { class: "bk-toolbar" }, h("span", { class: "bk-label" }, "One training window"), h("span", { class: "spacer" }), playButton),
    row,
    readout,
  );
  syncPlay();
  select(selected);
  return fig.element;
}

/* ---------- Figure 3: gradient descent on a valley ---------- */

const VALLEY = { a: 0.5, b: 6, angle: 25 };
const VALLEY_STEPS = 60;
const VALLEY_LEVELS = [0.03, 0.1, 0.3, 0.8, 1.8, 3.6, 6.5, 11, 18, 28];
const valleyCos = Math.cos((VALLEY.angle * Math.PI) / 180);
const valleySin = Math.sin((VALLEY.angle * Math.PI) / 180);
const toValley = (x, y) => [valleyCos * x + valleySin * y, -valleySin * x + valleyCos * y];
const valleyLoss = (x, y) => {
  const [u, v] = toValley(x, y);
  return 0.5 * (VALLEY.a * u * u + VALLEY.b * v * v);
};
const valleyGrad = (x, y) => {
  const [u, v] = toValley(x, y);
  const gu = VALLEY.a * u;
  const gv = VALLEY.b * v;
  return [valleyCos * gu - valleySin * gv, valleySin * gu + valleyCos * gv];
};

const OPTIMISERS = [
  { value: "plain", label: "Plain" },
  { value: "momentum", label: "Momentum" },
  { value: "adam", label: "Adam" },
];

function descentFigure(ctx, life) {
  const fig = figure({
    number: 3,
    title: "Gradient descent playground",
    info: "a loss surface with two weights",
    wide: true,
    caption: "A model with only two weights, so its loss surface can be drawn as a map: darker means lower loss. Each step measures the slope under the ball and moves against it. The valley is steep across and shallow along, which is typical: with plain steps, rates above about 0.33 overshoot the steep direction further every time and diverge. Click the map to drop the ball somewhere else.",
  });
  const state = { optimiser: "plain", logRate: -1, start: [-2.6, 1.5] };
  let run;
  let view = null;
  let lastFrame = 0;
  const rate = () => 10 ** state.logRate;

  function restart() {
    const [x, y] = state.start;
    run = { x, y, m: [0, 0], v: [0, 0], t: 0, path: [[x, y]], losses: [valleyLoss(x, y)], flips: 0, side: 0, reachedAt: 0, diverged: false, done: false };
    render();
    animation.start();
  }

  function advance() {
    const [gx, gy] = valleyGrad(run.x, run.y);
    const lr = rate();
    run.t += 1;
    if (state.optimiser === "plain") {
      run.x -= lr * gx;
      run.y -= lr * gy;
    } else if (state.optimiser === "momentum") {
      run.m = [0.9 * run.m[0] + gx, 0.9 * run.m[1] + gy];
      run.x -= lr * run.m[0];
      run.y -= lr * run.m[1];
    } else {
      run.m = [0.9 * run.m[0] + 0.1 * gx, 0.9 * run.m[1] + 0.1 * gy];
      run.v = [0.95 * run.v[0] + 0.05 * gx * gx, 0.95 * run.v[1] + 0.05 * gy * gy];
      const c1 = 1 - 0.9 ** run.t;
      const c2 = 1 - 0.95 ** run.t;
      run.x -= (lr * (run.m[0] / c1)) / (Math.sqrt(run.v[0] / c2) + 1e-8);
      run.y -= (lr * (run.m[1] / c1)) / (Math.sqrt(run.v[1] / c2) + 1e-8);
    }
    const loss = valleyLoss(run.x, run.y);
    run.path.push([run.x, run.y]);
    run.losses.push(loss);
    const across = toValley(run.x, run.y)[1];
    if (Math.abs(across) > 0.03) {
      const side = Math.sign(across);
      if (run.side && side !== run.side) run.flips += 1;
      run.side = side;
    }
    if (!run.reachedAt && loss < run.losses[0] * 1e-3) run.reachedAt = run.t;
    if (!Number.isFinite(loss) || loss > run.losses[0] * 60) {
      run.diverged = true;
      run.done = true;
    }
    if (run.t >= VALLEY_STEPS) run.done = true;
  }

  const animation = life.loop((time) => {
    if (time - lastFrame < 65) return true;
    lastFrame = time;
    advance();
    render();
    return !run.done;
  });

  const rateSlider = ctx.ui.slider({
    min: -2.3, max: -0.1, step: 0.01, value: state.logRate, label: "Learning rate",
    onInput: (value) => {
      state.logRate = value;
      rateControl.setValue(rate().toFixed(3));
      restart();
    },
  });
  const rateControl = control("Learning rate", rateSlider, { value: rate().toFixed(3) });
  const optimiserSwitch = ctx.ui.segmented(OPTIMISERS, state.optimiser, (value) => {
    state.optimiser = value;
    restart();
  });
  const replay = h("button", { class: "btn small", type: "button", onclick: () => restart() }, icon("reset", 13), "Replay");
  const stats = { step: ctx.ui.stat("Step", "0"), loss: ctx.ui.stat("Loss", "—", "train"), start: ctx.ui.stat("Start", "—") };
  const verdict = h("div", { class: "bk-learn-verdict" });

  const map = surface({
    height: 300,
    life,
    className: "bk-gd-map",
    label: "Contour map of a valley-shaped loss surface with the path taken by gradient descent",
    draw(root, width, height) {
      const unit = Math.min(width / 6.4, height / 4);
      const cx = width / 2;
      const cy = height / 2;
      view = { unit, cx, cy };
      const X = (x) => cx + x * unit;
      const Y = (y) => cy - y * unit;
      const clipId = `bk-gd-clip-${Math.round(Math.random() * 1e9)}`;
      root.append(svg("defs", {}, svg("clipPath", { id: clipId }, svg("rect", { x: 0, y: 0, width, height, rx: 10 }))));
      const layer = svg("g", { "clip-path": `url(#${clipId})` });
      layer.append(svg("rect", { class: "bk-gd-ground", x: 0, y: 0, width, height }));
      for (const level of [...VALLEY_LEVELS].reverse()) {
        layer.append(svg("ellipse", {
          class: "bk-gd-contour",
          cx, cy,
          rx: Math.sqrt((2 * level) / VALLEY.a) * unit,
          ry: Math.sqrt((2 * level) / VALLEY.b) * unit,
          transform: `rotate(${-VALLEY.angle} ${cx} ${cy})`,
        }));
      }
      layer.append(svg("path", { class: "bk-gd-target", d: `M${cx - 5},${cy}L${cx + 5},${cy}M${cx},${cy - 5}L${cx},${cy + 5}` }));
      const points = run.path.map(([x, y]) => [X(x), Y(y)]).map(([px, py]) => [Math.max(-4000, Math.min(4000, px)), Math.max(-4000, Math.min(4000, py))]);
      layer.append(svg("path", { class: "bk-gd-path", d: pathFrom(points) }));
      points.slice(1, -1).forEach(([px, py]) => layer.append(svg("circle", { class: "bk-gd-step", cx: px, cy: py, r: 2.4 })));
      const [sx, sy] = points[0];
      layer.append(svg("circle", { class: "bk-gd-start", cx: sx, cy: sy, r: 5 }));
      const [ex, ey] = points[points.length - 1];
      layer.append(svg("circle", { class: run.diverged ? "bk-gd-ball out" : "bk-gd-ball", cx: ex, cy: ey, r: 6.5 }));
      layer.append(svg("text", { class: "bk-gd-hint", x: 12, y: height - 12 }, "Click to drop the ball elsewhere"));
      root.append(layer);
    },
  });
  map.svg.addEventListener("click", (event) => {
    if (!view) return;
    const box = map.svg.getBoundingClientRect();
    const x = (event.clientX - box.left - view.cx) / view.unit;
    const y = -(event.clientY - box.top - view.cy) / view.unit;
    state.start = [Math.max(-3.1, Math.min(3.1, x)), Math.max(-1.95, Math.min(1.95, y))];
    restart();
  });

  const lossPlot = surface({
    height: 128,
    life,
    label: "Loss after each step, on a logarithmic scale",
    draw(root, width, height) {
      const area = { left: 40, right: width - 8, top: 10, bottom: height - 24 };
      const x = linear(0, VALLEY_STEPS, area.left, area.right);
      const y = logScale(1e-6, 1e3, area.bottom, area.top);
      axes(root, {
        x, y, area,
        xTicks: [0, 20, 40, 60],
        yTicks: [1e-6, 1e-3, 1, 1e3],
        yFormat: (value) => (value >= 1 ? fmt.int(value) : value.toExponential(0)),
        xLabel: "step",
      });
      const points = run.losses.map((loss, index) => [x(index), y(Math.max(1e-6, Math.min(1e3, loss)))]);
      root.append(svg("path", { class: "bk-learn-line train", d: pathFrom(points) }));
      const [lx, ly] = points[points.length - 1];
      root.append(svg("circle", { class: "bk-learn-dot train", cx: lx, cy: ly, r: 3.5 }));
    },
  });

  function describe() {
    const name = state.optimiser;
    const final = run.losses[run.losses.length - 1] / run.losses[0];
    const best = Math.min(...run.losses) / run.losses[0];
    if (run.diverged) {
      return { tone: "warn", icon: "alert", text: name === "momentum"
        ? "Diverged. Momentum adds each new step to the speed it already has, so a large rate builds up runaway swings. In a real run the loss would end as NaN and the workbench would stop."
        : "Diverged. Each step overshoots the floor of the valley by more than the step before, so the loss explodes. In a real run the loss would end as NaN and the workbench would stop." };
    }
    if (!run.done) return { tone: "info", icon: "trending-down", text: `Rolling downhill… step ${run.t} of ${VALLEY_STEPS}.` };
    if (name === "adam" && final >= 1e-3 && best < 0.02) {
      return { tone: "info", icon: "info", text: "Close, but jittering. Adam's steps stay roughly as long as the learning rate however gentle the slope, so near the bottom it hops around instead of settling. That is one reason real runs lower the rate towards the end." };
    }
    if (final < 1e-3) {
      if (run.flips >= 8) return { tone: "info", icon: "info", text: `Made it, the hard way. The steps are nearly too big for the steep direction, so the ball bounced across the valley ${run.flips} times on the way down.` };
      return { tone: "info", icon: "circle-check", text: `Reached the bottom: the loss fell a thousandfold by step ${run.reachedAt}.` };
    }
    if (run.flips >= 8) {
      return { tone: "info", icon: "info", text: name === "momentum"
        ? `Swinging. Momentum keeps carrying the ball past the bottom (${run.flips} crossings). It gets close, but with this much speed it takes a long time to settle.`
        : `Zig-zagging. The ball keeps overshooting the narrow direction of the valley (${run.flips} crossings), so most of each step is spent bouncing from wall to wall.` };
    }
    return { tone: "info", icon: "info", text: "Crawling. The steps are so small that after 60 of them the ball is still on the slope. The direction is right; the rate is too timid." };
  }

  function render() {
    stats.step.set(`${run.t} / ${VALLEY_STEPS}`);
    stats.loss.set(run.diverged ? "exploded" : fmt.loss(run.losses[run.losses.length - 1]));
    stats.start.set(fmt.loss(run.losses[0]));
    const { tone, icon: iconName, text } = describe();
    clear(verdict);
    verdict.append(note(text, { tone, iconName }));
    map.redraw();
    lossPlot.redraw();
  }

  fig.body.append(
    h("div", { class: "bk-learn-gd" },
      map.element,
      h("div", { class: "bk-learn-gd-side" },
        h("div", { class: "bk-toolbar" }, optimiserSwitch, h("span", { class: "spacer" }), replay),
        rateControl,
        h("div", { class: "bk-stats bk-learn-stats" }, Object.values(stats)),
        h("span", { class: "panel-label bk-section-label" }, "Loss after each step"),
        lossPlot.element,
      ),
    ),
    verdict,
  );
  restart();
  return fig.element;
}

/* ---------- Figure 4: the learning-rate schedule ---------- */

const SCHEDULES = {
  pretrain: { label: "Pre-training", peak: 6e-4, warmup: 100, steps: 2000, mode: "steps" },
  sft: { label: "Fine-tuning", peak: 1e-4, warmup: 20, steps: 400, mode: "steps" },
  dpo: { label: "Preference tuning", peak: 2e-5, warmup: 6, steps: 60, mode: "until" },
};

const roundSteps = (value) => {
  const digits = 10 ** Math.max(0, Math.floor(Math.log10(value)) - 1);
  return Math.round(value / digits) * digits;
};

function scheduleFigure(ctx, life) {
  const fig = figure({
    number: 4,
    title: "Learning-rate schedule",
    info: "the same formula as the engine",
    tone: "light",
    caption: "The rate the workbench uses at every step: a straight climb during warmup, then half a cosine wave down to a tenth of the peak at the step target. Choose <em>Until I stop</em> and there is no end to aim for, so the rate holds at its peak. Point at the curve to read off a step.",
  });
  const state = { ...SCHEDULES.pretrain, recipe: "pretrain", hover: null };
  const total = () => (state.mode === "steps" ? state.steps : null);
  const lrAt = (step) => learningRateAt(step, state.peak, state.warmup, total());

  const recipeSwitch = ctx.ui.segmented(Object.entries(SCHEDULES).map(([value, recipe]) => ({ value, label: recipe.label })), state.recipe, (value) => {
    Object.assign(state, SCHEDULES[value], { recipe: value });
    syncControls();
    update();
  });
  const modeSwitch = ctx.ui.segmented([{ value: "steps", label: "Step target" }, { value: "until", label: "Until I stop" }], state.mode, (value) => {
    state.mode = value;
    custom();
    update();
  });
  const custom = () => {
    state.recipe = null;
    recipeSwitch.set(null);
  };
  const peakSlider = ctx.ui.slider({ min: -5, max: -2.5, step: 0.01, value: Math.log10(state.peak), label: "Peak learning rate", onInput: (value) => {
    state.peak = Number((10 ** value).toPrecision(2));
    custom();
    update();
  } });
  const warmupSlider = ctx.ui.slider({ min: 0, max: 500, step: 1, value: state.warmup, label: "Warmup steps", onInput: (value) => {
    state.warmup = value;
    custom();
    update();
  } });
  const stepsSlider = ctx.ui.slider({ min: Math.log10(50), max: Math.log10(50000), step: 0.005, value: Math.log10(state.steps), label: "Step target", onInput: (value) => {
    state.steps = roundSteps(10 ** value);
    custom();
    update();
  } });
  const controls = {
    peak: control("Peak rate", peakSlider),
    warmup: control("Warmup", warmupSlider),
    steps: control("Steps shown", stepsSlider),
  };
  const stats = {
    full: ctx.ui.stat("Full rate from", "—"),
    half: ctx.ui.stat("Halfway", "—"),
    last: ctx.ui.stat("Last step", "—"),
    mean: ctx.ui.stat("Average", "—"),
  };
  const explain = h("p", { class: "bk-learn-readout light" });

  function syncControls() {
    recipeSwitch.set(state.recipe);
    modeSwitch.set(state.mode);
    peakSlider.setValue(Math.log10(state.peak));
    warmupSlider.setValue(state.warmup);
    stepsSlider.setValue(Math.log10(state.steps));
  }

  let geometry = null;
  const plot = surface({
    height: 220,
    life,
    label: "Learning rate at each training step",
    draw(root, width, height) {
      const area = { left: 52, right: width - 14, top: 16, bottom: height - 30 };
      const x = linear(0, state.steps, area.left, area.right);
      const y = linear(0, state.peak * 1.12, area.bottom, area.top);
      geometry = { x, area };
      axes(root, {
        x, y, area,
        xTicks: ticks(0, state.steps, Math.max(3, Math.floor((area.right - area.left) / 90))),
        yTicks: ticks(0, state.peak * 1.12, 4),
        xFormat: (value) => fmt.int(value),
        yFormat: (value) => (value === 0 ? "0" : fmt.lr(value)),
        xLabel: "step",
        yLabel: "learning rate",
      });
      const warm = Math.min(state.warmup, state.steps);
      if (warm > 0) {
        root.append(
          svg("rect", { class: "bk-sched-warm", x: area.left, y: area.top, width: Math.max(0, x(warm) - area.left), height: area.bottom - area.top }),
          svg("text", { class: "bk-sched-tag", x: area.left + 6, y: area.top + 13 }, x(warm) - area.left > 60 ? "warmup" : ""),
        );
      }
      if (state.mode === "steps") {
        const floor = state.peak * 0.1;
        root.append(
          svg("line", { class: "bk-learn-ref", x1: area.left, x2: area.right, y1: y(floor), y2: y(floor) }),
          svg("text", { class: "bk-learn-ref-label", x: area.right - 2, y: y(floor) - 6, "text-anchor": "end" }, "floor · 10% of peak"),
        );
      }
      const samples = Math.min(state.steps, 400);
      const points = [];
      for (let index = 0; index <= samples; index += 1) {
        const step = Math.min(state.steps - 1, Math.round((index * state.steps) / samples));
        points.push([x(step), y(lrAt(step))]);
      }
      root.append(svg("path", { class: "bk-sched-line", d: pathFrom(points) }));
      if (state.hover !== null) {
        const step = state.hover;
        const value = lrAt(step);
        const px = x(step);
        const right = px > (area.left + area.right) / 2;
        root.append(
          svg("line", { class: "bk-learn-drop", x1: px, x2: px, y1: area.top, y2: area.bottom }),
          svg("circle", { class: "bk-sched-dot", cx: px, cy: y(value), r: 4.5 }),
          svg("text", { class: "bk-sched-hover", x: px + (right ? -8 : 8), y: area.top + 14, "text-anchor": right ? "end" : "start" }, `step ${fmt.int(step)} · ${fmt.lr(value)}`),
        );
      }
    },
  });
  life.listen(plot.svg, "pointermove", (event) => {
    if (!geometry) return;
    const box = plot.svg.getBoundingClientRect();
    const px = event.clientX - box.left;
    if (px < geometry.area.left || px > geometry.area.right) return;
    state.hover = Math.max(0, Math.min(state.steps - 1, Math.round(geometry.x.invert(px))));
    plot.redraw();
  });
  life.listen(plot.svg, "pointerleave", () => {
    state.hover = null;
    plot.redraw();
  });

  function update() {
    controls.peak.setValue(fmt.lr(state.peak));
    controls.warmup.setValue(`${fmt.int(state.warmup)} steps`);
    controls.steps.setValue(`${fmt.int(state.steps)} steps`);
    const steps = state.steps;
    let sum = 0;
    for (let step = 0; step < steps; step += 1) sum += lrAt(step);
    stats.full.set(state.warmup > 0 ? `step ${fmt.int(Math.min(state.warmup - 1, steps - 1))}` : "step 0");
    stats.half.set(fmt.lr(lrAt(Math.floor(steps / 2))));
    stats.last.set(fmt.lr(lrAt(steps - 1)));
    stats.mean.set(fmt.lr(sum / steps));
    clear(explain);
    if (state.recipe === "dpo") {
      explain.append("Preference tuning in the workbench always runs this way: a warmup of a tenth of its steps (at most 10), then a constant rate. Its rate is thirty times smaller than pre-training's, because it only nudges a model that already works.");
    } else if (state.mode === "until") {
      explain.append("Training until you press Pause leaves the schedule without an end, so after warmup the rate stays at its peak. If you plan to stop at a particular point, a step target usually ends lower, because the last stretch is taken in small, careful steps.");
    } else {
      explain.append(`The rate climbs for ${fmt.int(Math.min(state.warmup, steps))} steps, glides down to ${fmt.lr(state.peak * 0.1)} by step ${fmt.int(steps)}, and averages about ${Math.round((sum / steps / state.peak) * 100)}% of the peak. Raise the step target and resume, and the curve stretches: the rate rises a little again, because the end has moved further away.`);
    }
    plot.redraw();
  }

  fig.body.append(
    h("div", { class: "bk-toolbar" }, recipeSwitch, h("span", { class: "spacer" }), modeSwitch),
    h("div", { class: "bk-controls bk-learn-controls" }, Object.values(controls)),
    plot.element,
    h("div", { class: "bk-stats bk-learn-stats" }, Object.values(stats)),
    explain,
  );
  update();
  return fig.element;
}

/* ---------- Figure 5: a tiny model trained live ---------- */

const LIVE = { steps: 3000, warmup: 100, batch: 32, evalEvery: 50, sampleEvery: 250, prompt: "the " };
const LIVE_RATES = [3e-4, 3e-3, 3e-2, 3e-1];

function trainerFigure(ctx, life) {
  const vocab = makeVocab(TOY_TEXT, TOY_VALIDATION);
  const texts = { full: encode(vocab, TOY_TEXT), tiny: encode(vocab, TOY_TINY) };
  const validation = encode(vocab, TOY_VALIDATION);
  const lnV = Math.log(vocab.size);
  const state = { rate: 3e-3, data: "full" };
  let run;

  const probe = new TinyCharModel({ vocab, seed: 7 });
  const fig = figure({
    number: 5,
    title: "Live training",
    info: `${fmt.int(probe.count)} weights · ${vocab.size} characters`,
    wide: true,
    caption: `A real network, trained in this page with the workbench's recipe: AdamW, gradient clipping, warmup and cosine decay. It reads three characters and predicts the fourth, learning from ${fmt.int(TOY_TEXT.length)} characters of simple sentences (or only the first two, with Two sentences), and is checked every ${LIVE.evalEvery} steps on ${fmt.int(TOY_VALIDATION.length)} characters it never trains on. Samples always use the same random numbers, so every change between them comes from learning.`,
  });

  const stats = {
    step: ctx.ui.stat("Step", "0"),
    train: ctx.ui.stat("Train loss", "—", "train"),
    val: ctx.ui.stat("Val loss", "—", "val"),
    lr: ctx.ui.stat("Rate", "—"),
    ppl: ctx.ui.stat("Perplexity", "—"),
  };
  const samples = h("ol", { class: "bk-learn-samples", "aria-live": "polite" });
  const verdict = h("div", { class: "bk-learn-verdict" });
  const trainButton = h("button", { class: "btn primary small", type: "button" });
  const resetButton = h("button", { class: "btn small", type: "button" }, icon("reset", 13), "Reset");

  function reset() {
    const model = new TinyCharModel({ vocab, seed: 7 });
    const ids = texts[state.data];
    const train = model.evaluate(ids);
    const val = model.evaluate(validation);
    run = {
      model, ids, random: mulberry32(99), step: 0, smooth: train, lr: 0,
      train: [[0, train]], val: [[0, val]], samples: [], best: val, bestStep: 0, peak: Math.max(train, val), broken: false,
    };
    addSample();
  }

  function addSample() {
    run.samples.unshift({ step: run.step, text: run.model.sample(LIVE.prompt, 64, { temperature: 0.8, seed: 2024 }) });
    clear(samples);
    for (const sample of run.samples) {
      samples.append(h("li", {},
        h("span", { class: "bk-learn-sample-step" }, `step ${fmt.int(sample.step)}`),
        h("span", { class: "bk-learn-sample-text" }, h("b", {}, LIVE.prompt), sample.text),
      ));
    }
  }

  function trainOne() {
    const positions = new Array(LIVE.batch);
    for (let index = 0; index < LIVE.batch; index += 1) positions[index] = Math.floor(run.random() * run.ids.length);
    run.lr = learningRateAt(run.step, state.rate, LIVE.warmup, LIVE.steps);
    const loss = run.model.trainStep(run.ids, positions, run.lr);
    if (!Number.isFinite(loss)) {
      run.broken = true;
      return;
    }
    run.step += 1;
    run.smooth = 0.95 * run.smooth + 0.05 * loss;
    run.peak = Math.max(run.peak, run.smooth);
    if (run.step % 10 === 0) run.train.push([run.step, run.smooth]);
    if (run.step % LIVE.evalEvery === 0) {
      const val = run.model.evaluate(validation);
      run.val.push([run.step, val]);
      run.peak = Math.max(run.peak, val);
      if (val < run.best) {
        run.best = val;
        run.bestStep = run.step;
      }
    }
    if (run.step % LIVE.sampleEvery === 0) addSample();
  }

  const finished = () => run.broken || run.step >= LIVE.steps;
  const loop = life.loop(() => {
    const started = performance.now();
    while (!finished() && performance.now() - started < 10) trainOne();
    render();
    return !finished();
  }, () => syncButton());

  function syncButton() {
    clear(trainButton);
    const running = loop.running;
    trainButton.append(icon(running ? "pause" : "play", 13), running ? "Pause" : finished() ? "Train again" : run.step ? "Resume" : "Train");
  }

  trainButton.addEventListener("click", () => {
    if (loop.running) {
      loop.stop();
      return;
    }
    if (finished()) {
      reset();
      render();
    }
    loop.start();
    syncButton();
  });
  resetButton.addEventListener("click", () => {
    loop.stop();
    reset();
    render();
    syncButton();
  });
  const restartWith = (change) => {
    const wasRunning = loop.running;
    loop.stop();
    change();
    reset();
    render();
    if (wasRunning) loop.start();
    syncButton();
  };
  const rateSwitch = ctx.ui.segmented(LIVE_RATES.map((value) => ({ value: String(value), label: String(value) })), String(state.rate), (value) => restartWith(() => {
    state.rate = Number(value);
  }));
  const dataSwitch = ctx.ui.segmented([
    { value: "full", label: "Full text", title: `${fmt.int(TOY_TEXT.length)} characters` },
    { value: "tiny", label: "Two sentences", title: `${TOY_TINY.length} characters: small enough to learn by heart` },
  ], state.data, (value) => restartWith(() => {
    state.data = value;
  }));

  const chart = surface({
    height: 236,
    life,
    label: "Training and validation loss of the live model",
    draw(root, width, height) {
      const area = { left: 34, right: width - 12, top: 14, bottom: height - 28 };
      const top = Math.max(4, Math.ceil(run.peak + 0.3));
      const x = linear(0, LIVE.steps, area.left, area.right);
      const y = linear(0, top, area.bottom, area.top);
      axes(root, {
        x, y, area,
        xTicks: [0, 1000, 2000, 3000],
        yTicks: ticks(0, top, 4),
        xFormat: (value) => (value ? `${value / 1000}k` : "0"),
        xLabel: "step",
        yLabel: "loss",
      });
      root.append(
        svg("line", { class: "bk-learn-ref", x1: area.left, x2: area.right, y1: y(lnV), y2: y(lnV) }),
        svg("text", { class: "bk-learn-ref-label", x: area.right - 2, y: y(lnV) - 6, "text-anchor": "end" }, `random guessing · ln ${vocab.size} ≈ ${lnV.toFixed(2)}`),
        svg("path", { class: "bk-learn-line train", d: pathFrom(run.train.map(([step, loss]) => [x(step), y(loss)])) }),
        svg("path", { class: "bk-learn-line val", d: pathFrom(run.val.map(([step, loss]) => [x(step), y(loss)])) }),
      );
      const [vs, vl] = run.val[run.val.length - 1];
      const [ts, tl] = run.train[run.train.length - 1];
      root.append(
        svg("circle", { class: "bk-learn-dot train", cx: x(ts), cy: y(tl), r: 3.5 }),
        svg("circle", { class: "bk-learn-dot val", cx: x(vs), cy: y(vl), r: 3.5 }),
      );
      if (run.bestStep && vl > run.best + 0.15) {
        root.append(
          svg("circle", { class: "bk-learn-best", cx: x(run.bestStep), cy: y(run.best), r: 5 }),
          svg("text", { class: "bk-learn-ref-label", x: x(run.bestStep) + 8, y: y(run.best) + 16 }, "best validation"),
        );
      }
    },
  });

  function describe() {
    const val = run.val[run.val.length - 1][1];
    if (run.broken) return { tone: "warn", icon: "alert", text: "The loss became NaN. The workbench stops a run when this happens and suggests a lower learning rate or a longer warmup." };
    if (run.step === 0) return { tone: "info", icon: "info", text: `Untrained, the model spreads its bets almost evenly over ${vocab.size} characters, so both losses sit on the random-guessing line and the sample is gibberish. Press Train.` };
    if (run.step > 300 && val > run.best + 0.15 && run.smooth < val - 0.3) {
      return { tone: "warn", icon: "alert", text: `Overfitting. The training loss keeps falling, but the validation loss has been rising since step ${fmt.int(run.bestStep)}${val > lnV ? " and is now worse than random guessing: the model has become confidently wrong about anything it has not memorised" : ""}. Look at the samples: it is reciting its text.` };
    }
    if (run.step > LIVE.warmup && run.smooth > lnV + 0.1) {
      return { tone: "warn", icon: "alert", text: "Too big a step. The loss has climbed above random guessing: each update overshoots and undoes the one before. Clipping keeps the numbers finite, and the decaying rate may rescue the run late on, but these steps are wasted." };
    }
    if (finished()) {
      return { tone: "info", icon: "circle-check", text: `Finished ${fmt.int(LIVE.steps)} steps. Best validation loss ${run.best.toFixed(2)} at step ${fmt.int(run.bestStep)}.${state.rate < 1e-3 ? " The steps were small, so the run ended long before the model learned what it could: try 0.003." : ""}` };
    }
    return { tone: "info", icon: "trending-down", text: "Learning. Both curves are falling together, so what the model picks up from its training text also helps on text it has never seen." };
  }

  function render() {
    const val = run.val[run.val.length - 1][1];
    stats.step.set(`${fmt.int(run.step)} / ${fmt.int(LIVE.steps)}`);
    stats.train.set(fmt.loss(run.smooth));
    stats.val.set(fmt.loss(val));
    stats.lr.set(run.step ? fmt.lr(run.lr) : "—");
    stats.ppl.set(Math.exp(val).toFixed(1));
    const { tone, icon: iconName, text } = describe();
    clear(verdict);
    verdict.append(note(text, { tone, iconName }));
    chart.redraw();
  }

  fig.body.append(
    h("div", { class: "bk-toolbar" },
      trainButton, resetButton,
      h("span", { class: "spacer" }),
      h("span", { class: "panel-label" }, "Peak rate"), rateSwitch,
      h("span", { class: "panel-label" }, "Text"), dataSwitch,
    ),
    h("div", { class: "bk-stats bk-learn-stats" }, Object.values(stats)),
    h("div", { class: "bk-learn-live" },
      h("div", {},
        chart.element,
        h("div", { class: "bk-learn-key" },
          h("span", {}, h("span", { class: "bk-dot train" }), "Training loss (smoothed)"),
          h("span", {}, h("span", { class: "bk-dot val" }), "Validation loss"),
        ),
      ),
      h("div", { class: "bk-learn-sample-box" },
        h("span", { class: "panel-label bk-section-label" }, `Samples · prompt “${LIVE.prompt.trim()}”`),
        samples,
      ),
    ),
    verdict,
  );
  reset();
  render();
  syncButton();
  return fig.element;
}

/* ---------- chapter ---------- */

export default {
  id: "how-models-learn",
  title: "How models learn",
  render(root, ctx) {
    const life = lifecycle();
    const shell = chapterShell(root, "how-models-learn",
      "A new model is millions of random numbers that predict nothing well. Training turns those numbers into a model that writes, one small correction at a time. This chapter follows a single training step from start to finish (the score, the slope, the nudge) and then shows how to read a training run as it unfolds.");
    const { article } = shell;

    article.append(prose(`
      <h2>Keeping score</h2>
      <p>Chapter 5 ended with a model whose every table and matrix is filled with small random numbers. Hand it a run of tokens and, at every position, it still produces a forecast: a probability for each token in its vocabulary. The forecast is just not any good yet. To improve it, training first needs a way to say <em>how</em> bad it is, as a single number that can be pushed down.</p>
      <p>The score every language model uses is simple. Look up the probability the model gave to the token that really came next, and take its negative logarithm:</p>
      <div class="bk-eq">loss = −ln <span class="bk-var">p</span>(the token that really came next)<span class="bk-eq-note">averaged over every position in the batch; called the cross-entropy loss</span></div>
      <p>A model that was certain and right scores 0. One that gave the right token even odds scores 0.69. One that gave it a one-in-a-hundred chance scores 4.6, and the score grows without limit as the probability approaches zero. You can think of the loss as <strong>surprise</strong>: how surprised the model was by what actually happened. The logarithm makes a confident mistake far more expensive than a hesitant one, which is exactly what we want, because a model that is sure of the wrong thing is worse than one that admits it does not know.</p>
    `));
    article.append(surpriseFigure(ctx, life));
    article.append(prose(`
      <p>The dashed line in Figure 1 explains the first number you will see on the Pre-training page. A freshly created model knows nothing, so its forecast is close to an even spread over the vocabulary: every token gets a probability of about 1/<span class="bk-var">V</span>, and the loss at every position is about ln <span class="bk-var">V</span>. For the workbench's tokenizers that is 8.32 with 4,096 tokens, 9.21 with 10,000, 9.68 with 16,000 and 10.8 with GPT-2's vocabulary. A run that starts far above that line has a problem; one that starts on it is behaving exactly as it should.</p>
      <p>Another common way to quote the same number is <strong>perplexity</strong>, which is simply <span class="bk-var">e</span> raised to the loss. It turns surprise back into a count: a perplexity of 20 means the model is, on average, as unsure as if it were choosing blindly among 20 equally likely tokens. A loss of 2.0, typical of a small model that has trained for a while on simple stories, is a perplexity of about 7.4.</p>

      <h2>The answers are already in the text</h2>
      <p>Most machine learning needs labelled examples: photos tagged "cat" or "dog", emails marked as spam. Language models get their labels for free. Take any run of tokens from the corpus: the input at each position is the token there, and the correct answer, the <strong>target</strong>, is simply the token that follows. Inputs and targets are the same text, shifted by one place. This trick is called <em>self-supervised</em> learning, and it is why a model can learn from billions of words that nobody ever annotated.</p>
    `));
    article.append(shiftFigure(ctx, life));
    article.append(prose(`
      <p>A training window holds as many tokens as the model's context, and every position in it is graded at once. The causal mask from Chapter 5 is what makes that honest: each position can only look to its left, so none of them can peek at its own answer. The workbench reads its corpus with a special end-of-text token between documents, so that predicting where a story finishes is part of the job too.</p>
      <p>One window is still a small sample, so each training step reads a <strong>batch</strong> of windows taken from random places in the corpus. With the Pocket shape's 128-token context and the default batch of 16, a single step makes and grades 2,048 predictions, and the loss shown for that step is their average.</p>

      <h2>Which way is downhill?</h2>
      <p>The loss depends on every weight in the model. Picture a landscape in which each possible setting of the weights is a place, and the loss there is the height of the ground. Training is a walk downhill. The walker cannot see the landscape (with millions of weights it has millions of dimensions) but it can feel the slope under its feet. That slope is the <strong>gradient</strong>: for every single weight, how much the loss would rise if that weight were nudged up a tiny amount. The gradient points uphill, so each training step moves every weight a little in the opposite direction:</p>
      <div class="bk-eq"><span class="bk-var">w</span> ← <span class="bk-var">w</span> − <span class="bk-var">η</span> × <span class="bk-frac"><span>∂ loss</span><span>∂ <span class="bk-var">w</span></span></span><span class="bk-eq-note">every weight steps against its own slope; η, the learning rate, sets the size of the step</span></div>
      <p>Measuring millions of slopes one weight at a time would take forever. <strong>Backpropagation</strong> gets all of them in one sweep. It starts at the loss and works backwards through the model, layer by layer, using the chain rule from calculus: the output layer works out how the loss depends on its own inputs and weights, hands that "blame" back to the layer below, which does the same, all the way down to the embedding table. Libraries like PyTorch, which the workbench's engine uses, do this bookkeeping automatically.</p>
      <p>The backward pass costs about twice as much as the forward pass, so a training step costs roughly three times as much as simply running the model. That is where the rule of thumb behind the workbench's time estimates comes from: training takes about six floating-point operations per weight for every token read, two forwards and four backwards.</p>
    `));
    article.append(descentFigure(ctx, life));
    article.append(prose(`
      <h2>How big a step?</h2>
      <p>The gradient says which way to go; the <strong>learning rate</strong> says how far. It is the most important setting in any training run, and Figure 3 shows why. Too small, and the model creeps downhill so slowly that the run ends long before it has learned what it could. Too large, and each step overshoots the bottom of the valley and lands on the opposite wall, higher than before. A little too large, and the loss zig-zags and wastes steps; much too large, and it explodes. Real loss surfaces are full of valleys that are steep in some directions and nearly flat in others, so the steep directions set the limit and the flat directions set the pace.</p>
      <p>When the loss explodes in a real run it soon becomes <code>NaN</code>, "not a number", and nothing can be learned from there. The workbench watches for this and stops the run with a hint to lower the learning rate or lengthen the warmup; you can then change the recipe and resume from the last checkpoint. Its defaults are a peak rate of 0.0006 for pre-training, 0.0001 for fine-tuning and 0.00002 for preference tuning: each stage starts from a better model and changes it more gently.</p>

      <h2>Warm up, then slow down</h2>
      <p>The workbench does not use one rate for the whole run. At the very start the weights are random, the gradients are large and erratic, and the optimiser's running averages (more on those below) have seen only a handful of batches, so taking full-size steps would be reckless. During the <strong>warmup</strong> the rate climbs in a straight line from almost nothing to its peak; pre-training uses the first 100 steps. After that the rate glides back down along half a cosine wave and ends at a tenth of the peak on the final step. Late in training, the model is near the bottom of a valley, and smaller steps let it settle there instead of bouncing from wall to wall.</p>
    `));
    article.append(scheduleFigure(ctx, life));
    article.append(prose(`
      <h2>Noisy steps</h2>
      <p>The loss the model really cares about is its average over all the text it could ever read. Computing that for every step would mean reading the whole corpus each time, so instead each step estimates the slope from one batch. Different batches give slightly different gradients, which makes every step a little noisy: this is called <strong>stochastic gradient descent</strong>. The noise is mostly harmless. Over many steps the random errors cancel out, and the jostling can even shake the model out of poor spots. It is also why the training loss on the Pre-training page wobbles from step to step while the overall trend falls. A bigger batch averages over more text and gives steadier steps, but it costs more memory and more time per step.</p>

      <h2>Smarter steps: AdamW and clipping</h2>
      <p>Plain gradient descent uses the same rate for every weight. Nearly all language models, the workbench's included, use a refinement called <strong>Adam</strong> (Kingma and Ba, 2014), in its <strong>AdamW</strong> variant (Loshchilov and Hutter, 2017). It adds three ideas:</p>
      <ul>
        <li><strong>Momentum.</strong> Instead of following each noisy gradient, follow a running average of recent ones. Directions that keep coming up add together, while random wobbles cancel. The workbench averages with a factor of 0.9, so the last ten or so steps count most.</li>
        <li><strong>A step size for every weight.</strong> Adam also keeps a running average of each weight's <em>squared</em> gradient (with a factor of 0.95 here) and divides the step by its square root. Weights whose gradients are large and jumpy take careful steps; weights that are rarely touched, such as the embeddings of uncommon tokens, still move when they do get a signal. A handy side effect is that each weight moves by roughly the learning rate on every step, whatever the scale of its gradient.</li>
        <li><strong>Weight decay</strong>, the W. On every step each weight is also shrunk slightly towards zero, by a tenth of the learning rate, so weights only stay large if the data keeps asking for it. Models trained this way tend to cope better with text they have not seen. Following GPT-2-style practice, the workbench applies decay to the weight matrices and embeddings, but not to biases and layer-norm parameters.</li>
      </ul>
      <p>The running averages are why training needs so much more memory than the model itself: next to each weight sit its gradient and two averages, four copies of the model in all, which the memory estimate on the Project setup page includes.</p>
      <p>One more safety catch is <strong>gradient clipping</strong>. Now and then an odd batch produces a gradient far larger than usual, which could fling the weights somewhere useless in a single step. Before every update the workbench measures the overall length of the gradient, taken across all weights at once, and if it is longer than 1.0, scales it down to exactly 1.0. The direction is kept; only the size of the step is capped.</p>
    `));
    article.append(callout("try", "In Figure 3, set the learning rate to about 0.4 and switch between the three optimisers. Plain steps diverge. Momentum swings wide before it settles. Adam never explodes, because its steps stay about as long as the rate, but at large rates it jitters around the bottom instead of stopping there. That is one more reason to lower the rate towards the end of a run."));
    article.append(prose(`
      <h2>Watching a model learn</h2>
      <p>Everything so far comes together in Figure 5: a real, if very small, neural network, trained right here in your browser with the same recipe as the workbench. Instead of tokens it predicts single characters, from the three before. Press <strong>Train</strong> and watch three things at once: the loss falling from the random-guessing line, the gap between the training and validation curves, and the samples changing from noise to letter patterns to recognisable words.</p>
    `));
    article.append(trainerFigure(ctx, life));
    article.append(prose(`
      <p>The early part of every run looks much the same. First the model learns which characters are common at all (spaces, <em>e</em>, <em>t</em>), and the loss drops steeply. Then it learns which characters follow which, and short common words appear. Longer-range patterns, like the shape of a sentence, come last and slowest. Real models go through the same stages with tokens instead of characters: frequent words first, then grammar, then facts and plots. Try the four learning rates: the smallest crawls, the two middle ones learn well, and the largest climbs above the random-guessing line before the shrinking schedule rescues it.</p>

      <h2>Learning or memorising?</h2>
      <p>A low training loss on its own proves little. A model large enough to memorise its training text can score almost zero on it without having learned anything that carries over. That is why the workbench sets aside some text the model never trains on (one document in fifty, as Chapter 6 described) and measures the <strong>validation loss</strong> on it at regular intervals: every 100 steps during pre-training, averaged over six batches.</p>
      <p>While the two curves fall together, the model is learning patterns that hold beyond its own training text. When the validation loss levels off or turns upwards while the training loss keeps falling, the model has started to <strong>overfit</strong>: it is memorising the particular text it reads instead of the language behind it. Switch Figure 5 to <strong>Two sentences</strong> to see this happen fast. The model learns the two sentences by heart, its training loss heads for zero, and its validation loss soars past the random-guessing line, because it has become confidently wrong about anything else.</p>
      <p>The cures are the ones Chapter 6 hinted at. More text helps most. Fewer passes over the same text help too, which is why the Pre-training page shows how many times your run will read the corpus. A smaller model has less room to memorise. With a corpus of several megabytes and a run of a few thousand steps, the gap between the curves usually stays small; with a tiny corpus and a long run, watch it closely.</p>

      <h2>Checkpoints</h2>
      <p>A training run can last hours, so the workbench regularly saves a <strong>checkpoint</strong>: the weights, the optimiser's running averages and the step count, everything needed to carry on exactly where it left off. During pre-training it saves every 250 steps by default and at least every ten minutes, at the end of a run, and whenever you press Pause. Only the latest checkpoint of each stage is kept. If the computer crashes or runs out of battery, at most the steps since the last checkpoint are lost, and <strong>Resume</strong> picks up from there with the optimiser state intact. The workbench also writes a short sample at regular intervals, and the Sample timeline collects them so you can see the writing improve, just like the samples in Figure 5.</p>
    `));
    article.append(callout("linux", `Training is the heaviest job the workbench does. Pocket trains comfortably on a phone's CPU; on a laptop or desktop, the <strong>Calibrate</strong> button on the Project setup page times a few real training steps on each device it finds (the CPU, and a GPU through CUDA, Metal or Vulkan) so that <strong>Auto</strong> can use the faster one. By default a run keeps going when you close the window, and Pause always saves a checkpoint first, so it is safe to stop and resume a long run whenever you need the machine back.`));
    article.append(prose(`
      <h2>Reading the Pre-training page</h2>
      <p>The Training run card on the Pre-training page shows the same quantities as Figure 5, for your own model:</p>
      <ul>
        <li><strong>Step</strong> counts optimiser updates. Each one reads one batch.</li>
        <li><strong>Loss</strong> is the training loss, and <strong>Val</strong> the latest validation loss. Expect them to start near ln <span class="bk-var">V</span> (8.32 for the 4K tokenizer), fall steeply for the first few hundred steps, then keep falling more and more slowly.</li>
        <li><strong>LR</strong> is the learning rate at the current step, following the schedule in Figure 4.</li>
        <li><strong>Tok/s</strong> is how many tokens of text your machine reads per second, and <strong>Time</strong> how long the run has been going.</li>
      </ul>
      <p>Some things are worth watching for. A loss that barely moves from its starting value after a few hundred steps suggests the learning rate is too low. Sudden spikes that do not settle suggest it is too high. A validation curve that turns upwards while the training curve keeps falling means overfitting. And a loss value only means something next to others measured the same way: a tokenizer with a bigger vocabulary cuts text into longer pieces, each harder to guess, so its loss per token is naturally higher. Compare runs that share a tokenizer and a validation set, not runs that do not.</p>
    `));
    article.append(callout("key", "Training repeats one small loop thousands of times: predict the next token at every position of a batch, score the predictions with the cross-entropy loss, use backpropagation to find the slope of that loss for every weight, and take a small step downhill with AdamW. The learning rate and its schedule decide how big each step is, and the validation loss tells you whether the model is learning the language or just memorising its text."));
    finishChapter(shell, ctx, { id: "how-models-learn-behavior", title: "How models learn behavior", blurb: "From a model that continues text to an assistant that answers: fine-tuning on conversations, and tuning on preferences." });
    return () => life.dispose();
  },
};
