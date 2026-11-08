// Chapter 8: How models learn behavior. From a model that continues text to one that answers: supervised
// fine-tuning on conversations, the chat template and its masked loss, and preference tuning with DPO.
import { h, clear, svg, tokenLabel } from "../dom.js";
import { icon } from "../icons.js";
import { chapterShell, finishChapter, prose, callout, figure, lifecycle, control, note, fmt } from "../book/kit.js";
import { surface, axes, linear, pathFrom } from "../book/plot.js";
import { pretokenize } from "../book/bpe.js";
import { dpoLoss, tunePreferences, sigmoid } from "../book/learning.js";

const DEFAULT_SYSTEM = "You are a friendly assistant. Give clear, short answers.";
const LN2 = Math.log(2);
const special = (name, role = false) => `<span class="bk-chip-inline${role ? "" : " orange"}">&lt;|${name}|&gt;</span>`;
const percentOf = (value) => `${Math.round(value * 100)}%`;
const signed = (value, digits = 1) => {
  const rounded = Number(value.toFixed(digits));
  if (rounded === 0) return (0).toFixed(digits);
  return `${rounded > 0 ? "+" : "−"}${Math.abs(rounded).toFixed(digits)}`;
};

/* ---------- the same message, sent to two models ---------- */

function versus() {
  const reply = (label, kind, text) => h("div", { class: ["bk-beh-reply", kind] },
    h("span", { class: "bk-label" }, label),
    h("p", {}, text),
  );
  return h("div", { class: "bk-beh-versus", role: "group", "aria-label": "The same message sent to a base model and to a fine-tuned model" },
    h("div", { class: "bk-beh-ask" }, h("span", { class: "bk-label" }, "You type"), h("p", {}, "What do cows eat?")),
    reply("A base model continues", "base", "“What do cows eat?” asked Lily, pressing her nose to the fence. Her grandfather laughed. “Come with me,” he said, and the two of them walked out into the long grass, where the herd…"),
    reply("A fine-tuned model answers", "tuned", "Mostly grass, and hay in the winter. Many farmers also give them grain or other feed to help them grow."),
  );
}

/* ---------- Figure 1: the chat template ---------- */

const TEMPLATE_EXAMPLES = [
  { value: "egg", label: "Cooking", user: "How long should I boil an egg?", assistant: "About seven minutes for a firm yolk, or five if you like it runny." },
  { value: "moon", label: "Science", user: "Why does the moon change shape?", assistant: "It doesn't. We just see more or less of its sunlit half as it travels around the Earth." },
  { value: "story", label: "Story", user: "Start a story about a sleepy dragon.", assistant: "Once upon a time, a dragon called Ember yawned so widely that she blew out her own fire." },
];
const FIELD_LIMIT = 220;
// Same rule as the engine's tokenizer: typed "<|" never becomes a special token.
const sanitize = (text) => text.replaceAll("<|", "<\u200b|");

// Pieces of a single-exchange conversation. Each piece: text, whether it is a special token, and its
// role in the figure: ctx (read, not graded), target (graded), gen (written by the model), stop.
function templatePieces(messages, chatTime) {
  const pieces = [];
  for (const { role, content } of messages) {
    const reply = role === "assistant";
    pieces.push({ text: `<|${role}|>`, special: true, state: "ctx" }, { text: "\n", state: "ctx" });
    const body = pretokenize(content.trim());
    if (reply && chatTime) {
      for (const text of body) pieces.push({ text, state: "gen" });
      pieces.push({ text: "<|end|>", special: true, state: "stop" });
      break;
    }
    for (const text of body) pieces.push({ text, state: reply ? "target" : "ctx" });
    pieces.push({ text: "<|end|>", special: true, state: reply ? "target" : "ctx" }, { text: "\n", state: "ctx" });
  }
  return pieces;
}

function templateFigure(ctx) {
  const fig = figure({
    number: 1,
    title: "Chat template",
    info: "one conversation · one stream of tokens",
    caption: `One conversation as the model receives it. Every turn opens with a role token and closes with ${special("end")}. Edit any of the three messages: in a training example only the blue tokens (the reply and its ${special("end")}) are learning targets. Switch to <em>At chat time</em> to see where the model takes over and where it stops. Words are split roughly the way a tokenizer would split them; your project's tokenizer may cut rare words into more pieces.`,
  });
  const first = TEMPLATE_EXAMPLES[0];
  const state = { system: DEFAULT_SYSTEM, user: first.user, assistant: first.assistant, chatTime: false };
  const inputs = {};

  const examples = ctx.ui.segmented(TEMPLATE_EXAMPLES.map(({ value, label }) => ({ value, label })), first.value, (value) => {
    const example = TEMPLATE_EXAMPLES.find((item) => item.value === value);
    Object.assign(state, { system: DEFAULT_SYSTEM, user: example.user, assistant: example.assistant });
    for (const [key, input] of Object.entries(inputs)) input.value = state[key];
    update();
  });
  const mode = ctx.ui.segmented([
    { value: "train", label: "Training example" },
    { value: "chat", label: "At chat time" },
  ], "train", (value) => {
    state.chatTime = value === "chat";
    update();
  });

  const field = (key, label, multiline) => {
    const attributes = {
      class: ["text-input", "bk-beh-field", multiline ? "bk-textarea" : ""],
      maxlength: FIELD_LIMIT,
      spellcheck: "false",
      "aria-label": label,
      oninput: () => {
        state[key] = inputs[key].value;
        examples.set(null);
        update();
      },
    };
    inputs[key] = multiline
      ? h("textarea", { ...attributes, rows: 2 }, state[key])
      : h("input", { ...attributes, type: "text", value: state[key] });
    return control(label, inputs[key]);
  };
  const fields = h("div", { class: "bk-beh-fields" },
    field("system", "System prompt", false),
    field("user", "User", true),
    field("assistant", "Assistant", true),
  );

  const stats = {
    tokens: ctx.ui.stat("Tokens", "—"),
    context: ctx.ui.stat("Read as context", "—"),
    graded: ctx.ui.stat("Learning targets", "—", "train"),
  };
  const relabel = (stat, text) => {
    stat.querySelector(".panel-label").textContent = text;
  };
  const stream = h("div", { class: "bk-tokens bk-beh-stream", role: "img", "aria-label": "The conversation as a stream of tokens" });
  const legend = h("div", { class: "bk-beh-legend" });
  const readout = h("p", { class: "bk-learn-readout" });
  const raw = h("pre", { class: "bk-beh-raw", "aria-label": "The raw text of the conversation" });

  const chip = (piece) => h("span", { class: ["token", piece.special ? "special" : "", `bk-beh-${piece.state}`] }, tokenLabel(piece.text));
  const legendItem = (piece, text) => h("span", {}, chip(piece), text);

  function update() {
    const messages = [
      { role: "system", content: state.system },
      { role: "user", content: state.user },
      { role: "assistant", content: state.assistant },
    ];
    const pieces = templatePieces(messages, state.chatTime);
    const count = (kind) => pieces.filter((piece) => piece.state === kind).length;
    clear(stream);
    pieces.forEach((piece, index) => {
      if (state.chatTime && piece.state !== "ctx" && pieces[index - 1]?.state === "ctx") {
        stream.append(h("span", { class: "bk-beh-handover" }, icon("chevron-right", 12), "model writes"));
      }
      stream.append(chip(piece));
    });

    const total = pieces.length;
    stats.tokens.set(fmt.int(total));
    clear(legend);
    clear(readout);
    if (state.chatTime) {
      const prompt = count("ctx");
      relabel(stats.context, "Sent as the prompt");
      relabel(stats.graded, "Written by the model");
      stats.context.set(fmt.int(prompt));
      stats.graded.set(fmt.int(total - prompt));
      legend.append(
        legendItem({ text: "Hi", state: "ctx" }, "Prompt the app sends"),
        legendItem({ text: " Hi", state: "gen" }, "Written by the model"),
        legendItem({ text: "<|end|>", special: true, state: "stop" }, "Its stop signal"),
      );
      readout.append(
        h("b", {}, "At chat time "),
        `the app builds the same template up to `,
        h("span", { class: "special-chip role" }, "<|assistant|>"),
        ` and a new line (${fmt.int(prompt)} tokens here), then lets the model write one token at a time. It stops as soon as the model writes `,
        h("span", { class: "special-chip" }, "<|end|>"),
        ". The model learned to write that token because every reply it was fine-tuned on ended with one.",
      );
    } else {
      const graded = count("target");
      relabel(stats.context, "Read as context");
      relabel(stats.graded, "Learning targets");
      stats.context.set(fmt.int(total - graded));
      stats.graded.set(`${fmt.int(graded)} · ${percentOf(graded / Math.max(1, total))}`);
      legend.append(
        legendItem({ text: "Hi", state: "ctx" }, "Context: read, not graded"),
        legendItem({ text: " Hi", state: "target" }, "Learning target"),
        legendItem({ text: "<|end|>", special: true, state: "target" }, "Graded special token"),
      );
      readout.append(
        h("b", {}, `The model reads all ${fmt.int(total)} tokens `),
        `but is graded on only ${fmt.int(graded)} of them: the reply and the `,
        h("span", { class: "special-chip" }, "<|end|>"),
        ` that closes it. The other ${fmt.int(total - graded)} positions have their targets set to −100, which the loss skips, so the model learns how the assistant answers without learning to imitate the person asking.`,
      );
    }
    if ([state.system, state.user, state.assistant].some((text) => text.includes("<|"))) {
      readout.append(" Typed markers stay ordinary characters: a “<|” you type is never a special token, so nobody can fake a turn by typing one.");
    }

    clear(raw);
    const safe = (text) => sanitize(text.trim());
    raw.append(ctx.ui.specialText(`<|system|>\n${safe(state.system)}<|end|>\n<|user|>\n${safe(state.user)}<|end|>\n<|assistant|>\n`));
    const reply = h("span", { class: state.chatTime ? "bk-beh-raw-gen" : "bk-beh-raw-target" }, ctx.ui.specialText(`${safe(state.assistant)}<|end|>`));
    raw.append(reply, state.chatTime ? "" : "\n");
  }

  fig.body.append(
    h("div", { class: "bk-toolbar bk-learn-toolbar" }, h("span", { class: "panel-label" }, "Example"), examples, h("span", { class: "spacer" }), mode),
    fields,
    h("div", { class: "bk-stats bk-learn-stats" }, Object.values(stats)),
    stream,
    legend,
    readout,
    h("span", { class: "panel-label bk-beh-raw-label" }, "The same conversation as raw text"),
    raw,
  );
  update();
  return fig.element;
}

/* ---------- Figure 2: the DPO loss ---------- */

const DPO_PRESETS = [
  { value: "start", label: "First step", chosen: 0, rejected: 0 },
  { value: "learning", label: "Learning", chosen: 1.5, rejected: -1.5 },
  { value: "both", label: "Both fall", chosen: -2, rejected: -5 },
  { value: "backwards", label: "Backwards", chosen: -1.5, rejected: 1.5 },
];
const MARGIN_RANGE = 12;
const LOSS_TOP = 4;

const likelihood = (ratio) => {
  if (Math.abs(ratio) < 0.05) return "as likely as before";
  const times = Math.exp(Math.abs(ratio));
  return `${times < 10 ? times.toFixed(1) : fmt.int(Math.round(times))}× ${ratio > 0 ? "more" : "less"} likely`;
};

function dpoFigure(ctx, life) {
  const fig = figure({
    number: 2,
    title: "DPO loss explorer",
    info: "one preference pair",
    caption: "The DPO loss for a single preference pair. Each answer's slider is a log-ratio: how much more (or less) likely the model being tuned makes that answer than the frozen reference copy does, summed over the answer's tokens. Only the gap between the two, the margin, matters. β sets how quickly the loss is satisfied, and so how far the model has to move.",
  });
  const state = { chosen: 1.5, rejected: -1.5, beta: 0.1 };
  const stats = {
    margin: ctx.ui.stat("Margin", "—"),
    loss: ctx.ui.stat("Loss", "—", "train"),
    prefer: ctx.ui.stat("Prefers chosen", "—"),
    push: ctx.ui.stat("Push from this pair", "—", "val"),
  };
  const matchPreset = () => DPO_PRESETS.find((preset) => Math.abs(preset.chosen - state.chosen) < 1e-9 && Math.abs(preset.rejected - state.rejected) < 1e-9)?.value ?? null;
  const presets = ctx.ui.segmented(DPO_PRESETS.map(({ value, label }) => ({ value, label })), "learning", (value) => {
    const preset = DPO_PRESETS.find((item) => item.value === value);
    state.chosen = preset.chosen;
    state.rejected = preset.rejected;
    chosenSlider.setValue(state.chosen);
    rejectedSlider.setValue(state.rejected);
    update();
  });
  const ratioSlider = (key, label) => ctx.ui.slider({
    min: -6, max: 6, step: 0.1, value: state[key], label,
    onInput: (value) => {
      state[key] = value;
      presets.set(matchPreset());
      update();
    },
  });
  const chosenSlider = ratioSlider("chosen", "Chosen answer: log-ratio against the reference");
  const rejectedSlider = ratioSlider("rejected", "Rejected answer: log-ratio against the reference");
  const betaSlider = ctx.ui.slider({
    min: 0.01, max: 0.5, step: 0.01, value: state.beta, label: "Beta",
    onInput: (value) => {
      state.beta = value;
      update();
    },
  });
  const chosenControl = control("Chosen answer", chosenSlider, { hint: " " });
  const rejectedControl = control("Rejected answer", rejectedSlider, { hint: " " });
  const betaControl = control("β (beta)", betaSlider, { hint: "Same range as the workbench's Beta slider; its default is 0.1." });
  const readout = h("p", { class: "bk-learn-readout" });

  const lossAt = (margin, beta) => dpoLoss(margin, 0, beta).loss;
  // The curve enters the plot where the loss drops below LOSS_TOP.
  const curve = (beta, x, y) => {
    const entry = -Math.log(Math.expm1(LOSS_TOP)) / beta;
    const start = Math.max(-MARGIN_RANGE, entry);
    const points = [];
    for (let index = 0; index <= 160; index += 1) {
      const margin = start + ((MARGIN_RANGE - start) * index) / 160;
      points.push([x(margin), y(lossAt(margin, beta))]);
    }
    return pathFrom(points);
  };
  const plot = surface({
    height: 220,
    life,
    label: "The DPO loss as a function of the margin",
    draw(root, width, height) {
      const area = { left: 34, right: width - 14, top: 14, bottom: height - 30 };
      const x = linear(-MARGIN_RANGE, MARGIN_RANGE, area.left, area.right);
      const y = linear(0, LOSS_TOP, area.bottom, area.top);
      axes(root, {
        x, y, area,
        xTicks: [-12, -8, -4, 0, 4, 8, 12],
        yTicks: [0, 1, 2, 3, 4],
        xFormat: (value) => (value < 0 ? `−${-value}` : value > 0 ? `+${value}` : "0"),
        xLabel: "margin: chosen − rejected",
        yLabel: "loss",
      });
      root.append(
        svg("line", { class: "bk-learn-ref", x1: area.left, x2: area.right, y1: y(LN2), y2: y(LN2) }),
        svg("text", { class: "bk-learn-ref-label", x: area.right - 2, y: y(LN2) - 6, "text-anchor": "end" }, "first step · ln 2 ≈ 0.69"),
      );
      if (Math.abs(state.beta - 0.1) > 1e-9) root.append(svg("path", { class: "bk-beh-ghost", d: curve(0.1, x, y) }));
      root.append(svg("path", { class: "bk-learn-curve", d: curve(state.beta, x, y) }));
      const margin = state.chosen - state.rejected;
      const loss = lossAt(margin, state.beta);
      const px = x(margin);
      const py = y(Math.min(LOSS_TOP, loss));
      const flip = px > area.right - 110;
      root.append(
        svg("line", { class: "bk-learn-drop", x1: px, x2: px, y1: py, y2: area.bottom }),
        svg("circle", { class: "bk-learn-marker", cx: px, cy: py, r: 5.5 }),
        svg("text", { class: "bk-learn-marker-label", x: px + (flip ? -10 : 10), y: Math.max(area.top + 10, py - 8), "text-anchor": flip ? "end" : "start" }, `loss ${loss.toFixed(2)}`),
      );
    },
  });
  const key = h("div", { class: "bk-learn-key" },
    h("span", {}, h("i", { class: "bk-beh-key-line" }), "β you chose"),
    h("span", { class: "bk-beh-key-ghost-wrap" }, h("i", { class: "bk-beh-key-line ghost" }), "β 0.1, the workbench default"),
  );
  const ghostKey = key.lastChild;

  function update() {
    const { margin, loss, weight, prefer } = dpoLoss(state.chosen, state.rejected, state.beta);
    for (const [element, ratio] of [[chosenControl, state.chosen], [rejectedControl, state.rejected]]) {
      element.setValue(signed(ratio));
      element.querySelector(".bk-control-hint").textContent = Math.abs(ratio) < 0.05 ? "as likely as under the reference" : `${likelihood(ratio)} than under the reference`;
    }
    betaControl.setValue(state.beta.toFixed(2));
    stats.margin.set(signed(margin));
    stats.loss.set(loss.toFixed(3));
    stats.prefer.set(percentOf(prefer));
    stats.push.set(percentOf(weight));
    ghostKey.hidden = Math.abs(state.beta - 0.1) < 1e-9;
    clear(readout);
    if (Math.abs(margin) < 0.05) {
      readout.append(h("b", {}, "No gap, no preference. "), `Both answers have moved by the same amount, so the margin is zero and the loss is ln 2 ≈ 0.693 whatever β is. That is its value at the very first step, when the tuned copy is still identical to the reference.`);
    } else if (margin > 0 && state.chosen < -0.05) {
      readout.append(h("b", {}, "Both answers fell. "), `The chosen answer is now ${likelihood(state.chosen)} and the rejected one ${likelihood(state.rejected)}, yet the loss is a contented ${loss.toFixed(2)}, because the rejected answer fell further. DPO scores only the gap, never whether the chosen answer itself became more likely.`);
    } else if (margin > 0) {
      const chosenText = Math.abs(state.chosen) < 0.05 ? "exactly as likely as under the reference" : `${likelihood(state.chosen)} than under the reference`;
      readout.append(h("b", {}, "Learning. "), `The chosen answer is ${chosenText}, and the rejected one is ${likelihood(state.rejected)}. A margin of ${signed(margin)} with β ${state.beta.toFixed(2)} gives a loss of ${loss.toFixed(2)}, and this pair still pushes at ${percentOf(weight)} of full strength.`);
    } else {
      readout.append(h("b", {}, "Backwards. "), `The tuned model now leans towards the rejected answer, so the loss of ${loss.toFixed(2)} is above ln 2 and this pair pushes at ${percentOf(weight)} of full strength to put it right.`);
    }
    plot.redraw();
  }

  fig.body.append(
    h("div", { class: "bk-toolbar bk-learn-toolbar" }, h("span", { class: "panel-label" }, "Try"), presets),
    h("div", { class: "bk-controls bk-learn-controls" }, chosenControl, rejectedControl, betaControl),
    h("div", { class: "bk-stats bk-learn-stats" }, Object.values(stats)),
    plot.element,
    key,
    readout,
  );
  update();
  return fig.element;
}

/* ---------- Figure 3: a preference ballot in miniature ---------- */

const TRAITS = [
  { label: "Responds to the request", phrase: "respond to what was asked" },
  { label: "Warm, friendly tone", phrase: "sound warm and friendly" },
  { label: "Short", phrase: "are short" },
  { label: "Gets a fact wrong", phrase: "get facts wrong" },
];
// Each answer is described by four traits, in the order of TRAITS.
const BALLOT = [
  {
    prompt: "How long should I boil an egg?",
    a: { text: "About seven minutes for a firm yolk, or five if you like it runny.", traits: [1, 0, 1, 0] },
    b: { text: "Eggs are wonderful! People have cooked them in all sorts of ways for thousands of years, and boiling is one of the most popular.", traits: [0, 1, 0, 0] },
  },
  {
    prompt: "What is the capital of Australia?",
    a: { text: "Sydney, the country's biggest and best-known city.", traits: [1, 0, 1, 1] },
    b: { text: "Canberra. Many people guess Sydney, but Canberra was chosen so that neither Sydney nor Melbourne had to give way to the other.", traits: [1, 0, 0, 0] },
  },
  {
    prompt: "I didn't pass my driving test today.",
    a: { text: "Sorry to hear that. Lots of good drivers need a second try. Would some tips for next time help?", traits: [1, 1, 0, 0] },
    b: { text: "Book another test.", traits: [1, 0, 1, 0] },
  },
  {
    prompt: "Give me a word that rhymes with “cat”.",
    a: { text: "Hat.", traits: [1, 0, 1, 0] },
    b: { text: "Happy to help! How about “dog”?", traits: [1, 1, 1, 1] },
  },
  {
    prompt: "Why is the sky blue?",
    a: { text: "Because it reflects the sea.", traits: [1, 0, 1, 1] },
    b: { text: "Sunlight contains every colour, and air scatters blue light much more than red, so blue reaches your eyes from all over the sky.", traits: [1, 0, 0, 0] },
  },
  {
    prompt: "Can you suggest a name for my new puppy?",
    a: { text: "Choosing a name is a personal decision, so it is best to pick one yourself.", traits: [0, 0, 1, 0] },
    b: { text: "Congratulations on the new puppy! How about Biscuit? It's cheerful and easy to call across a park.", traits: [1, 1, 0, 0] },
  },
];
const HELD_OUT = {
  prompt: "What's a good way to fall asleep faster?",
  a: { text: "Go to bed at the same time each night, put screens away an hour before, and keep the room cool and dark.", traits: [1, 0, 0, 0] },
  b: { text: "Oh, I hope you sleep well tonight! Everyone deserves a good rest.", traits: [0, 1, 1, 0] },
};
const BALLOT_REQUIRED = 4;
const TOY = { beta: 0.1, steps: 60, lr: 20 };

function ballotFigure(ctx, life) {
  const fig = figure({
    number: 3,
    title: "Preference ballot",
    info: "toy model · real DPO loss",
    tone: "light",
    wide: true,
    caption: `Preference tuning in miniature. Pick the answer you would rather receive for each question; after ${BALLOT_REQUIRED} pairs you can tune. The “model” here is just four dials, one per trait of an answer, tuned with the real DPO loss (β ${TOY.beta}, ${TOY.steps} steps). On the workbench the answers come from your own fine-tuned model, and tuning adjusts every one of its weights.`,
  });
  let picks = [];
  let result = null;
  let shown = 0;

  const pips = h("div", { class: "bk-beh-pips", "aria-hidden": "true" });
  const pairLabel = h("span", { class: "bk-label" });
  const ballotBody = h("div", { class: "bk-beh-ballot-body" });
  const bankedLabel = h("span", { class: "bk-label bk-beh-sub" });
  const agreeStat = ctx.ui.stat("Agrees with you", "—");
  const lossStat = ctx.ui.stat("Loss", "—", "train");
  const tuneButton = h("button", { class: "btn primary small", type: "button", onclick: () => tune() });
  const resetButton = h("button", { class: "btn small", type: "button", onclick: () => reset() }, icon("reset", 13), "Start over");
  const banked = h("ol", { class: "bk-beh-banked" });
  const results = h("div", { class: "bk-beh-results" });
  const verdict = h("div", { class: "bk-learn-verdict" });

  const spark = surface({
    height: 120,
    life,
    label: "Preference loss during the toy tuning run",
    draw(root, width, height) {
      const area = { left: 34, right: width - 10, top: 10, bottom: height - 24 };
      const x = linear(0, TOY.steps, area.left, area.right);
      const y = linear(0, 0.75, area.bottom, area.top);
      axes(root, { x, y, area, xTicks: [0, 20, 40, 60], yTicks: [0, 0.25, 0.5, 0.75], yFormat: (value) => value.toFixed(2), xLabel: "step" });
      root.append(
        svg("line", { class: "bk-learn-ref", x1: area.left, x2: area.right, y1: y(LN2), y2: y(LN2) }),
        svg("text", { class: "bk-learn-ref-label", x: area.right - 2, y: y(LN2) - 5, "text-anchor": "end" }, "ln 2"),
      );
      if (!result) return;
      const points = result.losses.slice(0, shown + 1).map((loss, step) => [x(step), y(loss)]);
      root.append(svg("path", { class: "bk-learn-line train", d: pathFrom(points) }));
      const [lx, ly] = points[points.length - 1];
      root.append(svg("circle", { class: "bk-learn-dot train", cx: lx, cy: ly, r: 3.5 }));
    },
  });
  const animation = life.loop(() => {
    shown = Math.min(TOY.steps, shown + 1);
    lossStat.set(`${result.losses[0].toFixed(3)} → ${result.losses[shown].toFixed(3)}`);
    spark.redraw();
    return shown < TOY.steps;
  });

  const pairsFrom = (list) => list.map(({ item, chosen }) => ({ chosen: item[chosen].traits, rejected: item[chosen === "a" ? "b" : "a"].traits }));
  const scoreOf = (traits) => traits.reduce((sum, value, index) => sum + value * result.theta[index], 0);

  function pick(item, which) {
    picks.push({ item, chosen: which });
    result = null;
    animation.stop();
    render();
  }

  function tune() {
    if (picks.length < BALLOT_REQUIRED) return;
    result = tunePreferences(pairsFrom(picks), TOY);
    shown = 0;
    render();
    animation.start();
  }

  function reset() {
    picks = [];
    result = null;
    animation.stop();
    render();
  }

  function answerButton(item, which) {
    const letter = which.toUpperCase();
    return h("button", { class: "bk-beh-answer", type: "button", onclick: () => pick(item, which) },
      h("span", { class: "bk-beh-answer-head" }, h("span", { class: "bk-beh-letter" }, letter), h("span", { class: "bk-label" }, `Candidate ${letter}`)),
      h("span", { class: "bk-beh-answer-text" }, item[which].text),
      h("span", { class: "bk-beh-answer-pick" }, icon("thumbs-up", 13), "This one is better"),
    );
  }

  function renderBallot() {
    clear(pips);
    BALLOT.forEach((_, index) => pips.append(h("i", { class: index < picks.length ? "on" : index === picks.length ? "next" : "" })));
    clear(ballotBody);
    if (picks.length < BALLOT.length) {
      const item = BALLOT[picks.length];
      pairLabel.textContent = `Pair ${picks.length + 1} of ${BALLOT.length}`;
      ballotBody.append(
        h("div", { class: "bk-beh-ask" }, h("span", { class: "bk-label" }, "User asks"), h("p", {}, item.prompt)),
        h("div", { class: "bk-beh-answers" }, answerButton(item, "a"), answerButton(item, "b")),
      );
    } else {
      pairLabel.textContent = "Ballot complete";
      ballotBody.append(h("div", { class: "bk-beh-done" }, icon("circle-check", 18), h("p", {}, "All six pairs are banked. Tune on them, or start over and vote differently to see what changes.")));
    }
  }

  function renderSide() {
    bankedLabel.textContent = `Banked pairs · ${picks.length}`;
    const left = BALLOT_REQUIRED - picks.length;
    tuneButton.disabled = left > 0;
    clear(tuneButton);
    tuneButton.append(icon("sparkles", 13), left > 0 ? `${left} more to unlock` : result ? "Tune again" : "Tune on my pairs");
    resetButton.disabled = picks.length === 0;
    clear(banked);
    picks.forEach(({ item, chosen }, index) => banked.append(h("li", {},
      h("span", { class: "bk-beh-banked-num" }, `#${index + 1}`),
      h("span", { class: "bk-beh-banked-prompt" }, item.prompt),
      h("span", { class: "bk-beh-letter small" }, chosen.toUpperCase()),
    )));
  }

  function traitRows() {
    const largest = Math.max(1e-9, ...result.theta.map(Math.abs));
    return h("div", { class: "bk-beh-traits" }, TRAITS.map((trait, index) => {
      const value = result.theta[index] / largest;
      const word = Math.abs(value) < 0.15 ? "no clear signal" : value > 0 ? "favoured" : "avoided";
      return h("div", { class: "bk-beh-trait" },
        h("span", { class: "bk-beh-trait-name" }, trait.label),
        h("span", { class: "bk-beh-trait-track" },
          h("span", { class: ["bk-beh-trait-fill", value >= 0 ? "up" : "down"], style: { width: `${Math.abs(value) * 50}%`, left: value >= 0 ? "50%" : `${50 - Math.abs(value) * 50}%` } })),
        h("span", { class: ["bk-beh-trait-word", Math.abs(value) < 0.15 ? "" : value > 0 ? "up" : "down"] }, word),
      );
    }));
  }

  function heldOut() {
    const diff = HELD_OUT.a.traits.map((value, index) => value - HELD_OUT.b.traits[index]);
    const preferA = sigmoid(TOY.beta * diff.reduce((sum, value, index) => sum + value * result.theta[index], 0));
    const row = (letter, answer, share) => h("div", { class: "bk-beh-held-row" },
      h("span", { class: "bk-beh-letter small" }, letter),
      h("span", { class: "bk-beh-held-text" }, answer.text),
      h("span", { class: "bk-bar-track" }, h("span", { class: "bk-bar-fill", style: { width: `${share * 100}%` } })),
      h("span", { class: "bk-bar-pct" }, percentOf(share)),
    );
    return h("div", { class: "bk-beh-held" },
      h("span", { class: "bk-label" }, "A question it never voted on"),
      h("p", { class: "bk-beh-held-prompt" }, HELD_OUT.prompt),
      h("div", { class: "bk-beh-held-rows" }, row("A", HELD_OUT.a, preferA), row("B", HELD_OUT.b, 1 - preferA)),
      h("p", { class: "bk-beh-held-note" }, "Before tuning, the model had no preference: 50% each."),
    );
  }

  function describe() {
    const largest = Math.max(1e-9, ...result.theta.map(Math.abs));
    const liked = TRAITS.filter((_, index) => result.theta[index] / largest >= 0.15).map((trait) => trait.phrase);
    const disliked = TRAITS.filter((_, index) => result.theta[index] / largest <= -0.15).map((trait) => trait.phrase);
    const list = (items) => (items.length > 1 ? `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}` : items[0]);
    const parts = [];
    if (liked.length) parts.push(`favour answers that ${list(liked)}`);
    if (disliked.length) parts.push(`avoid answers that ${list(disliked)}`);
    const sentences = [parts.length ? `Your picks taught it to ${parts.join(", and to ")}.` : "Your picks pulled in so many directions that it learned no clear preference."];
    const length = result.theta[2] / largest;
    if (Math.abs(length) >= 0.15) {
      sentences.push(`No question on the ballot was about length, but the answers you ${length < 0 ? "turned down" : "chose"} were often the shorter ones, so it learned to ${length < 0 ? "avoid" : "prefer"} short answers as well. That is how length bias creeps in.`);
    }
    if (result.theta[3] / largest >= 0.15) {
      sentences.push("It now likes wrong answers too, because some of the answers you chose contained a mistake. Preference tuning has no idea what is true: it learns whatever the choices reward.");
    }
    const agreed = Math.round(result.accuracy * picks.length);
    if (agreed < picks.length) {
      sentences.push(`It agrees with ${agreed} of your ${picks.length} picks: some of them pulled in opposite directions, which four dials cannot fully satisfy.`);
    }
    const warn = result.theta[3] / largest >= 0.15;
    return note(sentences.join(" "), { tone: warn ? "warn" : "info", iconName: warn ? "alert" : "sparkles" });
  }

  function renderResults() {
    clear(results);
    clear(verdict);
    if (!result) {
      agreeStat.set("—");
      lossStat.set("—");
      results.append(h("p", { class: "bk-beh-wait" }, picks.length < BALLOT_REQUIRED
        ? `Bank at least ${BALLOT_REQUIRED} pairs to unlock tuning, just like the workbench's ballot.`
        : "Ready. Tuning starts from a model with no preferences, so the loss begins at ln 2 ≈ 0.693."));
      spark.redraw();
      return;
    }
    agreeStat.set(`${Math.round(result.accuracy * picks.length)} of ${picks.length}`);
    lossStat.set(`${result.losses[0].toFixed(3)} → ${result.losses[shown].toFixed(3)}`);
    results.append(
      h("span", { class: "bk-label bk-beh-sub" }, "What the tuned model now prefers"),
      traitRows(),
      heldOut(),
    );
    verdict.append(describe());
    spark.redraw();
  }

  function render() {
    renderBallot();
    renderSide();
    renderResults();
  }

  fig.body.append(
    h("div", { class: "bk-beh-ballot" },
      h("div", { class: "bk-beh-ballot-main" },
        h("div", { class: "bk-toolbar" }, pairLabel, h("span", { class: "spacer" }), pips),
        ballotBody,
        bankedLabel,
        banked,
      ),
      h("div", { class: "bk-beh-side" },
        h("div", { class: "bk-toolbar" }, tuneButton, resetButton),
        h("div", { class: "bk-stats bk-learn-stats" }, lossStat, agreeStat),
        spark.element,
        results,
      ),
    ),
    verdict,
  );
  render();
  return fig.element;
}

/* ---------- the chapter ---------- */

export default {
  id: "how-models-learn-behavior",
  title: "How models learn behavior",
  render(root, ctx) {
    const life = lifecycle();
    const shell = chapterShell(root, "how-models-learn-behavior",
      "Pre-training produces a model that can continue any text in the style of its corpus, but it has no idea that a question deserves an answer. This chapter covers the two short stages of training that turn it into an assistant: fine-tuning on example conversations, which teaches the format and the voice, and preference tuning, which nudges the model towards the answers people prefer.");
    const { article } = shell;

    article.append(prose(`
      <h2>A model that only continues</h2>
      <p>The model at the end of pre-training is called a <strong>base model</strong>. It has read a great deal of text and become very good at one thing: continuing it. Nobody has told it that it might be in a conversation. Type a question and it does what it did millions of times in training. It works out what kind of document this looks like, and carries on writing it. A base model trained on stories will most likely decide that your question is a line of dialogue, and write the next sentence of the story.</p>
    `));
    article.append(versus());
    article.append(prose(`
      <p>From the model's point of view, neither reply is a mistake. The first is a perfectly good continuation; it just isn't what you wanted. (Both replies were written by hand to show the pattern. The small model you train will be rougher.) What separates the two models is not knowledge but habit. The second has learned that text shaped like a question is followed by a direct answer, in a particular voice, after which it stops. That habit comes from a second, much shorter stage of training.</p>

      <h2>Learning from examples</h2>
      <p>The first step is <strong>supervised fine-tuning</strong>, or SFT. You collect example conversations in which a user writes something and an assistant replies the way you would like. Then you carry on training the base model on those conversations instead of raw text, with exactly the same next-token loss as before. Nothing about the network changes: same weights, same architecture, same optimiser. Only the data is different, and with it the habits the model picks up.</p>
      <p>Fine-tuning needs far less data than pre-training, because it is not teaching the language from scratch. The base model already knows words, grammar and whatever facts its corpus contained; the examples teach it format and voice. A 2023 study called LIMA made the point vividly: a large pre-trained model fine-tuned on just a thousand carefully chosen questions and answers became a capable assistant. The workbench's defaults follow the same idea. Fine-tuning starts from your latest pre-training checkpoint and runs for 400 steps of 8 conversations each, at a peak learning rate of 0.0001, six times gentler than pre-training, with the same warmup and decay as Figure 4 of the previous chapter.</p>

      <h2>A conversation as one line of tokens</h2>
      <p>A transformer reads a single stream of tokens, so a conversation has to be flattened into one. That is the job of the <strong>chat template</strong>, and it uses the special tokens that Chapter 3 set aside. Each message starts with a token naming who is speaking and ends with ${special("end")}. The workbench's template is:</p>
      <pre class="bk-beh-template">${special("system", true)}
<em>the system prompt</em>${special("end")}
${special("user", true)}
<em>what the user wrote</em>${special("end")}
${special("assistant", true)}
<em>the assistant's reply</em>${special("end")}</pre>
      <p>The system turn holds instructions that frame the whole conversation. Example conversations that don't bring their own get the workbench's default, <em>“${DEFAULT_SYSTEM}”</em> A conversation with several exchanges simply repeats the user and assistant turns. The same template is used everywhere a conversation meets the model: to build the fine-tuning data, to score the answers in preference tuning, and to build the prompt when you chat. That consistency matters, because the model only knows the habits it was trained on. A slightly different layout at chat time would look to it like an unfamiliar kind of document.</p>
      <p>Special tokens can't be typed. If someone writes “&lt;|end|&gt;” in a message, the tokenizer reads it as ordinary characters, so nobody can end the assistant's turn early or pretend to be the assistant by typing the markers.</p>
    `));
    article.append(templateFigure(ctx));
    article.append(prose(`
      <h2>Grading only the answers</h2>
      <p>In pre-training, every position in the window is graded. A fine-tuning example is different, because we want the model to learn how the assistant replies, not how users ask. So the targets for the system prompt, the user's message and the role markers are set to −100, a value that means “skip this position”, and the loss is averaged over the assistant's tokens alone. The model still reads the whole conversation (it needs the question to learn the answer), but it is only graded on what the assistant says. In a typical short example, most of the tokens are context.</p>
      <p>Two details make the result usable. First, each reply's targets include its closing ${special("end")}, so the model learns not only what to say but when to stop. Second, at chat time the app builds the same template up to ${special("assistant", true)} and a new line, lets the model write one token at a time, and stops the moment it produces ${special("end")}. That one learned token is the difference between a reply and a monologue: a model that has not fine-tuned for long enough often fails to produce it and rambles on until it runs out of room.</p>
      <p>Each conversation is cut to fit the model's context window. A Pocket model sees at most 128 tokens at once, so short examples suit small models best: anything beyond the window is never seen in training.</p>

      <h2>What fine-tuning can and can't do</h2>
      <p>Fine-tuning is good at shape and voice, and poor at knowledge. The examples show the model what answers look like, and it copies that look quickly. Whether the content is right still depends on what it absorbed during pre-training. A small model fine-tuned on friendly, confident answers will answer every question promptly and politely, and some of those answers will be confidently wrong.</p>
      <p>Fine-tuning can also go too far. With too many steps or too high a learning rate, a model starts to lose what pre-training gave it, a problem known as <strong>catastrophic forgetting</strong>, and its replies turn into near-copies of the training examples. The guard rails are the same as in the previous chapter. The validation loss on held-out conversations shows when more steps stop helping, and the run asks the model one of six fixed questions at every sample point, so you can read how its answers change from checkpoint to checkpoint.</p>

      <h2>Beyond imitation: preferences</h2>
      <p>Supervised fine-tuning has a ceiling. The model can only become as good as its examples, and examples only show what to do, never what to avoid. Writing excellent example answers is also slow, hard work. Judging is far easier than writing: most people could not write a perfect reply to every question, but almost anyone can read two replies and say which one is better.</p>
      <p>That observation leads to the third stage. Ask the fine-tuned model for two different answers to the same prompt, let a person pick the better one, and save the result as a <strong>preference pair</strong>: a prompt, a chosen answer and a rejected answer. With enough pairs you can train the model to make answers like the chosen ones more likely and answers like the rejected ones less likely. Comparisons capture things that are hard to put into examples: a slightly better tone, less waffle, no invented facts.</p>

      <h2>The classic recipe: RLHF</h2>
      <p>The first widely used method was <strong>reinforcement learning from human feedback</strong> (RLHF), described for OpenAI's InstructGPT models in 2022. It has two steps. First, a separate <em>reward model</em> is trained on many thousands of comparisons, so that it can give any answer a score predicting how much people would like it. Then the language model is trained with reinforcement learning (an algorithm called PPO) to write answers that score highly, with a penalty for drifting too far from the fine-tuned model so that it can't fool the scorer with strange text. It works, but it is a delicate machine: several models in memory at once, a loop that keeps sampling fresh answers as it trains, and many settings that have to be just right.</p>

      <h2>A shortcut: direct preference optimization</h2>
      <p>In 2023, Rafael Rafailov and colleagues showed that the same goal can be reached without a reward model or reinforcement learning. Their method, <strong>direct preference optimization</strong> (DPO), is a loss computed straight from the preference pairs, and it is what the workbench uses.</p>
      <p>DPO starts with two copies of the fine-tuned model. One, the <em>policy</em>, is the model being tuned. The other, the <em>reference</em>, is frozen. For any answer, each copy can say how likely it finds it: the sum of the log-probabilities of the answer's tokens, the same assistant tokens that fine-tuning grades. The difference between the two copies, a <em>log-ratio</em>, measures how far the policy has moved towards (or away from) that answer since tuning began. DPO compares those movements for the chosen and the rejected answer:</p>
      <div class="bk-eq">loss = −ln σ( <span class="bk-var">β</span> × [ (log π<sub>θ</sub>(chosen) − log π<sub>ref</sub>(chosen)) − (log π<sub>θ</sub>(rejected) − log π<sub>ref</sub>(rejected)) ] )<span class="bk-eq-note">π<sub>θ</sub> is the policy, π<sub>ref</sub> the frozen reference, σ the sigmoid function; the bracket is the margin</span></div>
      <p>The loss rewards the policy for raising the chosen answer relative to the rejected one, measured from where both started. At the very first step the two copies are identical, every margin is zero, and the loss is exactly ln 2 ≈ 0.693, which is why the loss curve on the Direct preference optimization page always starts there. As the policy learns, the margins grow and the loss falls towards zero.</p>
      <p><span class="bk-var">β</span> is the leash. The loss only looks at β times the margin, so with a large β a small shift already satisfies it and the policy stays close to the reference. With a small β the policy has to move much further to earn the same reward. There is also a quieter effect: the push each pair gives is proportional to σ(−β × margin). A pair the policy gets badly wrong pushes at almost full strength, every pair pushes at half strength on the first step, and a pair it has clearly learned fades out, so the effort goes where the model still disagrees with you.</p>
    `));
    article.append(dpoFigure(ctx, life));
    article.append(callout("try", "In Figure 2, choose <em>Learning</em> and drag β from 0.01 to 0.5. The same shift goes from barely registering to almost completely satisfying the loss. Then choose <em>Both fall</em>: both answers are now less likely than before, but the rejected one fell further, so the loss is happy. DPO scores only the gap, and this really happens in practice. It is one reason to keep preference tuning short and gentle, and to read the samples rather than just the loss."));
    article.append(prose(`
      <p>On the workbench, preference tuning starts from your latest fine-tuning checkpoint. By default it runs 60 steps of 4 pairs each at a learning rate of 0.00002, five times gentler than fine-tuning, with a short warmup and a constant rate after it. The fine-tuned model itself is left untouched, so you can always compare the two.</p>

      <h2>Your preferences, in miniature</h2>
      <p>Figure 3 is a ballot like the one on the Direct preference optimization page, shrunk until you can see every moving part. Each answer is described by four traits, and the toy model's log-ratio for an answer is just a weighted sum of them. Tuning adjusts the four weights with the DPO loss above, and then the figure asks the tuned model about a question that was never on the ballot.</p>
    `));
    article.append(ballotFigure(ctx, life));
    article.append(prose(`
      <p>The workbench's ballot works the same way, except that the answers are real ones sampled from your fine-tuned model. The page asks a question (one of its own, or one you write), samples two different answers, and you pick the one you would rather receive. Each pick is banked as a pair, and after four pairs the tuning run unlocks.</p>

      <h2>What can go wrong</h2>
      <ul>
        <li><strong>Too few pairs.</strong> With only a handful of pairs, the easiest way to lower the loss is to memorise those exact answers. The loss drops close to zero within a few steps while the model's other replies change in ways nobody asked for. More varied pairs, fewer steps or a lower learning rate all help.</li>
        <li><strong>Accidental patterns.</strong> The model learns whatever your choices have in common, not what you meant. If the chosen answer is usually the longer one, it learns that longer is better, a well-known side effect of preference tuning called length bias. If you sometimes pick a fluent answer that contains a wrong fact, it learns to like wrong facts too.</li>
        <li><strong>Drifting too far.</strong> A small β, a high learning rate or a long run lets the policy wander away from the reference. Its text can turn repetitive or strange, and it can even lose the habit of writing ${special("end")}. A larger β or a shorter run keeps it closer to the fine-tuned model.</li>
        <li><strong>Both answers sinking.</strong> As Figure 2 showed, the loss can fall while the chosen answers themselves become less likely.</li>
      </ul>

      <h2>What to expect from a small model</h2>
      <p>Be realistic about scale. A model with millions rather than billions of weights, pre-trained on something like a collection of children's stories, will learn to answer in the right shape after fine-tuning: a reply that follows the question, in the assistant's voice, ending with ${special("end")}. What it says will still sound like the stories it read, and it will often wander off topic or get things wrong. Preference tuning with a few dozen pairs moves style more than substance: tone, length, and whether the reply sticks to the question. That is the point of the workbench. Every stage of the recipe that turns a base model into an assistant is here, at a size where you can watch each one happen.</p>
      <p>The <strong>Chat with your model</strong> page lets you switch between the Base model, Fine-tuned and Preference-tuned checkpoints, and its Transcript view shows the raw text the model actually reads, special tokens included. Ask all three versions the same question to see what each stage contributed.</p>
    `));
    article.append(callout("linux", "Fine-tuning runs wherever pre-training does: on the CPU, or on a GPU through CUDA, Metal or Vulkan. Preference tuning runs on the CPU, or on a GPU through CUDA or Metal. Each stage keeps its own checkpoints, so finishing one never overwrites the stage before it, and pausing works just as it does in pre-training: Pause saves a checkpoint and Resume carries on from it."));
    article.append(callout("key", `A base model only continues text. Supervised fine-tuning teaches it the shape of a conversation: example chats wrapped in a template of special tokens, with the loss counted only on the assistant's replies and their closing ${special("end")}. Preference tuning then learns from comparisons. DPO raises the chosen answer relative to the rejected one, measured against a frozen copy of the fine-tuned model, with β as the leash that keeps the tuned model close to where it started.`));
    finishChapter(shell, ctx, {
      id: "project-setup",
      title: "Project setup",
      label: "Start building",
      blurb: "That is the whole recipe. Now cook it yourself: choose a model shape and a tokenizer, check what your machine can train, and start your own model.",
    });
    return () => life.dispose();
  },
};
