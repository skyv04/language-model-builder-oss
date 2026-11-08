// Chapter 5: Anatomy of a Transformer — the block diagram, attention patterns and parameter counting.
import { h, svg, clear, clamp } from "../dom.js";
import { icon } from "../icons.js";
import {
  chapterShell, finishChapter, prose, callout, figure, control, lifecycle, note,
  seededRandom, softmax, percent, describeApiError, fmt,
} from "../book/kit.js";
import { surface } from "../book/plot.js";

/* ---------- model shapes and counting (the same formulas the engine uses) ---------- */

const SHAPES = [
  { id: "pocket", label: "Pocket", vocab: 4096, context: 128, width: 128, blocks: 4, heads: 4 },
  { id: "starter", label: "Starter", vocab: 10000, context: 256, width: 256, blocks: 6, heads: 8 },
  { id: "large", label: "Large", vocab: 10000, context: 512, width: 512, blocks: 8, heads: 8 },
  { id: "gpt2", label: "GPT-2 small", vocab: 50257, context: 1024, width: 768, blocks: 12, heads: 12 },
];
const SHAPE = Object.fromEntries(SHAPES.map((shape) => [shape.id, shape]));
const SHAPE_KEYS = ["vocab", "context", "width", "blocks", "heads"];

function breakdown({ vocab, context, width: d, blocks }) {
  return {
    tok: vocab * d,
    pos: context * d,
    attn: blocks * (4 * d * d + 4 * d),
    mlp: blocks * (8 * d * d + 5 * d),
    norms: blocks * 4 * d + 2 * d,
  };
}

function countParameters(shape) {
  const parts = breakdown(shape);
  return parts.tok + parts.pos + parts.attn + parts.mlp + parts.norms;
}

// The workbench's rough estimate of memory for one training step: weights, gradients and two
// AdamW moments (16 bytes per parameter), plus activations, attention maps and logits.
function trainingMemory(shape, batch = 16) {
  const { vocab, context, width, blocks, heads } = shape;
  const tokens = batch * context;
  return countParameters(shape) * 16
    + tokens * blocks * width * 4 * 20
    + batch * heads * context * context * blocks * 4 * 3
    + tokens * vocab * 4 * 3;
}

function rgba(hex, alpha) {
  const value = parseInt(hex.slice(1), 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

/* ---------- Figure 1: the block diagram ---------- */

const COLORS = {
  ids: "#b7c1d6",
  blue: "#2f8cff",
  teal: "#2fb5c5",
  stream: "#c9d4ee",
  block: "#6f86b8",
  norm: "#9aa8c7",
  attention: "#ff9f43",
  mlp: "#a77bff",
  green: "#4fd18b",
};

const PARTS = [
  {
    id: "tokens",
    name: "Token ids",
    color: COLORS.ids,
    text: (s) => `The input is a run of token ids straight from the tokenizer, at most ${fmt.int(s.context)} of them for this shape. Each id gets its own lane, and all the lanes travel up through the model side by side.`,
    shape: (s) => `up to ${fmt.int(s.context)} ids`,
    params: () => 0,
    zero: "None: an id is only a row number",
  },
  {
    id: "tok",
    name: "Token embedding",
    color: COLORS.blue,
    text: (s) => `A table with one row of ${s.width} numbers for each of the ${fmt.int(s.vocab)} tokens in the vocabulary. Every id fetches its row, exactly like the lookup in the previous chapter.`,
    shape: (s) => `${fmt.int(s.vocab)} rows × ${s.width}`,
    params: (s) => s.vocab * s.width,
  },
  {
    id: "pos",
    name: "Position embedding",
    color: COLORS.teal,
    text: (s) => `A second table with one learned row for each of the ${fmt.int(s.context)} positions. Row 0 is added to the first lane, row 1 to the second and so on. It is the model's only clue about word order.`,
    shape: (s) => `${fmt.int(s.context)} rows × ${s.width}`,
    params: (s) => s.context * s.width,
  },
  {
    id: "stream",
    name: "Residual stream",
    color: COLORS.stream,
    text: (s) => `The running vector of each lane, ${s.width} numbers wide. Every layer reads from it and adds its result back at a ⊕. Nothing in the stream is ever overwritten, only added to.`,
    shape: (s) => `${s.width} numbers per lane`,
    params: () => 0,
    zero: "None: it carries activations, not weights",
  },
  {
    id: "block",
    name: "Transformer block",
    color: COLORS.block,
    text: (s) => `One attention layer and one MLP, each with a layer norm in front and a shortcut around it. This shape stacks ${s.blocks} blocks. They are built alike, but each has its own weights and learns its own job.`,
    shape: (s) => `${s.blocks} blocks`,
    params: (s) => s.blocks * (12 * s.width * s.width + 13 * s.width),
  },
  {
    id: "ln",
    name: "Layer norm",
    color: COLORS.norm,
    text: (s) => `Rescales a lane's vector to an average of 0 and a typical size of 1, then applies a learned gain and bias to each of its ${s.width} numbers. Every block has two, one in front of each branch.`,
    shape: (s) => `2 per block, gain + bias`,
    params: (s) => s.blocks * 4 * s.width,
  },
  {
    id: "attn",
    name: "Attention",
    color: COLORS.attention,
    text: (s) => `The only place where lanes exchange information. Each lane forms a query, a key and a value; queries are matched against the keys of earlier lanes, and the matching values are blended and added to the stream. ${s.heads} heads do this side by side, each in a slice of ${s.width / s.heads} numbers.`,
    shape: (s) => `${s.heads} heads × ${s.width / s.heads}`,
    params: (s) => s.blocks * (4 * s.width * s.width + 4 * s.width),
  },
  {
    id: "mlp",
    name: "MLP",
    color: COLORS.mlp,
    text: (s) => `A small two-layer network that works on each lane separately: widen from ${s.width} to ${4 * s.width} numbers, pass them through a smooth gate called GELU, and project back down to ${s.width}. It holds about two thirds of every block's weights.`,
    shape: (s) => `${s.width} → ${4 * s.width} → ${s.width}`,
    params: (s) => s.blocks * (8 * s.width * s.width + 5 * s.width),
  },
  {
    id: "lnf",
    name: "Final layer norm",
    color: COLORS.norm,
    text: () => "One last normalisation of every lane after the final block, so that the output layer always receives vectors of a familiar size.",
    shape: (s) => `gain + bias, ${s.width} each`,
    params: (s) => 2 * s.width,
  },
  {
    id: "out",
    name: "Output scores",
    color: COLORS.blue,
    text: (s) => `Each lane's final vector is compared, by dot product, with every row of the token embedding table: ${fmt.int(s.vocab)} scores, one per token. Reusing the input table like this is called weight tying.`,
    shape: (s) => `${fmt.int(s.vocab)} scores per lane`,
    params: () => 0,
    zero: "None of its own: it reuses the token embedding",
  },
  {
    id: "softmax",
    name: "Softmax",
    color: COLORS.green,
    text: () => "Turns the scores into probabilities that add up to 1. Training measures how much probability went to the true next token; generation samples one token from them.",
    shape: (s) => `${fmt.int(s.vocab)} probabilities`,
    params: () => 0,
    zero: "None: it is a fixed formula",
  },
];
const PART = Object.fromEntries(PARTS.map((part) => [part.id, part]));
const RELATED = { tok: ["out", "tie"], out: ["tok", "tie"], block: ["ln", "attn", "mlp"], ln: ["lnf"], lnf: ["ln"] };

let diagramCount = 0;

function anatomyFigure(ctx) {
  const markerId = `bk-anat-arrow-${(diagramCount += 1)}`;
  let shape = SHAPE.pocket;
  let selected = "attn";
  const fig = figure({
    number: 1,
    title: "Inside the model",
    wide: true,
    caption: "A decoder-only transformer as this workbench builds it. Click any part, or step through with Previous and Next, to see what it does and how many parameters it holds. The block in the middle is repeated: 4 times in Pocket, 6 in Starter and 8 in Large, and every copy has its own weights.",
  });
  const root = svg("svg", { class: "bk-anat-svg", viewBox: "0 0 360 704", role: "group", "aria-label": "Diagram of a decoder-only transformer" });
  const groups = new Map();
  const dynamic = {};

  const part = (id) => {
    const info = PART[id];
    const g = svg("g", { class: "bk-anat-part", "data-part": id, tabindex: "0", role: "button", "aria-label": info.name });
    g.addEventListener("click", () => select(id));
    g.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select(id);
      }
    });
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push(g);
    root.append(g);
    return g;
  };
  const wire = (d, arrow = true) => {
    root.append(svg("path", { class: "bk-anat-wire", d, "marker-end": arrow ? `url(#${markerId})` : null }));
  };
  const box = (g, { x, y, w, h: height, color, title, key = null, align = "middle" }) => {
    g.append(svg("rect", {
      class: "bk-anat-box",
      x, y, width: w, height, rx: 9,
      style: { "--fill": rgba(color, 0.15), "--fill-on": rgba(color, 0.4), "--edge": rgba(color, 0.8) },
    }));
    const tx = align === "start" ? x + 12 : x + w / 2;
    const middle = y + height / 2;
    g.append(svg("text", { class: "bk-anat-title", x: tx, y: key ? middle - 4 : middle + 4, "text-anchor": align }, title));
    if (key) {
      dynamic[key] = svg("text", { class: "bk-anat-sub", x: tx, y: middle + 12, "text-anchor": align }, "");
      g.append(dynamic[key]);
    }
  };
  const adder = (g, cx, cy) => {
    g.append(
      svg("circle", { class: "bk-anat-add", cx, cy, r: 9 }),
      svg("path", { class: "bk-anat-plus", d: `M${cx - 4.5},${cy}H${cx + 4.5}M${cx},${cy - 4.5}V${cy + 4.5}` }),
    );
  };

  root.append(svg("defs", {},
    svg("marker", { id: markerId, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5.5, markerHeight: 5.5, orient: "auto-start-reverse" },
      svg("path", { class: "bk-anat-arrowhead", d: "M0,1 L9,5 L0,9 z" })),
  ));

  // Weight tying: the output layer reads the token embedding table.
  const tie = svg("g", { class: "bk-anat-tie" },
    svg("path", { d: "M24,87 H12 V571 H24" }),
    svg("text", { x: 12, y: 330, transform: "rotate(-90 12 330)", "text-anchor": "middle", "dominant-baseline": "central" }, "same table"),
  );
  root.append(tie);

  const block = part("block");
  block.append(
    svg("rect", { class: "bk-anat-shadow", x: 46, y: 172, width: 296, height: 312, rx: 14 }),
    svg("rect", { class: "bk-anat-shadow", x: 40, y: 166, width: 296, height: 312, rx: 14 }),
    svg("rect", { class: "bk-anat-frame", x: 34, y: 160, width: 296, height: 312, rx: 14 }),
  );
  dynamic.block = svg("text", { class: "bk-anat-block-label", x: 318, y: 179, "text-anchor": "end" }, "");
  block.append(dynamic.block);

  const stream = part("stream");
  stream.append(
    svg("line", { class: "bk-anat-hit", x1: 100, y1: 149, x2: 100, y2: 500 }),
    svg("line", { class: "bk-anat-band", x1: 100, y1: 149, x2: 100, y2: 500 }),
    svg("line", { class: "bk-anat-core", x1: 100, y1: 149, x2: 100, y2: 500 }),
    svg("circle", { class: "bk-anat-fork", cx: 100, cy: 190, r: 3.2 }),
    svg("circle", { class: "bk-anat-fork", cx: 100, cy: 330, r: 3.2 }),
    svg("text", { class: "bk-anat-stream-label", x: 86, y: 392, transform: "rotate(-90 86 392)", "text-anchor": "middle" }, "residual stream"),
  );

  wire("M60,30 V38 M140,30 V38 M220,30 V38 M300,30 V38 M60,38 H300", false);
  wire("M100,38 V60");
  wire("M260,38 V60");
  wire("M100,112 V129");
  wire("M260,112 V140 H111");
  wire("M100,190 H230 V198");
  wire("M230,224 V234");
  wire("M230,290 V310 H111");
  wire("M100,330 H230 V338");
  wire("M230,364 V374");
  wire("M230,430 V450 H111");
  wire("M100,526 V544");
  wire("M100,596 V614");

  const tokens = part("tokens");
  ["The", "·kettle", "·began", "·to"].forEach((label, index) => {
    const x = 24 + index * 80;
    tokens.append(
      svg("rect", { class: "bk-anat-box", x, y: 8, width: 72, height: 22, rx: 6, style: { "--fill": rgba(COLORS.ids, 0.12), "--fill-on": rgba(COLORS.ids, 0.26), "--edge": rgba(COLORS.ids, 0.55) } }),
      svg("text", { class: "bk-anat-chip-text", x: x + 36, y: 23, "text-anchor": "middle" }, label),
    );
  });

  box(part("tok"), { x: 24, y: 62, w: 152, h: 50, color: COLORS.blue, title: "Token embedding", key: "tok" });
  box(part("pos"), { x: 184, y: 62, w: 152, h: 50, color: COLORS.teal, title: "Position embedding", key: "pos" });

  const adders = part("stream");
  adder(adders, 100, 140);
  adder(adders, 100, 310);
  adder(adders, 100, 450);

  box(part("ln"), { x: 165, y: 200, w: 130, h: 24, color: COLORS.norm, title: "Layer norm" });
  box(part("attn"), { x: 140, y: 236, w: 180, h: 54, color: COLORS.attention, title: "Attention", key: "attn" });
  box(part("ln"), { x: 165, y: 340, w: 130, h: 24, color: COLORS.norm, title: "Layer norm" });
  box(part("mlp"), { x: 140, y: 376, w: 180, h: 54, color: COLORS.mlp, title: "MLP", key: "mlp" });
  box(part("lnf"), { x: 24, y: 500, w: 152, h: 26, color: COLORS.norm, title: "Final layer norm" });

  const out = part("out");
  box(out, { x: 24, y: 546, w: 312, h: 50, color: COLORS.blue, title: "Output scores", key: "out", align: "start" });
  const random = seededRandom(11);
  for (let index = 0; index < 30; index += 1) {
    const tall = index === 8;
    const height = tall ? 26 : 3 + Math.round(random() * 12);
    out.append(svg("rect", { class: tall ? "bk-anat-score top" : "bk-anat-score", x: 204 + index * 4, y: 588 - height, width: 3, height, rx: 1 }));
  }

  const soft = part("softmax");
  soft.append(
    svg("rect", { class: "bk-anat-box", x: 24, y: 616, width: 312, height: 78, rx: 9, style: { "--fill": rgba(COLORS.green, 0.12), "--fill-on": rgba(COLORS.green, 0.24), "--edge": rgba(COLORS.green, 0.75) } }),
    svg("text", { class: "bk-anat-title", x: 36, y: 637 }, "Softmax"),
    svg("text", { class: "bk-anat-sub", x: 324, y: 637, "text-anchor": "end" }, "probabilities for the next token"),
  );
  [["·boil", 0.52], ["·whistle", 0.23], ["·sing", 0.06]].forEach(([label, probability], index) => {
    const y = 657 + index * 15;
    soft.append(
      svg("text", { class: "bk-anat-sub bk-anat-word", x: 36, y }, label),
      svg("rect", { class: "bk-anat-track", x: 104, y: y - 7, width: 180, height: 7, rx: 3.5 }),
      svg("rect", { class: "bk-anat-fill", x: 104, y: y - 7, width: Math.max(3, probability * 180), height: 7, rx: 3.5 }),
      svg("text", { class: "bk-anat-sub", x: 324, y, "text-anchor": "end" }, `${Math.round(probability * 100)}%`),
    );
  });

  const side = {
    dot: h("span", { class: "bk-dot" }),
    count: h("span", {}),
    name: h("h3", { class: "bk-anat-name" }),
    text: h("p", { class: "bk-anat-text" }),
    shape: h("dd", {}),
    params: h("dd", {}),
    fill: h("span", { class: "bk-anat-share-fill" }),
  };
  const sidePanel = h("div", { class: "bk-anat-side" },
    h("div", { class: "bk-anat-kicker" }, side.dot, side.count),
    side.name,
    side.text,
    h("dl", { class: "bk-anat-facts" },
      h("div", {}, h("dt", {}, "Shape"), side.shape),
      h("div", {}, h("dt", {}, "Parameters"), side.params),
    ),
    h("div", { class: "bk-anat-share", title: "Share of all parameters" }, side.fill),
    h("div", { class: "bk-anat-nav" },
      h("button", { class: "btn small", onclick: () => step(-1) }, icon("chevron-left", 12), "Previous"),
      h("button", { class: "btn small", onclick: () => step(1) }, "Next", icon("chevron-right", 12)),
    ),
  );

  const presets = ctx.ui.segmented(
    SHAPES.slice(0, 3).map((item) => ({ value: item.id, label: item.label })),
    shape.id,
    (value) => {
      shape = SHAPE[value];
      applyShape();
    },
  );

  fig.body.append(
    h("div", { class: "bk-toolbar bk-anat-toolbar" },
      h("span", { class: "panel-label" }, "Preset"),
      presets,
      h("span", { class: "spacer" }),
      h("span", { class: "bk-anat-hint" }, icon("pointer", 13), "Click any part of the diagram"),
    ),
    h("div", { class: "bk-anat" }, h("div", { class: "bk-anat-diagram" }, root), sidePanel),
  );

  function select(id) {
    selected = id;
    update();
  }

  function step(delta) {
    const index = PARTS.findIndex((item) => item.id === selected);
    select(PARTS[(index + delta + PARTS.length) % PARTS.length].id);
  }

  function update() {
    const info = PART[selected];
    const total = countParameters(shape);
    const count = info.params(shape);
    side.dot.style.background = info.color;
    side.count.textContent = `Part ${PARTS.indexOf(info) + 1} of ${PARTS.length}`;
    side.name.textContent = info.name;
    side.text.textContent = info.text(shape);
    side.shape.textContent = info.shape(shape);
    side.params.textContent = count ? `${fmt.int(count)} · ${percent(count / total)} of ${shape.label}` : info.zero;
    side.fill.style.width = `${((count / total) * 100).toFixed(2)}%`;
    side.fill.style.background = info.color;
    const related = RELATED[selected] || [];
    for (const [id, list] of groups) {
      for (const g of list) {
        g.classList.toggle("sel", id === selected);
        g.classList.toggle("rel", related.includes(id));
        g.setAttribute("aria-pressed", String(id === selected));
      }
    }
    tie.classList.toggle("rel", related.includes("tie"));
  }

  function applyShape() {
    const { vocab, context, width, blocks, heads } = shape;
    dynamic.tok.textContent = `${fmt.int(vocab)} × ${width} table`;
    dynamic.pos.textContent = `${fmt.int(context)} × ${width} table`;
    dynamic.attn.textContent = `${heads} heads × ${width / heads} numbers`;
    dynamic.mlp.textContent = `${width} → ${4 * width} → ${width}`;
    dynamic.block.textContent = `Block, repeated × ${blocks}`;
    dynamic.out.textContent = `${fmt.int(vocab)} scores per lane`;
    fig.setInfo(`${fmt.int(countParameters(shape))} parameters`);
    update();
  }

  applyShape();
  return { element: fig.element };
}

/* ---------- Figure 2: attention inspector ---------- */

const ATTENTION_TEXT = "The small dog chased the ball, and then the dog slept by the door.";
const MAX_TOKENS = 32;
const DEMO_PATTERNS = [
  { value: "previous", label: "Previous token" },
  { value: "first", label: "First token" },
  { value: "same", label: "Same word" },
  { value: "nearby", label: "Nearby words" },
];
const PATTERN_TEXT = {
  previous: "This head mostly looks one position back. Heads like this let every position pick up the token just before it, a building block for words that span several tokens and for short phrases.",
  first: "This head mostly rests on the first position. Its weights must add up to 1, so when there is nothing useful to find, the attention has to land somewhere, and the start of the text is always visible.",
  self: "This head mostly looks at its own position, so it adds little from the rest of the context.",
  same: "This head looks back at earlier copies of the same word. Heads like this help a model keep names and topics consistent through a passage.",
  nearby: "This head looks mostly at the last few positions, with less and less attention further back. Local heads like this can help with short-range grammar, such as agreeing with a nearby word.",
  spread: "This head spreads its attention over many earlier positions, gathering a blurry summary of everything so far.",
  noRepeat: "No word repeats in this text, so a same-word head has nothing to find and rests on the first position instead. Try typing a word twice.",
};

const normalise = (text) => text.trim().toLowerCase();

function demoTokens(text, label) {
  return (text.match(/\s*[A-Za-z\u00C0-\u024F'’]+|\s*\d+|\s*[^\sA-Za-z\d]/g) || []).map((piece) => ({ text: piece, label: label(piece) }));
}

// Default position to look from: the last word (a repeated one for the same-word pattern), not trailing punctuation.
function defaultQuery(tokens, pattern) {
  const keys = tokens.map((token) => normalise(token.text));
  const wordy = (i) => /[\p{L}\p{N}]/u.test(tokens[i].text);
  if (pattern === "same") {
    for (let i = tokens.length - 1; i > 0; i -= 1) if (wordy(i) && keys.indexOf(keys[i]) < i) return i;
  }
  for (let i = tokens.length - 1; i >= 0; i -= 1) if (wordy(i)) return i;
  return Math.max(0, tokens.length - 1);
}

function hasRepeats(tokens) {
  const seen = new Set();
  for (const token of tokens) {
    const key = normalise(token.text);
    if (seen.has(key)) return true;
    seen.add(key);
  }
  return false;
}

// Hand-made attention patterns of the kinds that turn up in trained models.
function demoMatrix(tokens, pattern) {
  const n = tokens.length;
  const keys = tokens.map((token) => normalise(token.text));
  const random = seededRandom(97 + n * 31 + pattern.length * 7);
  return tokens.map((_, i) => {
    const repeats = [];
    for (let j = 0; j < i; j += 1) if (keys[j] === keys[i]) repeats.push(j);
    const scores = [];
    for (let j = 0; j <= i; j += 1) {
      let score = random() * 0.6;
      if (pattern === "previous") score += j === i - 1 ? 4.2 : j === i ? 1.2 : 0;
      else if (pattern === "first") score += j === 0 ? 3.6 : 0;
      else if (pattern === "same") score += repeats.length ? (repeats.includes(j) ? 5 : 0) : j === 0 ? 2.2 : j === i ? 1.5 : 0;
      else score -= 0.45 * (i - j);
      scores.push(score);
    }
    const row = new Array(n).fill(0);
    softmax(scores).forEach((weight, j) => {
      row[j] = weight;
    });
    return row;
  });
}

function averageHeads(heads) {
  const n = heads[0].length;
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => heads.reduce((sum, matrix) => sum + matrix[i][j], 0) / heads.length));
}

function classifyPattern(matrix, tokens) {
  const n = matrix.length;
  const keys = tokens.map((token) => normalise(token.text));
  let previous = 0;
  let first = 0;
  let self = 0;
  let local = 0;
  let same = 0;
  let sameRows = 0;
  let rows = 0;
  for (let i = 2; i < n; i += 1) {
    const row = matrix[i];
    previous += row[i - 1];
    first += row[0];
    self += row[i];
    local += row[i] + row[i - 1] + row[i - 2];
    rows += 1;
    let total = 0;
    let found = false;
    for (let j = 0; j < i; j += 1) {
      if (keys[j] === keys[i]) {
        total += row[j];
        found = true;
      }
    }
    if (found) {
      same += total;
      sameRows += 1;
    }
  }
  const scores = { previous: previous / rows, first: first / rows, self: self / rows, same: sameRows ? same / sameRows : 0 };
  const [best, value] = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  if (value >= 0.45) return best;
  if (local / rows >= 0.6) return "nearby";
  return "spread";
}

function parseLive(result, label) {
  if (!result || !Array.isArray(result.tokens) || !Array.isArray(result.layers) || !result.layers.length) return null;
  const total = result.tokens.length;
  const n = Math.min(total, MAX_TOKENS);
  if (!n) return null;
  const layers = [];
  for (const layer of result.layers) {
    if (!Array.isArray(layer) || !layer.length) return null;
    const heads = [];
    for (const head of layer) {
      if (!Array.isArray(head) || head.length < n) return null;
      heads.push(head.slice(0, n).map((row) => Array.from({ length: n }, (_, j) => {
        const value = Array.isArray(row) ? Number(row[j]) : NaN;
        return Number.isFinite(value) ? value : 0;
      })));
    }
    layers.push(heads);
  }
  const tokens = result.tokens.slice(0, n).map((token) => {
    const text = typeof token === "string" ? token : String(token && token.text != null ? token.text : "");
    return { text, label: label(text) };
  });
  return { tokens, layers, total };
}

const shorten = (text, limit) => (text.length > limit ? `${text.slice(0, limit - 1)}…` : text);

function attentionFigure(ctx, life) {
  const fig = figure({
    number: 2,
    title: "Where each position looks",
    wide: true,
    caption: "Each row is one position in the text, and each cell shows how much of that position's attention goes to an earlier one: the brighter, the more. The dim triangle is the causal mask. Hover a cell for its value, and click a row or a word to look from there. With a trained model in this project, the figure shows that model's real attention for every layer and head; otherwise it shows hand-made patterns of the kinds found in trained models.",
  });
  const state = {
    text: ATTENTION_TEXT,
    mode: "loading",
    busy: false,
    live: null,
    layer: 0,
    head: 0,
    pattern: "previous",
    query: null,
    error: null,
  };
  let view = { tokens: [], matrix: [], total: 0 };
  let geometry = null;
  let height = 160;
  let hover = null;
  let request = 0;
  const debounce = life.timer();

  const input = h("input", { class: "text-input", type: "text", maxlength: "200", value: state.text, spellcheck: "false", "aria-label": "Text to inspect" });
  const badge = h("span", { class: "bk-attn-badge" });
  const askButton = h("button", { class: "btn small", title: "Ask this project's trained model for its attention", onclick: () => fetchLive() }, icon("refresh", 12), "Ask my model");
  const resetButton = h("button", {
    class: "btn small ghost",
    title: "Restore the example",
    onclick: () => {
      input.value = ATTENTION_TEXT;
      onText();
    },
  }, icon("reset", 13));
  const picks = h("div", { class: "bk-attn-picks" });
  const message = h("div", { class: "bk-attn-message" });
  const queryWord = h("span", { class: "bk-attn-query-word" });
  const readout = h("p", { class: "bk-attn-readout" });
  const reading = h("div", { class: "bk-attn-read" });
  const summary = h("p", { class: "bk-attn-desc" });
  const heat = surface({ height, className: "bk-attn-heat", label: "Attention heatmap", life, draw });
  const placeholder = h("div", { class: "bk-loading" }, h("span", { class: "spinner" }), "Asking your model for its attention…");
  const main = h("div", { class: "bk-attn-main", hidden: true },
    h("div", { class: "bk-attn-heat-wrap" }, heat.element),
    h("div", { class: "bk-attn-side" },
      h("span", { class: "panel-label" }, "Looking from"),
      queryWord,
      readout,
      h("span", { class: "panel-label bk-attn-read-label" }, "Its attention over the text"),
      reading,
      summary,
    ),
  );

  fig.body.append(
    h("div", { class: "bk-toolbar" }, badge, h("span", { class: "spacer" }), askButton),
    h("div", { class: "bk-input-row bk-attn-input" }, h("span", { class: "panel-label" }, "Text"), input, resetButton),
    picks,
    message,
    placeholder,
    main,
  );

  input.addEventListener("input", onText);
  heat.svg.addEventListener("pointermove", (event) => setHover(cellAt(event)));
  heat.svg.addEventListener("pointerleave", () => setHover(null));
  heat.svg.addEventListener("click", (event) => {
    const cell = cellAt(event, true);
    if (!cell) return;
    state.query = cell.i;
    hover = null;
    refresh();
  });

  function onText() {
    state.text = input.value;
    state.query = null;
    hover = null;
    if (state.mode === "live") {
      state.busy = Boolean(state.text.trim());
      renderStatus();
      if (state.text.trim()) debounce.set(fetchLive, 450);
      else refresh();
    } else if (state.mode === "demo") {
      refresh();
    }
  }

  async function fetchLive() {
    debounce.clear();
    const text = state.text;
    const ticket = (request += 1);
    if (!text.trim()) {
      state.busy = false;
      if (state.mode === "loading") state.mode = "demo";
      refresh();
      return;
    }
    state.busy = true;
    renderStatus();
    try {
      const result = await ctx.api.post("/api/projects/current/attention", { text });
      if (life.disposed || ticket !== request) return;
      const parsed = parseLive(result, ctx.tokenLabel);
      if (!parsed) throw Object.assign(new Error("The engine's attention data had an unexpected shape."), { status: 500 });
      state.live = parsed;
      state.mode = "live";
      state.error = null;
      state.layer = Math.min(state.layer, parsed.layers.length - 1);
      if (state.head !== "mean") state.head = Math.min(state.head, parsed.layers[state.layer].length - 1);
    } catch (error) {
      if (life.disposed || ticket !== request) return;
      state.live = null;
      state.mode = "demo";
      state.error = describeApiError(error);
    }
    state.busy = false;
    refresh();
  }

  function computeView() {
    if (!state.text.trim()) {
      view = { tokens: [], matrix: [], total: 0 };
    } else if (state.mode === "live" && state.live) {
      const { tokens, layers, total } = state.live;
      const heads = layers[state.layer];
      view = { tokens, matrix: state.head === "mean" ? averageHeads(heads) : heads[state.head], total };
    } else {
      const all = demoTokens(state.text, ctx.tokenLabel);
      const tokens = all.slice(0, MAX_TOKENS);
      view = { tokens, matrix: demoMatrix(tokens, state.pattern), total: all.length };
    }
    if (state.query === null || state.query >= view.tokens.length) state.query = defaultQuery(view.tokens, state.mode === "demo" ? state.pattern : null);
  }

  function refresh() {
    computeView();
    placeholder.hidden = state.mode !== "loading";
    main.hidden = state.mode === "loading";
    renderStatus();
    renderPicks();
    renderMessage();
    heat.redraw();
    renderReading();
    renderReadout();
    renderSummary();
  }

  function renderStatus() {
    clear(badge);
    badge.className = "bk-attn-badge";
    if (state.busy) {
      badge.classList.add("busy");
      badge.append(h("span", { class: "spinner" }), state.mode === "live" ? "Updating…" : "Asking your model…");
    } else if (state.mode === "live") {
      const layers = state.live.layers.length;
      const heads = state.live.layers[0].length;
      badge.classList.add("live");
      badge.append(h("span", { class: "bk-dot" }), `Your model · ${layers} layer${layers === 1 ? "" : "s"} × ${heads} head${heads === 1 ? "" : "s"}`);
    } else if (state.mode === "demo") {
      badge.classList.add("demo");
      badge.append(h("span", { class: "bk-dot" }), "Demo · hand-made patterns, not a trained model");
    }
    askButton.hidden = state.mode === "live";
    askButton.disabled = state.busy;
    if (state.mode === "live") fig.setInfo(`Layer ${state.layer + 1} · ${state.head === "mean" ? "all heads" : `head ${state.head + 1}`}`);
    else fig.setInfo(state.mode === "demo" ? "Demo patterns" : "");
  }

  const chip = (label, active, onClick) => h("button", {
    class: ["chip", "bk-attn-chip", active ? "active" : ""],
    "aria-pressed": String(active),
    onclick: onClick,
  }, label);

  function renderPicks() {
    clear(picks);
    if (state.mode === "live" && state.live) {
      const { layers } = state.live;
      picks.append(
        h("div", { class: "bk-attn-pick" }, h("span", { class: "panel-label" }, "Layer"),
          layers.map((_, index) => chip(String(index + 1), state.layer === index, () => {
            state.layer = index;
            if (state.head !== "mean" && state.head >= layers[index].length) state.head = 0;
            refresh();
          }))),
        h("div", { class: "bk-attn-pick" }, h("span", { class: "panel-label" }, "Head"),
          layers[state.layer].map((_, index) => chip(String(index + 1), state.head === index, () => {
            state.head = index;
            refresh();
          })),
          chip("All, averaged", state.head === "mean", () => {
            state.head = "mean";
            refresh();
          })),
      );
    } else if (state.mode === "demo") {
      picks.append(h("div", { class: "bk-attn-pick" }, h("span", { class: "panel-label" }, "Pattern"),
        DEMO_PATTERNS.map((pattern) => chip(pattern.label, state.pattern === pattern.value, () => {
          state.pattern = pattern.value;
          state.query = null;
          refresh();
        }))));
    }
  }

  function renderMessage() {
    clear(message);
    if (state.mode !== "demo" || !state.error) return;
    const { kind, text } = state.error;
    if (kind === "not-ready") {
      message.append(note("This project has no trained model yet, so you are looking at hand-made patterns. Once Pre-training has saved a checkpoint, this figure shows your model's real attention, layer by layer and head by head.", {
        action: h("button", { class: "btn small", onclick: () => ctx.navigate("pretraining") }, "Open Pre-training", icon("chevron-right", 12)),
      }));
    } else if (kind === "unavailable") {
      message.append(note("The training engine isn't available in this window, so you are looking at hand-made patterns rather than a trained model's."));
    } else {
      message.append(note(`${text} Showing hand-made patterns instead.`, { tone: "warn" }));
    }
  }

  function renderReading() {
    clear(reading);
    const { tokens, matrix, total } = view;
    const i = state.query;
    queryWord.textContent = tokens[i] ? tokens[i].label : "—";
    tokens.forEach((token, j) => {
      const weight = j <= i ? matrix[i][j] : 0;
      const strength = 0.08 + 0.9 * Math.pow(clamp(weight, 0, 1), 0.7);
      reading.append(h("button", {
        class: ["bk-attn-tok", j === i ? "query" : "", j > i ? "future" : ""],
        style: j <= i ? { background: `rgba(255, 159, 67, ${strength.toFixed(3)})`, color: strength > 0.55 ? "#1d1d1f" : null } : null,
        title: j > i ? "A later position: hidden from this one" : `${percent(weight)} of the attention from “${tokens[i].label}”`,
        onclick: () => {
          state.query = j;
          hover = null;
          refresh();
        },
      }, token.label));
    });
    if (total > tokens.length) reading.append(h("span", { class: "bk-attn-more" }, `+${total - tokens.length} more tokens not shown`));
  }

  function renderReadout() {
    const { tokens, matrix } = view;
    if (!tokens.length) {
      readout.textContent = "Type some text above.";
      return;
    }
    if (hover) {
      const from = tokens[hover.i].label;
      const to = tokens[hover.j].label;
      readout.textContent = hover.j > hover.i
        ? `“${from}” cannot look ahead to “${to}”: later positions are masked.`
        : `“${from}” → “${to}”: ${percent(matrix[hover.i][hover.j])} of its attention.`;
      return;
    }
    const i = state.query;
    if (i === 0) {
      readout.textContent = "The first position can only look at itself, so all of its attention stays put.";
      return;
    }
    const top = matrix[i].slice(0, i + 1).map((weight, j) => [weight, j]).sort((a, b) => b[0] - a[0]).slice(0, 3);
    readout.textContent = `Strongest links: ${top.map(([weight, j]) => `“${tokens[j].label}” ${percent(weight, 0)}`).join(", ")}.`;
  }

  function renderSummary() {
    const { tokens, matrix } = view;
    if (tokens.length < 4) {
      summary.textContent = "Add a few more words to see what kind of pattern this head has.";
      return;
    }
    if (state.mode === "demo" && state.pattern === "same" && !hasRepeats(tokens)) {
      summary.textContent = PATTERN_TEXT.noRepeat;
      return;
    }
    summary.textContent = PATTERN_TEXT[classifyPattern(matrix, tokens)];
  }

  function cellAt(event, allowLabels = false) {
    if (!geometry) return null;
    const rect = heat.svg.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const { ox, left, top, cell, n } = geometry;
    const i = Math.floor((y - top) / cell);
    if (i < 0 || i >= n) return null;
    if (allowLabels && x >= ox && x < ox + left) return { i, j: null };
    const j = Math.floor((x - ox - left) / cell);
    if (j < 0 || j >= n) return null;
    return { i, j };
  }

  function setHover(cell) {
    hover = cell && cell.j !== null ? cell : null;
    const marker = heat.svg.querySelector(".bk-attn-hover");
    if (marker && geometry) {
      if (hover) {
        const { ox, left, top, cell: size } = geometry;
        marker.setAttribute("x", ox + left + hover.j * size - 1);
        marker.setAttribute("y", top + hover.i * size - 1);
        marker.style.display = "";
      } else {
        marker.style.display = "none";
      }
    }
    renderReadout();
  }

  function ensureHeight(needed) {
    if (needed === height) return false;
    height = needed;
    queueMicrotask(() => {
      if (!life.disposed) heat.setHeight(needed);
    });
    return true;
  }

  function draw(root, width) {
    const { tokens, matrix } = view;
    const n = tokens.length;
    geometry = null;
    if (!n) {
      if (ensureHeight(90)) return;
      root.append(svg("text", { class: "bk-attn-empty", x: width / 2, y: 48, "text-anchor": "middle" }, "Type some text to see where each position looks."));
      return;
    }
    const chars = Math.min(14, Math.max(...tokens.map((token) => token.label.length)));
    const narrow = width < 480;
    const left = clamp(chars * 6.4 + 14, 30, narrow ? 72 : 104);
    const top = clamp(chars * 6.4 * 0.8 + 16, 30, narrow ? 72 : 100);
    const pad = clamp(chars * 3.6, 10, narrow ? 36 : 52);
    const cell = clamp(Math.floor((width - left - pad) / n), 8, 28);
    if (ensureHeight(Math.ceil(top + cell * n + 10))) return;
    const ox = Math.max(0, Math.floor((width - (left + cell * n + pad)) / 2));
    geometry = { ox, left, top, cell, n };
    const font = clamp(cell * 0.62, 8.5, 11);
    const every = cell < 11 ? 2 : 1;
    const x0 = ox + left;

    let masked = "";
    const cells = svg("g", { class: "bk-attn-cells" });
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const x = x0 + j * cell;
        const y = top + i * cell;
        if (j > i) {
          masked += `M${x},${y}h${cell - 1}v${cell - 1}h${1 - cell}z`;
        } else {
          const weight = clamp(matrix[i][j], 0, 1);
          cells.append(svg("rect", { x, y, width: cell - 1, height: cell - 1, "fill-opacity": (0.06 + 0.94 * Math.pow(weight, 0.7)).toFixed(3) }));
        }
      }
    }
    root.append(svg("path", { class: "bk-attn-masked", d: masked }), cells);

    const labels = svg("g", { class: "bk-attn-labels", "font-size": font.toFixed(1) });
    tokens.forEach((token, index) => {
      const chosen = index === state.query;
      if (index % every && !chosen) return;
      const text = shorten(token.label, 14);
      labels.append(svg("text", { class: chosen ? "sel" : null, x: x0 - 6, y: top + index * cell + cell / 2, "text-anchor": "end", "dominant-baseline": "central" }, text));
      const cx = x0 + index * cell + cell / 2 + font * 0.3;
      labels.append(svg("text", { class: chosen ? "sel" : null, x: cx, y: top - 6, transform: `rotate(-55 ${cx} ${top - 6})`, "text-anchor": "start" }, text));
    });
    root.append(labels);

    const i = state.query;
    root.append(
      svg("rect", { class: "bk-attn-row", x: x0 - 1.5, y: top + i * cell - 1.5, width: (i + 1) * cell + 2, height: cell + 2, rx: 3 }),
      svg("rect", { class: "bk-attn-hover", width: cell + 1, height: cell + 1, rx: 2, style: { display: "none" } }),
    );
    if (hover) setHover(hover);
  }

  return {
    element: fig.element,
    start() {
      renderStatus();
      fetchLive();
    },
  };
}

/* ---------- Figure 3: parameter calculator ---------- */

const VOCABS = [256, 512, 1024, 2048, 4096, 8192, 10000, 16000, 32000, 50257, 65536];
const VOCAB_NOTES = {
  256: "one token per byte value",
  4096: "the workbench's 4K tokenizer",
  10000: "the workbench's 10K tokenizer",
  16000: "the workbench's 16K tokenizer",
  50257: "GPT-2's tokenizer",
  65536: "the largest the workbench allows",
};
const SEGMENTS = [
  { key: "tok", label: "Token embedding", color: "#2f8cff" },
  { key: "pos", label: "Position embedding", color: "#2fb5c5" },
  { key: "attn", label: "Attention", color: "#ff9f43" },
  { key: "mlp", label: "MLP", color: "#a77bff" },
  { key: "norms", label: "Layer norms", color: "#8e8e93" },
];

function validHeads(width) {
  const list = [];
  for (let heads = 1; heads <= 16; heads += 1) if (width % heads === 0 && (width / heads) % 2 === 0) list.push(heads);
  return list;
}

function memoryVerdict(bytes) {
  if (bytes < 700e6) return "Light enough to train on a phone or any laptop.";
  if (bytes < 3e9) return "Fine on most laptops. On a phone, close other apps or lower the batch size.";
  if (bytes < 12e9) return "Wants a desktop with plenty of memory, or a GPU.";
  return "Far beyond a phone or a laptop: models this size are trained on large data-centre GPUs.";
}

function paramsFigure(ctx) {
  const state = { ...SHAPE.pocket };
  const fig = figure({
    number: 3,
    title: "Parameter calculator",
    tone: "light",
    caption: "Every parameter, counted with the formula above. The total matches what Project setup reports for the same shape. The weights file stores 4 bytes per parameter, and training memory is the workbench's own rough estimate at a batch size of 16.",
  });
  const { slider, stat } = ctx.ui;
  let headOptions = validHeads(state.width);

  const vocabHint = h("div", { class: "bk-control-hint" });
  const headsHint = h("div", { class: "bk-control-hint" });
  const inputs = {
    vocab: slider({ min: 0, max: VOCABS.length - 1, step: 1, value: VOCABS.indexOf(state.vocab), label: "Vocabulary size", onInput: (index) => { state.vocab = VOCABS[index]; update(); } }),
    context: slider({ min: 16, max: 1024, step: 16, value: state.context, label: "Context length", onInput: (value) => { state.context = value; update(); } }),
    width: slider({ min: 32, max: 1024, step: 32, value: state.width, label: "Width", onInput: (value) => { state.width = value; syncHeads(); update(); } }),
    blocks: slider({ min: 1, max: 16, step: 1, value: state.blocks, label: "Blocks", onInput: (value) => { state.blocks = value; update(); } }),
    heads: slider({ min: 0, max: headOptions.length - 1, step: 1, value: headOptions.indexOf(state.heads), label: "Attention heads", onInput: (index) => { state.heads = headOptions[index]; update(); } }),
  };
  const controls = {
    vocab: control("Vocabulary (V)", inputs.vocab),
    context: control("Context (C)", inputs.context),
    width: control("Width (d)", inputs.width),
    blocks: control("Blocks (L)", inputs.blocks),
    heads: control("Heads", inputs.heads),
  };
  controls.vocab.append(vocabHint);
  controls.heads.append(headsHint);

  const presetButtons = SHAPES.map((shape) => h("button", { class: "chip bk-pc-preset", onclick: () => applyPreset(shape) }, shape.label));
  const total = h("span", { class: "bk-pc-big" });
  const compact = h("span", { class: "bk-pc-compact" });
  const formula = h("div", { class: "bk-pc-formula" });
  const bar = h("div", { class: "bk-pc-bar", role: "img", "aria-label": "Where the parameters are" });
  const legend = h("ul", { class: "bk-pc-legend" });
  const stats = {
    file: stat("Weights file", "—"),
    memory: stat("Training memory · batch 16", "—"),
    block: stat("One block", "—"),
  };
  const verdict = h("span", { class: "grow" });

  fig.body.append(
    h("div", { class: "bk-pc-presets" }, h("span", { class: "bk-label" }, "Start from"), presetButtons),
    h("div", { class: "bk-controls" }, Object.values(controls)),
    h("div", { class: "bk-pc-result" },
      h("div", { class: "bk-pc-total" }, total, h("span", { class: "bk-pc-unit" }, "parameters"), compact),
      formula,
      bar,
      legend,
      h("div", { class: "bk-stats" }, Object.values(stats)),
      h("div", { class: "bk-note bk-pc-verdict" }, icon("cpu", 14), verdict),
    ),
  );

  function syncHeads() {
    headOptions = validHeads(state.width);
    let best = headOptions[0];
    for (const option of headOptions) if (Math.abs(option - state.heads) < Math.abs(best - state.heads)) best = option;
    state.heads = best;
    inputs.heads.max = String(headOptions.length - 1);
    inputs.heads.setValue(headOptions.indexOf(best));
  }

  function applyPreset(shape) {
    for (const key of SHAPE_KEYS) state[key] = shape[key];
    inputs.vocab.setValue(VOCABS.indexOf(shape.vocab));
    inputs.context.setValue(shape.context);
    inputs.width.setValue(shape.width);
    inputs.blocks.setValue(shape.blocks);
    syncHeads();
    update();
  }

  function update() {
    const parts = breakdown(state);
    const sum = countParameters(state);
    const d = state.width;
    total.textContent = fmt.int(sum);
    compact.textContent = `≈ ${fmt.compact(sum)}`;
    formula.textContent = `= ${fmt.int(state.vocab)}·${d} + ${fmt.int(state.context)}·${d} + ${state.blocks}·(12·${d}² + 13·${d}) + 2·${d}`;
    clear(bar);
    clear(legend);
    for (const segment of SEGMENTS) {
      const value = parts[segment.key];
      bar.append(h("span", { class: "bk-pc-seg", title: `${segment.label}: ${fmt.int(value)}`, style: { width: `${((value / sum) * 100).toFixed(3)}%`, background: segment.color } }));
      legend.append(h("li", {},
        h("span", { class: "bk-dot", style: { background: segment.color } }),
        h("span", { class: "bk-pc-name" }, segment.label),
        h("span", { class: "bk-pc-count" }, fmt.int(value)),
        h("span", { class: "bk-pc-share" }, percent(value / sum)),
      ));
    }
    const memory = trainingMemory(state);
    stats.file.set(fmt.bytes(sum * 4));
    stats.memory.set(fmt.bytes(memory));
    stats.block.set(fmt.int(12 * d * d + 13 * d));
    verdict.textContent = memoryVerdict(memory);
    controls.vocab.setValue(fmt.int(state.vocab));
    vocabHint.textContent = VOCAB_NOTES[state.vocab] || " ";
    controls.context.setValue(`${fmt.int(state.context)} tokens`);
    controls.width.setValue(String(d));
    controls.blocks.setValue(String(state.blocks));
    controls.heads.setValue(String(state.heads));
    headsHint.textContent = `${d / state.heads} numbers per head; the total stays the same`;
    SHAPES.forEach((shape, index) => {
      const active = SHAPE_KEYS.every((key) => shape[key] === state[key]);
      presetButtons[index].classList.toggle("active", active);
      presetButtons[index].setAttribute("aria-pressed", String(active));
    });
  }

  update();
  return { element: fig.element };
}

/* ---------- chapter ---------- */

export default {
  id: "anatomy-of-a-transformer",
  title: "Anatomy of a Transformer",
  render(root, ctx) {
    const life = lifecycle();
    const shell = chapterShell(root, "anatomy-of-a-transformer",
      "The previous chapter ended with a row of vectors, one for each token. This chapter follows those vectors up through a transformer, the design inside nearly every modern language model, until they come out of the top as a forecast for the next token. Each part is simpler than its reputation, and by the end you will be able to count every number the model learns.");
    const { article } = shell;
    const anatomy = anatomyFigure(ctx);
    const attention = attentionFigure(ctx, life);
    const params = paramsFigure(ctx);

    article.append(prose(`
      <h2>The whole machine on one page</h2>
      <p>The transformer was introduced in 2017 by a group of researchers at Google, in a paper about translating text from one language into another. Their design had two halves: an <em>encoder</em> that reads the sentence to be translated, and a <em>decoder</em> that writes the translation one token at a time. Language models in the GPT family keep only the second half, so they are often called <strong>decoder-only</strong> transformers. The models you build in this workbench are of exactly that kind, and Figure 1 shows one in full.</p>
      <p>Read the diagram from top to bottom. Token ids come in and fetch their rows from the token embedding table, a second table contributes a row for each position, and the sum enters a long vertical line called the residual stream. The stream then passes through a stack of identical-looking <strong>blocks</strong>, each made of an attention layer and an MLP. After the last block, one more normalisation step tidies the vectors, and the output layer turns each of them into a score for every token in the vocabulary. Softmax turns the scores into probabilities.</p>
    `));
    article.append(anatomy.element);
    article.append(prose(`
      <p>One detail is easy to miss. The model does not make a single prediction at the end: every position gets its own lane through the stack and its own forecast at the top. The lane for <span class="bk-chip-inline">The</span> predicts what follows <em>The</em>, the lane for <span class="bk-chip-inline">·kettle</span> predicts what follows <em>The kettle</em>, and so on along the line. When the model generates text, only the forecast from the last lane is used. During training all of them are checked at once, so a single excerpt of 128 tokens provides 128 small lessons.</p>

      <h2>The residual stream</h2>
      <p>The vertical line running through Figure 1 is the most important design decision in the whole machine, and the easiest to overlook. Each lane carries a vector of <em>d</em> numbers, the <strong>residual stream</strong>, and the layers never replace it. They read it, compute something, and add the result back:</p>
      <div class="bk-eq"><span class="bk-var">x</span> ← <span class="bk-var">x</span> + attention(norm(<span class="bk-var">x</span>))<br><span class="bk-var">x</span> ← <span class="bk-var">x</span> + MLP(norm(<span class="bk-var">x</span>))<span class="bk-eq-note">x is one lane's vector; every block applies both lines, in this order</span></div>
      <p>A useful picture is a notebook that travels with each token. Every layer reads the current page and adds a short note of its own; nothing already written is ever erased. An early block might jot down that this token is a noun, or that a quotation has just opened. Later blocks read those notes and build on them, and by the top of the stack the page holds everything the model needs for its forecast.</p>
      <p>Adding instead of replacing also makes deep models trainable. A freshly initialised layer adds almost nothing, so the stream passes through it nearly untouched, and during training the corrections can travel straight down the stream to the earliest layers without fading away. The trick was borrowed from image-recognition networks built around 2015, which used it to train networks far deeper than had been practical before.</p>

      <h2>Attention: positions reading each other</h2>
      <p>An embedding describes a token on its own, but words lean on each other. Take <em>“Priya handed the parcel to Tom because he”</em>. To guess what comes next, the lane for <em>he</em> needs to know who <em>he</em> is, and that information sits in other lanes. Attention is the only place in the model where lanes exchange information.</p>
      <p>Inside an attention layer, every lane turns its vector into three new ones, each by multiplying it with a learned matrix:</p>
      <ul>
        <li>a <strong>query</strong>: what this position is looking for;</li>
        <li>a <strong>key</strong>: what this position can offer to others;</li>
        <li>a <strong>value</strong>: what it hands over when it is chosen.</li>
      </ul>
      <p>Each query is compared with the keys of its own position and every earlier one, using the dot product from the previous chapter; a large result means a good match. The results are divided by the square root of the head size (otherwise they grow with the length of the vectors, and softmax would pile all the weight onto a single position), and softmax turns them into weights that add up to 1. The layer's output for the lane is the weighted mix of the values:</p>
      <div class="bk-eq">weights = softmax(<span class="bk-frac"><span><span class="bk-var">q</span> · <span class="bk-var">k</span></span><span>√head size</span></span>)&emsp; output = Σ weight × <span class="bk-var">v</span><span class="bk-eq-note">one query q against the key k of each visible position; v is that position's value</span></div>
      <p>A lane may never look at later positions. While the model learns to predict the next token, a peek at that token would be cheating, so attention applies a <strong>causal mask</strong>: the lane for <em>he</em> sees “Priya handed the parcel to Tom because he” and nothing beyond. That is why the heatmap in Figure 2 is a triangle.</p>
      <p>Rather than one big attention, each layer runs several smaller <strong>heads</strong> side by side, each working in its own slice of the vector. Pocket splits its 128 numbers into 4 heads of 32, and Starter splits 256 into 8 heads of 32. Each head can specialise, one tracking the previous word while another follows the subject of the sentence. Their outputs are joined back together and mixed by one more matrix, the <em>output projection</em>. Changing the number of heads changes how the width is sliced, not how many weights there are.</p>
    `));
    article.append(attention.element);
    article.append(prose(`
      <p>Researchers who open up trained models keep finding the same few kinds of head. <strong>Previous-token heads</strong> look one step back. Many heads park most of their attention on the first token whenever they have nothing better to do, because their weights must add up to 1 and the start of the text is always visible. And <strong>induction heads</strong> find an earlier occurrence of the current token and look at whatever came right after it. They let a model continue a pattern it has seen only once: having read “Captain Ilse Varga” earlier in its context, it can complete “Captain Ilse V…” the second time round.</p>

      <h2>The MLP: one position at a time</h2>
      <p>Attention moves information between lanes. The <strong>MLP</strong> (short for multilayer perceptron) then works on each lane separately, with the same weights at every position. It widens the vector from <em>d</em> to 4<em>d</em> numbers, passes each of them through <strong>GELU</strong>, a smooth gate that lets positive values through and squashes negative ones to almost zero, and projects the result back down to <em>d</em>.</p>
      <p>One way to picture it: each of the 4<em>d</em> hidden numbers is a detector that fires when the incoming vector points in a particular direction, and when it fires it adds a direction of its own to the stream. Pocket has 512 such detectors in every block; GPT-2 small has 3,072. Studies of trained models suggest that much of what a model “knows”, such as that kettles boil and that Lisbon is in Portugal, is stored in these layers. They are also where most of the parameters live: about two thirds of every block.</p>

      <h2>Layer norm: keeping the numbers tame</h2>
      <p>Both branches of a block begin with <strong>layer normalisation</strong>. It takes the <em>d</em> numbers in a lane, shifts them so that their average is 0 and scales them so that their typical size is 1, then multiplies each by a learned gain and adds a learned bias. Without this step the numbers in the stream could drift towards huge or tiny values as the blocks pile up, and training would wobble or blow up. The models in this workbench normalise at the start of each branch, as GPT-2 does, and once more at the very top, before the output layer.</p>

      <h2>From the last vector to a prediction</h2>
      <p>At the top of the stack, each lane's final vector is compared with every row of the token embedding table, again by a dot product, giving one score per token in the vocabulary. These raw scores are often called <strong>logits</strong>, and softmax turns them into the probabilities you met in Chapter 2. Reusing the input table here is the <strong>weight tying</strong> mentioned in the previous chapter. It saves a whole second table: for Pocket that would be another 524,288 numbers, in a model where the embedding table already makes up 39% of all parameters.</p>
      <p>Tying also has a pleasing geometric meaning. To predict <span class="bk-chip-inline">·boil</span>, the final vector must point the same way as the embedding of <em>·boil</em>, so tokens that are interchangeable as inputs tend to be interchangeable as outputs too.</p>

      <h2>Counting parameters</h2>
      <p>Everything a model learns is stored in its <strong>parameters</strong>: the numbers in its tables, matrices and biases. Counting them takes nothing more than adding up the parts of Figure 1, with <em>V</em> for the vocabulary size, <em>C</em> for the context length, <em>d</em> for the width and <em>L</em> for the number of blocks:</p>
      <ul>
        <li>token embedding: <em>V</em> × <em>d</em>;</li>
        <li>position embedding: <em>C</em> × <em>d</em>;</li>
        <li>attention, in each block: 4<em>d</em>² + 4<em>d</em>, for the query, key, value and output matrices (each <em>d</em> × <em>d</em>) and their biases;</li>
        <li>MLP, in each block: 8<em>d</em>² + 5<em>d</em>, for the widening and narrowing matrices (<em>d</em> × 4<em>d</em> and 4<em>d</em> × <em>d</em>) and their biases;</li>
        <li>layer norms: 4<em>d</em> per block for the two inside it, plus 2<em>d</em> for the final one.</li>
      </ul>
      <div class="bk-eq">parameters = <span class="bk-var">V</span>·<span class="bk-var">d</span> + <span class="bk-var">C</span>·<span class="bk-var">d</span> + <span class="bk-var">L</span>·(12<span class="bk-var">d</span>² + 13<span class="bk-var">d</span>) + 2<span class="bk-var">d</span></div>
      <p>The <em>d</em>² terms dominate. Doubling the width roughly quadruples the cost of every block, while doubling the number of blocks only doubles it. Plug in the shape of GPT-2 small (vocabulary 50,257, context 1,024, width 768, 12 blocks) and you get 124,439,808, the model usually quoted as “124M”. That is about 93 times the size of Pocket, which has 1,334,016.</p>
    `));
    article.append(params.element);
    article.append(callout("try", "Start from Pocket in Figure 3 and double the width to 256: the total jumps from 1.3 to 4.2 million. Go back and double the blocks to 8 instead, and you only reach 2.1 million. Then change the number of heads and watch the total stay exactly where it is."));
    article.append(prose(`
      <h2>From parameters to memory</h2>
      <p>Every parameter is stored as a 4-byte number, so Pocket's 1,334,016 parameters make a weights file of about 5.3 MB. Training needs far more room than that. Next to each weight the optimiser keeps its gradient and two running averages, and every layer's intermediate results must be kept for every token in the batch until the backward pass has used them. <strong>Project setup</strong> adds all of this up for the shape you choose and warns you when the estimate approaches half of your computer's memory. At the default batch size of 16, the workbench estimates about 256 MB for Pocket, 1.7 GB for Starter and 7.4 GB for Large.</p>
    `));
    article.append(callout("linux", "Pocket is comfortable almost anywhere, phones included; Starter suits a laptop; Large wants a desktop with plenty of memory or a GPU, which the workbench can use through CUDA, Metal or Vulkan. If an estimate is too high, a smaller batch size on the Pre-training page shrinks the part that grows with the batch. The workbench also checks that the number of heads divides the width evenly and that each head gets an even number of values."));
    article.append(prose(`
      <p>You now know every part of the machine and what each one costs. What none of these parts contains yet is knowledge: straight after creation, every table and matrix holds small random numbers, and the forecast at the top is close to a uniform guess. Everything the model will ever know has to come from the text it reads during training, which is where the next chapter begins.</p>
    `));
    article.append(callout("key", "A transformer is a stack of blocks wrapped around a residual stream. Attention lets each position read from earlier positions, the MLP works on each position by itself, layer norms keep the numbers in range, and the embedding table that turns tokens into vectors also turns the final vectors back into a score for every token."));
    finishChapter(shell, ctx, { id: "training-data", title: "Training data", blurb: "Where pre-training text comes from, how it is cleaned, and why small models thrive on simple stories." });

    attention.start();
    return () => life.dispose();
  },
};
