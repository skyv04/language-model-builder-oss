// Chapter 6: Training data — where text comes from, how much is needed, who owns it, cleaning and deduplication.
import { h, clear, clamp } from "../dom.js";
import { icon } from "../icons.js";
import { chapterShell, finishChapter, prose, callout, figure, lifecycle, control, toggleChip, note, fmt } from "../book/kit.js";
import {
  MESSY_PAGE, FILTERS, parsePage, runPipeline, pipelineStats,
  splitWords, shingleList, jaccard, minhashSeeds, minhashSignature,
} from "../book/cleaning.js";

/* ---------- Figure 1: how much text? ---------- */

const DEFAULT_RUN_TOKENS = 2000 * 16 * 128; // default recipe: steps × excerpts per step × Pocket context
const BYTES_PER_TOKEN = 4;
const WORDS_PER_TOKEN = 0.75;

const KINDS = {
  human: { label: "Human scale", color: "#b58cff" },
  data: { label: "Datasets", color: "#2fb5c5" },
  run: { label: "Training runs", color: "#ff9f43" },
  bench: { label: "This workbench", color: "#30b158" },
};

const CORPORA = [
  {
    id: "novel", label: "A long novel", tag: "150,000 words", tokens: 2e5, kind: "human",
    text: "A thick book of about 150,000 words, the kind that keeps a keen reader busy for a couple of weeks of evenings.",
  },
  {
    id: "shakespeare", label: "Tiny Shakespeare", tag: "1.1 MB", tokens: 3.4e5, kind: "data",
    text: "A single megabyte of speeches from Shakespeare's plays and a favourite for first experiments. A model trains on it in minutes, and can start memorising it almost as quickly.",
  },
  {
    id: "pocket-run", label: "One default run", tag: "Pocket shape", tokens: DEFAULT_RUN_TOKENS, kind: "bench",
    text: "What the workbench's default pre-training recipe reads with the Pocket shape: 2,000 steps, each made of 16 excerpts that are 128 tokens long. When the corpus is smaller than that, some of it is read more than once.",
  },
  {
    id: "tinystories-slice", label: "TinyStories download", tag: "≈ 23 MB", tokens: 23e6 / BYTES_PER_TOKEN, kind: "bench",
    text: "The slice of TinyStories offered in the workbench catalog, roughly 27,000 stories. A default Pocket run gets through about two thirds of it.",
  },
  {
    id: "child", label: "Words a child hears", tag: "by age 13", tokens: 1e8 / WORDS_PER_TOKEN, kind: "human",
    text: "Estimates suggest that children have heard fewer than 100 million words by the age of thirteen. The BabyLM challenge uses that figure as a budget, asking how much a model can learn from a child-sized diet of language.",
  },
  {
    id: "lifetime", label: "A lifetime of reading", tag: "keen reader", tokens: 2.8e8 / WORDS_PER_TOKEN, kind: "human",
    text: "Our own rough sum: one 90,000-word book every week for sixty years comes to about 280 million words.",
  },
  {
    id: "tinystories", label: "TinyStories, complete", tag: "2023", tokens: 5e8, kind: "data",
    text: "About two million short stories that GPT-3.5 and GPT-4 wrote using only words a small child would know (Eldan and Li, 2023). Models with just a few million parameters learn to write fluent little tales from it.",
  },
  {
    id: "wikipedia", label: "English Wikipedia", tag: "as used for GPT-3", tokens: 3e9, kind: "data",
    text: "The English Wikipedia text that went into GPT-3 in 2020: carefully edited and densely factual, but small next to the web.",
  },
  {
    id: "openwebtext", label: "OpenWebText", tag: "2019", tokens: 9e9, kind: "data",
    text: "An open re-creation of the web pages OpenAI gathered to train GPT-2: pages that Reddit users had linked to in posts that other users voted up.",
  },
  {
    id: "gpt3", label: "GPT-3", tag: "2020", tokens: 3e11, kind: "run",
    text: "Tokens read during GPT-3's training run: mostly a filtered crawl of the web, mixed with books, Wikipedia and pages linked from Reddit, with the better sources read more than once.",
  },
  {
    id: "chinchilla", label: "Chinchilla", tag: "2022", tokens: 1.4e12, kind: "run",
    text: "A 70-billion-parameter model from DeepMind, trained on 1.4 trillion tokens to test how model size and data should grow together. It beat models several times its size.",
  },
  {
    id: "llama2", label: "Llama 2", tag: "2023", tokens: 2e12, kind: "run",
    text: "Meta's 2023 family of openly downloadable models, pre-trained on about 2 trillion tokens of publicly available text.",
  },
  {
    id: "fineweb", label: "FineWeb", tag: "2024", tokens: 1.5e13, kind: "data",
    text: "An openly published collection of cleaned and deduplicated English web pages, built by Hugging Face from a decade of Common Crawl snapshots.",
  },
  {
    id: "llama3", label: "Llama 3", tag: "2024", tokens: 1.5e13, kind: "run",
    text: "Meta's 2024 models were pre-trained on more than 15 trillion tokens, about seven times as much as Llama 2.",
  },
];

const LOG_LOW = 5;
const LOG_HIGH = Math.log10(3e13);
const LOG_TICKS = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13];
const LINEAR_MAX = 1.6e13;
const LINEAR_TICKS = [0, 5e12, 1e13, 1.5e13];

// Rounds to two significant figures, which is all the precision these numbers deserve.
function twoFigures(value) {
  if (value < 100) return value < 10 ? Math.round(value * 10) / 10 : Math.round(value);
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Math.round(value / unit) * unit;
}

const SCALE_UNITS = [[1e12, "T", "trillion"], [1e9, "B", "billion"], [1e6, "M", "million"], [1e3, "K", "thousand"]];

function shortCount(value) {
  if (value === 0) return "0";
  for (const [size, short] of SCALE_UNITS) if (value >= size) return `${twoFigures(value / size)}${short}`;
  return String(twoFigures(value));
}

function longCount(value) {
  for (const [size, , word] of SCALE_UNITS) if (value >= size) return `${twoFigures(value / size)} ${word}`;
  return fmt.int(twoFigures(value));
}

function readingTime(tokens) {
  const hours = (tokens * WORDS_PER_TOKEN) / 250 / 60;
  if (hours < 1) return "under an hour";
  if (hours < 24) return `${Math.round(hours)} hours`;
  const days = hours / 8;
  if (days < 365) return `${Math.round(days)} days`;
  const years = days / 365;
  return `${years < 10 ? twoFigures(years) : fmt.int(twoFigures(years))} years`;
}

function runsText(tokens) {
  const ratio = tokens / DEFAULT_RUN_TOKENS;
  if (ratio < 0.995) return `${Math.max(1, Math.round(ratio * 100))}% of one`;
  if (ratio < 10) return String(twoFigures(ratio));
  if (ratio < 1e6) return fmt.int(twoFigures(ratio));
  return longCount(ratio);
}

const bytesText = (bytes) => fmt.bytes(bytes).replace(/\.0 /, " ");

function sizesFigure(ctx, life) {
  const fig = figure({
    number: 1,
    title: "How much text?",
    info: "Log scale · approximate",
    wide: true,
    caption: "Text collections and training runs measured in tokens, using widely published approximate figures. Conversions to bytes and reading time assume four bytes and three quarters of a word per token, and a reader managing 250 words a minute. On the logarithmic axis every grid line is ten times the one before it. Click a row for details.",
  });
  let mode = "log";
  let items = CORPORA.slice();
  let selected = "pocket-run";
  let rowElements = new Map();

  const rows = h("div", { class: "bk-size-rows" });
  const grid = h("div", { class: "bk-size-grid", "aria-hidden": "true" });
  const axis = h("div", { class: "bk-size-axis", "aria-hidden": "true" });
  const hint = h("p", { class: "bk-size-hint" });
  const detail = h("div", { class: "bk-size-detail", "aria-live": "polite" });
  const legend = h("div", { class: "bk-size-legend" },
    Object.values(KINDS).map((kind) => h("span", { class: "bk-size-key" }, h("span", { class: "bk-size-dot", style: { "--kind": kind.color } }), kind.label)),
  );

  const position = (tokens) => (mode === "log"
    ? clamp((Math.log10(Math.max(tokens, 1)) - LOG_LOW) / (LOG_HIGH - LOG_LOW), 0, 1)
    : clamp(tokens / LINEAR_MAX, 0, 1)) * 100;

  function build() {
    clear(rows);
    rowElements = new Map();
    for (const item of items) {
      const fill = h("span", { class: "bk-size-fill" });
      const row = h("button", {
        class: ["bk-size-row", item.mine ? "mine" : ""],
        style: { "--kind": KINDS[item.kind].color },
        "aria-pressed": String(item.id === selected),
        onclick: () => select(item.id),
      },
      h("span", { class: "bk-size-name" }, h("span", { class: "bk-size-label" }, item.label), h("span", { class: "bk-size-tag" }, item.tag)),
      h("span", { class: "bk-size-track" }, fill),
      h("span", { class: "bk-size-value" }, shortCount(item.tokens)),
      );
      rowElements.set(item.id, { row, fill, item });
      rows.append(row);
    }
  }

  function render() {
    clear(grid);
    clear(axis);
    const tickValues = mode === "log" ? LOG_TICKS : LINEAR_TICKS;
    tickValues.forEach((value, index) => {
      const left = `${position(value).toFixed(2)}%`;
      grid.append(h("span", { class: "bk-size-gridline", style: { left } }));
      axis.append(h("span", { class: ["bk-size-tick", mode === "log" && index % 2 ? "minor" : ""], style: { left } }, shortCount(value)));
    });
    for (const { fill, item } of rowElements.values()) fill.style.width = `max(3px, ${position(item.tokens).toFixed(2)}%)`;
    hint.textContent = mode === "log"
      ? "Each grid line is ten times the one before it, so equal steps mean equal multiplication."
      : "The same numbers on an ordinary axis: everything below the big training runs shrinks to a sliver.";
    fig.setInfo(mode === "log" ? "Log scale · approximate" : "Linear scale · approximate");
  }

  const fact = (label, value, hintText) => h("div", { class: "bk-size-fact" },
    h("span", { class: "panel-label" }, label),
    h("span", { class: "bk-size-fact-value" }, value),
    hintText ? h("span", { class: "bk-size-fact-hint" }, hintText) : null,
  );

  function select(id) {
    const item = items.find((entry) => entry.id === id);
    if (!item) return;
    selected = id;
    for (const [key, { row }] of rowElements) row.setAttribute("aria-pressed", String(key === id));
    const kind = KINDS[item.kind];
    clear(detail);
    detail.append(
      h("div", { class: "bk-size-about bk-pop" },
        h("div", { class: "bk-size-title" },
          h("span", { class: "bk-size-dot", style: { "--kind": kind.color } }),
          h("b", {}, item.label),
          h("span", { class: "bk-size-kind" }, kind.label),
        ),
        h("p", {}, item.text),
      ),
      h("div", { class: "bk-size-facts bk-pop" },
        fact("Tokens", `≈ ${longCount(item.tokens)}`),
        fact("As plain text", `≈ ${bytesText(item.tokens * BYTES_PER_TOKEN)}`),
        fact("Reading time", readingTime(item.tokens), "8 hours a day"),
        fact("Default Pocket runs", runsText(item.tokens)),
      ),
    );
  }

  const scaleToggle = ctx.ui.segmented([
    { value: "log", label: "Log scale" },
    { value: "linear", label: "Linear scale" },
  ], mode, (value) => {
    mode = value;
    render();
  });

  fig.body.append(
    h("div", { class: "bk-toolbar bk-size-toolbar" }, scaleToggle, h("span", { class: "spacer" }), legend),
    h("div", { class: "bk-size-plot" }, grid, rows, axis),
    hint,
    detail,
  );
  build();
  render();
  select(selected);

  // With the engine running, add the reader's own corpus and planned run to the comparison.
  ctx.api.get("/api/projects/current").then((project) => {
    if (life.disposed || !project) return;
    const extra = [];
    const corpus = project.data && project.data.corpus_tokens;
    if (Number.isFinite(corpus) && corpus > 0) {
      extra.push({
        id: "my-corpus", label: "Your corpus", tag: project.data.corpus_label || "tokenized", tokens: corpus, kind: "bench", mine: true,
        text: `The pre-training text of “${project.name || "your project"}”, counted with the project's own tokenizer.`,
      });
    }
    const recipe = project.recipes && project.recipes.pretrain;
    const context = project.config && project.config.context;
    if (recipe && recipe.mode !== "until_stop" && recipe.steps > 0 && recipe.batch_size > 0 && context > 0) {
      extra.push({
        id: "my-run", label: "Your planned run", tag: `${fmt.int(recipe.steps)} steps`, tokens: recipe.steps * recipe.batch_size * context, kind: "bench", mine: true,
        text: `What your current pre-training recipe will read: ${fmt.int(recipe.steps)} steps × ${recipe.batch_size} excerpts × ${context} tokens.`,
      });
    }
    if (!extra.length) return;
    items = [...CORPORA, ...extra].sort((a, b) => a.tokens - b.tokens);
    build();
    render();
    select(selected);
  }).catch(() => {});

  return fig.element;
}

/* ---------- Figure 2: cleaning a scraped page ---------- */

const FILTER_BY_ID = new Map(FILTERS.map((filter) => [filter.id, filter]));
const KEPT_COLOR = "#1c2b4a";

function cleaningFigure(ctx) {
  const lines = parsePage(MESSY_PAGE);
  const goodCount = lines.filter((line) => line.good).length;
  const fig = figure({
    number: 2,
    title: "Cleaning a scraped page",
    info: `${lines.length} lines of HTML`,
    tone: "light",
    wide: true,
    caption: "A made-up web page, one line of its HTML source per row. Filters run from left to right, and each removed line is credited to the first filter that catches it. Every rule here is deliberately simple; real pipelines use the same ideas with far more care, and still make mistakes in both directions.",
  });
  const enabled = new Set();
  let minWords = 4;

  const doc = h("div", { class: "bk-clean-doc", tabindex: "0", "aria-label": "The page after the selected filters" });
  const chips = new Map();
  const counts = new Map();
  const about = h("div", { class: "bk-clean-about-text" });
  const startText = `This is the page as a crawler saved it, one line of HTML per row. Only ${goodCount} of its ${lines.length} lines belong to the article. Switch the filters on one at a time, from left to right, or press <b>Apply all</b>.`;

  function showAbout(filter) {
    about.innerHTML = filter ? `<b>${filter.label}.</b> ${filter.about}` : startText;
  }

  function setFilter(id, on) {
    const chip = chips.get(id);
    chip.classList.toggle("active", on);
    chip.setAttribute("aria-pressed", String(on));
    if (on) enabled.add(id);
    else enabled.delete(id);
  }

  for (const filter of FILTERS) {
    const count = h("span", { class: "bk-clean-count" });
    const chip = toggleChip([h("span", { class: "bk-clean-dot", style: { "--why": filter.color } }), filter.label, count], false, (on) => {
      if (on) enabled.add(filter.id);
      else enabled.delete(filter.id);
      showAbout(filter);
      update();
    }, { title: filter.standard ? "" : "Not included in Apply all" });
    chip.classList.add("bk-clean-chip");
    chips.set(filter.id, chip);
    counts.set(filter.id, count);
  }

  const slider = ctx.ui.slider({
    min: 1, max: 10, step: 1, value: minWords, label: "Minimum words per line",
    onInput: (value) => {
      minWords = value;
      if (!enabled.has("short")) {
        setFilter("short", true);
        showAbout(FILTER_BY_ID.get("short"));
      }
      update();
    },
  });
  const minControl = control("Short lines: minimum words", slider, { value: `${minWords} words` });

  const hideToggle = toggleChip("Hide removed lines", false, (on) => doc.classList.toggle("hide-removed", on));
  const applyAll = h("button", {
    class: "btn small primary",
    onclick: () => {
      for (const filter of FILTERS) setFilter(filter.id, filter.standard);
      minWords = 4;
      slider.setValue(minWords);
      about.innerHTML = "<b>Standard filters on.</b> Every line of junk is gone and the whole article survived, which is easy on a page written to be cleaned. Now try <b>Strict endings</b>, or drag the minimum-words slider up.";
      update();
    },
  }, icon("wand", 13), "Apply all");
  const reset = h("button", {
    class: "btn small",
    onclick: () => {
      for (const filter of FILTERS) setFilter(filter.id, false);
      minWords = 4;
      slider.setValue(minWords);
      showAbout(null);
      update();
    },
  }, icon("reset", 13), "Reset");

  // Before → after numbers.
  const cells = {};
  const statRow = (key, label) => {
    cells[key] = { before: h("span", { class: "bk-clean-num" }), after: h("span", { class: "bk-clean-num after" }) };
    return h("div", { class: "bk-clean-stat" }, h("span", { class: "bk-clean-stat-label" }, label), cells[key].before, h("span", { class: "bk-clean-arrow" }, "→"), cells[key].after);
  };
  const keptShare = h("span", { class: "bk-clean-share" });
  const bar = h("div", { class: "bk-clean-bar", role: "img" });
  const scoreGood = h("span", { class: "bk-clean-score-value" });
  const scoreJunk = h("span", { class: "bk-clean-score-value" });
  const meterGood = h("span", { class: "bk-clean-meter-fill good" });
  const meterJunk = h("span", { class: "bk-clean-meter-fill junk" });
  const score = (label, value, meter) => h("div", { class: "bk-clean-score" },
    h("span", { class: "bk-clean-score-label" }, label), value,
    h("span", { class: "bk-clean-meter" }, meter),
  );

  const side = h("div", { class: "bk-clean-side" },
    h("div", { class: "bk-clean-stats" },
      h("div", { class: "bk-clean-stat head" }, h("span", {}), h("span", { class: "bk-label" }, "Before"), h("span", {}), h("span", { class: "bk-label" }, "After")),
      statRow("lines", "Lines"),
      statRow("bytes", "Size"),
      statRow("tokens", "≈ Tokens"),
    ),
    bar,
    keptShare,
    h("div", { class: "bk-clean-scores" },
      score("Article lines kept", scoreGood, meterGood),
      score("Junk lines removed", scoreJunk, meterJunk),
    ),
    h("div", { class: "bk-clean-about" }, h("span", { class: "bk-label bk-section-label" }, "What this step does"), about),
  );

  const legendItem = (className, text) => h("span", { class: "bk-clean-legend-item" }, h("span", { class: `bk-clean-swatch ${className}` }), text);

  function update() {
    const results = runPipeline(lines, enabled, { minWords });
    const stats = pipelineStats(results);
    minControl.setValue(`${minWords} word${minWords === 1 ? "" : "s"}`);
    minControl.classList.toggle("bk-off", !enabled.has("short"));

    for (const filter of FILTERS) {
      const removed = stats.removedLines[filter.id];
      counts.get(filter.id).textContent = enabled.has(filter.id) ? String(removed) : "";
    }

    clear(doc);
    for (const line of results) {
      const filter = line.removedBy ? FILTER_BY_ID.get(line.removedBy) : null;
      const shown = line.removedBy === "markup" ? line.raw : line.current;
      const state = filter ? (line.good ? "removed lost" : "removed") : line.good ? "kept" : "kept junk";
      doc.append(h("div", { class: `bk-clean-line ${state}` },
        h("span", { class: "bk-clean-no" }, String(line.index + 1)),
        h("span", { class: "bk-clean-text" }, shown || " "),
        filter ? h("span", { class: "bk-clean-why", style: { "--why": filter.color } }, line.good ? `${filter.label}: article lost` : filter.label) : null,
      ));
    }

    const tokens = (bytes) => fmt.int(Math.round(bytes / BYTES_PER_TOKEN));
    cells.lines.before.textContent = String(stats.linesBefore);
    cells.lines.after.textContent = String(stats.linesAfter);
    cells.bytes.before.textContent = bytesText(stats.bytesBefore);
    cells.bytes.after.textContent = bytesText(stats.bytesAfter);
    cells.tokens.before.textContent = tokens(stats.bytesBefore);
    cells.tokens.after.textContent = tokens(stats.bytesAfter);
    const share = stats.bytesAfter / stats.bytesBefore;
    keptShare.textContent = share > 0.999 ? "Nothing removed yet." : `${Math.round(share * 100)}% of the bytes survive.`;

    clear(bar);
    const segments = [{ label: "Kept", color: KEPT_COLOR, bytes: stats.bytesAfter }];
    for (const filter of FILTERS) {
      if (stats.removedBytes[filter.id] > 0) segments.push({ label: filter.label, color: filter.color, bytes: stats.removedBytes[filter.id] });
    }
    for (const segment of segments) {
      bar.append(h("span", {
        class: "bk-clean-seg",
        title: `${segment.label}: ${bytesText(segment.bytes)}`,
        style: { width: `${((segment.bytes / stats.bytesBefore) * 100).toFixed(2)}%`, background: segment.color },
      }));
    }
    bar.setAttribute("aria-label", segments.map((segment) => `${segment.label} ${bytesText(segment.bytes)}`).join(", "));

    scoreGood.textContent = `${stats.goodKept} / ${stats.goodTotal}`;
    scoreJunk.textContent = `${stats.junkRemoved} / ${stats.junkTotal}`;
    meterGood.style.width = `${(stats.goodKept / Math.max(1, stats.goodTotal)) * 100}%`;
    meterJunk.style.width = `${(stats.junkRemoved / Math.max(1, stats.junkTotal)) * 100}%`;
    scoreGood.parentElement.classList.toggle("warn", stats.goodKept < stats.goodTotal);
  }

  fig.body.append(
    h("div", { class: "bk-clean-filters", role: "group", "aria-label": "Filters" }, [...chips.values()]),
    h("div", { class: "bk-clean-actions" }, applyAll, reset, h("span", { class: "spacer" }), minControl, hideToggle),
    h("div", { class: "bk-clean-main" },
      h("div", { class: "bk-clean-left" },
        doc,
        h("div", { class: "bk-clean-legend" },
          legendItem("junk", "junk still in the corpus"),
          legendItem("removed", "removed (the tag names the filter)"),
          legendItem("lost", "article text lost"),
        ),
      ),
      side,
    ),
  );
  showAbout(null);
  update();
  return fig.element;
}

/* ---------- Figure 3: near-duplicate detector ---------- */

const DUP_PRESETS = [
  {
    id: "news", label: "Copied news",
    a: "The town council voted on Tuesday to close Mill Street to cars every Sunday from June to September, turning it into a market for local growers and makers. Shop owners on the street asked for the change after a trial last summer brought in twice the usual number of visitors.",
    b: "On Tuesday the town council voted to close Mill Street to cars every Sunday from June to September, turning it into a market for local growers and makers. Shop owners asked for the change after a trial last summer brought in twice the usual number of visitors, the council said.",
  },
  {
    id: "spam", label: "Template spam",
    a: "Looking for the best plumber in Leeds? Our friendly team offers fast, affordable repairs 24 hours a day. Call now for a free quote and get 10% off your first job!",
    b: "Looking for the best plumber in Bristol? Our friendly team offers fast, affordable repairs 24 hours a day. Call now for a free quote and get 10% off your first job!",
  },
  {
    id: "reworded", label: "Reworded",
    a: "To fix a puncture, take the wheel off, lever one side of the tyre over the rim and pull out the tube. Find the hole, patch it, and check inside the tyre for whatever caused it.",
    b: "Got a flat? Remove the wheel, prise the tyre away from the rim with two levers and draw the inner tube out. Locate the leak, glue on a patch, then feel inside the tyre for thorns.",
  },
];
const DUP_THRESHOLD = 0.5;

function renderHighlighted(pane, parts, list, shared) {
  const marked = new Array(parts.length).fill(false);
  for (const shingle of list) {
    if (shared.has(shingle.key)) for (const index of shingle.parts) marked[index] = true;
  }
  clear(pane);
  let run = null;
  parts.forEach((part, index) => {
    const on = marked[index] || (!part.word && marked[index - 1] && marked[index + 1]);
    if (on) {
      if (!run) {
        run = h("mark", { class: "bk-dup-mark" });
        pane.append(run);
      }
      run.append(part.text);
    } else {
      run = null;
      pane.append(part.text);
    }
  });
}

function dedupFigure(ctx, life) {
  const fig = figure({
    number: 3,
    title: "Near-duplicate detector",
    info: "3-word shingles · 64 hashes",
    caption: "Both documents are cut into overlapping three-word shingles; highlighted words belong to shingles the two share. The exact overlap divides shared shingles by all distinct shingles. MinHash estimates the same number from two signatures of 64 values each, which is what makes the method affordable across billions of documents. Edit the texts to try your own.",
  });
  const seeds = minhashSeeds(64);
  let texts = { a: DUP_PRESETS[0].a, b: DUP_PRESETS[0].b };
  let editing = false;
  const debounce = life.timer();

  const paneA = h("div", { class: "bk-dup-text" });
  const paneB = h("div", { class: "bk-dup-text" });
  const area = (key) => {
    const element = h("textarea", { class: "text-input bk-textarea bk-dup-area", rows: 6, spellcheck: false, "aria-label": `Document ${key.toUpperCase()}`, hidden: true });
    element.addEventListener("input", () => {
      texts[key] = element.value;
      presets.set(null);
      debounce.set(update, 120);
    });
    return element;
  };
  const areaA = area("a");
  const areaB = area("b");

  const presets = ctx.ui.segmented(DUP_PRESETS.map((preset) => ({ value: preset.id, label: preset.label })), DUP_PRESETS[0].id, (id) => {
    const preset = DUP_PRESETS.find((entry) => entry.id === id);
    texts = { a: preset.a, b: preset.b };
    areaA.value = preset.a;
    areaB.value = preset.b;
    update();
  });
  const editLabel = h("span", {}, "Edit texts");
  const editButton = h("button", { class: "btn small", onclick: () => setEditing(!editing) }, icon("pencil", 12), editLabel);

  function setEditing(next) {
    editing = next;
    areaA.value = texts.a;
    areaB.value = texts.b;
    for (const element of [areaA, areaB]) element.hidden = !editing;
    for (const element of [paneA, paneB]) element.hidden = editing;
    editLabel.textContent = editing ? "Show overlap" : "Edit texts";
    if (editing) areaA.focus();
    update();
  }

  const stats = {
    a: ctx.ui.stat("Shingles in A", "—"),
    b: ctx.ui.stat("In B", "—"),
    shared: ctx.ui.stat("Shared", "—"),
    exact: ctx.ui.stat("Exact overlap", "—"),
    estimate: ctx.ui.stat("MinHash estimate", "—"),
  };
  const cells = seeds.map(() => h("span", { class: "bk-dup-cell" }));
  const signature = h("div", { class: "bk-dup-signature", role: "img" }, cells);
  const agreeText = h("span", { class: "bk-dup-agree" });
  const verdict = h("div", { class: "bk-dup-verdict" });

  function update() {
    const partsA = splitWords(texts.a);
    const partsB = splitWords(texts.b);
    const listA = shingleList(partsA);
    const listB = shingleList(partsB);
    const setA = new Set(listA.map((shingle) => shingle.key));
    const setB = new Set(listB.map((shingle) => shingle.key));
    const shared = new Set([...setA].filter((key) => setB.has(key)));
    const exact = jaccard(setA, setB);
    const ready = setA.size > 0 && setB.size > 0;
    const sigA = minhashSignature(setA, seeds);
    const sigB = minhashSignature(setB, seeds);
    let agree = 0;
    cells.forEach((cell, index) => {
      const same = ready && sigA[index] === sigB[index];
      cell.classList.toggle("on", same);
      if (same) agree += 1;
    });
    const estimate = ready ? agree / seeds.length : NaN;

    stats.a.set(String(setA.size));
    stats.b.set(String(setB.size));
    stats.shared.set(String(shared.size));
    stats.exact.set(ready ? exact.toFixed(2) : "—");
    stats.estimate.set(ready ? estimate.toFixed(2) : "—");
    agreeText.textContent = ready ? `${agree} of ${seeds.length} slots agree` : "";
    signature.setAttribute("aria-label", ready ? `${agree} of ${seeds.length} MinHash slots agree` : "No signature yet");

    clear(verdict);
    if (!ready) {
      verdict.className = "bk-dup-verdict";
      verdict.append(note("Each document needs at least three words to make a shingle."));
    } else {
      const duplicate = estimate >= DUP_THRESHOLD;
      verdict.className = `bk-dup-verdict ${duplicate ? "dup" : "keep"}`;
      verdict.append(
        icon(duplicate ? "layers" : "circle-check", 15),
        h("span", { html: duplicate
          ? `<b>Near-duplicates.</b> The estimated overlap of ${estimate.toFixed(2)} is above the threshold of ${DUP_THRESHOLD}, so a pipeline would keep only one of the two.`
          : `<b>Different documents.</b> The estimated overlap of ${estimate.toFixed(2)} is below the threshold of ${DUP_THRESHOLD}, so both would stay.` }),
      );
    }
    if (!editing) {
      renderHighlighted(paneA, partsA, listA, shared);
      renderHighlighted(paneB, partsB, listB, shared);
    }
  }

  const column = (label, pane, textArea) => h("div", { class: "bk-dup-col" }, h("span", { class: "panel-label bk-section-label" }, label), pane, textArea);
  fig.body.append(
    h("div", { class: "bk-toolbar" }, presets, h("span", { class: "spacer" }), editButton),
    h("div", { class: "bk-dup-texts" }, column("Document A", paneA, areaA), column("Document B", paneB, areaB)),
    h("div", { class: "bk-stats bk-dup-stats" }, Object.values(stats)),
    h("div", { class: "bk-dup-sig-row" },
      signature,
      h("p", { class: "bk-dup-sig-text" },
        h("b", {}, "MinHash signatures. "),
        "Each square is one of 64 hash functions. It lights up when both documents produce the same smallest hash, which happens with a probability equal to their overlap.",
        h("span", { class: "bk-dup-agree-wrap" }, agreeText),
      ),
    ),
    verdict,
  );
  update();
  return fig.element;
}

/* ---------- chapter ---------- */

export default {
  id: "training-data",
  title: "Training data",
  render(root, ctx) {
    const life = lifecycle();
    const shell = chapterShell(root, "training-data",
      "A language model learns everything it knows from the text it trains on: its vocabulary, its grammar, its facts, its tone and its blind spots. This chapter is about that text: where it comes from, how much is needed, who it belongs to, and how a raw pile of web pages is cleaned into something worth learning from.");
    const { article } = shell;

    article.append(prose(`
      <h2>You are what you read</h2>
      <p>Chapter 2 described a language model as a machine that has studied which pieces of text tend to follow which. Everything in that description depends on the text it studied. Train a network on Shakespeare's plays and it writes speeches for kings and clowns; train the same network on recipes and it tells you to preheat the oven; train it on both and you mostly get a muddle of the two. A model cannot know about anything its data never mentions, and it will not write better than the text it learned from.</p>
      <p>It also learns the quirks of its data as faithfully as the virtues. If a quarter of the training text is cookie notices and navigation menus, a quarter of the model's effort goes into learning to write cookie notices. If one paragraph appears ten thousand times, the model learns it by heart. If the text carries stereotypes or mistakes, the model absorbs those too, as the word vectors of Chapter 4 did. That is why the teams behind large models now spend as much care on their data as on their architecture, and why, in the workbench, choosing what your model will read comes straight after choosing its shape.</p>

      <h2>Where the text comes from</h2>
      <p>Large training sets are assembled from a handful of kinds of source, each with its own character:</p>
      <ul>
        <li><strong>The open web.</strong> By far the biggest source. Most projects start from Common Crawl, a non-profit that has been archiving the public web since 2008 and publishes a new snapshot of billions of pages every month or two. The web covers every topic and style imaginable, and most of it is junk: menus, adverts, spam, machine-made pages and endless near-copies.</li>
        <li><strong>Reference works.</strong> Wikipedia is small by comparison but carefully written, densely factual and openly licensed, so nearly every model trains on it, often more than once.</li>
        <li><strong>Books.</strong> Long, coherent text teaches a model to stay on topic for pages at a time. Books whose copyright has expired, such as the tens of thousands digitised by Project Gutenberg, can be used freely; most modern books cannot.</li>
        <li><strong>Code and science.</strong> Public code repositories and paper archives such as arXiv add precise, structured writing. Even models that are not meant to write code seem to reason better with some in the mix.</li>
        <li><strong>Conversation.</strong> Forums and question-and-answer sites show how people ask for help and explain things to each other, along with plenty of arguing and noise.</li>
        <li><strong>Synthetic text.</strong> A growing share is written by other language models on purpose: stories, textbook chapters and exercises produced to order. Two datasets in the workbench catalog, TinyStories and Cosmopedia, were made this way.</li>
      </ul>
      <p>The builders then choose a <strong>mixture</strong>: how much of each source the model sees during training. It rarely matches the raw sizes. GPT-3, for example, read its three-billion-token copy of Wikipedia more than three times over, but got through less than half of its filtered web crawl. Text judged to be better earns a bigger share of the model's attention.</p>

      <h2>How much text is that?</h2>
      <p>Training data is measured in <strong>tokens</strong>, the pieces a tokenizer cuts text into (Chapter 3). With a tokenizer like GPT-2's, a token of ordinary English averages about four characters, or roughly three quarters of a word, so a megabyte of plain text comes to about a quarter of a million tokens. A tokenizer with a smaller vocabulary cuts text into shorter pieces, so the same megabyte becomes somewhat more tokens.</p>
    `));
    article.append(sizesFigure(ctx, life));
    article.append(prose(`
      <p>The range is hard to take in. The largest runs in Figure 1 read nearly a hundred million times more text than a long novel, and even a deliberately small dataset like TinyStories holds more words than a child hears while growing up. Switch the figure to a linear scale to see why such charts are almost always drawn on a logarithmic one.</p>
      <p>How much text does a model need? In 2022 a team at DeepMind trained more than four hundred models of different sizes on different amounts of data, and found that for the best result from a fixed budget of computation, the training data should grow in step with the model, at roughly <strong>twenty tokens per parameter</strong>. Their 70-billion-parameter model Chinchilla, trained that way on 1.4 trillion tokens, beat models several times its size. By that rule of thumb the Pocket shape, with about 1.3 million parameters, would like some 27 million tokens. A default pre-training run reads about 4 million, so it stops well short, and if your machine can afford a longer run, your model will keep improving.</p>
      <p>The rule describes the cheapest way to <em>train</em> a model of a given quality, not the cheapest one to <em>use</em>. A smaller model trained for much longer costs less every time someone asks it a question, so recent models read far more than twenty tokens per parameter: the 8-billion-parameter version of Llama 3 saw more than 15 trillion tokens, nearly two thousand for each parameter.</p>
      <p>When a run needs more tokens than the corpus holds, the model reads the same text again. One complete pass over the data is called an <strong>epoch</strong>. A few passes over good text do little harm, but after many, a model starts to memorise its training text instead of learning patterns that carry over to text it has never seen. Chapter 7 shows how to spot that moment. The Pre-training page counts the passes for you.</p>
    `));
    article.append(callout("linux", `Text gets much smaller once it is tokenized. The workbench stores each token of your corpus as a 16-bit number, so the 23 MB TinyStories download becomes roughly 12 MB of token ids on disk, and training reads them straight from the file instead of loading them all into memory. On a phone or a small laptop, the size of the corpus is rarely what runs out; the model's shape and the batch size are (Chapter 5). Tokenizing a large corpus does take a while, but it happens once.`));
    article.append(prose(`
      <h2>Whose text is it?</h2>
      <p>Text on the web is easy to download, but that does not make it free to use. In most countries, nearly everything people write is protected by copyright the moment it is written, and the owner decides who may copy it. Some text is explicitly shared. Works in the <strong>public domain</strong>, such as old books whose copyright has run out, can be used by anyone. Authors can also choose an open licence such as Creative Commons: CC0 gives up all rights, CC BY allows reuse as long as the author is credited, CC BY-SA (the licence of Wikipedia) adds that anything built from the work must be shared on the same terms, and licences marked NC rule out commercial use.</p>
      <p>Whether training a model on copyrighted text without permission is allowed is one of the open legal questions of the moment. Courts in several countries are hearing cases about it, and the answer may differ from place to place. Meanwhile, many websites use a file called <em>robots.txt</em> to ask crawlers to stay away, some dataset builders honour requests to opt out, and some companies pay publishers for access. Privacy is a separate problem: scraped pages contain the names, email addresses and phone numbers of people who never expected to end up in a model, so careful pipelines detect and remove personal information.</p>
    `));
    article.append(callout("linux", `Every dataset card on the <strong>Pre-training data</strong> page shows its licence and a link to its source. All the built-in corpora are public domain or openly licensed, but their terms differ: the Wikipedia-based sets ask for credit and share-alike, and one of the fine-tuning sets rules out commercial use. If you plan to share a model you trained, check the licence of everything it read. This is general information, not legal advice.`, "In the workbench"));
    article.append(prose(`
      <h2>From web page to training text</h2>
      <p>A web page as it arrives from a crawler is mostly not prose. Figure 2 shows a typical one, a made-up repair guide on a made-up cycling site, as raw HTML. The article is in there, surrounded by a cookie banner, menus, a star rating, an advert, share buttons, comments and a footer, with scripts and styling code at either end. Switch on the filters one at a time and watch the page shrink towards the text a model should learn from.</p>
    `));
    article.append(cleaningFigure(ctx));
    article.append(prose(`
      <p>Each toggle in the figure stands for a whole stage of a real pipeline:</p>
      <ul>
        <li><strong>Text extraction.</strong> Instead of just deleting tags, real extractors study the structure of the page to guess which block holds the main content, and keep headings, paragraphs and lists apart.</li>
        <li><strong>Language identification.</strong> A small, fast classifier trained on character patterns labels each document with a language and a confidence score, and the pipeline keeps the languages it wants.</li>
        <li><strong>Quality rules.</strong> Simple rules catch a surprising amount: too few words, too many symbols, too many repeated lines, lines that do not end in punctuation. Google's C4 dataset, for example, kept only lines ending in a full stop, question mark, exclamation mark or quotation mark, dropped lines that mentioned JavaScript, and discarded every page containing a curly bracket, which kept out code at a stroke.</li>
        <li><strong>Model-based filters.</strong> Newer pipelines ask a large model to grade a sample of pages, for instance on how educational they are, then train a small classifier to copy those grades across billions of pages. Hugging Face's FineWeb-Edu kept only about a tenth of its source this way, and models trained on it scored noticeably better on knowledge and reasoning tests.</li>
        <li><strong>Safety and fairness.</strong> Filters for adult content, hate speech and personal information, and the removal of test questions from well-known benchmarks, so that later evaluations measure understanding rather than memory.</li>
      </ul>
      <p>Every filter is a trade-off. Each one throws away some good text with the bad, as the strict-endings rule in Figure 2 does to the headings and the list of tools, and the losses are not spread evenly. A 2021 study of C4 found that its list of banned words removed a disproportionate share of text written in African-American and Hispanic-aligned English, and of text mentioning gay and lesbian people, much of it not offensive at all. A filter decides whose writing a model learns from, so it deserves as much scrutiny as the model itself.</p>
    `));
    article.append(callout("try", `In Figure 2, press <strong>Apply all</strong>, then also switch on <strong>Strict endings</strong> and watch the headings and the list of tools disappear. Switch it off again and drag the minimum-words slider up to 8. Which version would you rather train on? There is no perfect setting, only a balance between keeping junk and losing good text.`));
    article.append(prose(`
      <h2>Duplicates</h2>
      <p>The web repeats itself constantly. One news story appears on hundreds of sites, disclaimers and licences are pasted under millions of pages, spam is stamped out from templates, and popular texts are copied wherever they fit. Left alone, a training set can contain thousands of copies of the same paragraph.</p>
      <p>Repetition costs twice. It wastes computation, because the model spends its steps rereading text it already knows. Worse, the model learns repeated text far more strongly than anything else, so it may reproduce it word for word, personal details included. Researchers at Google found in 2021 that models trained on deduplicated versions of standard datasets produced memorised text about ten times less often, and reached the same quality in fewer training steps.</p>
      <p>Exact copies are easy to catch: compute a short fingerprint, or <em>hash</em>, of every document and keep one document per fingerprint. Near-copies, which differ by a date, a name or an edited sentence, need a cleverer trick. The usual one cuts each document into overlapping runs of words called <strong>shingles</strong> and measures how many shingles two documents share. Comparing every pair of documents directly is out of the question at web scale, so pipelines use <strong>MinHash</strong>. Each document is boiled down to a short signature: for each of a few dozen different hash functions, the smallest hash value any of its shingles produces. Two documents agree in any one slot with a probability equal to the overlap of their shingle sets, so comparing two short signatures estimates the overlap without ever comparing the full texts.</p>
    `));
    article.append(dedupFigure(ctx, life));
    article.append(prose(`
      <p>Real deduplication adds one more trick, called locality-sensitive hashing: signatures are split into bands and sorted into buckets, so that only documents that land in the same bucket are ever compared. That is how collections of billions of documents can be deduplicated without comparing every pair.</p>

      <h2>Why small models want simple data</h2>
      <p>A model with a few million parameters has very little room. Trained on a slice of the open web, it has to spread that room across recipes, football scores, legal disclaimers, program code and dozens of languages, and it ends up good at none of them: its output has the texture of text but makes no sense. The same network trained on narrow, consistent text can become remarkably good at that one kind of writing.</p>
      <p>The clearest demonstration is <strong>TinyStories</strong>, a dataset created in 2023 by Ronen Eldan and Yuanzhi Li at Microsoft Research. They had GPT-3.5 and GPT-4 write millions of short stories using only words that a three- or four-year-old would understand, and kept the stories varied by asking for a few particular words and story features in each one. Models with fewer than ten million parameters, small enough to train on a laptop, learned to write stories with correct grammar, consistent characters and a simple plot. That is why it is the recommended starting point in the workbench.</p>
      <p>The other corpora in the catalog suit other goals. Tiny Shakespeare trains in minutes and its distinctive layout appears quickly, but at about a megabyte it is also memorised quickly. Simple English Wikipedia, the classic novels and the synthetic textbooks of Cosmopedia have far more variety, so they need a bigger shape and a longer run before the output reads well. Whatever you choose, expect your model's writing to sound like its data. That is not a flaw; it is the whole point.</p>

      <h2>Bringing your own text</h2>
      <p>The <strong>Add your own</strong> button on the Pre-training data page accepts plain-text files (<code>.txt</code> or <code>.md</code>, encoded as UTF-8) of up to 200 megabytes. A little preparation goes a long way:</p>
      <ul>
        <li><strong>Mark where documents end.</strong> Separate them with two or more blank lines, or with a line containing only <code>&lt;|endoftext|&gt;</code>. The model then learns where texts begin and end, and never mistakes the end of one for the start of the next.</li>
        <li><strong>Remove what you would not want repeated:</strong> page numbers, running headers, tables of contents, and the licence text that Project Gutenberg adds to the start and end of each of its books.</li>
        <li><strong>Aim for at least a few megabytes.</strong> A few hundred kilobytes trains quickly but is soon memorised.</li>
        <li><strong>Keep the style consistent.</strong> A model this small learns one voice far better than five.</li>
        <li><strong>Prefer many documents to one long one.</strong> The workbench sets aside one document in every fifty as validation text that the model never trains on, so it can check how well the model copes with writing it has not seen. With fewer than fifty documents, it holds back the last part of the text instead.</li>
      </ul>
    `));
    article.append(callout("key", "A model is a compressed summary of the text it read. Choosing that text, making sure it may be used, cleaning it and removing duplicates are not chores to rush through before the real work: they are among the strongest levers you have over what the model becomes."));
    finishChapter(shell, ctx, { id: "how-models-learn", title: "How models learn", blurb: "Loss, gradients and learning rates: what happens in a single training step, and how to tell when a run is going well." });

    return () => life.dispose();
  },
};
