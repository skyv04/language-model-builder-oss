// Chapter 1: Introduction — what the book and the workbench are, and the journey ahead.
import { h, clear } from "../dom.js";
import { icon } from "../icons.js";
import { chapterShell, finishChapter, prose, callout, figure, lifecycle, CHAPTERS } from "../book/kit.js";

const PHASES = {
  setup: { label: "Set up", color: "#2fb5c5" },
  pretrain: { label: "Pre-train", color: "#2f8cff" },
  finetune: { label: "Fine-tune", color: "#a77bff" },
  align: { label: "Prefer & chat", color: "#ff9f43" },
};

const STAGES = [
  {
    id: "project-setup", short: "Setup", title: "Project setup", icon: "folder-cog", phase: "setup", read: "anatomy-of-a-transformer",
    text: "Name the project and decide how big the model should be. Its shape — how much text it sees at once, how wide its internal vectors are and how many blocks are stacked — sets the parameter count. You also choose a tokenizer: a compact vocabulary trained on your own data, or the ready-made GPT-2 vocabulary.",
    points: ["A live blueprint of the model with its parameter count", "Memory needs and measured speed on your CPU and GPU", "Presets from Pocket to Large, or your own sliders"],
  },
  {
    id: "pretraining-data", short: "Data", title: "Pre-training data", icon: "file-down", phase: "pretrain", read: "training-data",
    text: "Choose the text your model will learn from. The catalog lists openly licensed corpora that download on demand, each with its source, license and a preview of what the model will actually read. You can also add your own plain-text files.",
    points: ["Simple stories are the best first choice for a tiny model", "Everything is stored locally; nothing is uploaded", "Preview the text before you commit to it"],
  },
  {
    id: "pretraining", short: "Pre-train", title: "Pre-training", icon: "brain", phase: "pretrain", read: "how-models-learn",
    text: "The long, important step. The model reads the corpus in small batches and nudges its parameters so that each next token becomes a little less surprising. You watch the loss fall, read samples that improve from gibberish to sentences, and can pause and resume at any time.",
    points: ["Training and validation loss, live", "A sample saved at every checkpoint", "Runs keep going after you close the window"],
  },
  {
    id: "sampling", short: "Sample", title: "Sampling", icon: "dice", phase: "pretrain", read: "what-is-a-language-model",
    text: "Hand your freshly trained base model the start of a text and let it continue. Temperature, top-k and top-p decide how adventurous it is. Click any generated token to see the alternatives it weighed, and branch the story from there.",
    points: ["Tokens shaded by how confident the model was", "Presets from Predictable to Creative", "Compare settings side by side"],
  },
  {
    id: "finetuning-data", short: "Chats", title: "Fine-tuning data", icon: "list-checks", phase: "finetune", read: "how-models-learn-behavior",
    text: "Pick a collection of example conversations — a user asks, an assistant answers. These don't add much knowledge; they teach the model the shape of a dialogue and the voice of a helpful reply.",
    points: ["Catalog of chat and instruction sets", "Examples rendered as chat bubbles", "Import your own .jsonl, .json or .csv"],
  },
  {
    id: "sft", short: "SFT", title: "Supervised fine-tuning", icon: "graduation", phase: "finetune", read: "how-models-learn-behavior",
    text: "Continue training from the base model, now on conversations wrapped in a chat template with special role tokens. Only the assistant's words count towards the loss, so the model learns to answer rather than to imitate the person asking.",
    points: ["Starts from your latest pre-training checkpoint", "A shorter run with a gentler learning rate", "Samples show the assistant voice emerging"],
  },
  {
    id: "dpo", short: "Prefer", title: "Direct preference optimization", icon: "thumbs-up", phase: "align", read: "how-models-learn-behavior",
    text: "Your fine-tuned model answers the same prompt twice and you pick the better reply. After a handful of choices, a short preference-tuning run nudges the model towards answers like the ones you chose — no separate reward model required.",
    points: ["A ballot of two answers per prompt", "Four or more choices unlock a run", "A frozen copy keeps the model from drifting too far"],
  },
  {
    id: "chat", short: "Chat", title: "Chat with your model", icon: "messages", phase: "align", read: "how-models-learn-behavior",
    text: "Talk to what you built. Switch between the base, fine-tuned and preference-tuned versions to feel what each stage changed. X-ray mode shows every token with its probability, and you can export the finished model as a zip file.",
    points: ["Base, SFT and DPO checkpoints", "Raw transcript with special tokens", "Share the model as files others can load"],
  },
];

const chapterTitle = (id) => (CHAPTERS.find((chapter) => chapter.id === id) || {}).title || id;

function completedSet(ctx) {
  const state = ctx.state && ctx.state();
  return new Set((state && state.settings && state.settings.completed) || []);
}

function journeyFigure(ctx, life) {
  const fig = figure({
    number: 2,
    title: "Your build, step by step",
    info: "",
    wide: true,
    caption: "The eight workbench pages under <em>Build your model</em>, in order. Select a step to see what happens there; finished steps get a check mark.",
  });
  let selected = 0;
  const buttons = [];
  const track = h("ol", { class: "bk-journey-track", role: "listbox", "aria-label": "Workbench steps" });
  const phaseRow = h("div", { class: "bk-journey-phases", "aria-hidden": "true" });
  let column = 1;
  for (const [key, phase] of Object.entries(PHASES)) {
    const span = STAGES.filter((stage) => stage.phase === key).length;
    phaseRow.append(h("span", { class: "bk-journey-phase", style: { gridColumn: `${column} / span ${span}`, "--c": phase.color } }, phase.label));
    column += span;
  }
  STAGES.forEach((stage, index) => {
    const color = PHASES[stage.phase].color;
    const button = h("button", {
      class: "bk-stop",
      role: "option",
      style: { "--c": color },
      "aria-label": `${index + 1}. ${stage.title}`,
      onclick: () => select(index, true),
      onkeydown: (event) => {
        const delta = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
        if (!delta) return;
        event.preventDefault();
        select((index + delta + STAGES.length) % STAGES.length, true);
        buttons[selected].focus();
      },
    },
    h("span", { class: "bk-stop-dot" }, icon(stage.icon, 19), h("span", { class: "bk-stop-check" }, icon("check", 11))),
    h("span", { class: "bk-stop-num" }, String(index + 1)),
    h("span", { class: "bk-stop-title" }, stage.short));
    buttons.push(button);
    track.append(h("li", {}, button));
  });

  const detail = h("div", { class: "bk-journey-detail", "aria-live": "polite" });

  function renderDetail() {
    const stage = STAGES[selected];
    const phase = PHASES[stage.phase];
    const done = completedSet(ctx).has(stage.id);
    clear(detail);
    detail.style.setProperty("--c", phase.color);
    detail.append(
      h("div", { class: "bk-journey-detail-main" },
        h("div", { class: "bk-journey-detail-head" },
          h("span", { class: "bk-journey-badge" }, icon(stage.icon, 18)),
          h("div", {},
            h("div", { class: "panel-label" }, `Step ${selected + 1} of ${STAGES.length} · ${phase.label}`),
            h("div", { class: "bk-journey-detail-title" }, stage.title, done ? h("span", { class: "bk-journey-done" }, icon("circle-check", 14), "done") : null),
          ),
        ),
        h("p", { class: "bk-journey-text" }, stage.text),
        h("ul", { class: "bk-journey-points" }, stage.points.map((point) => h("li", {}, point))),
      ),
      h("div", { class: "bk-journey-actions" },
        h("button", { class: "btn primary", onclick: () => ctx.navigate(stage.id) }, "Go there", icon("chevron-right", 15)),
        h("button", { class: "btn small ghost bk-journey-read", onclick: () => ctx.navigate(stage.read) }, icon("book-open", 13), `Background: ${chapterTitle(stage.read)}`),
      ),
    );
  }

  function refreshMarks() {
    const done = completedSet(ctx);
    let count = 0;
    STAGES.forEach((stage, index) => {
      const isDone = done.has(stage.id);
      if (isDone) count += 1;
      buttons[index].classList.toggle("done", isDone);
      buttons[index].classList.toggle("active", index === selected);
      buttons[index].setAttribute("aria-selected", String(index === selected));
      buttons[index].tabIndex = index === selected ? 0 : -1;
    });
    fig.setInfo(count ? `${count} of ${STAGES.length} done` : "8 steps · 4 phases");
  }

  function select(index, userAction) {
    selected = index;
    refreshMarks();
    renderDetail();
    if (userAction) detail.classList.remove("bk-pop");
    if (userAction) {
      void detail.offsetWidth;
      detail.classList.add("bk-pop");
    }
  }

  fig.body.append(h("div", { class: "bk-journey" }, phaseRow, track), detail);
  select(0, false);
  ctx.onState(() => {
    if (life.disposed) return;
    refreshMarks();
    renderDetail();
  });
  return fig.element;
}

// Parameter counts for the GPT-2 style recipe the engine uses (tied input/output embeddings).
function parameterCount({ vocab, context, width, blocks }) {
  return vocab * width + context * width + blocks * (12 * width * width + 13 * width) + 2 * width;
}

const PRESET_SHAPES = {
  pocket: { label: "Pocket preset", context: 128, width: 128, blocks: 4, vocab: 4096 },
  starter: { label: "Starter preset", context: 256, width: 256, blocks: 6, vocab: 10000 },
  large: { label: "Large preset", context: 512, width: 512, blocks: 8, vocab: 10000 },
};

const FAMOUS = [
  { label: "GPT-2 small", year: "2019", params: 124e6 },
  { label: "GPT-2 XL", year: "2019", params: 1.5e9 },
  { label: "GPT-3", year: "2020", params: 175e9 },
];

const BYTES_PER_PARAM_TRAINING = 16; // fp32 weights + gradients + two AdamW moment buffers

function formatCount(value) {
  if (value >= 1e12) return `${(value / 1e12).toFixed(value >= 1e13 ? 0 : 1).replace(/\.0$/, "")}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(value >= 1e10 ? 0 : 1).replace(/\.0$/, "")}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 1e7 ? 0 : 1).replace(/\.0$/, "")}M`;
  return `${Math.round(value / 1e3)}k`;
}

function formatBytes(value) {
  const units = [[1e12, "TB"], [1e9, "GB"], [1e6, "MB"], [1e3, "KB"]];
  for (const [size, unit] of units) {
    if (value >= size) return `${(value / size).toFixed(value / size >= 10 ? 0 : 1).replace(/\.0$/, "")} ${unit}`;
  }
  return `${Math.round(value)} B`;
}

function scaleFigure(ctx, life) {
  const fig = figure({
    number: 1,
    title: "How small is small?",
    info: "log scale",
    tone: "light",
    wide: true,
    caption: "Parameter counts on a logarithmic axis: every grid step is ten times larger than the last. Switch to <em>Memory to train</em> for a rough lower bound on the memory needed to train each model with the AdamW optimizer (16 bytes per parameter, before activations and data).",
  });
  let mode = "params";
  const rows = [];
  for (const shape of Object.values(PRESET_SHAPES)) {
    rows.push({ label: shape.label, tag: "this app", params: parameterCount(shape), ours: true });
  }
  for (const model of FAMOUS) rows.push({ label: model.label, tag: model.year, params: model.params, ours: false });

  const list = h("div", { class: "bk-scale-rows" });
  const axis = h("div", { class: "bk-scale-axis" });
  const overlay = h("div", { class: "bk-scale-overlay", "aria-hidden": "true" });
  const plot = h("div", { class: "bk-scale-plot" }, overlay, list, axis);
  const domains = { params: [3e5, 1e12], memory: [4e6, 1e13] };
  const ticksFor = { params: [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12], memory: [1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13] };
  const position = (value) => {
    const [low, high] = domains[mode];
    return Math.max(0, Math.min(100, ((Math.log10(value) - Math.log10(low)) / (Math.log10(high) - Math.log10(low))) * 100));
  };

  let rowElements = [];
  function build() {
    clear(list);
    rowElements = rows.map((row) => {
      const fill = h("div", { class: "bk-scale-fill" });
      const value = h("span", { class: "bk-scale-value" });
      list.append(h("div", { class: ["bk-scale-row", row.ours ? "ours" : "", row.mine ? "mine" : ""] },
        h("div", { class: "bk-scale-label" }, h("span", { class: "bk-scale-name" }, row.label), h("span", { class: "bk-scale-tag" }, row.tag)),
        h("div", { class: "bk-scale-track" }, fill),
        value,
      ));
      return { row, fill, value };
    });
  }

  function render() {
    clear(axis);
    clear(overlay);
    for (const tick of ticksFor[mode]) {
      overlay.append(h("span", { class: "bk-scale-gridline", style: { left: `${position(tick)}%` } }));
      axis.append(h("span", { style: { left: `${position(tick)}%` } }, mode === "params" ? formatCount(tick) : formatBytes(tick)));
    }
    if (mode === "memory") {
      overlay.append(h("span", { class: "bk-scale-marker", style: { left: `${position(8e9)}%` } }, h("span", {}, "8 GB of RAM")));
    }
    for (const { row, fill, value } of rowElements) {
      const amount = mode === "params" ? row.params : row.params * BYTES_PER_PARAM_TRAINING;
      fill.style.width = `${Math.max(1.5, position(amount))}%`;
      value.textContent = mode === "params" ? formatCount(amount) : formatBytes(amount);
    }
  }

  const toggle = ctx.ui.segmented([
    { value: "params", label: "Parameters" },
    { value: "memory", label: "Memory to train" },
  ], mode, (value) => {
    mode = value;
    render();
  });
  fig.body.append(h("div", { class: "bk-toolbar" }, toggle), plot);
  build();
  render();

  // If the engine is running and a project exists, add the reader's own model to the comparison.
  ctx.api.get("/api/projects/current").then((project) => {
    if (life.disposed || !project || !project.estimates || !Number.isFinite(project.estimates.parameters)) return;
    rows.unshift({ label: project.name || "Your project", tag: "yours", params: project.estimates.parameters, ours: true, mine: true });
    build();
    render();
  }).catch(() => {});
  return fig.element;
}

export default {
  id: "introduction",
  title: "Introduction",
  render(root, ctx) {
    const life = lifecycle();
    const shell = chapterShell(root, "introduction",
      "Language models can feel like magic until you build one. This book explains how they work, one idea at a time, and the workbench beside it lets you train a real — if very small — model of your own, here on your own computer.");
    const { article } = shell;

    article.append(prose(`
      <h2>Two halves of one app</h2>
      <p>The app you are using has two halves that are meant to be used together. The chapters, listed under <strong>Foundations</strong> and <strong>How models work</strong> in the sidebar, explain the ideas. The pages under <strong>Build your model</strong> form a <strong>workbench</strong>: a sequence of steps that takes you from an empty project to a model you can chat with.</p>
      <p>You can read first and build later, or jump straight into the workbench and come back whenever something on screen needs explaining. Each workbench page has a short guide of its own in a reading pane on the right; the book icon in the toolbar shows or hides it.</p>
      <p>Most chapters contain figures you can play with. They are not pictures or videos but small programs running inside this window. In chapter 2 a predictive-text toy learns from a few pages of stories the moment the page opens; in chapter 3 a tokenizer builds its vocabulary while you drag a slider; in chapter 7 a tiny neural network trains in front of you. None of them needs an internet connection, and a few become more interesting once you have trained a model of your own, because they can then look inside it.</p>

      <h2>What a language model does</h2>
      <p>At heart, a language model does one modest thing: given some text, it estimates how likely each possible continuation is. Show it <em>“The cat curled up on the”</em> and it will give good odds to <em>sofa</em>, <em>rug</em> and <em>windowsill</em>, and almost none to <em>spreadsheet</em>. Producing text is nothing more than asking that question over and over: choose a likely next piece, append it, and ask again.</p>
      <p>The rest of this book unpacks what is hidden inside that one sentence. How is text chopped into pieces the model can handle (<em>tokenization</em>)? How do those pieces become numbers (<em>embeddings</em>)? What does the network in the middle — a <em>Transformer</em> — actually compute? And how do millions of internal settings get tuned simply by showing the model a lot of text?</p>
    `));
    article.append(callout("key", "<p>A language model is a next-piece predictor. Telling a story, answering a question and holding a conversation are all built from the same move, repeated: <strong>predict, pick, append</strong>.</p>"));

    article.append(prose(`
      <h2>Small on purpose</h2>
      <p>The best-known chatbots contain hundreds of billions of <strong>parameters</strong> — the adjustable numbers inside the network that training tunes — and learn from trillions of words on warehouses full of specialised chips. The models you will build here are thousands of times smaller: roughly one million to a few tens of millions of parameters, trained on a few million to a few hundred million words.</p>
      <p>Small does not mean fake. Your model uses the same kind of architecture, the same training objective and the same fine-tuning steps as its giant relatives; it is the same recipe cooked in a much smaller pot. What changes is what you can expect from the result. A small model trained on simple children's stories will learn spelling, grammar and the rhythm of a story, and it will cheerfully invent characters and plots. It will not know the capital of Peru, and it will lose the thread after a few paragraphs. Finding exactly where those limits lie is part of the fun.</p>
    `));
    article.append(scaleFigure(ctx, life));
    article.append(prose(`
      <p>Small also means fast. A model with a few million parameters can learn to write recognisable English in well under an hour on an ordinary processor, and it fits easily into the memory of a phone. You can watch every stage happen, stop it, change a setting and try again — the kind of experimenting that is out of reach when a single training run costs a fortune.</p>

      <h2>Your machine, your model</h2>
      <p>This edition trains with real neural-network code on whatever hardware you have. The processor (<strong>CPU</strong>) works everywhere, from a desktop tower or a Mac to a single-board computer or a phone running Linux. If there is a graphics processor (<strong>GPU</strong>), the app can use that too: through <strong>CUDA</strong> on NVIDIA cards, through <strong>Metal</strong> on Apple silicon Macs, or through <strong>Vulkan</strong> on Linux, which covers most integrated and mobile GPUs. The workbench can time a few training steps on each device and pick the faster one for you; you can always override the choice.</p>
      <p>Everything stays on your computer. Datasets are downloaded once and stored locally, training happens locally, and the finished model is a set of ordinary files you can keep, share or delete.</p>
    `));
    article.append(callout("linux", "<p>Training needs several times more memory than the model itself: besides the weights there are gradients, optimizer statistics and the batch being processed. On a phone or a small board, start with the smallest shape, keep other heavy apps closed and let the estimates on <em>Project setup</em> tell you how much memory a shape needs before you commit to it.</p>"));

    article.append(prose(`
      <h2>The journey ahead</h2>
      <p>Building a model happens in eight steps grouped into four phases. Try the map below: each step corresponds to one page of the workbench.</p>
    `));
    article.append(journeyFigure(ctx, life));
    article.append(prose(`
      <p>The phases mirror how large models are made in industry, just scaled down:</p>
      <ul>
        <li><strong>Set up.</strong> Decide the size and shape of the network and how text will be split into tokens.</li>
        <li><strong>Pre-train.</strong> Feed the model a large body of ordinary text and train it to predict the next token, again and again. Nearly everything the model knows about language comes from this phase. The result is a <em>base model</em>: a tireless autocomplete that continues whatever you give it.</li>
        <li><strong>Fine-tune.</strong> Show the base model example conversations so that it learns to take turns and reply in the voice of an assistant.</li>
        <li><strong>Prefer and chat.</strong> Compare pairs of the model's answers, tell it which you prefer, and let a short training run lean it towards the better kind. Then talk to it.</li>
      </ul>

      <h2>How to read this book</h2>
      <p>The chapters are short and build on each other, but you need no background beyond curiosity and a little patience with numbers. When maths appears it comes with small worked examples, and the figures let you check every formula by changing its inputs. If a passage feels heavy, skip to the next figure, play with it, and come back to the text afterwards.</p>
      <p>Important terms are set in <strong>bold</strong> where they are introduced. Callouts summarise ideas worth keeping, point out things to try, or give advice for running the workbench on modest hardware.</p>
    `));
    article.append(callout("note", "<p>This open-source edition of Language Model Builder, for Linux, Windows (through WSL) and macOS, is an unofficial, independently written edition inspired by Felix Rieseberg's <em>Language Model Builder</em> for macOS. It is not affiliated with or endorsed by the original author. The text, figures and code of this edition were written from scratch; the <em>Acknowledgements</em> page lists the open-source projects and datasets it builds on.</p>", "About this edition"));

    finishChapter(shell, ctx, { id: "what-is-a-language-model", title: "What is a language model?", blurb: "Next-token prediction, probabilities, sampling and temperature." });
    return () => life.dispose();
  },
};
