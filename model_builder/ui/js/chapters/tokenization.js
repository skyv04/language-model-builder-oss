// Chapter 3: Tokenization — characters, words and subwords; byte-level BPE; vocabulary trade-offs.
import { h, clear, tokenLabel } from "../dom.js";
import { icon } from "../icons.js";
import { chapterShell, finishChapter, prose, callout, figure, control, lifecycle, note, describeApiError, fmt } from "../book/kit.js";
import { trainBPE, utf8, bytesLabel } from "../book/bpe.js";

const TRAINING_TEXT = "The old lighthouse keeper lit the lamp every evening. Every evening the light turned, and the turning light swept the sea. Sailors passing the lighthouse flashed their lights back, thanking the keeper. When fog rolled in, the keeper rang the bell, and the ringing bell kept the ships away from the rocks. In the morning the keeper slept, the lamp rested, and the gulls kept watch over the harbour instead.";
const COMPARE_TEXT = "The lighthouse keeper's 3 cats weren't impressed by 1,024 fireflies. Ça va? 🙂";
const MAX_TRAIN_CHARS = 8000;
const MAX_MERGES = 400;
const START_MERGES = 16;
const MAX_CHIPS = 240;

const hue = (id) => Math.round((id * 137.508) % 360);

function lightChip(bpe, id, fresh) {
  const raw = id < 256;
  return h("span", { class: "bk-tok" },
    h("span", {
      class: ["token", "light", "bk-bpe-chip", raw ? "raw" : "", fresh ? "bk-fresh" : ""],
      style: raw ? null : { "--hue": hue(id) },
      title: raw ? `byte ${id}` : `token ${id}`,
    }, tokenLabel(bpe.label(id))),
    h("span", { class: "bk-tok-id" }, String(id)),
  );
}

function miniChip(label, extra = "") {
  return h("span", { class: ["token", "light", "bk-bpe-chip", "bk-mini", extra] }, tokenLabel(label));
}

// Figure 1 — train a BPE tokenizer on a short text and scrub through its merges.
function bpeFigure(ctx, life, shared) {
  const fig = figure({
    number: 1,
    title: "Byte-pair encoding, one merge at a time",
    tone: "light",
    wide: true,
    caption: "A byte-level BPE tokenizer trained in your browser on the paragraph shown. Each chip is one token, with its id underneath; grey chips are single bytes (ids 0–255), coloured chips are tokens created by merges, and the orange ring marks the newest merge. Drag the slider or press Play to replay training.",
  });
  let text = TRAINING_TEXT;
  let bpe = trainBPE(text, MAX_MERGES);
  let limit = Math.min(START_MERGES, bpe.merges.length);
  const player = life.timer();
  const retrain = life.timer();
  let playing = false;

  const slider = ctx.ui.slider({
    min: 0, max: bpe.merges.length, step: 1, value: limit, label: "Merges learned",
    onInput: (value) => { stop(); setLimit(value); },
  });
  const sliderControl = control("Merges learned", slider, { value: "" });
  const playButton = h("button", { class: "btn small primary", onclick: () => (playing ? stop() : play()) });
  const stepBack = h("button", { class: "btn small icon-only", title: "Undo one merge", "aria-label": "Undo one merge", onclick: () => { stop(); setLimit(limit - 1); } }, icon("minus", 14));
  const stepForward = h("button", { class: "btn small icon-only", title: "Learn one more merge", "aria-label": "Learn one more merge", onclick: () => { stop(); setLimit(limit + 1); } }, icon("plus", 14));
  const editButton = h("button", { class: "btn small ghost", onclick: () => toggleEditor() }, icon("pencil", 13), "Use your own text");

  const textarea = h("textarea", { class: "text-input bk-textarea", rows: 5, spellcheck: "false", "aria-label": "Training text" });
  textarea.value = text;
  const editorNote = h("div", { class: "bk-bpe-editor-note" });
  const editor = h("div", { class: "bk-bpe-editor", hidden: true },
    textarea,
    h("div", { class: "bk-toolbar" }, editorNote, h("span", { class: "spacer" }),
      h("button", { class: "btn small ghost", onclick: () => { textarea.value = TRAINING_TEXT; scheduleRetrain(0); } }, icon("reset", 13), "Restore the original")),
  );
  textarea.addEventListener("input", () => scheduleRetrain(450));

  const vocabStat = ctx.ui.stat("Vocabulary", "—");
  const tokenStat = ctx.ui.stat("Tokens in the text", "—");
  const ratioStat = ctx.ui.stat("Characters per token", "—");
  const stats = h("div", { class: "stats bk-stats bk-bpe-stats" }, vocabStat, tokenStat, ratioStat);
  const chips = h("div", { class: "bk-tokens bk-bpe-text" });
  const newest = h("div", { class: "bk-bpe-newest" });
  const mergeList = h("div", { class: "bk-bpe-merges" });

  function toggleEditor() {
    editor.hidden = !editor.hidden;
    editButton.classList.toggle("active", !editor.hidden);
    if (!editor.hidden) textarea.focus();
  }

  function scheduleRetrain(delay) {
    retrain.set(() => {
      stop();
      const source = textarea.value.slice(0, MAX_TRAIN_CHARS);
      text = source.trim() ? source : TRAINING_TEXT;
      const before = limit;
      bpe = trainBPE(text, MAX_MERGES);
      slider.max = bpe.merges.length;
      limit = Math.min(before, bpe.merges.length);
      editorNote.textContent = textarea.value.length > MAX_TRAIN_CHARS
        ? `Using the first ${fmt.int(MAX_TRAIN_CHARS)} characters.`
        : `${fmt.int(text.length)} characters · ${bpe.merges.length} merges possible (pairs seen at least twice).`;
      render();
    }, delay);
  }

  function setLimit(value) {
    limit = Math.max(0, Math.min(bpe.merges.length, value));
    render();
  }

  function play() {
    if (limit >= bpe.merges.length) limit = 0;
    playing = true;
    render();
    const tick = () => {
      if (limit >= bpe.merges.length) {
        stop();
        return;
      }
      limit += 1;
      render();
      player.set(tick, limit < 12 ? 420 : 190);
    };
    player.set(tick, 260);
  }

  function stop() {
    if (!playing) return;
    playing = false;
    player.clear();
    render();
  }

  function render() {
    const ids = bpe.encode(text, limit);
    const freshId = limit > 0 ? bpe.merges[limit - 1].id : -1;
    slider.setValue(limit);
    sliderControl.setValue(`${limit} of ${bpe.merges.length}`);
    playButton.replaceChildren(icon(playing ? "pause" : "play", 13), playing ? "Pause" : limit >= bpe.merges.length ? "Replay" : "Play");
    stepBack.disabled = limit <= 0;
    stepForward.disabled = limit >= bpe.merges.length;
    fig.setInfo(`vocabulary 256 + ${limit} = ${fmt.int(256 + limit)}`);
    vocabStat.set(fmt.int(bpe.vocabSize(limit)));
    tokenStat.set(fmt.int(ids.length));
    ratioStat.set((Array.from(text).length / Math.max(ids.length, 1)).toFixed(2));

    const shown = ids.slice(0, 900);
    const fragment = document.createDocumentFragment();
    for (const id of shown) fragment.append(lightChip(bpe, id, id === freshId));
    if (ids.length > shown.length) fragment.append(h("span", { class: "bk-cmp-more" }, `+${fmt.int(ids.length - shown.length)} more`));
    chips.replaceChildren(fragment);

    clear(newest);
    if (limit === 0) {
      newest.append(h("span", { class: "bk-bpe-newest-label" }, "No merges yet"), h("span", { class: "muted" }, "every byte of the text is its own token."));
    } else {
      const merge = bpe.merges[limit - 1];
      newest.append(
        h("span", { class: "bk-bpe-newest-label" }, `Merge ${limit}`),
        miniChip(bpe.label(merge.a)), h("span", { class: "bk-op" }, "+"), miniChip(bpe.label(merge.b)),
        h("span", { class: "bk-op" }, "→"), miniChip(bpe.label(merge.id), "bk-fresh"),
        h("span", { class: "muted" }, `new id ${merge.id}, a pair seen ${merge.count} times`),
      );
    }

    clear(mergeList);
    const first = Math.max(0, limit - 7);
    if (limit === 0) mergeList.append(h("div", { class: "bk-merge-empty" }, "Merges will be listed here, newest first."));
    for (let rank = limit - 1; rank >= first; rank -= 1) {
      const merge = bpe.merges[rank];
      mergeList.append(h("button", {
        class: ["bk-merge", rank === limit - 1 ? "current" : ""],
        title: `Rewind to merge ${rank + 1}`,
        onclick: () => { stop(); setLimit(rank + 1); },
      },
      h("span", { class: "bk-merge-rank" }, `#${rank + 1}`),
      h("span", { class: "bk-merge-eq" }, miniChip(bpe.label(merge.a)), "+", miniChip(bpe.label(merge.b)), "→", miniChip(bpe.label(merge.id))),
      h("span", { class: "bk-merge-count" }, `×${merge.count}`)));
    }
    shared.bpe = bpe;
    shared.limit = limit;
    shared.notify();
  }

  fig.body.append(
    h("div", { class: "bk-bpe-top" },
      sliderControl,
      h("div", { class: "bk-bpe-buttons" }, stepBack, stepForward, playButton, editButton),
    ),
    editor,
    stats,
    newest,
    h("div", { class: "bk-grid-2 bk-bpe-main" },
      h("div", {}, h("span", { class: "bk-label bk-section-label" }, "The text, as tokens"), chips),
      h("div", {}, h("span", { class: "bk-label bk-section-label" }, "Latest merges"), mergeList),
    ),
  );
  editorNote.textContent = `${fmt.int(text.length)} characters · ${bpe.merges.length} merges possible (pairs seen at least twice).`;
  render();
  return fig.element;
}

// Figure 2 — the same sentence through several tokenizers, two of them from the engine.
function compareFigure(ctx, life, shared) {
  const fig = figure({
    number: 2,
    title: "One sentence, six tokenizers",
    info: "fewer tokens = more text per context",
    wide: true,
    caption: "Characters, raw bytes and whole words are computed here in the page, as is the mini BPE from Figure 1 (at its current number of merges). The last two rows ask the training engine to run your project's own tokenizer and GPT-2's. Edit the sentence to try other languages, numbers or code.",
  });
  const input = h("input", { class: "text-input", type: "text", spellcheck: "false", autocomplete: "off", "aria-label": "Sentence to tokenize" });
  input.value = COMPARE_TEXT;
  const debounce = life.timer();
  let request = 0;

  function makeRow(name, meta, { api = false } = {}) {
    const metaElement = h("span", { class: "bk-cmp-meta" }, meta);
    const count = h("span", { class: "bk-cmp-count" });
    const bar = h("span", { class: "bk-cmp-fill" });
    const body = h("div", { class: "bk-tokens bk-cmp-chips" });
    const element = h("div", { class: ["bk-cmp-row", api ? "api" : ""] },
      h("div", { class: "bk-cmp-head" }, h("span", { class: "bk-cmp-name" }, name), metaElement, h("span", { class: "spacer" }), count),
      h("div", { class: "bk-cmp-meter" }, bar),
      body,
    );
    return { element, metaElement, count, bar, body, value: null };
  }

  const rows = {
    chars: makeRow("Characters", "one token per Unicode character; ids are code points"),
    bytes: makeRow("UTF-8 bytes", "256 possible tokens; accents and emoji take several"),
    words: makeRow("Whole words", "a vocabulary would need every word ever written"),
    mini: makeRow("Mini BPE from Figure 1", ""),
    project: makeRow("Your project's tokenizer", "from the training engine", { api: true }),
    gpt2: makeRow("GPT-2", "50,257 tokens, trained on web text", { api: true }),
  };

  function fill(row, tokens, { ids = true } = {}) {
    row.value = tokens.length;
    row.count.replaceChildren(h("b", {}, fmt.int(tokens.length)), tokens.length === 1 ? " token" : " tokens");
    const fragment = document.createDocumentFragment();
    tokens.slice(0, MAX_CHIPS).forEach((token, index) => {
      const chip = h("span", { class: "token", style: { "--p": index % 2 ? 0.42 : 0.8 } }, tokenLabel(token.text));
      fragment.append(ids ? h("span", { class: "bk-tok" }, chip, h("span", { class: "bk-tok-id" }, String(token.id))) : chip);
    });
    if (tokens.length > MAX_CHIPS) fragment.append(h("span", { class: "bk-cmp-more" }, `+${fmt.int(tokens.length - MAX_CHIPS)} more`));
    row.body.replaceChildren(fragment);
  }

  function message(row, content) {
    row.value = null;
    row.count.replaceChildren();
    row.body.replaceChildren(content);
  }

  function scale() {
    const counts = Object.values(rows).map((row) => row.value).filter((value) => Number.isFinite(value));
    const max = Math.max(1, ...counts);
    for (const row of Object.values(rows)) {
      row.bar.style.width = Number.isFinite(row.value) ? `${(row.value / max) * 100}%` : "0%";
    }
  }

  function mini() {
    const { bpe, limit } = shared;
    if (!bpe) return;
    rows.mini.metaElement.textContent = `${fmt.int(bpe.vocabSize(limit))} tokens, trained on one paragraph`;
    fill(rows.mini, bpe.encode(input.value, limit).map((id) => ({ id, text: bpe.label(id) })));
  }

  function local() {
    const text = input.value;
    fill(rows.chars, Array.from(text, (char) => ({ id: char.codePointAt(0), text: char })));
    fill(rows.bytes, Array.from(utf8(text), (byte) => ({ id: byte, text: bytesLabel(Uint8Array.of(byte)) })));
    fill(rows.words, (text.match(/[\p{L}\p{M}'’]+|\p{N}+|[^\s\p{L}\p{N}]/gu) || []).map((word) => ({ text: word })), { ids: false });
    mini();
    scale();
  }

  function explain(kind, error, which) {
    if (kind === "not-ready" && which === "project") {
      return note(h("span", {}, "Train a tokenizer in Project setup to see yours here. It is built from your pre-training data at the start of the first run."), {
        action: h("button", { class: "btn small", onclick: () => ctx.navigate("project-setup") }, "Project setup", icon("chevron-right", 12)),
      });
    }
    const { text } = describeApiError(error);
    if (kind === "unavailable") return note("This row needs the training engine, which isn't available in this window.", { tone: "info" });
    if (kind === "offline") return note(`${text} Start the app normally to compare with ${which === "gpt2" ? "GPT-2" : "your tokenizer"}.`, { tone: "warn" });
    return note(text, { tone: "warn" });
  }

  async function remote() {
    const text = input.value;
    const ticket = ++request;
    for (const which of ["project", "gpt2"]) {
      const row = rows[which];
      if (!text.trim()) {
        fill(row, []);
        continue;
      }
      if (row.value === null && !row.body.querySelector(".token")) message(row, h("span", { class: "bk-cmp-wait" }, h("span", { class: "spinner" }), "Asking the engine…"));
    }
    await Promise.all(["project", "gpt2"].map(async (which) => {
      const row = rows[which];
      if (!text.trim()) return;
      try {
        const result = await ctx.api.post("/api/tokenize", { text, tokenizer: which });
        if (life.disposed || ticket !== request) return;
        const info = result.tokenizer || {};
        if (which === "project") row.metaElement.textContent = [info.label, info.vocab_size ? `${fmt.int(info.vocab_size)} tokens` : ""].filter(Boolean).join(" · ") || "from the training engine";
        else if (info.vocab_size) row.metaElement.textContent = `${fmt.int(info.vocab_size)} tokens (GPT-2's 50,257 plus chat markers)`;
        fill(row, Array.isArray(result.tokens) ? result.tokens : []);
      } catch (error) {
        if (life.disposed || ticket !== request) return;
        const { kind } = describeApiError(error);
        message(row, explain(kind, error, which));
      }
    }));
    if (!life.disposed && ticket === request) scale();
  }

  input.addEventListener("input", () => {
    local();
    debounce.set(remote, 350);
  });
  shared.subscribe(() => {
    mini();
    scale();
  });

  fig.body.append(
    h("div", { class: "bk-input-row" }, h("span", { class: "panel-label" }, "Text"), input,
      h("button", { class: "btn small ghost", title: "Restore the example", onclick: () => { input.value = COMPARE_TEXT; local(); debounce.set(remote, 0); } }, icon("reset", 13))),
    h("div", { class: "bk-cmp" }, ...Object.values(rows).map((row) => row.element)),
  );
  local();
  remote();
  return fig.element;
}

export default {
  id: "tokenization",
  title: "Tokenization",
  render(root, ctx) {
    const life = lifecycle();
    const listeners = new Set();
    const shared = {
      bpe: null,
      limit: 0,
      subscribe(fn) { listeners.add(fn); },
      notify() { for (const fn of listeners) fn(); },
    };
    life.add(() => listeners.clear());

    const shell = chapterShell(root, "tokenization",
      "Language models don't read letters or words. They read numbers. Before any learning can happen, text has to be cut into pieces and each piece swapped for an id from a fixed list. How that cutting is done matters more than you might expect.");
    const { article } = shell;

    article.append(prose(`
      <h2>Text in, numbers out</h2>
      <p>A neural network is a large pile of arithmetic, and arithmetic needs numbers: there is no way to multiply the letter <em>q</em> by anything. So every language model comes with a <strong>tokenizer</strong>, a fixed recipe that turns text into a sequence of integers and turns integers back into text. The pieces it knows form its <strong>vocabulary</strong>, and each piece's position in that list is its <strong>token id</strong>.</p>
      <p>Suppose the vocabulary has 10,000 entries. Then everything the model ever reads is a list of numbers between 0 and 9,999, and everything it predicts is a set of 10,000 probabilities, one per entry — exactly the bar charts of the previous chapter, only much longer.</p>
      <p>The tokenizer is built before the model is trained, by a simple counting procedure, and then frozen. From then on the two are a pair: the model learns something specific about token 4,711, and if you swapped in a tokenizer where 4,711 meant something else, everything it had learned would be scrambled. That is why the workbench locks the tokenizer as soon as your project has its first checkpoint.</p>

      <h2>Three ways to cut text</h2>
      <p><strong>Characters.</strong> The simplest choice is one token per character. The vocabulary is tiny — letters, digits and punctuation for English fit in about a hundred — and no input can ever be unrepresentable. The price is length: a 1,000-word essay becomes roughly 6,000 tokens, every prediction carries little information, and the model has to spend effort learning to spell before it can learn anything about meaning.</p>
      <p><strong>Words.</strong> At the other extreme, each word is a token. Sequences are short and every token means something, but the vocabulary explodes. English alone has hundreds of thousands of word forms, before you count names, typos, numbers, code and other languages. Any word missing from the list becomes a generic “unknown” token, and the model cannot even tell that <em>unhappiest</em> has something to do with <em>happy</em>.</p>
      <p><strong>Subwords.</strong> Practically every modern model uses a compromise. Common words get a token of their own, while rarer words are assembled from a few frequent fragments: <em>unhappiest</em> might come out as <span class="bk-chip-inline">un</span><span class="bk-chip-inline">happ</span><span class="bk-chip-inline">iest</span>. The vocabulary stays at a few thousand to a few hundred thousand entries, sequences stay reasonably short, and nothing is ever unknown: in the worst case a strange word falls apart into single characters.</p>

      <h2>Byte-pair encoding</h2>
      <p>The most popular way to choose subword pieces is <strong>byte-pair encoding</strong>, or BPE. It started life in 1994 as a data-compression trick, was adapted for machine translation in 2016, and in its byte-level form became the tokenizer of GPT-2 and many models since. The recipe is short enough to do by hand:</p>
      <ol>
        <li><strong>Start from bytes.</strong> Computers store text as UTF-8 bytes, and there are only 256 possible byte values. Those 256 values are the starting vocabulary, so any text in any language, emoji included, can be tokenized.</li>
        <li><strong>Split into chunks.</strong> Cut the training text into word-sized chunks at spaces and punctuation. A space stays attached to the word that follows it, so “ the” and “the” are different chunks. Merges never cross chunk boundaries.</li>
        <li><strong>Merge the most frequent pair.</strong> Count every pair of neighbouring tokens across all chunks. The most common pair becomes a new token with the next free id, and every occurrence is replaced by it.</li>
        <li><strong>Repeat</strong> until the vocabulary reaches the size you asked for.</li>
      </ol>
      <p>To tokenize new text later, the tokenizer replays its merges in the order it learned them. Frequent words collapse into single tokens, common fragments such as <em>ing</em> or <em>ed</em> become tokens, and anything rare stays in smaller pieces. Below, a tokenizer learns from one short paragraph, right in this page.</p>
    `));
    article.append(bpeFigure(ctx, life, shared));
    article.append(callout("try", "<p>Drag the slider to zero: every byte is its own token, so the text costs exactly one token per character. Then press <em>Play</em>. Within three merges the most common English word, <span class=\"bk-chip-inline\">·the</span>, has a token of its own, and by the end <span class=\"bk-chip-inline\">·lighthouse</span> and <span class=\"bk-chip-inline\">·keeper</span> are single tokens because this little text keeps repeating them. Paste in a paragraph of your own and watch different merges win.</p>"));

    article.append(prose(`
      <h2>Reading the chips</h2>
      <p>Token chips take a moment to get used to. A few things to look for, in the figure above and the one below:</p>
      <ul>
        <li><strong>Spaces belong to words.</strong> The dot at the start of <span class="bk-chip-inline">·keeper</span> is a space. <span class="bk-chip-inline">·keeper</span> and <span class="bk-chip-inline">keeper</span> are different tokens with unrelated ids, and so are <span class="bk-chip-inline">The</span> and <span class="bk-chip-inline">·the</span>. The model has to learn that they mean the same thing.</li>
        <li><strong>Tokens follow frequency, not grammar.</strong> A pair is merged because it is common, not because it is a syllable or a meaningful part of a word. Some tokens line up with word parts; many do not.</li>
        <li><strong>Numbers are chopped up.</strong> Before any merging, GPT-2-style tokenizers cut <em>1,024</em> into <span class="bk-chip-inline">·1</span><span class="bk-chip-inline">,</span><span class="bk-chip-inline">024</span>, and long numbers end up in chunks that have nothing to do with place value. A model that does arithmetic has to learn it over these odd pieces.</li>
        <li><strong>Unfamiliar text costs more.</strong> A tokenizer trained mostly on English text splits French, Thai or program code into many small pieces, and an emoji can take four byte tokens. More tokens means more computation for the same message and less of it fitting in the model's context — one reason models trained mostly on English serve other languages less well.</li>
      </ul>
    `));
    article.append(compareFigure(ctx, life, shared));

    article.append(prose(`
      <h2>How big should the vocabulary be?</h2>
      <p>A larger vocabulary means fewer tokens for the same text, so more text fits in the model's context window and each training step covers more ground. But every vocabulary entry costs something. The model keeps one row of numbers per token in its input table (the subject of the next chapter), and at every position its output layer has to score every entry. Rare tokens are also seen rarely during training, so their rows stay poorly trained.</p>
      <div class="bk-eq">embedding parameters = vocabulary size × model width<span class="bk-eq-note">4,096 tokens × 128 numbers per token = 524,288 parameters</span></div>
      <p>For a large model this table is small change. For the small models you will build here it can dominate. The workbench's Pocket preset has a width of 128 and about 1.3 million parameters in total; with a 4,096-token BPE, its embedding table is roughly 40% of the whole model. Give the same model GPT-2's vocabulary and the table grows to 6.4 million parameters, almost nine-tenths of everything, spent on a lookup table rather than on thinking. That is why Project setup offers compact BPE tokenizers trained on your own data (4K, 10K or 16K tokens) next to GPT-2's.</p>
    `));
    article.append(callout("linux", "<p>In this edition you choose the tokenizer in <strong>Project setup</strong>. A BPE tokenizer is trained automatically on your pre-training data in the first phase of your first pre-training run, so there is nothing extra to do. GPT-2's tokenizer needs no training at all, which is handy if you want to compare your model with published ones, but on a phone or an older laptop the smaller vocabularies train noticeably faster.</p>", "In the workbench"));

    article.append(prose(`
      <h2>Special tokens</h2>
      <p>Besides ordinary pieces of text, a tokenizer reserves a few ids for <strong>special tokens</strong>: markers that never come out of normal text. One typically marks the end of a document, so that during pre-training the model learns where one text stops and an unrelated one begins. Chat models add markers that frame a conversation. In this app they are <span class="bk-chip-inline orange">&lt;|system|&gt;</span>, <span class="bk-chip-inline orange">&lt;|user|&gt;</span>, <span class="bk-chip-inline orange">&lt;|assistant|&gt;</span> and <span class="bk-chip-inline orange">&lt;|end|&gt;</span>, and you will see them at work in the last chapter of this part. Each is a single dedicated id, so the model can recognise it without ambiguity; a careful chat system also makes sure that someone typing the same characters into a message does not produce the real thing. The workbench adds its four chat tokens to GPT-2's vocabulary, which is why it reports 50,261 tokens rather than 50,257.</p>

      <h2>Why tokens explain some odd behaviour</h2>
      <p>Since a model sees token ids rather than letters, some tasks that are trivial for people are oddly hard for it. How many letters are in a word, what it looks like spelled backwards, whether two words rhyme: to the model, <em>strawberry</em> may be two or three opaque ids, and the letters inside them are facts it must have picked up indirectly, not something it can look at. The same goes for digit-by-digit arithmetic on numbers that were cut into arbitrary chunks. When a capable model stumbles over a question a child could answer, the tokenizer is often part of the story.</p>
    `));
    article.append(callout("key", "<p>A model never sees text. It sees a sequence of token ids produced by a fixed tokenizer, and everything it learns — spelling, grammar, facts, style — it learns in terms of those ids.</p>"));

    finishChapter(shell, ctx, { id: "embeddings", title: "Embeddings", blurb: "How token ids become vectors, and why similar words end up close together." });
    return () => life.dispose();
  },
};
