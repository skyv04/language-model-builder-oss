// Chapter 2: What is a language model? — next-token prediction, sampling and temperature.
import { h, clear } from "../dom.js";
import { icon } from "../icons.js";
import { chapterShell, finishChapter, prose, callout, figure, control, lifecycle, sampleIndex, percent } from "../book/kit.js";
import { STORIES } from "../book/corpus.js";
import { NGramModel, corpusTokens, wordTokens, displayWord, reshape, SENTENCE_END } from "../book/ngram.js";

const DEFAULT_PROMPT = "Once upon a time, a little";
const ENDINGS = new Set([".", "!", "?"]);
const PUNCTUATION = /^[.,!?;:]$/;

// The history always begins with an artificial "." that marks the start of the text.
function contextLabel(history, length) {
  const from = history.length - length;
  return history.slice(from).map((word, index) => (from + index === 0 ? "⟨start⟩" : displayWord(word))).join(" ");
}

function playground(ctx, life) {
  const model = new NGramModel(corpusTokens(STORIES), 3);
  const fig = figure({
    number: 1,
    title: "Predict the next word",
    info: `${model.vocab.length}-word vocabulary`,
    wide: true,
    caption: "A counting model trained, when this page opened, on about 1,200 words of short stories written for this book. Bars show its probabilities for the next word after the temperature is applied. Click a bar to add that word, or let the model sample. Click a generated word to rewind to it.",
  });
  let lookBack = 2;
  let temperature = 1;
  let continuation = [];
  let writing = 0;
  const writer = life.timer();

  const input = h("input", { class: "text-input", type: "text", value: DEFAULT_PROMPT, spellcheck: "false", autocomplete: "off", "aria-label": "Prompt" });
  const story = h("div", { class: "bk-lm-story", "aria-live": "polite" });
  const bars = h("div", { class: "bk-bars bk-lm-bars" });
  const contextNote = h("div", { class: "bk-lm-context" });
  const writeButton = h("button", { class: "btn small", onclick: () => toggleWriting() });
  const undoButton = h("button", { class: "btn small", onclick: () => { stopWriting(); continuation.pop(); update(); } }, icon("corner-down-left", 13), "Undo");

  function history() {
    return [SENTENCE_END, ...wordTokens(input.value), ...continuation.map((item) => item.word)];
  }

  function renderStory() {
    clear(story);
    const prompt = input.value.trim();
    story.append(h("span", { class: "bk-lm-prompt" }, prompt || "(empty prompt)"));
    const promptTokens = wordTokens(prompt);
    let previous = promptTokens.length ? promptTokens[promptTokens.length - 1] : SENTENCE_END;
    continuation.forEach((item, index) => {
      let text = displayWord(item.word);
      const speech = (item.word === "said" || item.word === "asked") && (previous === "!" || previous === "?");
      if (ENDINGS.has(previous) && !PUNCTUATION.test(item.word) && !speech) text = text[0].toUpperCase() + text.slice(1);
      if (!PUNCTUATION.test(item.word)) story.append(" ");
      story.append(h("span", {
        class: "token bk-lm-word",
        style: { "--p": Math.sqrt(Math.max(item.prob, 0)) },
        title: `${percent(item.prob)} likely when chosen · click to rewind to here`,
        onclick: () => {
          stopWriting();
          continuation = continuation.slice(0, index);
          update();
        },
      }, text));
      previous = item.word;
    });
    story.append(h("span", { class: "bk-lm-caret" }));
  }

  function renderBars(probs) {
    const { items } = NGramModel.top(model, probs, 8, temperature);
    clear(bars);
    items.forEach((item, rank) => {
      const label = item.word === SENTENCE_END ? ". (end)" : displayWord(item.word);
      bars.append(h("button", {
        class: ["bk-bar", rank === 0 ? "top" : ""],
        title: `Add “${label}”`,
        onclick: () => {
          stopWriting();
          continuation.push({ word: item.word, prob: item.prob });
          update();
        },
      },
      h("span", { class: "bk-bar-text" }, label),
      h("span", { class: "bk-bar-track" }, h("span", { class: "bk-bar-fill", style: { width: `${Math.max(0.6, item.prob * 100)}%` } })),
      h("span", { class: "bk-bar-pct" }, percent(item.prob))));
    });
  }

  function renderContext(hist, used, tried) {
    clear(contextNote);
    const unknown = wordTokens(input.value).concat(continuation.map((item) => item.word)).filter((word) => !model.has(word));
    let text;
    if (used === 0) {
      text = h("span", {}, "None of the recent words appear in the stories, so the model can only fall back on how common each word is overall.");
    } else if (used < tried) {
      const seen = model.count(hist.slice(hist.length - used));
      text = h("span", {}, "“", h("b", {}, contextLabel(hist, tried)), "” never appears in the stories, so the model backs off to “", h("b", {}, contextLabel(hist, used)), `” (seen ${seen} time${seen === 1 ? "" : "s"}).`);
    } else {
      const seen = model.count(hist.slice(hist.length - used));
      text = h("span", {}, `Looking at the last ${used === 1 ? "word" : `${used} words`}: “`, h("b", {}, contextLabel(hist, used)), `” — seen ${seen} time${seen === 1 ? "" : "s"} in the stories.`);
    }
    contextNote.append(icon(used < tried ? "alert" : "eye", 14), text);
    if (unknown.length) {
      contextNote.append(h("span", { class: "bk-lm-unknown" }, ` Never seen: ${[...new Set(unknown)].slice(0, 3).map((word) => `“${word}”`).join(", ")}.`));
    }
  }

  function update() {
    const hist = history();
    const { probs, used, tried } = model.distribution(hist, lookBack);
    renderStory();
    renderBars(probs);
    renderContext(hist, used, tried);
    undoButton.disabled = continuation.length === 0;
    writeButton.replaceChildren(icon(writing ? "stop" : "play", 13), writing ? "Stop" : "Write 12 words");
    return probs;
  }

  function sampleOne() {
    const { probs } = model.distribution(history(), lookBack);
    const reshaped = reshape(probs, temperature);
    const index = sampleIndex(reshaped);
    continuation.push({ word: model.vocab[index], prob: reshaped[index] });
  }

  function stopWriting() {
    writing = 0;
    writer.clear();
  }

  function writeStep() {
    if (!writing) return;
    sampleOne();
    writing -= 1;
    update();
    if (writing > 0) writer.set(writeStep, 170);
  }

  function toggleWriting() {
    if (writing) {
      stopWriting();
      update();
      return;
    }
    writing = 12;
    update();
    writer.set(writeStep, 60);
  }

  input.addEventListener("input", () => {
    stopWriting();
    continuation = [];
    update();
  });

  const tempControl = control("Temperature", ctx.ui.slider({
    min: 0.2, max: 2, step: 0.05, value: temperature,
    label: "Temperature",
    onInput: (value) => {
      temperature = value;
      tempControl.setValue(value.toFixed(2));
      update();
    },
  }), { value: temperature.toFixed(2), hint: "Low: play it safe. High: take chances." });

  const lookControl = control("Look-back", ctx.ui.segmented([
    { value: 1, label: "1 word" },
    { value: 2, label: "2 words" },
    { value: 3, label: "3 words" },
  ], lookBack, (value) => {
    lookBack = value;
    update();
  }), { hint: "How many previous words the model may use." });

  const controls = h("div", { class: "bk-lm-controls" },
    lookControl,
    tempControl,
    h("div", { class: "bk-lm-buttons" },
      h("button", { class: "btn small primary", onclick: () => { stopWriting(); sampleOne(); update(); } }, icon("dice", 13), "Sample one"),
      writeButton,
      undoButton,
      h("button", { class: "btn small ghost", title: "Reset", onclick: () => { stopWriting(); input.value = DEFAULT_PROMPT; continuation = []; update(); } }, icon("reset", 13), "Reset"),
    ),
  );

  fig.body.append(
    h("div", { class: "bk-input-row" }, h("span", { class: "panel-label" }, "Prompt"), input),
    story,
    h("div", { class: "bk-grid-2 wide-left bk-lm-main" },
      h("div", {}, h("span", { class: "panel-label bk-section-label" }, "Next word"), bars),
      controls,
    ),
    contextNote,
  );
  update();
  return fig.element;
}

const LENS_PROMPT = "After a long day at the beach, the children were very";
const LENS_WORDS = [
  ["tired", 0.34], ["happy", 0.22], ["sleepy", 0.12], ["hungry", 0.1], ["sandy", 0.07],
  ["quiet", 0.05], ["sunburnt", 0.04], ["excited", 0.035], ["purple", 0.015], ["invisible", 0.01],
];

function shape(temperature, topK, topP) {
  const base = LENS_WORDS.map(([, prob]) => prob);
  const tempered = Array.from(reshape(Float64Array.from(base), temperature));
  const order = tempered.map((_, index) => index).sort((a, b) => tempered[b] - tempered[a]);
  const cut = new Array(base.length).fill(null);
  order.forEach((index, rank) => {
    if (rank >= topK) cut[index] = "top-k";
  });
  const keptTotal = order.filter((index) => !cut[index]).reduce((sum, index) => sum + tempered[index], 0);
  let cumulative = 0;
  for (const index of order) {
    if (cut[index]) continue;
    if (cumulative >= topP - 1e-9) cut[index] = "top-p";
    else cumulative += tempered[index] / keptTotal;
  }
  const finalTotal = tempered.reduce((sum, value, index) => sum + (cut[index] ? 0 : value), 0);
  const final = tempered.map((value, index) => (cut[index] ? 0 : value / finalTotal));
  return { tempered, final, cut };
}

function lens(ctx, life) {
  const fig = figure({
    number: 2,
    title: "Shaping the odds",
    info: "temperature → top-k → top-p",
    caption: "Made-up probabilities for the word after the prompt, reshaped by temperature and then filtered by top-k and top-p (in that order, as most software does). Faded bars are words that can no longer be picked. Draw samples to see how often each word actually comes up.",
  });
  let temperature = 1;
  let topK = LENS_WORDS.length;
  let topP = 1;
  let counts = new Array(LENS_WORDS.length).fill(0);
  let draws = [];
  let pending = 0;
  const drawTimer = life.timer();

  const rows = h("div", { class: "bk-bars bk-lens-bars" });
  const drawsRow = h("div", { class: "bk-lens-draws" });
  const stats = h("div", { class: "stats bk-lens-stats" });
  const topStat = ctx.ui.stat("Top word", "—");
  const aliveStat = ctx.ui.stat("Words in play", "—");
  const choiceStat = ctx.ui.stat("Like choosing among", "—");
  stats.append(topStat, aliveStat, choiceStat);

  function render() {
    const { tempered, final, cut } = shape(temperature, topK, topP);
    clear(rows);
    LENS_WORDS.forEach(([word], index) => {
      const removed = Boolean(cut[index]);
      rows.append(h("div", { class: ["bk-bar", "bk-lens-row", removed ? "cut" : ""] },
        h("span", { class: "bk-bar-text" }, word),
        h("span", { class: "bk-bar-track" },
          h("span", { class: "bk-bar-ghost", style: { width: `${tempered[index] * 100}%` } }),
          h("span", { class: "bk-bar-fill", style: { width: `${final[index] * 100}%` } }),
        ),
        h("span", { class: "bk-bar-pct" }, removed ? cut[index] : percent(final[index])),
        h("span", { class: "bk-lens-count" }, counts[index] ? `×${counts[index]}` : ""),
      ));
    });
    const alive = final.filter((value) => value > 0).length;
    const entropy = -final.reduce((sum, value) => sum + (value > 0 ? value * Math.log2(value) : 0), 0);
    const best = final.indexOf(Math.max(...final));
    topStat.set(`${LENS_WORDS[best][0]} ${percent(final[best], 0)}`);
    aliveStat.set(`${alive} of ${LENS_WORDS.length}`);
    choiceStat.set(`≈${(2 ** entropy).toFixed(1)} words`);
    clear(drawsRow);
    if (!draws.length) drawsRow.append(h("span", { class: "panel-muted" }, "Samples appear here."));
    for (const index of draws) drawsRow.append(h("span", { class: "token bk-lens-draw", style: { "--p": Math.sqrt(final[index] || 0.01) } }, LENS_WORDS[index][0]));
  }

  function resetDraws() {
    pending = 0;
    drawTimer.clear();
    counts = counts.map(() => 0);
    draws = [];
  }

  function drawStep() {
    if (pending <= 0) return;
    const { final } = shape(temperature, topK, topP);
    const index = sampleIndex(final);
    counts[index] += 1;
    draws.push(index);
    if (draws.length > 40) draws.shift();
    pending -= 1;
    render();
    if (pending > 0) drawTimer.set(drawStep, 55);
  }

  const tempControl = control("Temperature", ctx.ui.slider({
    min: 0.1, max: 2.5, step: 0.05, value: temperature, label: "Temperature",
    onInput: (value) => { temperature = value; tempControl.setValue(value.toFixed(2)); resetDraws(); render(); },
  }), { value: "1.00" });
  const kControl = control("Top-k", ctx.ui.slider({
    min: 1, max: LENS_WORDS.length, step: 1, value: topK, label: "Top-k",
    onInput: (value) => { topK = value; kControl.setValue(value >= LENS_WORDS.length ? "off" : String(value)); resetDraws(); render(); },
  }), { value: "off" });
  const pControl = control("Top-p", ctx.ui.slider({
    min: 0.05, max: 1, step: 0.05, value: topP, label: "Top-p",
    onInput: (value) => { topP = value; pControl.setValue(value >= 1 ? "off" : value.toFixed(2)); resetDraws(); render(); },
  }), { value: "off" });
  const sliders = [tempControl, kControl, pControl].map((element) => element.querySelector("input"));

  fig.body.append(
    h("p", { class: "bk-lens-prompt" }, LENS_PROMPT, h("span", { class: "bk-lens-gap" }, "…")),
    h("div", { class: "bk-controls" }, tempControl, kControl, pControl),
    h("div", { class: "bk-divider" }),
    rows,
    h("div", { class: "bk-divider" }),
    h("div", { class: "bk-toolbar" },
      stats,
      h("span", { class: "spacer" }),
      h("button", { class: "btn small primary", onclick: () => { pending += 20; if (!drawTimer.pending) drawStep(); } }, icon("dice", 13), "Draw 20"),
      h("button", {
        class: "btn small ghost",
        onclick: () => {
          temperature = 1; topK = LENS_WORDS.length; topP = 1;
          sliders[0].setValue(1); sliders[1].setValue(topK); sliders[2].setValue(1);
          tempControl.setValue("1.00"); kControl.setValue("off"); pControl.setValue("off");
          resetDraws();
          render();
        },
      }, icon("reset", 13), "Reset"),
    ),
    drawsRow,
  );
  render();
  return fig.element;
}

export default {
  id: "what-is-a-language-model",
  title: "What is a language model?",
  render(root, ctx) {
    const life = lifecycle();
    const shell = chapterShell(root, "what-is-a-language-model",
      "A language model is a machine for guessing what comes next. That sounds like a modest job, but a good enough guesser can write a story, translate a sentence or answer a question. This chapter shows how the guessing works, and how a stream of guesses becomes text.");
    const { article } = shell;

    article.append(prose(`
      <h2>The guessing game</h2>
      <p>Read this and notice what your mind does: <em>“She opened her umbrella because it had started to …”</em> Before you reach the gap you have already filled it. <em>Rain</em> is the obvious candidate; <em>drizzle</em> and <em>pour</em> are possible; <em>sing</em> would be a surprise. You did not consult a rule book. You have read and heard so much language that likely continuations simply occur to you, together with a feeling for how likely each one is.</p>
      <p>A language model plays exactly this game, but explicitly and with numbers. Given the text so far — the <strong>context</strong> — it produces a score for every possible next piece of text and turns those scores into <strong>probabilities</strong>: numbers between 0 and 1 that add up to 1 across all the options. A good model puts most of the probability on continuations that make sense.</p>
      <div class="bk-eq">P( <em>rain</em> | “…because it had started to” ) = 0.81<span class="bk-eq-note">read: the probability that the next word is “rain”, given the text so far</span></div>
      <p>The pieces a model predicts are called <strong>tokens</strong>. In this chapter a token is simply a word or a punctuation mark; the next chapter explains why real models use slightly different pieces.</p>

      <h2>Counting: the simplest language model</h2>
      <p>How could a computer learn such probabilities? The oldest trick is to count. Take a large body of text, and every time the words <em>a little</em> appear, note which word comes next. If <em>a little</em> is followed by <em>fox</em> twelve times, <em>bird</em> eight times and <em>girl</em> five times, then after <em>a little</em> the model predicts <em>fox</em> with probability 12/25, <em>bird</em> with 8/25, and so on.</p>
      <p>A model that looks back over a fixed number of words like this is called an <strong>n-gram model</strong>: a <em>bigram</em> model conditions on one previous word, a <em>trigram</em> model on two. Longer contexts make sharper, more sensible predictions — <em>on the</em> could be followed by almost anything, but <em>curled up on the</em> strongly suggests somewhere to sleep. The catch is that long contexts are rare. Most four-word sequences you could type have never appeared in any text the model has seen, and a pure counting model has nothing to say about them.</p>
      <p>The usual fix is to <strong>back off</strong>: if the long context was never seen, lean on the prediction from a shorter one, and if necessary on plain word frequencies. The playground below does exactly that, blending the counts for one, two and three words of context. It learned from a few pages of short stories the moment this page opened.</p>
    `));
    article.append(playground(ctx, life));
    article.append(callout("try", "<p>Start from “Once upon a time, a little” and switch the look-back between one, two and three words: the longer context knows that stories here often begin with a little <em>girl</em>. Press <em>Write 12 words</em> with each setting: one word of look-back gives word salad, while three words make the model recite whole phrases from its stories, a miniature version of memorisation. Then type a word the stories never use — <em>volcano</em>, say — and watch the model back off. Finally, drag the temperature to the far left and write again, then try the far right.</p>"));

    article.append(prose(`
      <h2>From counting to learning</h2>
      <p>Counting works surprisingly well for a toy, but it collapses at scale. With a vocabulary of 50,000 words there are two and a half billion possible word pairs and more than a hundred trillion triples; no library on Earth contains most of them even once. A counting model also cannot generalise. Having read <em>the cat ate the fish</em> tells it nothing about <em>the kitten ate the fish</em>, because in a table <em>cat</em> and <em>kitten</em> are simply different rows.</p>
      <p>Neural language models replace the table with a function. Instead of storing counts, they hold millions of parameters that turn a context into probabilities by calculation. Words that behave alike end up with similar internal representations (chapter 4), so whatever is learned about one carries over to the other. And a Transformer (chapter 5) can take hundreds or thousands of previous tokens into account, weighing the ones that matter, instead of a fixed window of two or three words.</p>
      <p>The job, however, stays exactly the same: given the context, produce a probability for every possible next token. Everything you will ever see a large model do is built on that single ability.</p>
    `));
    article.append(callout("key", "<p>A language model never writes text directly. It answers one question only: <em>how likely is each possible next token?</em> Writing is what happens when you ask that question again and again and act on the answers.</p>"));

    article.append(prose(`
      <h2>Writing by sampling</h2>
      <p>To produce text, we run a simple loop:</p>
      <ol>
        <li>Give the context to the model and get a probability for every token in its vocabulary.</li>
        <li>Choose one token according to those probabilities.</li>
        <li>Append it to the context.</li>
        <li>Repeat until there is enough text, or until the model produces a special token that means “stop”.</li>
      </ol>
      <p>The interesting part is step 2. The obvious strategy — always take the single most likely token, called <strong>greedy decoding</strong> — sounds sensible but works poorly. Text written that way is bland, and it easily falls into loops (<em>the dog was happy. the dog was happy. the dog …</em>), because the likeliest continuation of a repeated phrase is often the phrase again.</p>
      <p>Instead, models usually <strong>sample</strong>: they roll a weighted die. A token with probability 0.3 is chosen about three times in ten. This keeps the writing varied, and the same prompt can give a different text every time. It also means that an unlikely token is picked now and then — sometimes to delightful effect, sometimes sending the text off the rails.</p>

      <h2>Temperature, top-k and top-p</h2>
      <p>The probabilities a model produces can be reshaped before the die is rolled. The best-known knob is <strong>temperature</strong>, <span class="bk-var">T</span>. Each probability is raised to the power 1/<span class="bk-var">T</span>, and the results are rescaled so that they add up to 1 again:</p>
      <div class="bk-eq">p′<sub><i>i</i></sub> = <span class="bk-frac"><span>p<sub><i>i</i></sub><sup>1/<i>T</i></sup></span><span>Σ<sub><i>j</i></sub> p<sub><i>j</i></sub><sup>1/<i>T</i></sup></span></span><span class="bk-eq-note">raise every probability to the power 1/T, then rescale so they add up to 1</span></div>
      <p>At <span class="bk-var">T</span> = 1 nothing changes. Below 1 the gaps between likely and unlikely tokens are exaggerated and the favourite takes over; as <span class="bk-var">T</span> approaches 0, sampling turns into greedy decoding. Above 1 the distribution flattens and rare tokens get a real chance. (Internally, models first compute raw scores called <strong>logits</strong>; dividing the logits by <span class="bk-var">T</span> before turning them into probabilities is exactly the same operation.)</p>
      <p>Two filters are often applied as well. <strong>Top-k</strong> keeps only the <span class="bk-var">k</span> most likely tokens. <strong>Top-p</strong>, also called nucleus sampling, keeps the smallest group of top tokens whose probabilities add up to at least <span class="bk-var">p</span>. Top-p adapts to the situation: when the model is confident only a handful of tokens survive, and when it is unsure many do. Both protect against the long tail of individually unlikely tokens, which together can hold a surprising share of the probability.</p>
    `));
    article.append(lens(ctx, life));
    article.append(prose(`
      <p>The Sampling page of the workbench offers all three controls, plus <em>min-p</em>, which drops every token that is much less likely than the current favourite. There is no single right setting. Answers that should be accurate call for low temperatures; playful stories can take higher ones.</p>

      <h2>Fluent is not the same as true</h2>
      <p>Here is the most important thing to understand about language models, large or small. They are trained to produce text that is <em>likely</em> — text that resembles what they have read. Most of the time likely text and true text coincide, because most of what people write is roughly true. But nothing in the training objective checks facts. Ask a model about something it has rarely seen and it will still produce a fluent, confident continuation, because that is what text usually looks like. When the result is wrong we call it a <strong>hallucination</strong>, but from the model's point of view it is simply doing its job.</p>
      <p>Your own small model will make this vivid. Trained on children's stories, it will invent names, places and events with complete assurance. That is not a flaw in your training run; it is the nature of the machine.</p>

      <h2>Keeping score</h2>
      <p>If a language model is a guesser, how do we tell a good one from a bad one? Show it real text it has never seen and look at the probability it gave to each token that actually came next. A good model is rarely surprised: it gave high probability to what really happened. That measure of surprise, the <strong>loss</strong>, is the number you will watch fall during training, and chapter 7 takes it apart in detail. First, though, we need to look more closely at the pieces being predicted.</p>
    `));

    finishChapter(shell, ctx, { id: "tokenization", title: "Tokenization", blurb: "How text is cut into tokens, and why the pieces matter." });
    return () => life.dispose();
  },
};
