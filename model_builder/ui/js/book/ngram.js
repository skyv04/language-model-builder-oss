// A word-level n-gram language model with interpolated Witten-Bell smoothing.
export const SENTENCE_END = ".";
const ENDINGS = new Set([".", "!", "?"]);
const SPEECH = new Set(["said", "asked"]);
const NAMES = new Set(["mia", "tom", "sam", "i"]);

// Lowercase words and the punctuation marks . , ! ? — quotation marks are dropped.
export function wordTokens(text) {
  return (String(text).toLowerCase().replace(/[“”"]/g, " ").match(/[a-z]+(?:'[a-z]+)?|[.,!?;:]/g) || []);
}

// Corpus tokens with a sentence boundary at the start of every paragraph.
export function corpusTokens(text) {
  const tokens = [];
  for (const paragraph of String(text).split(/\n\s*\n/)) {
    const words = wordTokens(paragraph);
    if (!words.length) continue;
    if (!tokens.length || !ENDINGS.has(tokens[tokens.length - 1])) tokens.push(SENTENCE_END);
    tokens.push(...words);
  }
  return tokens;
}

// Readable text from word tokens: capitalised sentence starts and names, no space before punctuation.
export function detokenize(tokens, { startOfSentence = true } = {}) {
  let out = "";
  let capital = startOfSentence;
  let previous = "";
  for (const token of tokens) {
    const punctuation = /^[.,!?;:]$/.test(token);
    // After "!" or "?" inside dialogue ("…now!" said Sam) the sentence carries on in lower case.
    const speech = SPEECH.has(token) && (previous === "!" || previous === "?");
    let word = token;
    if (!punctuation && ((capital && !speech) || NAMES.has(token))) word = token[0].toUpperCase() + token.slice(1);
    out += punctuation || !out ? word : ` ${word}`;
    if (!punctuation) capital = false;
    if (ENDINGS.has(token)) capital = true;
    previous = token;
  }
  return out;
}

export function displayWord(token) {
  if (NAMES.has(token)) return token[0].toUpperCase() + token.slice(1);
  return token;
}

export class NGramModel {
  // `backoff` scales how much probability is handed down to shorter contexts (1 = classic Witten-Bell).
  // A smaller value trusts long contexts more, which reads better on a tiny corpus.
  constructor(tokens, maxContext = 3, { backoff = 0.35 } = {}) {
    this.maxContext = maxContext;
    this.backoff = backoff;
    this.vocab = [...new Set(tokens)].sort();
    this.index = new Map(this.vocab.map((word, index) => [word, index]));
    this.tables = Array.from({ length: maxContext + 1 }, () => new Map());
    for (let position = 0; position < tokens.length; position += 1) {
      for (let size = 0; size <= maxContext && size <= position; size += 1) {
        const key = tokens.slice(position - size, position).join(" ");
        let entry = this.tables[size].get(key);
        if (!entry) {
          entry = { total: 0, counts: new Map() };
          this.tables[size].set(key, entry);
        }
        entry.total += 1;
        entry.counts.set(tokens[position], (entry.counts.get(tokens[position]) || 0) + 1);
      }
    }
    this.tokenCount = tokens.length;
  }

  has(word) {
    return this.index.has(word);
  }

  count(history) {
    const entry = this.tables[history.length] && this.tables[history.length].get(history.join(" "));
    return entry ? entry.total : 0;
  }

  // Probability of every vocabulary word after `history`, mixing the longest seen contexts
  // (up to `lookBack` words) with shorter ones. Returns { probs, used, tried }.
  distribution(history, lookBack = this.maxContext) {
    const size = this.vocab.length;
    const unigram = this.tables[0].get("");
    let probs = new Float64Array(size);
    for (let index = 0; index < size; index += 1) {
      probs[index] = ((unigram.counts.get(this.vocab[index]) || 0) + 0.5) / (unigram.total + 0.5 * size);
    }
    let used = 0;
    const limit = Math.min(lookBack, this.maxContext, history.length);
    for (let length = 1; length <= limit; length += 1) {
      const entry = this.tables[length].get(history.slice(history.length - length).join(" "));
      if (!entry) break;
      const types = entry.counts.size;
      const weight = entry.total / (entry.total + this.backoff * types);
      const next = new Float64Array(size);
      for (let index = 0; index < size; index += 1) next[index] = (1 - weight) * probs[index];
      for (const [word, count] of entry.counts) next[this.index.get(word)] += (weight * count) / entry.total;
      probs = next;
      used = length;
    }
    return { probs, used, tried: limit };
  }

  // Top entries of a probability array (optionally reshaped by temperature), sorted descending.
  static top(model, probs, count = 8, temperature = 1) {
    const reshaped = reshape(probs, temperature);
    const order = Array.from(reshaped.keys()).sort((a, b) => reshaped[b] - reshaped[a]).slice(0, count);
    return { reshaped, items: order.map((index) => ({ index, word: model.vocab[index], prob: reshaped[index] })) };
  }
}

// p_i^(1/T), renormalised — the same as dividing log-probabilities (logits) by T.
export function reshape(probs, temperature = 1) {
  const t = Math.max(temperature, 0.01);
  const out = new Float64Array(probs.length);
  let max = -Infinity;
  for (let index = 0; index < probs.length; index += 1) {
    out[index] = Math.log(Math.max(probs[index], 1e-300)) / t;
    if (out[index] > max) max = out[index];
  }
  let total = 0;
  for (let index = 0; index < probs.length; index += 1) {
    out[index] = Math.exp(out[index] - max);
    total += out[index];
  }
  for (let index = 0; index < probs.length; index += 1) out[index] /= total;
  return out;
}
