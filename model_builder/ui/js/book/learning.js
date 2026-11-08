// Pure maths for Chapters 7 and 8: the engine's learning-rate schedule, a tiny character-level
// network trained with AdamW in the browser, and the DPO loss. No DOM access, so node can test it.

// Same schedule as learning_rate_at() in model.py: linear warmup, then cosine decay to
// floorRatio × peak at `total` steps. With no total (train until stopped) it stays at the peak.
export function learningRateAt(step, peak, warmup, total, floorRatio = 0.1) {
  if (warmup > 0 && step < warmup) return (peak * (step + 1)) / warmup;
  if (!total || total <= warmup) return peak;
  const progress = Math.min(1, (step - warmup) / Math.max(1, total - warmup));
  const floor = peak * floorRatio;
  return floor + 0.5 * (peak - floor) * (1 + Math.cos(Math.PI * progress));
}

// Deterministic PRNG (mulberry32), duplicated from kit.js so this module stays DOM-free.
export function mulberry32(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  const u = Math.max(random(), 1e-12);
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const logSigmoid = (x) => (x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x)));
export const sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));

/* ---------- DPO ---------- */

// chosen/rejected are log-ratios: log π_θ(answer) − log π_ref(answer), summed over the answer's tokens.
export function dpoLoss(chosen, rejected, beta) {
  const margin = chosen - rejected;
  return {
    margin,
    reward: beta * margin,
    loss: -logSigmoid(beta * margin),
    weight: sigmoid(-beta * margin),
    prefer: sigmoid(beta * margin),
  };
}

// Toy preference tuning: each answer is a vector of traits, and the tuned model's log-ratio for an
// answer is θ · traits. Plain gradient steps on the mean DPO loss over the banked pairs.
export function tunePreferences(pairs, { beta = 0.1, steps = 60, lr = 40 } = {}) {
  const size = pairs.length ? pairs[0].chosen.length : 0;
  const theta = new Array(size).fill(0);
  const losses = [];
  for (let step = 0; step <= steps; step += 1) {
    const grad = new Array(size).fill(0);
    let loss = 0;
    for (const pair of pairs) {
      const diff = pair.chosen.map((value, index) => value - pair.rejected[index]);
      const margin = diff.reduce((sum, value, index) => sum + value * theta[index], 0);
      loss -= logSigmoid(beta * margin);
      const weight = sigmoid(-beta * margin);
      diff.forEach((value, index) => {
        grad[index] -= beta * weight * value;
      });
    }
    losses.push(loss / Math.max(1, pairs.length));
    if (step === steps) break;
    for (let index = 0; index < size; index += 1) theta[index] -= (lr * grad[index]) / Math.max(1, pairs.length);
  }
  const accuracy = pairs.length
    ? pairs.filter((pair) => pair.chosen.reduce((sum, value, index) => sum + (value - pair.rejected[index]) * theta[index], 0) > 0).length / pairs.length
    : 0;
  return { theta, losses, accuracy };
}

/* ---------- a tiny character-level language model ---------- */

export function makeVocab(...texts) {
  const chars = [...new Set(texts.join(""))].sort();
  return { chars, index: new Map(chars.map((char, position) => [char, position])), size: chars.length };
}

export function encode(vocab, text) {
  const ids = new Int32Array(text.length);
  for (let position = 0; position < text.length; position += 1) ids[position] = vocab.index.get(text[position]) ?? 0;
  return ids;
}

const ADAM = { beta1: 0.9, beta2: 0.95, eps: 1e-8, decay: 0.1, clip: 1.0 };

/**
 * A neural network in the style of Bengio et al. (2003): the previous `context` characters are looked
 * up in an embedding table, joined, passed through one tanh layer and turned into one score per
 * character. Trained with the engine's recipe: AdamW (0.9, 0.95), weight decay 0.1 on matrices only,
 * gradient clipping at 1.0 and the warmup + cosine schedule above.
 */
export class TinyCharModel {
  constructor({ vocab, context = 3, embed = 10, hidden = 64, seed = 7 }) {
    this.vocab = vocab;
    this.V = vocab.size;
    this.C = context;
    this.D = embed;
    this.H = hidden;
    this.start = this.V; // extra embedding row for "before the start of the text"
    const random = mulberry32(seed);
    const init = (size, std) => Float64Array.from({ length: size }, () => gaussian(random) * std);
    const inputs = this.C * this.D;
    this.params = {
      E: init((this.V + 1) * this.D, 0.5),
      W1: init(inputs * this.H, 1 / Math.sqrt(inputs)),
      b1: new Float64Array(this.H),
      W2: init(this.H * this.V, 0.02),
      b2: new Float64Array(this.V),
    };
    this.decayed = new Set(["E", "W1", "W2"]);
    this.grads = {};
    this.m = {};
    this.v = {};
    for (const [name, values] of Object.entries(this.params)) {
      this.grads[name] = new Float64Array(values.length);
      this.m[name] = new Float64Array(values.length);
      this.v[name] = new Float64Array(values.length);
    }
    this.t = 0;
    this.count = Object.values(this.params).reduce((sum, values) => sum + values.length, 0);
    this.e = new Float64Array(inputs);
    this.h = new Float64Array(this.H);
    this.p = new Float64Array(this.V);
    this.dh = new Float64Array(this.H);
    this.de = new Float64Array(inputs);
    this.ctx = new Int32Array(this.C);
  }

  setContext(ids, position) {
    for (let slot = 0; slot < this.C; slot += 1) {
      const at = position - this.C + slot;
      this.ctx[slot] = at >= 0 ? ids[at] : this.start;
    }
  }

  // Forward pass for the context in this.ctx; leaves probabilities in this.p.
  forward() {
    const { E, W1, b1, W2, b2 } = this.params;
    const { C, D, H, V, e, h, p, ctx } = this;
    for (let slot = 0; slot < C; slot += 1) {
      const row = ctx[slot] * D;
      for (let d = 0; d < D; d += 1) e[slot * D + d] = E[row + d];
    }
    h.set(b1);
    for (let i = 0; i < C * D; i += 1) {
      const value = e[i];
      const base = i * H;
      for (let k = 0; k < H; k += 1) h[k] += value * W1[base + k];
    }
    for (let k = 0; k < H; k += 1) h[k] = Math.tanh(h[k]);
    p.set(b2);
    for (let k = 0; k < H; k += 1) {
      const value = h[k];
      const base = k * V;
      for (let v = 0; v < V; v += 1) p[v] += value * W2[base + v];
    }
    let max = -Infinity;
    for (let v = 0; v < V; v += 1) max = Math.max(max, p[v]);
    let total = 0;
    for (let v = 0; v < V; v += 1) {
      p[v] = Math.exp(p[v] - max);
      total += p[v];
    }
    for (let v = 0; v < V; v += 1) p[v] /= total;
    return p;
  }

  // Mean cross-entropy (in nats) of predicting every character of `ids` from the ones before it.
  evaluate(ids) {
    let total = 0;
    for (let position = 0; position < ids.length; position += 1) {
      this.setContext(ids, position);
      total -= Math.log(Math.max(this.forward()[ids[position]], 1e-12));
    }
    return total / Math.max(1, ids.length);
  }

  // One optimisation step on `positions` (indices into ids). Returns the batch loss before the update.
  trainStep(ids, positions, lr) {
    const { W1, W2 } = this.params;
    const g = this.grads;
    const { C, D, H, V, e, h, p, dh, de, ctx } = this;
    for (const values of Object.values(g)) values.fill(0);
    const scale = 1 / positions.length;
    let loss = 0;
    for (const position of positions) {
      this.setContext(ids, position);
      this.forward();
      const target = ids[position];
      loss -= Math.log(Math.max(p[target], 1e-12));
      p[target] -= 1;
      for (let v = 0; v < V; v += 1) {
        p[v] *= scale;
        g.b2[v] += p[v];
      }
      for (let k = 0; k < H; k += 1) {
        const base = k * V;
        let sum = 0;
        for (let v = 0; v < V; v += 1) {
          g.W2[base + v] += h[k] * p[v];
          sum += W2[base + v] * p[v];
        }
        dh[k] = sum * (1 - h[k] * h[k]);
        g.b1[k] += dh[k];
      }
      for (let i = 0; i < C * D; i += 1) {
        const base = i * H;
        let sum = 0;
        for (let k = 0; k < H; k += 1) {
          g.W1[base + k] += e[i] * dh[k];
          sum += W1[base + k] * dh[k];
        }
        de[i] = sum;
      }
      for (let slot = 0; slot < C; slot += 1) {
        const row = ctx[slot] * D;
        for (let d = 0; d < D; d += 1) g.E[row + d] += de[slot * D + d];
      }
    }
    this.lastNorm = this.clip();
    this.adamw(lr);
    return loss * scale;
  }

  // Global gradient-norm clipping, like torch.nn.utils.clip_grad_norm_(…, 1.0).
  clip() {
    let squares = 0;
    for (const values of Object.values(this.grads)) for (const value of values) squares += value * value;
    const norm = Math.sqrt(squares);
    const factor = ADAM.clip / (norm + 1e-6);
    if (factor < 1) for (const values of Object.values(this.grads)) for (let i = 0; i < values.length; i += 1) values[i] *= factor;
    return norm;
  }

  adamw(lr) {
    this.t += 1;
    const { beta1, beta2, eps, decay } = ADAM;
    const correction1 = 1 - beta1 ** this.t;
    const correction2 = Math.sqrt(1 - beta2 ** this.t);
    for (const [name, values] of Object.entries(this.params)) {
      const grad = this.grads[name];
      const m = this.m[name];
      const v = this.v[name];
      const shrink = this.decayed.has(name) ? 1 - lr * decay : 1;
      for (let i = 0; i < values.length; i += 1) {
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        values[i] = values[i] * shrink - (lr / correction1) * (m[i] / (Math.sqrt(v[i]) / correction2 + eps));
      }
    }
  }

  // Continues `prompt` one character at a time; the same seed gives the same dice rolls.
  sample(prompt, length = 60, { temperature = 0.8, seed = 1234 } = {}) {
    const random = mulberry32(seed);
    const ids = Array.from(prompt, (char) => this.vocab.index.get(char) ?? 0);
    let out = "";
    for (let index = 0; index < length; index += 1) {
      this.setContext(ids, ids.length);
      const probs = this.forward();
      let weights = probs;
      if (temperature !== 1) {
        weights = Array.from(probs, (value) => value ** (1 / temperature));
        const total = weights.reduce((sum, value) => sum + value, 0);
        weights = weights.map((value) => value / total);
      }
      let r = random();
      let pick = weights.length - 1;
      for (let v = 0; v < weights.length; v += 1) {
        r -= weights[v];
        if (r <= 0) {
          pick = v;
          break;
        }
      }
      ids.push(pick);
      out += this.vocab.chars[pick];
    }
    return out;
  }
}

/* ---------- the little corpus for the live training figure (written for this book) ---------- */

export const TOY_TEXT = [
  "the little cat sat by the warm stove. the baker gave the cat a bit of bread. the cat ate the bread and went to sleep.",
  "in the morning the sun came up over the hill. the baker opened the shop and the town woke up.",
  "a boy ran to the shop to buy a loaf of bread. the boy said thank you and ran home. his mother was happy and gave him a hug.",
  "the rain came in the afternoon. the cat sat by the window and watched the rain. the boy played with a red ball in the hall.",
  "the ball rolled under the table and the cat ran after it. the mother laughed and the boy laughed too.",
  "at night the town was quiet. the baker went to bed early. the cat sat on the roof and looked at the moon.",
  "the moon was big and round and bright. a small bird sang in the tree by the shop. the bird was not afraid of the cat.",
  "the cat was not hungry, so it let the bird sing. the next day the boy went to the shop again.",
  "the baker had made a cake with red berries. the boy wanted the cake, but he had only one coin.",
  "the baker smiled and gave him a small piece. the boy shared the piece with his mother.",
  "the cat came home and sat by the stove again. the stove was warm and the cat was sleepy.",
  "the boy read a book about a brave dog. the dog in the book found a lost girl in the snow.",
  "the girl was cold, but the dog kept her warm. the boy liked the dog in the book.",
  "he asked his mother for a dog. his mother said they had a cat, and a cat was enough. the cat opened one eye and went back to sleep.",
].join(" ") + " ";

export const TOY_VALIDATION = "the girl ran to the shop to buy a cake. the baker gave her a loaf of bread and a small piece of cake. "
  + "the dog sat by the stove and went to sleep. in the afternoon the rain came over the hill and the town was quiet. "
  + "the boy watched the bird in the tree and the cat watched the boy. ";

// The first two sentences only: small enough for the network to learn by heart.
export const TOY_TINY = "the little cat sat by the warm stove. the baker gave the cat a bit of bread. ";
