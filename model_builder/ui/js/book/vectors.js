// Word vectors for the embeddings chapter: loads data/embeddings.json (a GloVe subset) and
// provides cosine similarity, nearest neighbours and vector arithmetic.

export const EMBEDDINGS_URL = new URL("../../data/embeddings.json", import.meta.url);

export function dot(a, b) {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) sum += a[index] * b[index];
  return sum;
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function cosine(a, b) {
  const denominator = norm(a) * norm(b);
  return denominator ? dot(a, b) / denominator : 0;
}

export class WordVectors {
  constructor(json) {
    this.source = json.source;
    this.license = json.license;
    this.url = json.url;
    this.dims = json.dims;
    this.categories = json.categories;
    this.words = json.words.map(([word, category, rank, x, y], index) => ({ word, category, rank, x, y, index }));
    this.byWord = new Map(this.words.map((entry) => [entry.word, entry]));
    this.vectors = json.vectors.map((row) => {
      const vector = Float32Array.from(row, (value) => value / json.scale);
      const length = norm(vector) || 1;
      for (let index = 0; index < vector.length; index += 1) vector[index] /= length;
      return vector;
    });
  }

  has(word) {
    return this.byWord.has(word);
  }

  get(word) {
    return this.byWord.get(word) || null;
  }

  vector(word) {
    const entry = this.byWord.get(word);
    return entry ? this.vectors[entry.index] : null;
  }

  // Words ranked by cosine similarity to a vector, skipping any in `exclude`.
  nearest(vector, count = 8, exclude = new Set()) {
    const length = norm(vector) || 1;
    const scored = [];
    for (const entry of this.words) {
      if (exclude.has(entry.word)) continue;
      scored.push({ ...entry, score: dot(vector, this.vectors[entry.index]) / length });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, count);
  }

  neighbours(word, count = 8) {
    const vector = this.vector(word);
    return vector ? this.nearest(vector, count, new Set([word])) : [];
  }

  // a − b + c, the classic analogy query ("king − man + woman").
  offset(a, b, c) {
    const va = this.vector(a);
    const vb = this.vector(b);
    const vc = this.vector(c);
    if (!va || !vb || !vc) return null;
    return va.map((value, index) => value - vb[index] + vc[index]);
  }

  suggestions(prefix, count = 6) {
    const query = prefix.trim().toLowerCase();
    if (!query) return [];
    const starts = this.words.filter((entry) => entry.word.startsWith(query));
    const contains = this.words.filter((entry) => !entry.word.startsWith(query) && entry.word.includes(query));
    return [...starts.sort((a, b) => a.rank - b.rank), ...contains.sort((a, b) => a.rank - b.rank)].slice(0, count);
  }

  // Words with a similar spelling (typos, swapped letters), closest first.
  similarSpelling(text, count = 4) {
    const query = text.trim().toLowerCase();
    if (!query) return [];
    const limit = query.length <= 4 ? 1 : 2;
    return this.words
      .map((entry) => ({ entry, distance: editDistance(query, entry.word, limit) }))
      .filter((item) => item.distance <= limit)
      .sort((a, b) => a.distance - b.distance || a.entry.rank - b.entry.rank)
      .slice(0, count)
      .map((item) => item.entry);
  }
}

// Optimal-string-alignment distance (insert, delete, substitute, swap neighbours),
// giving up early once every alignment costs more than `limit`.
export function editDistance(a, b, limit = Infinity) {
  if (Math.abs(a.length - b.length) > limit) return limit + 1;
  let before = null;
  let previous = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i += 1) {
    const row = [i];
    let best = i;
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let value = Math.min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + cost);
      if (before && i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) value = Math.min(value, before[j - 2] + 1);
      row.push(value);
      best = Math.min(best, value);
    }
    if (best > limit) return limit + 1;
    before = previous;
    previous = row;
  }
  return previous[b.length];
}

let pending = null;

export function loadWordVectors() {
  if (!pending) {
    pending = fetch(EMBEDDINGS_URL)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((json) => new WordVectors(json))
      .catch((error) => {
        pending = null;
        throw error;
      });
  }
  return pending;
}
