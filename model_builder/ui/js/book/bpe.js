// A small byte-level BPE tokenizer that trains in the browser (chapter 3).
// Same recipe as GPT-2-style tokenizers: split text into word-like pieces, start from UTF-8 bytes,
// and repeatedly merge the most frequent adjacent pair into a new token.

// GPT-2's pre-tokenizer: contractions, words with their leading space, numbers, punctuation, whitespace.
const PRETOKEN = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const encoder = new TextEncoder();
const strict = new TextDecoder("utf-8", { fatal: true });
const PAIR = 65536;

export function pretokenize(text) {
  return String(text).match(PRETOKEN) || [];
}

export function utf8(text) {
  return encoder.encode(text);
}

// Readable label for a run of bytes: valid UTF-8 is shown as text, anything else as hex bytes.
export function bytesLabel(bytes) {
  try {
    return strict.decode(bytes);
  } catch {
    return Array.from(bytes, (byte) => (byte < 128 ? String.fromCharCode(byte) : `<${byte.toString(16).padStart(2, "0")}>`)).join("");
  }
}

export class BPE {
  constructor(merges, pieces) {
    this.merges = merges;
    this.pieces = pieces;
    this.rank = new Map(merges.map((merge, rank) => [merge.a * PAIR + merge.b, rank]));
    this.cache = new Map();
  }

  vocabSize(limit = this.merges.length) {
    return 256 + Math.min(limit, this.merges.length);
  }

  label(id) {
    return bytesLabel(this.pieces[id]);
  }

  encodePiece(piece, limit) {
    const key = `${limit}\u0000${piece}`;
    const cached = this.cache.get(key);
    if (cached) return cached;
    let ids = Array.from(utf8(piece));
    while (ids.length > 1) {
      let bestRank = Infinity;
      for (let index = 0; index + 1 < ids.length; index += 1) {
        const rank = this.rank.get(ids[index] * PAIR + ids[index + 1]);
        if (rank !== undefined && rank < limit && rank < bestRank) bestRank = rank;
      }
      if (bestRank === Infinity) break;
      const { a, b, id } = this.merges[bestRank];
      const next = [];
      for (let index = 0; index < ids.length; index += 1) {
        if (index + 1 < ids.length && ids[index] === a && ids[index + 1] === b) {
          next.push(id);
          index += 1;
        } else {
          next.push(ids[index]);
        }
      }
      ids = next;
    }
    if (this.cache.size > 4000) this.cache.clear();
    this.cache.set(key, ids);
    return ids;
  }

  // Token ids for `text`, using only the first `limit` merges.
  encode(text, limit = this.merges.length) {
    const out = [];
    for (const piece of pretokenize(text)) out.push(...this.encodePiece(piece, limit));
    return out;
  }
}

// Learn up to `maxMerges` merges from `text`. Stops early when no pair occurs at least `minCount` times.
export function trainBPE(text, maxMerges = 200, { minCount = 2 } = {}) {
  const frequency = new Map();
  for (const piece of pretokenize(text)) frequency.set(piece, (frequency.get(piece) || 0) + 1);
  const words = [...frequency].map(([piece, count]) => ({ ids: Array.from(utf8(piece)), count }));
  const pieces = Array.from({ length: 256 }, (_, byte) => Uint8Array.of(byte));
  const merges = [];
  while (merges.length < maxMerges) {
    const pairs = new Map();
    for (const word of words) {
      const { ids, count } = word;
      for (let index = 0; index + 1 < ids.length; index += 1) {
        const key = ids[index] * PAIR + ids[index + 1];
        pairs.set(key, (pairs.get(key) || 0) + count);
      }
    }
    let bestKey = -1;
    let bestCount = 0;
    for (const [key, count] of pairs) {
      if (count > bestCount || (count === bestCount && key < bestKey)) {
        bestKey = key;
        bestCount = count;
      }
    }
    if (bestCount < minCount) break;
    const a = Math.floor(bestKey / PAIR);
    const b = bestKey % PAIR;
    const id = pieces.length;
    const joined = new Uint8Array(pieces[a].length + pieces[b].length);
    joined.set(pieces[a]);
    joined.set(pieces[b], pieces[a].length);
    pieces.push(joined);
    merges.push({ a, b, id, count: bestCount });
    for (const word of words) {
      const { ids } = word;
      if (ids.length < 2) continue;
      const next = [];
      for (let index = 0; index < ids.length; index += 1) {
        if (index + 1 < ids.length && ids[index] === a && ids[index + 1] === b) {
          next.push(id);
          index += 1;
        } else {
          next.push(ids[index]);
        }
      }
      word.ids = next;
    }
  }
  return new BPE(merges, pieces);
}
