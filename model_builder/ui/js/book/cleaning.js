// Pure helpers for Chapter 6: a toy web-page cleaning pipeline and MinHash near-duplicate detection.
import { seededRandom } from "./kit.js";

// A made-up page from a made-up site, as a crawler would save it. Lines starting with "+" are the
// article text a person would want to keep; every other line is markup, boilerplate or junk.
export const MESSY_PAGE = `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>How to Fix a Flat Tyre Without a Workshop | Spoke &amp; Chain</title>
<style>body { font-family: sans-serif; } .ad { border: 1px solid #ccc; }</style>
<script>window.stats = window.stats || []; function track(e) { stats.push(e); }</script>
</head>
<body>
<div class="cookies">We use cookies to improve your experience. <a href="/privacy">Privacy policy</a> <button>Accept all</button></div>
<nav>Home | Shop | Guides | Forum | Log in</nav>
<div class="crumbs">Home &rsaquo; Guides &rsaquo; Repairs &rsaquo; Tyres</div>
+<h1>How to fix a flat tyre without a workshop</h1>
<p class="byline">By Ruth Okafor &middot; 12 March 2024 &middot; 6 min read</p>
<div class="share">Share | Tweet | Pin it | Email</div>
<div class="rating">&#9733;&#9733;&#9733;&#9733;&#9734; 4.6/5 (212 votes)</div>
+<p>A flat tyre is the most common thing to go wrong on a bike, and the good news is that you can fix one at the roadside in about fifteen minutes with a few cheap tools.</p>
+<h2>What you will need</h2>
+<ul><li>Two or three tyre levers (plastic ones are kinder to your rims)</li>
+<li>A spare inner tube or a puncture repair kit</li>
+<li>A pump that fits your type of valve</li></ul>
<div class="ad">BEST PRICES ON TYRES!!! FREE DELIVERY &gt;&gt; www.cheap-tyres-4u.example</div>
+<h2>Step 1: Take the wheel off</h2>
+<p>Shift into the smallest rear sprocket, open the brake if it has a quick release, and undo the axle. Lift the frame and the wheel should drop out.</p>
+<h2>Step 2: Get the tyre off the rim</h2>
+<p>Let out any air that&rsquo;s left. Hook a lever under the edge of the tyre opposite the valve, prise it over the rim and clip the lever to a spoke. Run the second lever round until one side of the tyre comes free.</p>
+<p class="tip">Tip: never use a screwdriver as a lever. It will pinch the new tube and you will be back where you started.</p>
+<h2>Step 3: Find the hole</h2>
+<p>Pump a little air into the old tube and listen for the hiss, or hold it close to your lips, which are surprisingly good at feeling a thin jet of air. If there is water nearby, dunk the tube and watch for bubbles.</p>
<p>Read more: <a href="/guides/tubeless">Should you go tubeless?</a></p>
+<h2>Step 4: Patch it or replace it</h2>
+<p>Roughen the area around the hole with the sandpaper from your kit, spread a thin layer of glue, wait until it turns dull, then press the patch on firmly for a full minute.</p>
<p class="tip">Tip: never use a screwdriver as a lever. It will pinch the new tube and you will be back where you started.</p>
+<h2>Step 5: Put it all back together</h2>
+<p>Push the valve through the rim, tuck the tube inside the tyre and work the tyre back over the rim with your thumbs, finishing at the valve. Pump it up to the pressure printed on the side of the tyre.</p>
<p>&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;</p>
<div class="related">Related posts: <a href="/chain">Chain care for beginners</a> &middot; <a href="/brakes">Brake pads</a> &middot; <a href="/winter">Winter riding</a></div>
<h3>4 comments</h3>
<p>Great post!!! Thanks</p>
+<p>Tip from a shop mechanic: before you fit the new tube, run your fingers slowly round the inside of the tyre, or whatever caused the first puncture will cause the second.</p>
<p>Great post!!! Thanks</p>
<p>Danke f&uuml;r die Anleitung, das hat bei meinem Rad sofort funktioniert!</p>
<p>EARN $500 A DAY FROM HOME, NO EXPERIENCE NEEDED: http://quick-cash.example/join</p>
<footer>&copy; 2024 Spoke &amp; Chain Ltd. All rights reserved. | Privacy | Terms | Cookies</footer>
<div class="newsletter">Subscribe to our newsletter for weekly tips!</div>
<script>track({ event: "pageview" });</script>
</body>
</html>
`;

const ENTITIES = {
  amp: "&", lt: "<", gt: ">", quot: "\"", apos: "'", nbsp: " ", rsquo: "’", lsquo: "‘", ldquo: "“", rdquo: "”",
  mdash: "—", ndash: "–", middot: "·", rsaquo: "›", raquo: "»", copy: "©", uuml: "ü", hellip: "…",
};

export function decodeEntities(text) {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, name) => {
    if (name[0] === "#") {
      const code = name[1] === "x" || name[1] === "X" ? parseInt(name.slice(2), 16) : parseInt(name.slice(1), 10);
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : match;
    }
    return ENTITIES[name.toLowerCase()] ?? match;
  });
}

// Splits a page into lines and works out what plain-text extraction leaves of each one:
// <head>, <script> and <style> blocks vanish, tags become spaces and entities become characters.
export function parsePage(source) {
  let skipping = null;
  return source.replace(/^\s*\n/, "").replace(/\s+$/, "").split("\n").map((line, index) => {
    const good = line.startsWith("+");
    const raw = good ? line.slice(1) : line;
    const lower = raw.toLowerCase();
    let hidden = false;
    if (skipping) {
      hidden = true;
      if (lower.includes(`</${skipping}>`)) skipping = null;
    } else {
      const open = lower.match(/<(head|script|style)[\s>]/);
      if (open) {
        hidden = true;
        if (!lower.includes(`</${open[1]}>`)) skipping = open[1];
      }
    }
    const text = hidden ? "" : decodeEntities(raw.replace(/<[^>]*>/g, " ")).replace(/\s+/g, " ").trim();
    return { index, raw, text, good };
  });
}

const BOILERPLATE = [
  /\bcookies?\b/i, /\bprivacy\b/i, /all rights reserved/i, /©/, /\bsubscribe\b/i, /\bnewsletter\b/i,
  /\blog ?in\b|\bsign (in|up)\b/i, /\btweet\b|\bpin it\b|^share\b/i, /^home\b/i, /\bread more\b/i,
  /\brelated (posts|articles)\b/i, /\bmin read\b/i, /[›»]/,
];

const ENGLISH_WORDS = new Set(("the a an and or but to of in on at for with from by is are was were be it its this that you your i my we our " +
  "they he she his her not no so if as up out off into than then there what which who when how all can will would just about more very").split(" "));
const OTHER_WORDS = new Set(("der die das und ist nicht ein eine mit für bei hat sich auf den dem zu le la les et est une des du que qui pour pas " +
  "el los las y es por para con una del").split(" "));

export const wordCount = (text) => text.split(/\s+/).filter(Boolean).length;

function letterShare(text) {
  const visible = text.replace(/\s/g, "");
  if (!visible.length) return 1;
  return (visible.match(/\p{L}/gu) || []).length / visible.length;
}

function isSpam(text) {
  if (/https?:\/\/|\bwww\.|\.(com|net|org|example)\b/i.test(text)) return true;
  const letters = (text.match(/\p{L}/gu) || []).length;
  const upper = (text.match(/\p{Lu}/gu) || []).length;
  return letters >= 12 && upper / letters > 0.6;
}

function looksForeign(text) {
  const words = text.toLowerCase().match(/\p{L}+/gu) || [];
  if (words.length < 4) return false;
  let english = 0;
  let other = 0;
  for (const word of words) {
    if (ENGLISH_WORDS.has(word)) english += 1;
    if (OTHER_WORDS.has(word)) other += 1;
  }
  return other >= 2 && other > english;
}

// Pipeline stages in the order they run. A removed line is credited to the first stage that catches it.
export const FILTERS = [
  {
    id: "markup", label: "Strip markup", color: "#8e8e93", standard: true,
    about: "Text extraction. Drops the page's head, scripts and styles, turns the remaining tags into plain line breaks and decodes entities such as <code>&amp;rsquo;</code> back into characters. Real extractors also guess which block of the page is the main article.",
  },
  {
    id: "boilerplate", label: "Boilerplate", color: "#0a84ff", standard: true,
    about: "Drops lines that match phrases repeated across thousands of sites: cookie notices, menus, share buttons, “read more” links, bylines and copyright footers.",
    test: (text) => BOILERPLATE.some((pattern) => pattern.test(text)),
  },
  {
    id: "spam", label: "Links & shouting", color: "#e0559b", standard: true,
    about: "A crude spam detector: drops lines that contain a web address or are written mostly in capital letters.",
    test: isSpam,
  },
  {
    id: "symbols", label: "Mostly symbols", color: "#a77bff", standard: true,
    about: "Drops lines in which letters make up less than 60% of the visible characters: star ratings, separators, tables of numbers.",
    test: (text) => letterShare(text) < 0.6,
  },
  {
    id: "language", label: "Not English", color: "#2fb5c5", standard: true,
    about: "A rough language check: drops lines that contain more common German, French or Spanish words than common English ones. Real pipelines use a trained language identifier that also reports how confident it is.",
    test: looksForeign,
  },
  {
    id: "short", label: "Short lines", color: "#ff9f0a", standard: true,
    about: "Drops lines with fewer words than the slider allows. That catches menu fragments and one-line comments, but set it high and headings and list items go too.",
    test: (text, options) => wordCount(text) < options.minWords,
  },
  {
    id: "duplicates", label: "Duplicates", color: "#30b158", standard: true,
    about: "Keeps only the first copy of any line that appears more than once, ignoring capital letters and spacing.",
  },
  {
    id: "strict", label: "Strict endings", color: "#a2845e", standard: false,
    about: "Keeps only lines that end like a sentence: with a full stop, a question mark, an exclamation mark or a closing quote. Some well-known datasets used exactly this rule. Watch what it does to the headings and the list of tools.",
    test: (text) => !/[.!?"'”’]$/.test(text),
  },
];

export function runPipeline(lines, enabled, { minWords = 4 } = {}) {
  const seen = new Set();
  const markup = enabled.has("markup");
  return lines.map((line) => {
    const current = markup ? line.text : line.raw;
    let removedBy = null;
    if (markup && !line.text) {
      removedBy = "markup";
    } else {
      for (const filter of FILTERS) {
        if (filter.id === "markup" || !enabled.has(filter.id)) continue;
        if (filter.id === "duplicates") {
          const key = current.toLowerCase().replace(/\s+/g, " ").trim();
          if (seen.has(key)) {
            removedBy = filter.id;
            break;
          }
          seen.add(key);
        } else if (filter.test(current, { minWords })) {
          removedBy = filter.id;
          break;
        }
      }
    }
    return { ...line, current, removedBy };
  });
}

const encoder = new TextEncoder();
export const utf8Length = (text) => encoder.encode(text).length;

// Totals for the figure: lines and bytes before and after, where the removed bytes went,
// and how well the filters separated article text from junk.
export function pipelineStats(results) {
  const stats = {
    linesBefore: results.length, linesAfter: 0, bytesBefore: 0, bytesAfter: 0,
    removedLines: {}, removedBytes: {}, goodTotal: 0, goodKept: 0, junkTotal: 0, junkRemoved: 0,
  };
  for (const filter of FILTERS) {
    stats.removedLines[filter.id] = 0;
    stats.removedBytes[filter.id] = 0;
  }
  for (const line of results) {
    const rawBytes = utf8Length(line.raw) + 1;
    stats.bytesBefore += rawBytes;
    if (line.good) stats.goodTotal += 1;
    else stats.junkTotal += 1;
    if (line.removedBy) {
      stats.removedLines[line.removedBy] += 1;
      stats.removedBytes[line.removedBy] += rawBytes;
      if (!line.good) stats.junkRemoved += 1;
    } else {
      const keptBytes = utf8Length(line.current) + 1;
      stats.linesAfter += 1;
      stats.bytesAfter += keptBytes;
      stats.removedBytes.markup += rawBytes - keptBytes;
      if (line.good) stats.goodKept += 1;
    }
  }
  return stats;
}

/* ---------- near-duplicate detection ---------- */

// Words and the gaps between them, so a text can be redrawn with highlights.
export function splitWords(text) {
  const parts = text.match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*|[^\p{L}\p{N}]+/gu) || [];
  return parts.map((part) => ({ text: part, word: /[\p{L}\p{N}]/u.test(part) ? part.toLowerCase().replace(/’/g, "'") : null }));
}

// Overlapping runs of k words ("shingles"), each with the indices of the parts it covers.
export function shingleList(parts, k = 3) {
  const words = [];
  parts.forEach((part, index) => {
    if (part.word) words.push(index);
  });
  const list = [];
  for (let start = 0; start + k <= words.length; start += 1) {
    const covered = words.slice(start, start + k);
    list.push({ key: covered.map((index) => parts[index].word).join(" "), parts: covered });
  }
  return list;
}

export function jaccard(a, b) {
  if (!a.size || !b.size) return NaN;
  let shared = 0;
  for (const item of a) if (b.has(item)) shared += 1;
  return shared / (a.size + b.size - shared);
}

function fnv1a(text) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function mix32(value) {
  let hash = value >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return hash >>> 0;
}

export function minhashSeeds(count = 64, seed = 2024) {
  const random = seededRandom(seed);
  return Array.from({ length: count }, () => Math.floor(random() * 4294967296) >>> 0);
}

// For each of the seeded hash functions, the smallest hash any shingle produces.
export function minhashSignature(keys, seeds) {
  const signature = new Array(seeds.length).fill(Infinity);
  for (const key of keys) {
    const base = fnv1a(key);
    for (let index = 0; index < seeds.length; index += 1) {
      const value = mix32(base ^ seeds[index]);
      if (value < signature[index]) signature[index] = value;
    }
  }
  return signature;
}
