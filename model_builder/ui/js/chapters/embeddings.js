// Chapter 4: Embeddings — token ids become vectors; similarity, directions, and learned tables.
import { h, svg, clear, clamp } from "../dom.js";
import { icon } from "../icons.js";
import { chapterShell, finishChapter, prose, callout, figure, lifecycle, note } from "../book/kit.js";
import { surface } from "../book/plot.js";
import { loadWordVectors, dot, norm } from "../book/vectors.js";

const CATEGORY_COLORS = {
  animals: "#ffb46b",
  food: "#ff7d8e",
  colors: "#ffe066",
  numbers: "#5fd4ff",
  people: "#b58cff",
  jobs: "#dcb6ff",
  body: "#ff9fd0",
  places: "#4fd18b",
  nature: "#9be6a6",
  time: "#5ea8ff",
  travel: "#ffcf5c",
  home: "#d9b38c",
  verbs: "#ff8a5c",
  adjectives: "#c4e36a",
  ideas: "#f2a6ff",
  hobbies: "#6ee6d8",
  tech: "#93b4ff",
  function: "#b7c1d6",
};

const DEFAULT_SENTENCE = "the queen gave the boy a red apple";
const TRY_WORDS = ["dog", "piano", "monday", "hot", "happy", "apple"];
const FIRST_WORD = "dog";

const PRESETS = [
  { a: "king", b: "man", c: "woman" },
  { a: "paris", b: "france", c: "italy" },
  { a: "walked", b: "walk", c: "swim" },
  { a: "bigger", b: "big", c: "small" },
  { a: "actor", b: "man", c: "woman" },
  { a: "moscow", b: "russia", c: "egypt" },
  { a: "doctor", b: "man", c: "woman", tag: "bias" },
  { a: "president", b: "man", c: "woman", tag: "miss" },
];

const PRESET_NOTES = {
  bias: "This answer comes from patterns in the news and encyclopedia text the vectors were learned from, not from any fact about doctors or nurses. The section below the figure has more on this.",
  miss: "No word on the map fits well here. The arithmetic always produces some nearest word, sensible or not; notice how far the best candidate lands from the target.",
};

let uid = 0;

function categoryColor(wv, entry) {
  return CATEGORY_COLORS[wv.categories[entry.category].id] || "#b7c1d6";
}

function dotSwatch(color) {
  return h("span", { class: "bk-dot", style: { background: color } });
}

// Opaque diverging colours: blue for negative numbers, orange for positive, fading into the panel.
const BASES = { light: [244, 245, 248], dark: [29, 43, 80] };
const NEGATIVE = [47, 140, 255];
const POSITIVE = [255, 159, 67];

function cellColor(value, tone) {
  const t = Math.min(1, Math.abs(value) / 0.3) ** 0.8;
  const base = BASES[tone];
  const target = value < 0 ? NEGATIVE : POSITIVE;
  const mix = base.map((channel, index) => Math.round(channel + (target[index] - channel) * t));
  return `rgb(${mix.join(", ")})`;
}

function strip(vector, { label = "", tone = "light" } = {}) {
  return h("span", { class: "bk-strip", role: "img", "aria-label": `${label || "vector"}: ${vector.length} numbers` },
    Array.from(vector, (value, index) => h("span", {
      class: "bk-cell",
      style: { background: cellColor(value, tone) },
      title: `${label ? `${label}, ` : ""}number ${index + 1}: ${value.toFixed(3)}`,
    })),
  );
}

function loading() {
  return h("div", { class: "bk-loading" }, h("span", { class: "spinner" }), "Loading word vectors…");
}

function failure() {
  return note("The word vectors (data/embeddings.json) could not be loaded, so this figure is empty. The rest of the chapter still works.", { tone: "warn" });
}

/* ---------- Figure 1: from ids to rows ---------- */

function lookupFigure() {
  const fig = figure({
    number: 1,
    title: "From token ids to rows",
    info: "one row of 50 numbers per word",
    tone: "light",
    caption: "Type a few words. Each one is swapped for its id (here simply its position in this book's 479-word list) and then for that row of the table: 50 numbers, drawn as cells, blue for negative and orange for positive. Hover a cell to read its value. A repeated word fetches exactly the same row; a word that is not in the table has no row at all.",
  });
  let wv = null;
  const input = h("input", { class: "text-input", type: "text", value: DEFAULT_SENTENCE, spellcheck: "false", autocomplete: "off", "aria-label": "Words to look up" });
  const reset = h("button", {
    class: "btn small icon-only",
    title: "Restore the example",
    "aria-label": "Restore the example",
    onclick: () => {
      input.value = DEFAULT_SENTENCE;
      update();
    },
  }, icon("reset", 14));
  const table = h("div", { class: "bk-lookup" }, loading());
  const summary = h("p", { class: "bk-lookup-summary" });
  fig.body.append(h("div", { class: "bk-input-row" }, input, reset), table, summary);
  input.addEventListener("input", () => update());

  function update() {
    if (!wv) return;
    const words = (input.value.toLowerCase().match(/[a-z]+/g) || []).slice(0, 14);
    clear(table);
    table.append(h("div", { class: "bk-lookup-row head", "aria-hidden": "true" },
      h("span", {}, "token"), h("span", { class: "bk-lookup-id" }, "id"), h("span", {}, "its row: 50 numbers")));
    const counts = new Map();
    let missing = 0;
    for (const word of words) {
      const entry = wv.get(word);
      const repeat = counts.has(word);
      counts.set(word, (counts.get(word) || 0) + 1);
      if (!entry) missing += 1;
      table.append(h("div", { class: ["bk-lookup-row", entry ? "" : "missing", repeat && entry ? "repeat" : ""] },
        h("span", { class: "bk-lookup-word" }, h("span", { class: "token light" }, word)),
        h("span", { class: "bk-lookup-id" }, entry ? String(entry.index) : "–"),
        entry ? strip(wv.vectors[entry.index], { label: word }) : h("span", { class: "bk-lookup-none" }, "not in this table"),
      ));
    }
    if (!words.length) {
      table.append(h("div", { class: "bk-lookup-none" }, "Type some words above."));
      summary.textContent = "";
      return;
    }
    const found = words.length - missing;
    const repeated = [...counts].find(([word, count]) => count > 1 && wv.has(word));
    let text = `${found} of ${words.length} word${words.length === 1 ? "" : "s"} found, so a grid of ${found} × 50 numbers is what the model actually receives.`;
    if (repeated) text += ` “${repeated[0]}” appears ${repeated[1] === 2 ? "twice" : `${repeated[1]} times`} and gets the identical row each time (outlined).`;
    if (missing) text += " A real tokenizer never has missing words: it would split them into smaller pieces that are in its table.";
    summary.textContent = text;
  }

  return {
    element: fig.element,
    ready(data) {
      wv = data;
      update();
    },
    fail() {
      clear(table).append(failure());
    },
  };
}

/* ---------- Figure 2: the word map ---------- */

function mapFigure(life) {
  const fig = figure({
    number: 2,
    title: "479 words, flattened from 50 dimensions to 2",
    info: "GloVe 6B · 50d",
    wide: true,
    caption: "Each dot is one of 479 everyday words, placed by flattening its 50-number GloVe vector onto the two directions in which these words vary most. Colours are categories assigned for this book; the vectors know nothing about them. Click a dot or search for a word to list its nearest neighbours, measured in all 50 dimensions; lines show where those neighbours landed after flattening. Pick a category to zoom in on it, and drag to pan while zoomed. Vectors: GloVe 6B 50d by Pennington, Socher and Manning (Stanford NLP), public domain under the ODC PDDL 1.0.",
  });
  let wv = null;
  let bounds = null;
  let selected = null;
  let neighbours = [];
  let focus = null;
  let hover = null;
  let geometry = null;
  let hoverLayer = null;
  let drag = null;
  let animation = null;
  let frameRequest = 0;
  const view = { cx: 0, cy: 0, zoom: 1 };

  const map = surface({ height: 480, className: "bk-map-svg", label: "Map of 479 word vectors", draw, life });
  map.element.classList.add("bk-map");
  const zoomIn = h("button", { class: "btn small icon-only", title: "Zoom in", "aria-label": "Zoom in", onclick: () => zoomBy(1.8) }, icon("plus", 14));
  const zoomOut = h("button", { class: "btn small icon-only", title: "Zoom out", "aria-label": "Zoom out", onclick: () => zoomBy(1 / 1.8) }, icon("minus", 14));
  const home = h("button", { class: "btn small icon-only", title: "Show all words", "aria-label": "Show all words", onclick: () => setFocus(null) }, icon("reset", 14));
  map.element.append(h("div", { class: "bk-map-tools" }, zoomOut, zoomIn, home), h("div", { class: "bk-map-status" }, loading()));
  const legend = h("div", { class: "bk-legend", role: "group", "aria-label": "Categories" });

  const search = h("input", { class: "text-input", type: "search", placeholder: "Find a word…", spellcheck: "false", autocomplete: "off", "aria-label": "Find a word on the map" });
  const suggestions = h("div", { class: "bk-suggest" });
  const detail = h("div", { class: "bk-map-detail" });
  const side = h("div", { class: "bk-map-side" }, h("label", { class: "bk-search" }, icon("search", 14), search), suggestions, detail);
  fig.body.append(h("div", { class: "bk-map-layout" }, h("div", { class: "bk-map-main" }, map.element, legend), side));

  const tween = life.loop((time) => {
    if (!animation) return false;
    if (animation.start === null) animation.start = time;
    const progress = Math.min(1, (time - animation.start) / 340);
    const eased = progress < 0.5 ? 2 * progress * progress : 1 - (-2 * progress + 2) ** 2 / 2;
    const { from, to } = animation;
    view.cx = from.cx + (to.cx - from.cx) * eased;
    view.cy = from.cy + (to.cy - from.cy) * eased;
    view.zoom = Math.exp(Math.log(from.zoom) + (Math.log(to.zoom) - Math.log(from.zoom)) * eased);
    map.redraw();
    if (progress >= 1) {
      animation = null;
      return false;
    }
    return true;
  });
  life.add(() => cancelAnimationFrame(frameRequest));

  function requestRedraw() {
    if (frameRequest || life.disposed) return;
    frameRequest = requestAnimationFrame(() => {
      frameRequest = 0;
      map.redraw();
    });
  }

  function homeView() {
    return { cx: (bounds.x0 + bounds.x1) / 2, cy: (bounds.y0 + bounds.y1) / 2, zoom: 1 };
  }

  function clampView(next) {
    return {
      cx: clamp(next.cx, bounds.x0, bounds.x1),
      cy: clamp(next.cy, bounds.y0, bounds.y1),
      zoom: clamp(next.zoom, 1, 14),
    };
  }

  function animateTo(target) {
    const to = target.zoom <= 1.0001 ? homeView() : clampView(target);
    animation = { from: { ...view }, to, start: null };
    map.element.classList.toggle("zoomed", to.zoom > 1.0001);
    zoomOut.disabled = to.zoom <= 1.0001;
    zoomIn.disabled = to.zoom >= 13.99;
    home.disabled = to.zoom <= 1.0001 && focus === null;
    tween.start();
  }

  function baseScale(width, height) {
    const pad = 26;
    return Math.min((width - pad * 2) / bounds.w, (height - pad * 2) / bounds.h);
  }

  function toData(px, py) {
    const { width, height, k } = geometry;
    return { x: view.cx + (px - width / 2) / k, y: view.cy - (py - height / 2) / k };
  }

  function zoomBy(factor, px, py) {
    if (!geometry) return;
    const { width, height } = geometry;
    const anchorX = px ?? width / 2;
    const anchorY = py ?? height / 2;
    const zoom = clamp((animation ? animation.to.zoom : view.zoom) * factor, 1, 14);
    const point = toData(anchorX, anchorY);
    const k = baseScale(width, height) * zoom;
    animateTo({ cx: point.x - (anchorX - width / 2) / k, cy: point.y + (anchorY - height / 2) / k, zoom });
  }

  function setFocus(index) {
    focus = index === focus ? null : index;
    for (const chip of legend.children) {
      const active = Number(chip.dataset.index) === focus;
      chip.classList.toggle("active", active);
      chip.setAttribute("aria-pressed", String(active));
    }
    if (focus === null || !geometry) {
      animateTo({ zoom: 1 });
      return;
    }
    const members = wv.words.filter((entry) => entry.category === focus);
    const xs = members.map((entry) => entry.x);
    const ys = members.map((entry) => entry.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const { width, height } = geometry;
    const fitX = (width - 150) / Math.max(x1 - x0, 0.02);
    const fitY = (height - 90) / Math.max(y1 - y0, 0.02);
    const zoom = clamp(Math.min(fitX, fitY) / baseScale(width, height), 1, 9);
    animateTo({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, zoom });
  }

  function select(entry, { reveal = false } = {}) {
    selected = entry;
    neighbours = entry ? wv.neighbours(entry.word, 8) : [];
    if (entry && reveal && geometry) {
      const px = geometry.positions[entry.index * 2];
      const py = geometry.positions[entry.index * 2 + 1];
      if (px < 40 || px > geometry.width - 40 || py < 40 || py > geometry.height - 40) {
        animateTo({ cx: entry.x, cy: entry.y, zoom: animation ? animation.to.zoom : view.zoom });
      }
    }
    map.redraw();
    renderDetail();
  }

  /* drawing */

  function classify(entry, near) {
    if (selected) {
      if (entry === selected) return "sel";
      if (near.has(entry.word)) return "near";
      if (focus !== null && entry.category !== focus) return "dim";
      return "faded";
    }
    if (focus !== null) return entry.category === focus ? "focus" : "dim";
    return "normal";
  }

  const RADIUS = { sel: 6.5, near: 4.8, focus: 4.4, normal: 3.4, faded: 3.2, dim: 2.6 };

  function draw(root, width, height) {
    hoverLayer = null;
    geometry = null;
    if (!wv) return;
    const k = baseScale(width, height) * view.zoom;
    const positions = new Float32Array(wv.words.length * 2);
    for (const entry of wv.words) {
      positions[entry.index * 2] = width / 2 + (entry.x - view.cx) * k;
      positions[entry.index * 2 + 1] = height / 2 - (entry.y - view.cy) * k;
    }
    geometry = { width, height, k, positions };
    const at = (entry) => [positions[entry.index * 2], positions[entry.index * 2 + 1]];
    const inside = ([x, y]) => x > -12 && x < width + 12 && y > -12 && y < height + 12;
    const near = new Set(neighbours.map((entry) => entry.word));

    const links = svg("g", { class: "bk-map-links" });
    if (selected) {
      const [x0, y0] = at(selected);
      for (const entry of neighbours) {
        const [x1, y1] = at(entry);
        links.append(svg("line", { x1: x0, y1: y0, x2: x1, y2: y1 }));
      }
    }

    const states = new Map(wv.words.map((entry) => [entry, classify(entry, near)]));
    const layer = { dim: 0, faded: 1, normal: 2, focus: 3, near: 4, sel: 5 };
    const dots = svg("g", { class: "bk-map-dots" });
    for (const entry of [...wv.words].sort((a, b) => layer[states.get(a)] - layer[states.get(b)])) {
      const point = at(entry);
      if (!inside(point)) continue;
      const state = states.get(entry);
      dots.append(svg("circle", { class: state, cx: point[0].toFixed(1), cy: point[1].toFixed(1), r: RADIUS[state], fill: categoryColor(wv, entry) }));
    }

    const labels = svg("g", { class: "bk-map-labels" });
    const placed = [];
    const place = (entry, state, force = false) => {
      const [x, y] = at(entry);
      if (!inside([x, y])) return false;
      const size = state === "sel" ? 13.5 : 11;
      const w = entry.word.length * size * 0.6 + 4;
      const r = RADIUS[state];
      const options = [
        [x + r + 3, y - 7, "start"],
        [x - r - 3 - w, y - 7, "end"],
        [x - w / 2, y - r - 16, "middle"],
        [x - w / 2, y + r + 2, "middle"],
      ];
      for (const [left, top, anchor] of options) {
        const box = { left, top, right: left + w, bottom: top + 14 };
        if (box.left < 2 || box.right > width - 2 || box.top < 2 || box.bottom > height - 2) continue;
        if (!force && placed.some((other) => box.left < other.right && box.right > other.left && box.top < other.bottom && box.bottom > other.top)) continue;
        placed.push(box);
        const tx = anchor === "start" ? left : anchor === "end" ? box.right : left + w / 2;
        labels.append(svg("text", {
          class: state,
          x: tx.toFixed(1),
          y: (top + 11).toFixed(1),
          "text-anchor": anchor,
          fill: state === "near" || state === "focus" ? categoryColor(wv, entry) : null,
        }, entry.word));
        return true;
      }
      return false;
    };
    // The top-right corner holds the zoom buttons.
    placed.push({ left: width - 96, top: 0, right: width, bottom: 40 });
    if (selected) place(selected, "sel", true);
    for (const entry of neighbours) place(wv.get(entry.word), "near");
    const budget = Math.round((width * height) / (selected ? 7000 : 4200));
    let count = 0;
    const rest = wv.words
      .filter((entry) => entry !== selected && !near.has(entry.word) && (focus === null || entry.category === focus))
      .sort((a, b) => a.rank - b.rank);
    for (const entry of rest) {
      if (count >= budget) break;
      if (place(entry, states.get(entry))) count += 1;
    }

    hoverLayer = svg("g", { class: "bk-map-hover" });
    root.append(links, dots, labels, hoverLayer);
    renderHover();
  }

  function renderHover() {
    if (!hoverLayer || !geometry) return;
    clear(hoverLayer);
    if (!hover || hover === selected) return;
    const x = geometry.positions[hover.index * 2];
    const y = geometry.positions[hover.index * 2 + 1];
    const text = hover.word;
    const w = text.length * 7.4 + 14;
    const left = clamp(x - w / 2, 2, geometry.width - w - 2);
    const top = y - 32 < 2 ? y + 12 : y - 32;
    hoverLayer.append(
      svg("circle", { cx: x, cy: y, r: 8.5, class: "ring" }),
      svg("rect", { x: left, y: top, width: w, height: 21, rx: 6 }),
      svg("text", { x: left + w / 2, y: top + 14.5, "text-anchor": "middle" }, text),
    );
  }

  function nearestAt(px, py, radius = 14) {
    if (!geometry) return null;
    let best = null;
    let bestDistance = radius * radius;
    for (const entry of wv.words) {
      const dx = geometry.positions[entry.index * 2] - px;
      const dy = geometry.positions[entry.index * 2 + 1] - py;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = entry;
      }
    }
    return best;
  }

  function local(event) {
    const rect = map.svg.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  map.element.addEventListener("pointerdown", (event) => {
    if (!wv || event.button > 0 || event.target.closest(".bk-map-tools")) return;
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, cx: view.cx, cy: view.cy, moved: false };
  });
  map.element.addEventListener("pointermove", (event) => {
    if (!wv) return;
    if (drag && drag.id === event.pointerId) {
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) > 4 && view.zoom > 1.0001) {
        drag.moved = true;
        animation = null;
        map.element.setPointerCapture(event.pointerId);
        map.element.classList.add("panning");
      }
      if (drag.moved && geometry) {
        Object.assign(view, clampView({ cx: drag.cx - dx / geometry.k, cy: drag.cy + dy / geometry.k, zoom: view.zoom }));
        requestRedraw();
        return;
      }
    }
    if (event.pointerType === "touch") return;
    const [px, py] = local(event);
    const found = nearestAt(px, py);
    if (found !== hover) {
      hover = found;
      renderHover();
    }
  });
  const endDrag = (event) => {
    if (!drag || drag.id !== event.pointerId) return;
    const wasClick = !drag.moved && event.type === "pointerup";
    drag = null;
    map.element.classList.remove("panning");
    if (!wasClick || !wv || event.target.closest(".bk-map-tools")) return;
    const [px, py] = local(event);
    const found = nearestAt(px, py, event.pointerType === "touch" ? 22 : 14);
    if (found) select(found);
    else if (selected) select(null);
  };
  map.element.addEventListener("pointerup", endDrag);
  map.element.addEventListener("pointercancel", endDrag);
  map.element.addEventListener("pointerleave", () => {
    if (hover) {
      hover = null;
      renderHover();
    }
  });
  map.element.addEventListener("dblclick", (event) => {
    if (!wv || event.target.closest(".bk-map-tools")) return;
    const [px, py] = local(event);
    zoomBy(2, px, py);
  });

  /* side panel */

  function wordChip(entry, onclick) {
    return h("button", { class: "chip bk-word-chip", onclick }, dotSwatch(categoryColor(wv, entry)), entry.word);
  }

  function renderSuggestions() {
    clear(suggestions);
    const query = search.value.trim().toLowerCase();
    if (!query || !wv) return;
    const list = wv.suggestions(query, 6);
    if (!list.length) {
      const close = wv.similarSpelling(query, 4);
      suggestions.append(h("span", { class: "bk-suggest-none" }, `“${query}” is not among these 479 words.${close.length ? " Did you mean:" : ""}`));
      for (const entry of close) suggestions.append(wordChip(entry, () => select(entry, { reveal: true })));
      return;
    }
    for (const entry of list) suggestions.append(wordChip(entry, () => select(entry, { reveal: true })));
  }

  search.addEventListener("input", renderSuggestions);
  search.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !wv) return;
    const query = search.value.trim().toLowerCase();
    const entry = wv.get(query) || wv.suggestions(query, 1)[0] || wv.similarSpelling(query, 1)[0];
    if (entry) select(entry, { reveal: true });
  });

  function renderDetail() {
    clear(detail);
    if (!wv) return;
    if (!selected) {
      detail.append(
        h("p", { class: "bk-map-hint" }, "Click any dot, or search for a word, to see which words sit closest to it."),
        h("div", { class: "bk-suggest" }, TRY_WORDS.filter((word) => wv.has(word)).map((word) => wordChip(wv.get(word), () => select(wv.get(word), { reveal: true })))),
      );
      return;
    }
    const category = wv.categories[selected.category];
    const vector = wv.vectors[selected.index];
    let farthest = null;
    for (const entry of wv.words) {
      const score = dot(vector, wv.vectors[entry.index]);
      if (!farthest || score < farthest.score) farthest = { word: entry.word, score };
    }
    const first = neighbours[0];
    detail.append(
      h("div", { class: "bk-map-word" },
        h("span", { class: "bk-map-word-text" }, selected.word),
        h("span", { class: "bk-map-cat" }, dotSwatch(categoryColor(wv, selected)), category.label),
      ),
      h("div", { class: "panel-label bk-section-label" }, "Closest in all 50 dimensions"),
      h("div", { class: "bk-bars bk-neighbours" }, neighbours.map((entry, index) => h("button", {
        class: ["bk-bar", index === 0 ? "top" : ""],
        title: `Select ${entry.word}`,
        onclick: () => select(wv.get(entry.word), { reveal: true }),
      },
      h("span", { class: "bk-bar-text" }, entry.word),
      h("span", { class: "bk-bar-track" }, h("span", { class: "bk-bar-fill", style: { width: `${Math.max(0, entry.score) * 100}%` } })),
      h("span", { class: "bk-bar-pct" }, entry.score.toFixed(2)),
      ))),
      h("div", { class: "panel-label bk-section-label bk-strip-label" }, "The numbers themselves"),
      h("div", { class: "bk-strip-pair" },
        h("span", { class: "bk-strip-name" }, selected.word), strip(vector, { label: selected.word, tone: "dark" }),
        h("span", { class: "bk-strip-name" }, first.word), strip(wv.vectors[first.index], { label: first.word, tone: "dark" }),
      ),
      h("p", { class: "bk-map-foot" }, `Cosine similarity ${first.score.toFixed(2)}. Least alike on this map: ${farthest.word} (${farthest.score.toFixed(2)}).`),
    );
  }

  return {
    element: fig.element,
    ready(data) {
      wv = data;
      const xs = wv.words.map((entry) => entry.x);
      const ys = wv.words.map((entry) => entry.y);
      bounds = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
      bounds.w = bounds.x1 - bounds.x0;
      bounds.h = bounds.y1 - bounds.y0;
      Object.assign(view, homeView());
      map.element.querySelector(".bk-map-status").remove();
      zoomOut.disabled = true;
      home.disabled = true;
      wv.categories.forEach((category, index) => {
        legend.append(h("button", {
          class: "chip bk-legend-chip",
          dataset: { index: String(index) },
          "aria-pressed": "false",
          title: `Highlight ${category.label.toLowerCase()}`,
          onclick: () => setFocus(index),
        }, dotSwatch(CATEGORY_COLORS[category.id] || "#b7c1d6"), category.label));
      });
      select(wv.get(FIRST_WORD));
    },
    fail() {
      const status = map.element.querySelector(".bk-map-status");
      if (status) clear(status).append(failure());
    },
  };
}

/* ---------- Figure 3: analogies ---------- */

function analogyFigure(life) {
  const fig = figure({
    number: 3,
    title: "Arithmetic with meanings",
    info: "a − b + c ≈ ?",
    wide: true,
    caption: "Type three words from the map, or pick an example. The figure computes a − b + c and lists the words whose vectors point most nearly the same way as the result, leaving out the three you typed. The diagram shows the flat slice of the 50-dimensional space that contains your three words: the orange arrow from b to a is copied so that it starts at c, and its tip is the target. Real words never land exactly on the target; the diagram shows their shadows on this slice.",
  });
  const marker = `bk-arrow-${(uid += 1)}`;
  let wv = null;
  let current = null;
  const fields = ["first word", "word to subtract", "word to add"].map((label, index) => h("input", {
    class: "text-input bk-analogy-input",
    type: "text",
    value: [PRESETS[0].a, PRESETS[0].b, PRESETS[0].c][index],
    spellcheck: "false",
    autocomplete: "off",
    "aria-label": label,
  }));
  const operator = (text) => h("span", { class: "bk-analogy-op", "aria-hidden": "true" }, text);
  const answer = h("span", { class: "bk-analogy-answer", "aria-live": "polite" }, "…");
  const presetButtons = PRESETS.map((preset) => h("button", {
    class: "chip",
    onclick: () => {
      [preset.a, preset.b, preset.c].forEach((word, index) => {
        fields[index].value = word;
      });
      compute();
    },
  }, `${preset.a} − ${preset.b} + ${preset.c}`, preset.tag ? h("span", { class: "bk-preset-tag" }, preset.tag === "bias" ? "a bias" : "a miss") : null));
  const plot = surface({ height: 300, className: "bk-plane", label: "Diagram of the analogy", draw, life });
  const results = h("div", { class: "bk-bars bk-analogy-results" }, loading());
  const messages = h("div", { class: "bk-analogy-messages" });
  fig.body.append(
    h("div", { class: "bk-analogy-eq" }, fields[0], operator("−"), fields[1], operator("+"), fields[2], operator("≈"), answer),
    h("div", { class: "bk-analogy-presets" }, presetButtons),
    h("div", { class: "bk-analogy-main" },
      h("div", { class: "bk-analogy-plot" }, plot.element),
      h("div", {}, h("div", { class: "panel-label bk-section-label" }, "Closest words to the target"), results, messages),
    ),
  );
  const timer = life.timer();
  for (const field of fields) {
    field.addEventListener("input", () => timer.set(compute, 280));
    field.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        timer.clear();
        compute();
      }
    });
  }

  function plane(words, ranked) {
    const [A, B, C] = words.map((word) => wv.vector(word));
    const ab = A.map((value, index) => value - B[index]);
    const abLength = norm(ab);
    if (abLength < 1e-6) return null;
    const u = ab.map((value) => value / abLength);
    const cb = C.map((value, index) => value - B[index]);
    const along = dot(cb, u);
    const w = cb.map((value, index) => value - along * u[index]);
    const wLength = norm(w);
    if (wLength < 1e-6) return null;
    const v = w.map((value) => value / wLength);
    const project = (X) => {
      const d = X.map((value, index) => value - B[index]);
      return [dot(d, u), dot(d, v)];
    };
    return {
      a: project(A),
      b: [0, 0],
      c: project(C),
      t: [abLength + along, wLength],
      results: ranked.slice(0, 3).map((entry) => ({ word: entry.word, point: project(wv.vectors[entry.index]) })),
    };
  }

  function compute() {
    if (!wv) return;
    const words = fields.map((field) => field.value.trim().toLowerCase());
    const bad = words.map((word) => !wv.has(word));
    fields.forEach((field, index) => field.classList.toggle("bk-bad", bad[index]));
    presetButtons.forEach((button, index) => {
      const preset = PRESETS[index];
      button.classList.toggle("active", preset.a === words[0] && preset.b === words[1] && preset.c === words[2]);
    });
    clear(results);
    clear(messages);
    const badIndex = bad.indexOf(true);
    if (badIndex >= 0) {
      const word = words[badIndex];
      const options = wv.similarSpelling(word, 3).concat(wv.suggestions(word, 3));
      const unique = [...new Set(options.map((entry) => entry.word))].slice(0, 3);
      current = null;
      answer.textContent = "?";
      answer.classList.add("empty");
      messages.append(note(word
        ? `“${word}” is not one of the 479 words on the map.${unique.length ? ` Did you mean ${unique.length === 1 ? unique[0] : `${unique.slice(0, -1).join(", ")} or ${unique[unique.length - 1]}`}?` : " Try words like king, paris or walked."}`
        : "Fill in all three words.", { tone: "warn" }));
      plot.redraw();
      return;
    }
    const target = wv.offset(...words);
    const exclude = new Set(words);
    const ranked = wv.nearest(target, 5, exclude);
    const overall = wv.nearest(target, 1)[0];
    current = { words, ranked, coords: plane(words, ranked) };
    answer.textContent = ranked[0].word;
    answer.classList.remove("empty");
    results.append(...ranked.map((entry, index) => h("div", { class: ["bk-bar", index === 0 ? "top" : ""] },
      h("span", { class: "bk-bar-text" }, entry.word),
      h("span", { class: "bk-bar-track" }, h("span", { class: "bk-bar-fill", style: { width: `${Math.max(0, entry.score) * 100}%` } })),
      h("span", { class: "bk-bar-pct" }, entry.score.toFixed(2)),
    )));
    if (exclude.has(overall.word)) {
      messages.append(note(`Scores are cosine similarities with the target. If the three input words were allowed, “${overall.word}” itself would come first (${overall.score.toFixed(2)}).`));
    }
    const preset = PRESETS.find((item) => item.a === words[0] && item.b === words[1] && item.c === words[2]);
    if (preset && preset.tag) messages.append(note(PRESET_NOTES[preset.tag], { tone: "warn" }));
    if (!current.coords) messages.append(note("Pick three different words to see the diagram."));
    plot.redraw();
  }

  function draw(root, width, height) {
    if (!current || !current.coords) {
      root.append(svg("text", { class: "bk-plane-empty", x: width / 2, y: height / 2, "text-anchor": "middle" }, wv ? "Choose three words from the map" : ""));
      return;
    }
    const { a, b, c, t, results: shadows } = current.coords;
    const [wordA, wordB, wordC] = current.words;
    const points = [a, b, c, t, ...shadows.map((item) => item.point)];
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs);
    const y0 = Math.min(...ys);
    const y1 = Math.max(...ys);
    const padX = 64;
    const padY = 34;
    const k = Math.min((width - padX * 2) / Math.max(x1 - x0, 1e-6), (height - padY * 2) / Math.max(y1 - y0, 1e-6));
    const sx = (x) => width / 2 + (x - (x0 + x1) / 2) * k;
    const sy = (y) => height / 2 - (y - (y0 + y1) / 2) * k;
    const P = (point) => [sx(point[0]), sy(point[1])];

    root.append(svg("defs", {}, svg("marker", { id: marker, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
      svg("path", { d: "M0,0 L10,5 L0,10 z", class: "bk-plane-head" }))));
    const segment = (from, to, className, withHead = false) => {
      const [fx, fy] = P(from);
      const [tx, ty] = P(to);
      const length = Math.hypot(tx - fx, ty - fy) || 1;
      const shorten = withHead ? 7 : 0;
      return svg("line", {
        class: className,
        x1: fx,
        y1: fy,
        x2: tx - ((tx - fx) / length) * shorten,
        y2: ty - ((ty - fy) / length) * shorten,
        "marker-end": withHead ? `url(#${marker})` : null,
      });
    };
    const dotAt = (point, className, r) => {
      const [x, y] = P(point);
      return svg("circle", { class: className, cx: x, cy: y, r });
    };
    const best = shadows[0];
    const others = shadows.slice(1);
    root.append(
      segment(b, c, "bk-plane-side"),
      segment(a, t, "bk-plane-side dashed"),
      segment(b, a, "bk-plane-arrow", true),
      segment(c, t, "bk-plane-arrow copy", true),
      segment(t, best.point, "bk-plane-gap"),
      ...others.map((item) => dotAt(item.point, "bk-plane-other", 4)),
      dotAt(b, "bk-plane-word", 5),
      dotAt(a, "bk-plane-word", 5),
      dotAt(c, "bk-plane-word", 5),
      dotAt(t, "bk-plane-target", 8),
      dotAt(best.point, "bk-plane-best", 6),
    );

    // Greedy label placement: try eight spots around each point, preferring the side facing
    // away from the figure, and keep the first spot that hits no line, dot or earlier label.
    const lines = [[b, c], [a, t], [b, a], [c, t], [t, best.point]].map(([from, to]) => [P(from), P(to)]);
    const boxes = [[b, 6], [a, 6], [c, 6], [t, 9], [best.point, 7], ...others.map((item) => [item.point, 5])].map(([point, r]) => {
      const [x, y] = P(point);
      return { x0: x - r, y0: y - r, x1: x + r, y1: y + r };
    });
    const crossesLine = (box) => lines.some(([[fx, fy], [tx, ty]]) => {
      for (let i = 0; i <= 24; i += 1) {
        const x = fx + ((tx - fx) * i) / 24;
        const y = fy + ((ty - fy) * i) / 24;
        if (x > box.x0 && x < box.x1 && y > box.y0 && y < box.y1) return true;
      }
      return false;
    });
    const blocked = (box) => box.x0 < 2 || box.x1 > width - 2 || box.y0 < 2 || box.y1 > height - 2
      || boxes.some((o) => box.x0 < o.x1 && box.x1 > o.x0 && box.y0 < o.y1 && box.y1 > o.y0)
      || crossesLine(box);
    const center = P([(a[0] + b[0] + c[0] + t[0]) / 4, (a[1] + b[1] + c[1] + t[1]) / 4]);
    const SPOTS = [[1, 0], [-1, 0], [0, 1], [0, -1], [0.75, 0.75], [0.75, -0.75], [-0.75, 0.75], [-0.75, -0.75]];
    const place = (point, text, className, { from = center, gap = 9, size = 12, required = true } = {}) => {
      const [x, y] = P(point);
      let ax = x - from[0];
      let ay = y - from[1];
      const length = Math.hypot(ax, ay);
      [ax, ay] = length > 1 ? [ax / length, ay / length] : [0.7, 0.7];
      const textWidth = text.length * size * 0.62 + 2;
      const options = SPOTS
        .map(([ux, uy]) => {
          const px = x + ux * gap;
          const py = y + uy * gap;
          const anchor = ux > 0.3 ? "start" : ux < -0.3 ? "end" : "middle";
          const left = anchor === "start" ? px : anchor === "end" ? px - textWidth : px - textWidth / 2;
          const top = uy > 0.3 ? py : uy < -0.3 ? py - size - 1 : py - size / 2 - 1;
          return { anchor, px, baseline: top + size - 2, box: { x0: left, y0: top, x1: left + textWidth, y1: top + size + 1 }, score: ux * ax + uy * ay };
        })
        .sort((p, q) => q.score - p.score);
      const choice = options.find((option) => !blocked(option.box)) || (required ? options[0] : null);
      if (!choice) return;
      boxes.push(choice.box);
      root.append(svg("text", { class: className, x: choice.px, y: choice.baseline, "text-anchor": choice.anchor }, text));
    };
    place(b, wordB, "bk-plane-label");
    place(a, wordA, "bk-plane-label");
    place(c, wordC, "bk-plane-label");
    place(t, "target", "bk-plane-target-label", { gap: 12 });
    place(best.point, best.word, "bk-plane-best-label", { from: P(t), gap: 10 });
    for (const item of others) place(item.point, item.word, "bk-plane-other-label", { from: P(t), gap: 7, size: 11, required: false });
  }

  return {
    element: fig.element,
    ready(data) {
      wv = data;
      compute();
    },
    fail() {
      clear(results).append(failure());
    },
  };
}

/* ---------- chapter ---------- */

export default {
  id: "embeddings",
  title: "Embeddings",
  render(root, ctx) {
    const life = lifecycle();
    const shell = chapterShell(root, "embeddings",
      "A token id is a name tag, not a description: the number 4,711 says nothing about what the token means. So the first thing a language model does with every id is trade it for a list of numbers called an embedding. Training arranges those lists so that tokens used in similar ways end up with similar numbers, and that is where meaning begins.");
    const { article } = shell;
    const lookup = lookupFigure();
    const map = mapFigure(life);
    const analogies = analogyFigure(life);

    article.append(prose(`
      <h2>Names are not meanings</h2>
      <p>The previous chapter turned text into a list of token ids. Those ids are arbitrary. They record the order in which the tokenizer happened to learn its pieces, so neighbouring ids need not have anything in common: id 4,711 might be <em>umbrella</em> and id 4,712 the middle of somebody's surname. Doing arithmetic on ids would be like averaging two phone numbers and expecting to reach a person halfway between the two callers.</p>
      <p>So every language model begins with a lookup. It owns a table, the <strong>embedding table</strong>, with one row for every token in its vocabulary. Each row is a list of numbers of a fixed length, called the model's <strong>width</strong> and often written <em>d</em>. Reading token 4,711 means fetching row 4,711. That row, the token's <strong>embedding</strong>, is all the rest of the model ever sees; the id itself is never used again.</p>
    `));
    article.append(lookup.element);
    article.append(prose(`
      <p>A list of numbers like this is called a <strong>vector</strong>. With two numbers you could draw it as an arrow on squared paper: so far to the right, so far up. With 50, 128 or 768 numbers no drawing is possible, but the geometry carries over unchanged. Vectors still have a length and a direction, and two of them can point roughly the same way or in completely different ways. That is the central trick of embeddings: <em>meaning becomes direction</em>.</p>

      <h2>Learned, not designed</h2>
      <p>Nobody writes these numbers by hand. When training starts, every row of the table is random noise. Training, the subject of the chapter after next, then adjusts every number in the model a tiny step at a time, always in whichever direction makes the next-token predictions slightly better. The rows of the embedding table are adjusted along with everything else.</p>
      <p>That single pressure has a remarkable side effect. Suppose <em>Tuesday</em> and <em>Thursday</em> tend to be followed by the same kinds of words: <em>morning</em>, <em>evening</em>, <em>at noon</em>. Then an adjustment that improves the prediction after one also improves it after the other, so both rows get nudged the same way, again and again, until they sit close together. Nobody tells the model that both are days of the week. It discovers that treating them alike pays off.</p>
      <p>The idea that a word can be understood through the words around it is older than computers; linguists in the 1950s called it the <em>distributional hypothesis</em>. Embeddings put it to work on a very large scale.</p>
    `));
    article.append(callout("note", `The figures in this chapter use <strong>GloVe</strong>, a set of word vectors published in 2014 by Jeffrey Pennington, Richard Socher and Christopher Manning at Stanford and dedicated to the public domain. They were learned from about six billion words of Wikipedia and newswire text by a method that is not a transformer: it fits vectors so that their dot products reflect how often two words occur near each other. The result behaves much like the embedding table of a trained language model, and at 50 numbers per word a few hundred words fit easily inside this book. We picked 479 everyday words from GloVe's vocabulary of 400,000.`, "Where these vectors come from"));
    article.append(prose(`
      <h2>How alike are two vectors?</h2>
      <p>The usual way to compare two embeddings is by the angle between them, expressed as their <strong>cosine similarity</strong>. Multiply the two lists entry by entry and add up the products (that sum is the <em>dot product</em>), then divide by the two lengths:</p>
      <div class="bk-eq">cos(<span class="bk-var">a</span>, <span class="bk-var">b</span>) = <span class="bk-frac"><span><span class="bk-var">a</span> · <span class="bk-var">b</span></span><span>‖<span class="bk-var">a</span>‖ ‖<span class="bk-var">b</span>‖</span></span><span class="bk-eq-note">a · b = a₁b₁ + a₂b₂ + … + a₅₀b₅₀, and ‖a‖ is the length of a: the square root of a · a</span></div>
      <p>The result is 1 for vectors that point the same way, 0 for vectors at right angles, and −1 for vectors that point in opposite directions. In GloVe, <em>king</em> and <em>queen</em> score 0.78, while <em>king</em> and <em>banana</em> score 0.22. The angle is used rather than the straight-line distance because the length of an embedding tends to carry other information, such as how common the word is, while its direction reflects how the word is used.</p>
      <p>Dot products are not only a measuring tool for curious readers. They are the basic operation inside a transformer: attention compares tokens with dot products, and the final layer scores every candidate next token with one. A model that points related tokens in similar directions has made all of that work easier.</p>

      <h2>A map of meaning</h2>
      <p>Fifty dimensions cannot be drawn, so Figure 2 flattens them. A standard technique called <strong>principal component analysis</strong> finds the two directions along which these 479 vectors spread out the most and places each word by how far it reaches along each one. It is like photographing the shadow of a wire sculpture: the overall shape survives, but points that are far apart in space can fall on top of each other in the shadow. That is why the neighbour lists in the figure are computed from all 50 numbers, and why a few neighbours turn up surprisingly far away on the map.</p>
    `));
    article.append(map.element);
    article.append(prose(`
      <p>Some things worth discovering on the map:</p>
      <ul>
        <li><strong>Tight families.</strong> Numbers, weekdays and months form dense clumps; <em>seven</em> and <em>five</em> score 0.99. In the contexts where they appear, these words are almost interchangeable.</li>
        <li><strong>Opposites are neighbours.</strong> <em>hot</em> and <em>cold</em> score 0.80. They mean opposite things but turn up in the same sentences (“the water was too …”), so their vectors end up close. <em>Similar</em> here means <em>used alike</em>, not <em>means the same</em>.</li>
        <li><strong>Senses are blended.</strong> Each word gets one vector, whichever meaning a sentence intends. <em>orange</em> sits among the colours rather than the fruit, and the closest neighbours of <em>apple</em> include <em>software</em> and <em>computer</em>, a trace of how often news text mentions a certain company.</li>
        <li><strong>Style leaves a mark.</strong> The nearest neighbours of <em>happy</em> are <em>i</em>, <em>good</em>, <em>think</em> and <em>you</em>. It is a word of chatty, first-person sentences, and its vector remembers the company it keeps.</li>
      </ul>

      <h2>Directions that mean something</h2>
      <p>Because training only cares about how words are used, a difference in usage that recurs across many words tends to become a consistent direction in the space. The step from <em>man</em> to <em>king</em> resembles the step from <em>woman</em> to <em>queen</em>; the step from <em>walk</em> to <em>walked</em> resembles the step from <em>swim</em> to <em>swam</em>. That makes possible a famous party trick: take the vector for <em>king</em>, subtract <em>man</em>, add <em>woman</em>, and ask which word lies closest to the result.</p>
    `));
    article.append(analogies.element);
    article.append(prose(`
      <p>The trick works more often than you might expect, for grammar (<em>bigger − big + small</em> gives <em>smaller</em>) as well as geography (<em>moscow − russia + egypt</em> gives <em>cairo</em>). Still, treat it as a demonstration rather than a law. By convention the three input words are left out of the answer; without that rule, <em>king − man + woman</em> would mostly return <em>king</em> itself, because the target is still closer to it than to anything else. And misses are common: <em>president − man + woman</em> lands on <em>spoke</em>. The impressive examples you see quoted were usually picked from many attempts.</p>
    `));
    article.append(callout("note", `Try <em>doctor − man + woman</em> in Figure 3. The answer is <em>nurse</em>. The vectors did not invent that; they learned it from the way their training text, years of news and encyclopedia articles, wrote about men and women at work. A language model learns from text in exactly the same way, so it absorbs the same associations and can repeat them in what it writes. That is one reason the choice of training data (Chapter 6) and the behaviour training of Chapter 8 matter so much.`, "Vectors absorb their text"));
    article.append(prose(`
      <h2>Embeddings inside a transformer</h2>
      <p>A language model's embedding table plays the same role as GloVe's vectors: it gives each token a starting point in a space where useful similarities are cheap to compute. There are three important differences.</p>
      <ul>
        <li><strong>Context changes the vector.</strong> A GloVe vector is fixed: <em>orange</em> gets the same numbers in “an orange sky” and in “orange juice”. In a transformer the lookup is only the first step. Every layer then mixes in information from the surrounding tokens (that mixing is called attention, and it is the subject of the next chapter), so by the top of the model the two oranges have moved apart. These are often called <em>contextual</em> embeddings.</li>
        <li><strong>Position is added.</strong> A lookup says nothing about word order: <em>dog bites man</em> and <em>man bites dog</em> fetch exactly the same rows. So the models in this workbench own a second, smaller table with one learned row per position in the context window, and add the row for position 0, 1, 2 and so on to each token's row before anything else happens.</li>
        <li><strong>The table is used twice.</strong> At the very end the model must score every token in the vocabulary as a possible next token. It does so by taking the dot product of its final vector with every row of the same embedding table, so tokens whose rows point the same way as the final vector get high scores. Reusing the table like this is called <em>weight tying</em>. It saves parameters, and it means each row is trained from both ends: as an input and as an answer.</li>
      </ul>
    `));
    article.append(callout("linux", `The <strong>Project setup</strong> page draws your model's blueprint, and these embedding rows are its first stage. The width slider sets how many numbers each token gets: 128 in the Pocket preset, 256 in Starter and 512 in Large. Wider rows can hold finer distinctions but cost memory and time; the table alone holds vocabulary × width numbers, about half a million for Pocket with a 4K tokenizer. Your model learns its table from your own pre-training data, so expect sensible neighbours for the words your corpus uses often and something closer to noise for the rare ones.`, "In the workbench"));
    article.append(callout("key", "An embedding turns a token id into a direction in space. Nobody designs those directions: training arranges them so that tokens used alike point alike, and that lets every later layer treat similar tokens in similar ways."));
    finishChapter(shell, ctx, { id: "anatomy-of-a-transformer", title: "Anatomy of a Transformer", blurb: "Attention, MLPs and the residual stream: what happens to the vectors after the lookup." });

    loadWordVectors()
      .then((data) => {
        if (life.disposed) return;
        lookup.ready(data);
        map.ready(data);
        analogies.ready(data);
      })
      .catch((error) => {
        if (life.disposed) return;
        console.warn("Embeddings chapter: could not load word vectors", error);
        lookup.fail();
        map.fail();
        analogies.fail();
      });

    return () => life.dispose();
  },
};
