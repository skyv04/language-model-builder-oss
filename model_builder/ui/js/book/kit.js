// Shared building blocks for the textbook chapters: page shell, prose, figures, callouts and lifecycle.
import { h, fmt } from "../dom.js";
import { icon } from "../icons.js";

export const CHAPTERS = [
  { id: "introduction", title: "Introduction", part: "Foundations" },
  { id: "what-is-a-language-model", title: "What is a language model?", part: "Foundations" },
  { id: "tokenization", title: "Tokenization", part: "Foundations" },
  { id: "embeddings", title: "Embeddings", part: "Foundations" },
  { id: "anatomy-of-a-transformer", title: "Anatomy of a Transformer", part: "How models work" },
  { id: "training-data", title: "Training data", part: "How models work" },
  { id: "how-models-learn", title: "How models learn", part: "How models work" },
  { id: "how-models-learn-behavior", title: "How models learn behavior", part: "How models work" },
];

// Collects everything a chapter starts (listeners on window/document, timers, animation loops,
// observers, in-flight requests) so a single dispose() on navigation stops all of it.
export function lifecycle() {
  const disposers = [];
  let disposed = false;
  const add = (fn) => {
    if (disposed) fn();
    else disposers.push(fn);
    return fn;
  };
  return {
    get disposed() {
      return disposed;
    },
    add,
    listen(target, type, handler, options) {
      target.addEventListener(type, handler, options);
      add(() => target.removeEventListener(type, handler, options));
    },
    // A single re-armable timeout: set() replaces any pending call.
    timer() {
      let id = null;
      const control = {
        set(fn, ms) {
          clearTimeout(id);
          id = null;
          if (disposed) return;
          id = setTimeout(() => {
            id = null;
            if (!disposed) fn();
          }, ms);
        },
        clear() {
          clearTimeout(id);
          id = null;
        },
        get pending() {
          return id !== null;
        },
      };
      add(() => control.clear());
      return control;
    },
    // requestAnimationFrame loop; step(time) returns false to stop by itself.
    loop(step, onStop) {
      let frameId = 0;
      let running = false;
      const halt = () => {
        if (!running) return;
        running = false;
        cancelAnimationFrame(frameId);
        if (onStop) onStop();
      };
      const frame = (time) => {
        if (!running) return;
        let keep;
        try {
          keep = step(time);
        } catch (error) {
          halt();
          throw error;
        }
        if (keep === false) {
          halt();
          return;
        }
        if (running) frameId = requestAnimationFrame(frame);
      };
      const control = {
        start() {
          if (running || disposed) return;
          running = true;
          frameId = requestAnimationFrame(frame);
        },
        stop: halt,
        get running() {
          return running;
        },
      };
      add(() => {
        running = false;
        cancelAnimationFrame(frameId);
      });
      return control;
    },
    observeResize(element, fn) {
      const observer = new ResizeObserver(fn);
      observer.observe(element);
      add(() => observer.disconnect());
      return observer;
    },
    dispose() {
      disposed = true;
      for (const fn of disposers.splice(0).reverse()) {
        try {
          fn();
        } catch (error) {
          console.error(error);
        }
      }
    },
  };
}

// Static, trusted chapter prose written as HTML.
export function prose(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content;
}

const PART_TONE = { Foundations: "foundations", "How models work": "models" };

export function chapterShell(root, id, lead) {
  const index = CHAPTERS.findIndex((chapter) => chapter.id === id);
  const chapter = CHAPTERS[index];
  root.classList.add("bk-page");
  const meta = h("div", { class: "bk-meta" });
  const article = h("article", { class: "book bk-chapter", dataset: { tone: PART_TONE[chapter.part] } },
    h("header", { class: "bk-head" },
      h("div", { class: "bk-kicker" }, h("span", {}, chapter.part), h("span", { class: "bk-kicker-sep" }), h("span", {}, `Chapter ${index + 1}`)),
      h("h1", {}, chapter.title),
      lead ? h("p", { class: "bk-lead", html: lead }) : null,
      meta,
    ),
  );
  root.append(article);
  return { article, chapter, index, meta };
}

// Adds the reading-time line under the lead and the closing "Next" card.
export function finishChapter(shell, ctx, next) {
  const words = (shell.article.textContent || "").split(/\s+/).filter(Boolean).length;
  const figures = shell.article.querySelectorAll(".bk-figure").length;
  shell.meta.append(
    h("span", { class: "bk-meta-item" }, icon("clock", 13), `${Math.max(1, Math.round(words / 230))} min read`),
    figures ? h("span", { class: "bk-meta-item" }, icon("pointer", 13), `${figures} interactive figure${figures === 1 ? "" : "s"}`) : null,
  );
  shell.article.append(nextCard(ctx, next));
}

export function nextCard(ctx, { id, title, label = "Next chapter", blurb = "" }) {
  return h("nav", { class: "bk-next-wrap", "aria-label": "Continue reading" },
    h("button", { class: "bk-next", onclick: () => ctx.navigate(id) },
      h("span", { class: "bk-next-text" },
        h("span", { class: "bk-next-label" }, label),
        h("span", { class: "bk-next-title" }, `Next: ${title}`),
        blurb ? h("span", { class: "bk-next-blurb" }, blurb) : null,
      ),
      h("span", { class: "bk-next-arrow" }, icon("chevron-right", 20)),
    ),
  );
}

const CALLOUTS = {
  key: { label: "Key idea", icon: "sparkles" },
  linux: { label: "On your machine", icon: "cpu" },
  try: { label: "Try this", icon: "pointer" },
  note: { label: "A closer look", icon: "info" },
};

export function callout(kind, html, label) {
  const spec = CALLOUTS[kind] || CALLOUTS.note;
  return h("aside", { class: `bk-callout ${kind}` },
    h("div", { class: "bk-callout-label" }, icon(spec.icon, 13), label || spec.label),
    h("div", { class: "bk-callout-body", html }),
  );
}

// A numbered figure: a dark instrument panel (default) or a light card, plus a caption.
export function figure({ number, title, info = "", caption = "", wide = false, tone = "dark" }) {
  const dark = tone === "dark";
  const infoElement = h("span", { class: dark ? "panel-label bk-info" : "bk-label bk-info" }, info);
  const head = h("div", { class: "bk-panel-head" },
    h("span", { class: dark ? "panel-label" : "bk-label" }, title),
    h("span", { class: "spacer" }),
    infoElement,
  );
  const body = h("div", { class: "bk-panel-body" });
  const panel = h("div", { class: dark ? "panel-dark bk-panel" : "bk-card bk-panel" }, head, body);
  const element = h("figure", { class: ["bk-figure", wide ? "wide" : "", dark ? "dark" : "light"] },
    panel,
    caption ? h("figcaption", {}, h("b", {}, `Figure ${number}.`), h("span", { html: ` ${caption}` })) : null,
  );
  return {
    element,
    panel,
    head,
    body,
    setInfo(text) {
      infoElement.textContent = text;
    },
  };
}

// Labelled control: title + live value, control underneath.
export function control(label, input, { value = "", hint = "" } = {}) {
  const valueElement = h("span", { class: "bk-control-value" }, value);
  const element = h("div", { class: "bk-control" },
    h("div", { class: "bk-control-head" }, h("span", { class: "bk-control-label" }, label), valueElement),
    input,
    hint ? h("div", { class: "bk-control-hint" }, hint) : null,
  );
  element.setValue = (text) => {
    valueElement.textContent = text;
  };
  return element;
}

export function toggleChip(label, active, onChange, { title = "" } = {}) {
  const chip = h("button", { class: ["chip", "bk-toggle", active ? "active" : ""], title: title || null, "aria-pressed": String(active) }, label);
  chip.addEventListener("click", () => {
    const next = !chip.classList.contains("active");
    chip.classList.toggle("active", next);
    chip.setAttribute("aria-pressed", String(next));
    onChange(next);
  });
  return chip;
}

export function note(content, { tone = "info", action = null, iconName = null } = {}) {
  return h("div", { class: `bk-note ${tone}` },
    icon(iconName || (tone === "warn" ? "alert" : "info"), 14),
    h("span", { class: "grow" }, content),
    action,
  );
}

// Deterministic PRNG (mulberry32) so demos are reproducible.
export function seededRandom(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleIndex(probs, random = Math.random) {
  let r = random();
  for (let index = 0; index < probs.length; index += 1) {
    r -= probs[index];
    if (r <= 0) return index;
  }
  return probs.length - 1;
}

export function softmax(values, temperature = 1) {
  const t = Math.max(temperature, 1e-6);
  let max = -Infinity;
  for (const value of values) max = Math.max(max, value / t);
  const exps = values.map((value) => Math.exp(value / t - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

export function percent(value, digits = 1) {
  if (!Number.isFinite(value)) return "—";
  if (value > 0 && value < 0.001) return "<0.1%";
  return `${(value * 100).toFixed(digits)}%`;
}

// Friendly text for engine/API failures. Returns { kind, text }.
export function describeApiError(error) {
  const status = error && error.status;
  if (status === 409) return { kind: "not-ready", text: (error && error.message) || "The project is not ready for this yet." };
  if (status === 404 || status === 501 || status === 405) return { kind: "unavailable", text: "The training engine is not available in this window." };
  if (!status) return { kind: "offline", text: "Could not reach the training engine." };
  return { kind: "error", text: (error && error.message) || `The engine answered with an error (${status}).` };
}

export { fmt };
