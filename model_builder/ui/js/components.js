// Shared widgets used by the shell, workbench pages and chapters.
import { h, svg, fmt, clear, clamp, tokenLabel } from "./dom.js";
import { icon } from "./icons.js";

/* ---------- toasts, menus, modals ---------- */

export function toast(message, kind = "info", timeout = 4500) {
  const host = document.getElementById("toasts");
  if (!host) return;
  const element = h("div", { class: `toast ${kind}` }, message);
  host.append(element);
  setTimeout(() => element.remove(), timeout);
}

let openMenuElement = null;

function closeMenu() {
  if (openMenuElement) {
    openMenuElement.remove();
    openMenuElement = null;
  }
}

function placeNear(element, anchor, { align = "start", gap = 6 } = {}) {
  const rect = anchor.getBoundingClientRect();
  document.body.append(element);
  const width = element.offsetWidth;
  const height = element.offsetHeight;
  let left = align === "end" ? rect.right - width : rect.left;
  let top = rect.bottom + gap;
  if (top + height > window.innerHeight - 8) top = Math.max(8, rect.top - height - gap);
  left = clamp(left, 8, window.innerWidth - width - 8);
  element.style.left = `${left}px`;
  element.style.top = `${top}px`;
}

export function menu(anchor, items, options = {}) {
  closeMenu();
  const element = h("div", { class: "menu", role: "menu" });
  for (const item of items) {
    if (item === "-") {
      element.append(h("div", { class: "menu-sep" }));
    } else if (item.title) {
      element.append(h("div", { class: "menu-title" }, item.title));
    } else {
      element.append(h("button", {
        role: "menuitem",
        disabled: Boolean(item.disabled),
        style: item.danger ? { color: "var(--red)" } : null,
        onclick: () => {
          closeMenu();
          if (item.action) item.action();
        },
      },
      h("span", { style: { width: "14px", display: "inline-flex" } }, item.icon ? icon(item.icon, 14) : null),
      h("span", { class: "grow" }, item.label),
      item.detail ? h("span", { class: "muted", style: { fontSize: "11px" } }, item.detail) : null));
    }
  }
  placeNear(element, anchor, options);
  openMenuElement = element;
  setTimeout(() => {
    const dismiss = (event) => {
      if (!element.contains(event.target)) {
        closeMenu();
        document.removeEventListener("mousedown", dismiss, true);
      }
    };
    document.addEventListener("mousedown", dismiss, true);
  });
  return element;
}

export function modal({ title, body, actions = [{ label: "OK", primary: true }], onClose, width }) {
  return new Promise((resolve) => {
    const backdrop = h("div", { class: "modal-backdrop" });
    const close = (value) => {
      backdrop.remove();
      document.removeEventListener("keydown", onKey, true);
      if (onClose) onClose(value);
      resolve(value);
    };
    const onKey = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close(null);
      } else if (event.key === "Enter" && !(event.target instanceof HTMLTextAreaElement)) {
        const primary = actions.find((action) => action.primary);
        if (primary) {
          event.preventDefault();
          close(primary.value !== undefined ? primary.value : true);
        }
      }
    };
    const box = h("div", { class: "modal", role: "dialog", "aria-modal": "true", style: width ? { width } : null },
      title ? h("h2", {}, title) : null,
      body,
      h("div", { class: "modal-actions" }, actions.map((action) => h("button", {
        class: ["btn", action.primary ? "primary" : "", action.danger ? "danger" : ""],
        onclick: () => close(action.value !== undefined ? action.value : Boolean(action.primary)),
      }, action.label))),
    );
    backdrop.append(box);
    backdrop.addEventListener("mousedown", (event) => {
      if (event.target === backdrop) close(null);
    });
    document.addEventListener("keydown", onKey, true);
    document.body.append(backdrop);
    const focusable = box.querySelector("input, textarea, select, .btn.primary");
    if (focusable) setTimeout(() => focusable.focus(), 30);
  });
}

export async function confirm({ title, message, confirm: label = "OK", danger = false }) {
  const result = await modal({
    title,
    body: h("p", { class: "muted" }, message || ""),
    actions: [{ label: "Cancel", value: false }, { label, primary: !danger, danger, value: true }],
  });
  return result === true;
}

export async function prompt({ title, message, value = "", confirm: label = "OK", placeholder = "" }) {
  const input = h("input", { class: "text-input", value, placeholder, style: { marginTop: "10px" } });
  setTimeout(() => input.select(), 40);
  const result = await modal({
    title,
    body: h("div", {}, message ? h("p", { class: "muted" }, message) : null, input),
    actions: [{ label: "Cancel", value: false }, { label, primary: true, value: true }],
  });
  return result === true ? input.value.trim() : null;
}

/* ---------- small controls ---------- */

export function statusPill(label, kind = "") {
  return h("span", { class: `status-pill ${kind}` }, h("span", { class: "status-dot" }), label);
}

export function segmented(options, value, onChange, { className = "" } = {}) {
  const element = h("div", { class: `segmented ${className}`, role: "tablist" });
  const buttons = new Map();
  for (const option of options) {
    const button = h("button", {
      role: "tab",
      disabled: Boolean(option.disabled),
      title: option.title || "",
      onclick: () => {
        if (element.value === option.value) return;
        element.set(option.value);
        if (onChange) onChange(option.value);
      },
    }, option.icon ? icon(option.icon, 13) : null, option.label);
    buttons.set(option.value, button);
    element.append(button);
  }
  element.set = (next) => {
    element.value = next;
    for (const [key, button] of buttons) button.classList.toggle("active", key === next);
  };
  element.set(value);
  return element;
}

export function fillRange(input) {
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const percent = max > min ? ((Number(input.value) - min) / (max - min)) * 100 : 0;
  input.style.setProperty("--fill", `${percent}%`);
}

export function slider({ min = 0, max = 100, step = 1, value = 0, onInput, onChange, disabled = false, label }) {
  const input = h("input", { type: "range", min, max, step, value, disabled, "aria-label": label || "" });
  fillRange(input);
  input.addEventListener("input", () => {
    fillRange(input);
    if (onInput) onInput(Number(input.value));
  });
  input.addEventListener("change", () => {
    if (onChange) onChange(Number(input.value));
  });
  input.setValue = (next) => {
    input.value = next;
    fillRange(input);
  };
  return input;
}

const NICE_MANTISSAS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8];

export function niceRound(value) {
  if (value <= 0) return 0;
  const exponent = Math.floor(Math.log10(value));
  const base = 10 ** exponent;
  let best = base;
  for (const mantissa of [...NICE_MANTISSAS, 10]) {
    const candidate = mantissa * base;
    if (Math.abs(candidate - value) < Math.abs(best - value)) best = candidate;
  }
  return Math.round(best);
}

// Logarithmic slider with labelled ticks, snapping to friendly numbers (used for step targets).
export function logSlider({ min, max, value, ticks = [], format = fmt.compact, onInput, onChange, disabled = false }) {
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  const toPosition = (number) => ((Math.log10(clamp(number, min, max)) - lo) / (hi - lo)) * 1000;
  const fromPosition = (position) => niceRound(10 ** (lo + (position / 1000) * (hi - lo)));
  const tickHost = h("div", { class: "ticks" });
  const input = slider({
    min: 0, max: 1000, step: 1, value: toPosition(value), disabled,
    onInput: (position) => {
      element.value = clamp(fromPosition(position), min, max);
      markTicks();
      if (onInput) onInput(element.value);
    },
    onChange: () => {
      if (onChange) onChange(element.value);
    },
  });
  const element = h("div", { class: "tick-slider" }, input, tickHost);
  element.value = value;
  const tickLabels = ticks.map((tick) => {
    const label = h("span", {
      style: { left: `${toPosition(tick) / 10}%` },
      onclick: () => {
        if (input.disabled) return;
        element.set(tick);
        if (onInput) onInput(tick);
        if (onChange) onChange(tick);
      },
    }, format(tick));
    tickHost.append(label);
    return [tick, label];
  });
  function markTicks() {
    for (const [tick, label] of tickLabels) label.classList.toggle("active", tick === element.value);
  }
  element.set = (next) => {
    element.value = next;
    input.setValue(toPosition(next));
    markTicks();
  };
  element.setDisabled = (flag) => {
    input.disabled = flag;
  };
  markTicks();
  return element;
}

export function stepper({ values, value, format = String, onChange, disabled = false }) {
  let index = Math.max(0, values.indexOf(value));
  if (values.indexOf(value) < 0) {
    index = values.findIndex((candidate) => candidate >= value);
    if (index < 0) index = values.length - 1;
  }
  const display = h("span", { class: "stepper-value" });
  const minus = h("button", { "aria-label": "Decrease", onclick: () => move(-1) }, icon("minus", 12));
  const plus = h("button", { "aria-label": "Increase", onclick: () => move(1) }, icon("plus", 12));
  const element = h("div", { class: "stepper" }, minus, display, plus);
  function render() {
    element.value = values[index];
    display.textContent = format(values[index]);
    minus.disabled = element.disabled || index <= 0;
    plus.disabled = element.disabled || index >= values.length - 1;
  }
  function move(delta) {
    const next = clamp(index + delta, 0, values.length - 1);
    if (next === index) return;
    index = next;
    render();
    if (onChange) onChange(values[index]);
  }
  element.disabled = disabled;
  element.set = (next) => {
    const found = values.indexOf(next);
    if (found >= 0) index = found;
    render();
  };
  element.setDisabled = (flag) => {
    element.disabled = flag;
    render();
  };
  render();
  return element;
}

export function settingRow(title, help, ...controls) {
  return h("div", { class: "setting-row" },
    h("div", { class: "setting-text" }, h("div", { class: "setting-title" }, title), help ? h("div", { class: "setting-help" }, help) : null),
    h("div", { class: "setting-control" }, controls),
  );
}

export function meter(fraction = 0, { light = false } = {}) {
  const fill = h("div");
  const element = h("div", { class: `meter ${light ? "light" : ""}` }, fill);
  element.set = (value) => {
    fill.style.width = `${clamp(value || 0, 0, 1) * 100}%`;
  };
  element.set(fraction);
  return element;
}

export function stat(label, value = "—", kind = "") {
  const valueElement = h("div", { class: `value ${kind}` }, value);
  const element = h("div", { class: "stat" }, h("span", { class: "panel-label" }, label), valueElement);
  element.set = (next) => {
    valueElement.textContent = next;
  };
  return element;
}

/* ---------- special-token aware text ---------- */

const SPECIAL = /(<\|[a-z_]+\|>)/g;
const ROLE_TOKENS = new Set(["<|system|>", "<|user|>", "<|assistant|>"]);

export function specialText(text, { roleClass = "role" } = {}) {
  const fragment = document.createDocumentFragment();
  for (const part of String(text).split(SPECIAL)) {
    if (!part) continue;
    if (/^<\|[a-z_]+\|>$/.test(part)) {
      fragment.append(h("span", { class: `special-chip ${ROLE_TOKENS.has(part) ? roleClass : ""}` }, part));
    } else {
      fragment.append(document.createTextNode(part));
    }
  }
  return fragment;
}

/* ---------- token chips and alternatives popover ---------- */

let openPopover = null;

export function closePopover() {
  if (openPopover) {
    openPopover.remove();
    openPopover = null;
  }
}

export function alternativesPopover(anchor, token, { onPick, title = "What was possible here" } = {}) {
  closePopover();
  const alternatives = token.alternatives && token.alternatives.length ? token.alternatives : [{ id: token.id, text: token.text, prob: token.prob }];
  const maxProb = Math.max(...alternatives.map((alternative) => alternative.prob), 1e-9);
  const rows = alternatives.map((alternative) => {
    const chosen = alternative.id === token.id;
    const row = h(onPick ? "button" : "div", {
      class: `alt-row ${chosen ? "chosen" : ""}`,
      title: onPick && !chosen ? "Continue from here with this token" : "",
      onclick: onPick && !chosen ? () => {
        closePopover();
        onPick(alternative);
      } : null,
    },
    h("span", { class: "alt-text" }, tokenLabel(alternative.text)),
    h("span", { class: "alt-bar" }, h("div", { style: { width: `${(alternative.prob / maxProb) * 100}%` } })),
    h("span", { class: "alt-pct" }, formatProb(alternative.prob)));
    return row;
  });
  const element = h("div", { class: "popover", role: "dialog" },
    h("div", { class: "panel-label", style: { marginBottom: "8px" } }, title),
    rows,
    onPick ? h("div", { class: "panel-note", style: { marginTop: "8px", opacity: 0.8 } }, "Pick another token to branch the text from this point.") : null,
  );
  placeNear(element, anchor);
  openPopover = element;
  setTimeout(() => {
    const dismiss = (event) => {
      if (!element.contains(event.target) && event.target !== anchor) {
        closePopover();
        document.removeEventListener("mousedown", dismiss, true);
        document.removeEventListener("keydown", onKey, true);
      }
    };
    const onKey = (event) => {
      if (event.key === "Escape") {
        closePopover();
        document.removeEventListener("mousedown", dismiss, true);
        document.removeEventListener("keydown", onKey, true);
      }
    };
    document.addEventListener("mousedown", dismiss, true);
    document.addEventListener("keydown", onKey, true);
  });
  return element;
}

export function formatProb(prob) {
  if (!Number.isFinite(prob)) return "—";
  if (prob >= 0.995) return "100%";
  if (prob >= 0.1) return `${Math.round(prob * 100)}%`;
  if (prob >= 0.01) return `${(prob * 100).toFixed(1)}%`;
  return "<1%";
}

export function tokenChip(token, { light = false, prompt = false, onPick, onSelect, popoverTitle } = {}) {
  const special = /^<\|[a-z_]+\|>$/.test(token.text);
  const chip = h("span", {
    class: ["token", light ? "light" : "", prompt ? "prompt" : "", special ? "special" : ""],
    style: { "--p": prompt ? 1 : clamp(Number.isFinite(token.prob) ? Math.sqrt(token.prob) : 1, 0, 1) },
    title: prompt ? "" : `${formatProb(token.prob)} likely`,
    tabindex: prompt ? null : 0,
  }, tokenLabel(token.text));
  if (!prompt) {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".token.selected").forEach((node) => node.classList.remove("selected"));
      chip.classList.add("selected");
      if (onSelect) onSelect(token, chip);
      alternativesPopover(chip, token, popoverTitle ? { onPick, title: popoverTitle } : { onPick });
    });
  }
  return chip;
}

/* ---------- loss chart ---------- */

function niceStep(range, target) {
  const raw = range / Math.max(1, target);
  const exponent = Math.floor(Math.log10(raw || 1));
  const base = 10 ** exponent;
  for (const multiplier of [1, 2, 2.5, 5, 10]) {
    if (raw <= multiplier * base) return multiplier * base;
  }
  return 10 * base;
}

function downsample(points, limit) {
  if (points.length <= limit) return points;
  const stride = points.length / limit;
  const result = [];
  for (let index = 0; index < limit; index += 1) result.push(points[Math.floor(index * stride)]);
  result.push(points[points.length - 1]);
  return result;
}

export function lossChart({ height = 260, emptyText = "The loss curve appears here as soon as training starts." } = {}) {
  const element = h("div", { class: "chart", style: { height: `${height}px` } });
  const surface = svg("svg", { role: "img", "aria-label": "Loss curve" });
  const empty = h("div", { class: "chart-empty" }, emptyText);
  const bubble = h("div", { class: "chart-bubble", hidden: true });
  const tooltip = h("div", { class: "chart-tooltip", hidden: true });
  element.append(surface, empty, bubble, tooltip);
  let data = { train: [], val: [], target: 0 };
  let geometry = null;
  let pinned = null;

  function render() {
    clear(surface);
    const width = element.clientWidth || 600;
    const chartHeight = element.clientHeight || height;
    surface.setAttribute("viewBox", `0 0 ${width} ${chartHeight}`);
    const { train, val } = data;
    empty.hidden = train.length > 0;
    bubble.hidden = train.length === 0;
    if (!train.length) {
      geometry = null;
      return;
    }
    const pad = { left: 30, right: 74, top: 8, bottom: 26 };
    const lastStep = train[train.length - 1][0];
    const xMax = Math.max(lastStep * 1.06, 10);
    const values = [...train.map((point) => point[1]), ...val.map((point) => point[1])].filter(Number.isFinite);
    const top = Math.max(...values, 1);
    const yStep = niceStep(top, 4);
    const yMax = Math.ceil(top / yStep) * yStep;
    const x = (step) => pad.left + (step / xMax) * (width - pad.left - pad.right);
    const y = (loss) => pad.top + (1 - loss / yMax) * (chartHeight - pad.top - pad.bottom);
    geometry = { x, y, pad, width, chartHeight, xMax };

    const grid = svg("g", { class: "grid" });
    const axis = svg("g", { class: "axis" });
    for (let tick = 0; tick <= yMax + 1e-9; tick += yStep) {
      grid.append(svg("line", { x1: pad.left, x2: width - pad.right + 40, y1: y(tick), y2: y(tick) }));
      axis.append(svg("text", { x: pad.left - 8, y: y(tick) + 3.5, "text-anchor": "end" }, Number(tick.toFixed(2)).toString()));
    }
    const xStep = niceStep(xMax, 4);
    for (let tick = 0; tick <= xMax; tick += xStep) {
      axis.append(svg("line", { x1: x(tick), x2: x(tick), y1: chartHeight - pad.bottom, y2: chartHeight - pad.bottom + 12 }));
      axis.append(svg("text", { x: x(tick) + 5, y: chartHeight - pad.bottom + 14 }, fmt.int(tick)));
    }
    axis.append(svg("line", { x1: pad.left, x2: pad.left, y1: pad.top, y2: chartHeight - pad.bottom }));
    axis.append(svg("line", { x1: pad.left, x2: width - pad.right + 40, y1: chartHeight - pad.bottom, y2: chartHeight - pad.bottom }));
    surface.append(grid, axis);

    const path = (points) => downsample(points, Math.max(200, Math.floor(width))).map((point, index) => `${index ? "L" : "M"}${x(point[0]).toFixed(1)},${y(point[1]).toFixed(1)}`).join("");
    surface.append(svg("line", { class: "cursor-line", x1: x(lastStep), x2: x(lastStep), y1: pad.top, y2: chartHeight - pad.bottom }));
    if (val.length > 1) surface.append(svg("path", { class: "val-line", d: path(val) }));
    else if (val.length === 1) surface.append(svg("circle", { cx: x(val[0][0]), cy: y(val[0][1]), r: 2.5, fill: "var(--val)" }));
    surface.append(svg("path", { class: "train-line", d: path(train) }));
    const last = train[train.length - 1];
    surface.append(svg("circle", { cx: x(last[0]), cy: y(last[1]), r: 4, fill: "var(--train)", stroke: "#0d1a35", "stroke-width": 1.5 }));
    bubble.textContent = fmt.loss(last[1]);
    bubble.style.left = `${x(last[0])}px`;
    bubble.style.top = `${y(last[1]) - 14}px`;
    if (pinned) showTooltip(pinned, true);
  }

  function nearest(points, step) {
    if (!points.length) return null;
    let low = 0;
    let high = points.length - 1;
    while (low < high) {
      const mid = (low + high) >> 1;
      if (points[mid][0] < step) low = mid + 1;
      else high = mid;
    }
    const candidates = [points[low], points[Math.max(0, low - 1)]];
    return candidates.reduce((best, point) => (Math.abs(point[0] - step) < Math.abs(best[0] - step) ? point : best));
  }

  function showTooltip(step, keep) {
    if (!geometry) return;
    const trainPoint = nearest(data.train, step);
    const valPoint = nearest(data.val, step);
    if (!trainPoint) return;
    const hover = surface.querySelector(".hover-line") || surface.appendChild(svg("line", { class: "cursor-line hover-line" }));
    const xPosition = geometry.x(trainPoint[0]);
    hover.setAttribute("x1", xPosition);
    hover.setAttribute("x2", xPosition);
    hover.setAttribute("y1", geometry.pad.top);
    hover.setAttribute("y2", geometry.chartHeight - geometry.pad.bottom);
    clear(tooltip);
    tooltip.append(
      h("div", { class: "panel-label", style: { marginBottom: "3px" } }, `Step ${fmt.int(trainPoint[0])}${keep ? " · pinned" : ""}`),
      h("div", {}, h("span", { style: { color: "var(--train)" } }, "train "), fmt.loss(trainPoint[1])),
      valPoint && Math.abs(valPoint[0] - trainPoint[0]) <= Math.max(1, geometry.xMax * 0.03)
        ? h("div", {}, h("span", { style: { color: "var(--val)" } }, "val "), fmt.loss(valPoint[1]), h("span", { class: "panel-muted" }, ` @ ${fmt.int(valPoint[0])}`))
        : null,
    );
    tooltip.hidden = false;
    const left = xPosition + 12 + tooltip.offsetWidth > geometry.width ? xPosition - tooltip.offsetWidth - 12 : xPosition + 12;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${geometry.pad.top + 4}px`;
  }

  function stepAt(event) {
    if (!geometry) return null;
    const rect = element.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const ratio = (px - geometry.pad.left) / (geometry.width - geometry.pad.left - geometry.pad.right);
    return clamp(ratio, 0, 1) * geometry.xMax;
  }

  element.addEventListener("mousemove", (event) => {
    if (pinned !== null) return;
    const step = stepAt(event);
    if (step !== null) showTooltip(step, false);
  });
  element.addEventListener("mouseleave", () => {
    if (pinned !== null) return;
    tooltip.hidden = true;
    const hover = surface.querySelector(".hover-line");
    if (hover) hover.remove();
  });
  element.addEventListener("click", (event) => {
    const step = stepAt(event);
    if (step === null) return;
    pinned = pinned === null ? step : null;
    if (pinned === null) {
      tooltip.hidden = true;
      const hover = surface.querySelector(".hover-line");
      if (hover) hover.remove();
    } else {
      showTooltip(pinned, true);
    }
  });

  const observer = new ResizeObserver(() => render());
  observer.observe(element);
  element.update = (train, val, target = 0) => {
    data = { train: train || [], val: val || [], target };
    render();
  };
  element.destroy = () => observer.disconnect();
  return element;
}

/* ---------- dataset tiles ---------- */

export function datasetTile(info, { selected = false, downloaded = false, active = false, onClick } = {}) {
  const state = downloaded
    ? h("span", { class: "tile-state ready", title: "Downloaded" }, icon("circle-check", 16))
    : h("span", { class: "tile-state", title: "Downloads when you use it" }, icon("download", 15));
  return h("button", { class: ["tile", selected ? "selected" : ""], onclick: onClick, dataset: { id: info.id } },
    h("span", { class: "tile-icon", style: { background: info.color || "var(--accent)" } }, icon(info.icon || "file", 18)),
    h("span", { class: "grow" },
      h("div", { class: "tile-title" }, info.name, active ? h("span", { class: "muted", style: { fontWeight: 500 } }, " · in use") : null),
      h("div", { class: "tile-meta" }, info.size_label || ""),
    ),
    state,
  );
}

/* ---------- misc ---------- */

export function sigmaLine(text) {
  return h("div", { class: "sigma-line" }, h("span", { class: "sigma" }, "∑"), text);
}

export function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const area = h("textarea", { style: { position: "fixed", opacity: 0 } }, text);
  document.body.append(area);
  area.select();
  document.execCommand("copy");
  area.remove();
  return Promise.resolve();
}
