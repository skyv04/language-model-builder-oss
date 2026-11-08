// Small SVG plotting helpers for textbook figures (scales, ticks, axes, responsive surfaces).
import { h, svg } from "../dom.js";

export function linear(d0, d1, r0, r1) {
  const k = (r1 - r0) / (d1 - d0 || 1);
  const scale = (value) => r0 + (value - d0) * k;
  scale.invert = (pixel) => d0 + (pixel - r0) / k;
  scale.domain = [d0, d1];
  scale.range = [r0, r1];
  return scale;
}

export function logScale(d0, d1, r0, r1) {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  const k = (r1 - r0) / (l1 - l0 || 1);
  const scale = (value) => r0 + (Math.log10(Math.max(value, 1e-300)) - l0) * k;
  scale.invert = (pixel) => 10 ** (l0 + (pixel - r0) / k);
  scale.domain = [d0, d1];
  scale.range = [r0, r1];
  return scale;
}

export function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(raw || 1));
  const normalized = raw / magnitude;
  const factor = normalized < 1.5 ? 1 : normalized < 3 ? 2 : normalized < 7 ? 5 : 10;
  return factor * magnitude;
}

export function ticks(min, max, count = 5) {
  if (!(max > min)) return [min];
  const step = niceStep(max - min, count);
  const first = Math.ceil(min / step - 1e-9) * step;
  const result = [];
  for (let value = first; value <= max + step * 1e-6; value += step) result.push(Number(value.toPrecision(12)));
  return result;
}

// A responsive SVG whose drawing is rebuilt at the real pixel width whenever the container resizes.
export function surface({ height = 240, className = "", label = "", draw, life }) {
  const root = svg("svg", { class: ["bk-svg", className], role: "img", "aria-label": label, height });
  // A zero-height probe tracks the width only, so changing the height from draw() never re-triggers the observer.
  const probe = h("div", { class: "bk-svg-probe", "aria-hidden": "true" });
  const element = h("div", { class: "bk-svg-wrap", style: { height: `${height}px` } }, probe, root);
  const state = { width: 0, height };
  const redraw = () => {
    if (!state.width) return;
    while (root.firstChild) root.firstChild.remove();
    root.setAttribute("width", state.width);
    root.setAttribute("viewBox", `0 0 ${state.width} ${state.height}`);
    draw(root, state.width, state.height);
  };
  const observer = new ResizeObserver((entries) => {
    const width = Math.round(entries[0].contentRect.width);
    if (width > 0 && width !== state.width) {
      state.width = width;
      redraw();
    }
  });
  observer.observe(probe);
  if (life) life.add(() => observer.disconnect());
  return {
    element,
    svg: root,
    redraw,
    get width() {
      return state.width;
    },
    setHeight(next) {
      state.height = next;
      root.setAttribute("height", next);
      element.style.height = `${next}px`;
      redraw();
    },
    destroy: () => observer.disconnect(),
  };
}

// Grid lines plus tick labels for a plot area { left, right, top, bottom } in pixels.
export function axes(root, { x, y, xTicks = [], yTicks = [], xFormat = String, yFormat = String, area, xLabel = "", yLabel = "" }) {
  const grid = svg("g", { class: "bk-grid" });
  const axis = svg("g", { class: "bk-axis" });
  for (const value of yTicks) {
    const py = y(value);
    grid.append(svg("line", { x1: area.left, x2: area.right, y1: py, y2: py }));
    axis.append(svg("text", { x: area.left - 6, y: py, "text-anchor": "end", "dominant-baseline": "middle" }, yFormat(value)));
  }
  for (const value of xTicks) {
    const px = x(value);
    grid.append(svg("line", { class: "v", x1: px, x2: px, y1: area.top, y2: area.bottom }));
    axis.append(svg("text", { x: px, y: area.bottom + 15, "text-anchor": "middle" }, xFormat(value)));
  }
  axis.append(svg("line", { class: "base", x1: area.left, x2: area.right, y1: area.bottom, y2: area.bottom }));
  if (xLabel) axis.append(svg("text", { class: "label", x: area.right, y: area.bottom + 30, "text-anchor": "end" }, xLabel));
  if (yLabel) axis.append(svg("text", { class: "label", x: area.left - 6, y: area.top - 8, "text-anchor": "start" }, yLabel));
  root.append(grid, axis);
  return { grid, axis };
}

export function pathFrom(points) {
  let d = "";
  let pen = false;
  for (const [px, py] of points) {
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`;
    pen = true;
  }
  return d;
}

export function sampleFunction(fn, x0, x1, count = 160) {
  const points = [];
  for (let index = 0; index <= count; index += 1) {
    const value = x0 + ((x1 - x0) * index) / count;
    points.push([value, fn(value)]);
  }
  return points;
}
