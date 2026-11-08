// Small DOM and formatting helpers shared by the shell, pages, and chapters.
const SVG_NS = "http://www.w3.org/2000/svg";

function setAttributes(element, attributes, isSvg) {
  for (const [key, value] of Object.entries(attributes || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") {
      element.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "class" || key === "className") {
      const name = Array.isArray(value) ? value.filter(Boolean).join(" ") : value;
      if (isSvg) element.setAttribute("class", name);
      else element.className = name;
    } else if (key === "style" && typeof value === "object") {
      for (const [property, setting] of Object.entries(value)) {
        if (setting === null || setting === undefined) continue;
        if (property.startsWith("--")) element.style.setProperty(property, setting);
        else element.style[property] = setting;
      }
    } else if (key === "dataset") {
      Object.assign(element.dataset, value);
    } else if (key === "html") {
      element.innerHTML = value;
    } else if (!isSvg && key in element && typeof value !== "string") {
      element[key] = value;
    } else {
      element.setAttribute(key, value === true ? "" : String(value));
    }
  }
}

function appendChildren(element, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false || child === true) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function h(tag, attributes = {}, ...children) {
  const element = document.createElement(tag);
  setAttributes(element, attributes, false);
  appendChildren(element, children);
  return element;
}

export function svg(tag, attributes = {}, ...children) {
  const element = document.createElementNS(SVG_NS, tag);
  setAttributes(element, attributes, true);
  appendChildren(element, children);
  return element;
}

export function clear(element) {
  while (element.firstChild) element.firstChild.remove();
  return element;
}

export function debounce(fn, wait = 200) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

export function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function compact(value) {
  const abs = Math.abs(value);
  const units = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "k"]];
  for (const [size, suffix] of units) {
    if (abs >= size) {
      const scaled = value / size;
      const digits = Math.abs(scaled) >= 100 ? 0 : 1;
      return `${scaled.toFixed(digits).replace(/\.0$/, "")}${suffix}`;
    }
  }
  return String(Math.round(value));
}

export const fmt = {
  int: (value) => Math.round(value).toLocaleString("en-US"),
  compact,
  params: (value) => `≈${compact(value)} params`,
  bytes(value) {
    if (!Number.isFinite(value)) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let index = 0;
    let scaled = value;
    while (scaled >= 1000 && index < units.length - 1) {
      scaled /= 1000;
      index += 1;
    }
    return `${scaled >= 100 || index === 0 ? Math.round(scaled) : scaled.toFixed(1)} ${units[index]}`;
  },
  duration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "—";
    const total = Math.round(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    if (hours >= 24) return `${Math.floor(hours / 24)}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
    if (minutes > 0) return `${minutes}m ${String(secs).padStart(2, "0")}s`;
    return `${secs}s`;
  },
  loss: (value) => (Number.isFinite(value) ? value.toFixed(3) : "—"),
  lr: (value) => (Number.isFinite(value) ? value.toExponential(1).replace(/e([+-])(\d)$/, "e$10$2") : "—"),
  pct: (value, digits = 1) => `${(value * 100).toFixed(digits)}%`,
  tokens: (value) => `${compact(value)} tok`,
};

export function tokenLabel(text) {
  // Show leading spaces as a middle dot and line breaks as a return glyph, like a token inspector.
  if (text === "\n") return "↵";
  return text.replace(/\n/g, "↵").replace(/^ /, "·").replace(/\t/g, "⇥");
}
