// Application shell: sidebar, toolbar, router, reading pane and progress tracking.
import { api, events } from "./api.js";
import { h, svg, fmt, clear, tokenLabel } from "./dom.js";
import { icon } from "./icons.js";
import * as ui from "./components.js";

export const NAV = [
  { id: "welcome", title: "Welcome", icon: "sparkles", tone: "welcome" },
  {
    section: "Foundations", tone: "foundations", kind: "chapter", items: [
      { id: "introduction", title: "Introduction", icon: "hand" },
      { id: "what-is-a-language-model", title: "What is a language model?", icon: "question" },
      { id: "tokenization", title: "Tokenization", icon: "type" },
      { id: "embeddings", title: "Embeddings", icon: "scatter" },
    ],
  },
  {
    section: "How models work", tone: "models", kind: "chapter", items: [
      { id: "anatomy-of-a-transformer", title: "Anatomy of a Transformer", icon: "layers" },
      { id: "training-data", title: "Training data", icon: "database" },
      { id: "how-models-learn", title: "How models learn", icon: "trending-down" },
      { id: "how-models-learn-behavior", title: "How models learn behavior", icon: "drama" },
    ],
  },
  {
    section: "Build your model", tone: "build", kind: "page", items: [
      { id: "project-setup", title: "Project setup", icon: "folder-cog" },
      { id: "pretraining-data", title: "Pre-training data", icon: "file-down", counted: false },
      { id: "pretraining", title: "Pre-training", icon: "brain" },
      { id: "sampling", title: "Sampling", icon: "dice" },
      { id: "finetuning-data", title: "Fine-tuning data", icon: "list-checks", counted: false },
      { id: "sft", title: "Supervised fine-tuning", icon: "graduation" },
      { id: "dpo", title: "Direct preference optimization", icon: "thumbs-up" },
      { id: "chat", title: "Chat with your model", icon: "messages" },
    ],
  },
  { section: "About", tone: "about", kind: "page", items: [{ id: "acknowledgements", title: "Acknowledgements", icon: "heart" }] },
];

const PAGES = new Map();
const ORDER = [];
for (const entry of NAV) {
  if (!entry.section) {
    PAGES.set(entry.id, { ...entry, kind: "page", sectionTitle: null });
    ORDER.push(entry.id);
    continue;
  }
  for (const item of entry.items) {
    PAGES.set(item.id, { ...item, kind: entry.kind, tone: entry.tone, sectionTitle: entry.section });
    ORDER.push(item.id);
  }
}
const COUNTED = ORDER.filter((id) => id !== "welcome" && PAGES.get(id).counted !== false);

let appState = null;
let guides = {};
let currentId = null;
let renderToken = 0;
let pageScope = [];
const stateListeners = new Set();
const chrome = {};

const $ = (selector) => document.querySelector(selector);

function settings() {
  return (appState && appState.settings) || {};
}

function completedPages() {
  return new Set(settings().completed || []);
}

export async function refreshState() {
  try {
    appState = await api.get("/api/state");
  } catch (error) {
    ui.toast(`Could not reach the engine: ${error.message}`, "error");
    return appState;
  }
  renderSidebarDynamic();
  updateToolbar();
  for (const fn of [...stateListeners]) {
    try {
      fn(appState);
    } catch (error) {
      console.error(error);
    }
  }
  return appState;
}

async function saveSettings(patch) {
  try {
    const result = await api.post("/api/settings", patch);
    if (appState) appState.settings = result.settings;
  } catch (error) {
    ui.toast(error.message, "error");
  }
}

export function navigate(id) {
  if (!PAGES.has(id)) id = "welcome";
  history.replaceState(null, "", `#/${id}`);
  renderPage(id);
}

// The toolbar arrows step through the pages in sidebar order, like turning the pages of a book.
function neighbour(offset) {
  const index = ORDER.indexOf(currentId) + offset;
  return index >= 0 && index < ORDER.length ? ORDER[index] : null;
}

function goPrevious() {
  const id = neighbour(-1);
  if (id) navigate(id);
}

function goNext() {
  const id = neighbour(1);
  if (id) navigate(id);
}

function runCleanup() {
  for (const fn of pageScope.splice(0)) {
    try {
      fn();
    } catch (error) {
      console.error(error);
    }
  }
}

function makeContext(entry, scope) {
  return {
    api, events, h, svg, fmt, icon, ui, tokenLabel, page: entry,
    navigate: (id) => navigate(id),
    state: () => appState,
    refresh: refreshState,
    saveSettings,
    on(type, fn) {
      const off = events.on(type, fn);
      scope.push(off);
      return off;
    },
    onState(fn) {
      stateListeners.add(fn);
      scope.push(() => stateListeners.delete(fn));
    },
    addCleanup(fn) {
      scope.push(fn);
    },
    toast: ui.toast,
  };
}

function placeholder(root, entry, error) {
  root.append(
    h("div", { class: "empty-state" },
      icon(entry.icon || "info", 28),
      h("h3", {}, entry.title),
      h("p", {}, error ? `This page could not be loaded: ${error.message}` : "This page is not available yet."),
    ),
  );
}

async function renderPage(id) {
  const token = ++renderToken;
  runCleanup();
  const entry = PAGES.get(id);
  currentId = id;
  const content = $("#content");
  clear(content);
  content.scrollTop = 0;
  const root = h("div", { class: ["page", entry.kind === "chapter" ? "chapter-page" : `page-${id}`] });
  content.append(root);
  updateToolbar();
  renderSidebarDynamic();
  renderReadingPane();
  const folder = entry.kind === "chapter" ? "chapters" : "pages";
  let module;
  try {
    module = await import(`./${folder}/${id}.js`);
  } catch (error) {
    console.error(error);
    if (token === renderToken) placeholder(root, entry, error);
    return;
  }
  if (token !== renderToken) return;
  const scope = [];
  pageScope = scope;
  try {
    const result = await module.default.render(root, makeContext(entry, scope));
    if (typeof result === "function") {
      if (token === renderToken) scope.push(result);
      else result();
    }
  } catch (error) {
    console.error(error);
    if (token === renderToken) {
      clear(root);
      placeholder(root, entry, error);
    }
  }
}

/* ---------- sidebar ---------- */

function buildSidebar() {
  const sidebar = $("#sidebar");
  clear(sidebar);
  chrome.sidebarToggle = h("button", { class: "glass-btn", title: "Hide sidebar", "aria-label": "Hide sidebar", onclick: toggleSidebar }, icon("panel-left", 17));
  sidebar.append(h("div", { class: "sidebar-top" }, chrome.sidebarToggle));

  chrome.projectName = h("div", { class: "project-name" }, "No project");
  chrome.projectDot = h("span", { class: "status-dot" });
  chrome.projectStatus = h("span", {}, "Create a project to begin");
  chrome.projectButton = h("button", { class: "project-switcher", title: "Switch project", onclick: (event) => openProjectMenu(event.currentTarget) },
    h("span", { class: "project-badge" }, icon("box", 17)),
    h("span", { class: "project-text" }, chrome.projectName, h("span", { class: "project-status" }, chrome.projectDot, chrome.projectStatus)),
    icon("chevrons-up-down", 14),
  );
  sidebar.append(chrome.projectButton);

  const nav = h("nav", { class: "nav", "aria-label": "Sections" });
  chrome.navItems = new Map();
  const makeItem = (item, tone) => {
    const button = h("button", { class: "nav-item", dataset: { tone, page: item.id }, onclick: () => navigate(item.id) },
      h("span", { class: "nav-icon" }, icon(item.icon, 16)),
      h("span", { class: "label" }, item.title),
    );
    chrome.navItems.set(item.id, button);
    return button;
  };
  for (const entry of NAV) {
    if (!entry.section) {
      nav.append(makeItem(entry, entry.tone));
      continue;
    }
    nav.append(h("div", { class: "nav-section" },
      h("div", { class: "nav-section-title" }, entry.section),
      entry.items.map((item) => makeItem(item, entry.tone)),
    ));
  }
  sidebar.append(nav);

  chrome.progressValue = h("span", {}, "0%");
  chrome.progressFill = h("div");
  sidebar.append(h("div", { class: "sidebar-footer" },
    h("div", { class: "progress-meter" },
      h("div", { class: "progress-meter-head" }, h("span", {}, "Progress"), chrome.progressValue),
      h("div", { class: "progress-meter-bar" }, chrome.progressFill),
    ),
    h("button", { class: "btn ghost small icon-only", title: "Settings", "aria-label": "Settings", onclick: openSettings }, icon("sliders", 16)),
  ));
}

function currentProject() {
  if (!appState || !appState.current) return null;
  return (appState.projects || []).find((project) => project.id === appState.current) || null;
}

function renderSidebarDynamic() {
  if (!chrome.navItems) return;
  const done = completedPages();
  for (const [id, button] of chrome.navItems) {
    button.classList.toggle("active", id === currentId);
    const existing = button.querySelector(".done");
    if (done.has(id) && !existing) button.append(h("span", { class: "done", title: "Completed" }, icon("check", 13)));
    if (!done.has(id) && existing) existing.remove();
  }
  const finished = COUNTED.filter((id) => done.has(id)).length;
  const percent = Math.round((finished / COUNTED.length) * 100);
  chrome.progressValue.textContent = `${percent}%`;
  chrome.progressFill.style.width = `${percent}%`;

  const project = currentProject();
  chrome.projectName.textContent = project ? project.name : "No project yet";
  const status = project ? project.status || { label: "Not trained yet", tone: "grey" } : { label: "Create one to start building", tone: "grey" };
  chrome.projectStatus.textContent = status.label;
  chrome.projectDot.className = `status-dot ${status.tone || ""} ${status.pulse ? "pulse" : ""}`;
}

function toggleSidebar() {
  const app = $("#app");
  const hidden = !app.classList.contains("sidebar-hidden");
  app.classList.toggle("sidebar-hidden", hidden);
  chrome.toolbarSidebar.hidden = !hidden;
  saveSettings({ sidebar_hidden: hidden });
}

/* ---------- toolbar ---------- */

function buildToolbar() {
  const toolbar = $("#toolbar");
  clear(toolbar);
  chrome.toolbarSidebar = h("button", { class: "glass-btn", title: "Show sidebar", "aria-label": "Show sidebar", hidden: true, onclick: toggleSidebar }, icon("panel-left", 17));
  chrome.previous = h("button", { title: "Previous page", "aria-label": "Previous page", onclick: goPrevious }, icon("chevron-left", 18));
  chrome.next = h("button", { title: "Next page", "aria-label": "Next page", onclick: goNext }, icon("chevron-right", 18));
  chrome.crumbSection = h("span", { class: "section" });
  chrome.crumbSep = h("span", { class: "sep" }, "›");
  chrome.crumbPage = h("span", { class: "crumb-page" });
  chrome.crumbCount = h("span", { class: "count" });
  chrome.complete = h("button", { title: "Mark this page as complete", "aria-label": "Mark complete", onclick: toggleComplete }, icon("circle-check", 18));
  chrome.guide = h("button", { title: "Show the guide", "aria-label": "Toggle guide", onclick: toggleReadingPane }, icon("book-open", 18));
  toolbar.append(
    chrome.toolbarSidebar,
    h("div", { class: "glass-group" }, chrome.previous, chrome.next),
    h("div", { class: "breadcrumb" }, chrome.crumbSection, chrome.crumbSep, chrome.crumbPage, chrome.crumbCount),
    h("div", { class: "spacer" }),
    h("div", { class: "glass-group" }, chrome.complete, chrome.guide),
  );
  $("#content").addEventListener("scroll", (event) => {
    toolbar.classList.toggle("scrolled", event.target.scrollTop > 4);
  }, { passive: true });
}

function updateToolbar() {
  if (!chrome.crumbPage || !currentId) return;
  const entry = PAGES.get(currentId);
  chrome.crumbSection.textContent = entry.sectionTitle || "";
  chrome.crumbSection.hidden = !entry.sectionTitle;
  chrome.crumbSep.hidden = !entry.sectionTitle;
  chrome.crumbPage.textContent = entry.title;
  const index = COUNTED.indexOf(currentId);
  chrome.crumbCount.textContent = index >= 0 ? `${index + 1} of ${COUNTED.length}` : "";
  for (const [button, id, label] of [[chrome.previous, neighbour(-1), "Previous"], [chrome.next, neighbour(1), "Next"]]) {
    button.disabled = !id;
    button.title = id ? `${label}: ${PAGES.get(id).title}` : `${label} page`;
  }
  const done = completedPages().has(currentId);
  chrome.complete.disabled = currentId === "welcome";
  chrome.complete.classList.toggle("complete", done);
  chrome.complete.title = done ? "Marked complete (click to undo)" : "Mark this page as complete";
  const hasGuide = Boolean(guides[currentId]);
  chrome.guide.disabled = !hasGuide;
  chrome.guide.classList.toggle("on", hasGuide && Boolean(settings().reading_pane));
}

async function toggleComplete() {
  if (!currentId || currentId === "welcome") return;
  const done = completedPages();
  if (done.has(currentId)) done.delete(currentId);
  else done.add(currentId);
  if (appState) appState.settings = { ...settings(), completed: [...done] };
  renderSidebarDynamic();
  updateToolbar();
  await saveSettings({ completed: [...done] });
}

/* ---------- reading pane ---------- */

function renderReadingPane() {
  const pane = $("#reading-pane");
  const html = guides[currentId];
  const show = Boolean(html) && Boolean(settings().reading_pane);
  pane.hidden = !show;
  if (!show) return;
  pane.innerHTML = "";
  pane.append(h("h2", {}, "Guide"));
  const body = h("div", { class: "guide-body" });
  body.innerHTML = html;
  pane.append(body);
  pane.scrollTop = 0;
}

function toggleReadingPane() {
  const next = !settings().reading_pane;
  if (appState) appState.settings = { ...settings(), reading_pane: next };
  renderReadingPane();
  updateToolbar();
  saveSettings({ reading_pane: next });
}

/* ---------- project menu and settings ---------- */

function openProjectMenu(anchor) {
  const projects = (appState && appState.projects) || [];
  const items = [{ title: "Projects" }];
  for (const project of projects) {
    items.push({
      label: project.name,
      detail: project.status ? project.status.label : "",
      icon: project.id === appState.current ? "check" : null,
      action: () => switchProject(project.id),
    });
  }
  if (!projects.length) items.push({ label: "No projects yet", disabled: true });
  items.push("-");
  items.push({ label: "New project…", icon: "plus", action: () => newProjectDialog() });
  const project = currentProject();
  if (project) {
    items.push({ label: "Rename project…", icon: "pencil", action: () => renameProjectDialog(project) });
    items.push({ label: "Open project folder", icon: "folder-open", action: () => api.post("/api/projects/current/reveal").catch((error) => ui.toast(error.message, "error")) });
    items.push("-");
    items.push({ label: "Delete project…", icon: "trash", danger: true, action: () => deleteProjectDialog(project) });
  }
  ui.menu(anchor, items);
}

async function switchProject(id) {
  try {
    await api.post("/api/projects/select", { id });
    await refreshState();
    renderPage(currentId);
  } catch (error) {
    ui.toast(error.message, "error");
  }
}

export async function newProjectDialog() {
  const name = await ui.prompt({ title: "New project", message: "Give your model a name. You can change it later.", value: "My Model", confirm: "Create" });
  if (!name) return;
  try {
    await api.post("/api/projects", { name });
    await refreshState();
    navigate("project-setup");
  } catch (error) {
    ui.toast(error.message, "error");
  }
}

async function renameProjectDialog(project) {
  const name = await ui.prompt({ title: "Rename project", value: project.name, confirm: "Rename" });
  if (!name || name === project.name) return;
  try {
    await api.patch("/api/projects/current", { name });
    await refreshState();
  } catch (error) {
    ui.toast(error.message, "error");
  }
}

async function deleteProjectDialog(project) {
  const ok = await ui.confirm({
    title: `Delete “${project.name}”?`,
    message: "This removes the project folder, including its tokenizer, checkpoints and samples. Downloaded datasets are kept.",
    confirm: "Delete",
    danger: true,
  });
  if (!ok) return;
  try {
    await api.del(`/api/projects/${encodeURIComponent(project.id)}`);
    await refreshState();
    renderPage(currentId);
  } catch (error) {
    ui.toast(error.message, "error");
  }
}

function openSettings() {
  const state = appState || {};
  const current = settings();
  const devices = state.devices || [];
  const deviceChoice = ui.segmented(
    [{ value: "auto", label: "Automatic" }, ...devices.map((device) => ({ value: device.id, label: device.label || device.name }))],
    current.device || "auto",
    (value) => saveSettings({ device: value }),
  );
  const deviceList = h("div", { class: "stack", style: { gap: "6px", marginTop: "10px" } },
    devices.map((device) => h("div", { class: "row hint" }, icon(device.kind === "cpu" ? "cpu" : "zap", 14), h("span", {}, `${device.name}`), h("span", { class: "muted" }, device.detail || ""))),
  );
  const keepTraining = h("label", { class: "switch" },
    h("input", { type: "checkbox", checked: current.keep_training !== false, onchange: (event) => saveSettings({ keep_training: event.target.checked }) }),
    h("span"),
  );
  ui.modal({
    title: "Settings",
    body: h("div", { class: "stack", style: { gap: "4px" } },
      h("div", { class: "setting-row" },
        h("div", { class: "setting-text" }, h("div", { class: "setting-title" }, "Compute device"),
          h("div", { class: "setting-help" }, "Automatic picks the faster device for your model shape once you calibrate it in Project setup.")),
      ),
      deviceChoice,
      deviceList,
      h("div", { class: "setting-row", style: { marginTop: "10px" } },
        h("div", { class: "setting-text" }, h("div", { class: "setting-title" }, "Keep training after closing the window"),
          h("div", { class: "setting-help" }, "Runs continue in the background; reopen the app to check on them.")),
        keepTraining,
      ),
      h("div", { class: "setting-row" },
        h("div", { class: "setting-text" }, h("div", { class: "setting-title" }, "Data folder"),
          h("div", { class: "setting-help mono" }, state.app ? state.app.data_dir : "")),
        h("button", { class: "btn small", onclick: () => api.post("/api/reveal-data").catch((error) => ui.toast(error.message, "error")) }, "Open"),
      ),
      h("div", { class: "setting-row" },
        h("div", { class: "setting-text" }, h("div", { class: "setting-title" }, "Reading progress"),
          h("div", { class: "setting-help" }, "Clear the completed marks in the sidebar.")),
        h("button", { class: "btn small", onclick: async () => { await saveSettings({ completed: [] }); renderSidebarDynamic(); updateToolbar(); } }, "Reset"),
      ),
      h("p", { class: "hint", style: { marginTop: "12px" } },
        `Language Model Builder, open-source edition ${state.app ? state.app.version : ""} · engine ${state.app ? state.app.engine : ""}`),
    ),
    actions: [{ label: "Done", primary: true }],
  });
}

/* ---------- startup ---------- */

async function loadGuides() {
  try {
    guides = (await import("./guides.js")).guides || {};
  } catch {
    guides = {};
  }
}

function watchConnection() {
  let banner = null;
  let timer = null;
  events.on("connection", (message) => {
    if (message.connected) {
      clearTimeout(timer);
      if (banner) {
        banner.remove();
        banner = null;
        refreshState();
      }
      return;
    }
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (banner) return;
      banner = h("div", { class: "banner warn offline-banner" }, h("span", { class: "spinner" }), "Reconnecting to the training engine…");
      document.body.append(banner);
    }, 2500);
  });
  events.on("state", () => refreshState());
}

function bindKeys() {
  document.addEventListener("keydown", (event) => {
    const mod = event.ctrlKey || event.metaKey;
    if (event.altKey && event.key === "ArrowLeft") goPrevious();
    else if (event.altKey && event.key === "ArrowRight") goNext();
    else if (mod && event.key === "[") goPrevious();
    else if (mod && event.key === "]") goNext();
    else if (mod && event.key === "\\") toggleSidebar();
    else return;
    event.preventDefault();
  });
}

async function start() {
  buildSidebar();
  buildToolbar();
  bindKeys();
  watchConnection();
  await Promise.all([loadGuides(), refreshState()]);
  if (settings().sidebar_hidden) {
    $("#app").classList.add("sidebar-hidden");
    chrome.toolbarSidebar.hidden = false;
  }
  events.start();
  const requested = location.hash.replace(/^#\/?/, "");
  const initial = PAGES.has(requested) ? requested : settings().last_page && PAGES.has(settings().last_page) ? settings().last_page : "welcome";
  navigate(initial);
  window.addEventListener("hashchange", () => {
    const id = location.hash.replace(/^#\/?/, "");
    if (id && id !== currentId && PAGES.has(id)) navigate(id);
  });
  setInterval(() => {
    if (currentId && settings().last_page !== currentId) {
      if (appState) appState.settings = { ...settings(), last_page: currentId };
      saveSettings({ last_page: currentId });
    }
  }, 4000);
}

start();
