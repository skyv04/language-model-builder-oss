// Welcome: the front door — a hero, the way through the book and the workbench, the reader's progress,
// and the compute this machine offers.
import { NAV } from "../app.js";
import { clear } from "../dom.js";
import { loadProject } from "./shared.js";

const TONE = { foundations: "#f08a24", models: "#23a6c4", build: "var(--accent)", about: "#57575d", chat: "var(--purple)" };
const BLURB = {
  foundations: "What a language model is, and how text turns into numbers it can work with.",
  models: "Inside a transformer, and how training shapes what comes out of it.",
  build: "Train a tiny GPT of your own, from raw text to a model you can talk to.",
};

const SECTIONS = NAV.filter((entry) => entry.section && entry.tone !== "about");
const FLAT = NAV.flatMap((entry) => (entry.section ? entry.items.map((item) => ({ ...item, tone: entry.tone, kind: entry.kind })) : [entry]));
const TITLE = new Map(FLAT.map((item) => [item.id, item.title]));
const COUNTED = FLAT.filter((item) => item.id !== "welcome" && item.counted !== false);
// Counted pages that sit outside the three path columns (Acknowledgements), so the total can say where the rest come from.
const SHOWN = new Set(SECTIONS.flatMap((section) => section.items.map((item) => item.id)));
const EXTRA = COUNTED.filter((item) => !SHOWN.has(item.id));
const CHAPTERS = FLAT.filter((item) => item.kind === "chapter");

const hasCheckpoint = (project, stage) => project?.stages?.[stage]?.checkpoint_step != null;

// The build page that moves this project forward: the running job, otherwise the first stage without a checkpoint.
function nextBuildPage(project, job) {
  if (!project) return "project-setup";
  if (job && job.project === project.id) return { pretrain: "pretraining", sft: "sft", dpo: "dpo" }[job.kind] || "project-setup";
  if (!hasCheckpoint(project, "pretrain")) return project.stages?.pretrain?.ready_reason ? "pretraining-data" : "pretraining";
  if (!hasCheckpoint(project, "sft")) return /dataset/i.test(project.stages?.sft?.ready_reason || "") ? "finetuning-data" : "sft";
  if (!hasCheckpoint(project, "dpo")) return "dpo";
  return "chat";
}

// "GPU (Vulkan)" → "Vulkan"
const backendOf = (device) => (/\(([^)]+)\)/.exec(device.label || "") || [])[1] || "";

export default {
  id: "welcome",
  title: "Welcome",
  async render(root, ctx) {
    const { h, ui, icon } = ctx;
    root.classList.add("flush");
    let project; // undefined while loading, null when there is none
    let projectError = null;
    let probing = false;
    let timer = null;
    let disposed = false;

    const cards = h("div", { class: "wl-cards" });
    const path = h("section", { class: "card wl-path", "aria-label": "Your path" });
    const machine = h("section", { class: "panel-dark wl-machine", "aria-label": "This machine" });

    const hero = h("section", { class: "wl-hero" },
      h("div", { class: "wl-hero-inner" },
        h("span", { class: "wl-kicker" }, icon("sparkles", 12), "Open-source edition"),
        h("h1", { class: "wl-title" }, "Language Model Builder"),
        h("p", { class: "wl-pitch" },
          "Language models can feel like magic. They aren't. Find out how one turns text into numbers and numbers back into words, then build a small one yourself on this computer: choose what it reads, watch it learn, teach it to hold a conversation and talk to it."),
        h("div", { class: "wl-hero-actions" },
          h("button", { class: "btn primary large", onclick: () => ctx.navigate("introduction") }, icon("book-open", 15), "Start reading"),
          h("button", { class: "btn large wl-glass", onclick: () => ctx.navigate("project-setup") }, icon("wand", 15), "Build a model"))),
      h("span", { class: "wl-credit" }, "Caspar David Friedrich, ", h("em", {}, "Wanderer above the Sea of Fog"), ", c. 1818"));

    const note = h("p", { class: "wl-note" },
      icon("info", 13),
      h("span", {}, "This is an unofficial open-source edition: an independent re-implementation of Felix Rieseberg's Language Model Builder for the Mac, not made, reviewed or endorsed by him. "),
      h("button", { class: "link-btn", onclick: () => ctx.navigate("acknowledgements") }, "Credits and licences ›"));

    root.append(hero, h("div", { class: "wl-body" }, cards, h("div", { class: "wl-grid" }, path, machine), note));

    const completed = () => new Set(ctx.state()?.settings?.completed || []);

    /* ---------- the three doors ---------- */

    function door({ tone, iconName, title, subtitle, status, dot = "", pulse = false, target }) {
      return h("button", { class: "wl-card", style: { "--tone": TONE[tone] }, onclick: () => ctx.navigate(target) },
        h("span", { class: "wl-card-icon" }, icon(iconName, 20)),
        h("span", { class: "wl-card-go", "aria-hidden": "true" }, icon("chevron-right", 15)),
        h("strong", { class: "wl-card-title" }, title),
        h("span", { class: "wl-card-sub" }, subtitle),
        h("span", { class: "wl-card-status" }, h("span", { class: ["status-dot", dot, pulse ? "pulse" : ""] }), h("span", {}, status)));
    }

    function chatStatus() {
      if (project === undefined) return { status: "Checking your checkpoints…" };
      if (projectError) return { status: "Couldn't read the project", dot: "red" };
      if (!project) return { status: "Train a model first" };
      if (hasCheckpoint(project, "dpo")) return { status: "Preference-tuned model ready", dot: "green" };
      if (hasCheckpoint(project, "sft")) return { status: "Fine-tuned model ready", dot: "green" };
      if (hasCheckpoint(project, "pretrain")) return { status: "Base model only: not chat-trained yet", dot: "orange" };
      return { status: "No checkpoint yet" };
    }

    function paintCards() {
      const state = ctx.state() || {};
      const done = completed();
      const read = CHAPTERS.filter((item) => done.has(item.id)).length;
      const unread = CHAPTERS.find((item) => !done.has(item.id));
      const summary = (state.projects || []).find((item) => item.id === state.current) || null;
      const target = nextBuildPage(project || null, state.job);
      const status = summary?.status || { label: summary ? "Not trained yet" : "No project yet", tone: "" };
      clear(cards).append(
        door({
          tone: "foundations",
          iconName: "book-open",
          title: read ? "Keep reading" : "Learn the ideas",
          subtitle: unread ? (read ? `Up next: ${unread.title}` : `${CHAPTERS.length} short illustrated chapters, no maths degree needed`) : "Every chapter read. Revisit any of them from the sidebar.",
          status: `${read} of ${CHAPTERS.length} chapters read`,
          dot: read === CHAPTERS.length ? "green" : read ? "orange" : "",
          target: unread ? unread.id : CHAPTERS[0].id,
        }),
        door({
          tone: "build",
          iconName: "brain",
          title: summary ? "Keep building" : "Build your own model",
          subtitle: summary ? `${summary.name} · next: ${TITLE.get(target)}` : "Set up a project, choose its data and pre-train it",
          status: status.label,
          dot: status.tone === "grey" ? "" : status.tone || "",
          pulse: Boolean(status.pulse),
          target,
        }),
        door({
          tone: "chat",
          iconName: "messages",
          title: "Chat with your model",
          subtitle: "Talk to what you trained and watch it pick every word",
          target: "chat",
          ...chatStatus(),
        }));
    }

    /* ---------- the path through the app ---------- */

    function column(section, index, done) {
      const counted = section.items.filter((item) => item.counted !== false);
      const finished = counted.filter((item) => done.has(item.id)).length;
      return h("div", { class: "wl-col", style: { "--tone": TONE[section.tone] } },
        h("div", { class: "wl-col-head" },
          h("span", { class: "wl-col-num" }, String(index + 1)),
          h("strong", {}, section.section),
          h("span", { class: "wl-col-count" }, `${finished}/${counted.length}`)),
        h("p", { class: "wl-col-blurb" }, BLURB[section.tone] || ""),
        h("div", { class: "wl-steps" }, section.items.map((item) => {
          const isDone = done.has(item.id);
          return h("button", {
            class: ["wl-step", item.counted === false ? "minor" : "", isDone ? "done" : ""],
            title: isDone ? `${item.title} (done)` : item.title,
            onclick: () => ctx.navigate(item.id),
          },
          h("span", { class: "wl-step-icon" }, icon(item.icon, 14)),
          h("span", { class: "wl-step-title" }, item.title),
          isDone ? h("span", { class: "wl-step-check", "aria-label": "Done" }, icon("check", 12)) : null);
        })));
    }

    function paintPath() {
      const done = completed();
      const finished = COUNTED.filter((item) => done.has(item.id)).length;
      const percent = Math.round((finished / COUNTED.length) * 100);
      const next = COUNTED.find((item) => !done.has(item.id));
      clear(path).append(
        h("div", { class: "wl-path-head" },
          h("div", {},
            h("span", { class: "section-label" }, "Your path"),
            h("h2", {}, finished ? "Pick up where you left off" : "Read a little, then build")),
          h("span", { class: "spacer" }),
          next
            ? h("button", { class: "btn small", onclick: () => ctx.navigate(next.id) }, finished ? `Continue: ${next.title}` : `Begin: ${next.title}`, icon("chevron-right", 12))
            : h("span", { class: "wl-all-done" }, icon("circle-check", 14), "Everything done")),
        h("div", { class: "wl-progress" },
          ui.meter(finished / COUNTED.length, { light: true }),
          h("span", {}, `${finished} of ${COUNTED.length} done · ${percent}%`)),
        h("div", { class: "wl-cols" }, SECTIONS.map((section, index) => column(section, index, done))),
        h("p", { class: "wl-path-hint" }, "Mark a page as done with ", h("span", { class: "wl-inline-icon" }, icon("circle-check", 12)),
          " in the toolbar. ",
          EXTRA.length
            ? ["The total also counts ", EXTRA.map((item, index) => [index ? ", " : "",
              h("button", { class: "link-btn", onclick: () => ctx.navigate(item.id) }, item.title)]), ". "]
            : null,
          "Your progress is kept on this computer."));
    }

    /* ---------- this machine ---------- */

    async function lookAgain(button) {
      button.disabled = true;
      probing = true;
      paintMachine();
      try {
        await ctx.api.post("/api/devices/refresh", {});
        await ctx.refresh();
      } catch (error) {
        ctx.toast(`Couldn't look for a GPU: ${error.message}`, "error");
      } finally {
        probing = false;
        if (!disposed) paintMachine();
      }
    }

    function deviceRow(device) {
      const gpu = device.kind !== "cpu";
      return h("div", { class: ["wl-device", gpu ? "gpu" : "cpu"] },
        h("span", { class: "wl-device-icon" }, icon(gpu ? "zap" : "cpu", 16)),
        h("span", { class: "wl-device-text" },
          h("span", { class: "wl-device-name" }, device.name || device.label),
          h("span", { class: "wl-device-detail" }, device.detail || "")),
        h("span", { class: "wl-device-tag" }, device.label || (gpu ? "GPU" : "CPU")));
    }

    function paintMachine() {
      const state = ctx.state() || {};
      const devices = state.devices || [];
      const app = state.app || {};
      const scanning = probing || app.gpu_probe === "running";
      const gpus = devices.filter((device) => device.kind !== "cpu");
      const summary = devices.map((device) => {
        if (device.kind === "cpu") return `CPU: ${device.name}`;
        const backend = backendOf(device);
        return `GPU: ${device.name}${backend ? ` via ${backend}` : ""}`;
      }).join(" · ");
      const choice = state.settings?.device || "auto";
      const chosen = devices.find((device) => device.id === choice);
      const fallback = { cuda: "the CUDA GPU", mps: "the Metal GPU" }[state.auto_device] || "the CPU";
      const timed = project?.estimates?.calibrated ? devices.find((device) => device.id === project.estimates.auto_device) : null;
      const footer = [app.platform, app.engine, app.version ? `v${app.version}` : null].filter(Boolean).join(" · ");
      const lookButton = h("button", { class: "btn small", onclick: (event) => lookAgain(event.currentTarget) }, icon("refresh", 12), "Look again");
      clear(machine).append(...[
        h("div", { class: "wl-machine-head" },
          h("span", { class: "panel-label" }, "This machine"),
          h("span", { class: "spacer" }),
          scanning ? h("span", { class: "wl-scan" }, h("span", { class: "spinner" }), "Looking for a GPU…") : null),
        h("p", { class: "wl-machine-summary" }, summary || "Asking the engine…"),
        h("div", { class: "wl-devices" }, devices.map(deviceRow)),
        !gpus.length && !scanning && devices.length
          ? h("div", { class: "wl-nogpu" },
            h("p", {}, "No GPU the engine can use turned up, so training runs on the CPU. The small models in this app still train there, just more slowly."),
            lookButton)
          : null,
        h("div", { class: "wl-machine-row" },
          h("span", { class: "panel-label" }, "Trains on"),
          h("span", {}, chosen
            ? `${chosen.label || chosen.name}, chosen in Settings`
            : timed && gpus.length ? `Automatic: ${timed.name || timed.label}, the ${devices.length > 2 ? "fastest" : "faster"} device for ${project.name}`
            : gpus.length ? `Automatic: the faster device once Project setup has timed them; until then, ${fallback}` : "Automatic: the CPU")),
        footer ? h("div", { class: "wl-machine-foot" }, footer) : null,
      ].filter(Boolean));
    }

    /* ---------- data ---------- */

    function paint() {
      if (disposed) return;
      paintCards();
      paintPath();
      paintMachine();
    }

    async function reload() {
      try {
        project = ctx.state()?.current ? await loadProject(ctx) : null;
        projectError = null;
      } catch (error) {
        project = null;
        projectError = error;
      }
      paint();
    }

    const reloadSoon = () => {
      clearTimeout(timer);
      timer = setTimeout(reload, 250);
    };
    ctx.onState(reloadSoon);
    ctx.on("checkpoint", reloadSoon);
    ctx.on("job_end", reloadSoon);

    paint();
    await reload();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  },
};
