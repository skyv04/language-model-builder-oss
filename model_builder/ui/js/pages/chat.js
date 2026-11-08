// Chat with your model: talk to the base, fine-tuned or preference-tuned checkpoint, X-ray every reply,
// read the raw transcript the model sees, and export a checkpoint as a shareable .zip.
import { clear } from "../dom.js";
import { noProjectState, errorBanner, watchProject, STAGES, STAGE_SHORT, STAGE_PAGE } from "./shared.js";
import { generate, loadSampler, saveSampler, requestSettings, randomSeed, readStore, writeStore, stopLabel, speedLine } from "../workbench/generate.js";
import { POPOVER_TITLE, tokensLabel } from "../workbench/sample-text.js";
import {
  DEFAULT_SYSTEM, templateText, rawText, visibleText, turnsSeen, xrayLegend, transcriptPanel, settingsCard, composer,
  chooseExport, showExported,
} from "../workbench/chat-view.js";

const memory = new Map(); // project id → {messages, stage}; survives page changes
const SAMPLER_KEY = "lmb.chat.sampler";
const VIEW_KEY = "lmb.chat.view";
const SYSTEM_KEY = "lmb.chat.system";

const STAGE_NAME = { pretrain: "Base model", sft: "Fine-tuned", dpo: "Preference-tuned" };
const STAGE_ABOUT = {
  pretrain: "Only ever learned to continue text, so expect it to carry on from your message rather than answer it.",
  sft: "Learned the back-and-forth of a conversation from example chats.",
  dpo: "Fine-tuned, then nudged towards the replies that were preferred.",
};
const MISSING = {
  pretrain: "There is no pre-trained checkpoint yet.",
  sft: "There is no fine-tuned checkpoint yet, so nothing has taught this model to hold a conversation.",
  dpo: "There is no preference-tuned checkpoint yet.",
};
const PAGE_TITLE = { pretraining: "Pre-training", sft: "Supervised fine-tuning", dpo: "Direct preference optimization" };
const RUNNING = { pretrain: "Pre-training", sft: "Fine-tuning", dpo: "Preference tuning" };
const STARTERS = [
  "Tell me a story about a brave little fox.",
  "What do bees make?",
  "Can you help me plan a picnic?",
  "Why is the sky blue?",
];

const stagesOf = (project) => STAGES.filter((stage) => project?.stages?.[stage]?.checkpoint_step != null);
const newId = () => `m${Date.now().toString(36)}${Math.floor(Math.random() * 1e5)}`;

export default {
  id: "chat",
  title: "Chat with your model",
  async render(root, ctx) {
    const { h, ui, icon, fmt } = ctx;
    const settings = loadSampler(SAMPLER_KEY, { seed_locked: false });
    const view = { xray: false, panel: "transcript", ...readStore(VIEW_KEY, {}) };
    let system = String(readStore(SYSTEM_KEY, { text: "" }).text || "");
    const host = h("div", { class: "ct-host" });
    root.append(host);
    let project = null;
    let job = ctx.state()?.job || null;
    let mem = null;
    let built = null;
    let controller = null;
    let ticket = 0;
    let busy = false;
    let hintStage = null;
    let refreshTimer = null;
    let transcriptFrame = 0;
    let exporting = false;
    let noticeKind = null;
    const views = new Map(); // assistant message id → {body, meta}

    /* ---------- streaming: one reply at a time ---------- */

    function finishStreaming() {
      for (const message of mem?.messages || []) {
        if (!message.streaming) continue;
        message.streaming = false;
        message.end = message.end || { stop_reason: "stopped", tokens: message.tokens.length };
        if (!message.tokens.length) message.failed = true;
      }
    }

    function cancel() {
      if (controller) controller.abort();
      controller = null;
      ticket += 1;
      finishStreaming();
      setBusy(false);
    }

    function setBusy(next) {
      busy = next;
      if (!built) return;
      built.composer.setBusy(busy);
      paintAllMeta();
    }

    const history = () => mem.messages
      .filter((message) => message.role === "user" || (message.role === "assistant" && !message.failed && !message.streaming))
      .map((message) => ({ role: message.role, content: message.role === "user" ? message.content : visibleText(message.tokens) }));

    async function reply({ seed } = {}) {
      const stage = mem.stage;
      const sent = history();
      const message = {
        id: newId(), role: "assistant", stage, step: project.stages[stage]?.checkpoint_step ?? null, tokens: [], end: null,
        error: null, failed: false, streaming: true, promptText: null, contextTokens: null, contextLimit: null, sentTurns: sent.length, seed: null,
      };
      mem.messages.push(message);
      appendMessage(message);
      if (controller) controller.abort();
      const mine = ++ticket;
      const abort = new AbortController();
      controller = abort;
      setBusy(true);
      paintTranscript();
      const live = () => mine === ticket;
      const body = {
        stage,
        messages: sent,
        system: system.trim() ? system : null,
        settings: requestSettings(settings, seed !== undefined ? { seed } : {}),
      };
      let outcome;
      try {
        outcome = { result: await generate(ctx, body, {
          signal: abort.signal,
          onStart(start) {
            if (!live()) return;
            message.promptText = start.prompt_text ?? rawText(start.prompt_tokens || []);
            message.contextTokens = start.context_tokens ?? null;
            message.contextLimit = start.context_limit ?? null;
            message.seed = start.seed ?? null;
            if (start.step != null) message.step = start.step;
            if (!settings.seed_locked && start.seed != null) {
              settings.seed = start.seed;
              saveSampler(settings, SAMPLER_KEY);
              built?.settings.setSeed(settings.seed, false);
            }
            paintMeta(message);
            scheduleTranscript();
          },
          onToken(token) {
            if (!live()) return;
            message.tokens.push(token);
            streamToken(message, token);
            scheduleTranscript();
          },
        }) };
      } catch (error) {
        outcome = { error, result: error.partial || null };
      }
      if (!live()) return;
      controller = null;
      message.streaming = false;
      if (outcome.error) {
        const { error } = outcome;
        message.error = error.status === 409 ? error.message : `The model couldn't reply: ${error.message}`;
        if (!message.tokens.length) message.failed = true;
      }
      message.end = outcome.result?.end || (outcome.error ? null : { stop_reason: "stopped", tokens: message.tokens.length });
      if (!outcome.error && !message.tokens.length && message.end?.stop_reason === "stopped") message.failed = true;
      paintReply(message);
      setBusy(false);
      paintTranscript();
      if (outcome.error && outcome.error.status === 409) refreshSoon();
    }

    function send(text) {
      const content = String(text || "").trim();
      if (!content || busy || !built) return false;
      const message = { id: newId(), role: "user", content };
      mem.messages.push(message);
      appendMessage(message);
      reply();
      return true;
    }

    function retry() {
      const last = mem.messages[mem.messages.length - 1];
      if (busy || !last || last.role !== "assistant") return;
      mem.messages.pop();
      views.get(last.id)?.element.remove();
      views.delete(last.id);
      const seed = randomSeed();
      if (settings.seed_locked) {
        settings.seed = seed;
        saveSampler(settings, SAMPLER_KEY);
        built.settings.setSeed(seed, true);
      }
      reply({ seed });
    }

    function newChat() {
      cancel();
      mem.messages = [];
      renderThread();
      paintTranscript();
      built.composer.focus();
    }

    /* ---------- thread ---------- */

    function systemPill() {
      const custom = Boolean(system.trim());
      return h("button", {
        class: ["ct-system", custom ? "" : "default"],
        title: "The hidden first turn of every conversation. Click to edit it.",
        onclick: () => openPanel("settings", { focusSystem: true }),
      }, h("span", { class: "ct-system-tag" }, "System"), h("span", { class: "ct-system-text" }, custom ? system.trim() : DEFAULT_SYSTEM), icon("pencil", 11));
    }

    function greeting() {
      const base = mem.stage === "pretrain";
      return h("div", { class: "ct-greeting" },
        h("span", { class: "ct-greeting-icon" }, icon(base ? "scroll" : "messages", 22)),
        h("h3", {}, base ? "Talk to the base model" : "Say hello to your model"),
        h("p", {}, base
          ? "This checkpoint has read a lot of text but has never seen a conversation, so it will probably keep writing in the style of its training data instead of answering. Try it, then compare with a fine-tuned checkpoint."
          : "Everything it says comes from the small network you trained, one token at a time. Turn on X-ray to see how sure it was of each word."),
        h("div", { class: "ct-starters" }, STARTERS.map((text) => h("button", { class: "chip", onclick: () => send(text) }, text))));
    }

    function renderThread() {
      views.clear();
      ui.closePopover();
      const { thread } = built;
      clear(thread).append(systemPill());
      if (!mem.messages.length) {
        thread.append(greeting());
        return;
      }
      for (const message of mem.messages) thread.append(messageNode(message));
      scrollToEnd(true);
    }

    function appendMessage(message) {
      const { thread } = built;
      thread.querySelector(".ct-greeting")?.remove();
      thread.append(messageNode(message));
      scrollToEnd(true);
    }

    function messageNode(message) {
      if (message.role === "user") return h("div", { class: "ct-msg user" }, h("div", { class: "bubble user" }, message.content));
      const body = h("div", { class: "ct-reply" });
      const meta = h("div", { class: "ct-meta" });
      const element = h("div", { class: "ct-msg assistant" }, body, meta);
      views.set(message.id, { element, body, meta });
      paintReply(message);
      return element;
    }

    function chip(token) {
      return ui.tokenChip(token, { light: true, popoverTitle: POPOVER_TITLE });
    }

    function paintReply(message) {
      const entry = views.get(message.id);
      if (!entry) return;
      const { body } = entry;
      body.classList.toggle("xray", view.xray);
      clear(body);
      if (view.xray) {
        for (const token of message.tokens) body.append(chip(token));
      } else {
        const text = visibleText(message.tokens);
        if (text) body.append(ui.specialText(text));
        else if (!message.streaming && !message.error) {
          body.append(h("span", { class: "ct-quiet" }, message.failed
            ? "Stopped before the model wrote anything."
            : "The model ended its turn straight away, without writing anything."));
        }
      }
      if (message.streaming) body.append(h("span", { class: "ct-caret" }));
      if (message.error) body.append(h("div", { class: "ct-reply-error" }, icon("alert", 13), h("span", {}, message.error)));
      paintMeta(message);
    }

    function streamToken(message, token) {
      const entry = views.get(message.id);
      if (!entry) return;
      if (view.xray) {
        const caret = entry.body.querySelector(".ct-caret");
        const node = chip(token);
        if (caret) caret.before(node);
        else entry.body.append(node);
      } else {
        paintReply(message);
      }
      scrollToEnd();
    }

    function paintMeta(message) {
      const entry = views.get(message.id);
      if (!entry) return;
      const { meta } = entry;
      clear(meta).append(h("span", { class: ["ct-meta-stage", message.stage] },
        STAGE_SHORT[message.stage], message.step != null ? h("span", {}, ` · step ${fmt.int(message.step)}`) : null));
      if (message.streaming) {
        meta.append(h("span", {}, message.tokens.length ? `Writing… ${tokensLabel(message.tokens.length)}` : "Reading the conversation…"));
        return;
      }
      const end = message.end;
      if (end && end.stop_reason === "stopped") meta.append(h("span", {}, `Stopped by you · ${tokensLabel(message.tokens.length)}`));
      else if (end) meta.append(h("span", {}, `${stopLabel(end, { chat: true })} · ${speedLine(end)}`));
      if (message.seed != null) meta.append(h("span", { class: "ct-meta-seed" }, `seed ${message.seed}`));
      const actions = h("span", { class: "ct-meta-actions" });
      const text = visibleText(message.tokens);
      if (text) {
        actions.append(h("button", {
          class: "ct-meta-btn",
          title: "Copy this reply",
          onclick: () => ui.copyText(text).then(() => ui.toast("Reply copied"), (error) => ui.toast(`Couldn't copy: ${error.message}`, "error")),
        }, "Copy"));
      }
      if (!busy && mem.messages[mem.messages.length - 1] === message) {
        actions.append(h("button", { class: "ct-meta-btn", title: "Write this reply again with a fresh seed", onclick: retry }, icon("refresh", 11), "Try again"));
      }
      meta.append(actions);
    }

    function paintAllMeta() {
      for (const message of mem?.messages || []) if (message.role === "assistant") paintMeta(message);
    }

    function scrollToEnd(force = false) {
      const scroller = built?.scroller;
      if (!scroller) return;
      const near = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
      if (force || near) scroller.scrollTop = scroller.scrollHeight;
    }

    /* ---------- transcript ---------- */

    function scheduleTranscript() {
      if (transcriptFrame) return;
      transcriptFrame = requestAnimationFrame(() => {
        transcriptFrame = 0;
        paintTranscript();
      });
    }

    function paintTranscript() {
      if (!built || view.panel !== "transcript") return;
      const last = [...mem.messages].reverse().find((message) => message.role === "assistant" && message.promptText != null);
      const limit = project.config?.context || null;
      if (last) {
        const windowSize = last.contextLimit || limit;
        // When a reply fills the window the engine still yields the token it predicted from the full window, one past the limit.
        const used = (last.contextTokens || 0) + last.tokens.length;
        built.transcript.set({
          text: last.promptText + rawText(last.tokens),
          streaming: last.streaming,
          used: windowSize ? Math.min(used, windowSize) : used,
          limit: windowSize,
          droppedTurns: Math.max(0, last.sentTurns - turnsSeen(last.promptText)),
        });
        return;
      }
      built.transcript.set({ text: templateText(system, history(), { open: busy }), streaming: busy, used: null, limit, preview: true });
    }

    /* ---------- header, stage picker, panels ---------- */

    function selectStage(stage) {
      if (!stagesOf(project).includes(stage)) {
        hintStage = hintStage === stage ? null : stage;
        paintHeader();
        return;
      }
      hintStage = null;
      if (mem.stage === stage) return;
      mem.stage = stage;
      paintHeader();
      if (!mem.messages.length) renderThread();
    }

    function paintHeader() {
      const available = stagesOf(project);
      built.name.textContent = project.name;
      clear(built.pipeline);
      STAGES.forEach((stage, index) => {
        if (index) built.pipeline.append(h("span", { class: "ct-arrow", "aria-hidden": "true" }, icon("chevron-right", 12)));
        const has = available.includes(stage);
        const training = job && job.kind === stage && job.project === project.id;
        const info = project.stages[stage] || {};
        built.pipeline.append(h("button", {
          class: ["ct-stage", stage === mem.stage ? "active" : "", has ? "" : "missing"],
          "aria-pressed": String(stage === mem.stage),
          title: has ? `${STAGE_NAME[stage]} checkpoint, step ${fmt.int(info.checkpoint_step)}` : MISSING[stage],
          onclick: () => selectStage(stage),
        }, training ? h("span", { class: "ct-stage-dot", title: `${RUNNING[stage]} is running` }) : null, STAGE_SHORT[stage]));
      });
      const info = project.stages[mem.stage] || {};
      clear(built.subtitle).append(
        h("strong", {}, STAGE_NAME[mem.stage]),
        info.checkpoint_step != null ? ` · step ${fmt.int(info.checkpoint_step)}` : "",
        h("span", { class: "ct-subtitle-about" }, ` — ${STAGE_ABOUT[mem.stage]}`));
      const hint = built.hint;
      clear(hint);
      const missingTarget = hintStage && !available.includes(hintStage) ? hintStage : null;
      if (missingTarget) {
        const page = STAGE_PAGE[missingTarget];
        hint.append(icon("info", 13), h("span", {}, `${MISSING[missingTarget]} `),
          h("button", { class: "link-btn", onclick: () => ctx.navigate(page) }, `Open ${PAGE_TITLE[page]} ›`));
      } else if (mem.stage === "pretrain") {
        const lesson = "Base models aren't chat-trained: they learned to continue text, not to take turns.";
        if (available.includes("sft")) {
          hint.append(icon("info", 13), h("span", {}, `${lesson} Switch to SFT to compare.`));
        } else {
          hint.append(icon("info", 13), h("span", {}, `${lesson} `),
            h("button", { class: "link-btn", onclick: () => ctx.navigate("sft") }, "Fine-tune it on conversations ›"));
        }
      }
      hint.hidden = !hint.childNodes.length;
      built.jobNote.hidden = !job;
      if (job) {
        const text = job.kind === "calibrate"
          ? "A speed calibration is running, so replies may be slow until it finishes."
          : job.project && job.project !== project.id
            ? "Another project is training right now, so replies may be slower than usual."
            : `${RUNNING[job.kind] || "Training"} is running. Replies come from the last saved checkpoint and share the machine with the run, so they may be slower.`;
        clear(built.jobNote).append(icon("info", 14), h("span", {}, text));
      }
    }

    function paintToggles() {
      built.xrayButton.classList.toggle("active", view.xray);
      built.xrayButton.setAttribute("aria-pressed", String(view.xray));
      built.transcriptButton.classList.toggle("active", view.panel === "transcript");
      built.transcriptButton.setAttribute("aria-pressed", String(view.panel === "transcript"));
      built.settingsButton.classList.toggle("active", view.panel === "settings");
      built.settingsButton.setAttribute("aria-pressed", String(view.panel === "settings"));
      built.legend.hidden = !view.xray;
      built.layout.classList.toggle("solo", !view.panel);
      built.side.hidden = !view.panel;
      built.side.replaceChildren(view.panel === "settings" ? built.settings : view.panel === "transcript" ? built.transcript : "");
    }

    function openPanel(panel, { focusSystem = false, toggle = false } = {}) {
      view.panel = toggle && view.panel === panel ? null : panel;
      writeStore(VIEW_KEY, view);
      paintToggles();
      if (view.panel === "transcript") paintTranscript();
      if (focusSystem && view.panel === "settings") built.settings.focusSystem();
    }

    function toggleXray() {
      view.xray = !view.xray;
      writeStore(VIEW_KEY, view);
      ui.closePopover();
      paintToggles();
      for (const message of mem.messages) if (message.role === "assistant") paintReply(message);
    }

    async function share() {
      if (exporting) return;
      const stage = await chooseExport(project, mem.stage);
      if (!stage || !built) return;
      exporting = true;
      const button = built.shareButton;
      button.disabled = true;
      button.replaceChildren(h("span", { class: "spinner" }), "Exporting…");
      try {
        const result = await ctx.api.post("/api/projects/current/export", { stage });
        showNotice(null);
        showExported(result, stage);
      } catch (error) {
        showNotice(error.status === 409 ? error.message : `Couldn't export the model: ${error.message}`);
      } finally {
        exporting = false;
        button.disabled = false;
        button.replaceChildren(icon("share", 13), "Share model…");
      }
    }

    function showNotice(message, kind = "export") {
      if (!built) return;
      const { notice } = built;
      noticeKind = message ? kind : null;
      notice.hidden = !message;
      clear(notice);
      if (message) {
        notice.append(icon("alert", 15), h("span", {}, message), h("span", { class: "spacer" }),
          h("button", { class: "btn ghost small icon-only", title: "Dismiss", "aria-label": "Dismiss", onclick: () => showNotice(null) }, icon("x", 13)));
      }
    }

    /* ---------- layout ---------- */

    function toggle(iconName, label, title, onclick) {
      return h("button", { class: "btn small ct-toggle", title, "aria-pressed": "false", onclick }, icon(iconName, 13), label);
    }

    function build() {
      cancel();
      mem = memory.get(project.id) || { messages: [], stage: null };
      memory.set(project.id, mem);
      finishStreaming();
      const name = h("h2", { class: "ct-name" });
      const pipeline = h("div", { class: "ct-pipeline", role: "group", "aria-label": "Checkpoint to chat with" });
      const xrayButton = toggle("eye", "X-ray", "Show every token of the replies with its probability", toggleXray);
      const transcriptButton = toggle("scroll", "Transcript", "Show the raw text the model reads", () => openPanel("transcript", { toggle: true }));
      const settingsButton = toggle("sliders", "Settings", "System prompt, sampling and seed", () => openPanel("settings", { toggle: true }));
      const shareButton = h("button", { class: "btn small", title: "Export a checkpoint as a .zip", onclick: share }, icon("share", 13), "Share model…");
      const newButton = h("button", { class: "btn small", title: "Start a fresh conversation", onclick: newChat }, icon("new-chat", 13), "New chat");
      const subtitle = h("div", { class: "ct-subtitle" });
      const hint = h("div", { class: "ct-hint-line", hidden: true });
      const jobNote = h("div", { class: "banner ct-banner", hidden: true });
      const notice = h("div", { class: "banner error ct-banner", hidden: true, role: "alert" });
      const thread = h("div", { class: "ct-thread" });
      const scroller = h("div", { class: "ct-scroll" }, thread);
      const legend = xrayLegend();
      const box = composer({ onSend: send, onStop: () => (controller ? controller.abort() : cancel()) });
      const transcript = transcriptPanel({ onSampler: () => openPanel("settings") });
      const card = settingsCard(settings, {
        system,
        onSystem(text) {
          system = text;
          writeStore(SYSTEM_KEY, { text });
          const pill = built.thread.querySelector(".ct-system");
          if (pill) pill.replaceWith(systemPill());
          paintTranscript();
        },
        onChange() {
          saveSampler(settings, SAMPLER_KEY);
          transcript.setSampler(settings);
        },
        onSeed(seed) {
          Object.assign(settings, { seed, seed_locked: true });
          saveSampler(settings, SAMPLER_KEY);
          card.setSeed(seed, true);
        },
        onShuffle() {
          Object.assign(settings, { seed: randomSeed(), seed_locked: true });
          saveSampler(settings, SAMPLER_KEY);
          card.setSeed(settings.seed, true);
        },
        onLock() {
          settings.seed_locked = !settings.seed_locked;
          saveSampler(settings, SAMPLER_KEY);
          card.setSeed(settings.seed, settings.seed_locked);
        },
      });
      card.setSeed(settings.seed, settings.seed_locked);
      transcript.setSampler(settings);
      const side = h("div", { class: "ct-side" });
      const layout = h("div", { class: "ct-layout" },
        h("section", { class: "ct-chat" }, legend, scroller, box),
        side);
      host.replaceChildren(h("div", { class: "ct-page" },
        h("div", { class: "ct-head" },
          h("div", { class: "ct-title" }, name, pipeline),
          h("div", { class: "ct-actions" }, xrayButton, transcriptButton, settingsButton, shareButton, newButton)),
        subtitle, hint, jobNote, notice, layout));
      built = {
        projectId: project.id, name, pipeline, subtitle, hint, jobNote, notice, thread, scroller, legend, composer: box,
        transcript, settings: card, side, layout, xrayButton, transcriptButton, settingsButton, shareButton,
      };
    }

    function noCheckpoint() {
      const running = job && job.kind === "pretrain" && job.project === project.id;
      return h("section", { class: "panel-dark wb-sampler wb-sampler-empty ct-empty" },
        h("div", { class: "wb-sampler-head" }, h("span", { class: "panel-label" }, "Chat")),
        h("div", { class: "wb-sampler-blank" },
          h("span", { class: "wb-blank-icon" }, icon("messages", 24)),
          h("h3", {}, running ? "Waiting for the first checkpoint" : "Nothing to chat with yet"),
          h("p", {}, running
            ? "Pre-training is running. You can talk to the model as soon as the run saves its first checkpoint, but to get real answers it also needs fine-tuning on conversations."
            : "Chat needs a trained checkpoint. Pre-train the model first; then fine-tune it on example conversations (SFT) so it learns to take turns."),
          h("div", { class: "row", style: { justifyContent: "center" } },
            h("button", { class: "btn primary", onclick: () => ctx.navigate("pretraining") }, "Go to Pre-training", icon("chevron-right", 13)),
            h("button", { class: "btn", onclick: () => ctx.navigate("sft") }, "Supervised fine-tuning"))));
    }

    function noProject() {
      return h("div", {},
        noProjectState(ctx, { message: "Create a project and train it; then you can talk to your model here and see how it picks every word." }),
        h("p", { class: "ct-empty-link" }, "New here? ",
          h("button", { class: "link-btn", onclick: () => ctx.navigate("project-setup") }, "Start at Project setup ›")));
    }

    function apply(nextProject, nextJob, error) {
      job = nextJob;
      if (error && built) {
        // Keep the conversation on screen; a failed refresh is usually a passing hiccup.
        showNotice(`Couldn't refresh the project: ${error.message}`, "refresh");
        return;
      }
      if (error || !nextProject) {
        cancel();
        built = null;
        project = nextProject;
        host.replaceChildren(error ? errorBanner(error) : noProject());
        return;
      }
      project = nextProject;
      const stages = stagesOf(project);
      if (!stages.length) {
        cancel();
        built = null;
        host.replaceChildren(noCheckpoint());
        return;
      }
      const fresh = !built || built.projectId !== project.id;
      if (fresh) build();
      else if (noticeKind === "refresh") showNotice(null);
      if (!stages.includes(mem.stage)) mem.stage = stages[stages.length - 1];
      paintHeader();
      if (fresh) {
        paintToggles();
        renderThread();
      }
      paintTranscript();
    }

    const refresh = watchProject(ctx, apply);
    const refreshSoon = () => {
      clearTimeout(refreshTimer);
      refreshTimer = setTimeout(refresh, 400);
    };
    ctx.on("checkpoint", refreshSoon);
    ctx.on("job", (message) => {
      const before = job;
      job = message.job;
      if (built && (!before || !job || before.kind !== job.kind || before.state !== job.state)) paintHeader();
    });
    ctx.on("job_end", () => {
      job = null;
      refreshSoon();
    });

    await refresh();
    return () => {
      cancel();
      clearTimeout(refreshTimer);
      cancelAnimationFrame(transcriptFrame);
      ui.closePopover();
    };
  },
};
