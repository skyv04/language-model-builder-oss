// Sampling: prompt a checkpoint and watch it continue token by token; steer it, compare decoders, revisit samples.
import { noProjectState, errorBanner, watchProject, STAGES, STAGE_SHORT, STAGE_TITLE } from "./shared.js";
import {
  PRESETS, PRESET_ORDER, generate, loadSampler, saveSampler, requestSettings, randomSeed, readStore, writeStore, stopLabel, speedLine,
} from "../workbench/generate.js";
import { decodingPanel, sideHead } from "../workbench/sampler.js";
import { recentList, compareColumns, tokensLabel } from "../workbench/sample-text.js";
import { samplingCard } from "../workbench/sampling-card.js";

const memory = new Map(); // project id → {prompt, run, editing, stage}; survives page changes
const MAX_RECENT = 8;
const VIEW_KEY = "lmb.sampling.view";

const idsOf = (tokens) => tokens.map((token) => token.id);
const stagesOf = (project) => STAGES.filter((stage) => project.stages[stage]?.checkpoint_step != null);
const shorten = (text, size = 48) => (text.length > size ? `${text.slice(0, size - 1).trimEnd()}…` : text);

export default {
  id: "sampling",
  title: "Sampling",
  async render(root, ctx) {
    const { h, ui, icon } = ctx;
    const settings = loadSampler();
    const view = readStore(VIEW_KEY, { chips: true });
    const host = h("div", { class: "wb-sampling-host" });
    root.append(host);
    let project = null;
    let job = ctx.state()?.job || null;
    let mem = null;
    let built = null;
    let controller = null;
    let ticket = 0;
    let busy = null;
    let peekTimer = null;
    let refreshTimer = null;
    let recents = [];

    /* ---------- streaming plumbing: one stream at a time, newest wins ---------- */

    function cancelStreams() {
      if (controller) controller.abort();
      controller = null;
      ticket += 1;
      clearTimeout(peekTimer);
      setBusy(null);
    }

    function stream(kind, body, { onStart, onToken } = {}) {
      if (controller) controller.abort();
      clearTimeout(peekTimer);
      const mine = ++ticket;
      const abort = new AbortController();
      controller = abort;
      setBusy(kind === "peek" ? null : kind);
      const live = () => mine === ticket;
      return generate(ctx, body, {
        signal: abort.signal,
        onStart: (start) => live() && onStart && onStart(start),
        onToken: (token) => live() && onToken && onToken(token),
      }).then((result) => ({ result, live: live() }), (error) => ({ error, result: error.partial || null, live: live() }))
        .finally(() => {
          if (!live()) return;
          controller = null;
          setBusy(null);
        });
    }

    function setBusy(kind) {
      busy = kind;
      if (!built) return;
      const hasRun = Boolean(mem.run) && !mem.editing;
      built.card.setBusy(kind, { hasRun, canGenerate: stagesOf(project).includes(mem.stage) });
      built.compareButton.disabled = Boolean(kind);
    }

    const failure = (error) => (error.status === 409 ? error.message : `The model could not write: ${error.message}`);

    /* ---------- runs ---------- */

    function updateCount() {
      const run = mem.run;
      if (!run || mem.editing) return built.card.setCount(0);
      built.card.setCount(run.promptTokens.length + run.tokens.length,
        `${tokensLabel(run.promptTokens.length)} of prompt + ${tokensLabel(run.tokens.length)} written by the model`);
    }

    function showMeta(run) {
      const end = run?.end;
      if (!end || end.stop_reason === "error") built.card.setMeta("");
      else if (end.stop_reason === "stopped") built.card.setMeta(`Stopped by you · ${tokensLabel(run.tokens.length)} written`);
      else built.card.setMeta(`${stopLabel(end)} · ${speedLine(end)}`);
    }

    function showRun({ streaming = false } = {}) {
      built.card.output.setRun(mem.run, { streaming });
      built.card.setMode({ editing: mem.editing, chips: view.chips, hasRun: Boolean(mem.run) });
      showMeta(streaming ? null : mem.run);
      updateCount();
      setBusy(busy);
    }

    // Streams into `run`. A fresh run takes its prompt tokens and seed from the start event.
    async function play(run, body, { fresh = false } = {}) {
      const { card } = built;
      card.setError(null);
      card.next.set(null);
      card.setMeta("");
      card.output.setStreaming(true);
      const outcome = await stream("generate", body, {
        onStart(start) {
          if (fresh) {
            run.promptTokens = start.prompt_tokens || [];
            run.seed = start.seed;
            card.output.setRun(run, { streaming: true });
            if (!settings.seed_locked) settings.seed = start.seed;
            card.setSeed(settings.seed, settings.seed_locked);
          }
          run.step = start.step;
          updateCount();
        },
        onToken(token) {
          run.tokens.push(token);
          card.output.addToken(token, run.tokens.length - 1);
          updateCount();
        },
      });
      if (!outcome.live || !built || built.card !== card) return;
      card.output.setStreaming(false);
      if (outcome.error) card.setError(failure(outcome.error));
      run.end = outcome.result?.end || { stop_reason: outcome.error ? "error" : "stopped" };
      showMeta(run);
      if (run.tokens.length) remember(run);
      if (!outcome.error && run.end.stop_reason !== "stopped" && run.end.stop_reason !== "context") peek();
    }

    function startRun({ seed } = {}) {
      const run = {
        id: `s${Date.now().toString(36)}${Math.floor(Math.random() * 1e4)}`, stage: mem.stage, prompt: mem.prompt,
        promptTokens: [], tokens: [], end: null, seed: null, step: null, settings: { ...requestSettings(settings), seed: undefined }, time: Date.now(),
      };
      mem.run = run;
      mem.editing = false;
      showRun({ streaming: true });
      play(run, { stage: mem.stage, prompt: run.prompt, settings: requestSettings(settings, { seed }) }, { fresh: true });
    }

    function continueRun() {
      const run = mem.run;
      if (!run) return;
      run.stage = mem.stage;
      play(run, { stage: mem.stage, prompt_ids: idsOf([...run.promptTokens, ...run.tokens]), settings: requestSettings(settings) });
    }

    // Picking a runner-up rewrites the text from that token on.
    function branch(index, alternative, token) {
      const run = mem.run;
      if (!run || index >= run.tokens.length) return;
      run.tokens = run.tokens.slice(0, index);
      run.tokens.push({ id: alternative.id, text: alternative.text, prob: alternative.prob, alternatives: token.alternatives });
      run.end = null;
      built.card.output.setRun(run, { streaming: true });
      continueRun();
    }

    function schedulePeek(delay = 350) {
      clearTimeout(peekTimer);
      if (busy) return;
      peekTimer = setTimeout(peek, delay);
    }

    // Asks for one token to list the checkpoint's top candidates for what comes next.
    async function peek() {
      const run = mem.run;
      if (!built || !run || mem.editing || busy) return;
      const all = [...run.promptTokens, ...run.tokens];
      if (!all.length) return built.card.next.set(null);
      built.card.next.set({ state: "loading" });
      const outcome = await stream("peek", {
        stage: mem.stage, prompt_ids: idsOf(all), settings: { ...requestSettings(settings, { seed: settings.seed }), max_tokens: 1 },
      });
      if (!outcome.live || !built) return;
      if (outcome.error) return built.card.next.set({ state: "error", error: failure(outcome.error) });
      const token = outcome.result?.tokens[0];
      if (!token) return built.card.next.set(null);
      const alternatives = token.alternatives?.length ? token.alternatives : [{ id: token.id, text: token.text, prob: token.prob }];
      built.card.next.set({ state: "ok", alternatives });
    }

    function pickNext(alternative, peeked) {
      const run = mem.run;
      if (!run) return;
      const token = { id: alternative.id, text: alternative.text, prob: alternative.prob, alternatives: peeked.alternatives };
      run.tokens.push(token);
      built.card.output.addToken(token, run.tokens.length - 1);
      updateCount();
      remember(run);
      peek();
    }

    /* ---------- compare ---------- */

    async function compareAll() {
      const { compare } = built;
      const seed = settings.seed_locked ? settings.seed : randomSeed();
      const prompt = mem.prompt;
      compare.hidden = false;
      compare.setCaption(`${prompt ? `“${shorten(prompt)}”` : "Empty prompt"} · seed ${seed} · ${STAGE_SHORT[mem.stage]} checkpoint`);
      for (const name of PRESET_ORDER) {
        compare.columns.get(name).text.setRun(null);
        compare.setStatus(name, "waiting");
      }
      compare.scrollIntoView({ behavior: "smooth", block: "nearest" });
      for (const name of PRESET_ORDER) {
        const column = compare.columns.get(name);
        const { label, ...values } = PRESETS[name];
        const decoder = { ...values, max_tokens: settings.max_tokens };
        compare.setSettings(name, decoder, seed);
        compare.setStatus(name, "writing…");
        const run = { promptTokens: [], tokens: [] };
        column.text.setRun(run, { streaming: true });
        const outcome = await stream("compare", { stage: mem.stage, prompt, settings: { ...decoder, seed } }, {
          onToken: (token) => {
            run.tokens.push(token);
            column.text.addToken(token, run.tokens.length - 1);
          },
        });
        if (!outcome.live) return;
        column.text.setStreaming(false);
        if (outcome.error || outcome.result?.aborted) {
          compare.setStatus(name, outcome.error ? "failed" : "stopped");
          if (outcome.error) built.card.setError(failure(outcome.error));
          return;
        }
        compare.setStatus(name, tokensLabel(run.tokens.length));
      }
    }

    /* ---------- recent samples ---------- */

    const recentKey = () => `lmb.samples.${project.id}`;

    function remember(run) {
      const entry = JSON.parse(JSON.stringify({ ...run, stageLabel: STAGE_SHORT[run.stage] }));
      recents = [entry, ...recents.filter((item) => item.id !== run.id)].slice(0, MAX_RECENT);
      writeStore(recentKey(), { list: recents });
      built.recent.set(recents, run.id);
    }

    function openRecent(sample) {
      cancelStreams();
      mem.run = JSON.parse(JSON.stringify(sample));
      mem.prompt = sample.prompt || "";
      mem.editing = false;
      if (stagesOf(project).includes(sample.stage)) mem.stage = sample.stage;
      built.card.setPrompt(mem.prompt);
      built.card.setError(null);
      update();
      showRun();
      built.recent.set(recents, sample.id);
      peek();
    }

    /* ---------- layout ---------- */

    function noCheckpoint() {
      const running = job?.kind === "pretrain" || project.stages.pretrain?.state === "running";
      return h("div", { class: "wb-sampling" },
        h("section", { class: "panel-dark wb-sampler wb-sampler-empty" },
          h("div", { class: "wb-sampler-head" }, h("span", { class: "panel-label" }, "Sampling")),
          h("div", { class: "wb-sampler-blank" },
            h("span", { class: "wb-blank-icon" }, icon("dice", 24)),
            h("h3", {}, running ? "Waiting for the first checkpoint" : "Nothing to sample yet"),
            h("p", {}, running
              ? "Pre-training is running. Sampling opens as soon as the run saves its first checkpoint."
              : "Sampling reads from a saved checkpoint. Pre-train the model for a few hundred steps, then come back and give it a prompt."),
            h("button", { class: "btn primary", onclick: () => ctx.navigate("pretraining") }, "Go to Pre-training", icon("chevron-right", 13)))));
    }

    function jobNote() {
      if (!job) return "";
      if (job.kind === "calibrate") return "A speed calibration is running, so sampling may be slow until it finishes.";
      return `${STAGE_TITLE[job.kind] || "Training"} is running. Samples come from the last saved checkpoint and share the machine with the run, so they may be slower.`;
    }

    function update() {
      const info = project.stages[mem.stage] || {};
      built.card.setStages(stagesOf(project), mem.stage, info.checkpoint_step);
      built.card.setMode({ editing: mem.editing, chips: view.chips, hasRun: Boolean(mem.run) });
      built.card.setSeed(settings.seed, settings.seed_locked);
      built.card.setNote(jobNote());
      updateCount();
      setBusy(busy);
    }

    const actions = {
      onStage(stage) {
        mem.stage = stage;
        update();
        schedulePeek();
      },
      onToggleChips() {
        view.chips = !view.chips;
        writeStore(VIEW_KEY, view);
        built.card.output.setChips(view.chips);
        showRun({ streaming: busy === "generate" });
      },
      onEdit() {
        mem.editing = !mem.editing || !mem.run;
        if (mem.editing) cancelStreams();
        update();
        showRun();
        if (mem.editing) built.card.textarea.focus();
        else schedulePeek(0);
      },
      onPromptInput(text) {
        mem.prompt = text;
      },
      onGenerate: () => startRun(),
      onStop: () => (controller ? controller.abort() : cancelStreams()),
      onContinue: continueRun,
      onRetry() {
        settings.seed = randomSeed();
        saveSampler(settings);
        if (mem.run) mem.prompt = mem.run.prompt;
        built.card.setPrompt(mem.prompt);
        startRun({ seed: settings.seed });
      },
      onPickAlternative: branch,
      onPickNext: pickNext,
      onSeed(seed) {
        Object.assign(settings, { seed, seed_locked: true });
        saveSampler(settings);
        built.card.setSeed(settings.seed, true);
      },
      onShuffle() {
        Object.assign(settings, { seed: randomSeed(), seed_locked: true });
        saveSampler(settings);
        built.card.setSeed(settings.seed, true);
      },
      onLock() {
        settings.seed_locked = !settings.seed_locked;
        saveSampler(settings);
        built.card.setSeed(settings.seed, settings.seed_locked);
      },
    };

    function build() {
      cancelStreams();
      mem = memory.get(project.id) || { prompt: "", run: null, editing: true, stage: null };
      memory.set(project.id, mem);
      const card = samplingCard(actions);
      const decoding = decodingPanel(settings, {
        onChange: () => {
          saveSampler(settings);
          schedulePeek();
        },
      });
      const compareButton = h("button", { class: "btn wb-compare-btn", onclick: () => compareAll() }, icon("columns", 14), "Compare all three presets");
      const recent = recentList({
        onOpen: openRecent,
        onClear: () => {
          recents = [];
          writeStore(recentKey(), { list: [] });
          recent.set(recents, null);
        },
      });
      const compare = compareColumns({
        onClose: () => {
          if (busy === "compare") cancelStreams();
          compare.hidden = true;
        },
      });
      const side = h("aside", { class: "wb-side" }, decoding,
        h("section", { class: "wb-side-section" }, sideHead("Compare"), compareButton,
          h("p", { class: "wb-side-note" }, "One prompt and one seed, written three ways, side by side.")),
        recent);
      host.replaceChildren(h("div", { class: "wb-sampling" }, h("div", { class: "wb-sampling-main" }, card.element, compare), side));
      built = { projectId: project.id, card, compareButton, compare, recent };
      recents = readStore(recentKey(), { list: [] }).list || [];
      recent.set(recents, mem.run?.id);
      card.setPrompt(mem.prompt);
      card.output.setChips(view.chips);
    }

    function apply(nextProject, nextJob, error) {
      job = nextJob;
      if (error || !nextProject) {
        cancelStreams();
        built = null;
        project = nextProject;
        host.replaceChildren(error ? errorBanner(error)
          : noProjectState(ctx, { message: "Create a project and pre-train it; then you can prompt the model here and watch it write." }));
        return;
      }
      project = nextProject;
      const stages = stagesOf(project);
      if (!stages.length) {
        cancelStreams();
        built = null;
        host.replaceChildren(noCheckpoint());
        return;
      }
      const fresh = !built || built.projectId !== project.id;
      if (fresh) build();
      if (!stages.includes(mem.stage)) mem.stage = stages.includes("pretrain") ? "pretrain" : stages[0];
      update();
      if (fresh) {
        showRun();
        if (mem.run && !mem.editing) schedulePeek(0);
      }
    }

    const refresh = watchProject(ctx, apply);
    const refreshSoon = () => {
      clearTimeout(refreshTimer);
      refreshTimer = setTimeout(refresh, 400);
    };
    ctx.on("checkpoint", refreshSoon);
    ctx.on("job", (message) => {
      job = message.job;
      if (built) built.card.setNote(jobNote());
    });
    ctx.on("job_end", () => {
      job = null;
      refreshSoon();
    });

    await refresh();
    return () => {
      cancelStreams();
      clearTimeout(refreshTimer);
      ui.closePopover();
    };
  },
};
