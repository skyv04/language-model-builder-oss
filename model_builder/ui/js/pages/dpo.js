// Direct preference optimisation: preference ballot, banked pairs and the tuning run.
import { runCard } from "../workbench/run.js";
import { runLength, sampleTimeline } from "../workbench/controls.js";
import { pageHeader, noProjectState, errorBanner, watchProject, recipeRows, sectionHead } from "./shared.js";

const STAGE = "dpo";
const STEP_VALUES = [10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500];
// The last sampled pair survives navigation, so coming back does not cost two new generations.
const memory = { projectId: null, ballot: null };

export default {
  id: "dpo",
  title: "Direct preference optimization",
  async render(root, ctx) {
    const { h, fmt, icon, ui } = ctx;
    let project = null;
    let job = ctx.state()?.job || null;
    let built = null;
    let pairs = [];
    let required = 4;
    let ballot = null;
    let customPrompt = null;
    let editing = false;
    let sampling = false;
    let ballotError = "";
    let disposed = false;

    const header = pageHeader(ctx, { stage: STAGE });
    const body = h("div", { class: "wb-body" });
    root.append(header, body);

    const sftReady = () => project && project.stages.sft.checkpoint_step != null;
    const unlocked = () => project && (pairs.length >= required || project.stages.dpo.checkpoint_step != null || (job && job.kind === STAGE));

    async function loadPairs() {
      try {
        const data = await ctx.api.get("/api/projects/current/preferences");
        pairs = data.pairs || [];
        required = data.required || 4;
      } catch {
        pairs = [];
      }
    }

    async function sample(promptText) {
      if (sampling || !sftReady()) return;
      sampling = true;
      ballotError = "";
      renderBallot();
      try {
        const result = await ctx.api.post("/api/projects/current/ballot", promptText ? { prompt: promptText } : {});
        if (disposed) return;
        ballot = result;
        customPrompt = null;
        memory.projectId = project.id;
        memory.ballot = result;
      } catch (error) {
        ballotError = error.message;
      } finally {
        sampling = false;
        if (!disposed) renderBallot();
      }
    }

    async function pick(which) {
      if (!ballot) return;
      const chosen = which === "a" ? ballot.a.text : ballot.b.text;
      const rejected = which === "a" ? ballot.b.text : ballot.a.text;
      try {
        await ctx.api.post("/api/projects/current/preferences", { prompt: ballot.prompt, chosen, rejected });
        pairs.push({ prompt: ballot.prompt, chosen, rejected });
        ctx.toast(pairs.length >= required ? "Pair banked. The tuning run is unlocked." : `Pair banked · ${Math.max(0, required - pairs.length)} more to unlock tuning`);
        ballot = null;
        memory.ballot = null;
        renderAll();
        sample(null);
      } catch (error) {
        ballotError = error.message;
        renderBallot();
      }
    }

    /* ---------- ballot card ---------- */

    const nameLabel = h("span", { class: "panel-label" });
    const pairLabel = h("span", { class: "panel-label" });
    const askHost = h("div", { class: "wb-ask" });
    const candA = h("div", { class: "wb-candidate" });
    const candB = h("div", { class: "wb-candidate" });
    const sampleButton = h("button", { class: "btn primary wb-sample-btn", onclick: () => sample(customPrompt || (ballot ? ballot.prompt : null)) });
    const footNote = h("div", { class: "panel-note wb-ballot-note" });
    const bars = h("div", { class: "wb-bars" });
    const bankCount = h("span", { class: "wb-bank-count" });
    const tuneValue = h("div", { class: "wb-tune-value" });
    const editButton = h("button", { class: "btn small icon-only ghost", title: "Write your own question", onclick: () => { editing = true; renderBallot(); } }, icon("pencil", 13));
    const shuffleButton = h("button", { class: "btn small icon-only ghost", title: "Ask a different built-in question", onclick: () => sample(null) }, icon("shuffle", 13));
    const ballotCard = h("section", { class: "panel-dark wb-ballot" },
      h("div", { class: "panel-section" },
        h("div", { class: "panel-header" }, h("span", { class: "panel-label" }, "Preference ballot"), h("span", { class: "spacer" }), nameLabel),
        h("div", { class: "panel-header wb-ask-head" }, h("span", { class: "panel-label" }, "User asks"), editButton, shuffleButton, h("span", { class: "spacer" }), pairLabel),
        askHost,
        h("div", { class: "wb-candidates" }, candA, candB)),
      h("div", { class: "panel-section wb-ballot-foot" },
        h("div", { class: "row wb-ballot-actions" }, sampleButton, footNote),
        h("div", { class: "wb-bank-row" },
          h("div", { class: "wb-bank" }, h("span", { class: "panel-label" }, "Pairs banked"), h("div", { class: "row" }, bars, bankCount)),
          h("div", { class: "wb-bank" }, h("span", { class: "panel-label" }, "Tuning run"), tuneValue))));

    function candidate(host, letter, which) {
      const text = ballot ? ballot[which].text : "";
      const same = ballot && ballot.a.text.trim() === ballot.b.text.trim();
      host.classList.toggle("empty", !ballot);
      host.replaceChildren(
        h("span", { class: "panel-label" }, `Candidate ${letter}`),
        h("div", { class: "wb-candidate-text" }, sampling ? h("span", { class: "panel-muted" }, h("span", { class: "spinner" }), " Sampling…") : ballot ? text : h("span", { class: "panel-dim" }, sftReady() ? "An answer from your fine-tuned model appears here." : "Waiting for a fine-tuned checkpoint.")),
        h("button", { class: "wb-pick", disabled: !ballot || sampling || same, onclick: () => pick(which) }, icon("pointer", 13), "This one is better"));
    }

    function renderBallot() {
      nameLabel.textContent = project ? project.name : "";
      pairLabel.textContent = `Pair ${pairs.length + 1}`;
      if (editing) {
        const input = h("input", { class: "wb-ask-input", value: customPrompt || ballot?.prompt || "", placeholder: "Type a question for your model…" });
        const done = () => {
          editing = false;
          customPrompt = input.value.trim() || null;
          renderBallot();
          if (customPrompt) sample(customPrompt);
        };
        input.addEventListener("keydown", (event) => {
          if (event.key === "Enter") done();
          if (event.key === "Escape") {
            editing = false;
            renderBallot();
          }
        });
        askHost.replaceChildren(input, h("button", { class: "btn small primary", onclick: done }, "Ask"));
        setTimeout(() => input.focus());
      } else {
        const question = customPrompt || ballot?.prompt;
        askHost.replaceChildren(h("div", { class: question ? "wb-ask-text" : "wb-ask-text placeholder" }, question || "Sample a pair to get a question, or write your own."));
      }
      candidate(candA, "A", "a");
      candidate(candB, "B", "b");
      editButton.disabled = !sftReady() || sampling;
      shuffleButton.disabled = !sftReady() || sampling;
      clear(sampleButton);
      sampleButton.append(icon("refresh", 13), ballot ? "Sample again" : "Sample two answers");
      sampleButton.disabled = !sftReady() || sampling;
      const same = ballot && ballot.a.text.trim() === ballot.b.text.trim();
      footNote.classList.toggle("wb-warn", Boolean(ballotError || same));
      if (!sftReady()) {
        footNote.replaceChildren(project?.stages?.dpo?.ready_reason || "Fine-tune the model first; the ballot asks your fine-tuned model for answers.", " ",
          h("button", { class: "wb-link", onclick: () => ctx.navigate("sft") }, "Open fine-tuning", icon("chevron-right", 12)));
      } else if (ballotError) footNote.textContent = ballotError;
      else if (same) footNote.textContent = "Both answers came out identical, so there is nothing to compare. Sample again.";
      else footNote.textContent = "Pick the answer you would rather receive. Your choice is the signal the model learns from.";
      clear(bars);
      for (let index = 0; index < required; index += 1) bars.append(h("i", { class: index < pairs.length ? "on" : "" }));
      bankCount.textContent = fmt.int(pairs.length);
      const left = Math.max(0, required - pairs.length);
      tuneValue.textContent = left ? `${left} more to unlock` : "Unlocked";
      tuneValue.classList.toggle("ready", !left);
    }

    function clear(node) {
      node.replaceChildren();
      return node;
    }

    /* ---------- tuning run and banked pairs ---------- */

    async function patchRecipe(values) {
      try {
        apply(await ctx.api.patch("/api/projects/current", { recipes: { [STAGE]: values } }), job);
      } catch (error) {
        ctx.toast(error.message, "error");
        refresh();
      }
    }

    function buildRun() {
      const card = runCard(ctx, {
        stage: STAGE,
        title: "Tuning run",
        caption: "Lower means the model prefers your chosen answers over the rejected ones.",
        startLabel: "Start tuning",
        resumeLabel: "Resume tuning",
        startNote: (p) => `from the fine-tuned model at step ${fmt.int(p.stages.sft.checkpoint_step ?? 0)}`,
        chartHeight: 200,
        emptyText: "The preference loss appears here once tuning starts.",
        footer: (p) => [
          ["Base", `SFT step ${fmt.int(p.stages.dpo.base?.step ?? p.stages.sft.checkpoint_step ?? 0)}`],
          ["Pairs", fmt.int(pairs.length)],
          ["Beta", String(p.recipes.dpo.beta)],
          ["Model", fmt.params(p.estimates.parameters)],
        ],
      });
      const length = runLength(ctx, { stage: STAGE, variant: "spin", values: STEP_VALUES, patch: patchRecipe,
        help: "Tuning starts from the fine-tuned checkpoint and stops at this step. The fine-tuned model itself stays untouched." });
      const betaValue = h("span", { class: "setting-value" });
      const beta = ui.slider({ min: 0.01, max: 0.5, step: 0.01, value: project.recipes.dpo.beta, label: "Beta",
        onInput: (value) => { betaValue.textContent = value.toFixed(2); },
        onChange: (value) => patchRecipe({ beta: Number(value.toFixed(2)) }) });
      const recipe = recipeRows({ recipe: project.recipes.dpo, patch: patchRecipe, lr: { min: 1e-7, max: 1e-4 }, batches: [1, 2, 4, 8, 16] });
      recipe.prepend(ui.settingRow("Beta", "How far the model may drift from the fine-tuned version", h("span", { class: "wb-lr" }, beta, betaValue)));
      const timeline = sampleTimeline(ctx, { stage: STAGE, chat: true, caption: "A reply is sampled at each checkpoint of the tuning run." });
      const element = h("div", { class: "wb-stack wb-dpo-run" },
        card,
        h("div", { class: "columns wb-below" }, length, h("div", { class: "wb-block" }, sectionHead("Recipe"), recipe)),
        timeline);
      return { element, card, length, recipe, beta, betaValue, timeline };
    }

    const pairsHost = h("div", { class: "wb-block wb-pairs" });
    function renderPairs() {
      if (!pairs.length) {
        pairsHost.replaceChildren();
        return;
      }
      const clearButton = h("button", { class: "btn small", onclick: async () => {
        const ok = await ui.confirm({ title: "Clear banked pairs?", message: "This removes every preference pair you picked for this project. Existing tuning checkpoints are kept.", confirm: "Clear pairs", danger: true });
        if (!ok) return;
        try {
          await ctx.api.del("/api/projects/current/preferences");
          pairs = [];
          renderAll();
        } catch (error) {
          ctx.toast(error.message, "error");
        }
      } }, icon("trash", 13), "Clear");
      pairsHost.replaceChildren(
        sectionHead("Banked pairs", h("span", { class: "hint" }, `${fmt.int(pairs.length)} saved`), h("span", { class: "spacer" }), clearButton),
        h("div", { class: "card wb-pair-list" }, pairs.slice().reverse().map((pair, index) => h("div", { class: "wb-pair" },
          h("div", { class: "wb-pair-prompt" }, h("span", { class: "wb-pair-num" }, `#${pairs.length - index}`), pair.prompt),
          h("div", { class: "wb-pair-answers" },
            h("div", { class: "wb-pair-answer chosen" }, h("span", { class: "wb-badge green" }, icon("check", 11), "Chosen"), h("div", {}, pair.chosen)),
            h("div", { class: "wb-pair-answer rejected" }, h("span", { class: "wb-badge" }, icon("x", 11), "Rejected"), h("div", {}, pair.rejected)))))));
    }

    let runBuilt = null;
    const runHost = h("div", { class: "wb-run-host" });
    // Like the reference app, the status row only appears once there is a tuning run to talk about.
    function syncHeader() {
      header.update(project, job);
      header.hidden = Boolean(project) && !unlocked() && !job;
    }

    function renderAll() {
      syncHeader();
      renderBallot();
      renderPairs();
      if (unlocked()) {
        if (!runBuilt) {
          runBuilt = buildRun();
          runHost.replaceChildren(runBuilt.element);
        }
        const running = Boolean(job && job.kind === STAGE);
        runBuilt.card.update(project, job);
        runBuilt.length.update(project, running);
        runBuilt.recipe.update(project.recipes.dpo, running);
        runBuilt.beta.setValue(project.recipes.dpo.beta);
        runBuilt.beta.disabled = running;
        runBuilt.betaValue.textContent = Number(project.recipes.dpo.beta).toFixed(2);
        runBuilt.timeline.update(project);
      } else if (runBuilt) {
        runBuilt.card.destroy();
        runBuilt = null;
        runHost.replaceChildren();
      }
    }

    function apply(nextProject, nextJob, error) {
      job = nextJob;
      if (error) {
        built = null;
        body.replaceChildren(errorBanner(error));
        header.update(null, job);
        return;
      }
      const changed = !project || !nextProject || project.id !== nextProject.id;
      project = nextProject;
      header.update(project, job);
      if (!project) {
        built = null;
        body.replaceChildren(noProjectState(ctx, { message: "Create a project, pre-train and fine-tune it, then teach it your preferences here." }));
        return;
      }
      if (changed) ballot = memory.projectId === project.id ? memory.ballot : null;
      if (!built) {
        built = true;
        body.replaceChildren(h("div", { class: "wb-stack wb-dpo" }, ballotCard, runHost, pairsHost));
      }
      renderAll();
      if (changed && !ballot && sftReady() && !job) sample(null);
    }

    const refresh = watchProject(ctx, async (nextProject, nextJob, error) => {
      if (nextProject) await loadPairs();
      apply(nextProject, nextJob, error);
    });

    ctx.on("job", (message) => {
      job = message.job;
      syncHeader();
      if (runBuilt) {
        runBuilt.card.onJob(job);
        runBuilt.length.update(project, job.kind === STAGE);
        runBuilt.recipe.update(project.recipes.dpo, job.kind === STAGE);
      }
    });
    ctx.on("progress", (message) => runBuilt && runBuilt.card.onProgress(message));
    ctx.on("eval", (message) => runBuilt && runBuilt.card.onEval(message));
    ctx.on("sample", (message) => runBuilt && runBuilt.timeline.onSample(message));
    ctx.on("job_end", (message) => runBuilt && runBuilt.card.onJobEnd(message));

    await refresh();
    return () => {
      disposed = true;
      if (runBuilt) runBuilt.card.destroy();
    };
  },
};
