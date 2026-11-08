// Pre-training: live training run card, run length, recipe and sample timeline.
import { runCard } from "../workbench/run.js";
import { runLength, sampleTimeline } from "../workbench/controls.js";
import {
  pageHeader, noProjectState, errorBanner, watchProject, datasetMap, recipeRows, sectionHead,
  sourcesLabel, paceSeconds, stepsLabel, activeDevice, deviceLabel,
} from "./shared.js";

const STAGE = "pretrain";

export default {
  id: "pretraining",
  title: "Pre-training",
  async render(root, ctx) {
    const { h, fmt, icon, ui } = ctx;
    let project = null;
    let job = ctx.state()?.job || null;
    let datasets = new Map();
    let built = null;

    const header = pageHeader(ctx, { stage: STAGE, onReset: () => built && built.card.reload() });
    const body = h("div", { class: "wb-body" });
    root.append(header, body);

    const datasetOf = (p) => datasets.get(p?.data?.pretrain) || null;

    function corpusLine(p, tokens, total = false) {
      const corpus = p.data.corpus_tokens;
      const head = `≈${fmt.compact(tokens)} training tokens${total ? "" : ""}`;
      if (!corpus) return `${head} · the corpus is measured when it is tokenized`;
      return `${head} · ≈${(tokens / corpus).toFixed(2)} passes over the ${fmt.compact(corpus)}-token corpus`;
    }

    function pipeline(p) {
      const info = datasetOf(p);
      const recipe = p.recipes.pretrain;
      const seconds = paceSeconds(ctx, p, STAGE);
      const step = (label, value, note, extra) => h("div", { class: "wb-pipe-step" },
        h("span", { class: "panel-label" }, label), h("div", { class: "wb-pipe-value" }, value), h("div", { class: "panel-note wb-pipe-note" }, note), extra || null);
      const arrow = () => h("span", { class: "wb-pipe-arrow" }, icon("chevron-right", 16));
      const data = info
        ? step("Data", info.name, info.status?.downloaded ? info.size_label : `${info.size_label} · downloads at start`)
        : step("Data", "Not chosen", "Pick a corpus to read",
          h("button", { class: "btn small primary wb-pipe-link", onclick: () => ctx.navigate("pretraining-data") }, "Choose data", icon("chevron-right", 12)));
      const tokenizer = step("Tokenizer", p.tokenizer.label, p.tokenizer.trained ? "Trained and ready" : "Trained on your data first");
      const model = step("Model", fmt.params(p.estimates.parameters), `${p.config.blocks} blocks · width ${p.config.width} · context ${p.config.context}`);
      const run = step("This run", recipe.mode === "until_stop" ? "Until you pause" : stepsLabel(recipe.steps),
        seconds && recipe.mode !== "until_stop" ? `about ${fmt.duration(seconds * recipe.steps)} on ${deviceLabel(ctx, activeDevice(ctx, p))}` : `batch ${recipe.batch_size} × ${p.config.context} tokens`);
      return h("div", { class: "wb-pipeline" },
        h("div", { class: "wb-pipe-row" }, data, arrow(), tokenizer, arrow(), model, arrow(), run),
        h("p", { class: "panel-note wb-pipe-caption" },
          "Start runs the whole pipeline: it fetches the corpus if needed, trains the tokenizer, turns the text into token ids and then begins training."));
    }

    async function patchRecipe(values) {
      try {
        const next = await ctx.api.patch("/api/projects/current", { recipes: { [STAGE]: values } });
        apply(next, job);
      } catch (error) {
        ctx.toast(error.message, "error");
        refresh();
      }
    }

    function build() {
      const card = runCard(ctx, {
        stage: STAGE,
        title: "Training run",
        caption: "Lower is better. Validation loss is measured on text the model never trains on.",
        startLabel: "Start pre-training",
        resumeLabel: "Resume pre-training",
        startNote: () => "from freshly initialised weights",
        emptyText: "The loss curve draws itself here once training starts.",
        pipeline,
        sigmaText: (p, tokens) => corpusLine(p, tokens),
        footer: (p) => {
          const info = datasetOf(p);
          return [
            ["Data", info ? info.name : "—", info ? null : { class: "wb-foot-item missing" }],
            ["Tokenizer", p.tokenizer.label],
            ["Model", fmt.params(p.estimates.parameters)],
            ["Vocab", fmt.compact(p.tokenizer.vocab_size)],
            ["Sources", info ? sourcesLabel(p, info) : "—"],
          ];
        },
      });
      const length = runLength(ctx, {
        stage: STAGE,
        variant: "slider",
        min: 100,
        max: 50000,
        ticks: [200, 500, 1000, 2000, 5000, 10000, 20000],
        help: "Training resumes from the latest checkpoint and stops once the step counter reaches this target.",
        sigmaText: (p, steps) => corpusLine(p, steps * p.recipes.pretrain.batch_size * p.config.context, true),
        patch: patchRecipe,
      });
      const recipe = recipeRows({ recipe: project.recipes.pretrain, patch: patchRecipe });
      const recipeBlock = h("div", { class: "wb-block" }, sectionHead("Recipe"), recipe);
      const timeline = sampleTimeline(ctx, { stage: STAGE, caption: "A short sample is saved with every checkpoint, so you can watch the writing improve." });
      const element = h("div", { class: "wb-stack" }, card, h("div", { class: "columns wb-below" }, length, recipeBlock), timeline);
      return { element, card, length, recipe, timeline };
    }

    function apply(nextProject, nextJob, error) {
      job = nextJob;
      if (error) {
        built = null;
        body.replaceChildren(errorBanner(error));
        header.update(null, job);
        return;
      }
      project = nextProject;
      header.update(project, job);
      if (!project) {
        if (built) built.card.destroy();
        built = null;
        body.replaceChildren(noProjectState(ctx, { message: "Create a project to pick a corpus and pre-train a model from scratch." }));
        return;
      }
      if (!built) {
        built = build();
        body.replaceChildren(built.element);
      }
      const running = Boolean(job && job.kind === STAGE);
      built.card.update(project, job);
      built.length.update(project, running);
      built.recipe.update(project.recipes.pretrain, running);
      built.timeline.update(project);
    }

    const refresh = watchProject(ctx, async (nextProject, nextJob, error) => {
      if (nextProject) datasets = await datasetMap(ctx);
      apply(nextProject, nextJob, error);
    });

    ctx.on("job", (message) => {
      job = message.job;
      header.update(project, job);
      if (!built) return;
      built.card.onJob(job);
      const running = job.kind === STAGE;
      built.length.update(project, running);
      built.recipe.update(project.recipes.pretrain, running);
    });
    ctx.on("progress", (message) => built && built.card.onProgress(message));
    ctx.on("eval", (message) => built && built.card.onEval(message));
    ctx.on("sample", (message) => built && built.timeline.onSample(message));
    ctx.on("job_end", (message) => built && built.card.onJobEnd(message));

    await refresh();
    return () => {
      if (built) built.card.destroy();
      ui.closePopover();
    };
  },
};
