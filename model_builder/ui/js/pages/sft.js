// Supervised fine-tuning: run card that starts from the latest pre-training checkpoint.
import { runCard } from "../workbench/run.js";
import { runLength, sampleTimeline } from "../workbench/controls.js";
import { pageHeader, noProjectState, errorBanner, watchProject, datasetMap, recipeRows, sectionHead } from "./shared.js";

const STAGE = "sft";
const STEP_VALUES = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 5000];

export default {
  id: "sft",
  title: "Supervised fine-tuning",
  async render(root, ctx) {
    const { h, fmt, icon, ui } = ctx;
    let project = null;
    let job = ctx.state()?.job || null;
    let datasets = new Map();
    let built = null;

    const header = pageHeader(ctx, { stage: STAGE });
    const body = h("div", { class: "wb-body" });
    root.append(header, body);

    const datasetOf = (p) => datasets.get(p?.data?.finetune) || null;
    const baseStep = (p) => p.stages.sft.base?.step ?? p.stages.pretrain.checkpoint_step;

    async function patchRecipe(values) {
      try {
        apply(await ctx.api.patch("/api/projects/current", { recipes: { [STAGE]: values } }), job);
      } catch (error) {
        ctx.toast(error.message, "error");
        refresh();
      }
    }

    function build() {
      const card = runCard(ctx, {
        stage: STAGE,
        title: "Fine-tuning run",
        caption: "Lower means the assistant replies in the data are getting easier to predict.",
        startLabel: "Start fine-tuning",
        resumeLabel: "Resume fine-tuning",
        startNote: (p) => (baseStep(p) != null ? `from the base model at step ${fmt.int(baseStep(p))}` : "needs a pre-training checkpoint"),
        emptyText: "The fine-tuning loss appears here once the run starts.",
        footer: (p) => {
          const info = datasetOf(p);
          return [
            ["Base", baseStep(p) != null ? `step ${fmt.int(baseStep(p))}` : "—", baseStep(p) == null ? { class: "wb-foot-item missing" } : null],
            ["Data", info ? info.name : "—", info ? null : { class: "wb-foot-item missing" }],
            ["Model", fmt.params(p.estimates.parameters)],
            ["Vocab", fmt.compact(p.tokenizer.vocab_size)],
          ];
        },
      });
      const length = runLength(ctx, {
        stage: STAGE,
        variant: "spin",
        values: STEP_VALUES,
        help: "Fine-tuning resumes from its latest checkpoint and stops once the step counter reaches this target.",
        patch: patchRecipe,
      });
      const recipe = recipeRows({ recipe: project.recipes.sft, patch: patchRecipe, lr: { min: 1e-6, max: 1e-3 }, warmups: [0, 5, 10, 20, 50, 100, 200] });
      const dataHost = h("div", { class: "wb-data-ref" });
      const baseHost = h("div", { class: "wb-data-ref" });
      const left = h("div", { class: "stack wb-col" }, length, h("div", { class: "wb-block" }, sectionHead("Recipe"), recipe));
      const right = h("div", { class: "stack wb-col" },
        h("div", { class: "wb-block" }, sectionHead("Dataset"), dataHost),
        h("div", { class: "wb-block" }, sectionHead("Starts from"), baseHost));
      const timeline = sampleTimeline(ctx, { stage: STAGE, chat: true, caption: "Each checkpoint answers the same question, so you can watch the replies take shape." });
      const element = h("div", { class: "wb-stack" }, card, h("div", { class: "columns wb-below" }, left, right), timeline);
      return { element, card, length, recipe, timeline, dataHost, baseHost };
    }

    function renderRefs() {
      const info = datasetOf(project);
      built.dataHost.replaceChildren(
        info
          ? h("button", { class: "wb-link wb-ref-title", onclick: () => ctx.navigate("finetuning-data") }, info.name, icon("chevron-right", 13))
          : h("button", { class: "wb-link wb-ref-title", onclick: () => ctx.navigate("finetuning-data") }, "Choose a fine-tuning dataset", icon("chevron-right", 13)),
        h("p", { class: "hint" }, info ? info.description : "Fine-tuning needs example conversations. Only the assistant replies are used as training targets."),
      );
      const step = baseStep(project);
      built.baseHost.replaceChildren(
        h("button", { class: "wb-link wb-ref-title", onclick: () => ctx.navigate("pretraining") },
          step != null ? `Pre-trained model · step ${fmt.int(step)}` : "No pre-training checkpoint yet", icon("chevron-right", 13)),
        h("p", { class: "hint" }, step != null
          ? "Fine-tuning copies the latest pre-training checkpoint and keeps training it on conversations. The base model stays untouched."
          : "Pre-train the model first. Fine-tuning always starts from the newest pre-training checkpoint."),
      );
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
        body.replaceChildren(noProjectState(ctx, { message: "Create a project and pre-train it before teaching it to follow a conversation." }));
        return;
      }
      if (!built) {
        built = build();
        body.replaceChildren(built.element);
      }
      const running = Boolean(job && job.kind === STAGE);
      built.card.update(project, job);
      built.length.update(project, running);
      built.recipe.update(project.recipes.sft, running);
      built.timeline.update(project);
      renderRefs();
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
      built.length.update(project, job.kind === STAGE);
      built.recipe.update(project.recipes.sft, job.kind === STAGE);
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
