// Acknowledgements: the original app this edition is modelled on, the datasets, software, fonts, artwork and
// papers it relies on, and the details of this build.
import { clear } from "../dom.js";

const ORIGINAL_URL = "https://languagemodelbuilder.com";

const SOFTWARE = [
  { name: "Python", role: "The language the engine is written in.", licence: "PSF-2.0", url: "https://www.python.org" },
  { name: "PyTorch", role: "Builds, trains and samples the models on the CPU, and on NVIDIA GPUs (CUDA) or Apple GPUs (Metal) where present.", licence: "BSD-3-Clause", url: "https://pytorch.org" },
  { name: "NumPy", role: "Array maths for datasets, statistics and charts.", licence: "BSD-3-Clause", url: "https://numpy.org" },
  { name: "Hugging Face tokenizers", role: "Trains and runs the byte-level BPE tokenizers.", licence: "Apache-2.0", url: "https://github.com/huggingface/tokenizers" },
  { name: "safetensors", role: "Stores checkpoints and the weights inside exported models.", licence: "Apache-2.0", url: "https://github.com/huggingface/safetensors" },
  { name: "Apache Arrow (PyArrow)", role: "Reads the Parquet files many datasets are published as.", licence: "Apache-2.0", url: "https://arrow.apache.org" },
  { name: "tinygrad", role: "Optional: trains on Vulkan GPUs, such as the Adreno in many Arm laptops and phones.", licence: "MIT", url: "https://github.com/tinygrad/tinygrad" },
  { name: "wgpu-py and wgpu-native", role: "Optional: the WebGPU layer tinygrad uses to reach Vulkan.", licence: "BSD-2-Clause · MIT / Apache-2.0", url: "https://github.com/pygfx/wgpu-py", extra: { label: "wgpu-native", url: "https://github.com/gfx-rs/wgpu-native" } },
  { name: "Mesa (Turnip)", role: "Its Turnip driver gives Adreno GPUs Vulkan, including under PRoot on Android. Installed separately, not bundled.", licence: "MIT", url: "https://www.mesa3d.org" },
  { name: "WebKitGTK", role: "Draws this window on Linux. Part of your system, not bundled.", licence: "LGPL-2.1 · BSD", url: "https://webkitgtk.org" },
  { name: "PyGObject", role: "Connects Python to GTK and WebKitGTK on Linux. Part of your system, not bundled.", licence: "LGPL-2.1-or-later", url: "https://pygobject.gnome.org" },
  { name: "pywebview", role: "Opens this window on macOS, drawn by the Mac's own WebKit.", licence: "BSD-3-Clause", url: "https://pywebview.flowrl.com" },
  { name: "PyObjC", role: "Connects pywebview to the Mac's Cocoa and WebKit frameworks.", licence: "MIT", url: "https://github.com/ronaldoussoren/pyobjc" },
];

const BUNDLED = [
  { name: "Inter", by: "Rasmus Andersson", role: "The interface typeface.", licence: "SIL Open Font License 1.1", file: "licenses/Inter-OFL.txt", url: "https://rsms.me/inter/" },
  { name: "JetBrains Mono", by: "JetBrains", role: "The monospaced face for tokens, code and numbers.", licence: "SIL Open Font License 1.1", file: "licenses/JetBrainsMono-OFL.txt", url: "https://www.jetbrains.com/lp/mono/" },
  { name: "Lucide", by: "Lucide contributors", role: "The icons, converted into the app by tools/build_icons.py.", licence: "ISC License", file: "licenses/Lucide-ISC.txt", url: "https://lucide.dev" },
];

const PAPERS = [
  {
    authors: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin",
    year: 2017, title: "Attention Is All You Need", venue: "NeurIPS",
    url: "https://arxiv.org/abs/1706.03762", label: "arXiv 1706.03762",
    why: "Introduced the transformer and the attention mechanism every model here is built from.",
    related: "anatomy-of-a-transformer",
  },
  {
    authors: "Radford, Wu, Child, Luan, Amodei & Sutskever",
    year: 2019, title: "Language Models are Unsupervised Multitask Learners", venue: "OpenAI (the GPT-2 report)",
    url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", label: "PDF",
    why: "The decoder-only GPT-2 design that the models you build follow, at a much smaller scale.",
    related: "pretraining",
  },
  {
    authors: "Sennrich, Haddow & Birch",
    year: 2016, title: "Neural Machine Translation of Rare Words with Subword Units", venue: "ACL",
    url: "https://arxiv.org/abs/1508.07909", label: "arXiv 1508.07909",
    why: "Brought byte-pair encoding to language, the way your tokenizer splits text into pieces.",
    related: "tokenization",
  },
  {
    authors: "Loshchilov & Hutter",
    year: 2019, title: "Decoupled Weight Decay Regularization", venue: "ICLR",
    url: "https://arxiv.org/abs/1711.05101", label: "arXiv 1711.05101",
    why: "AdamW, the optimiser that updates the weights at every training step.",
    related: "how-models-learn",
  },
  {
    authors: "Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin et al.",
    year: 2022, title: "Training language models to follow instructions with human feedback", venue: "NeurIPS",
    url: "https://arxiv.org/abs/2203.02155", label: "arXiv 2203.02155",
    why: "InstructGPT: fine-tuning on demonstrations, then on human preferences, turns a text predictor into an assistant.",
    related: "how-models-learn-behavior",
  },
  {
    authors: "Rafailov, Sharma, Mitchell, Ermon, Manning & Finn",
    year: 2023, title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", venue: "NeurIPS",
    url: "https://arxiv.org/abs/2305.18290", label: "arXiv 2305.18290",
    why: "Learning from preference pairs directly, without a separate reward model. The method behind the DPO step.",
    related: "dpo",
  },
];

const TITLES = {
  "anatomy-of-a-transformer": "Anatomy of a Transformer",
  pretraining: "Pre-training",
  tokenization: "Tokenization",
  "how-models-learn": "How models learn",
  "how-models-learn-behavior": "How models learn behavior",
  dpo: "Direct preference optimization",
  embeddings: "Embeddings",
};

export default {
  id: "acknowledgements",
  title: "Acknowledgements",
  async render(root, ctx) {
    const { h, ui, icon } = ctx;
    let disposed = false;

    const link = (href, text, { quiet = false } = {}) => h("a", {
      class: ["ak-link", quiet ? "quiet" : ""], href, target: "_blank", rel: "noopener noreferrer", title: href,
    }, text, icon("external", 11));
    const goto = (page, text) => h("button", { class: "link-btn ak-goto", onclick: () => ctx.navigate(page) }, text);
    const section = (iconName, title, lede, ...body) => h("section", { class: "ak-section" },
      h("div", { class: "ak-section-head" },
        h("span", { class: "ak-section-icon" }, icon(iconName, 15)),
        h("h2", {}, title)),
      lede ? h("p", { class: "ak-lede" }, lede) : null,
      ...body);

    async function showLicence(item) {
      let text;
      try {
        const response = await fetch(item.file, { cache: "no-cache" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        text = await response.text();
      } catch (error) {
        ui.toast(`Couldn't open the licence: ${error.message}`, "error");
        return;
      }
      await ui.modal({
        title: `${item.name}: ${item.licence}`,
        body: h("pre", { class: "ak-licence-text" }, text),
        actions: [{ label: "Close", primary: true }],
        width: "min(640px, calc(100vw - 40px))",
      });
    }

    /* ---------- the original ---------- */

    const original = h("section", { class: "card ak-original" },
      h("div", { class: "ak-original-main" },
        h("span", { class: "ak-original-icon" }, icon("heart", 20)),
        h("div", {},
          h("span", { class: "section-label" }, "The original"),
          h("h2", {}, "Language Model Builder by Felix Rieseberg"),
          h("p", {},
            "The idea of a book about language models with a workbench attached, where you read how they work and then build one step by step on your own computer, comes from Felix Rieseberg's Mac app. If you are on a Mac, try the real thing."),
          h("div", { class: "row wrap ak-original-links" }, link(ORIGINAL_URL, "languagemodelbuilder.com")))),
      h("div", { class: "ak-disclaimer" },
        icon("info", 14),
        h("p", {},
          h("strong", {}, "This open-source edition is unofficial. "),
          "It is an independent re-implementation, written separately, and is not affiliated with, reviewed or endorsed by Felix Rieseberg. Its text, code and mistakes are its own; please don't send him bug reports about it.")));

    /* ---------- datasets, loaded from the engine's catalogue ---------- */

    const datasetsBody = h("div", { class: "ak-datasets" }, h("div", { class: "ak-loading" }, h("span", { class: "spinner" }), "Loading the dataset catalogue…"));

    function datasetRow(info) {
      return h("li", { class: "ak-ds" },
        h("span", { class: "ak-ds-icon", style: { background: info.color || "var(--accent)" } }, icon(info.icon || "file", 15)),
        h("div", { class: "ak-ds-text" },
          h("div", { class: "ak-ds-name" }, h("strong", {}, info.name), info.tagline ? h("span", { class: "ak-ds-tagline" }, info.tagline) : null),
          h("div", { class: "ak-ds-links" },
            h("span", { class: "ak-licence-chip" }, info.license_url ? link(info.license_url, info.license || "Licence") : info.license || "Licence not stated"),
            info.source_url ? link(info.source_url, info.source_label || "Source", { quiet: true }) : info.source_label ? h("span", {}, info.source_label) : null,
            info.paper_url ? link(info.paper_url, info.paper_label || "Paper", { quiet: true }) : null)));
    }

    function datasetGroup(title, note, items) {
      return h("div", { class: "ak-ds-group" },
        h("div", { class: "ak-ds-group-head" }, h("h3", {}, title), h("span", { class: "ak-count" }, String(items.length))),
        h("p", { class: "ak-ds-group-note" }, note),
        items.length ? h("ul", { class: "ak-ds-list" }, items.map(datasetRow)) : h("p", { class: "hint" }, "None listed."));
    }

    async function loadDatasets() {
      clear(datasetsBody).append(h("div", { class: "ak-loading" }, h("span", { class: "spinner" }), "Loading the dataset catalogue…"));
      let lists;
      try {
        lists = await Promise.all(["pretrain", "finetune"].map((kind) => ctx.api.get(`/api/datasets?kind=${kind}`)));
      } catch (error) {
        if (disposed) return;
        clear(datasetsBody).append(h("div", { class: "banner error" }, icon("alert", 15),
          h("span", {}, `Couldn't load the dataset catalogue: ${error.message}`),
          h("span", { class: "spacer" }),
          h("button", { class: "btn small", onclick: loadDatasets }, "Try again")));
        return;
      }
      if (disposed) return;
      const [pretrain, finetune] = lists.map((result) => (result.datasets || []));
      const custom = [...pretrain, ...finetune].filter((info) => info.custom).length;
      clear(datasetsBody).append(
        datasetGroup("Pre-training", "Plain text the model learns language from.", pretrain.filter((info) => !info.custom)),
        datasetGroup("Fine-tuning", "Conversations and instructions that teach it to answer.", finetune.filter((info) => !info.custom)));
      if (custom) {
        datasetsBody.append(h("p", { class: "hint ak-custom-note" }, icon("info", 12),
          ` ${custom === 1 ? "One dataset you imported yourself isn't" : `${custom} datasets you imported yourself aren't`} listed; their licences are yours to check.`));
      }
    }

    /* ---------- software, fonts and icons ---------- */

    const software = h("ul", { class: "ak-grid" }, SOFTWARE.map((item) => h("li", { class: "ak-item" },
      h("div", { class: "ak-item-head" }, h("strong", {}, item.name), h("span", { class: "ak-licence-chip" }, item.licence)),
      h("p", {}, item.role),
      h("div", { class: "ak-item-links" }, link(item.url, item.url.replace(/^https?:\/\/(www\.)?/, ""), { quiet: true }),
        item.extra ? link(item.extra.url, item.extra.label, { quiet: true }) : null))));

    const bundled = h("ul", { class: "ak-grid" }, BUNDLED.map((item) => h("li", { class: "ak-item" },
      h("div", { class: "ak-item-head" }, h("strong", {}, item.name), h("span", { class: "ak-licence-chip" }, item.licence)),
      h("p", {}, `${item.role} By ${item.by}.`),
      h("div", { class: "ak-item-links" },
        h("button", { class: "link-btn ak-goto", onclick: () => showLicence(item) }, icon("file-text", 12), "Read the licence"),
        link(item.url, item.url.replace(/^https?:\/\/(www\.)?/, "").replace(/\/$/, ""), { quiet: true })))));

    /* ---------- word vectors and the GPT-2 vocabulary ---------- */

    const vectors = h("div", { class: "ak-stack" },
      h("div", { class: "ak-item wide" },
        h("div", { class: "ak-item-head" }, h("strong", {}, "GloVe 6B word vectors"), h("span", { class: "ak-licence-chip" }, "PDDL 1.0")),
        h("p", {},
          "Jeffrey Pennington, Richard Socher and Christopher D. Manning, Stanford NLP Group (2014). Trained on Wikipedia 2014 and Gigaword 5. A small, hand-picked subset of the 50-dimensional vectors drives the word maps in ",
          goto("embeddings", "Embeddings"), "."),
        h("div", { class: "ak-item-links" },
          link("https://nlp.stanford.edu/projects/glove/", "nlp.stanford.edu/projects/glove", { quiet: true }),
          link("https://opendatacommons.org/licenses/pddl/1-0/", "Open Data Commons PDDL 1.0", { quiet: true }),
          link("https://aclanthology.org/D14-1162/", "Paper (EMNLP 2014)", { quiet: true }))),
      h("div", { class: "ak-item wide" },
        h("div", { class: "ak-item-head" }, h("strong", {}, "GPT-2 tokenizer"), h("span", { class: "ak-licence-chip" }, "Modified MIT")),
        h("p", {},
          "Released by OpenAI. Nothing is bundled: only if you pick the GPT-2 vocabulary in ", goto("project-setup", "Project setup"),
          " does the app download its tokenizer.json from Hugging Face."),
        h("div", { class: "ak-item-links" },
          link("https://huggingface.co/openai-community/gpt2", "huggingface.co/openai-community/gpt2", { quiet: true }),
          link("https://github.com/openai/gpt-2/blob/master/LICENSE", "Licence", { quiet: true }))));

    /* ---------- artwork ---------- */

    const artwork = h("div", { class: "ak-art" },
      h("img", { class: "ak-art-thumb", src: "img/wanderer.jpg", alt: "A man in a dark coat stands on a rocky summit, looking out over a sea of fog towards distant mountains.", loading: "lazy", decoding: "async" }),
      h("div", { class: "ak-art-text" },
        h("strong", {}, "Wanderer above the Sea of Fog"),
        h("p", {}, "Caspar David Friedrich, about 1818. Oil on canvas, Hamburger Kunsthalle, Hamburg."),
        h("p", {}, "In the public domain. It greets you on the ", goto("welcome", "Welcome"), " page: someone at the edge of the known, about to find out what lies beneath the fog."),
        h("div", { class: "ak-item-links" },
          link("https://en.wikipedia.org/wiki/Wanderer_above_the_Sea_of_Fog", "About the painting", { quiet: true }),
          link("https://commons.wikimedia.org/wiki/File:Caspar_David_Friedrich_-_Wanderer_above_the_sea_of_fog.jpg", "Wikimedia Commons", { quiet: true }))));

    /* ---------- papers ---------- */

    const papers = h("ol", { class: "ak-papers" }, PAPERS.map((paper) => h("li", { class: "ak-paper" },
      h("div", { class: "ak-paper-cite" },
        h("span", { class: "ak-paper-authors" }, `${paper.authors} (${paper.year}). `),
        h("cite", {}, paper.title), h("span", { class: "ak-paper-venue" }, `. ${paper.venue}.`)),
      h("p", {}, paper.why),
      h("div", { class: "ak-item-links" },
        link(paper.url, paper.label, { quiet: true }),
        h("span", { class: "ak-related" }, "Related: ", goto(paper.related, TITLES[paper.related] || paper.related))))));

    /* ---------- this build ---------- */

    const about = h("section", { class: "panel-dark ak-build" });
    function paintBuild() {
      const app = ctx.state()?.app || {};
      const value = (text, mono = false) => h("span", { class: ["ak-build-value", mono ? "mono" : ""] }, text || "—");
      const revealButton = h("button", { class: "btn small", title: "Open the data folder in your file manager" }, icon("folder-open", 12), `Show in ${app.file_manager || "Files"}`);
      revealButton.addEventListener("click", async () => {
        revealButton.disabled = true;
        try {
          await ctx.api.post("/api/reveal-data", {});
        } catch (error) {
          ui.toast(`Couldn't open the folder: ${error.message}`, "error");
        } finally {
          revealButton.disabled = false;
        }
      });
      clear(about).append(
        h("div", { class: "ak-build-head" }, h("span", { class: "panel-label" }, "This build"), h("span", { class: "spacer" }),
          h("span", { class: "ak-build-badge" }, "Unofficial open-source edition")),
        h("dl", { class: "ak-build-grid" },
          h("dt", {}, "Version"), h("dd", {}, value(app.version ? `v${app.version}` : "")),
          h("dt", {}, "Engine"), h("dd", {}, value(app.engine)),
          h("dt", {}, "Platform"), h("dd", {}, value(app.platform)),
          h("dt", {}, "Data folder"), h("dd", { class: "ak-build-folder" }, value(app.data_dir, true), app.data_dir ? revealButton : null)));
    }

    root.append(h("div", { class: "ak" },
      h("header", { class: "ak-head" },
        h("h1", {}, "Thank you"),
        h("p", {}, "This app stands on other people's work: an app that showed the way, openly licensed data and software, a painting, and decades of research. Here is who to thank, and the terms their work comes with.")),
      original,
      section("database", "Training data",
        "The datasets the workbench can download for you. Each keeps its own licence, and a model trained on one inherits its terms, so read them before you share a model.",
        datasetsBody),
      section("box", "Open-source software", "The engine and the window are built on these projects.", software),
      section("type", "Fonts and icons", "Bundled with the app; their licence texts ship alongside them.", bundled),
      section("scatter", "Word vectors and vocabularies", null, vectors),
      section("sparkles", "Artwork", null, artwork),
      section("scroll", "Papers worth reading", "The research behind the ideas the book explains and the workbench lets you try.", papers),
      about));

    paintBuild();
    ctx.onState(() => {
      if (!disposed) paintBuild();
    });
    await loadDatasets();
    return () => {
      disposed = true;
    };
  },
};
