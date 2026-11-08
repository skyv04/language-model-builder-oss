// Dataset catalog pages (pre-training data, fine-tuning data): tiles, detail card, download, preview, import.
import { h, fmt, clear } from "../dom.js";
import { icon } from "../icons.js";
import * as ui from "../components.js";
import { noProjectState, errorBanner, watchProject, tint, forgetDatasets } from "../pages/shared.js";
import { previewPlaceholder, textPreview, chatPreview } from "./data-preview.js";

const COPY = {
  pretrain: {
    accept: ".txt,.md,text/plain,text/markdown",
    importHint: "Plain text (.txt or .md). Blank lines separate documents. The file stays on this computer.",
    stage: "pretrain",
    job: "pre-training",
    preview: "What your model will read",
    empty: "Create a project to choose the text your model learns language from.",
  },
  finetune: {
    accept: ".jsonl,.json,.csv",
    importHint: "JSONL rows with a messages list, or prompt/response columns (.jsonl, .json or .csv). The file stays on this computer.",
    stage: "sft",
    job: "fine-tuning",
    preview: "Example training row",
    empty: "Create a project to choose the conversations your model practises.",
  },
};

function shortUrl(url) {
  try {
    const parsed = new URL(url);
    const path = parsed.pathname.replace(/^\/(datasets\/)?/, "").replace(/\/$/, "");
    return path && path.length <= 64 ? path : parsed.host.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function linkChip(label, value, href) {
  return h("a", { class: "wb-link-chip", href, target: "_blank", rel: "noopener noreferrer", title: href },
    h("span", {}, label), h("b", {}, value), icon("external", 11));
}

export function dataPage({ id, title, kind }) {
  const copy = COPY[kind];
  return {
    id,
    title,
    async render(root, ctx) {
      root.classList.add("wb-data-page");
      const host = h("div", { class: "wb-data-host" });
      root.append(host);
      let project = null;
      let job = null;
      let datasets = [];
      let selectedId = null;
      let download = ctx.state()?.download || null;
      let notice = null;
      let uploading = false;
      let disposed = false;
      let layout = null;
      let progressRefs = null;
      const previews = new Map();

      const fileInput = h("input", { type: "file", accept: copy.accept, hidden: true, onchange: () => importFile(fileInput.files[0]) });
      const infoOf = (datasetId) => datasets.find((info) => info.id === datasetId) || null;
      const stageBusy = () => Boolean(job && job.kind === copy.stage);

      async function loadDatasets() {
        try {
          datasets = (await ctx.api.get(`/api/datasets?kind=${kind}`)).datasets || [];
        } catch (error) {
          notice = { kind: "error", text: error.message };
        }
      }

      function select(datasetId) {
        selectedId = datasetId;
        notice = null;
        render();
      }

      async function useDataset(info, { thenDownload = false } = {}) {
        notice = null;
        try {
          if (project.data[kind] !== info.id) {
            project = await ctx.api.patch("/api/projects/current", { data: { [kind]: info.id } });
            notice = { kind: "info", text: `${info.name} is now this project's ${kind === "pretrain" ? "pre-training" : "fine-tuning"} data.` };
          }
          if (thenDownload) download = (await ctx.api.post(`/api/datasets/${encodeURIComponent(info.id)}/download`)).download;
        } catch (error) {
          notice = { kind: "error", text: error.message };
        }
        render();
      }

      async function cancelDownload(info) {
        try {
          const result = await ctx.api.post(`/api/datasets/${encodeURIComponent(info.id)}/cancel`);
          if (result && result.download) download = result.download;
        } catch (error) {
          notice = { kind: "error", text: error.message };
        }
        render();
      }

      async function removeDataset(info) {
        const custom = info.custom;
        const ok = await ui.confirm({
          title: custom ? `Remove ${info.name}?` : `Delete the downloaded copy of ${info.name}?`,
          message: custom
            ? "The imported file is deleted from this computer. Projects that use it need another dataset."
            : "The files are removed from this computer. The dataset stays in the catalog and downloads again when it is used.",
          confirm: custom ? "Remove" : "Delete",
          danger: true,
        });
        if (!ok) return;
        try {
          await ctx.api.del(`/api/datasets/${encodeURIComponent(info.id)}`);
          forgetDatasets();
          previews.delete(info.id);
          await loadDatasets();
          if (custom && selectedId === info.id) selectedId = null;
          await ctx.refresh();
        } catch (error) {
          notice = { kind: "error", text: error.message };
        }
        render();
      }

      async function importFile(file) {
        fileInput.value = "";
        if (!file) return;
        const name = await ui.prompt({ title: "Add your own data", message: copy.importHint, value: file.name.replace(/\.[^.]+$/, ""), confirm: "Import" });
        if (!name) return;
        uploading = true;
        render();
        try {
          const result = await ctx.api.upload("/api/datasets/import", file, { kind, name });
          forgetDatasets();
          await loadDatasets();
          selectedId = result.dataset.id;
          notice = { kind: "info", text: `Imported ${result.dataset.name}. Choose “Use this dataset” to train on it.` };
        } catch (error) {
          notice = { kind: "error", text: error.message };
        }
        uploading = false;
        if (!disposed) render();
      }

      async function ensurePreview(info) {
        const downloaded = Boolean(info.status?.downloaded);
        const entry = previews.get(info.id);
        if (entry && entry.downloaded === downloaded && entry.state !== "error") return;
        previews.set(info.id, { state: "loading", downloaded });
        let next;
        try {
          next = { state: "ok", downloaded, data: await ctx.api.get(`/api/datasets/${encodeURIComponent(info.id)}/preview`) };
        } catch (error) {
          next = { state: error.status === 409 ? "missing" : "error", downloaded, error };
        }
        previews.set(info.id, next);
        if (!disposed && selectedId === info.id) render();
      }

      function previewNode(info) {
        const entry = previews.get(info.id);
        if (!entry || entry.state === "loading") return previewPlaceholder({ label: copy.preview, message: "Fetching a few examples…", busy: true });
        if (entry.state === "missing") {
          if (download && download.dataset === info.id && download.state === "downloading") {
            return previewPlaceholder({ label: copy.preview, message: "The preview appears as soon as the download finishes.", busy: true });
          }
          const busy = download && download.state === "downloading";
          return previewPlaceholder({
            label: copy.preview,
            message: "Download this dataset to preview it here.",
            action: h("button", { class: "btn small", disabled: Boolean(busy), onclick: () => startDownload(info) }, icon("download", 13), "Download"),
          });
        }
        if (entry.state === "error") return previewPlaceholder({ label: copy.preview, message: entry.error.message });
        entry.node ||= entry.data.kind === "chat" ? chatPreview(entry.data, info) : textPreview(entry.data, info);
        return entry.node;
      }

      async function startDownload(info) {
        notice = null;
        try {
          download = (await ctx.api.post(`/api/datasets/${encodeURIComponent(info.id)}/download`)).download;
        } catch (error) {
          notice = { kind: "error", text: error.message };
        }
        render();
      }

      function tile(info) {
        const color = info.color || "#0a84ff";
        const inUse = project.data[kind] === info.id;
        const element = ui.datasetTile(info, { selected: info.id === selectedId, downloaded: Boolean(info.status?.downloaded), onClick: () => select(info.id) });
        element.style.setProperty("--ds", color);
        element.style.setProperty("--ds-ring", tint(color, 0.2));
        element.style.setProperty("--ds-soft", tint(color, 0.07));
        element.title = info.name;
        const titleRow = element.querySelector(".tile-title");
        if (titleRow) {
          titleRow.replaceChildren(h("span", { class: "wb-tile-name" }, info.name));
          if (info.recommended) titleRow.append(h("span", { class: "wb-rec", title: "Recommended" }, icon("sparkles", 12)));
        }
        const meta = element.querySelector(".tile-meta");
        if (meta) {
          meta.textContent = inUse ? "In use" : "";
          meta.classList.toggle("in-use", inUse);
        }
        if (download && download.dataset === info.id && download.state === "downloading") {
          element.querySelector(".tile-state")?.replaceChildren(h("span", { class: "spinner wb-tile-spinner" }));
        }
        return element;
      }

      function updateProgress() {
        if (!progressRefs || !download || download.dataset !== progressRefs.id) return;
        const { received = 0, total = null } = download;
        const fraction = total ? Math.min(1, received / total) : null;
        progressRefs.bar.set(fraction || 0);
        progressRefs.bar.classList.toggle("indeterminate", fraction == null);
        progressRefs.label.textContent = total ? `${fmt.bytes(received)} of ${fmt.bytes(total)}` : `${fmt.bytes(received)} so far`;
        progressRefs.pct.textContent = fraction == null ? "" : fmt.pct(fraction, 0);
      }

      function progressBlock(info) {
        const bar = ui.meter(0, { light: true });
        bar.classList.add("wb-dl-meter");
        const label = h("span", { class: "wb-dl-label" });
        const pct = h("span", { class: "wb-dl-pct" });
        progressRefs = { id: info.id, bar, label, pct };
        updateProgress();
        return h("div", { class: "wb-dl" },
          h("div", { class: "wb-dl-row" },
            h("span", { class: "spinner" }), h("b", {}, "Downloading"), label, h("span", { class: "spacer" }), pct,
            h("button", { class: "btn small", onclick: () => cancelDownload(info) }, "Cancel")),
          bar);
      }

      function actions(info) {
        if (download && download.dataset === info.id && download.state === "downloading") return progressBlock(info);
        const inUse = project.data[kind] === info.id;
        const downloaded = Boolean(info.status?.downloaded);
        const otherDownload = Boolean(download && download.state === "downloading");
        const busy = stageBusy();
        const waitTitle = otherDownload ? "Another download is still running" : "";
        let main;
        if (inUse && downloaded) {
          main = h("button", { class: "btn wb-selected", disabled: true }, icon("check", 14), "Selected");
        } else if (inUse) {
          main = h("button", { class: "btn primary", disabled: otherDownload, title: waitTitle, onclick: () => startDownload(info) }, icon("download", 14), "Download now");
        } else if (downloaded) {
          main = h("button", { class: "btn primary", disabled: busy, onclick: () => useDataset(info) }, icon("check", 14), "Use this dataset");
        } else {
          main = h("button", { class: "btn primary", disabled: busy || otherDownload, title: waitTitle, onclick: () => useDataset(info, { thenDownload: true }) }, icon("download", 14), "Download & use");
        }
        const items = [];
        if (info.custom) items.push({ label: "Remove dataset…", icon: "trash", danger: true, disabled: inUse && busy, action: () => removeDataset(info) });
        else if (downloaded) items.push({ label: "Delete downloaded files…", icon: "trash", danger: true, disabled: inUse && busy, action: () => removeDataset(info) });
        const more = items.length
          ? h("button", { class: "btn ghost icon-only wb-more", title: "More actions", onclick: (event) => ui.menu(event.currentTarget, items, { align: "end" }) }, icon("ellipsis", 16))
          : null;
        return h("div", { class: "wb-dataset-actions" }, main, more);
      }

      function detail(info) {
        const color = info.color || "#0a84ff";
        const inUse = project.data[kind] === info.id;
        const downloaded = Boolean(info.status?.downloaded);
        const fetching = Boolean(download && download.dataset === info.id && download.state === "downloading");
        const status = downloaded
          ? h("span", { class: "wb-ds-status ready" }, icon("circle-check", 12), info.custom ? "Imported" : "Downloaded",
            info.status?.bytes ? ` · ${fmt.bytes(info.status.bytes)} on disk` : "")
          : h("span", { class: "wb-ds-status" }, icon("download", 12),
            fetching ? "Downloading to this computer" : inUse ? "Downloads when the run starts" : "Not downloaded yet");
        const chips = [];
        if (info.source_url) chips.push(linkChip(info.source_label || "Source", shortUrl(info.source_url), info.source_url));
        else if (info.source_label) chips.push(h("span", { class: "wb-link-chip static" }, h("span", {}, "Source"), h("b", {}, info.source_label)));
        if (info.paper_url) chips.push(linkChip("Paper", info.paper_label || shortUrl(info.paper_url), info.paper_url));
        if (info.license) {
          chips.push(info.license_url ? linkChip("License", info.license, info.license_url)
            : h("span", { class: "wb-link-chip static" }, h("span", {}, "License"), h("b", {}, info.license)));
        }
        const busyNote = stageBusy() && !inUse
          ? h("div", { class: "wb-dataset-note" }, icon("lock", 13), `${copy.job[0].toUpperCase()}${copy.job.slice(1)} is running. Pause it to switch datasets.`)
          : null;
        return h("section", { class: "wb-dataset", style: { "--ds": color, "--ds-soft": tint(color, 0.1), "--ds-faint": tint(color, 0.025), "--ds-line": tint(color, 0.3) } },
          h("div", { class: "wb-dataset-head" },
            h("span", { class: "wb-dataset-icon" }, icon(info.icon || "file", 22)),
            h("div", { class: "wb-dataset-titles" },
              h("h2", {}, info.name),
              h("div", { class: "wb-dataset-meta" }, info.size_label ? h("span", {}, info.size_label) : null, status)),
            actions(info)),
          info.description || info.tagline ? h("p", { class: "wb-dataset-desc" }, info.description || info.tagline) : null,
          chips.length ? h("div", { class: "wb-link-chips" }, ...chips) : null,
          busyNote,
          h("div", { class: "wb-dataset-preview" }, previewNode(info)));
      }

      function catalog() {
        const current = infoOf(project.data[kind]);
        const add = h("button", { class: "tile add wb-add-tile", disabled: uploading, title: copy.importHint, onclick: () => fileInput.click() },
          h("span", { class: "wb-add-icon" }, uploading ? h("span", { class: "spinner" }) : icon("plus", 13)),
          h("span", {}, uploading ? "Importing…" : "Add your own"));
        return h("aside", { class: "wb-catalog" },
          h("div", { class: "section-label" }, "Dataset catalog"),
          h("div", { class: "wb-catalog-list" }, ...datasets.map(tile), add),
          fileInput,
          h("p", { class: "wb-catalog-foot" },
            current ? ["This project trains on ", h("b", {}, current.name), "."] : "No dataset chosen yet. Pick one to use it for this project."));
      }

      function render() {
        if (disposed || !layout) return;
        const scroller = host.querySelector(".wb-read, .wb-thread-scroll");
        const keep = scroller ? { id: selectedId, top: scroller.scrollTop } : null;
        progressRefs = null;
        if (layout.error) {
          host.replaceChildren(errorBanner(layout.error));
          return;
        }
        if (!project) {
          host.replaceChildren(noProjectState(ctx, { message: copy.empty }));
          return;
        }
        const info = infoOf(selectedId);
        const banner = notice
          ? h("div", { class: ["banner", "wb-data-notice", notice.kind === "error" ? "error" : ""] },
            icon(notice.kind === "error" ? "alert" : "info", 15), h("span", { class: "grow" }, notice.text),
            h("button", { class: "btn ghost small icon-only", title: "Dismiss", onclick: () => { notice = null; render(); } }, icon("x", 13)))
          : null;
        host.replaceChildren(banner || "", h("div", { class: "wb-data" }, catalog(),
          info ? detail(info) : h("section", { class: "wb-dataset wb-dataset-empty" }, h("p", { class: "hint" }, "Choose a dataset from the catalog."))));
        if (info) ensurePreview(info);
        const next = host.querySelector(".wb-read, .wb-thread-scroll");
        if (keep && next && keep.id === selectedId) next.scrollTop = keep.top;
      }

      function defaultSelection() {
        const current = project?.data?.[kind];
        if (current && infoOf(current)) return current;
        return (datasets.find((info) => info.recommended) || datasets[0] || {}).id || null;
      }

      const refresh = watchProject(ctx, async (next, nextJob, error) => {
        job = nextJob;
        layout = { error: error || null };
        const switched = (next?.id ?? null) !== (project?.id ?? null);
        project = next;
        if (project && !error) {
          await loadDatasets();
          if (disposed) return;
          if (switched || !infoOf(selectedId)) selectedId = defaultSelection();
        }
        render();
      });

      ctx.on("download", (message) => {
        const previous = download;
        download = message;
        const same = previous && previous.dataset === message.dataset && previous.state === message.state;
        if (message.state === "downloading") {
          if (same) updateProgress();
          else render();
          return;
        }
        if (same) return;
        const info = infoOf(message.dataset);
        if (message.state === "error") notice = { kind: "error", text: `Download failed${info ? ` (${info.name})` : ""}: ${message.error || "unknown error"}` };
        else if (message.state === "done" && info) notice = { kind: "info", text: `${info.name} is downloaded and ready.` };
        forgetDatasets();
        loadDatasets().then(render);
      });
      const onJob = (nextJob) => {
        const before = stageBusy();
        job = nextJob;
        if (before !== stageBusy()) render();
      };
      ctx.on("job", (message) => onJob(message.job || null));
      ctx.on("job_end", () => onJob(null));

      await refresh();
      return () => {
        disposed = true;
      };
    },
  };
}
