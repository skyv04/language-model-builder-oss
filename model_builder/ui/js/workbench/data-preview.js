// Dataset previews: "what your model will read" (plain text) and "example training row" (chat).
import { h, clear } from "../dom.js";
import { icon } from "../icons.js";
import { tint } from "../pages/shared.js";

function headLine(label, sub, ...extra) {
  return h("div", { class: "wb-preview-head" },
    h("div", { class: "wb-preview-titles" }, h("span", { class: "section-label" }, label), sub ? h("span", { class: "wb-preview-sub" }, sub) : null),
    h("span", { class: "spacer" }), ...extra);
}

export function previewPlaceholder({ label, message, action, busy = false }) {
  return h("div", { class: "wb-preview" },
    headLine(label),
    h("div", { class: "wb-preview-empty" },
      busy ? h("span", { class: "spinner" }) : h("span", { class: "wb-preview-empty-icon" }, icon("eye", 20)),
      h("p", {}, message), action || null));
}

export function textPreview(preview, info) {
  const body = h("div", { class: "wb-read" });
  preview.documents.forEach((text, index) => {
    if (index) body.append(h("div", { class: "wb-eot" }, h("span", { class: "special-chip wb-eot-chip" }, "<|endoftext|>")));
    body.append(h("div", { class: "wb-read-doc" }, text));
  });
  if (preview.truncated) {
    body.append(h("div", { class: "wb-read-more" }, "· · ·", h("span", {}, `and much more · ${(info.size_label || "").split("·")[0].trim() || "the full corpus"} in total`)));
  }
  return h("div", { class: "wb-preview text" }, headLine("What your model will read", null), body);
}

// One conversation at a time, with a pager when the preview holds several.
export function chatPreview(preview, info) {
  const examples = (preview.examples || []).filter((example) => example.messages && example.messages.length);
  const color = info.color || "#ff2d55";
  let index = 0;
  const counter = h("span", { class: "wb-pager-count" });
  const prev = h("button", { class: "btn small ghost icon-only", title: "Previous example", onclick: () => show(index - 1) }, icon("chevron-left", 14));
  const next = h("button", { class: "btn small ghost icon-only", title: "Next example", onclick: () => show(index + 1) }, icon("chevron-right", 14));
  const pager = h("div", { class: "wb-pager", hidden: examples.length < 2 }, prev, counter, next);
  const thread = h("div", { class: "wb-thread" });
  const element = h("div", { class: "wb-preview chat", style: { "--ds": color, "--ds-soft": tint(color, 0.12), "--ds-line": tint(color, 0.32) } },
    headLine("Example training row", "One conversation the model will learn to continue", pager),
    h("div", { class: "wb-thread-scroll" }, thread),
    h("div", { class: "wb-target-note" },
      h("span", { class: "wb-target-icon" }, icon("target", 16)),
      h("div", {}, h("b", {}, "Only the assistant's words are graded"), h("span", {}, "Everything before a reply is context. The loss is measured on the reply alone, so that is what the model learns to write."))));

  function bubble(message) {
    if (message.role === "system") {
      return h("div", { class: "wb-turn system" }, h("span", { class: "wb-role" }, "System"), h("div", { class: "wb-system-text" }, message.content));
    }
    const user = message.role === "user";
    return h("div", { class: ["wb-turn", user ? "user" : "assistant"] },
      h("span", { class: "wb-role" }, user ? "User" : "Assistant"),
      h("div", { class: "wb-turn-row" },
        user ? null : h("span", { class: "wb-avatar" }, icon("sparkles", 13)),
        h("div", { class: "wb-bubble" }, message.content),
        user ? h("span", { class: "wb-avatar" }, icon("user", 13)) : null));
  }

  function show(next) {
    if (!examples.length) {
      clear(thread).append(h("p", { class: "hint" }, "This dataset has no conversations to show yet."));
      return;
    }
    index = (next + examples.length) % examples.length;
    counter.textContent = `${index + 1} of ${examples.length}`;
    clear(thread).append(...examples[index].messages.map(bubble));
  }
  show(0);
  return element;
}
