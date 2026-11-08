#!/usr/bin/env python3
"""Mock of the Language Model Builder engine API (spec section 5) for UI development.

Stdlib only. Serves model_builder/ui at / and fakes every endpoint with plausible data:
simulated training jobs that stream SSE events, dataset downloads, NDJSON generation with
top-5 alternatives, attention, tokenisation, DPO ballots and export. Nothing is persisted.

    python3 tools/mock_server.py --port 8772 [--speed 1] [--empty] [--sse-lifetime 0]

--speed scales how fast simulated jobs, downloads and token streams advance, --empty starts with no projects,
--sse-lifetime N closes each /api/events stream after N seconds (the client reconnects); useful
for headless screenshots, which wait for the network to go idle. Mock-only helper:
POST /api/_mock/speed {"speed": 0.05} changes --speed at runtime (e.g. to freeze a download mid-way).
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import queue
import random
import re
import threading
import time
import uuid
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "model_builder" / "ui"
VERSION = "0.1.0-mock"
LOCK = threading.RLock()
ARGS = argparse.Namespace(speed=1.0, empty=False, sse_lifetime=0.0)
STAGES = ("pretrain", "sft", "dpo")
STAGE_PARENT = {"sft": "pretrain", "dpo": "sft"}

PRESETS = {
    "pocket": {"context": 128, "width": 128, "blocks": 4, "heads": 4},
    "starter": {"context": 256, "width": 256, "blocks": 6, "heads": 8},
    "large": {"context": 512, "width": 512, "blocks": 8, "heads": 8},
}
LIMITS = {"context": [16, 1024], "width": [32, 1024], "blocks": [1, 16], "heads": [1, 16]}
BPE_SIZES = (4000, 10000, 16000)
GPT2_VOCAB = 50261
DEFAULT_RECIPES = {
    "pretrain": {"batch_size": 16, "learning_rate": 6e-4, "warmup": 100, "steps": 2000, "mode": "steps",
                 "eval_every": 100, "sample_every": 250, "checkpoint_every": 250},
    "sft": {"batch_size": 8, "learning_rate": 1e-4, "warmup": 20, "steps": 400, "mode": "steps",
            "eval_every": 50, "sample_every": 100, "checkpoint_every": 100},
    "dpo": {"beta": 0.1, "learning_rate": 2e-5, "steps": 60, "batch_size": 4},
}
DEVICES = [
    {"id": "cpu", "kind": "cpu", "name": "Cortex-A720 × 8", "label": "CPU", "detail": "8 cores · PyTorch"},
    {"id": "vulkan", "kind": "gpu", "name": "Adreno (TM) 840", "label": "GPU (Vulkan)", "detail": "Vulkan 1.3 · tinygrad"},
]
# Seconds per optimisation step for the Starter shape at batch 16; scaled by model size.
BASE_STEP_SECONDS = {"cpu": 0.62, "vulkan": 0.21}
STARTER_PARAMS = 7_364_608


def _ds(id, kind, name, tagline, description, license, license_url, source_label, source_url,
        size_label, approx_bytes, icon, color, recommended=False, paper_label=None, paper_url=None, **extra):
    info = dict(id=id, kind=kind, name=name, tagline=tagline, description=description, license=license,
                license_url=license_url, source_label=source_label, source_url=source_url,
                paper_label=paper_label, paper_url=paper_url, size_label=size_label,
                approx_bytes=approx_bytes, icon=icon, color=color, recommended=recommended, custom=False)
    info.update(extra)
    return info


CATALOG = [
    _ds("tinystories", "pretrain", "TinyStories", "Short stories in very simple English",
        "Machine-written children's stories that only use words a small child would know. Because the "
        "language is so plain, even a tiny model learns grammar, names and simple plots within a few "
        "thousand steps. The best first corpus.",
        "CDLA-Sharing-1.0", "https://cdla.dev/sharing-1-0/", "Hugging Face", "https://huggingface.co/datasets/roneneldan/TinyStories",
        "19 MB · 21k stories", 19_400_000, "moon", "#8e5cf7", True,
        "TinyStories paper (2023)", "https://arxiv.org/abs/2305.07759", documents=21_000),
    _ds("wikitext-2", "pretrain", "WikiText-2", "Featured Wikipedia articles",
        "A cleaned set of good and featured English Wikipedia articles. Facts, dates and long sentences: "
        "a much harder read than stories, so expect higher loss and more confident nonsense.",
        "CC BY-SA 3.0", "https://creativecommons.org/licenses/by-sa/3.0/", "Hugging Face", "https://huggingface.co/datasets/Salesforce/wikitext",
        "13 MB · 600 articles", 13_100_000, "globe", "#0a84ff", False,
        "Pointer Sentinel paper (2016)", "https://arxiv.org/abs/1609.07843", documents=600),
    _ds("tiny-shakespeare", "pretrain", "Tiny Shakespeare", "Forty thousand lines of plays",
        "Every play speech run together into one long script. Small and quick to train on; the model soon "
        "invents characters and archaic verse, though the plots never quite make sense.",
        "Public domain", "https://en.wikipedia.org/wiki/Public_domain", "GitHub", "https://github.com/karpathy/char-rnn",
        "1.1 MB · 1 .txt", 1_115_000, "theater", "#ff375f", documents=1),
    _ds("simple-wikipedia", "pretrain", "Simple English Wikipedia", "Encyclopedia articles in plain words",
        "Articles written for people learning English: short sentences and common words, covering every "
        "subject. A good middle ground between stories and the full encyclopedia.",
        "CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/", "Hugging Face", "https://huggingface.co/datasets/wikimedia/wikipedia",
        "48 MB · 230k articles", 48_000_000, "books", "#30b0c7", documents=230_000),
    _ds("childrens-classics", "pretrain", "Children's classics", "Sixty out-of-copyright books",
        "Novels and fairy tales from Project Gutenberg whose copyright has expired. Richer vocabulary than "
        "TinyStories and older spelling, so the model needs longer to sound fluent.",
        "Public domain (US)", "https://www.gutenberg.org/policy/license.html", "Project Gutenberg", "https://www.gutenberg.org/",
        "9 MB · 60 books", 9_300_000, "book", "#ff9f0a", documents=60),
    _ds("arxiv-abstracts", "pretrain", "arXiv abstracts", "Summaries of science papers",
        "The opening summaries of physics, maths and computer-science papers. Dense jargon and LaTeX: an "
        "interesting stress test for a small tokenizer.",
        "CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/", "Kaggle", "https://www.kaggle.com/datasets/Cornell-University/arxiv",
        "30 MB · 120k abstracts", 30_000_000, "atom", "#5e5ce6", documents=120_000),
    _ds("rosetta-code", "pretrain", "Rosetta Code", "Small programs in many languages",
        "The same programming tasks solved in hundreds of languages. Teaches the model indentation, brackets "
        "and the rhythm of source code rather than prose.",
        "GFDL 1.2", "https://www.gnu.org/licenses/old-licenses/fdl-1.2.html", "Rosetta Code", "https://rosettacode.org/",
        "24 MB · 1.2k tasks", 24_000_000, "code", "#34c759", documents=1_200),
    _ds("everyday-conversations", "finetune", "Everyday Conversations", "Short friendly chats",
        "Two thousand brief exchanges between a user and a helpful assistant about ordinary things: greetings, "
        "small talk, simple questions. Ideal for teaching a small model the turn-taking of a chat.",
        "Apache-2.0", "https://www.apache.org/licenses/LICENSE-2.0", "Hugging Face", "https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k",
        "2.2k chats", 2_600_000, "bubbles", "#ff2d55", True, rows=2_200),
    _ds("no-robots", "finetune", "No Robots", "Instructions answered by people",
        "Ten thousand instructions and answers written by human annotators: brainstorming, rewriting, "
        "classification, open questions. Varied and high quality.",
        "CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/", "Hugging Face", "https://huggingface.co/datasets/HuggingFaceH4/no_robots",
        "9.5k rows", 16_000_000, "people", "#ff9f0a", rows=9_500),
    _ds("dolly-15k", "finetune", "Dolly 15k", "Employee-written instructions",
        "Instruction and response pairs written by thousands of employees of a software company, across "
        "categories such as summarisation, classification and creative writing.",
        "CC BY-SA 3.0", "https://creativecommons.org/licenses/by-sa/3.0/", "Hugging Face", "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "15k rows", 13_000_000, "list", "#34c759", rows=15_000),
    _ds("oasst1", "finetune", "OpenAssistant", "Volunteer conversation trees",
        "Multi-turn conversations written and ranked by volunteers in many languages. We keep the "
        "highest-rated English branch of each tree.",
        "Apache-2.0", "https://www.apache.org/licenses/LICENSE-2.0", "Hugging Face", "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "10k chats", 21_000_000, "chat", "#0a84ff", False, "OpenAssistant paper (2023)", "https://arxiv.org/abs/2304.07327", rows=10_000),
    _ds("gsm8k", "finetune", "GSM8K", "Grade-school maths with worked answers",
        "Word problems with step-by-step solutions. Small models rarely get the arithmetic right, but they do "
        "pick up the shape of a worked answer.",
        "MIT", "https://opensource.org/license/mit", "Hugging Face", "https://huggingface.co/datasets/openai/gsm8k",
        "7.5k rows", 4_300_000, "math", "#ff9500", False, "Verifier paper (2021)", "https://arxiv.org/abs/2110.14168", rows=7_500),
    _ds("code-alpaca", "finetune", "Code Alpaca", "Short coding requests",
        "Twenty thousand small programming instructions with code answers. Pairs well with a code-heavy "
        "pre-training corpus.",
        "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/", "Hugging Face", "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k",
        "20k rows", 8_000_000, "code", "#30b0c7", rows=20_000),
]

# ---------------------------------------------------------------- original sample text

STORY_DOCS = [
    "Pip the hedgehog found a red button under the big oak tree. He rolled it home and showed his sister. "
    "\"It could be a wheel,\" she said. They built a tiny cart and pulled it around the garden until the sun went down.",
    "Mia wanted to bake a cake for her grandpa. She put in flour, two eggs and far too much sugar. The cake was very sweet, "
    "but Grandpa ate every crumb and said it was the best cake in the whole world.",
    "A little cloud could not make any rain. The other clouds laughed at him. One hot day a flower looked up and asked for help. "
    "The little cloud tried very hard, and a single drop fell. The flower smiled, and the cloud felt proud.",
    "Tom and his dog Biscuit went to the beach. Biscuit chased the waves and Tom built a castle with a moat. "
    "When the tide came in, the moat filled with water. \"Now it is a real castle,\" Tom said.",
]
WIKI_DOCS = [
    "The Varden Bridge is a stone arch bridge that crosses the river Ell in the north of the county. It was completed in 1842 "
    "after six years of work and replaced an older wooden crossing that had been washed away by floods.",
    "Copperleaf moths are a family of small moths found in temperate forests. Adults are active at dusk and are attracted to "
    "light. The larvae feed mainly on the leaves of birch and alder.",
]
SHAKESPEARE_DOCS = [
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\n"
    "You are all resolved rather to die than to famish?",
]
CODE_DOCS = [
    "def fizzbuzz(n):\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            print(\"FizzBuzz\")\n"
    "        elif i % 3 == 0:\n            print(\"Fizz\")\n        else:\n            print(i)",
]
CHAT_EXAMPLES = [
    [{"role": "user", "content": "Hi! What should I have for breakfast?"},
     {"role": "assistant", "content": "Good morning! If you have a few minutes, porridge with some fruit is warm and filling. "
                                      "In a hurry? A banana and a slice of toast will keep you going."}],
    [{"role": "user", "content": "Can you help me name my new goldfish?"},
     {"role": "assistant", "content": "Of course! How about Bubbles, Sunny or Captain Fin? If it is a bit shy, Whisper could be nice too."}],
    [{"role": "user", "content": "I'm bored. Any ideas?"},
     {"role": "assistant", "content": "You could take a short walk, draw the view from your window, or try cooking something new. "
                                      "Which sounds most fun?"}],
]
PREVIEWS = {"tinystories": STORY_DOCS, "wikitext-2": WIKI_DOCS, "simple-wikipedia": WIKI_DOCS,
            "tiny-shakespeare": SHAKESPEARE_DOCS, "rosetta-code": CODE_DOCS}

CONTINUATIONS = [
    " there was a little girl named Lily. She loved to play in the park with her red ball. One day, the ball rolled into the pond "
    "and a friendly duck pushed it back to her. Lily said thank you and they played together until it was time to go home.",
    " the sun was shining and the birds were singing. Ben went outside to look for his kite. He found it stuck in a tree, "
    "so he asked his dad for help. Together they pulled it down and flew it high over the hill.",
    " a small boy who wanted to see the sea. He packed a sandwich and walked all morning. When he finally saw the water, "
    "it was bigger and bluer than anything he had imagined, and he laughed out loud.",
    " a cat named Mittens who did not like the rain. Every time it rained she hid under the bed. One day her friend the dog "
    "showed her a puddle full of shiny leaves, and Mittens decided that rain was not so bad after all.",
]
CHAT_REPLIES = [
    "Hello! I am happy to help. What would you like to talk about today?",
    "That sounds like a lovely idea. You could start small and see how it feels, then try a little more each day.",
    "I think the best thing to do is take a deep breath and make a short list. Then you can do one thing at a time.",
    "Sure! Here is a simple one: a sleepy owl wanted to stay up late, but the stars told her a story and she fell asleep.",
    "Good question. A cloud is made of tiny drops of water that float in the air. When the drops get heavy, they fall as rain.",
]
BALLOT_PROMPTS = [
    "Tell me a short story about a brave mouse.",
    "What is a good way to start the day?",
    "Why is the sky blue?",
    "Can you suggest a name for a friendly robot?",
    "How do I make a paper boat?",
    "Write two lines about the sea.",
]
SAMPLE_PROMPTS = {"pretrain": "Once upon a time", "sft": "<|user|>\nTell me about the moon.<|end|>\n<|assistant|>\n",
                  "dpo": "<|user|>\nWhat makes a good friend?<|end|>\n<|assistant|>\n"}
EARLY_GIBBERISH = [
    " the the was and a , . the he to she the the , and was",
    " was she a day . The little he the the friend and to the",
    " upon a time , there was a girl . She was a big the dog and she",
]
COMMON_WORDS = [" the", " a", " and", " was", " to", ",", ".", " she", " he", " it", " little", " big", " day", " happy",
                " friend", " said", " went", " saw", " very", " one", " time", " they", " with", " her", " his", " in",
                " on", " play", " home", " tree", " ball", " dog", " cat", " sun", " park", " mom", " smiled", "\n"]

# ---------------------------------------------------------------- events


class Hub:
    """Fan-out of SSE messages to every connected /api/events client."""

    def __init__(self):
        self.lock = threading.Lock()
        self.subscribers: set[queue.Queue] = set()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self.lock:
            self.subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.subscribers.discard(q)

    def publish(self, message: dict) -> None:
        with self.lock:
            targets = list(self.subscribers)
        for q in targets:
            try:
                q.put_nowait(message)
            except queue.Full:
                pass


HUB = Hub()


def emit(kind: str, **payload) -> None:
    HUB.publish({"type": kind, **payload})


# ---------------------------------------------------------------- state

STATE: dict = {
    "settings": {"device": "auto", "reading_pane": False, "completed": [], "last_page": "welcome",
                 "sidebar_hidden": False, "keep_training": False},
    "projects": {},          # id -> internal project dict (insertion ordered)
    "current": None,
    "job": None,             # public Job dict while a job runs
    "download": None,
    "downloaded": {"tinystories", "everyday-conversations", "tiny-shakespeare"},
    "custom": [],            # imported DatasetInfo dicts
    "custom_text": {},       # dataset id -> preview payload
}
JOB_CONTROL = {"stop": threading.Event(), "thread": None, "project": None}
DOWNLOAD_CONTROL = {"cancel": threading.Event()}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "model"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def parameter_count(config: dict, vocab: int) -> int:
    d = config["width"]
    return vocab * d + config["context"] * d + config["blocks"] * (12 * d * d + 13 * d) + 2 * d


def training_memory(config: dict, vocab: int, batch: int) -> int:
    params = parameter_count(config, vocab)
    tokens = batch * config["context"]
    return int(params * 16 + tokens * config["blocks"] * config["width"] * 80
               + batch * config["heads"] * config["context"] ** 2 * config["blocks"] * 12 + tokens * vocab * 12)


def tokenizer_label(kind: str, vocab: int) -> str:
    return "GPT-2 BPE" if kind == "gpt2" else f"Fast {round(vocab / 1000)}K BPE"


def dataset_info(dataset_id: str | None) -> dict | None:
    for info in CATALOG + STATE["custom"]:
        if info["id"] == dataset_id:
            return info
    return None


def dataset_status(info: dict) -> dict:
    downloaded = info["custom"] or info["id"] in STATE["downloaded"]
    status = {"downloaded": downloaded, "bytes": info["approx_bytes"] if downloaded else 0}
    if info["kind"] == "pretrain":
        status["documents"] = info.get("documents") if downloaded else None
    else:
        status["rows"] = info.get("rows") if downloaded else None
    return status


def empty_stage() -> dict:
    return {"state": "empty", "step": 0, "target": None, "loss": None, "val_loss": None, "lr": None, "tok_per_s": None,
            "elapsed": 0.0, "tokens_seen": 0, "checkpoint_step": None, "device": None, "base": None, "ready_reason": None,
            "complete": False}


def new_project(name: str) -> dict:
    pid = f"{slugify(name)}-{uuid.uuid4().hex[:4]}"
    return {
        "id": pid, "name": name, "created": now_iso(),
        "config": dict(PRESETS["starter"], preset="starter"),
        "tokenizer": {"kind": "bpe", "vocab_size": 10000, "trained": False},
        "data": {"pretrain": None, "finetune": None, "corpus_tokens": None},
        "recipes": deepcopy(DEFAULT_RECIPES),
        "calibration": {},       # device -> seconds per step (for the current shape)
        "stages": {stage: empty_stage() for stage in STAGES},
        "metrics": {stage: {"train": [], "val": [], "lr": []} for stage in STAGES},
        "samples": {stage: [] for stage in STAGES},
        "preferences": [],
    }


def vocab_of(project: dict) -> int:
    tok = project["tokenizer"]
    return GPT2_VOCAB if tok["kind"] == "gpt2" else tok["vocab_size"]


def step_seconds(project: dict) -> dict:
    """Estimated (or 'calibrated') seconds per step on each device for the current shape and batch."""
    scale = (parameter_count(project["config"], vocab_of(project)) / STARTER_PARAMS) * \
        (project["recipes"]["pretrain"]["batch_size"] * project["config"]["context"] / 4096)
    out = {"cpu": None, "vulkan": None, "cuda": None}
    for device, seconds in BASE_STEP_SECONDS.items():
        measured = project["calibration"].get(device)
        out[device] = round(measured if measured else seconds * scale, 4)
    return out


def auto_device(project: dict) -> str:
    """What "Auto" really uses: the fastest calibrated device, else CUDA if present, else the CPU."""
    measured = {k: v for k, v in project["calibration"].items() if v}
    if measured:
        return min(measured, key=measured.get)
    return "cuda" if any(device["id"] == "cuda" for device in DEVICES) else "cpu"


def chosen_device(project: dict) -> str:
    device = STATE["settings"]["device"]
    return device if device in ("cpu", "vulkan") else auto_device(project)


def missing_for(project: dict, stage: str) -> str | None:
    if stage == "pretrain":
        return None if project["data"]["pretrain"] else "Choose a pre-training dataset first."
    if stage == "sft":
        if project["stages"]["pretrain"]["checkpoint_step"] is None:
            return "Pre-train the model first; fine-tuning starts from its latest checkpoint."
        return None if project["data"]["finetune"] else "Choose a fine-tuning dataset first."
    if project["stages"]["sft"]["checkpoint_step"] is None:
        return "Fine-tune the model first; preference tuning starts from the SFT checkpoint."
    if len(project["preferences"]) < 4:
        return "Bank at least four preference pairs to unlock a tuning run."
    return None


def stage_view(project: dict, stage: str) -> dict:
    raw = project["stages"][stage]
    info = {key: value for key, value in raw.items() if key != "complete"}
    recipe = project["recipes"][stage]
    job = STATE["job"]
    running = bool(job and job["stage"] == stage and JOB_CONTROL["project"] == project["id"] and job["kind"] != "calibrate")
    target = None if recipe.get("mode") == "until_stop" else recipe["steps"]
    info["target"] = target
    if running:
        info["state"] = "running"
    elif raw["checkpoint_step"] is not None:
        done = target is not None and raw["checkpoint_step"] >= target
        info["state"] = "complete" if done else "paused"
    else:
        reason = missing_for(project, stage)
        info["state"] = "ready" if reason is None else "empty"
        info["ready_reason"] = reason
    if stage in STAGE_PARENT and info["base"] is None:
        parent = project["stages"][STAGE_PARENT[stage]]
        if parent["checkpoint_step"] is not None:
            info["base"] = {"stage": STAGE_PARENT[stage], "step": parent["checkpoint_step"]}
    return info


def project_status(project: dict) -> dict:
    job = STATE["job"]
    if job and JOB_CONTROL["project"] == project["id"]:
        if job["kind"] == "calibrate":
            return {"label": "Measuring speed…", "tone": "blue", "pulse": True}
        label = {"pretrain": "Pre-training…", "sft": "Fine-tuning…", "dpo": "Preference tuning…"}[job["kind"]]
        return {"label": label, "tone": "red", "pulse": True}
    for stage, done in (("dpo", "Preference-tuned"), ("sft", "Fine-tuned"), ("pretrain", "Pre-trained")):
        raw = project["stages"][stage]
        if raw["checkpoint_step"] is not None:
            if stage_view(project, stage)["state"] == "complete":
                return {"label": done, "tone": "green", "pulse": False}
            return {"label": f"Paused at step {raw['checkpoint_step']:,}", "tone": "orange", "pulse": False}
    if project["data"]["pretrain"]:
        return {"label": "Data selected", "tone": "blue", "pulse": False}
    return {"label": "Not trained yet", "tone": "grey", "pulse": False}


def summary_view(project: dict) -> dict:
    return {"id": project["id"], "name": project["name"], "created": project["created"], "status": project_status(project)}


def project_view(project: dict) -> dict:
    vocab = vocab_of(project)
    config = project["config"]
    batch = project["recipes"]["pretrain"]["batch_size"]
    params = parameter_count(config, vocab)
    memory = training_memory(config, vocab, batch)
    total = 7.5e9
    warning = None
    if memory > 0.45 * total:
        warning = (f"Training this shape at batch {batch} needs roughly {memory / 1e9:.1f} GB, but this machine has "
                   f"{total / 1e9:.1f} GB in total. Pick a smaller shape or batch size.")
    calibrated = bool(project["calibration"])
    # Like the real engine: no guesses before calibration, only measured seconds per step.
    seconds = step_seconds(project) if calibrated else {"cpu": None, "vulkan": None, "cuda": None}
    pretrain_info = dataset_info(project["data"]["pretrain"])
    corpus_tokens = project["data"]["corpus_tokens"]
    tok = project["tokenizer"]
    view = summary_view(project)
    view.update(
        config=dict(config),
        tokenizer={"kind": tok["kind"], "vocab_size": vocab, "trained": tok["trained"] or tok["kind"] == "gpt2",
                   "label": tokenizer_label(tok["kind"], vocab)},
        data={"pretrain": project["data"]["pretrain"], "finetune": project["data"]["finetune"],
              "corpus_tokens": corpus_tokens,
              "corpus_documents": pretrain_info.get("documents") if (pretrain_info and corpus_tokens) else None,
              "corpus_dataset": project["data"]["pretrain"] if corpus_tokens else None,
              "corpus_label": (f"{pretrain_info['name']} · {pretrain_info['size_label']}" if pretrain_info else None)},
        recipes=deepcopy(project["recipes"]),
        estimates={"parameters": params, "weights_bytes": params * 4, "training_memory_bytes": memory,
                   "memory_warning": warning, "tokens_per_step": batch * config["context"],
                   "step_seconds": seconds,
                   "calibrated": calibrated, "calibrated_at": project.get("calibrated_at"),
                   "auto_device": auto_device(project)},
        locked=project["stages"]["pretrain"]["checkpoint_step"] is not None,
        stages={stage: stage_view(project, stage) for stage in STAGES},
        limits=deepcopy(LIMITS),
        presets=deepcopy(PRESETS),
        preferences=len(project["preferences"]),
        folder=f"~/.local/share/language-model-builder/projects/{project['id']}",
    )
    return view


def current_project() -> dict:
    project = STATE["projects"].get(STATE["current"]) if STATE["current"] else None
    if not project:
        raise ApiError(409, "No project is open. Create one to get started.")
    return project


def state_view() -> dict:
    return {
        "app": {"version": VERSION, "engine": "mock", "data_dir": "~/.local/share/language-model-builder",
                "platform": "Linux aarch64 (mock)"},
        "settings": deepcopy(STATE["settings"]),
        "devices": deepcopy(DEVICES),
        "projects": [summary_view(p) for p in STATE["projects"].values()],
        "current": STATE["current"],
        "job": deepcopy(STATE["job"]),
        "download": deepcopy(STATE["download"]),
    }

# ---------------------------------------------------------------- curves and seeded projects


def loss_curve(stage: str, step: float) -> float:
    if stage == "pretrain":
        return 1.55 + 2.4 * math.exp(-step / 600) + 5.25 * math.exp(-step / 40)
    if stage == "sft":
        return 1.35 + 1.25 * math.exp(-step / 120)
    return 0.33 + 0.36 * math.exp(-step / 30)


def noisy_loss(stage: str, step: float, rng: random.Random) -> float:
    spread = {"pretrain": 0.05, "sft": 0.06, "dpo": 0.03}[stage]
    return max(0.02, loss_curve(stage, step) + rng.gauss(0, spread))


def val_loss(stage: str, step: float, rng: random.Random) -> float:
    gap = {"pretrain": 0.08, "sft": 0.1, "dpo": 0.04}[stage]
    return max(0.02, loss_curve(stage, step) + gap + rng.gauss(0, 0.015))


def lr_at(recipe: dict, step: int, target: int | None) -> float:
    peak = recipe["learning_rate"]
    warm = recipe.get("warmup", 0)
    if warm and step < warm:
        return peak * (step + 1) / warm
    total = target or max(step + 1, recipe.get("steps", 1000))
    progress = min(1.0, (step - warm) / max(1, total - warm))
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def cadence(recipe: dict, key: str, stage: str) -> int:
    defaults = {"eval_every": 10, "sample_every": 20, "checkpoint_every": 20}
    return int(recipe.get(key) or defaults[key]) if stage != "dpo" else defaults[key]


def sample_text(stage: str, step: int, rng: random.Random) -> str:
    if stage == "pretrain":
        if step < 300:
            return rng.choice(EARLY_GIBBERISH)
        text = rng.choice(CONTINUATIONS)
        if step < 900:
            words = text.split(" ")
            for i in range(1, len(words), 4):
                words[i] = rng.choice(["the", "was", "and", "a", "she", "little"])
            text = " ".join(words)
        return text.strip()
    return rng.choice(CHAT_REPLIES)


def seed_stage(project: dict, stage: str, upto: int, seed: int) -> None:
    rng = random.Random(seed)
    recipe = project["recipes"][stage]
    target = recipe.get("steps")
    metrics = project["metrics"][stage]
    every = 10 if stage == "pretrain" else (5 if stage == "sft" else 1)
    for step in range(0, upto + 1, every):
        metrics["train"].append([step, round(noisy_loss(stage, step, rng), 4)])
        metrics["lr"].append([step, lr_at(recipe, step, target)])
    for step in range(cadence(recipe, "eval_every", stage), upto + 1, cadence(recipe, "eval_every", stage)):
        metrics["val"].append([step, round(val_loss(stage, step, rng), 4)])
    for step in range(cadence(recipe, "sample_every", stage), upto + 1, cadence(recipe, "sample_every", stage)):
        project["samples"][stage].append({"step": step, "prompt": SAMPLE_PROMPTS[stage],
                                          "text": sample_text(stage, step, rng), "time": time.time() - (upto - step) * 0.3})
    seconds = BASE_STEP_SECONDS["vulkan"]
    tokens_per_step = recipe["batch_size"] * project["config"]["context"]
    project["stages"][stage].update(
        step=upto, loss=metrics["train"][-1][1], val_loss=metrics["val"][-1][1] if metrics["val"] else None,
        lr=metrics["lr"][-1][1], tok_per_s=round(tokens_per_step / seconds), elapsed=round(upto * seconds, 1),
        tokens_seen=upto * tokens_per_step, checkpoint_step=upto, device="vulkan",
        base=({"stage": STAGE_PARENT[stage], "step": project["stages"][STAGE_PARENT[stage]]["checkpoint_step"]}
              if stage in STAGE_PARENT else None))


def seed_projects() -> None:
    if ARGS.empty:
        return
    helper = new_project("Story Helper")
    helper["created"] = "2025-01-04T18:20:00"
    helper["data"].update(pretrain="tinystories", finetune="everyday-conversations", corpus_tokens=4_652_000)
    helper["tokenizer"]["trained"] = True
    seed_stage(helper, "pretrain", 2000, 1)
    seed_stage(helper, "sft", 400, 2)
    helper["calibration"] = {"cpu": 0.64, "vulkan": 0.22}
    helper["calibrated_at"] = time.time() - 2 * 86400
    helper["preferences"] = [
        {"prompt": BALLOT_PROMPTS[0], "chosen": "A small mouse named Tilly climbed the tall clock to save her friend, and everyone cheered.",
         "rejected": "Mouse mouse the clock was was big and the the."},
        {"prompt": BALLOT_PROMPTS[1], "chosen": "Drink a glass of water, open the curtains and take a slow breath before you start.",
         "rejected": "The day is a day and you start it by the day."},
    ]
    demo = new_project("Bedtime Stories")
    demo["created"] = "2025-01-06T09:12:00"
    demo["data"].update(pretrain="tinystories", finetune="everyday-conversations", corpus_tokens=4_652_000)
    demo["tokenizer"]["trained"] = True
    seed_stage(demo, "pretrain", 1200, 3)
    fresh = new_project("Fresh Model")
    for project in (helper, fresh, demo):
        STATE["projects"][project["id"]] = project
    STATE["current"] = demo["id"]


# ---------------------------------------------------------------- jobs


def set_phase(job: dict, phase: str, label: str, progress: float | None = None) -> None:
    with LOCK:
        job.update(phase=phase, phase_label=label, progress=progress, state="running")
    emit("job", job=deepcopy(job))


def finish(job: dict, result: str, error: str | None = None) -> None:
    with LOCK:
        STATE["job"] = None
        JOB_CONTROL["project"] = None
        job["state"] = "stopped"
    payload = {"job": deepcopy(job), "result": result}
    if error:
        payload["error"] = error
    emit("job_end", **payload)
    emit("state")


def start_job(kind: str) -> dict:
    with LOCK:
        if STATE["job"]:
            raise ApiError(409, "Another job is already running. Pause it before starting a new one.")
        project = current_project()
        stage = None if kind == "calibrate" else kind
        if stage:
            reason = missing_for(project, stage)
            if reason:
                raise ApiError(409, reason)
            info = stage_view(project, stage)
            if info["state"] == "complete":
                raise ApiError(409, "This run already reached its step target. Raise the target to keep training.")
        device = chosen_device(project)
        target = stage_view(project, stage)["target"] if stage else None
        job = {"id": uuid.uuid4().hex[:8], "kind": kind, "stage": stage, "state": "starting",
               "phase": "calibrate" if kind == "calibrate" else "train", "phase_label": "Starting…", "progress": None,
               "step": project["stages"][stage]["step"] if stage else 0, "target": target, "device": device,
               "started": time.time()}
        STATE["job"] = job
        JOB_CONTROL["project"] = project["id"]
        JOB_CONTROL["stop"].clear()
        thread = threading.Thread(target=run_job, args=(project, job), daemon=True)
        JOB_CONTROL["thread"] = thread
    emit("job", job=deepcopy(job))
    emit("state")
    thread.start()
    return job


def stop_job() -> dict:
    with LOCK:
        job = STATE["job"]
        if not job:
            raise ApiError(409, "Nothing is running.")
        job["state"] = "stopping"
        JOB_CONTROL["stop"].set()
    emit("job", job=deepcopy(job))
    return job


def run_job(project: dict, job: dict) -> None:
    try:
        if job["kind"] == "calibrate":
            run_calibration(project, job)
        else:
            run_training(project, job)
    except Exception as exc:  # keep the mock alive and tell the UI
        finish(job, "error", f"{type(exc).__name__}: {exc}")


def tick_phase(job: dict, phase: str, label: str, seconds: float, stage: str | None) -> bool:
    set_phase(job, phase, label, 0.0)
    ticks = 12
    for i in range(ticks):
        if JOB_CONTROL["stop"].is_set():
            return False
        time.sleep(seconds / ticks / max(0.1, ARGS.speed))
        job["progress"] = (i + 1) / ticks
        emit("progress", stage=stage, step=job.get("step", 0), target=job.get("target"), loss=None, lr=None, tok_per_s=None,
             elapsed=0, tokens_seen=0, phase=phase, progress=job["progress"])
    return True


def run_calibration(project: dict, job: dict) -> None:
    results = {}
    scale_source = dict(project["calibration"])
    project["calibration"] = {}
    estimate = step_seconds(project)
    project["calibration"] = scale_source
    tokens = project["recipes"]["pretrain"]["batch_size"] * project["config"]["context"]
    for device in ("cpu", "vulkan"):
        name = next(d["label"] for d in DEVICES if d["id"] == device)
        if not tick_phase(job, "calibrate", f"Timing a few steps on the {name}…", 2.0, None):
            finish(job, "paused")
            return
        seconds = round(estimate[device] * random.uniform(0.9, 1.1), 4)
        results[device] = {"step_seconds": seconds, "tok_per_s": round(tokens / seconds)}
    project["calibration"] = {device: r["step_seconds"] for device, r in results.items()}
    project["calibrated_at"] = time.time()
    emit("calibration", results=results)
    finish(job, "complete")


def run_training(project: dict, job: dict) -> None:
    stage = job["stage"]
    raw = project["stages"][stage]
    recipe = project["recipes"][stage]
    rng = random.Random()
    if stage == "pretrain":
        info = dataset_info(project["data"]["pretrain"])
        if info and not dataset_status(info)["downloaded"]:
            set_phase(job, "prepare", f"Downloading {info['name']}…", 0.0)
            if not simulate_download(info, job):
                finish(job, "paused")
                return
        if not project["tokenizer"]["trained"] and project["tokenizer"]["kind"] != "gpt2":
            label = tokenizer_label("bpe", project["tokenizer"]["vocab_size"])
            if not tick_phase(job, "tokenizer", f"Training the {label} tokenizer…", 3.0, stage):
                finish(job, "paused")
                return
            project["tokenizer"]["trained"] = True
        if not project["data"]["corpus_tokens"]:
            if not tick_phase(job, "tokenize", "Tokenizing the corpus…", 3.0, stage):
                finish(job, "paused")
                return
            project["data"]["corpus_tokens"] = int((info["approx_bytes"] if info else 5_000_000) / 4.2)
    target = None if recipe.get("mode") == "until_stop" else recipe["steps"]
    step = raw["checkpoint_step"] or 0
    metrics = project["metrics"][stage]
    for key in ("train", "val", "lr"):
        metrics[key] = [point for point in metrics[key] if point[0] <= step]
    project["samples"][stage] = [s for s in project["samples"][stage] if s["step"] <= step]
    if stage in STAGE_PARENT:
        raw["base"] = {"stage": STAGE_PARENT[stage], "step": project["stages"][STAGE_PARENT[stage]]["checkpoint_step"]}
    device = job["device"]
    seconds = step_seconds(project)[device] * (0.5 if stage != "pretrain" else 1.0)
    tokens_per_step = recipe["batch_size"] * project["config"]["context"]
    elapsed, tokens_seen = raw["elapsed"] if step else 0.0, raw["tokens_seen"] if step else 0
    set_phase(job, "train", {"pretrain": "Pre-training", "sft": "Fine-tuning", "dpo": "Preference tuning"}[stage],
              step / target if target else None)
    per_tick = {"pretrain": 4, "sft": 2, "dpo": 1}[stage]
    evals, samples, checkpoints = (cadence(recipe, k, stage) for k in ("eval_every", "sample_every", "checkpoint_every"))
    loss = raw["loss"]
    while not JOB_CONTROL["stop"].is_set() and (target is None or step < target):
        time.sleep(0.2)
        for _ in range(max(1, round(per_tick * ARGS.speed))):
            if target is not None and step >= target:
                break
            step += 1
            elapsed += seconds * rng.uniform(0.95, 1.05)
            tokens_seen += tokens_per_step
            loss = noisy_loss(stage, step, rng)
            if step % evals == 0:
                value = round(val_loss(stage, step, rng), 4)
                metrics["val"].append([step, value])
                raw["val_loss"] = value
                emit("eval", stage=stage, step=step, val_loss=value)
            if step % samples == 0:
                text = sample_text(stage, step, rng)
                project["samples"][stage].append({"step": step, "prompt": SAMPLE_PROMPTS[stage], "text": text, "time": time.time()})
                emit("sample", stage=stage, step=step, prompt=SAMPLE_PROMPTS[stage], text=text)
            if step % checkpoints == 0:
                raw["checkpoint_step"] = step
                emit("checkpoint", stage=stage, step=step)
        lr = lr_at(recipe, step, target)
        metrics["train"].append([step, round(loss, 4)])
        metrics["lr"].append([step, lr])
        raw.update(step=step, loss=round(loss, 4), lr=lr, tok_per_s=round(tokens_per_step / seconds * rng.uniform(0.96, 1.04)),
                   elapsed=round(elapsed, 2), tokens_seen=tokens_seen, device=device)
        job.update(step=step, progress=(step / target) if target else None)
        emit("progress", stage=stage, step=step, target=target, loss=raw["loss"], lr=lr, tok_per_s=raw["tok_per_s"],
             elapsed=raw["elapsed"], tokens_seen=tokens_seen, phase="train", progress=job["progress"])
    if raw["checkpoint_step"] != step:
        raw["checkpoint_step"] = step
        emit("checkpoint", stage=stage, step=step)
    finish(job, "complete" if target is not None and step >= target else "paused")


def simulate_download(info: dict, job: dict | None = None) -> bool:
    total = info["approx_bytes"]
    cancel = DOWNLOAD_CONTROL["cancel"]
    cancel.clear()
    STATE["download"] = {"dataset": info["id"], "received": 0, "total": total, "state": "downloading"}
    emit("download", **STATE["download"])
    ticks = 24
    for i in range(ticks):
        time.sleep(4.0 / ticks / max(0.01, ARGS.speed))
        if cancel.is_set() or (job and JOB_CONTROL["stop"].is_set()):
            STATE["download"] = dict(STATE["download"], state="cancelled")
            emit("download", **STATE["download"])
            return False
        STATE["download"] = dict(STATE["download"], received=int(total * (i + 1) / ticks))
        emit("download", **STATE["download"])
        if job:
            job["progress"] = (i + 1) / ticks
            emit("progress", stage="pretrain", step=0, target=job.get("target"), loss=None, lr=None, tok_per_s=None,
                 elapsed=0, tokens_seen=0, phase="prepare", progress=job["progress"])
    STATE["downloaded"].add(info["id"])
    STATE["download"] = dict(STATE["download"], received=total, state="done")
    emit("download", **STATE["download"])
    emit("state")
    return True


def start_download(dataset_id: str) -> dict:
    info = dataset_info(dataset_id)
    if not info:
        raise ApiError(404, "Unknown dataset.")
    with LOCK:
        current = STATE["download"]
        if current and current["state"] == "downloading":
            raise ApiError(409, "Another download is already in progress.")
        if dataset_status(info)["downloaded"]:
            return {"dataset": dataset_id, "received": info["approx_bytes"], "total": info["approx_bytes"], "state": "done"}
        STATE["download"] = {"dataset": dataset_id, "received": 0, "total": info["approx_bytes"], "state": "downloading"}
    threading.Thread(target=simulate_download, args=(info,), daemon=True).start()
    return deepcopy(STATE["download"])

# ---------------------------------------------------------------- tokens and generation

SPECIALS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
TOKEN_RE = re.compile(r"<\|[a-z]+\|>|\n| ?[A-Za-z]+|\d| ?[^\sA-Za-z\d]| +(?=\S)| +")


class Vocab:
    """Grows as new text is seen, so ids stay stable for the lifetime of the mock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.texts = list(SPECIALS)
        self.ids = {text: i for i, text in enumerate(self.texts)}

    def id(self, text: str) -> int:
        with self.lock:
            if text not in self.ids:
                self.ids[text] = len(self.texts)
                self.texts.append(text)
            return self.ids[text]

    def text(self, token_id: int) -> str:
        return self.texts[token_id] if 0 <= token_id < len(self.texts) else "\ufffd"


VOCAB = Vocab()


def split_text(text: str) -> list[str]:
    pieces = []
    for match in TOKEN_RE.finditer(text):
        piece = match.group(0)
        word = piece.strip()
        if word.isalpha() and len(word) > 8:  # long words become two BPE-like pieces
            cut = len(piece) - len(word) + (len(word) + 1) // 2
            pieces.extend([piece[:cut], piece[cut:]])
        else:
            pieces.append(piece)
    return pieces


def tokenize(text: str) -> list[dict]:
    return [{"id": VOCAB.id(piece), "text": piece} for piece in split_text(text)]


def byte_tokens(text: str) -> list[dict]:
    out = []
    for value in text.encode("utf-8"):
        char = chr(value)
        out.append({"id": value, "text": char if 32 <= value < 127 or char == "\n" else f"<0x{value:02X}>"})
    return out


def gen_token(text: str, rng: random.Random, temperature: float, top_k: int, sure: bool = False) -> dict:
    chosen = VOCAB.id(text)
    pool = [w for w in COMMON_WORDS if w != text]
    others = rng.sample(pool, 4)
    conf = 0.92 if sure else rng.betavariate(5, 2) ** max(0.35, temperature)
    weights = sorted((rng.random() for _ in others), reverse=True)
    tail = rng.uniform(0.3, 1.2) * max(0.2, temperature)
    alt_probs = [(1 - conf) * w / (sum(weights) + tail) for w in weights]
    if not sure and rng.random() < 0.15 * temperature:
        conf, alt_probs[0] = alt_probs[0], conf
    alternatives = [{"id": chosen, "text": text, "prob": conf}]
    alternatives += [{"id": VOCAB.id(o), "text": o, "prob": p} for o, p in zip(others, alt_probs)]
    alternatives.sort(key=lambda a: -a["prob"])
    alternatives = alternatives[:max(1, min(5, top_k or 5))]
    if not any(a["id"] == chosen for a in alternatives):
        alternatives[-1] = {"id": chosen, "text": text, "prob": conf}
    return {"id": chosen, "text": text, "prob": round(conf, 4),
            "alternatives": [dict(a, prob=round(a["prob"], 4)) for a in alternatives]}


DEFAULT_SYSTEM = "You are a friendly assistant. Give clear, short answers."


def chat_template(system: str | None, messages: list[dict]) -> str:
    # Like the real engine: a system turn is always present (default text when `system` is None).
    parts = [f"<|system|>\n{(DEFAULT_SYSTEM if system is None else system).strip()}<|end|>\n"]
    for message in messages:
        role = message.get("role")
        if role in ("user", "assistant"):
            parts.append(f"<|{role}|>\n{str(message.get('content', '')).strip()}<|end|>\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def best_stage(project: dict) -> str | None:
    for stage in ("dpo", "sft", "pretrain"):
        if project["stages"][stage]["checkpoint_step"] is not None:
            return stage
    return None


def prepare_generation(project: dict, body: dict) -> dict:
    stage = body.get("stage") or "pretrain"
    if stage not in STAGES:
        raise ApiError(400, "Unknown stage.")
    checkpoint = project["stages"][stage]["checkpoint_step"]
    if checkpoint is None:
        names = {"pretrain": "pre-trained", "sft": "fine-tuned", "dpo": "preference-tuned"}
        raise ApiError(409, f"There is no {names[stage]} checkpoint yet. Train this stage first.")
    settings = body.get("settings") or {}
    seed = settings.get("seed")
    if seed in (None, ""):
        seed = random.randrange(1, 2 ** 31)
    rng = random.Random(int(seed))
    chat = body.get("messages") is not None
    if chat:
        prompt_tokens = tokenize(chat_template(body.get("system"), body.get("messages") or []))
    elif body.get("prompt_ids"):
        prompt_tokens = [{"id": int(i), "text": VOCAB.text(int(i))} for i in body["prompt_ids"]]
        chat = any(t["text"] == "<|assistant|>" for t in prompt_tokens[-80:])
    else:
        prompt_tokens = tokenize(body.get("prompt") or "") or tokenize("<|endoftext|>")
    limit = project["config"]["context"]
    if len(prompt_tokens) >= limit:
        prompt_tokens = prompt_tokens[-(limit - 1):]
    if chat:
        source = rng.choice(CHAT_REPLIES) + "<|end|>"
    elif stage == "pretrain" and checkpoint < 300:
        source = rng.choice(EARLY_GIBBERISH) + "<|endoftext|>"
    else:
        source = rng.choice(CONTINUATIONS) + "<|endoftext|>"
        prompt_text = "".join(t["text"] for t in prompt_tokens)
        if prompt_text.endswith((" ", "\n", "|>")) or not prompt_text:
            source = source.lstrip(" ")
    return {"stage": stage, "checkpoint": checkpoint, "seed": int(seed), "rng": rng, "chat": chat,
            "prompt_tokens": prompt_tokens, "limit": limit, "source": split_text(source),
            "temperature": float(settings.get("temperature", 0.8) or 0.8),
            "top_k": int(settings.get("top_k", 40) or 40), "max_tokens": int(settings.get("max_tokens", 120) or 120)}


def run_generation(plan: dict, write) -> None:
    prompt_tokens = plan["prompt_tokens"]
    write({"type": "start", "prompt_tokens": prompt_tokens, "prompt_text": "".join(t["text"] for t in prompt_tokens),
           "context_tokens": len(prompt_tokens), "context_limit": plan["limit"], "seed": plan["seed"],
           "step": plan["checkpoint"], "stage": plan["stage"]})
    started = time.time()
    first = None
    produced = 0
    stop_reason = "end"
    for piece in plan["source"]:
        if produced >= plan["max_tokens"]:
            stop_reason = "length"
            break
        if len(prompt_tokens) + produced >= plan["limit"]:
            stop_reason = "context"
            break
        time.sleep((0.03 + plan["rng"].random() * 0.03) / max(0.1, ARGS.speed))
        special = piece in ("<|end|>", "<|endoftext|>")
        write({"type": "token", "token": gen_token(piece, plan["rng"], plan["temperature"], plan["top_k"], sure=special)})
        produced += 1
        if first is None:
            first = time.time() - started
        if special:
            stop_reason = "end"
            break
    elapsed = max(1e-3, time.time() - started)
    write({"type": "end", "stop_reason": stop_reason, "tokens": produced, "tok_per_s": round(produced / elapsed, 1),
           "ttft": round(first or 0.0, 3)})


def attention(project: dict, body: dict) -> dict:
    stage = body.get("stage") or best_stage(project)
    if not stage or project["stages"][stage]["checkpoint_step"] is None:
        raise ApiError(409, "Train the model first; attention comes from a saved checkpoint.")
    tokens = tokenize(body.get("text") or "")[: project["config"]["context"]]
    n = len(tokens)
    rng = random.Random(sum(t["id"] for t in tokens) + n)
    layers = []
    for _ in range(project["config"]["blocks"]):
        heads = []
        for _ in range(project["config"]["heads"]):
            pattern = rng.choice(["previous", "first", "self", "spread"])
            rows = []
            for i in range(n):
                scores = []
                for j in range(i + 1):
                    score = rng.random() * 0.6
                    score += 3.0 if (pattern == "previous" and j == i - 1) else 0.0
                    score += 2.5 if (pattern == "first" and j == 0) or (pattern == "self" and j == i) else 0.0
                    scores.append(math.exp(score * 1.5))
                total = sum(scores)
                rows.append([round(s / total, 4) for s in scores] + [0.0] * (n - i - 1))
            heads.append(rows)
        layers.append(heads)
    return {"tokens": tokens, "layers": layers}


def ballot(project: dict, body: dict) -> dict:
    if project["stages"]["sft"]["checkpoint_step"] is None:
        raise ApiError(409, "Fine-tune the model first; the ballot samples answers from the SFT checkpoint.")
    prompt = (body.get("prompt") or "").strip() or random.choice(BALLOT_PROMPTS)
    first, second = random.sample(CHAT_REPLIES, 2)
    if random.random() < 0.5:
        words = second.split(" ")
        second = " ".join(w if i % 3 else "the" for i, w in enumerate(words))
    time.sleep(0.8 / max(0.1, ARGS.speed))
    return {"prompt": prompt, "a": {"text": first}, "b": {"text": second}}


def human_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size} B"


def import_dataset(kind: str, name: str, filename: str, data: bytes) -> dict:
    if kind not in ("pretrain", "finetune"):
        raise ApiError(400, "kind must be pretrain or finetune.")
    if not data.strip():
        raise ApiError(400, "The file is empty.")
    text = data.decode("utf-8", errors="replace")
    suffix = Path(filename).suffix.lower()
    if kind == "pretrain":
        documents = [d.strip() for d in re.split(r"\n\s*\n", text) if d.strip()]
        preview = {"kind": "text", "documents": documents[:4], "truncated": len(documents) > 4}
        extra = {"documents": len(documents)}
        size_label = f"{human_bytes(len(data))} · 1 {suffix or '.txt'}"
    else:
        examples = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                cells = [c.strip().strip('"') for c in line.split(",", 1)]
                row = {"prompt": cells[0], "response": cells[1]} if len(cells) == 2 else None
            if isinstance(row, dict) and isinstance(row.get("messages"), list):
                examples.append({"messages": row["messages"]})
            elif isinstance(row, dict) and (row.get("prompt") or row.get("instruction")):
                user = row.get("prompt") or row.get("instruction")
                examples.append({"messages": [{"role": "user", "content": user},
                                              {"role": "assistant", "content": row.get("response") or row.get("output") or ""}]})
        if not examples:
            raise ApiError(400, "No conversations found. Use JSONL rows with a messages list, or prompt/response columns.")
        preview = {"kind": "chat", "examples": examples[:5]}
        extra = {"rows": len(examples)}
        size_label = f"{len(examples):,} rows"
    info = _ds(f"custom-{slugify(name)}-{uuid.uuid4().hex[:4]}", kind, name, f"Imported from {filename}",
               f"Your own file, {filename}. It stays on this computer.", "Your own data", "", "Imported file", "",
               size_label, len(data), "file", "#8e8e93", **extra)
    info["custom"] = True
    STATE["custom"].append(info)
    STATE["custom_text"][info["id"]] = preview
    return info


def preview_dataset(dataset_id: str) -> dict:
    info = dataset_info(dataset_id)
    if not info:
        raise ApiError(404, "Unknown dataset.")
    if not dataset_status(info)["downloaded"]:
        raise ApiError(409, "Download this dataset to preview it.")
    if info["id"] in STATE["custom_text"]:
        return STATE["custom_text"][info["id"]]
    if info["kind"] == "finetune":
        return {"kind": "chat", "examples": [{"messages": m} for m in CHAT_EXAMPLES]}
    return {"kind": "text", "documents": PREVIEWS.get(info["id"], STORY_DOCS), "truncated": True}

# ---------------------------------------------------------------- project updates


def patch_project(project: dict, body: dict) -> None:
    if "name" in body:
        name = str(body["name"]).strip()
        if not name:
            raise ApiError(409, "The project needs a name.")
        project["name"] = name[:80]
    locked = project["stages"]["pretrain"]["checkpoint_step"] is not None
    if ("config" in body or "tokenizer" in body) and locked:
        raise ApiError(409, "This model already has pre-training checkpoints. Reset pre-training to change its shape or tokenizer.")
    if body.get("config"):
        changes = body["config"]
        config = dict(project["config"])
        preset = changes.get("preset")
        if preset in PRESETS:
            config.update(PRESETS[preset], preset=preset)
        for key, (low, high) in LIMITS.items():
            if key in changes:
                config[key] = int(min(high, max(low, int(changes[key]))))
                if preset not in PRESETS:
                    config["preset"] = "custom"
        if preset == "custom":
            config["preset"] = "custom"
        if config["width"] % config["heads"] or (config["width"] // config["heads"]) % 2:
            raise ApiError(409, "The width must split evenly into attention heads with an even size per head.")
        if config["preset"] in PRESETS and any(config[k] != v for k, v in PRESETS[config["preset"]].items()):
            config["preset"] = "custom"
        if config != project["config"]:
            project["config"] = config
            project["calibration"] = {}
    if body.get("tokenizer"):
        changes = body["tokenizer"]
        tok = dict(project["tokenizer"])
        kind = changes.get("kind", tok["kind"])
        if kind not in ("bpe", "gpt2"):
            raise ApiError(409, "Unknown tokenizer kind.")
        if "vocab_size" in changes and int(changes["vocab_size"]) not in BPE_SIZES:
            raise ApiError(409, "Vocabulary size must be one of 4000, 10000, 16000.")
        new = {"kind": kind, "vocab_size": int(changes.get("vocab_size", tok["vocab_size"])), "trained": False}
        if (new["kind"], new["vocab_size"]) != (tok["kind"], tok["vocab_size"]):
            project["tokenizer"] = new
            project["data"]["corpus_tokens"] = None
            project["calibration"] = {}
    for kind in ("pretrain", "finetune"):
        if kind not in (body.get("data") or {}):
            continue
        value = body["data"][kind] or None
        info = dataset_info(value) if value else None
        if value and not info:
            raise ApiError(404, "Unknown dataset.")
        if info and info["kind"] != kind:
            raise ApiError(409, f"{info['name']} is not a {'pre-training' if kind == 'pretrain' else 'fine-tuning'} dataset.")
        if value != project["data"][kind]:
            project["data"][kind] = value
            if kind == "pretrain":
                project["data"]["corpus_tokens"] = None
                if not locked and project["tokenizer"]["kind"] == "bpe":
                    project["tokenizer"]["trained"] = False
    for stage, values in (body.get("recipes") or {}).items():
        if stage not in project["recipes"] or not isinstance(values, dict):
            continue
        recipe = project["recipes"][stage]
        for key, value in values.items():
            if key == "mode":
                if value in ("steps", "until_stop"):
                    recipe["mode"] = value
            elif key in recipe:
                number = float(value)
                recipe[key] = number if key in ("learning_rate", "beta") else max(1 if key != "warmup" else 0, int(round(number)))


def reset_stage(project: dict, stage: str) -> None:
    if stage not in STAGES:
        raise ApiError(409, "Unknown stage.")
    job = STATE["job"]
    if job and JOB_CONTROL["project"] == project["id"]:
        raise ApiError(409, "Pause the running job before resetting.")
    for name in STAGES[STAGES.index(stage):]:
        project["stages"][name] = empty_stage()
        project["metrics"][name] = {"train": [], "val": [], "lr": []}
        project["samples"][name] = []


# ---------------------------------------------------------------- HTTP


class Handler(BaseHTTPRequestHandler):
    server_version = "LMBMock/0.1"

    def log_message(self, fmt, *args):  # quiet: only 4xx/5xx responses
        if len(args) > 1 and str(args[1])[:1] in ("4", "5"):
            super().log_message(fmt, *args)

    # ---- helpers
    def send_json(self, payload, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def json_body(self) -> dict:
        raw = self.read_body()
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except ValueError:
            raise ApiError(400, "Invalid JSON body.")
        return value if isinstance(value, dict) else {}

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def do_PATCH(self):
        self.dispatch("PATCH")

    def do_DELETE(self):
        self.dispatch("DELETE")

    def dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        path = url.path
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if not path.startswith("/api/"):
                if method != "GET":
                    raise ApiError(405, "Method not allowed.")
                return self.serve_static(path)
            if path == "/api/events" and method == "GET":
                return self.serve_events()
            if path == "/api/projects/current/generate" and method == "POST":
                return self.serve_generate()
            result = self.route(method, path, query)
            self.send_json(result if result is not None else {"ok": True})
        except ApiError as error:
            self.send_json({"error": str(error)}, error.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:  # pragma: no cover - surfaced to the UI as a 500
            self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)

    def route(self, method: str, path: str, query: dict):
        parts = [unquote(p) for p in path.split("/")[2:]]
        key = (method, "/".join(parts))
        if key == ("GET", "state"):
            return state_view()
        if key == ("POST", "_mock/speed"):  # mock-only: slow down or speed up simulated jobs and downloads
            ARGS.speed = max(0.01, float(self.json_body().get("speed", 1.0)))
            return {"speed": ARGS.speed}
        if key == ("POST", "settings"):
            body = self.json_body()
            with LOCK:
                STATE["settings"].update({k: v for k, v in body.items() if k in STATE["settings"]})
            return {"settings": deepcopy(STATE["settings"])}
        if key == ("POST", "reveal-data"):
            return {"ok": True}
        if key == ("POST", "devices/refresh"):
            emit("state")
            return {"devices": deepcopy(DEVICES), "probing": False}
        if parts[0] == "projects":
            return self.route_projects(method, parts[1:])
        if parts[0] == "jobs":
            if key == ("POST", "jobs"):
                kind = self.json_body().get("kind")
                if kind not in ("pretrain", "sft", "dpo", "calibrate"):
                    raise ApiError(400, "Unknown job kind.")
                return {"job": deepcopy(start_job(kind))}
            if key == ("POST", "jobs/stop"):
                return {"job": deepcopy(stop_job())}
        if parts[0] == "datasets":
            return self.route_datasets(method, parts[1:], query)
        if key == ("POST", "tokenize"):
            body = self.json_body()
            text = body.get("text") or ""
            kind = body.get("tokenizer") or "project"
            if kind == "bytes":
                return {"tokens": byte_tokens(text), "tokenizer": {"kind": "bytes", "vocab_size": 256, "label": "Bytes"}}
            if kind == "gpt2":
                return {"tokens": tokenize(text), "tokenizer": {"kind": "gpt2", "vocab_size": GPT2_VOCAB, "label": "GPT-2 BPE"}}
            project = current_project()
            view = project_view(project)["tokenizer"]
            if not view["trained"]:
                raise ApiError(409, "This project has no trained tokenizer yet. Start pre-training to train one.")
            return {"tokens": tokenize(text), "tokenizer": {k: view[k] for k in ("kind", "vocab_size", "label")}}
        raise ApiError(404, f"No such endpoint: {method} {path}")

    def route_projects(self, method: str, rest: list[str]):
        if not rest:
            if method == "POST":
                name = str(self.json_body().get("name") or "").strip() or "Untitled model"
                project = new_project(name[:80])
                with LOCK:
                    STATE["projects"][project["id"]] = project
                    STATE["current"] = project["id"]
                emit("state")
                return {"project": project_view(project)}
            raise ApiError(405, "Method not allowed.")
        if rest == ["select"] and method == "POST":
            pid = self.json_body().get("id")
            if pid not in STATE["projects"]:
                raise ApiError(404, "Unknown project.")
            STATE["current"] = pid
            emit("state")
            return {"project": project_view(STATE["projects"][pid])}
        if rest[0] != "current":
            if method == "DELETE" and len(rest) == 1:
                if rest[0] not in STATE["projects"]:
                    raise ApiError(404, "Unknown project.")
                if STATE["job"] and JOB_CONTROL["project"] == rest[0]:
                    raise ApiError(409, "Pause the running job before deleting this project.")
                del STATE["projects"][rest[0]]
                if STATE["current"] == rest[0]:
                    STATE["current"] = next(iter(STATE["projects"]), None)
                emit("state")
                return {"ok": True}
            raise ApiError(404, "No such endpoint.")
        project = current_project()
        sub = "/".join(rest[1:])
        if sub == "" and method == "GET":
            return project_view(project)
        if sub == "" and method == "PATCH":
            with LOCK:
                patch_project(project, self.json_body())
            emit("state")
            return project_view(project)
        if sub == "reset" and method == "POST":
            with LOCK:
                reset_stage(project, self.json_body().get("stage") or "pretrain")
            emit("state")
            return project_view(project)
        if sub == "reveal" and method == "POST":
            return {"ok": True}
        if len(rest) == 4 and rest[1] == "runs" and rest[2] in STAGES and method == "GET":
            if rest[3] == "metrics":
                return deepcopy(project["metrics"][rest[2]])
            if rest[3] == "samples":
                return {"samples": deepcopy(project["samples"][rest[2]])}
        if sub == "attention" and method == "POST":
            return attention(project, self.json_body())
        if sub == "ballot" and method == "POST":
            return ballot(project, self.json_body())
        if sub == "preferences":
            if method == "GET":
                return {"pairs": deepcopy(project["preferences"]), "count": len(project["preferences"]), "required": 4}
            if method == "POST":
                body = self.json_body()
                if not all(isinstance(body.get(k), str) and body.get(k) for k in ("prompt", "chosen", "rejected")):
                    raise ApiError(400, "prompt, chosen and rejected are required.")
                project["preferences"].append({k: body[k] for k in ("prompt", "chosen", "rejected")})
                emit("state")
                return {"count": len(project["preferences"])}
            if method == "DELETE":
                project["preferences"] = []
                emit("state")
                return {"count": 0}
        if sub == "export" and method == "POST":
            stage = self.json_body().get("stage") or best_stage(project)
            if not stage or stage not in STAGES or project["stages"][stage]["checkpoint_step"] is None:
                raise ApiError(409, "There is no checkpoint to export yet.")
            time.sleep(0.6)
            folder = Path.home() / "Downloads"
            return {"path": str(folder / f"{slugify(project['name'])}-{stage}.zip"), "folder": str(folder)}
        raise ApiError(404, "No such endpoint.")

    def route_datasets(self, method: str, rest: list[str], query: dict):
        if not rest and method == "GET":
            kind = query.get("kind")
            items = [dict(info, status=dataset_status(info)) for info in CATALOG + STATE["custom"]
                     if kind in (None, "", info["kind"])]
            return {"datasets": items}
        if rest == ["import"] and method == "POST":
            filename = unquote(self.headers.get("X-Filename") or "data.txt")
            name = (query.get("name") or Path(filename).stem).strip() or "My data"
            info = import_dataset(query.get("kind") or "pretrain", name, filename, self.read_body())
            emit("state")
            return {"dataset": dict(info, status=dataset_status(info))}
        if len(rest) == 1 and method == "DELETE":
            info = dataset_info(rest[0])
            if not info:
                raise ApiError(404, "Unknown dataset.")
            if info["custom"]:
                STATE["custom"] = [d for d in STATE["custom"] if d["id"] != info["id"]]
            else:
                STATE["downloaded"].discard(info["id"])
            return {"ok": True}
        if len(rest) == 2 and rest[1] == "preview" and method == "GET":
            return preview_dataset(rest[0])
        if len(rest) == 2 and rest[1] == "download" and method == "POST":
            return {"download": start_download(rest[0])}
        if len(rest) == 2 and rest[1] == "cancel" and method == "POST":
            DOWNLOAD_CONTROL["cancel"].set()
            return {"download": deepcopy(STATE["download"])}
        raise ApiError(404, "No such endpoint.")

    # ---- streams
    def serve_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = HUB.subscribe()
        started = time.time()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                wait = 15.0
                if ARGS.sse_lifetime:
                    left = ARGS.sse_lifetime - (time.time() - started)
                    if left <= 0:
                        break
                    wait = min(wait, left)
                try:
                    message = q.get(timeout=wait)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(f"data: {json.dumps(message)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HUB.unsubscribe(q)

    def serve_generate(self) -> None:
        plan = prepare_generation(current_project(), self.json_body())
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def write(event: dict) -> None:
            self.wfile.write((json.dumps(event) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            run_generation(plan, write)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def serve_static(self, path: str) -> None:
        relative = unquote(path).lstrip("/") or "index.html"
        target = (UI_DIR / relative).resolve()
        if UI_DIR.resolve() not in target.parents and target != UI_DIR.resolve():
            raise ApiError(404, "Not found.")
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            raise ApiError(404, "Not found.")
        data = target.read_bytes()
        ctype = {".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml",
                 ".woff2": "font/woff2", ".json": "application/json", ".html": "text/html"}.get(target.suffix)
        ctype = ctype or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--speed", type=float, default=1.0, help="speed factor for simulated jobs and downloads")
    parser.add_argument("--empty", action="store_true", help="start without any projects")
    parser.add_argument("--sse-lifetime", type=float, default=0.0, help="close SSE streams after N seconds (0 = never)")
    ARGS = parser.parse_args()
    seed_projects()
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    server.daemon_threads = True
    print(f"Mock engine on http://{ARGS.host}:{ARGS.port}/ (ui: {UI_DIR})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
