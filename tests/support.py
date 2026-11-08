"""Shared helpers for the engine tests: a throwaway data folder and a tiny, quickly trained project."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from unittest import mock


def _documents() -> list[str]:
    rng = random.Random(7)
    words = ("the a cat dog bird sat ran slept on under near mat rug tree big small red blue happy quiet "
             "garden river stone and with friendly assistant answers clear short").split()
    documents = []
    for _ in range(30):
        sentences = [" ".join(rng.choice(words) for _ in range(rng.randint(4, 9))).capitalize() + "."
                     for _ in range(8)]
        sentences.append("You are a friendly assistant. Give clear, short answers.")
        sentences.append(f"What is {rng.randint(1, 9)} plus {rng.randint(1, 9)}?")
        documents.append(" ".join(sentences))
    return documents


DOCUMENTS = _documents()
TINY_TEXT = "\n\n".join(DOCUMENTS)
PRETRAIN_FILE = "\n<|endoftext|>\n".join(DOCUMENTS)
CHAT_JSONL = "\n".join(
    json.dumps({"messages": [{"role": "user", "content": f"What is {a} plus {b}?"},
                             {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}]})
    for a in range(1, 7) for b in range(1, 7)
)


class isolated_environment:
    """Point every data, cache, runtime and download folder at a fresh temporary directory."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.root = None
        self.patcher = None

    def start(self) -> "isolated_environment":
        self.root = tempfile.mkdtemp(prefix=self.prefix)
        folders = {"LMB_DATA_DIR": "data", "LMB_CACHE_DIR": "cache", "XDG_RUNTIME_DIR": "run", "XDG_DOWNLOAD_DIR": "downloads"}
        values = {}
        for variable, name in folders.items():
            values[variable] = os.path.join(self.root, name)
            os.makedirs(values[variable])
        values.update(LMB_DISABLE_VULKAN="1", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
        self.patcher = mock.patch.dict(os.environ, values)
        self.patcher.start()
        return self

    def stop(self) -> None:
        self.patcher.stop()
        shutil.rmtree(self.root, ignore_errors=True)


def tiny_project(name: str, steps: int = 6):
    """A project with its own small datasets, a 1-block model and recipes that train in seconds."""
    from model_builder import datasets
    from model_builder import project as projects

    text = datasets.import_custom(f"{name} text", "pretrain", "tiny.txt", PRETRAIN_FILE.encode())
    chat = datasets.import_custom(f"{name} chat", "finetune", "chat.jsonl", CHAT_JSONL.encode())
    project = projects.create(name)
    project.update({
        "config": {"context": 64, "width": 32, "blocks": 1, "heads": 2},
        "tokenizer": {"kind": "bpe", "vocab_size": 4096},
        "data": {"pretrain": text.id, "finetune": chat.id},
        "recipes": {
            "pretrain": {"steps": steps, "batch_size": 4, "warmup": 2, "eval_every": 3, "sample_every": 3, "checkpoint_every": 3},
            "sft": {"steps": 4, "batch_size": 4, "warmup": 1, "eval_every": 2, "sample_every": 4, "checkpoint_every": 4},
            "dpo": {"steps": 3, "batch_size": 2},
        },
    })
    return project
