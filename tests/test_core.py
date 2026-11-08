"""Fast checks of the engine core: tokenizer, chat formatting, projects, events and a tiny training run.

Run from the repository root:  .venv/bin/python -m unittest tests.test_core -v
Everything runs on the CPU in a throwaway data folder and finishes in well under a minute.
"""

from __future__ import annotations

import json
import re
import shutil
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.support import TINY_TEXT, isolated_environment, tiny_project

from model_builder import app, data, project as projects, server, tokenizer, training, worker
from model_builder.training import read_jsonl

_env = None


def setUpModule() -> None:
    global _env
    _env = isolated_environment("lmb-test-core-")
    _env.start()


def tearDownModule() -> None:
    _env.stop()


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = tokenizer.train_bpe([TINY_TEXT] * 4, 320)

    def test_any_text_round_trips(self) -> None:
        sample = "Grüße, 世界! 🙂\ttabs and\nnew lines   and spaces"
        self.assertEqual(self.tok.decode(self.tok.encode(sample)), sample)

    def test_special_token_text_in_user_input_stays_plain_text(self) -> None:
        self.assertNotIn(self.tok.end_id, self.tok.encode("say <|end|> now"))
        self.assertIn(self.tok.end_id, self.tok.encode("<|end|>", special=True))

    def test_save_and_load_keep_ids(self) -> None:
        path = Path(tempfile.mkdtemp()) / "tokenizer.json"
        self.addCleanup(shutil.rmtree, path.parent)
        self.tok.save(path)
        loaded = tokenizer.TextTokenizer.load(path)
        self.assertEqual(loaded.encode(TINY_TEXT[:200]), self.tok.encode(TINY_TEXT[:200]))
        self.assertEqual((loaded.kind, loaded.end_id, loaded.vocab_size), (self.tok.kind, self.tok.end_id, self.tok.vocab_size))

    def test_chat_targets_cover_only_assistant_replies(self) -> None:
        messages = [{"role": "user", "content": "Hi there"}, {"role": "assistant", "content": "Hello friend"}]
        ids, targets = data.encode_chat(self.tok, messages)
        learned = [(position, target) for position, target in enumerate(targets) if target != data.IGNORE]
        self.assertEqual([target for _, target in learned], self.tok.encode("Hello friend") + [self.tok.end_id])
        for position, target in learned:
            self.assertEqual(target, ids[position + 1])
        self.assertEqual(ids[0], self.tok.system_id)
        prompt, _ = data.encode_chat(self.tok, messages[:1], add_generation_prompt=True)
        self.assertEqual(prompt[-2:], [self.tok.assistant_id, *self.tok.encode("\n")])

    def test_ui_previews_use_the_engine_default_system_prompt(self) -> None:
        # The chat page previews the transcript before the engine sees it, so its copies must not drift.
        ui = Path(data.__file__).parent / "ui" / "js"
        copies = {path.relative_to(ui).as_posix(): match.group(1)
                  for path in ui.rglob("*.js")
                  for match in re.finditer(r'DEFAULT_SYSTEM = "([^"]*)"', path.read_text(encoding="utf-8"))}
        self.assertIn("workbench/chat-view.js", copies)
        for name, text in copies.items():
            self.assertEqual(text, data.DEFAULT_SYSTEM, name)


class ProjectTests(unittest.TestCase):
    def test_shape_changes_are_clamped_and_validated(self) -> None:
        project = projects.create("Shape test")
        project.update({"config": {"context": 99_999}})
        self.assertEqual(project.data["config"]["context"], projects.SHAPE_LIMITS["context"][1])
        self.assertEqual(project.data["config"]["preset"], "custom")
        project.update({"config": {"preset": "pocket"}})
        self.assertEqual(project.data["config"], {**projects.PRESETS["pocket"], "preset": "pocket"})
        with self.assertRaises(projects.ProjectError):
            project.update({"config": {"heads": 3}})  # 128 doesn't split into 3 heads
        with self.assertRaises(projects.ProjectError):
            project.update({"tokenizer": {"vocab_size": 1234}})
        with self.assertRaises(projects.ProjectError):
            project.update({"name": "   "})
        reloaded = projects.load(project.id)
        self.assertEqual(reloaded.data["config"]["preset"], "pocket")

    def test_recipe_values_are_clamped(self) -> None:
        project = projects.create("Recipe test")
        project.update({"recipes": {"pretrain": {"batch_size": 10_000, "learning_rate": -1, "mode": "bogus"},
                                    "nonsense": {"steps": 3}}})
        recipe = project.data["recipes"]["pretrain"]
        self.assertEqual(recipe["batch_size"], projects.RECIPE_LIMITS["batch_size"][1])
        self.assertEqual(recipe["learning_rate"], projects.RECIPE_LIMITS["learning_rate"][0])
        self.assertEqual(recipe["mode"], "steps")
        self.assertNotIn("nonsense", project.data["recipes"])

    def test_project_ids_cannot_escape_the_projects_folder(self) -> None:
        for bad in ("../x", "a/b", "", ".hidden", "UPPER"):
            with self.assertRaises(KeyError):
                projects.load(bad)

    def test_memory_warning_uses_free_memory_on_phones(self) -> None:
        phone = {"MemTotal": 15 << 30, "MemAvailable": 3 << 30}
        with mock.patch.object(projects, "_meminfo", side_effect=phone.get):
            self.assertIn("free right now", projects.memory_warning(int(3.9e9), 8))
            self.assertIsNone(projects.memory_warning(int(1.7e9), 16))
        desktop = {"MemTotal": 16 << 30, "MemAvailable": 15 << 30}
        with mock.patch.object(projects, "_meminfo", side_effect=desktop.get):
            self.assertIn("in total", projects.memory_warning(int(7.9e9), 16))
            self.assertIsNone(projects.memory_warning(int(3.9e9), 8))


class EventHubTests(unittest.TestCase):
    def test_payload_may_contain_a_kind_field(self) -> None:
        hub = server.EventHub()
        client = hub.subscribe()
        hub.emit("job", kind="pretrain", state="running")
        self.assertEqual(json.loads(client.get_nowait()), {"type": "job", "kind": "pretrain", "state": "running"})


class WindowChoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        info = {"port": 1, "token": "t"}
        patches = [mock.patch.object(app, "ensure_server", return_value=info), mock.patch.object(app, "notify_closed"),
                   mock.patch.object(app, "run_webkit", return_value=None), mock.patch.object(app, "run_chromium", return_value=None),
                   mock.patch.object(app, "run_browser", return_value=0)]
        self.mocks = [patch.start() for patch in patches]
        for patch in patches:
            self.addCleanup(patch.stop)

    def test_auto_falls_back_to_the_browser(self) -> None:
        self.assertEqual(app.main("auto"), 0)
        self.mocks[4].assert_called_once()

    def test_an_unavailable_requested_window_does_not_open_the_browser(self) -> None:
        with mock.patch("sys.stderr"):
            self.assertEqual(app.main("webkit"), 1)
        self.mocks[3].assert_not_called()
        self.mocks[4].assert_not_called()


class TrainLoopTests(unittest.TestCase):
    def test_speed_ignores_the_warmup_step(self) -> None:
        class SlowFirstStep:
            calls = 0

            def train_step(self, tokens, targets, lr):
                self.calls += 1
                time.sleep(0.5 if self.calls == 1 else 0.01)  # the first step compiles kernels
                return 1.0

            def eval_loss(self, tokens, targets):
                return 1.0

            def sync(self):
                pass

            def close(self):
                pass

        folder = Path(tempfile.mkdtemp(prefix="lmb-loop-"))
        self.addCleanup(shutil.rmtree, folder, True)
        files = training.RunFiles(folder, job_id="loop", kind="pretrain", stage="pretrain", device="cpu")
        plan = training.LoopPlan("pretrain", steps=6, batch_size=2, learning_rate=1e-3, warmup=0,
                                 eval_every=100, sample_every=100, checkpoint_every=100)
        batch = np.zeros((2, 8), dtype=np.int64)
        model = training.GPT(training.ModelConfig(vocab_size=64, context=16, width=32, blocks=1, heads=2))
        with mock.patch.object(training, "make_stepper", return_value=SlowFirstStep()):
            result = training.train_loop(plan=plan, files=files, model=model, device="cpu",
                                         next_batch=lambda: (batch, batch), eval_batches=[],
                                         make_sample=lambda model, index: {"prompt": "", "text": ""},
                                         stop=threading.Event())
        self.assertEqual(result, "complete")
        speeds = [record["tok_per_s"] for record in read_jsonl(folder / "metrics.jsonl") if "loss" in record]
        self.assertIsNone(speeds[0])
        self.assertGreater(speeds[-1], 16 / 0.1)  # 16 tokens per ~0.01 s step, not dragged down by the 0.5 s one


class PipelineTests(unittest.TestCase):
    """Import data, pre-train, resume, fine-tune and preference-tune a tiny model through the worker."""

    def run_job(self, project, kind: str) -> dict:
        saved = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            with mock.patch("os.nice"):
                code = worker.run(project.id, kind, "cpu", f"test-{kind}")
        finally:
            for signum, handler in saved.items():
                signal.signal(signum, handler)
        status = json.loads((project.run_dir(kind) / "status.json").read_text())
        self.assertEqual(code, 0, status.get("error"))
        return status

    def test_full_pipeline(self) -> None:
        project = tiny_project("Pipeline", steps=6)
        status = self.run_job(project, "pretrain")
        self.assertEqual(status["state"], "complete")
        self.assertEqual(project.stage_info("pretrain")["state"], "complete")
        meta = json.loads((project.folder / "tokenizer.meta.json").read_text())
        self.assertLess(meta["vocab_size"], 4096)  # the tiny corpus can't fill the requested vocabulary
        self.assertTrue(project.locked())
        with self.assertRaises(projects.ProjectError):
            project.update({"config": {"blocks": 2}})

        # Raising the target turns the finished stage into a resumable one; resuming continues the curve.
        project.update({"recipes": {"pretrain": {"steps": 9}}})
        info = project.stage_info("pretrain")
        self.assertEqual((info["state"], info["step"], info["target"]), ("paused", 6, 9))
        self.run_job(project, "pretrain")
        steps = [record["step"] for record in read_jsonl(project.run_dir("pretrain") / "metrics.jsonl") if "loss" in record]
        self.assertEqual(steps, list(range(1, 10)))
        self.assertTrue(read_jsonl(project.run_dir("pretrain") / "samples.jsonl"))

        self.assertEqual(project.missing_for("sft"), None)
        self.run_job(project, "sft")
        sft = project.stage_info("sft")
        self.assertEqual((sft["state"], sft["base"]), ("complete", {"stage": "pretrain", "step": 9}))

        self.assertIsNotNone(project.missing_for("dpo"))
        for index in range(4):
            projects.add_preference(project, f"Question {index}?", f"A kind answer {index}.", f"no {index}")
        self.assertEqual(project.missing_for("dpo"), None)
        self.run_job(project, "dpo")
        records = [record for record in read_jsonl(project.run_dir("dpo") / "metrics.jsonl") if "loss" in record]
        self.assertEqual(len(records), 3)
        self.assertAlmostEqual(records[0]["loss"], 0.6931, places=2)  # policy == reference at the start
        self.assertIn("accuracy", records[-1])
        self.assertEqual(project.status()["label"], "Preference-tuned")


if __name__ == "__main__":
    unittest.main()
