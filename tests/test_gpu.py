"""Vulkan GPU trainer (tinygrad + wgpu) parity with the torch reference.

Needs a physical GPU adapter; run with ``LMB_TEST_GPU=1``.  The vocabulary (203) is deliberately
not a multiple of the GPU's padding granularity, and the batch repeats tokens, so the padded
output projection and the distinct-token embedding backward are both exercised.
"""

from __future__ import annotations

import copy
import os
import unittest
from unittest import mock

import numpy as np
import torch

from model_builder.model import GPT, ModelConfig, make_optimizer

GPU_TESTS = os.environ.get("LMB_TEST_GPU") == "1"
CONFIG = dict(vocab_size=203, context=16, width=32, blocks=2, heads=2)
BATCH, LENGTH = 3, 16
LEARNING_RATES = (3e-3, 1e-2, 5e-3, 2e-3)


def _model(seed: int = 0, **overrides) -> GPT:
    torch.manual_seed(seed)
    return GPT(ModelConfig(**{**CONFIG, **overrides}))


def _batch(seed: int, masked: bool = True, vocab: int = CONFIG["vocab_size"]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=(BATCH, LENGTH), dtype=np.int32)
    targets = rng.integers(0, vocab, size=(BATCH, LENGTH), dtype=np.int32)
    tokens[0, :5] = 7  # repeated tokens: their embedding gradients must add up
    tokens[1, -3:] = vocab - 1
    if masked:
        targets[0, :6] = -100
        targets[2, 9:] = -100
    return tokens, targets


def _torch_step(model: GPT, optimizer: torch.optim.AdamW, tokens, targets, lr: float) -> tuple[float, float]:
    """One reference step; returns (loss, gradient norm before clipping)."""
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss = model.loss(torch.from_numpy(tokens).long(), torch.from_numpy(targets).long())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), float(norm)


def _assert_models_close(test: unittest.TestCase, actual: GPT, reference: GPT, reference_optimizer, total_lr: float) -> None:
    """Weights agree, except where Adam normalized a gradient at rounding-noise level.

    Adam steps by about lr * g / |g|, so where a gradient nearly cancels out (the attention key
    bias cannot change the softmax at all; big vocabularies have many such output-layer entries)
    float32 rounding on either device picks the direction; there only the step bound can hold.
    """
    for (name, got), (_, want) in zip(actual.named_parameters(), reference.named_parameters()):
        rms = reference_optimizer.state[want]["exp_avg_sq"].sqrt()
        noisy = rms < 1e-3 * rms.max()
        got, want = got.detach(), want.detach()
        error = (got - want).abs()
        test.assertLessEqual(float(error.max()), 2 * total_lr, name)
        wrong = (error > 2e-5 + 1e-3 * want.abs()) & ~noisy
        test.assertFalse(bool(wrong.any()), f"{name}: {int(wrong.sum())} elements, max error {float(error[wrong].max()) if wrong.any() else 0:.3g}")


@unittest.skipUnless(GPU_TESTS, "set LMB_TEST_GPU=1 to run the Vulkan GPU tests")
class GpuTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from model_builder.gpu_model import gpu_available

        cls.adapter = gpu_available()
        if cls.adapter is None:
            raise unittest.SkipTest("no physical GPU adapter")
        torch.set_num_threads(2)

    def _trainer(self, model: GPT, optimizer: torch.optim.AdamW, step: int = 0):
        from model_builder.gpu_model import GpuTrainer

        trainer = GpuTrainer(model, optimizer, step)
        self.addCleanup(trainer.close)
        return trainer

    def test_adapter_is_physical(self) -> None:
        self.assertEqual(set(self.adapter), {"name", "backend", "adapter_type"})
        self.assertIn(self.adapter["adapter_type"], ("IntegratedGPU", "DiscreteGPU"))

    def test_software_adapter_is_rejected(self) -> None:
        import wgpu

        from model_builder import gpu_model, gpu_runtime

        software = mock.Mock(info={"adapter_type": "CPU", "device": "llvmpipe", "backend_type": "Vulkan"})
        gpu_model.gpu_available.cache_clear()
        self.addCleanup(gpu_model.gpu_available.cache_clear)
        with mock.patch.object(wgpu.gpu, "request_adapter_sync", return_value=software):
            with self.assertRaisesRegex(RuntimeError, "software"):
                gpu_runtime.request_adapter()
            self.assertIsNone(gpu_model.gpu_available())

    def test_eval_loss_matches_torch_with_masked_targets(self) -> None:
        model = _model()
        trainer = self._trainer(model, make_optimizer(model, 1e-3))
        for seed, masked in ((1, False), (2, True)):
            tokens, targets = _batch(seed, masked)
            with torch.no_grad():
                expected = model.loss(torch.from_numpy(tokens).long(), torch.from_numpy(targets).long()).item()
            self.assertAlmostEqual(trainer.eval_loss(tokens, targets) / expected, 1.0, delta=1e-4)

    def test_training_steps_match_torch_adamw_with_clipping(self) -> None:
        norms = []
        for gain in (1.0, 0.1):  # a small final LayerNorm gain shrinks all gradients below the clip norm
            reference = _model()
            with torch.no_grad():
                reference.ln_f.weight.mul_(gain)
            model = copy.deepcopy(reference)
            reference_optimizer = make_optimizer(reference, LEARNING_RATES[0])
            optimizer = make_optimizer(model, LEARNING_RATES[0])
            trainer = self._trainer(model, optimizer)
            for index, lr in enumerate(LEARNING_RATES):
                tokens, targets = _batch(10 + index)
                if index == 2:
                    tokens = tokens * 0 + 3  # one token everywhere: all positions share an embedding row
                expected, norm = _torch_step(reference, reference_optimizer, tokens, targets, lr)
                norms.append(norm)
                self.assertAlmostEqual(trainer.train_step(tokens, targets, lr) / expected, 1.0, delta=2e-4)
            self.assertEqual(trainer.step, len(LEARNING_RATES))
            self.assertEqual(len(trainer.programs), 1)  # learning-rate changes reuse one compiled step
            trainer.sync_to_torch(model, optimizer)
            _assert_models_close(self, model, reference, reference_optimizer, sum(LEARNING_RATES))
            for group in optimizer.param_groups:
                self.assertEqual(group["lr"], LEARNING_RATES[-1])
        self.assertTrue(any(norm > 1.0 for norm in norms) and any(norm < 1.0 for norm in norms), norms)

    def test_tied_embedding_gradient(self) -> None:
        reference = _model(seed=3)
        model = copy.deepcopy(reference)
        reference_optimizer = make_optimizer(reference, 1e-3)
        optimizer = make_optimizer(model, 1e-3)
        trainer = self._trainer(model, optimizer)
        tokens, targets = _batch(20)
        _torch_step(reference, reference_optimizer, tokens, targets, 1e-3)
        trainer.train_step(tokens, targets, 1e-3)
        trainer.sync_to_torch(model, optimizer)
        # After one AdamW step exp_avg is 0.1 x the clipped gradient: output-layer and lookup parts.
        got = optimizer.state[model.tok_emb.weight]["exp_avg"]
        want = reference_optimizer.state[reference.tok_emb.weight]["exp_avg"]
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-6 * float(want.abs().max()))
        unused = np.setdiff1d(np.arange(CONFIG["vocab_size"]), tokens)
        self.assertTrue(len(unused) and bool(want[unused].abs().sum() > 0))  # the output layer reaches them too
        padding = trainer.values["tok_emb.weight"].numpy()[CONFIG["vocab_size"] :]
        self.assertGreater(len(padding), 0)
        self.assertFalse(padding.any(), "padded vocabulary rows must stay zero")

    def test_resume_from_cpu_then_continue_on_cpu(self) -> None:
        reference = _model(seed=5)
        model = copy.deepcopy(reference)
        reference_optimizer = make_optimizer(reference, 1e-3)
        optimizer = make_optimizer(model, 1e-3)
        batches = [_batch(30 + index) for index in range(4)]
        for index, lr in enumerate(LEARNING_RATES):
            _torch_step(reference, reference_optimizer, *batches[index], lr)
        _torch_step(model, optimizer, *batches[0], LEARNING_RATES[0])  # CPU -> GPU with AdamW state
        trainer = self._trainer(model, optimizer, step=1)
        for index in (1, 2):
            trainer.train_step(*batches[index], LEARNING_RATES[index])
        trainer.sync_to_torch(model, optimizer)  # GPU -> CPU
        self.assertEqual(float(optimizer.state[model.ln_f.weight]["step"]), 3.0)
        _torch_step(model, optimizer, *batches[3], LEARNING_RATES[3])
        _assert_models_close(self, model, reference, reference_optimizer, sum(LEARNING_RATES))
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            got, want = optimizer.state[parameter], reference_optimizer.state[expected]
            torch.testing.assert_close(got["exp_avg_sq"], want["exp_avg_sq"], rtol=2e-3, atol=1e-12)

    def test_large_vocabulary_split_output_layer(self) -> None:
        from model_builder.gpu_model import HEAD_SPLIT_VOCAB

        vocab = HEAD_SPLIT_VOCAB + 7232  # pads to 40064 and runs the output layer in chunks
        reference = _model(seed=6, vocab_size=vocab)
        model = copy.deepcopy(reference)
        reference_optimizer = make_optimizer(reference, 1e-3)
        optimizer = make_optimizer(model, 1e-3)
        trainer = self._trainer(model, optimizer)
        tokens, targets = _batch(50, vocab=vocab)
        with torch.no_grad():
            expected = reference.loss(torch.from_numpy(tokens).long(), torch.from_numpy(targets).long()).item()
        self.assertAlmostEqual(trainer.eval_loss(tokens, targets) / expected, 1.0, delta=1e-4)
        for index, lr in enumerate(LEARNING_RATES[:2]):
            batch = _batch(51 + index, vocab=vocab)
            expected, _ = _torch_step(reference, reference_optimizer, *batch, lr)
            self.assertAlmostEqual(trainer.train_step(*batch, lr) / expected, 1.0, delta=2e-4)
            if index == 0:
                # Compare weights after the first update only: Adam's first step is about lr * sign(g),
                # so the many nearly-cancelling output-layer gradients of a big vocabulary flip a few
                # elements on either device, and later steps compound that, not any GPU error.
                trainer.sync_to_torch(model, optimizer)
                _assert_models_close(self, model, reference, reference_optimizer, lr)

    def test_fully_masked_batch_is_finite(self) -> None:
        model = _model()
        trainer = self._trainer(model, make_optimizer(model, 1e-3))
        tokens, targets = _batch(40)
        self.assertEqual(trainer.train_step(tokens, np.full_like(targets, -100), 1e-3), 0.0)
        self.assertTrue(np.isfinite(trainer.eval_loss(tokens, targets)))


if __name__ == "__main__":
    unittest.main()
