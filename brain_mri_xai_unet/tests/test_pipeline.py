"""Checks for missing inputs, checkpoint loading, and sample inference."""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CHECKPOINT_PATH, DEFAULT_IMAGE, INFERENCE_SIZE, MASK_THRESHOLD
from pipeline import PipelineError, load_unet, preprocess_mri, run_pipeline, select_device


class PipelineTests(unittest.TestCase):
    def test_missing_checkpoint(self):
        with self.assertRaises(PipelineError) as ctx:
            load_unet(checkpoint_path=os.path.join(ROOT, "models", "does_not_exist.pth"))
        self.assertIn("not found", str(ctx.exception).lower())

    def test_invalid_checkpoint_format(self):
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
            handle.write(b"not-a-pytorch-checkpoint")
            path = handle.name
        try:
            with self.assertRaises(PipelineError) as ctx:
                load_unet(checkpoint_path=path, device=select_device(force_cpu=True))
            message = str(ctx.exception).lower()
            self.assertTrue(
                "invalid" in message or "could not be loaded" in message,
                msg=message,
            )
        finally:
            os.remove(path)

    def test_missing_image(self):
        device = select_device(force_cpu=True)
        with self.assertRaises(PipelineError) as ctx:
            preprocess_mri(os.path.join(ROOT, "sample_data", "missing.jpg"), device)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_unreadable_image(self):
        device = select_device(force_cpu=True)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(b"this is not an image")
            path = handle.name
        try:
            with self.assertRaises(PipelineError) as ctx:
                preprocess_mri(path, device)
            self.assertIn("could not be opened", str(ctx.exception).lower())
        finally:
            os.remove(path)

    def test_checkpoint_loads(self):
        self.assertTrue(os.path.isfile(CHECKPOINT_PATH), "checkpoint missing from repo")
        model = load_unet(CHECKPOINT_PATH, select_device(force_cpu=True))
        n_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(n_params, 92737)

    def test_sample_inference(self):
        self.assertTrue(os.path.isfile(DEFAULT_IMAGE), "sample MRI missing from repo")
        with tempfile.TemporaryDirectory() as output_dir:
            result = run_pipeline(
                image_path=DEFAULT_IMAGE,
                output_dir=output_dir,
                checkpoint_path=CHECKPOINT_PATH,
                force_cpu=True,
                with_gradcam=False,
                warmup=False,
            )
            self.assertEqual(result["input_shape"], (1, 3, INFERENCE_SIZE, INFERENCE_SIZE))
            self.assertEqual(result["output_shape"], (1, 1, INFERENCE_SIZE, INFERENCE_SIZE))
            self.assertEqual(result["mask_shape"], (INFERENCE_SIZE, INFERENCE_SIZE))
            self.assertEqual(result["threshold"], MASK_THRESHOLD)
            self.assertGreater(result["inference_ms"], 0.0)
            for key in ("original", "mask", "overlay", "probability", "panel"):
                self.assertTrue(os.path.isfile(result["paths"][key]), msg=key)


if __name__ == "__main__":
    unittest.main()
