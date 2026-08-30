"""Lightweight checks for the real-time assist pipeline (stdlib unittest)."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.assist import ASSIST_CONTINUE, ASSIST_HOLD_FOR_REVIEW, ASSIST_SLOW_AND_FLAG, decide_assist
from src.preprocess import load_image_tensor
from src.robot_sim import SimulatedRobotAssist


class AssistLogicTests(unittest.TestCase):
    def test_low_confidence_holds(self) -> None:
        scores = {name: 0.25 for name in config.CLASS_NAMES}
        scores["glioma"] = 0.40
        decision = decide_assist(0, 0.40, scores, inference_ms=5.0)
        self.assertEqual(decision.assist_state, ASSIST_HOLD_FOR_REVIEW)

    def test_no_tumor_continues(self) -> None:
        scores = {name: 0.01 for name in config.CLASS_NAMES}
        scores["no_tumor"] = 0.95
        idx = config.CLASS_NAMES.index("no_tumor")
        decision = decide_assist(idx, 0.95, scores, inference_ms=5.0)
        self.assertEqual(decision.assist_state, ASSIST_CONTINUE)

    def test_tumor_slows(self) -> None:
        scores = {name: 0.01 for name in config.CLASS_NAMES}
        scores["glioma"] = 0.90
        decision = decide_assist(0, 0.90, scores, inference_ms=5.0)
        self.assertEqual(decision.assist_state, ASSIST_SLOW_AND_FLAG)

    def test_invalid_class_index(self) -> None:
        with self.assertRaises(ValueError):
            decide_assist(99, 0.9, {}, inference_ms=1.0)


class PreprocessTests(unittest.TestCase):
    def test_missing_image_errors(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_image_tensor(ROOT / "sample_data" / "does_not_exist.jpg")

    def test_sample_image_shape(self) -> None:
        if not config.SAMPLE_IMAGE.is_file():
            self.skipTest("sample image not present")
        tensor = load_image_tensor(config.SAMPLE_IMAGE)
        self.assertEqual(tuple(tensor.shape), (1, 3, *config.IMAGE_SIZE))


class ClassifierLoadTests(unittest.TestCase):
    def test_missing_weights_error(self) -> None:
        from models.classifier import load_classifier

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.pth"
            with self.assertRaises(FileNotFoundError):
                load_classifier(weights_path=missing, device="cpu")


class LiveInferenceTests(unittest.TestCase):
    def test_sample_inference(self) -> None:
        if not config.CLASSIFIER_WEIGHTS.is_file() or not config.SAMPLE_IMAGE.is_file():
            self.skipTest("model or sample image not present")

        from src.inference import classify_image

        decision = classify_image(
            config.SAMPLE_IMAGE,
            weights_path=config.CLASSIFIER_WEIGHTS,
            device="cpu",
            warmup=1,
            runs=2,
        )
        self.assertIn(decision.condition, config.CLASS_NAMES)
        self.assertGreater(decision.inference_ms, 0.0)
        self.assertAlmostEqual(sum(decision.scores.values()), 1.0, places=4)
        robot = SimulatedRobotAssist()
        cmd = robot.apply(decision)
        self.assertEqual(cmd["source"], "software_simulation")
        self.assertIn(cmd["command"], {ASSIST_CONTINUE, ASSIST_SLOW_AND_FLAG, ASSIST_HOLD_FOR_REVIEW})

    def test_main_missing_image_exit_code(self) -> None:
        import main as app_main

        buf = io.StringIO()
        with mock.patch.object(sys, "stderr", buf):
            code = app_main.main(["--image", str(ROOT / "no_such.jpg"), "--runs", "1", "--warmup", "0"])
        self.assertEqual(code, 1)
        self.assertIn("not found", buf.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
