"""Load the U-Net checkpoint, run sample MRI inference, print timings."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CHECKPOINT_PATH, DEFAULT_IMAGE, DEFAULT_OUTPUT_DIR
from pipeline import PipelineError, run_pipeline


def main():
    try:
        result = run_pipeline(
            image_path=DEFAULT_IMAGE,
            output_dir=DEFAULT_OUTPUT_DIR,
            checkpoint_path=CHECKPOINT_PATH,
            force_cpu=True,
            with_gradcam=False,
            warmup=True,
        )
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"model loaded: {result['checkpoint_path']}")
    print(f"inference device: {result['device']}")
    print(f"parameter count: {result['parameter_count']}")
    print(f"sample image: {result['image_path']}")
    print(f"input dimensions: {result['input_shape']}")
    print(f"output dimensions: {result['output_shape']}")
    print(f"inference time: {result['inference_ms']:.1f} ms")
    print(
        "output min/max/mean: "
        f"{result['prob_min']:.4f} / {result['prob_max']:.4f} / {result['prob_mean']:.4f}"
    )
    print(
        "mask statistics: "
        f"positive_pixels={result['positive_pixels']} "
        f"coverage={result['coverage_pct']:.2f}% "
        f"threshold={result['threshold']}"
    )
    print(f"output path: {result['paths']['mask']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
