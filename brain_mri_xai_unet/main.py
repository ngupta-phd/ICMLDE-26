"""CLI for XAI U-Net brain MRI edge inference."""

import argparse
import sys
import traceback

from config import CHECKPOINT_PATH, DEFAULT_IMAGE, DEFAULT_OUTPUT_DIR
from pipeline import PipelineError, run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run LightweightUNet on a 2D brain MRI slice and save "
            "the segmentation mask, overlay, and probability map."
        )
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Path to an MRI image (default: sample_data/test_mri.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved visualizations (default: outputs/)",
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_PATH,
        help="Path to unet_brain_mri.pth",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (edge-style), even if CUDA is available",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Also save a bottleneck Grad-CAM overlay (extra backward pass)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip a warmup forward pass before timing",
    )
    return parser.parse_args()


def print_report(result):
    print("\n============================================")
    print("   XAI U-Net Brain MRI Edge Inference")
    print("============================================")
    print(f"Device:                  {result['device']}")
    print(f"Checkpoint:              {result['checkpoint_path']}")
    print(f"Input image:             {result['image_path']}")
    print(f"Input tensor:            {result['input_shape']}")
    print(f"U-Net output:            {result['output_shape']}")
    print(f"Mask shape:              {result['mask_shape']}")
    print(f"Threshold:               {result['threshold']}")
    print(f"Positive pixels:         {result['positive_pixels']}")
    print(f"Mask coverage:           {result['coverage_pct']:.2f}%")
    print(
        "Probability min/max/mean:"
        f" {result['prob_min']:.4f} / {result['prob_max']:.4f} / {result['prob_mean']:.4f}"
    )
    print(f"Parameters:              {result['parameter_count']}")
    warmup_note = " (after warmup)" if result["warmup"] else ""
    print(f"Inference time:          {result['inference_ms']:.1f} ms{warmup_note}")
    print(f"Original MRI:            {result['paths']['original']}")
    print(f"Segmentation mask:       {result['paths']['mask']}")
    print(f"MRI + mask overlay:      {result['paths']['overlay']}")
    print(f"Probability map:         {result['paths']['probability']}")
    print(f"Combined panel:          {result['paths']['panel']}")
    if "gradcam" in result["paths"]:
        print(f"Grad-CAM overlay:        {result['paths']['gradcam']}")
    print("============================================")
    print("Research/educational prototype - not a medical diagnostic system.")
    print()


def main():
    args = parse_args()
    try:
        result = run_pipeline(
            image_path=args.image,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            force_cpu=args.cpu,
            with_gradcam=args.gradcam,
            warmup=not args.no_warmup,
        )
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1

    print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
