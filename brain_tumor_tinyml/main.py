"""TinyML medical-robotics real-time assist (research prototype).

Input MRI frame → MobileNetV3-Small → assist recommendation.
Optional U-Net localization is off by default (extra latency).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import config
from models.classifier import load_classifier
from models.unet import load_unet_model
from src.inference import classify_image
from src.robot_sim import SimulatedRobotAssist, format_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TinyML Medical Robotics Real-Time Assist (research prototype, not a medical device)."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=config.SAMPLE_IMAGE,
        help="Path to an RGB/MRI slice image (default: sample_data/test_mri.jpg)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=config.CLASSIFIER_WEIGHTS,
        help="Path to MobileNetV3-Small .pth weights",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Also run optional LightweightUNet localization (not the real-time default).",
    )
    parser.add_argument(
        "--unet-weights",
        type=Path,
        default=config.UNET_WEIGHTS,
        help="Path to U-Net .pth weights (used only with --segment)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: cpu (default, edge-oriented) or cuda if available",
    )
    parser.add_argument("--warmup", type=int, default=config.WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=config.TIMING_RUNS, help="Timed forward passes after warmup")
    parser.add_argument(
        "--stream",
        type=int,
        default=0,
        metavar="N",
        help="Repeat inference N times to simulate a real-time camera loop (0 = once)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Use --device cpu.", file=sys.stderr)
        return 1

    if not args.image.is_file():
        print(f"Error: Input image not found: {args.image}", file=sys.stderr)
        return 1
    if not args.weights.is_file():
        print(
            f"Error: Classifier weights not found: {args.weights}\n"
            "Place best_mobilenetv3.pth in models/ or pass --weights.",
            file=sys.stderr,
        )
        return 1
    if args.segment and not args.unet_weights.is_file():
        print(
            f"Error: U-Net weights not found: {args.unet_weights}\n"
            "Place unet_brain_mri.pth in models/ or omit --segment.",
            file=sys.stderr,
        )
        return 1

    print("TinyML Medical Robotics Real-Time Assist")
    print("Decision support / research prototype - not a medical device.")
    print(f"Device: {device_name}")
    print(f"Image:  {args.image}")
    print(f"Model:  {args.weights}")
    print()

    robot = SimulatedRobotAssist()
    loops = args.stream if args.stream > 0 else 1

    try:
        model = load_classifier(weights_path=args.weights, device=device_name)
        unet = load_unet_model(weights_path=args.unet_weights, device=device_name) if args.segment else None
        last_report = ""
        for i in range(loops):
            decision = classify_image(
                image_path=args.image,
                weights_path=args.weights,
                device=device_name,
                segment=args.segment,
                unet_weights=args.unet_weights,
                warmup=args.warmup if i == 0 else 0,
                runs=args.runs if i == 0 else 1,
                model=model,
                unet=unet,
            )
            command = robot.apply(decision)
            last_report = format_report(decision, command)
            if loops > 1:
                print(
                    f"[frame {i + 1}/{loops}] {decision.assist_state} "
                    f"{decision.condition} {decision.inference_ms:.2f} ms"
                )
        print(last_report)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
