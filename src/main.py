import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add repo root to path for relative imports

import yaml

from src.inference import EddyDetector
from src.utils import parse_args_from_yaml


def main():
    args = parse_args_from_yaml("config/inference.yaml")  # Load config from YAML
    detector = EddyDetector(args)
    detector.setup_dataset()
    if detector.setup_model():
        detector.run_inference()
        print(f"Inference complete. Results saved in: {args.output_dir}")

        detector.create_eddy_previews(confidence_threshold=0.999)


if __name__ == "__main__":
    main()
