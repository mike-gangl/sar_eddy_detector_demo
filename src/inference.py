# src/inference.py

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import SARImageNormalizer, SARTileDataset
from src.model import get_model
from src.utils import merge_csv_bboxes, parse_bbox
from src.visualize_eddy_bbox import create_preview_with_boxes


class EddyDetector:
    """Main class for eddy detection in SAR imagery. Simplified for demo."""

    def __init__(self, config):
        """
        Initialize the eddy detector.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._create_transform()
        self.dataset = None
        self.model = None
        self.positive_detections = []

    def _create_transform(self) -> transforms.Compose:
        """Create the transformation pipeline for SAR imagery. Simplified for demo."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.NEAREST
                ),
                SARImageNormalizer.global_min_max_normalize,
                SARImageNormalizer.per_tile_normalize,
                SARImageNormalizer.boost_dark_images,
            ]
        )

    def setup_dataset(self) -> None:
        """Initialize the SAR dataset."""
        self.dataset = SARTileDataset(
            geotiff_dir=self.config.geotiff_dir,
            land_shapefile=self.config.land_shapefile,
            window_size=self.config.window_size,
            stride_factor=self.config.stride_factor,
            land_threshold=self.config.land_threshold,
            nodata_threshold=self.config.nodata_threshold,
            var_threshold=self.config.var_threshold,
            preprocessed_dir=self.config.preprocessed_dir,  # Not used in demo config
            transform=self.transform,
        )

    def setup_model(self) -> bool:
        """
        Initialize and load the pretrained model. Simplified for demo.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        self.model = get_model(self.config).to(self.device)  # Use simplified get_model

        if not self.config.pretrain:
            print(
                "ERROR: No pretrained model specified. Please provide --pretrain argument in config."
            )
            return False

        if not os.path.isfile(self.config.pretrain):
            print(f"ERROR: No checkpoint found at '{self.config.pretrain}'")
            return False

        self.model.eval()
        return True

    def run_inference(self) -> None:
        """Run inference on the dataset and save results."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            prefetch_factor=4,
            pin_memory=True,
        )

        # Prepare output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Running inference on SAR imagery...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Processing tiles")
            ):
                self._process_batch(batch_idx, batch, output_dir)

        # Save detection results
        self._save_detection_results()

    def _process_batch(
        self, batch_idx: int, batch: Dict[str, Any], output_dir: Path
    ) -> None:
        """
        Process a batch of SAR tiles.

        Args:
            batch_idx: Index of the current batch
            batch: Dictionary containing batch data
            output_dir: Directory to save positive detections
        """
        # Move images to device and extract batch metadata
        images = batch["image"].to(self.device)

        # Run model inference on batch
        outputs = self.model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()

        # Process each sample in the batch
        for i in range(len(batch["filename"])):
            # fmt: off
            # Save positive detections
            self._handle_detection(
                batch_idx, i, preds[i], probs[i],
                {k: v[i] for k, v in batch.items()},
                images[i], output_dir
            )

    def _handle_detection(
        self,
        batch_idx: int,
        sample_idx: int,
        prediction: int,
        probabilities: torch.Tensor,
        metadata: Dict[str, Any],
        image: torch.Tensor,
        output_dir: Path,
    ) -> None:
        """
        Handle a single detection result.

        Args:
            batch_idx: Current batch index
            sample_idx: Index of sample within the batch
            prediction: Predicted class index
            probabilities: Model confidence scores
            metadata: Sample metadata (filename, bbox, etc.)
            image: Image tensor
            output_dir: Directory to save positive detections
        """
        # Skip if not a positive detection
        if prediction != self.config.positive_class_index:
            return

        # Extract metadata
        filename = metadata["filename"]
        bbox = metadata["bbox"].cpu().numpy()
        # bbox_latlon = metadata["bbox_latlon"].cpu().numpy()
        confidence = float(probabilities[prediction].cpu().numpy())

        # Store detection information
        self.positive_detections.append(
            {
                "filename": filename,
                "bbox": bbox,
                # "bbox_latlon": bbox_latlon,
                "confidence": confidence,
            }
        )

        print(
            f"Found positive eddy in {filename} with confidence {confidence:.2f} at lat-lon: {bbox}"
        )

        # Save the detection as GeoTIFF # Commented out for demo to keep it simple
        # self._save_detection_geotiff(
        #     batch_idx,
        #     sample_idx,
        #     filename,
        #     image,
        #     metadata["transform"],
        #     metadata["crs"],
        #     output_dir,
        # )

    def _save_detection_results(self) -> None:
        """Save the detection results to CSV."""
        df = pd.DataFrame(self.positive_detections)
        df.to_csv(self.config.identification_table_path, index=False)
        merged_results = merge_csv_bboxes(self.config.identification_table_path)
        for fname, merges in merged_results.items():
            print(f"File: {fname}")
            for m in merges:
                print(
                    f"  Merged bbox: {m['bbox']}, Combined confidence: {m['confidence']:.3f}"
                )
        # save to merged dataframe csv
        rows_list = []
        for filename, detections in merged_results.items():
            for detection in detections:
                bbox_str = " ".join(map(str, detection["bbox"]))
                confidence = detection["confidence"]
                rows_list.append(
                    {"filename": filename, "bbox": bbox_str, "confidence": confidence}
                )

        df = pd.DataFrame(rows_list)
        merged_filename = (
            f"{os.path.splitext(self.config.identification_table_path)[0]}_merged.csv"
        )
        df.to_csv(merged_filename, index=False)
        self.config.merged_identification_table_path = merged_filename
        print(
            f"Saved {len(self.positive_detections)} positive detections to: {self.config.identification_table_path} and merged bounding box results to: {merged_filename}"
        )

    def create_eddy_previews(self, confidence_threshold=0.999, merged=False) -> None:
        """
        For each TIFF file processed, create a downsampled PNG preview with bounding boxes
        """
        # Create output directory with threshold
        output_dir = os.path.join(
            os.path.dirname(self.config.identification_table_path),
            f"previews_{confidence_threshold}",
        )
        os.makedirs(output_dir, exist_ok=True)

        if not merged:  # i.e., plot each individual positive eddy bbox
            df = pd.read_csv(self.config.identification_table_path)
        else:
            df = pd.read_csv(self.config.merged_identification_table_path)
        df["bbox"] = df["bbox"].apply(parse_bbox)

        tif_folder = self.dataset.preprocessed_dir

        for i, file in enumerate(df["filename"].unique()):
            # Filter data for current file
            file_df = df[df["filename"] == file]

            # Extract bounding boxes and confidences
            bounding_boxes = file_df["bbox"].tolist()
            confidences = file_df["confidence"].tolist()

            output_png = os.path.join(
                output_dir,
                f"{os.path.splitext(file)[0]}{'_merged' if merged else ''}.png",
            )

            # Pass both bounding boxes and confidences to the function
            create_preview_with_boxes(
                os.path.join(tif_folder, file),
                bounding_boxes,
                output_png,
                confidences=confidences,
                confidence_threshold=confidence_threshold,
            )
