import os
import sys

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show

from utils.bbox import boxes_overlap, merge_boxes, parse_bbox
from utils.compress_sar_with_jpeg2000 import expand_from_jp2

# set matplotlib default font size to 24
matplotlib.rcParams.update({"font.size": 24})


def create_preview_with_boxes(
    input_tif,
    bounding_boxes,
    out_png,
    scale_factor=0.1,
    confidence_threshold=0.999,
    confidences=None,
):
    """
    Create a downsampled PNG preview of a GeoTIFF and draw bounding boxes.

    :param input_tif: Path to the large GeoTIFF.
    :param bounding_boxes: List of bounding boxes in lat/lon [(xmin, ymin, xmax, ymax), ...].
    :param out_png: Output path for the PNG preview.
    :param scale_factor: How much to downsample (0.1 => 10% of original).
    :param confidence_threshold: Minimum confidence score to include (default: 0.999).
    :param confidences: List of confidence values corresponding to bounding_boxes.
    """
    # Filter bounding boxes by confidence if provided
    if confidences is not None:
        filtered_boxes = []
        filtered_confidences = []
        for box, conf in zip(bounding_boxes, confidences):
            if conf >= confidence_threshold:
                filtered_boxes.append(box)
                filtered_confidences.append(conf)
        bounding_boxes = filtered_boxes
        confidences = filtered_confidences
        print(
            f"Using {len(bounding_boxes)} boxes with confidence >= {confidence_threshold}"
        )

    with rasterio.open(input_tif) as src:
        # Calculate the new dimensions
        new_width = int(src.width * scale_factor)
        new_height = int(src.height * scale_factor)

        # Read the data at reduced resolution
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype(np.float32)

        if input_tif.endswith("jp2") or src.driver == "JP2OpenJPEG":
            # Expand the data if it was compressed with JPEG2000
            data = expand_from_jp2(
                input_tif,
                out_shape=(new_height, new_width),
                resampling=rasterio.enums.Resampling.bilinear,
            )

        # Adjust the transform so pixel coordinates match the downsampled data
        # (rasterio.transform.scale expects how many *original* pixels each new pixel covers)
        scale_x = src.width / float(new_width)
        scale_y = src.height / float(new_height)
        new_transform = src.transform * src.transform.scale(scale_x, scale_y)

        # Plot the data
        fig, ax = plt.subplots(figsize=(10, 8))
        show(
            data,
            transform=new_transform,
            ax=ax,
            cmap="gray",
            vmin=0,
            vmax=np.nanpercentile(data, 98),
        )  # or any other colormap

        # For each bounding box in lat/lon, draw directly using geographic coordinates
        for i, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
            # Create a width and height for the rectangle
            width = xmax - xmin
            height = ymax - ymin

            # Create rectangle with geographic coordinates directly
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                alpha=0.7,
            )
            ax.add_patch(rect)

            # Add confidence label if available
            # if confidences is not None:
            #     plt.text(xmin, ymin, f"{confidences[i]:.3f}",
            #              color='red', fontsize=8, backgroundcolor='white')

        plt.title(f"Preview with Merged Bounding Boxes (conf ≥ {confidence_threshold})")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    # Example usage
    csv_file = sys.argv[1]
    tif_file = sys.argv[2]
    output_dir = os.path.join(os.path.dirname(csv_file), "previews_0.9")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    df["bbox"] = df["bbox"].apply(parse_bbox)

    file_df = df[df["filename"] == os.path.basename(tif_file)]

    bounding_boxes = file_df["bbox"].tolist()
    confidences = file_df["confidence"].tolist()

    output_png = os.path.join(
        output_dir, f"{os.path.splitext(os.path.basename(tif_file))[0]}.png"
    )
    print(f"Creating eddy bounding box preview for {tif_file} at {output_png}")

    # Pass both bounding boxes and confidences to the function
    create_preview_with_boxes(
        tif_file,
        bounding_boxes,
        output_png,
        confidences=confidences,
        confidence_threshold=0.9,
    )
