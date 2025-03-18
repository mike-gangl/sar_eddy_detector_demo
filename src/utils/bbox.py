import numpy as np
import pandas as pd


def merge_csv_bboxes(csv_path):
    """
    Merges overlapping bounding boxes from a CSV file.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Convert bbox column from string to tuple of floats.
    df["bbox_parsed"] = df["bbox"].apply(parse_bbox)

    # Group rows by filename.
    results = {}
    for filename, group in df.groupby("filename"):
        # For each row, make a dict with bbox and confidence.
        boxes = [
            {"bbox": row, "confidences": [conf]}
            for row, conf in zip(group["bbox_parsed"], group["confidence"])
        ]

        merged = merge_overlapping_boxes(boxes)
        results[filename] = merged

    return results


def parse_bbox(bbox_str):
    # Remove brackets and extra whitespace, then split into float values.
    numbers = bbox_str.strip("[]").split()
    return tuple(map(float, numbers))


def merge_overlapping_boxes(boxes):
    """
    Expects boxes as a list of dicts with keys 'bbox' and 'confidences' (a list).
    Iteratively merge boxes that overlap.
    """
    merged = []
    changed = True
    while changed:
        changed = False
        new_merged = []
        while boxes:
            current = boxes.pop(0)
            merged_this = current
            i = 0
            while i < len(boxes):
                if boxes_overlap(merged_this["bbox"], boxes[i]["bbox"]):
                    # Merge the boxes.
                    merged_bbox = merge_boxes(merged_this["bbox"], boxes[i]["bbox"])
                    merged_confidences = (
                        merged_this["confidences"] + boxes[i]["confidences"]
                    )
                    merged_this = {
                        "bbox": merged_bbox,
                        "confidences": merged_confidences,
                    }
                    boxes.pop(i)
                    changed = True
                    # Restart checking with the new merged box.
                    i = 0
                else:
                    i += 1
            new_merged.append(merged_this)
        boxes = new_merged
        merged = boxes
    # Now average the confidences.
    for item in merged:
        item["confidence"] = np.mean(item["confidences"])
        del item["confidences"]
    return merged


def boxes_overlap(bbox1, bbox2):
    # bbox format: (xmin, ymin, xmax, ymax)
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    # They overlap if the boxes intersect in both dimensions.
    return not (xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2)


def merge_boxes(bbox1, bbox2):
    # Return the union of the two boxes.
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    return (min(xmin1, xmin2), min(ymin1, ymin2), max(xmax1, xmax2), max(ymax1, ymax2))
