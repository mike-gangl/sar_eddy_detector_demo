#!/usr/bin/env python

import argparse
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features


def clamp_top_percentile(data, percentile, debug=False):
    """
    Clamp (in-place) values above the given percentile to that percentile's value.

    Args:
        data (np.ndarray): Input array (float).
        percentile (float): If 95 => clamp top 5% to the 95th percentile value.
        debug (bool): Whether to print debug info.

    Returns:
        float or None: The clamp value at the given percentile, or None if no valid data.
    """
    valid = np.isfinite(data)
    if not np.any(valid):
        return None

    clamp_val = np.nanpercentile(data[valid], percentile)
    data[data > clamp_val] = clamp_val

    if debug:
        print(
            f"[DEBUG] Clamping above the {percentile}th percentile => {clamp_val:.4f}"
        )
    return clamp_val


def scale_to_uint16(data, ocean_min, ocean_max, debug=False):
    """
    Scale a float array to the 0..65535 range (uint16) given a min and max.

    Args:
        data (np.ndarray): Array of float data.
        ocean_min (float): Minimum valid value.
        ocean_max (float): Maximum valid value.
        debug (bool): Whether to print debug info.

    Returns:
        np.ndarray: The scaled uint16 array.
    """
    valid = np.isfinite(data)
    denom = (ocean_max - ocean_min) if (ocean_max != ocean_min) else 1.0

    # Initialize an empty uint16 array
    scaled = np.zeros_like(data, dtype=np.uint16)

    # Scale valid data to 0..65535
    scaled[valid] = np.round(((data[valid] - ocean_min) / denom) * 65535).astype(
        np.uint16
    )

    if debug:
        print(f"[DEBUG] Scaling data to uint16 with range [{ocean_min}, {ocean_max}]")
    return scaled


def expand_from_jp2(jp2_path, debug=False, **rasterio_read_kwargs):
    """
    Reads the compressed JP2 (uint16) file, looks for the "min_val" / "max_val" tags,
    and re-expands to the original float range. Returns a float array with nodata as NaNs.

    Args:
        jp2_path (str): Path to the compressed JP2.
        debug (bool): Whether to print debug messages.

    Returns:
        np.ndarray: Re-expanded float data.
    """
    with rasterio.open(jp2_path) as src:
        comp_data_uint16 = src.read(**rasterio_read_kwargs).astype(np.float32)
        tags = src.tags()

        if debug:
            print(f"[DEBUG] Tags read from JP2: {tags}")

        # Retrieve min/max from tags or use fallback values
        ocean_min = float(tags.get("min_val", 0))
        ocean_max = float(tags.get("max_val", 1))
        rng = ocean_max - ocean_min if ocean_max != ocean_min else 1.0

        # 0 => nodata
        comp_valid = comp_data_uint16 > 0
        comp_data_uint16[~comp_valid] = np.nan

        comp_data = (comp_data_uint16 / 65535.0) * rng + ocean_min

        if debug:
            print(
                "[DEBUG] Re-expanded Data Range:",
                f"[{np.nanmin(comp_data):.6f}, {np.nanmax(comp_data):.6f}]",
            )

    return comp_data


def mask_land(
    input_tif,
    shapefile,
    masked_tif,
    nodata_value=-9999,
    outlier_percentile=None,
    debug=False,
):
    """
    Masks land areas in a float GeoTIFF using a land-polygons shapefile, optionally
    masks top outliers, then writes the result with a specified 'nodata_value'.

    1) Rasterize the shapefile polygons (set them to NaN).
    2) If outlier_percentile is given, also mask out all pixels above that percentile.
    3) Write the result as a new float GeoTIFF with 'nodata_value'.

    Args:
        input_tif (str): Path to the original float GeoTIFF.
        shapefile (str): Path to land polygons shapefile.
        masked_tif (str): Path to output masked float GeoTIFF.
        nodata_value (float): Sentinel for masked (NaN) data.
        outlier_percentile (float): E.g., 99 means top 1% of ocean is masked out.
        debug (bool): If True, print additional debug messages.
    """
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        data = src.read(1).astype(np.float32)

        # Replace original nodata with NaN
        orig_nodata = src.nodata if src.nodata is not None else nodata_value
        data[data == orig_nodata] = np.nan

        if debug:
            print(
                "[DEBUG] Data range after reading input:",
                f"[{np.nanmin(data):.6f}, {np.nanmax(data):.6f}]",
            )

        # Rasterize land polygons => set them to NaN
        land_gdf = gpd.read_file(shapefile)
        if land_gdf.crs is None:
            land_gdf = land_gdf.set_crs("EPSG:4326")
        land_gdf_file_crs = land_gdf.to_crs(src.crs)

        shapes = ((geom, 1) for geom in land_gdf_file_crs.geometry)
        land_mask = features.rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            fill=0,
            transform=src.transform,
            all_touched=True,
            dtype=np.uint8,
        )
        data[land_mask == 1] = np.nan

        if debug:
            print(
                "[DEBUG] Data range after land masking:",
                f"[{np.nanmin(data):.6f}, {np.nanmax(data):.6f}]",
            )

        # Optionally exclude top outliers
        if outlier_percentile is not None and 0 < outlier_percentile < 100:
            clamp_val = np.nanpercentile(data, outlier_percentile)
            data[data > clamp_val] = np.nan
            if debug:
                print(
                    f"[DEBUG] Outlier masking above {outlier_percentile}th percentile: {clamp_val:.4f}"
                )
                print(
                    "[DEBUG] Data range after outlier masking:",
                    f"[{np.nanmin(data):.6f}, {np.nanmax(data):.6f}]",
                )

        # Update profile and write masked float raster with nodata
        profile.update({"nodata": nodata_value, "dtype": "float32"})

    # Convert np.nan => nodata_value before writing
    data = np.where(np.isfinite(data), data, nodata_value)

    with rasterio.open(masked_tif, "w", **profile) as dst:
        dst.write(data, 1)


def compress_sar_to_jp2_lossless(
    input_tif, output_jp2=None, use_same_percentile=None, debug=False
):
    """
    Reads a float GeoTIFF (already land/outlier masked), finds min/max, rescales
    to 16-bit, and writes a lossless JP2 (with reversible wavelet). Optionally
    clamps top values to avoid saturation at extremely high outliers.

    Args:
        input_tif (str): Path to the masked float GeoTIFF.
        output_jp2 (str): Output JP2 path. If None, it appends "_compressed_lossless".
        use_same_percentile (float): E.g., 95 => clamp top 5% for final scaling.
        debug (bool): If True, print debug messages.

    Returns:
        (str, (float, float)): (Path to output JP2, (ocean_min, ocean_max)).
    """
    if output_jp2 is None:
        base, _ = os.path.splitext(input_tif)
        output_jp2 = base + "_compressed_lossless.jp2"

    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        nodata_in = src.nodata if src.nodata is not None else -9999

        data = src.read().astype(np.float32)
        data[data == nodata_in] = np.nan

        if debug:
            print(
                "[DEBUG] Data range before scaling to uint16:",
                f"[{np.nanmin(data):.6f}, {np.nanmax(data):.6f}]",
            )

        valid = np.isfinite(data)
        if not np.any(valid):
            raise ValueError("No valid ocean pixels found for scaling!")

        # Optionally clamp top percentile
        if use_same_percentile is not None and 0 < use_same_percentile < 100:
            clamp_top_percentile(data, use_same_percentile, debug=debug)

        # Compute min/max across valid pixels
        ocean_min = np.nanmin(data[valid])
        ocean_max = np.nanmax(data[valid])

        # Update profile for JP2 in lossless mode (uint16)
        profile.update(
            {
                "driver": "JP2OpenJPEG",
                "count": data.shape[0],
                "dtype": "uint16",
                "quality": 100,
                "REVERSIBLE": "YES",  # Lossless wavelet transform
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "nodata": 0,  # We'll map nodata => 0
            }
        )

        # Scale to uint16
        scaled = scale_to_uint16(data, ocean_min, ocean_max, debug=debug)

        # Write JP2
        with rasterio.open(output_jp2, "w", **profile) as dst:
            dst.write(scaled)
            dst.update_tags(min_val=str(ocean_min), max_val=str(ocean_max))

        # Optional debug: re-read tags
        if debug:
            with rasterio.open(output_jp2) as test_src:
                test_tags = test_src.tags()
                print("[DEBUG] Tags written to JP2:", test_tags)

    return output_jp2, (ocean_min, ocean_max)


def validate_compressed(original_masked_tif, compressed_jp2, debug=False):
    """
    Opens the masked TIF (float) and the compressed JP2, re-expands the JP2 to float,
    and prints correlation & RMSE. Returns (orig_data, comp_data).

    Args:
        original_masked_tif (str): Path to the masked float GeoTIFF.
        compressed_jp2 (str): Path to the compressed JP2.
        debug (bool): If True, print debug messages.

    Returns:
        tuple: (orig_data, comp_data) as np.ndarray.
    """
    # Read original data
    with rasterio.open(original_masked_tif) as src_orig:
        orig_data = src_orig.read().astype(np.float32)
        nodata_in = src_orig.nodata if src_orig.nodata is not None else -9999
        orig_data[orig_data == nodata_in] = np.nan

    # Re-expand from JP2 using the extracted tags
    comp_data = reexpand_from_jp2(compressed_jp2, debug=debug)

    if debug:
        print(
            "[DEBUG] Original Data Range:",
            f"[{np.nanmin(orig_data):.6f}, {np.nanmax(orig_data):.6f}]",
        )
        print(
            "[DEBUG] Compressed Re-expanded Data Range:",
            f"[{np.nanmin(comp_data):.6f}, {np.nanmax(comp_data):.6f}]",
        )

    # Compute correlation and RMSE
    o = orig_data.ravel()
    c = comp_data.ravel()
    valid_mask = np.isfinite(o) & np.isfinite(c)
    if not np.any(valid_mask):
        if debug:
            print("[DEBUG] No valid ocean pixels to compare.")
        return orig_data, comp_data

    corr = np.corrcoef(o[valid_mask], c[valid_mask])[0, 1]
    rmse = np.sqrt(np.mean((o[valid_mask] - c[valid_mask]) ** 2))

    print(f"Correlation: {corr:.4f}, RMSE: {rmse:.6f}")
    return orig_data, comp_data


def plot_histogram_comparison(orig_data, comp_data, bins=50):
    """
    Plot the distributions of original vs. compressed data with a KL metric in the title.
    """

    def compute_kl_divergence(a, b, bins=50, eps=1e-12):
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return 0.0
        hist_a, edges = np.histogram(a, bins=bins, density=True)
        hist_b, _ = np.histogram(b, bins=edges, density=True)

        hist_a = np.clip(hist_a, eps, None)
        hist_b = np.clip(hist_b, eps, None)
        return np.sum(hist_a * np.log(hist_a / hist_b))

    ovals = orig_data.ravel()
    cvals = comp_data.ravel()
    valid_mask = np.isfinite(ovals) & np.isfinite(cvals)
    if not np.any(valid_mask):
        print("No valid data for histogram comparison.")
        return

    kl = compute_kl_divergence(ovals[valid_mask], cvals[valid_mask], bins=bins)

    data_min = np.nanmin([ovals[valid_mask], cvals[valid_mask]])
    data_max = np.nanmax([ovals[valid_mask], cvals[valid_mask]])
    hist_range = (data_min, data_max)

    plt.figure(figsize=(6, 5))
    plt.hist(
        ovals[valid_mask],
        bins=bins,
        alpha=0.5,
        density=True,
        label="Original",
        range=hist_range,
    )
    plt.hist(
        cvals[valid_mask],
        bins=bins,
        alpha=0.5,
        density=True,
        label="Compressed",
        range=hist_range,
    )
    plt.title(f"Value Distribution (KL={kl:.2f})")
    plt.xlim(hist_range[0], hist_range[1])
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison_with_residual(orig_data, comp_data, band=0):
    """
    Plot side-by-side original, compressed, and residual (C - O) for a given band index.
    """
    o = orig_data[band].copy()
    c = comp_data[band].copy()

    # Common scale ignoring outliers
    vmin = np.nanmin([o, c])
    vmax = np.nanmax([o, c])
    if np.isfinite(vmax) and np.isfinite(vmin):
        vmax = np.nanpercentile([o, c], 99)

    res = c - o
    res_absmax = np.nanmax(np.abs(res))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axs[0].imshow(o, vmin=vmin, vmax=vmax)
    axs[0].set_title("Original")
    plt.colorbar(im0, ax=axs[0], orientation="horizontal")

    im1 = axs[1].imshow(c, vmin=vmin, vmax=vmax)
    axs[1].set_title("Compressed")
    plt.colorbar(im1, ax=axs[1], orientation="horizontal")

    im2 = axs[2].imshow(res, vmin=-res_absmax, vmax=res_absmax, cmap="RdBu")
    axs[2].set_title("Residual (C - O)")
    plt.colorbar(im2, ax=axs[2], orientation="horizontal")

    plt.tight_layout()
    plt.show()


def main():
    """
    Example usage:
      python compress_sar_tiff_with_land_mask.py \
        --input my_sar.tif \
        --shapefile land_polygons.shp \
        --outlier_percentile 99 \
        --compress_percentile 95 \
        --visualize
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input SAR GeoTIFF (float32).")
    parser.add_argument("--shapefile", required=True, help="Land polygons shapefile.")
    parser.add_argument(
        "--temp_masked_tif",
        default="masked_temp.tif",
        help="Intermediate masked output.",
    )
    parser.add_argument(
        "--outlier_percentile",
        type=float,
        default=None,
        help="Exclude top X%% of data by setting them to nodata, e.g. 99 => cut top 1%%.",
    )
    parser.add_argument(
        "--compress_percentile",
        type=float,
        default=99,
        help="Clamp top X%% of data in the final scaling step, e.g. 95 => saturate top 5%%.",
    )
    parser.add_argument("--output", default=None, help="Output JP2 path.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Plot histograms & residuals.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print debug information.",
    )
    args = parser.parse_args()

    # 1. Mask land and optionally top outliers
    if args.debug:
        print(f"[DEBUG] Masking land from {args.input} using {args.shapefile}")
    mask_land(
        args.input,
        args.shapefile,
        args.temp_masked_tif,
        nodata_value=-9999,
        outlier_percentile=args.outlier_percentile,
        debug=args.debug,
    )

    # 2. Compress to JP2 (16-bit, lossless)
    if args.debug:
        print("[DEBUG] Compressing masked TIF to JP2.")
    jp2_file, (omin, omax) = compress_sar_to_jp2_lossless(
        args.temp_masked_tif,
        output_jp2=args.output,
        use_same_percentile=args.compress_percentile,
        debug=args.debug,
    )

    print(f"Compressed JP2 => {jp2_file}")
    print(f"Ocean range used for scaling: [{omin:.6f}, {omax:.6f}]")

    # 3. Validate correlation & RMSE
    if args.debug:
        print("[DEBUG] Validating compression results...")
    orig_data, comp_data = validate_compressed(
        args.temp_masked_tif, jp2_file, debug=args.debug
    )

    # 4. Optional visualizations
    if orig_data is not None and comp_data is not None and args.visualize:
        print("\nVisualizing histogram comparison...")
        plot_histogram_comparison(orig_data, comp_data)

        print("\nVisualizing comparison with residual (Band 1)...")
        plot_comparison_with_residual(orig_data, comp_data, band=0)

    print("\nDone. Temporary masked file:", args.temp_masked_tif)


if __name__ == "__main__":
    main()
