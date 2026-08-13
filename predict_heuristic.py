"""
predict_heuristic.py

Heuristic baseline for aerial-image water fringe analysis.

Pipeline:
1. Detect water using RGB color + local texture conditions.
2. Detect forest using texture roughness + green-channel dominance.
3. Detect sand/road using brightness, color, and texture conditions.
4. Apply morphological post-processing.
5. Visualize terrain boundaries on the original image.
"""

import os
import sys
import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from scipy import ndimage


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = 512

# Forest detection thresholds
FOREST_STD_MIN = 6.0
FOREST_GREEN_RATIO = 0.95
FOREST_MIN_AREA_PX = 500


# ============================================================
# Utility functions
# ============================================================

def find_image(filename):
    """
    Search for an image in the validation or training dataset.
    """
    for split in ["val", "train"]:
        path = f"data/processed/{split}/images/{filename}"

        if os.path.exists(path):
            return path

    return None


def compute_texture_map(image_rgb):
    """
    Compute a multi-scale local standard-deviation texture map.
    """
    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    def local_std(gray_image, kernel_size):
        mean = cv2.boxFilter(
            gray_image,
            -1,
            (kernel_size, kernel_size)
        )

        squared_mean = cv2.boxFilter(
            gray_image ** 2,
            -1,
            (kernel_size, kernel_size)
        )

        return np.sqrt(
            np.clip(squared_mean - mean ** 2, 0, None)
        )

    texture_small = local_std(gray, 7)
    texture_large = local_std(gray, 31)

    return texture_small * 0.4 + texture_large * 0.6


# ============================================================
# Heuristic segmentation
# ============================================================

def detect_forest(image_rgb, texture_map):
    """
    Detect forest using texture roughness and green-channel dominance.
    """
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)

    rough_condition = texture_map > FOREST_STD_MIN

    green_condition = (
        (g > r * FOREST_GREEN_RATIO)
        & (g > b * 0.90)
    )

    forest_mask = (
        rough_condition & green_condition
    ).astype(np.uint8)

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (35, 35)
    )

    forest_mask = cv2.morphologyEx(
        forest_mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    forest_mask = cv2.morphologyEx(
        forest_mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(forest_mask)
    )

    filtered = np.zeros_like(forest_mask)

    for label_id in range(1, num_labels):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area >= FOREST_MIN_AREA_PX:
            filtered[labels == label_id] = 1

    return filtered.astype(bool)


def compute_heuristic_water(image_rgb):
    """
    Detect water using RGB thresholds and local texture smoothness.
    """
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    mean_img = cv2.boxFilter(
        gray,
        -1,
        (31, 31)
    )

    squared_mean = cv2.boxFilter(
        gray ** 2,
        -1,
        (31, 31)
    )

    local_std = np.sqrt(
        np.clip(
            squared_mean - mean_img ** 2,
            0,
            None
        )
    )

    water_condition = (
        (r < 85)
        & (g < 110)
        & (b < 110)
        & (g > r)
        & (local_std < 5.0)
    )

    water_mask = water_condition.astype(np.uint8)

    kernel = np.ones(
        (9, 9),
        dtype=np.uint8
    )

    water_mask = cv2.morphologyEx(
        water_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return water_mask


# ============================================================
# Prediction
# ============================================================

def predict_single(image_path):

    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]

    print(f"\n[Heuristic prediction] {filename}")

    # --------------------------------------------------------
    # 1. Load image
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:

        image = src.read([1, 2, 3])
        image = np.moveaxis(
            image,
            0,
            -1
        )

    image_512 = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # 2. Texture analysis
    # --------------------------------------------------------

    texture_map = compute_texture_map(
        image_512
    )

    # --------------------------------------------------------
    # 3. Heuristic water detection
    # --------------------------------------------------------

    water_mask = compute_heuristic_water(
        image_512
    )

    water_mask = ndimage.binary_fill_holes(
        water_mask
    ).astype(np.uint8)

    kernel_7 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    water_mask = cv2.morphologyEx(
        water_mask,
        cv2.MORPH_OPEN,
        kernel_7,
        iterations=2
    )

    water_mask = cv2.morphologyEx(
        water_mask,
        cv2.MORPH_CLOSE,
        kernel_7,
        iterations=3
    )

    water_mask = ndimage.binary_fill_holes(
        water_mask
    ).astype(np.uint8)

    # Keep the largest connected water region
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            water_mask
        )
    )

    if num_labels > 1:

        largest_label = (
            1
            + np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )

        water_mask = (
            labels == largest_label
        ).astype(np.uint8)

    # Binary map for visualization
    water_probability = (
        water_mask.astype(np.float32)
    )

    # --------------------------------------------------------
    # 4. Sand / road detection
    # --------------------------------------------------------

    r = image_512[:, :, 0].astype(float)
    g = image_512[:, :, 1].astype(float)
    b = image_512[:, :, 2].astype(float)

    brightness = (
        r + g + b
    ) / 3.0

    color_difference = (
        np.abs(r - g)
        + np.abs(g - b)
        + np.abs(r - b)
    )

    asphalt_condition = (
        (color_difference < 20)
        & (brightness > 40)
        & (brightness < 160)
        & (texture_map < 8.0)
    )

    sand_condition = (
        (
            (brightness > 115)
            & (r > b * 0.85)
        )
        | asphalt_condition
    )

    sand_mask = sand_condition.astype(
        np.uint8
    )

    sand_mask[water_mask == 1] = 0

    kernel_3 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    kernel_15 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    sand_mask = cv2.morphologyEx(
        sand_mask,
        cv2.MORPH_OPEN,
        kernel_3,
        iterations=1
    )

    sand_mask = cv2.morphologyEx(
        sand_mask,
        cv2.MORPH_CLOSE,
        kernel_15,
        iterations=1
    )

    # --------------------------------------------------------
    # 5. Forest detection
    # --------------------------------------------------------

    forest_mask = detect_forest(
        image_512,
        texture_map
    )

    forest_mask = (
        forest_mask
        & ~water_mask.astype(bool)
        & ~sand_mask.astype(bool)
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            forest_mask.astype(np.uint8)
        )
    )

    forest_single = np.zeros_like(
        forest_mask,
        dtype=np.uint8
    )

    contours = []

    if num_labels > 1:

        areas = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        max_area = areas.max()

        keep_ids = [
            index + 1
            for index, area in enumerate(areas)
            if area >= max_area * 0.10
        ]

        forest_clean = np.zeros_like(
            forest_mask,
            dtype=np.uint8
        )

        for keep_id in keep_ids:
            forest_clean[
                labels == keep_id
            ] = 1

        kernel_merge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (20, 20)
        )

        forest_merged = cv2.dilate(
            forest_clean,
            kernel_merge,
            iterations=1
        )

        forest_merged[
            water_mask == 1
        ] = 0

        forest_merged[
            sand_mask == 1
        ] = 0

        forest_merged = (
            ndimage.binary_fill_holes(
                forest_merged
            ).astype(np.uint8)
        )

        contours, _ = cv2.findContours(
            forest_merged,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

        if contours:

            cv2.drawContours(
                forest_single,
                [contours[0]],
                -1,
                1,
                -1
            )

    else:

        forest_single = (
            forest_mask.copy()
            .astype(np.uint8)
        )

        contours, _ = cv2.findContours(
            forest_single,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

    forest_single[
        water_mask == 1
    ] = 0

    forest_single[
        sand_mask == 1
    ] = 0

    # --------------------------------------------------------
    # 6. Boundary extraction
    # --------------------------------------------------------

    main_forest_contour = (
        contours[0]
        if contours
        else None
    )

    smooth_forest = None

    if main_forest_contour is not None:

        epsilon = (
            0.0008
            * cv2.arcLength(
                main_forest_contour,
                True
            )
        )

        smooth_forest = cv2.approxPolyDP(
            main_forest_contour,
            epsilon,
            True
        )

    water_contours, _ = cv2.findContours(
        water_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    water_contours = sorted(
        water_contours,
        key=cv2.contourArea,
        reverse=True
    )

    main_water_contour = (
        water_contours[0]
        if water_contours
        else None
    )

    smooth_water = None

    if main_water_contour is not None:

        epsilon_water = (
            0.0005
            * cv2.arcLength(
                main_water_contour,
                True
            )
        )

        smooth_water = cv2.approxPolyDP(
            main_water_contour,
            epsilon_water,
            True
        )

    sand_contours, _ = cv2.findContours(
        sand_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    smooth_sands = []

    for contour in sand_contours:

        if cv2.contourArea(contour) > 200:

            epsilon_sand = (
                0.001
                * cv2.arcLength(
                    contour,
                    True
                )
            )

            smooth_contour = (
                cv2.approxPolyDP(
                    contour,
                    epsilon_sand,
                    True
                )
            )

            smooth_sands.append(
                smooth_contour
            )

    # --------------------------------------------------------
    # 7. Draw boundaries
    # --------------------------------------------------------

    image_contour = image_512.copy()

    if smooth_forest is not None:

        cv2.drawContours(
            image_contour,
            [smooth_forest],
            -1,
            (255, 220, 0),
            3
        )

        cv2.drawContours(
            image_contour,
            [smooth_forest],
            -1,
            (30, 180, 30),
            2
        )

    if smooth_water is not None:

        cv2.drawContours(
            image_contour,
            [smooth_water],
            -1,
            (255, 220, 0),
            3
        )

        cv2.drawContours(
            image_contour,
            [smooth_water],
            -1,
            (30, 100, 200),
            2
        )

    for smooth_sand in smooth_sands:

        cv2.drawContours(
            image_contour,
            [smooth_sand],
            -1,
            (255, 220, 0),
            3
        )

        cv2.drawContours(
            image_contour,
            [smooth_sand],
            -1,
            (30, 140, 230),
            2
        )

    image_contour = (
        image_contour.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # 8. Calculate area ratios
    # --------------------------------------------------------

    total_pixels = IMG_SIZE * IMG_SIZE

    water_ratio = (
        water_mask.sum()
        / total_pixels
        * 100
    )

    forest_ratio = (
        forest_single.sum()
        / total_pixels
        * 100
    )

    sand_ratio = (
        sand_mask.sum()
        / total_pixels
        * 100
    )

    other_ratio = (
        100
        - water_ratio
        - forest_ratio
        - sand_ratio
    )

    # --------------------------------------------------------
    # 9. Visualization
    # --------------------------------------------------------

    image_show = (
        image_512.astype(np.float32)
        / 255.0
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(21, 7)
    )

    fig.patch.set_facecolor(
        "#0f0f1a"
    )

    for axis in axes:
        axis.set_facecolor(
            "#0f0f1a"
        )

    # Original image
    axes[0].imshow(
        image_show
    )

    axes[0].set_title(
        "Original Image",
        fontsize=13,
        fontweight="bold",
        color="white"
    )

    axes[0].axis("off")

    # Heuristic water map
    heatmap = axes[1].imshow(
        water_probability,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    colorbar = plt.colorbar(
        heatmap,
        ax=axes[1],
        fraction=0.046,
        pad=0.04
    )

    colorbar.set_label(
        "Heuristic Water Map",
        color="white",
        fontsize=8
    )

    axes[1].set_title(
        "Heuristic Water Map",
        fontsize=13,
        fontweight="bold",
        color="white"
    )

    axes[1].axis("off")

    # Final boundary overlay
    axes[2].imshow(
        image_contour
    )

    water_legend = mpatches.Patch(
        edgecolor=(0.12, 0.39, 0.78),
        facecolor="none",
        linewidth=2,
        label=f"Water Boundary {water_ratio:.1f}%"
    )

    forest_legend = mpatches.Patch(
        edgecolor=(0.12, 0.70, 0.12),
        facecolor="none",
        linewidth=2,
        label=f"Forest Boundary {forest_ratio:.1f}%"
    )

    sand_legend = mpatches.Patch(
        edgecolor=(0.85, 0.60, 0.30),
        facecolor="none",
        linewidth=2,
        label=f"Sand/Road Boundary {sand_ratio:.1f}%"
    )

    other_legend = mpatches.Patch(
        facecolor="#555",
        label=f"Other {other_ratio:.1f}%"
    )

    axes[2].legend(
        handles=[
            water_legend,
            forest_legend,
            sand_legend,
            other_legend
        ],
        loc="lower right",
        fontsize=10,
        facecolor="#0f0f1a",
        edgecolor="white",
        labelcolor="white"
    )

    axes[2].set_title(
        "Heuristic Terrain Boundary",
        fontsize=13,
        fontweight="bold",
        color="white"
    )

    axes[2].axis("off")

    fig.suptitle(
        (
            "HEURISTIC BASELINE | "
            f"Color + Texture | {filename}"
        ),
        fontsize=13,
        fontweight="bold",
        color="white"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # 10. Save result
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        f"results/"
        f"heuristic_pred_{base_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "AP_HR_2021_0179_01"
    )

    filename = (
        target
        if target.endswith(".tif")
        else f"{target}.tif"
    )

    image_path = find_image(
        filename
    )

    if image_path:

        predict_single(
            image_path
        )

    else:

        print(
            f"[Warning] Image not found: "
            f"{filename}"
        )


if __name__ == "__main__":
    main()
