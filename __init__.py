# SPDX-License-Identifier: MIT
"""
Blade-coating segmentation: video preprocessing, UNet training and inference
for separating substrate, coating (wet film) and tool (blade) in process videos.
"""

# Canonical class order used everywhere (index in this list == class index in the model output).
CLASS_NAMES = ["background", "substrate", "coating", "tool"]

# RGB colors, one row per class in CLASS_NAMES. Use the same colors when you create labels in CVAT.
CLASS_PALETTE = [
    [0, 0, 0],        # background
    [51, 221, 255],   # substrate
    [250, 50, 83],    # coating
    [36, 179, 83],    # tool
]
