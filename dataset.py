# SPDX-License-Identifier: MIT
"""
Dataset for blade-coating frames annotated in CVAT and exported as "Segmentation mask 1.1".

Differences to image_segmentation.utils.segmentation_dataset.SegmentationDataset:
  * Labels are mapped by NAME (case-insensitive, with aliases) to the canonical CLASS_NAMES order,
    so several CVAT tasks whose labelmap.txt lists classes in a different order can be combined safely.
  * Masks are matched by file stem (CVAT exports images as .jpg/.png but masks always as .png).
  * Frames whose mask is entirely background are skipped by default: CVAT exports such masks for
    images nobody annotated, and training on them would teach the model "there is no substrate".
  * Every sample carries its video id (parsed from the frame name written by video_preprocessing),
    so train/val can be split per video. A random frame split leaks near-identical neighbouring
    frames into validation and makes the scores look far better than they are.
"""
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from blade_coating_segmentation import CLASS_NAMES
from image_segmentation.utils.segmentation_dataset import build_color_lut, rgb_mask_to_index

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Map whatever names were used in CVAT onto the canonical classes.
DEFAULT_LABEL_ALIASES = {
    "background": "background", "bg": "background",
    "substrate": "substrate", "glass": "substrate",
    "coating": "coating", "film": "coating", "wet film": "coating", "wet_film": "coating", "ink": "coating",
    "tool": "tool", "blade": "tool", "applicator": "tool", "doctor blade": "tool", "doctor_blade": "tool",
}


def parse_cvat_labelmap(path: Path) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Lines like 'substrate:51,221,255::' -> [('substrate', (51,221,255)), ...]."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 2 or not parts[1].strip():
                continue
            r, g, b = (int(v) for v in parts[1].split(","))
            entries.append((parts[0].strip(), (r, g, b)))
    return entries


def video_id_from_frame_name(name: str) -> str:
    """'<video_id>_f0001234.png' -> '<video_id>'. Falls back to the stem for foreign file names."""
    stem = Path(name).stem
    head, sep, tail = stem.rpartition("_f")
    return head if sep and tail.isdigit() else stem


class BladeCoatingDataset(Dataset):
    def __init__(
        self,
        export_dirs: Sequence[str],
        transform=None,
        class_names: Sequence[str] = CLASS_NAMES,
        label_aliases: Optional[Dict[str, str]] = None,
        ignore_index: int = 255,
        skip_empty_masks: bool = True,
        grayscale: bool = False,
        verbose: bool = True,
    ):
        """
        export_dirs: one or more CVAT export folders, each containing
                     JPEGImages/ (possibly with sub-folders), SegmentationClass/ and labelmap.txt.
        """
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index
        self.grayscale = grayscale
        self.transform = transform or v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        aliases = {k.lower(): v for k, v in (label_aliases or DEFAULT_LABEL_ALIASES).items()}

        self.samples: List[Tuple[Path, Path, np.ndarray]] = []  # (image, mask, lut)
        self.video_ids: List[str] = []

        for export_dir in export_dirs:
            export_dir = Path(export_dir)
            lut = self._build_lut(export_dir / "labelmap.txt", aliases)
            img_root, mask_root = export_dir / "JPEGImages", export_dir / "SegmentationClass"
            masks_by_rel_stem = {
                p.relative_to(mask_root).with_suffix("").as_posix(): p for p in mask_root.rglob("*.png")
            }
            n_added = n_missing = n_empty = 0
            for img_path in sorted(p for p in img_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
                mask_path = masks_by_rel_stem.get(img_path.relative_to(img_root).with_suffix("").as_posix())
                if mask_path is None:
                    n_missing += 1
                    continue
                if skip_empty_masks and self._is_background_only(mask_path, lut):
                    n_empty += 1
                    continue
                self.samples.append((img_path, mask_path, lut))
                self.video_ids.append(video_id_from_frame_name(img_path.name))
                n_added += 1
            if verbose:
                print(f"[BladeCoatingDataset] {export_dir.name}: {n_added} samples "
                      f"({n_missing} without mask, {n_empty} background-only skipped)")

        if not self.samples:
            raise RuntimeError("No annotated samples found. Check the export folder layout.")

    def _build_lut(self, labelmap_path: Path, aliases: Dict[str, str]) -> np.ndarray:
        color_to_idx = {}
        for name, color in parse_cvat_labelmap(labelmap_path):
            canonical = aliases.get(name.lower(), name.lower())
            if canonical not in self.class_names:
                print(f"[BladeCoatingDataset] WARNING: label '{name}' in {labelmap_path} is not mapped to "
                      f"{self.class_names}; its pixels will be ignored. Add it to label_aliases.")
                continue
            color_to_idx[color] = self.class_names.index(canonical)
        return build_color_lut(color_to_idx, ignore_index=self.ignore_index)

    def _is_background_only(self, mask_path: Path, lut: np.ndarray) -> bool:
        idx = self.load_mask_index(mask_path, lut)
        bg = self.class_names.index("background") if "background" in self.class_names else 0
        return bool(np.all((idx == bg) | (idx == self.ignore_index)))

    @staticmethod
    def load_mask_index(mask_path: Path, lut: np.ndarray) -> np.ndarray:
        return rgb_mask_to_index(np.array(Image.open(mask_path).convert("RGB")), lut)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.get(idx, self.transform)

    def get(self, idx, transform):
        img_path, mask_path, lut = self.samples[idx]
        image = Image.open(img_path).convert("L" if self.grayscale else "RGB")
        mask = tv_tensors.Mask(torch.as_tensor(self.load_mask_index(mask_path, lut), dtype=torch.int64))
        image, mask = transform(image, mask)
        return {"image": image, "mask": torch.as_tensor(mask, dtype=torch.long), "id": img_path.name}

    def class_pixel_frequencies(self, max_samples: int = 300, indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """Fraction of labelled pixels per class, estimated on (a subset of) the dataset."""
        indices = list(indices if indices is not None else range(len(self)))
        random.Random(0).shuffle(indices)
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for i in indices[:max_samples]:
            _, mask_path, lut = self.samples[i]
            m = self.load_mask_index(mask_path, lut)
            m = m[m != self.ignore_index]
            counts += np.bincount(m.ravel(), minlength=self.num_classes)[: self.num_classes]
        return counts / max(1.0, counts.sum())


def split_by_video(dataset: BladeCoatingDataset, val_percent: float = 0.15, seed: int = 0) -> Tuple[Subset, Subset]:
    """All frames of one video go to either train or val, never both."""
    groups = defaultdict(list)
    for i, vid in enumerate(dataset.video_ids):
        groups[vid].append(i)
    vids = sorted(groups)
    random.Random(seed).shuffle(vids)

    if len(vids) < 2:
        print("[split_by_video] WARNING: only one video in the dataset - falling back to a random frame split.")
        idx = list(range(len(dataset)))
        random.Random(seed).shuffle(idx)
        n_val = max(1, int(len(idx) * val_percent))
        return Subset(dataset, idx[n_val:]), Subset(dataset, idx[:n_val])

    target_val = max(1, int(round(len(dataset) * val_percent)))
    val_idx, val_vids = [], []
    for vid in vids:
        if len(val_idx) >= target_val or len(val_vids) == len(vids) - 1:
            break
        val_idx.extend(groups[vid])
        val_vids.append(vid)
    train_idx = [i for vid in vids if vid not in set(val_vids) for i in groups[vid]]
    print(f"[split_by_video] train: {len(train_idx)} frames / {len(vids) - len(val_vids)} videos, "
          f"val: {len(val_idx)} frames / {len(val_vids)} videos")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


class SubsetWithTransform(Dataset):
    """Same underlying files, different transform (train augmentations vs. plain val)."""
    def __init__(self, subset: Subset, transform):
        self.subset, self.transform = subset, transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        return self.subset.dataset.get(self.subset.indices[i], self.transform)


# ---------------------------------------------------------
# Transforms
# ---------------------------------------------------------

def get_train_transforms(crop_size: Tuple[int, int] = (512, 512), ignore_index: int = 255):
    """
    Random crops keep GPU memory low and let the network see the blade edge / film front at many
    positions. Augmentations model what changes between your recordings: lighting, slight camera
    pose changes, focus. The horizontal flip mirrors the coating direction, which is physically
    plausible (stroke left->right vs right->left).
    """
    fill = {tv_tensors.Image: 0, tv_tensors.Mask: ignore_index}
    return v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.9, 1.15),
                        interpolation=InterpolationMode.BILINEAR, fill=fill),
        v2.RandomCrop(crop_size, pad_if_needed=True, fill=fill),
        v2.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.2, hue=0.02),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_val_transforms():
    # Frames are already resized to multiples of 32 by video_preprocessing -> no resizing here.
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
