# SPDX-License-Identifier: MIT
"""
Inference for the blade-coating UNet: single images or whole videos.

For videos it writes an overlay video and a CSV with per-frame measurements
(area fraction of each class, tool position, coating front position) which you can use to
monitor the process (blade speed, coverage over time, coating defects).

Usage (from the repo root):
    python -m blade_coating_segmentation.inference video ^
        --checkpoint artifacts/blade_coating/best_UNetS_bladecoating.pt ^
        --preprocess datasets/blade_frames/preprocess.json ^
        --input "D:/blade_videos/run_017.mp4" --out outputs/run_017 --every 1

    python -m blade_coating_segmentation.inference image --checkpoint ... --preprocess ... --input frame.png --out outputs/frame
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from blade_coating_segmentation import CLASS_NAMES, CLASS_PALETTE
from blade_coating_segmentation.video_preprocessing import FramePreprocessor
from image_segmentation.utils.UNet import UNet, UNetS, UNetXS

ARCHITECTURES = {"UNet": UNet, "UNetS": UNetS, "UNetXS": UNetXS}


class BladeCoatingSegmenter:
    def __init__(self, checkpoint_path: str, preprocess_config: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model", ckpt)
        self.class_names: List[str] = ckpt.get("class_names", CLASS_NAMES)
        self.palette = np.array(ckpt.get("class_palette", CLASS_PALETTE), dtype=np.uint8)
        self.n_channels: int = ckpt.get("n_channels", 3)
        arch = ckpt.get("arch", "UNetS")

        self.model = ARCHITECTURES[arch](n_channels=self.n_channels, n_classes=len(self.class_names))
        self.model.load_state_dict(state)  # strict: a silent partial load would give garbage masks
        self.model.to(self.device).eval()

        # Must be identical to what was used to extract the training frames
        self.preprocessor = FramePreprocessor.load(Path(preprocess_config)) if preprocess_config else None

    def _to_tensor(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        if self.n_channels == 1:
            arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[..., None] for f in frames_bgr])
        else:
            arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
        return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0).to(self.device)

    @torch.no_grad()
    def predict(self, frames_bgr: List[np.ndarray], preprocess: bool = True) -> (List[np.ndarray], List[np.ndarray]):
        """
        frames_bgr: raw frames (as read by OpenCV). Returns (processed_frames_bgr, label_maps (H,W) uint8).
        All frames in one call must have the same size (true for frames from one video).
        """
        if preprocess and self.preprocessor is not None:
            frames_bgr = [self.preprocessor(f) for f in frames_bgr]
        x = self._to_tensor(frames_bgr)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits = self.model(x)
        labels = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
        return frames_bgr, list(labels)

    def overlay(self, frame_bgr: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        color_bgr = self.palette[labels][..., ::-1]
        out = frame_bgr.copy()
        fg = labels != self.class_names.index("background") if "background" in self.class_names else np.ones_like(labels, bool)
        out[fg] = cv2.addWeighted(frame_bgr, 1 - alpha, color_bgr, alpha, 0)[fg]
        return out

    def measurements(self, labels: np.ndarray) -> Dict[str, float]:
        """Simple per-frame process quantities derived from the mask (pixel units)."""
        h, w = labels.shape
        m: Dict[str, float] = {}
        for i, name in enumerate(self.class_names):
            m[f"area_frac_{name}"] = float(np.mean(labels == i))

        if "tool" in self.class_names:
            tool = labels == self.class_names.index("tool")
            m["tool_centroid_x"] = float(np.mean(np.nonzero(tool)[1])) if tool.any() else float("nan")
        if "coating" in self.class_names:
            coat_cols = np.nonzero((labels == self.class_names.index("coating")).any(axis=0))[0]
            # Leftmost/rightmost image column containing coating; the side that moves is the coating front.
            m["coating_x_min"] = float(coat_cols.min()) if coat_cols.size else float("nan")
            m["coating_x_max"] = float(coat_cols.max()) if coat_cols.size else float("nan")
        if "substrate" in self.class_names and "coating" in self.class_names:
            sub = np.sum(labels == self.class_names.index("substrate"))
            coat = np.sum(labels == self.class_names.index("coating"))
            # Fraction of visible substrate area already covered by coating
            m["coverage"] = float(coat / (sub + coat)) if (sub + coat) else float("nan")
        return m


def run_on_image(seg: BladeCoatingSegmenter, image_path: str, out_prefix: str):
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(image_path)
    frames, labels = seg.predict([frame])
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{out_prefix}_mask.png", seg.palette[labels[0]][..., ::-1])
    cv2.imwrite(f"{out_prefix}_overlay.png", seg.overlay(frames[0], labels[0]))
    print(seg.measurements(labels[0]))


def run_on_video(seg: BladeCoatingSegmenter, video_path: str, out_prefix: str, every: int = 1,
                 batch_size: int = 8, write_video: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    writer = None
    csv_file = open(f"{out_prefix}_measurements.csv", "w", newline="")
    csv_writer = None
    buffer, buffer_idx = [], []

    def flush():
        nonlocal writer, csv_writer
        frames, labels = seg.predict(buffer)
        for idx, f, l in zip(buffer_idx, frames, labels):
            row = {"frame_idx": idx, "time_s": round(idx / fps, 3), **seg.measurements(l)}
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                csv_writer.writeheader()
            csv_writer.writerow(row)
            if write_video:
                ov = seg.overlay(f, l)
                if writer is None:
                    writer = cv2.VideoWriter(f"{out_prefix}_overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / every, (ov.shape[1], ov.shape[0]))
                writer.write(ov)
        buffer.clear()
        buffer_idx.clear()

    idx = -1
    with tqdm(total=n_total, unit="frame", desc=Path(video_path).name) as pbar:
        while cap.grab():
            idx += 1
            pbar.update(1)
            if idx % every:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            buffer.append(frame)
            buffer_idx.append(idx)
            if len(buffer) == batch_size:
                flush()
        if buffer:
            flush()

    cap.release()
    csv_file.close()
    if writer is not None:
        writer.release()
    print(f"Wrote {out_prefix}_measurements.csv" + (f" and {out_prefix}_overlay.mp4" if write_video else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["image", "video"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--preprocess", default=None, help="preprocess.json written by video_preprocessing extract")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True, help="Output path prefix")
    p.add_argument("--every", type=int, default=1, help="Video: process every n-th frame")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-video", action="store_true", help="Video: only write the CSV")
    a = p.parse_args()

    seg = BladeCoatingSegmenter(a.checkpoint, a.preprocess)
    if a.mode == "image":
        run_on_image(seg, a.input, a.out)
    else:
        run_on_video(seg, a.input, a.out, a.every, a.batch_size, not a.no_video)


if __name__ == "__main__":
    main()
