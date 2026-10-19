# SPDX-License-Identifier: MIT
"""
Turn raw blade-coating videos into a compact, de-duplicated set of training frames.

With ~100 GB of video you cannot (and should not) use every frame: consecutive frames are
nearly identical, and you can only hand-label a few hundred. This module therefore:

  1. `extract`  - samples each video at a fixed time interval, drops frames that barely changed
                  (idle substrate before/after the stroke), optionally drops blurry frames,
                  crops to a region of interest, resizes, and saves lossless PNGs + a manifest CSV.
                  Runs videos in parallel and is resumable (finished videos are skipped).
  2. `select`   - picks the N most visually diverse frames across all videos (farthest-point
                  sampling, with a per-video cap) and copies them into one folder for annotation
                  in CVAT. Also writes a `labelmap.txt` with the class colors.

The same `FramePreprocessor` is used later at inference time, so the model always sees frames
processed exactly like the training frames.

Usage (from the repo root):
    python -m blade_coating_segmentation.video_preprocessing extract ^
        --videos "D:/blade_videos" --out "datasets/blade_frames" --every-sec 0.5 --width 960 --workers 4

    python -m blade_coating_segmentation.video_preprocessing select ^
        --frames "datasets/blade_frames" --out "datasets/blade_to_annotate" --n 600 --per-video-cap 15
"""
import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from tqdm import tqdm

from blade_coating_segmentation import CLASS_NAMES, CLASS_PALETTE

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"}
PREPROCESS_CONFIG_NAME = "preprocess.json"
MANIFEST_NAME = "manifest.csv"


# ---------------------------------------------------------
# Frame preprocessing (shared by extraction and inference)
# ---------------------------------------------------------

@dataclass
class FramePreprocessor:
    """
    Deterministic per-frame preprocessing.

    roi:        (x, y, w, h) crop in ORIGINAL pixel coordinates, or None for the full frame.
                Crop away everything that is never substrate/coating/tool (machine frame, text overlays).
    width:      target width after resizing (aspect ratio kept). None keeps the cropped size.
    multiple:   output H and W are rounded to a multiple of this (UNet has 4 down-samplings -> 16;
                32 is a safe default).
    clahe:      apply CLAHE contrast equalisation on the luminance channel. Helps when the wet film
                is only faintly visible against the substrate. Must be the same for train and inference.
    """
    roi: Optional[Tuple[int, int, int, int]] = None
    width: Optional[int] = 960
    multiple: int = 32
    clahe: bool = False

    def output_size(self, frame_w: int, frame_h: int) -> Tuple[int, int]:
        """(W, H) after resizing a frame that has ALREADY been cropped to the ROI."""
        if self.width is None:
            w, h = frame_w, frame_h
        else:
            w = self.width
            h = frame_h * self.width / frame_w
        w = max(self.multiple, int(round(w / self.multiple)) * self.multiple)
        h = max(self.multiple, int(round(h / self.multiple)) * self.multiple)
        return w, h

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.roi is not None:
            x, y, w, h = self.roi
            frame_bgr = frame_bgr[y:y + h, x:x + w]

        out_w, out_h = self.output_size(frame_bgr.shape[1], frame_bgr.shape[0])
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != (out_w, out_h):
            interp = cv2.INTER_AREA if out_w < frame_bgr.shape[1] else cv2.INTER_LINEAR
            frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=interp)

        if self.clahe:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[..., 0] = clahe.apply(lab[..., 0])
            frame_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame_bgr

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FramePreprocessor":
        with open(path) as f:
            cfg = json.load(f)
        if cfg.get("roi") is not None:
            cfg["roi"] = tuple(cfg["roi"])
        return cls(**cfg)


def video_id_from_path(video_path: Path, videos_root: Path) -> str:
    """Unique, filesystem-safe id from the path relative to the video root (avoids name clashes)."""
    rel = video_path.relative_to(videos_root).with_suffix("")
    return "__".join(rel.parts).replace(" ", "_")


def _thumbnail(frame_bgr: np.ndarray, width: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h = max(1, int(round(gray.shape[0] * width / gray.shape[1])))
    return cv2.resize(gray, (width, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def _sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------
# Step 1: extraction
# ---------------------------------------------------------

def extract_frames_from_video(
    video_path: str,
    videos_root: str,
    out_root: str,
    preprocessor: FramePreprocessor,
    every_sec: float = 0.5,
    min_change: float = 2.0,
    min_sharpness: float = 0.0,
    max_frames: Optional[int] = None,
) -> Tuple[str, int, str]:
    """
    Extract frames from one video. Returns (video_id, n_saved, status).

    every_sec:     sampling interval in seconds (0.5 s -> 2 candidate frames per second).
    min_change:    mean absolute grey-level difference (0-255) to the last SAVED frame below which a
                   candidate is considered a duplicate and dropped. 0 disables de-duplication.
    min_sharpness: variance-of-Laplacian threshold to reject motion-blurred frames. 0 disables it.
                   Inspect a few values with `_sharpness` on your footage before enabling.
    """
    cv2.setNumThreads(1)  # we parallelise over videos, not inside OpenCV
    video_path, videos_root, out_root = Path(video_path), Path(videos_root), Path(out_root)
    vid = video_id_from_path(video_path, videos_root)
    out_dir = out_root / vid
    done_marker = out_dir / ".done"
    if done_marker.exists():
        n = sum(1 for _ in out_dir.glob("*.png"))
        return vid, n, "skipped (already done)"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return vid, 0, "ERROR: could not open"

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1e-3 or fps > 1000:
        fps = 30.0  # some containers report garbage; fall back to a sane default
    stride = max(1, int(round(fps * every_sec)))

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    last_thumb = None
    frame_idx = -1

    while True:
        # grab() advances without the costly retrieve/colour conversion; only retrieve sampled frames
        if not cap.grab():
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue

        frame = preprocessor(frame)

        if min_sharpness > 0 and _sharpness(frame) < min_sharpness:
            continue

        thumb = _thumbnail(frame)
        if last_thumb is not None and min_change > 0:
            if thumb.shape == last_thumb.shape and float(np.mean(np.abs(thumb - last_thumb))) < min_change:
                continue
        last_thumb = thumb

        name = f"{vid}_f{frame_idx:07d}.png"
        cv2.imwrite(str(out_dir / name), frame)
        rows.append({
            "file": f"{vid}/{name}",
            "video_id": vid,
            "video_path": str(video_path),
            "frame_idx": frame_idx,
            "time_s": round(frame_idx / fps, 3),
            "height": frame.shape[0],
            "width": frame.shape[1],
        })
        if max_frames is not None and len(rows) >= max_frames:
            break

    cap.release()

    with open(out_dir / MANIFEST_NAME, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "video_id", "video_path", "frame_idx", "time_s", "height", "width"])
        writer.writeheader()
        writer.writerows(rows)
    done_marker.touch()
    return vid, len(rows), "ok"


def find_videos(videos_root: Path) -> List[Path]:
    return sorted(p for p in videos_root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def merge_manifests(out_root: Path) -> int:
    rows = []
    for m in sorted(out_root.glob(f"*/{MANIFEST_NAME}")):
        with open(m, newline="") as f:
            rows.extend(csv.DictReader(f))
    if rows:
        with open(out_root / MANIFEST_NAME, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def extract_all(
    videos_root: str,
    out_root: str,
    preprocessor: FramePreprocessor,
    every_sec: float = 0.5,
    min_change: float = 2.0,
    min_sharpness: float = 0.0,
    max_frames_per_video: Optional[int] = None,
    workers: int = 4,
):
    videos_root, out_root = Path(videos_root), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_path = out_root / PREPROCESS_CONFIG_NAME
    if cfg_path.exists():
        existing = FramePreprocessor.load(cfg_path)
        if existing != preprocessor:
            raise ValueError(
                f"{cfg_path} exists with different settings ({existing}). Use a new --out folder "
                f"or delete it; mixing preprocessing settings in one dataset breaks training."
            )
    preprocessor.save(cfg_path)

    videos = find_videos(videos_root)
    print(f"Found {len(videos)} videos under {videos_root}")
    if not videos:
        return

    total = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(extract_frames_from_video, str(v), str(videos_root), str(out_root), preprocessor,
                        every_sec, min_change, min_sharpness, max_frames_per_video)
            for v in videos
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Videos", unit="video"):
            vid, n, status = fut.result()
            total += n
            if status != "ok":
                tqdm.write(f"[{vid}] {status}")

    n_rows = merge_manifests(out_root)
    print(f"Done. {n_rows} frames in {out_root} (manifest: {out_root / MANIFEST_NAME})")


# ---------------------------------------------------------
# Step 2: select diverse frames for annotation
# ---------------------------------------------------------

def select_for_annotation(
    frames_root: str,
    out_dir: str,
    n: int = 600,
    per_video_cap: Optional[int] = 15,
    thumb_width: int = 32,
    seed: int = 0,
):
    """
    Farthest-point sampling on small grey thumbnails: each newly picked frame is the one that is
    most different from everything picked so far. This covers different process stages
    (blade approaching, mid-stroke, film drying, empty substrate), lighting and setups with far
    fewer labels than random sampling.
    """
    frames_root, out_dir = Path(frames_root), Path(out_dir)
    manifest = frames_root / MANIFEST_NAME
    if not manifest.exists():
        merge_manifests(frames_root)
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No frames listed in {manifest}. Run `extract` first.")

    print(f"Computing thumbnails for {len(rows)} frames...")
    feats, keep_rows = [], []
    for r in tqdm(rows, unit="frame"):
        img = cv2.imread(str(frames_root / r["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        t = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (thumb_width, thumb_width // 2),
                       interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        t = (t - t.mean()) / (t.std() + 1e-6)  # brightness/contrast invariant -> diversity in *content*
        feats.append(t)
        keep_rows.append(r)
    feats = np.stack(feats)
    rows = keep_rows
    video_ids = np.array([r["video_id"] for r in rows])

    rng = np.random.default_rng(seed)
    n = min(n, len(rows))
    selected = [int(rng.integers(len(rows)))]
    per_video_count = {video_ids[selected[0]]: 1}
    min_dist = np.linalg.norm(feats - feats[selected[0]], axis=1)
    min_dist[selected[0]] = -1

    for _ in tqdm(range(n - 1), desc="Selecting", unit="frame"):
        idx = int(np.argmax(min_dist))
        if min_dist[idx] < 0:
            break  # everything is either selected or blocked by the per-video cap
        selected.append(idx)
        v = video_ids[idx]
        per_video_count[v] = per_video_count.get(v, 0) + 1
        min_dist = np.minimum(min_dist, np.linalg.norm(feats - feats[idx], axis=1))
        min_dist[selected] = -1
        if per_video_cap is not None and per_video_count[v] >= per_video_cap:
            min_dist[video_ids == v] = -1

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "selected.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for i in selected:
            shutil.copy2(frames_root / rows[i]["file"], out_dir / Path(rows[i]["file"]).name)
            writer.writerow(rows[i])

    write_cvat_labelmap(out_dir / "labelmap.txt")
    print(f"Copied {len(selected)} frames from {len(per_video_count)} videos to {out_dir}")


def write_cvat_labelmap(path: Path):
    """labelmap.txt in the format CVAT uses for 'Segmentation mask 1.1' exports."""
    with open(path, "w") as f:
        f.write("# label:color_rgb:parts:actions\n")
        for name, (r, g, b) in zip(CLASS_NAMES, CLASS_PALETTE):
            f.write(f"{name}:{r},{g},{b}::\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def _parse_roi(s: Optional[str]):
    if not s:
        return None
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    return tuple(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="Sample, de-duplicate and preprocess frames from all videos")
    e.add_argument("--videos", required=True, help="Root folder with videos (searched recursively)")
    e.add_argument("--out", required=True, help="Output folder for frames")
    e.add_argument("--every-sec", type=float, default=0.5)
    e.add_argument("--min-change", type=float, default=2.0)
    e.add_argument("--min-sharpness", type=float, default=0.0)
    e.add_argument("--max-frames-per-video", type=int, default=None)
    e.add_argument("--roi", type=_parse_roi, default=None, help="Crop x,y,w,h in original pixels")
    e.add_argument("--width", type=int, default=960, help="Resize width (0 = keep size)")
    e.add_argument("--clahe", action="store_true")
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))

    s = sub.add_parser("select", help="Pick diverse frames for annotation")
    s.add_argument("--frames", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=600)
    s.add_argument("--per-video-cap", type=int, default=15)
    s.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.cmd == "extract":
        pre = FramePreprocessor(roi=args.roi, width=args.width or None, clahe=args.clahe)
        extract_all(args.videos, args.out, pre, args.every_sec, args.min_change, args.min_sharpness,
                    args.max_frames_per_video, args.workers)
    else:
        select_for_annotation(args.frames, args.out, args.n, args.per_video_cap, seed=args.seed)


if __name__ == "__main__":
    main()
