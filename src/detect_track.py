"""
Phase 1: Player Detection + Tracking
=====================================
Model:   YOLOv8x (ultralytics)
Tracker: ByteTrack (built into ultralytics >= 8.0)

Detection model choice:
  - YOLOv8x is the extra-large variant of YOLOv8, pretrained on COCO.
  - Class 0 ("person") is the only class used.
  - The x variant is chosen for accuracy on small/occluded players in sports
    footage. Swap to yolov8m.pt for ~3x faster CPU processing with modest
    accuracy loss.

Tracker choice:
  - ByteTrack tracks both high- and low-confidence detections, making it
    more robust to occlusion than SORT/DeepSort.
  - It assigns consistent integer track_ids across frames.
  - IDs may reset after a long full occlusion; Phase 3 merges these.

Outputs:
  data/{video_stem}/tracks.csv          — frame-level detections
  data/{video_stem}/crops/{tid}_{fid}.jpg — sampled player crops for ReID
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

YOLO_MODEL = "yolov8x.pt"
PERSON_CLASS = 0
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
CROP_INTERVAL = 8     # save a crop every N processed frames per track
MAX_CROPS_PER_TRACK = 20


def run_detection(video_path, output_dir, skip_frames=1, device="cpu", max_frames=0):
    """
    Run YOLO + ByteTrack on a video file.

    Args:
        video_path:  Path to source MP4.
        output_dir:  Base output directory (data/{video_stem}/).
        skip_frames: Process 1 out of every N frames. 1 = all frames.
        device:      "cpu" or "cuda".
        max_frames:  Stop after processing this many frames (0 = no limit). For testing.

    Returns:
        (tracks_csv_path, fps, total_frames)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"[detect_track] Loading model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap_frames = min(total_frames, max_frames) if max_frames > 0 else total_frames
    print(f"[detect_track] {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    if skip_frames > 1:
        print(f"[detect_track] Processing every {skip_frames}th frame ({cap_frames // skip_frames} frames)")
    if max_frames > 0:
        print(f"[detect_track] Capped at first {max_frames} frames (test mode)")

    records = []
    crop_counters = {}   # track_id -> frames-since-last-crop
    crop_counts = {}     # track_id -> total crops saved

    tracks_csv = output_dir / "tracks.csv"

    with tqdm(total=cap_frames, desc="Detecting + Tracking", unit="frame") as pbar:
        frame_idx = 0
        while True:
            if max_frames > 0 and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[PERSON_CLASS],
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    device=device,
                    verbose=False,
                    imgsz=640,
                )
                result = results[0]

                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()

                    for box, track_id, conf in zip(boxes, ids, confs):
                        x1, y1, x2, y2 = box.astype(int)
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        records.append({
                            "frame_id": frame_idx,
                            "track_id": int(track_id),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "conf": round(float(conf), 4),
                        })

                        # Save representative crops for ReID
                        tid = int(track_id)
                        crop_counters[tid] = crop_counters.get(tid, 0) + 1
                        crop_counts.setdefault(tid, 0)

                        w_box = x2 - x1
                        h_box = y2 - y1
                        if (crop_counters[tid] >= CROP_INTERVAL
                                and crop_counts[tid] < MAX_CROPS_PER_TRACK
                                and w_box > 20 and h_box > 40):
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                crop_path = crops_dir / f"{tid}_{frame_idx}.jpg"
                                cv2.imwrite(str(crop_path), crop)
                                crop_counts[tid] += 1
                                crop_counters[tid] = 0

            frame_idx += 1
            pbar.update(1)

    cap.release()

    with open(tracks_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame_id", "track_id", "x1", "y1", "x2", "y2", "conf"]
        )
        writer.writeheader()
        writer.writerows(records)

    unique_tracks = len(set(r["track_id"] for r in records))
    print(f"[detect_track] {len(records)} detections, {unique_tracks} unique tracks")
    print(f"[detect_track] Crops saved: {sum(crop_counts.values())} across {len(crop_counts)} tracks")
    print(f"[detect_track] tracks.csv -> {tracks_csv}")

    return str(tracks_csv), fps, total_frames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Detect and track players")
    parser.add_argument("--video", required=True, help="Path to input MP4")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    run_detection(args.video, args.output_dir, args.skip_frames, args.device)
