"""
Phase 2: Re-Identification (ReID) Feature Extraction
======================================================
For each tracked player (track_id), extracts three types of features
from the sampled crop images produced by Phase 1.

Jersey Number OCR
  Model:   EasyOCR (English, digit-only mode)
  Region:  Upper 50% of each player crop (where jersey numbers appear)
  Voting:  Most frequently detected number across all crops wins.
  Fallback: If EasyOCR is not installed, jersey recognition is skipped
            and identity resolution relies on embedding similarity only.

Appearance Embeddings
  Model:   torchvision ResNet50 (pretrained on ImageNet, IMAGENET1K_V2)
  Why:     Always available with PyTorch. The final FC layer is replaced
           with nn.Identity() to expose the 2048-dim feature vector.
           L2-normalized for cosine similarity comparison.
  Per-track: Mean-pooled across all valid crops → stable representation
             even if some crops are blurred or partially occluded.

Team Color (Hue)
  Method:  Median HSV hue from the jersey region (15-65% height, 10-90% width).
           Near-black and near-white pixels (background/skin) are masked out.
  Use:     KMeans(k=2) in Phase 3 separates Team A from Team B.

Outputs:
  data/{video_stem}/reid.json
    {
      "1": {
        "jersey_number": "23",     // null if not detected
        "jersey_conf": 0.82,
        "embedding": [...],        // 2048 floats, L2-normalized
        "team_hue": 115.4,         // median HSV hue, 0-180 (OpenCV scale)
        "crop_count": 12
      },
      ...
    }
"""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[reid] WARNING: easyocr not installed. Jersey number recognition disabled.")

MIN_CROP_W = 25
MIN_CROP_H = 50


def _build_embedding_model(device):
    """ResNet50 with FC replaced by Identity → 2048-dim embedding output."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet = models.resnet50(weights=weights)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    return resnet.to(device)


def _extract_jersey_number(ocr_reader, crop_bgr):
    """
    Run digit OCR on the upper torso region of a player crop.
    Returns (number_str, confidence) or (None, 0.0).
    """
    h, w = crop_bgr.shape[:2]
    torso = crop_bgr[int(h * 0.1): int(h * 0.6), :]
    if torso.size == 0:
        return None, 0.0

    try:
        results = ocr_reader.readtext(
            torso,
            allowlist="0123456789",
            min_size=6,
            text_threshold=0.6,
            low_text=0.3,
        )
    except Exception:
        return None, 0.0

    if not results:
        return None, 0.0

    best = max(results, key=lambda r: r[2])
    text = best[1].strip()
    conf = float(best[2])
    if text and conf > 0.5:
        return text, conf
    return None, 0.0


def _extract_team_hue(crop_bgr):
    """
    Compute the median HSV hue of the jersey region.
    Returns a float in [0, 180] (OpenCV HSV scale).
    """
    h, w = crop_bgr.shape[:2]
    jersey = crop_bgr[int(h * 0.15): int(h * 0.65), int(w * 0.10): int(w * 0.90)]
    if jersey.size == 0:
        return 0.0

    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    # Exclude near-black (dark shadows) and near-white (skin/sky)
    mask = cv2.inRange(hsv, (0, 40, 50), (180, 255, 255))
    hues = hsv[:, :, 0][mask > 0]
    if len(hues) == 0:
        return 0.0
    return float(np.median(hues))


def _extract_jersey_sv(crop_bgr):
    """
    Compute mean HSV saturation and value of the jersey region (all pixels,
    no hue masking). Saturation distinguishes colored jerseys from white;
    value distinguishes white from dark. Returns (saturation, value) floats
    in [0, 255] (OpenCV scale).
    """
    h, w = crop_bgr.shape[:2]
    jersey = crop_bgr[int(h * 0.15): int(h * 0.65), int(w * 0.10): int(w * 0.90)]
    if jersey.size == 0:
        return 0.0, 0.0

    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]))
    val = float(np.mean(hsv[:, :, 2]))
    return sat, val


def extract_reid_features(tracks_csv, crops_dir, output_dir, device="cpu"):
    """
    Extract jersey number, appearance embedding, and team color per track.

    Args:
        tracks_csv:  Path to tracks.csv (unused here but kept for API consistency).
        crops_dir:   Directory of player crop images ({track_id}_{frame_id}.jpg).
        output_dir:  Where to write reid.json.
        device:      "cpu" or "cuda".

    Returns:
        Path to reid.json.
    """
    crops_dir = Path(crops_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reid_path = output_dir / "reid.json"

    # Group crops by track_id
    crop_files = sorted(crops_dir.glob("*.jpg"))
    crops_by_track = {}
    for f in crop_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                tid = int(parts[0])
                crops_by_track.setdefault(tid, []).append(f)
            except ValueError:
                continue

    if not crops_by_track:
        print("[reid] No crops found. Skipping ReID.")
        reid_path.write_text("{}")
        return str(reid_path)

    print(f"[reid] {len(crops_by_track)} tracks, {len(crop_files)} total crops")

    # Build models
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    embedding_model = _build_embedding_model(torch_device)
    ocr_reader = easyocr.Reader(["en"], verbose=False) if EASYOCR_AVAILABLE else None

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    reid_data = {}

    for tid, crop_paths in tqdm(crops_by_track.items(), desc="ReID extraction", unit="track"):
        jersey_votes = []
        jersey_confs = []
        embeddings = []
        hues = []
        sats = []
        vals = []

        for crop_path in sorted(crop_paths):
            img_bgr = cv2.imread(str(crop_path))
            if img_bgr is None:
                continue
            ch, cw = img_bgr.shape[:2]
            if cw < MIN_CROP_W or ch < MIN_CROP_H:
                continue

            # --- OCR ---
            if ocr_reader is not None:
                num, conf = _extract_jersey_number(ocr_reader, img_bgr)
                if num:
                    jersey_votes.append(num)
                    jersey_confs.append(conf)

            # --- Appearance embedding ---
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor = transform(pil_img).unsqueeze(0).to(torch_device)
            with torch.no_grad():
                emb = embedding_model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-9:
                emb = emb / norm
            embeddings.append(emb)

            # --- Team hue + saturation + value ---
            hues.append(_extract_team_hue(img_bgr))
            sat, val = _extract_jersey_sv(img_bgr)
            sats.append(sat)
            vals.append(val)

        # --- Aggregate per track ---
        jersey_number = None
        jersey_conf = 0.0
        if jersey_votes:
            counts = Counter(jersey_votes)
            jersey_number = counts.most_common(1)[0][0]
            winning_confs = [c for n, c in zip(jersey_votes, jersey_confs) if n == jersey_number]
            jersey_conf = round(float(np.mean(winning_confs)), 3)

        mean_embedding = None
        if embeddings:
            mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-9:
                mean_emb = mean_emb / norm
            mean_embedding = mean_emb.tolist()

        mean_hue = round(float(np.mean(hues)), 2) if hues else 0.0
        mean_sat = round(float(np.mean(sats)), 2) if sats else 0.0
        mean_val = round(float(np.mean(vals)), 2) if vals else 0.0

        reid_data[str(tid)] = {
            "jersey_number": jersey_number,
            "jersey_conf": jersey_conf,
            "embedding": mean_embedding,
            "team_hue": mean_hue,
            "team_sat": mean_sat,
            "team_val": mean_val,
            "crop_count": len(embeddings),
        }

    with open(reid_path, "w") as f:
        json.dump(reid_data, f, indent=2)

    recognized = sum(1 for v in reid_data.values() if v["jersey_number"])
    print(f"[reid] Jersey numbers recognized: {recognized}/{len(reid_data)} tracks")
    print(f"[reid] reid.json -> {reid_path}")

    return str(reid_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Extract ReID features")
    parser.add_argument("--tracks-csv", required=True)
    parser.add_argument("--crops-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    extract_reid_features(args.tracks_csv, args.crops_dir, args.output_dir, args.device)
