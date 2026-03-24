# Ultimate Frisbee Video Analyzer

Analyzes Ultimate Frisbee game footage to detect all players, track them across frames, identify each player by jersey number or appearance, separate them by team, and extract per-player video clips with timestamps. Results are displayed in a local web viewer.

---

## Pipeline

```
MP4 Video
    │
    ▼
Phase 1 — detect_track.py   YOLOv8x + ByteTrack
    │  Detects every person in every frame.
    │  Assigns a consistent track_id across frames.
    │  Output: data/{video}/tracks.csv
    │          data/{video}/crops/{tid}_{fid}.jpg
    │
    ▼
Phase 2 — reid.py            EasyOCR + ResNet50
    │  Per track, extracts:
    │    Jersey number (OCR on upper torso crop)
    │    Appearance embedding (2048-dim ResNet50, L2-normalized)
    │    Jersey HSV saturation + value (for team separation)
    │  Output: data/{video}/reid.json
    │
    ▼
Phase 3 — identity.py        Agglomerative clustering + KMeans
    │  Resolves track_ids → named players:
    │    1. Tracks with the same jersey number → same player
    │    2. Agglomerative clustering on cosine distance (avg linkage)
    │       merges fragmented tracks from occlusions
    │    3. KMeans (k=2) on jersey saturation+value → Team A / Team B
    │  Output: data/{video}/players.json
    │
    ▼
Phase 4 — clip_extractor.py  FFmpeg stream copy
    Groups per-player frames into time segments, merges short gaps,
    pads each clip, and extracts via FFmpeg (no re-encode, very fast).
    Output: output/{video}/clips/*.mp4
            output/{video}/player_clips.json
            output/{video}/index.html   (web viewer)
            output/{video}/report.txt   (human-readable player list)
```

---

## Installation

**Requirements:** Python 3.9+, CUDA optional.

```bash
pip install -r requirements.txt
```

FFmpeg is bundled via `imageio-ffmpeg` — no system install needed.
YOLOv8x weights (~130 MB) and EasyOCR models (~100 MB) download automatically on first run.

---

## Usage

```bash
# Full pipeline — process video and open web viewer
python src/pipeline.py --video "path/to/game.mp4" --serve

# CPU, skip every other frame for 2x speed
python src/pipeline.py --video "path/to/game.mp4" --skip-frames 2

# All options
python src/pipeline.py \
  --video "path/to/game.mp4" \
  --data-dir data \
  --output-dir output \
  --device cpu \
  --skip-frames 1 \
  --similarity-threshold 0.72 \
  --merge-gap 3.0 \
  --padding 2.0 \
  --serve \
  --port 8080
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Path to input MP4 |
| `--data-dir` | `data/` | Intermediate data (tracks, crops, reid) |
| `--output-dir` | `output/` | Final clips + web viewer |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--skip-frames` | `1` | Process 1 of every N frames (`2` = 2x faster) |
| `--similarity-threshold` | `0.72` | Cosine similarity floor for merging broken tracks (lower = more merging) |
| `--merge-gap` | `3.0` | Merge clip segments separated by fewer than N seconds |
| `--padding` | `2.0` | Pad each clip by N seconds on each side |
| `--serve` | off | Launch web viewer after processing |
| `--port` | `8080` | Web server port |

### Running individual phases

```bash
# Phase 1 — detection + tracking
python src/detect_track.py --video game.mp4 --output-dir data/game

# Phase 2 — ReID features
python src/reid.py --tracks-csv data/game/tracks.csv --crops-dir data/game/crops --output-dir data/game

# Phase 3 — identity resolution
python src/identity.py --reid-json data/game/reid.json --tracks-csv data/game/tracks.csv --output-dir data/game

# Phase 4 — clip extraction
python src/clip_extractor.py --video game.mp4 --tracks-csv data/game/tracks.csv \
  --players-json data/game/players.json --output-dir output/game
```

---

## Output

```
output/{video_stem}/
├── player_clips.json     # all players, clips, and timestamps
├── report.txt            # human-readable player + timestamp table
├── index.html            # web viewer
└── clips/
    ├── player_0_clip_0.mp4
    └── ...
```

**`report.txt` example:**
```
  Team A (7 players)
  #23           3 clip(s)  (41.5s on camera)
                0:00 → 0:14   (14.2s)  clips/player_1_clip_0.mp4
                0:40 → 0:47   (7.1s)   clips/player_1_clip_1.mp4
```

**`player_clips.json` schema:**
```json
{
  "video": "game.mp4",
  "video_fps": 29.97,
  "video_duration_sec": 95.5,
  "total_players": 21,
  "players": [
    {
      "player_id": "player_0",
      "label": "#23",
      "jersey_number": "23",
      "team": "A",
      "team_color_hex": "#3B82F6",
      "total_clip_count": 3,
      "total_appearance_sec": 41.5,
      "clips": [
        {
          "clip_file": "clips/player_0_clip_0.mp4",
          "start_sec": 0.0,
          "end_sec": 14.2,
          "duration_sec": 14.2,
          "start_time_str": "0:00",
          "end_time_str": "0:14"
        }
      ]
    }
  ]
}
```

---

## Model Selection

| Component | Model | Why |
|-----------|-------|-----|
| Detection | YOLOv8x (COCO, class 0) | Highest YOLOv8 accuracy for small/occluded players; swap to `yolov8m` for 3x CPU speedup |
| Tracking | ByteTrack (Ultralytics built-in) | Tracks low-confidence detections through occlusion; faster than DeepSort (no appearance network) |
| Jersey OCR | EasyOCR (digit-only) | Simple API, good accuracy on printed numbers; voted across all crops per track |
| Appearance ReID | ResNet50 (ImageNet, torchvision) | No extra download; 2048-dim L2-normalized features sufficient for same-video track merging |
| Team separation | KMeans k=2 on HSV sat+val | Saturation + value cleanly separates white jerseys from colored without hue-noise confusion |

---

## Performance

| Hardware | Video | Skip | Approx speed |
|----------|-------|------|-------------|
| CPU 8-core | 1080p | 1 | ~2–4 fps |
| CPU 8-core | 1080p | 2 | ~4–8 fps |
| GPU RTX 3080 | 4K | 1 | ~25–40 fps |

For a 10-minute 60fps video, CPU with `--skip-frames 2` takes ~1–2 hours.

---

## Known Limitations

- Jersey OCR requires the number to face the camera at reasonable distance. Expect 40–70% recognition on typical drone footage.
- A player fully occluded for > 1 second may get a new track_id. Phase 3 merges these via embedding similarity.
- Team separation assumes two visually distinct jersey colors. Two similarly colored teams may mis-cluster.
