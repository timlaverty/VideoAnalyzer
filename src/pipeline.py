"""
Ultimate Frisbee Video Analyzer — Main Pipeline
================================================
Orchestrates all four phases in sequence:

  Phase 1: detect_track.py  — YOLO detection + ByteTrack tracking
  Phase 2: reid.py          — Jersey OCR + appearance embeddings
  Phase 3: identity.py      — Resolve track_ids to player identities
  Phase 4: clip_extractor.py — Extract per-player video clips

Usage:
    python src/pipeline.py --video "path/to/game.mp4"
    python src/pipeline.py --video "path/to/game.mp4" --device cuda --serve
"""

import argparse
import http.server
import json
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clip_extractor import extract_clips
from detect_track import run_detection
from identity import build_player_registry
from reid import extract_reid_features


def _get_video_info(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total_frames / fps if fps > 0 else 0
    return fps, total_frames, duration


def print_player_report(player_clips_json, output_dir):
    """Print and save a human-readable player report with timestamps."""
    with open(player_clips_json) as f:
        data = json.load(f)

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  Player Report — {data['video']}")
    lines.append(f"  Duration: {data['video_duration_sec']:.1f}s  |  "
                 f"Players: {data['total_players']}  |  "
                 f"Processed: {data['processed_at']}")
    lines.append(f"{'='*70}")

    for team in ("A", "B", "?"):
        team_players = [p for p in data["players"] if p["team"] == team]
        if not team_players:
            continue
        team_label = f"Team {team}" if team != "?" else "Unknown Team"
        lines.append(f"\n  {team_label} ({len(team_players)} players)")
        lines.append(f"  {'-'*50}")

        for p in team_players:
            jersey = f"#{p['jersey_number']}" if p["jersey_number"] else "(no #)"
            lines.append(f"\n  {p['label']:<12}  {jersey:<8}  "
                         f"{p['total_clip_count']} clip(s)  "
                         f"({p['total_appearance_sec']:.1f}s on camera)")
            for c in p["clips"]:
                lines.append(f"              {c['start_time_str']:>7} → {c['end_time_str']:<7}  "
                             f"({c['duration_sec']:.1f}s)  {c['clip_file']}")

    lines.append(f"\n{'='*70}\n")
    report = "\n".join(lines)
    print(report)

    report_path = Path(output_dir) / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[pipeline] Report saved -> {report_path}")
    return str(report_path)


def serve_results(output_dir, port=8080):
    """Start a local HTTP server and open the web viewer in a browser."""
    output_dir = Path(output_dir).resolve()
    web_src = Path(__file__).parent.parent / "web" / "index.html"
    web_dst = output_dir / "index.html"

    if web_src.exists() and not web_dst.exists():
        shutil.copy(web_src, web_dst)

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_dir), **kwargs)

        def log_message(self, fmt, *args):
            pass  # suppress per-request logs

    server = http.server.HTTPServer(("localhost", port), _Handler)
    url = f"http://localhost:{port}/index.html"

    print(f"\n[pipeline] Web viewer: {url}")
    print("[pipeline] Press Ctrl+C to stop.\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[pipeline] Server stopped.")


def run_pipeline(args):
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[pipeline] ERROR: Video not found: {video_path}")
        sys.exit(1)

    stem = video_path.stem
    data_dir = Path(args.data_dir) / stem
    output_dir = Path(args.output_dir) / stem

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Ultimate Frisbee Video Analyzer ===")
    print(f"  Video      : {video_path}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Device     : {args.device}")
    print(f"  Skip frames: {args.skip_frames}")
    print()

    fps, total_frames, duration = _get_video_info(str(video_path))
    print(f"[pipeline] {total_frames} frames @ {fps:.2f}fps ({duration:.1f}s)\n")

    # ------------------------------------------------------------------ #
    # Phase 1: Detection + Tracking                                        #
    # ------------------------------------------------------------------ #
    print("--- Phase 1: Detection + Tracking ---")
    tracks_csv, _, _ = run_detection(
        str(video_path),
        str(data_dir),
        skip_frames=args.skip_frames,
        device=args.device,
        max_frames=getattr(args, "max_frames", 0),
        imgsz=getattr(args, "imgsz", 1280),
    )

    # ------------------------------------------------------------------ #
    # Phase 2: ReID Feature Extraction                                     #
    # ------------------------------------------------------------------ #
    print("\n--- Phase 2: ReID Feature Extraction ---")
    crops_dir = data_dir / "crops"
    reid_json = extract_reid_features(
        tracks_csv,
        str(crops_dir),
        str(data_dir),
        device=args.device,
        embedding_model_name=getattr(args, "embedding_model", "osnet"),
    )

    # ------------------------------------------------------------------ #
    # Phase 3: Identity Resolution                                         #
    # ------------------------------------------------------------------ #
    print("\n--- Phase 3: Identity Resolution ---")
    players_json = build_player_registry(
        reid_json,
        tracks_csv,
        str(data_dir),
        similarity_threshold=args.similarity_threshold,
    )

    # ------------------------------------------------------------------ #
    # Phase 4: Clip Extraction                                             #
    # ------------------------------------------------------------------ #
    print("\n--- Phase 4: Clip Extraction ---")
    player_clips_json = extract_clips(
        str(video_path),
        tracks_csv,
        players_json,
        str(output_dir),
        fps=fps,
        video_duration_sec=duration,
        merge_gap=args.merge_gap,
        padding=args.padding,
    )

    # Copy web viewer
    web_src = Path(__file__).parent.parent / "web" / "index.html"
    web_dst = output_dir / "index.html"
    if web_src.exists():
        shutil.copy(web_src, web_dst)
        print(f"[pipeline] Web viewer -> {web_dst}")

    print_player_report(player_clips_json, output_dir)

    print(f"\n=== Done ===")
    print(f"  Results      : {player_clips_json}")
    print(f"  Web viewer   : {web_dst}")
    print(f"\n  To view results:")
    print(f"    python src/pipeline.py --video \"{video_path}\" --serve")
    print(f"  Or open directly: python -m http.server 8080 --directory \"{output_dir}\"")

    if args.serve:
        serve_results(output_dir, port=args.port)

    return player_clips_json


def main():
    parser = argparse.ArgumentParser(
        description="Ultimate Frisbee Video Analyzer — full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True,
                        help="Path to input MP4 video")
    parser.add_argument("--data-dir", default="data",
                        help="Intermediate data directory")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory for clips and web viewer")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Inference device")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Process 1 out of every N frames (1 = all)")
    parser.add_argument("--similarity-threshold", type=float, default=0.72,
                        help="Cosine similarity floor for merging broken tracks (distance = 1 - threshold)")
    parser.add_argument("--merge-gap", type=float, default=3.0,
                        help="Merge clips separated by fewer than N seconds")
    parser.add_argument("--padding", type=float, default=2.0,
                        help="Pad each clip by N seconds on each side")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="YOLO inference image size (1280 recommended for 4K; 640 for faster CPU)")
    parser.add_argument("--embedding-model", default="osnet", choices=["osnet", "resnet50"],
                        help="Appearance embedding backbone")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop detection after N frames (0 = all). Use for quick smoke tests.")
    parser.add_argument("--serve", action="store_true",
                        help="Open web viewer after processing")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web server port")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
