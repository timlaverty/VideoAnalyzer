"""
Phase 4: Clip Extraction
=========================
For each identified player from Phase 3, this module:

  1. Collects all frame IDs where the player appears (from tracks.csv).
  2. Groups consecutive frames into time segments.
  3. Merges segments with gaps smaller than merge_gap seconds (default 3s).
     Short gaps are bridged because a player temporarily off-screen
     (occluded, out of frame) mid-play should not split one clip into two.
  4. Pads each merged segment by padding_sec on each side (default 2s)
     to include context before/after the player appears.
  5. Extracts the clip using FFmpeg with -c copy (stream copy, no
     re-encoding). This is fast — extraction of a 5-second clip from a
     10-minute video takes under a second.

FFmpeg resolution order:
  1. imageio-ffmpeg bundled binary (installed via pip, no system install needed)
  2. System PATH fallback

Outputs:
  output/{video_stem}/clips/player_N_clip_M.mp4
  output/{video_stem}/player_clips.json
"""

import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


def _get_ffmpeg_exe():
    """Return path to ffmpeg binary. Prefers imageio-ffmpeg bundled binary."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"  # fall back to system PATH


def _check_ffmpeg():
    exe = _get_ffmpeg_exe()
    try:
        subprocess.run([exe, "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _seconds_to_hms(seconds):
    """Format seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _extract_clip_ffmpeg(video_path, out_path, start_sec, duration_sec):
    """
    Extract a clip using FFmpeg stream copy.
    Seeking is done before the -i flag (input seek) for speed.
    Returns True on success.
    """
    cmd = [
        _get_ffmpeg_exe(),
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(video_path),
        "-t", f"{duration_sec:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "1",
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        snippet = result.stderr[-400:] if result.stderr else "(no stderr)"
        print(f"[clip_extractor] ffmpeg error for {out_path.name}: {snippet}")
        return False
    return True


def _frames_to_segments(frame_ids, fps, merge_gap_sec, padding_sec, video_duration_sec):
    """
    Convert a sorted list of frame IDs to padded (start_sec, end_sec) segments.
    Gaps smaller than merge_gap_sec between consecutive frames are bridged.
    """
    if not frame_ids:
        return []

    frame_ids = sorted(set(frame_ids))
    gap_frames = merge_gap_sec * fps

    raw_segments = []
    seg_start = frame_ids[0]
    seg_end = frame_ids[0]

    for fid in frame_ids[1:]:
        if fid - seg_end <= gap_frames:
            seg_end = fid
        else:
            raw_segments.append((seg_start, seg_end))
            seg_start = fid
            seg_end = fid
    raw_segments.append((seg_start, seg_end))

    result = []
    for start_f, end_f in raw_segments:
        start_sec = max(0.0, start_f / fps - padding_sec)
        end_sec = min(video_duration_sec, end_f / fps + padding_sec)
        result.append((round(start_sec, 3), round(end_sec, 3)))

    return result


def extract_clips(video_path, tracks_csv, players_json, output_dir,
                  fps, video_duration_sec, merge_gap=3.0, padding=2.0):
    """
    Extract per-player clips from a video.

    Args:
        video_path:         Source MP4 path.
        tracks_csv:         Path to tracks.csv from Phase 1.
        players_json:       Path to players.json from Phase 3.
        output_dir:         Output directory (clips/ subdir created automatically).
        fps:                Video frame rate.
        video_duration_sec: Total video duration in seconds.
        merge_gap:          Merge clips with gaps < N seconds.
        padding:            Pad each clip by N seconds on each side.

    Returns:
        Path to player_clips.json.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if not _check_ffmpeg():
        print(
            "[clip_extractor] ERROR: ffmpeg not found in PATH.\n"
            "  Install from https://ffmpeg.org and ensure it is in your system PATH."
        )
        sys.exit(1)

    with open(players_json) as f:
        players = json.load(f)

    # Build track_id -> player_id lookup
    track_to_player = {}
    for player in players:
        for tid in player["track_ids"]:
            track_to_player[tid] = player["player_id"]

    # Aggregate frames per player from tracks CSV
    player_frames = defaultdict(list)
    with open(tracks_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            pid = track_to_player.get(tid)
            if pid:
                player_frames[pid].append(int(row["frame_id"]))

    player_clip_data = []

    for player in tqdm(players, desc="Extracting clips", unit="player"):
        pid = player["player_id"]
        frames = player_frames.get(pid, [])

        if not frames:
            continue

        segments = _frames_to_segments(frames, fps, merge_gap, padding, video_duration_sec)

        clips = []
        for clip_idx, (start_sec, end_sec) in enumerate(segments):
            duration = round(end_sec - start_sec, 3)
            if duration < 0.5:
                continue

            clip_filename = f"{pid}_clip_{clip_idx}.mp4"
            clip_path = clips_dir / clip_filename

            success = _extract_clip_ffmpeg(video_path, clip_path, start_sec, duration)

            clips.append({
                "clip_index": clip_idx,
                "clip_file": f"clips/{clip_filename}",
                "extraction_success": success,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration,
                "start_time_str": _seconds_to_hms(start_sec),
                "end_time_str": _seconds_to_hms(end_sec),
            })

        total_appearance = round(sum(c["duration_sec"] for c in clips), 2)

        player_clip_data.append({
            "player_id": pid,
            "label": player["label"],
            "jersey_number": player.get("jersey_number"),
            "team": player.get("team", "?"),
            "team_color_hex": player.get("team_color_hex", "#6B7280"),
            "total_clip_count": len(clips),
            "total_appearance_sec": total_appearance,
            "clips": clips,
        })

    # Sort: Team A first, then by label
    player_clip_data.sort(key=lambda p: (p["team"], p["label"]))

    output = {
        "video": video_path.name,
        "video_fps": round(fps, 3),
        "video_duration_sec": round(video_duration_sec, 2),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "total_players": len(player_clip_data),
        "players": player_clip_data,
    }

    clips_json_path = output_dir / "player_clips.json"
    with open(clips_json_path, "w") as f:
        json.dump(output, f, indent=2)

    total_clips = sum(p["total_clip_count"] for p in player_clip_data)
    print(f"[clip_extractor] {len(player_clip_data)} players, {total_clips} clips extracted")
    print(f"[clip_extractor] player_clips.json -> {clips_json_path}")

    return str(clips_json_path)


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="Phase 4: Extract per-player clips")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks-csv", required=True)
    parser.add_argument("--players-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-gap", type=float, default=3.0)
    parser.add_argument("--padding", type=float, default=2.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    _fps = cap.get(cv2.CAP_PROP_FPS)
    _frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    _duration = _frames / _fps if _fps > 0 else 0

    extract_clips(
        args.video, args.tracks_csv, args.players_json, args.output_dir,
        _fps, _duration, args.merge_gap, args.padding,
    )
