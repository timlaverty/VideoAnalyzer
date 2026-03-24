"""
Phase 3: Identity Resolution
==============================
Maps track_ids to named player identities using a three-pass priority system.

Priority 1 — Jersey number (most reliable)
  Tracks that share the same jersey number are merged into one player.
  Only tracks with jersey_conf > 0.6 are trusted for this merge.

Priority 2 — Appearance embedding similarity
  Remaining tracks (no reliable jersey number) are compared pairwise via
  cosine similarity on their ResNet50 embeddings. Pairs above
  similarity_threshold (default 0.85) are merged into one player.
  This handles track fragmentation from occlusions: the same physical
  player may have 2-3 track_ids in a long video; this pass merges them.

Priority 3 — Team color clustering
  KMeans (k=2) is run on the per-track median jersey hue. This assigns
  each track to "Team A" or "Team B". Within a player, team is determined
  by majority vote across all constituent tracks.

Short tracks (< MIN_TRACK_FRAMES detections) are discarded as false
positives or momentary bounding-box glitches.

Outputs:
  data/{video_stem}/players.json
    [
      {
        "player_id": "player_0",
        "label": "#23",
        "jersey_number": "23",    // null if not recognized
        "team": "A",              // "A", "B", or "?"
        "team_color_hex": "#3B82F6",
        "appearance_hue": 115.4,
        "track_ids": [1, 5, 12]   // all track_ids for this player
      },
      ...
    ]
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

MIN_TRACK_FRAMES = 15  # discard tracks shorter than this

TEAM_COLORS = {
    "A": "#3B82F6",   # blue
    "B": "#EF4444",   # red
    "?": "#6B7280",   # gray (unknown)
}


def _cosine_sim(a, b):
    """Cosine similarity of two L2-normalized vectors."""
    return float(np.dot(np.array(a), np.array(b)))


def _hue_to_hex(hue_0_180):
    """Approximate RGB hex from OpenCV HSV hue (0-180 scale)."""
    h = (hue_0_180 / 180.0) * 360.0
    if 0 <= h < 30 or 330 <= h <= 360:
        return "#C83232"   # red
    elif 30 <= h < 90:
        return "#C8C832"   # yellow
    elif 90 <= h < 150:
        return "#32C832"   # green
    elif 150 <= h < 210:
        return "#32C8A0"   # teal
    elif 210 <= h < 270:
        return "#3264C8"   # blue
    elif 270 <= h < 330:
        return "#9632C8"   # purple
    return "#808080"


def build_player_registry(reid_json_path, tracks_csv_path, output_dir,
                           similarity_threshold=0.85, min_track_frames=MIN_TRACK_FRAMES):
    """
    Build a player registry from ReID features.

    Args:
        reid_json_path:       Path to reid.json from Phase 2.
        tracks_csv_path:      Path to tracks.csv from Phase 1 (for frame counts).
        output_dir:           Where to write players.json.
        similarity_threshold: Cosine similarity cutoff for merging tracks.
        min_track_frames:     Discard tracks with fewer than this many frames.

    Returns:
        Path to players.json.
    """
    reid_json_path = Path(reid_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(reid_json_path) as f:
        reid_data = json.load(f)

    # Count frames per track
    track_frame_counts = {}
    with open(tracks_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1

    # Filter to valid tracks
    valid_tracks = {
        int(tid): data
        for tid, data in reid_data.items()
        if track_frame_counts.get(int(tid), 0) >= min_track_frames
    }
    print(f"[identity] {len(reid_data)} total tracks -> {len(valid_tracks)} pass min_frames={min_track_frames}")

    # ------------------------------------------------------------------ #
    # Pass 1: Group by jersey number                                       #
    # ------------------------------------------------------------------ #
    jersey_groups = {}    # jersey_number -> [track_id, ...]
    no_jersey_tracks = []

    for tid, data in valid_tracks.items():
        jn = data.get("jersey_number")
        if jn and data.get("jersey_conf", 0) >= 0.6:
            jersey_groups.setdefault(jn, []).append(tid)
        else:
            no_jersey_tracks.append(tid)

    print(f"[identity] Jersey groups: {len(jersey_groups)} | Unrecognized tracks: {len(no_jersey_tracks)}")

    # ------------------------------------------------------------------ #
    # Pass 2: Agglomerative clustering by appearance embedding             #
    # Distance threshold 0.40 ≈ cosine similarity 0.60                    #
    # ------------------------------------------------------------------ #
    emb_tracks = [
        (tid, valid_tracks[tid]["embedding"])
        for tid in no_jersey_tracks
        if valid_tracks[tid].get("embedding") is not None
    ]

    merged_groups = {}   # cluster_key -> [tid, ...]

    if len(emb_tracks) >= 2:
        emb_matrix = np.array([e for _, e in emb_tracks], dtype=np.float32)
        cos_sim = np.clip(emb_matrix @ emb_matrix.T, -1.0, 1.0)
        cos_dist = 1.0 - cos_sim
        np.fill_diagonal(cos_dist, 0.0)

        dist_threshold = 1.0 - similarity_threshold
        Z = linkage(squareform(cos_dist), method="average")
        cluster_labels = fcluster(Z, t=dist_threshold, criterion="distance")

        for (tid, _), label in zip(emb_tracks, cluster_labels):
            merged_groups.setdefault(int(label), []).append(tid)
    elif len(emb_tracks) == 1:
        merged_groups[0] = [emb_tracks[0][0]]

    # Tracks with no embedding each become their own group
    no_emb_key = max(merged_groups, default=-1) + 1
    for tid in no_jersey_tracks:
        if valid_tracks[tid].get("embedding") is None:
            merged_groups[no_emb_key] = [tid]
            no_emb_key += 1

    # ------------------------------------------------------------------ #
    # Pass 3: Team color clustering                                        #
    # Primary: KMeans k=2 on (team_sat, team_val) — separates colored     #
    # jerseys from white without being fooled by low-saturation hue noise. #
    # Fallback: KMeans k=2 on hue alone if sat/val not in reid.json.      #
    # ------------------------------------------------------------------ #
    team_assignments = {}
    try:
        if all(valid_tracks[tid].get("team_sat") is not None for tid in valid_tracks):
            feature_tids = list(valid_tracks.keys())
            feature_vals = [
                [valid_tracks[tid]["team_sat"], valid_tracks[tid]["team_val"]]
                for tid in feature_tids
            ]
        else:
            # Fallback for old reid.json without sat/val
            feature_tids = [tid for tid in valid_tracks
                            if valid_tracks[tid].get("team_hue", -1) >= 0]
            feature_vals = [[valid_tracks[tid]["team_hue"]] for tid in feature_tids]

        if len(feature_vals) >= 2:
            n_clusters = min(2, len(feature_vals))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(feature_vals)

            # Sort clusters so that Team A = lower mean value (likely colored/dark)
            # and Team B = higher mean value (likely white). Adjust if needed.
            cluster_vals = {}
            for tid, label in zip(feature_tids, labels):
                cluster_vals.setdefault(label, []).append(feature_vals[feature_tids.index(tid)][0])

            cluster_mean = {lbl: np.mean(vs) for lbl, vs in cluster_vals.items()}
            sorted_clusters = sorted(cluster_mean, key=lambda l: cluster_mean[l])
            cluster_to_team = {sorted_clusters[i]: chr(ord("A") + i)
                               for i in range(len(sorted_clusters))}

            for tid, label in zip(feature_tids, labels):
                team_assignments[tid] = cluster_to_team[label]
    except Exception as e:
        print(f"[identity] Team clustering failed: {e}")

    # ------------------------------------------------------------------ #
    # Build final player list                                              #
    # ------------------------------------------------------------------ #
    players = []
    player_counter = 0

    def _make_player(label, jersey_number, track_ids):
        nonlocal player_counter

        teams = [team_assignments.get(tid, "?") for tid in track_ids]
        team = Counter(teams).most_common(1)[0][0] if teams else "?"

        hues = [valid_tracks[tid]["team_hue"]
                for tid in track_ids
                if tid in valid_tracks and valid_tracks[tid].get("team_hue", -1) >= 0]
        mean_hue = round(float(np.mean(hues)), 1) if hues else 0.0

        pid = f"player_{player_counter}"
        player_counter += 1
        return {
            "player_id": pid,
            "label": label,
            "jersey_number": jersey_number,
            "team": team,
            "team_color_hex": TEAM_COLORS.get(team, TEAM_COLORS["?"]),
            "appearance_hue": mean_hue,
            "track_ids": sorted(track_ids),
        }

    # Jersey-identified players (sorted by number for stable ordering)
    for jersey_number in sorted(jersey_groups, key=lambda x: int(x) if x.isdigit() else 999):
        track_ids = jersey_groups[jersey_number]
        label = f"#{jersey_number}"
        players.append(_make_player(label, jersey_number, track_ids))

    # Embedding-merged players
    for _, track_ids in sorted(merged_groups.items()):
        # Check if any track in this group now has a jersey number from Pass 1
        # (shouldn't happen, but guard against it)
        known_jerseys = [
            valid_tracks[tid]["jersey_number"]
            for tid in track_ids
            if valid_tracks[tid].get("jersey_number")
        ]
        if known_jerseys:
            jn = Counter(known_jerseys).most_common(1)[0][0]
            label = f"#{jn}"
        else:
            label = f"Player {player_counter + 1}"
            jn = None
        players.append(_make_player(label, jn, track_ids))

    players_path = output_dir / "players.json"
    with open(players_path, "w") as f:
        json.dump(players, f, indent=2)

    team_a = sum(1 for p in players if p["team"] == "A")
    team_b = sum(1 for p in players if p["team"] == "B")
    print(f"[identity] {len(players)} players identified (Team A: {team_a}, Team B: {team_b})")
    print(f"[identity] players.json -> {players_path}")

    return str(players_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Resolve player identities")
    parser.add_argument("--reid-json", required=True)
    parser.add_argument("--tracks-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.72)
    parser.add_argument("--min-track-frames", type=int, default=MIN_TRACK_FRAMES)
    args = parser.parse_args()

    build_player_registry(
        args.reid_json,
        args.tracks_csv,
        args.output_dir,
        args.similarity_threshold,
        args.min_track_frames,
    )
