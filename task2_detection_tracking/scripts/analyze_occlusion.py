from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def find_candidate_frame(df: pd.DataFrame) -> tuple[int, dict[str, float]]:
    best_frame = int(df["frame"].iloc[0])
    best_score = -1.0
    best_info: dict[str, float] = {}
    for frame, g in df.groupby("frame"):
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        max_iou = 0.0
        mean_iou = 0.0
        n_pairs = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                v = iou(boxes[i], boxes[j])
                max_iou = max(max_iou, v)
                mean_iou += v
                n_pairs += 1
        mean_iou = mean_iou / n_pairs if n_pairs else 0.0
        n_objects = len(g)
        score = max_iou + 0.03 * n_objects + 0.5 * mean_iou
        if score > best_score:
            best_score = score
            best_frame = int(frame)
            best_info = {"max_iou": max_iou, "mean_pair_iou": mean_iou, "n_objects": float(n_objects), "score": score}
    return best_frame, best_info


def extract_frames(video: Path, frames: list[int], out_dir: Path) -> list[Path]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    saved: list[Path] = []
    for frame_idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 3-4 frames for occlusion / ID-switch analysis.")
    parser.add_argument("--track-csv", type=Path, required=True)
    parser.add_argument("--annotated-video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/occlusion_analysis"))
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=None, help="Manually specify first frame to extract.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.track_csv)
    if df.empty:
        raise RuntimeError("tracks.csv is empty; no detections/tracks to analyze.")
    if args.start_frame is None:
        center, info = find_candidate_frame(df)
        start = max(0, center - args.num_frames // 2)
    else:
        start = args.start_frame
        center = start
        info = {}
    frames = list(range(start, start + args.num_frames))
    saved = extract_frames(args.annotated_video, frames, args.out_dir)

    window_df = df[df["frame"].isin(frames)].copy()
    frame_stats = []
    for frame, g in window_df.groupby("frame"):
        ids = sorted(int(x) for x in g["track_id"].unique())
        frame_stats.append({"frame": int(frame), "n_tracks": len(ids), "track_ids": ids})

    notes = [
        "# 遮挡与 ID 跳变分析记录模板",
        "",
        f"- track csv: `{args.track_csv}`",
        f"- annotated video: `{args.annotated_video}`",
        f"- extracted frames: `{args.out_dir}`",
        f"- selected frames: {frames}",
        f"- automatic candidate info: `{json.dumps(info, ensure_ascii=False)}`",
        "",
        "## 连续帧统计",
        "",
    ]
    for st in frame_stats:
        notes.append(f"- frame {st['frame']}: n_tracks={st['n_tracks']}, ids={st['track_ids']}")
    notes += [
        "",
        "## 请在报告中填写的观察结论",
        "",
        "1. 观察这 3-4 帧中发生遮挡或密集交汇的目标 ID。",
        "2. 如果同一个物体在遮挡前后仍显示相同 Tracking ID，说明跟踪算法成功维持了身份。",
        "3. 如果物体短暂消失后以新 ID 出现，说明发生目标丢失或重新初始化。",
        "4. 如果两个物体交汇后 ID 互换，说明发生 ID switch。",
        "5. 原因分析可从检测框重叠、检测置信度下降、运动预测误差、ByteTrack/BoT-SORT 的关联阈值等角度说明。",
        "",
        "## 自动截取的帧",
        "",
    ]
    for p in saved:
        notes.append(f"![{p.name}]({p.name})")
    (args.out_dir / "occlusion_analysis_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Saved frames to {args.out_dir}")
    print(f"Wrote notes: {args.out_dir / 'occlusion_analysis_notes.md'}")


if __name__ == "__main__":
    main()
