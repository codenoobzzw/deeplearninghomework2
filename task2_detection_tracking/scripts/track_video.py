from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def side_of_line(point: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def parse_line(args_line: list[int] | None, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if args_line is None:
        return (int(width * 0.1), int(height * 0.55)), (int(width * 0.9), int(height * 0.55))
    if len(args_line) != 4:
        raise ValueError("--line expects four integers: x1 y1 x2 y2")
    x1, y1, x2, y2 = args_line
    return (x1, y1), (x2, y2)


def put_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 tracking and line-crossing counting on a video.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=str, required=True, help="Video path or camera index.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tracking_demo"))
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--line", nargs="*", type=int, default=None, help="x1 y1 x2 y2. Default is horizontal mid-lower line.")
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Optional class ids to track/count.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    p1, p2 = parse_line(args.line, width, height)
    out_video = args.out_dir / "tracked.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracks_csv = args.out_dir / "tracks.csv"
    events_csv = args.out_dir / "crossing_events.csv"
    tracks_f = tracks_csv.open("w", newline="", encoding="utf-8")
    events_f = events_csv.open("w", newline="", encoding="utf-8")
    track_writer = csv.DictWriter(
        tracks_f,
        fieldnames=[
            "frame",
            "time_sec",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy",
            "line_side",
        ],
    )
    event_writer = csv.DictWriter(events_f, fieldnames=["frame", "time_sec", "track_id", "class_id", "class_name", "direction", "count"])
    track_writer.writeheader()
    event_writer.writeheader()

    previous_side: dict[int, float] = {}
    counted_ids: set[int] = set()
    count = 0
    crossing_events: list[dict[str, Any]] = []
    history: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.max_history))

    results = model.track(
        source=args.source,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        classes=args.classes,
        verbose=False,
    )

    frame_idx = -1
    try:
        for frame_idx, result in enumerate(results):
            frame = result.orig_img.copy()
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            cv2.line(frame, p1, p2, (0, 255, 255), 3)
            put_label(frame, f"Crossing count: {count}", (30, 40), scale=0.9)
            put_label(frame, f"Line: ({p1[0]},{p1[1]})-({p2[0]},{p2[1]})", (30, 75), scale=0.55)

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                clss = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                for box, tid, cls_id, conf in zip(xyxy, ids, clss, confs):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    side = side_of_line((cx, cy), p1, p2)
                    class_name = str(model.names.get(int(cls_id), cls_id)) if isinstance(model.names, dict) else str(cls_id)
                    track_writer.writerow(
                        {
                            "frame": frame_idx,
                            "time_sec": frame_idx / fps,
                            "track_id": tid,
                            "class_id": int(cls_id),
                            "class_name": class_name,
                            "confidence": float(conf),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "cx": cx,
                            "cy": cy,
                            "line_side": side,
                        }
                    )

                    prev = previous_side.get(tid)
                    if prev is not None and tid not in counted_ids:
                        if prev == 0:
                            previous_side[tid] = side
                        elif side != 0 and np.sign(prev) != np.sign(side):
                            counted_ids.add(tid)
                            count += 1
                            direction = "A_to_B" if prev < side else "B_to_A"
                            event = {
                                "frame": frame_idx,
                                "time_sec": frame_idx / fps,
                                "track_id": int(tid),
                                "class_id": int(cls_id),
                                "class_name": class_name,
                                "direction": direction,
                                "count": count,
                            }
                            event_writer.writerow(event)
                            crossing_events.append(event)
                    previous_side[tid] = side

                    color = (
                        int(37 * (tid % 7) % 255),
                        int(53 * (tid % 11) % 255),
                        int(97 * (tid % 13) % 255),
                    )
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"ID {tid} {class_name} {conf:.2f}"
                    put_label(frame, label, (int(x1), max(20, int(y1) - 8)), scale=0.55)
                    cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
                    history[tid].append((int(cx), int(cy)))
                    pts = list(history[tid])
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i - 1], pts[i], color, 2)

            put_label(frame, f"Crossing count: {count}", (30, 40), scale=0.9)
            writer.write(frame)
    finally:
        writer.release()
        tracks_f.close()
        events_f.close()

    summary = {
        "source": args.source,
        "weights": str(args.weights),
        "out_video": str(out_video),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames_reported_by_video": total_frames,
        "frames_processed": frame_idx + 1,
        "line": {"p1": p1, "p2": p2},
        "tracker": args.tracker,
        "conf": args.conf,
        "iou": args.iou,
        "crossing_count": count,
        "counted_track_ids": sorted(map(int, counted_ids)),
        "events": crossing_events,
    }
    (args.out_dir / "tracking_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
