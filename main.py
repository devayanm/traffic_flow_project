import argparse
import os
import cv2
import json
import time
import csv
import numpy as np
import subprocess
from ultralytics import YOLO
from tracker import CentroidTracker


def download_youtube(url, out_path):
    """Download YouTube video using yt-dlp and return the saved file path."""
    print(f"Downloading {url} with yt-dlp ...")
    os.makedirs(out_path, exist_ok=True)
    output_file = os.path.join(out_path, "traffic.mp4")
    subprocess.run(
        ["yt-dlp", "-f", "best[ext=mp4]", "-o", output_file, url], check=True
    )
    print("Downloaded to", output_file)
    return output_file


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def get_vehicle_class_indexes(model, class_names_to_keep):
    # model.names maps class index -> name
    name_to_idx = {v: k for k, v in model.model.names.items()}
    wanted = []
    for n in class_names_to_keep:
        if n in name_to_idx:
            wanted.append(name_to_idx[n])
    return wanted


def main(args):
    cfg = load_config()
    out_dir = cfg.get("output_dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    if args.download_youtube:
        input_path = download_youtube(args.youtube_url, out_path=".")
    else:
        input_path = args.input_video
    if input_path is None:
        raise ValueError("No input video provided.")

    model_name = cfg.get("model", "yolov8n.pt")
    print("Loading model", model_name)
    model = YOLO(model_name)

    target_class_names = cfg.get("classes", ["car", "truck", "bus", "motorcycle"])
    min_conf = cfg.get("min_confidence", 0.35)
    wanted_class_idxs = get_vehicle_class_indexes(model, target_class_names)
    print("Target class indexes:", wanted_class_idxs)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video: {input_path} {width}x{height} @ {fps}fps, {total_frames} frames")

    lanes = []
    if cfg.get("lanes") == "equal_vertical_3":
        w_third = width // 3
        lanes = [
            [(0, 0), (w_third, 0), (w_third, height), (0, height)],
            [(w_third, 0), (2 * w_third, 0), (2 * w_third, height), (w_third, height)],
            [(2 * w_third, 0), (width, 0), (width, height), (2 * w_third, height)],
        ]
    else:
        lanes = cfg.get("lanes", [])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = os.path.join(out_dir, "annotated.mp4")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    tracker = CentroidTracker(
        maxDisappeared=cfg.get("max_disappeared", 30),
        maxDistance=cfg.get("max_distance", 80),
    )

    csv_path = os.path.join(out_dir, "vehicle_counts.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["vehicle_id", "lane", "frames_seen", "first_frame", "first_timestamp_seconds"]
    )

    frame_idx = 0
    seen_lane_assignments = {}
    lane_counts = {1: 0, 2: 0, 3: 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = model(frame)[0]
            boxes = []
            confidences = []
            class_ids = []

            for (*xyxy, conf, cls) in results.boxes.data.tolist():
                cls = int(cls)
                if conf < min_conf:
                    continue
                if len(wanted_class_idxs) > 0 and cls not in wanted_class_idxs:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                boxes.append((x1, y1, x2, y2))
                confidences.append(float(conf))
                class_ids.append(cls)

            objects = tracker.update(boxes, frame_idx)

            overlay = frame.copy()
            for li, poly in enumerate(lanes, start=1):
                pts = np.array(poly, np.int32)
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
                cx = int(np.mean([p[0] for p in poly]))
                cy = 30
                cv2.putText(
                    overlay,
                    f"Lane {li}",
                    (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            for objectID, centroid in objects.items():
                best_box = None
                min_d = 1e9
                for b in boxes:
                    bx = int((b[0] + b[2]) / 2)
                    by = int((b[1] + b[3]) / 2)
                    d = np.hypot(centroid[0] - bx, centroid[1] - by)
                    if d < min_d:
                        min_d = d
                        best_box = b
                if best_box is None:
                    cv2.circle(
                        overlay,
                        (int(centroid[0]), int(centroid[1])),
                        4,
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        overlay,
                        f"ID {objectID}",
                        (int(centroid[0]) + 5, int(centroid[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                else:
                    x1, y1, x2, y2 = best_box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        overlay,
                        f"ID {objectID}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                assigned_lane = None
                for li, poly in enumerate(lanes, start=1):
                    pts = np.array(poly, np.int32)
                    inside = cv2.pointPolygonTest(
                        pts, (int(centroid[0]), int(centroid[1])), False
                    )
                    if inside >= 0:
                        assigned_lane = li
                        break
                if assigned_lane is None:
                    x = int(centroid[0])
                    if x < width / 3:
                        assigned_lane = 1
                    elif x < 2 * width / 3:
                        assigned_lane = 2
                    else:
                        assigned_lane = 3

                if objectID not in seen_lane_assignments:
                    seen_lane_assignments[objectID] = assigned_lane
                    lane_counts[assigned_lane] += 1
                    first_ts = frame_idx / fps
                    frames_seen = tracker.frames_seen.get(objectID, 1)
                    csv_writer.writerow(
                        [
                            objectID,
                            assigned_lane,
                            frames_seen,
                            tracker.first_frame.get(objectID, frame_idx),
                            round(first_ts, 3),
                        ]
                    )
                    csv_file.flush()

                cv2.putText(
                    overlay,
                    f"Lane {assigned_lane}",
                    (int(centroid[0]) + 5, int(centroid[1]) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            for li in range(1, 4):
                cv2.putText(
                    overlay,
                    f"Lane {li}: {lane_counts[li]}",
                    (10, 30 + 25 * (li - 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    overlay,
                    f"Lane {li}: {lane_counts[li]}",
                    (10, 30 + 25 * (li - 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            writer.write(overlay)

            if frame_idx % 50 == 0:
                print(
                    f"Frame {frame_idx}/{total_frames} processed. Lane counts: {lane_counts}"
                )

        print("Processing finished.")
        print("Lane totals:", lane_counts)

    finally:
        cap.release()
        writer.release()
        csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_youtube",
        action="store_true",
        help="Download the YouTube video first",
    )
    parser.add_argument(
        "--youtube_url", type=str, default="https://www.youtube.com/watch?v=MNn9qKG2UFI"
    )
    parser.add_argument(
        "--input_video", type=str, default=None, help="Path to local video"
    )
    args = parser.parse_args()
    main(args)
