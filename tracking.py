import numpy as np


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


class IoUTracker:

    def __init__(self, threshold=0.25, max_missing=12):
        self.threshold = threshold
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2]
        returns: list of (track_id, box)
        """

        for track in self.tracks.values():
            track["missing"] += 1

        unmatched = set(self.tracks.keys())

        for detection in detections:
            best_id = None
            best_iou = 0.0

            for track_id in unmatched:
                score = calculate_iou(
                    detection,
                    self.tracks[track_id]["box"]
                )

                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            if best_id is not None and best_iou >= self.threshold:
                self.tracks[best_id]["box"] = detection
                self.tracks[best_id]["missing"] = 0
                unmatched.remove(best_id)

            else:
                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = {
                    "box": detection,
                    "missing": 0
                }

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missing"] > self.max_missing
        ]

        for track_id in expired:
            del self.tracks[track_id]

        return [
            (track_id, track["box"])
            for track_id, track in self.tracks.items()
            if track["missing"] == 0
        ]