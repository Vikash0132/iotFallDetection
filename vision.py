import cv2
import numpy as np

from tracking import IoUTracker
from alerts import AlertManager


class FallAnalyzer:

    def __init__(self, config):
        settings = config["fall_detection"]

        self.threshold = settings["horizontal_ratio"]
        self.confirmation_frames = settings["confirmation_frames"]
        self.recovery_frames = settings["recovery_frames"]

        self.states = {}

    def update(self, camera, person_id, box):
        key = (camera, person_id)

        x1, y1, x2, y2 = box

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        ratio = width / height

        state = self.states.setdefault(
            key,
            {
                "suspicious": 0,
                "normal": 0,
                "fallen": False
            }
        )

        newly_fallen = False

        if ratio >= self.threshold:
            state["suspicious"] += 1
            state["normal"] = 0

            if (
                state["suspicious"] >= self.confirmation_frames
                and not state["fallen"]
            ):
                state["fallen"] = True
                newly_fallen = True

        else:
            state["normal"] += 1
            state["suspicious"] = 0

            if state["normal"] >= self.recovery_frames:
                state["fallen"] = False

        return state["fallen"], newly_fallen


class VisionEngine:

    def __init__(self, config, model):
        self.config = config
        self.model = model

        tracking_config = config["tracking"]

        self.trackers = {}
        self.fall_analyzer = FallAnalyzer(config)
        self.alerts = AlertManager(config)

        self.process_every = max(
            1,
            config["video"]["process_every_n_frames"]
        )

        for camera in config["grid"]["cameras"]:
            name = camera["name"]

            self.trackers[name] = IoUTracker(
                threshold=tracking_config["iou_threshold"],
                max_missing=tracking_config["max_missing_frames"]
            )

        self.frame_number = 0
        self.cached_annotations = []

    def _get_camera_roi(self, frame, camera):
        height, width = frame.shape[:2]

        x1 = int(camera["x1"] * width)
        y1 = int(camera["y1"] * height)
        x2 = int(camera["x2"] * width)
        y2 = int(camera["y2"] * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        return x1, y1, x2, y2

    def process(self, frame):
        self.frame_number += 1

        if self.frame_number % self.process_every != 0:
            self._draw_cached(frame)
            return frame

        annotations = []
        model_config = self.config["model"]

        for camera in self.config["grid"]["cameras"]:
            name = camera["name"]

            rx1, ry1, rx2, ry2 = self._get_camera_roi(
                frame, camera
            )

            roi = frame[ry1:ry2, rx1:rx2]

            if roi.size == 0:
                continue

            results = self.model.predict(
                source=roi,
                conf=model_config["confidence"],
                imgsz=model_config["image_size"],
                classes=[0],
                device=model_config["device"] or None,
                verbose=False
            )

            detections = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)

                    detections.append([
                        x1 + rx1,
                        y1 + ry1,
                        x2 + rx1,
                        y2 + ry1
                    ])

            tracks = self.trackers[name].update(detections)

            camera_falls = 0

            for person_id, box in tracks:
                fallen, newly_fallen = self.fall_analyzer.update(
                    name,
                    person_id,
                    box
                )

                x1, y1, x2, y2 = map(int, box)

                if fallen:
                    camera_falls += 1
                    color = (0, 0, 255)
                    label = f"FALL - Person {person_id}"

                    if newly_fallen:
                        self.alerts.report_fall(
                            name,
                            person_id,
                            frame.copy()
                        )

                else:
                    color = (0, 255, 0)
                    label = f"Person {person_id}"

                annotations.append(
                    (x1, y1, x2, y2, color, label)
                )

            cv2.putText(
                frame,
                f"{name} | People: {len(tracks)} | Falls: {camera_falls}",
                (rx1 + 10, ry1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.rectangle(
                frame,
                (rx1, ry1),
                (rx2, ry2),
                (255, 255, 0),
                2
            )

        self.cached_annotations = annotations
        self._draw_cached(frame)

        return frame

    def _draw_cached(self, frame):
        for x1, y1, x2, y2, color, label in self.cached_annotations:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )