from collections import namedtuple
from typing import List, Set, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)

alarm = False


BoundingBox = namedtuple("BoundingBox", ["x1", "y1", "x2", "y2"])


class Head:
    def __init__(self, bbox: List[float], depth: float = None):
        self.bbox = BoundingBox(*map(int, bbox))
        self.center = (
            (self.bbox.x1 + self.bbox.x2) // 2,
            (self.bbox.y1 + self.bbox.y2) // 2,
        )
        self.depth = depth
        self.area = self._calculate_area()

    def _calculate_area(self):
        return (self.bbox.x2 - self.bbox.x1) * (self.bbox.y2 - self.bbox.y1)

    def draw(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        return cv2.rectangle(
            image,
            (self.bbox.x1, self.bbox.y1),
            (self.bbox.x2, self.bbox.y2),
            color,
            thickness,
        )

    def draw_with_score(self, image: np.ndarray, score: float) -> np.ndarray:
        image = cv2.putText(
            image,
            f"{score:.2f}",
            (self.bbox.x1, self.bbox.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
        return self.draw(image)


def calculate_head_pair_metrics(
    head1: Head, head2: Head, image_width: int
) -> Tuple[float, float]:
    area_ratio = head1.area / head2.area
    x_distance = abs(head1.center[0] - head2.center[0]) / image_width
    return area_ratio, x_distance


class HeadGroup:
    def __init__(
        self,
        bboxes: List[List[float]],
        image_width: int,
        base_thres_x: float = 0.1,
        area_ratio_min: float = 0.6,
        area_ratio_max: float = 1.4,
        area_factor: float = 0.5,
    ):
        self.heads = [Head(bbox) for bbox in bboxes]
        self.image_width = image_width
        self.base_thres_x = base_thres_x
        self.area_ratio_min = area_ratio_min
        self.area_ratio_max = area_ratio_max
        self.area_factor = area_factor
        self.groups = self._group_heads()

    def _calculate_dynamic_threshold(self, area1: float, area2: float) -> float:
        avg_area = (area1 + area2) / 2
        max_area = max(head.area for head in self.heads)
        normalized_avg_area = 2 * (avg_area / max_area) - 1  # Range: -1 to +1

        return (
            self.base_thres_x
            + self.base_thres_x * self.area_factor * normalized_avg_area
        )

    def _group_heads(self) -> List[Set[int]]:
        groups = []
        visited = set()

        for i, head1 in enumerate(self.heads):
            if i in visited:
                continue

            group = {i}
            stack = [i]

            while stack:
                idx = stack.pop()
                if idx not in visited:
                    visited.add(idx)
                    for j, head2 in enumerate(self.heads):
                        if idx != j and j not in visited:
                            area_ratio, x_distance = calculate_head_pair_metrics(
                                self.heads[idx], head2, self.image_width
                            )
                            dynamic_threshold = self._calculate_dynamic_threshold(
                                self.heads[idx].area, head2.area
                            )

                            if (
                                x_distance < dynamic_threshold
                                and self.area_ratio_min
                                <= area_ratio
                                <= self.area_ratio_max
                            ):
                                group.add(j)
                                stack.append(j)

            if len(group) > 1:
                groups.append(group)

        return groups

    def draw_bounding_boxes(self, image: np.ndarray) -> np.ndarray:
        for group in self.groups:
            group_heads = [self.heads[i] for i in group]
            x_min = min(head.bbox.x1 for head in group_heads)
            y_min = min(head.bbox.y1 for head in group_heads)
            x_max = max(head.bbox.x2 for head in group_heads)
            y_max = max(head.bbox.y2 for head in group_heads)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(
                image,
                f"{len(group)}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        return image


def setup_capture(resolution, fps):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution set to: {actual_width}x{actual_height}")
    print(f"FPS set to: {actual_fps}")

    return cap, actual_width, actual_height


def process_frame(od_model, color_img, resolution, actual_width):
    od_preds = od_model.predict(color_img, classes=[0], conf=0.3, device="cuda")
    nHeads = len(od_preds[0].boxes)
    od_img = cv2.putText(
        color_img,
        f"Total Number of Heads: {nHeads}",
        (int(resolution[0] // 2 - 200), 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    bboxes = od_preds[0].boxes.xyxy
    scores = od_preds[0].boxes.conf
    for bbox, score in zip(bboxes, scores):
        head = Head(bbox)
        od_img = head.draw_with_score(od_img, round(float(score), 2))

    group = HeadGroup(bboxes, int(actual_width))
    od_img = group.draw_bounding_boxes(od_img)

    nGroups = len(group.groups)
    od_img = cv2.putText(
        od_img,
        f"Total Number of Crowds: {nGroups}",
        (int(resolution[0] // 2 - 200), 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    od_img = cv2.resize(od_img, (resolution[0], resolution[1]))

    return od_img


def init_model():
    resolution = (1920, 1080)
    fps = 0

    cap, actual_width, actual_height = setup_capture(resolution, fps)
    od_model = YOLO("best.pt").to("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        ret, color_img = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        od_img = process_frame(od_model, color_img, resolution, actual_width)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        ret, buffer = cv2.imencode(".jpg", od_img)
        od_img = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + od_img + b"\r\n")

    cv2.destroyAllWindows()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(init_model(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/alarm")
def update_image():
    global alarm
    return jsonify(success=alarm)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
