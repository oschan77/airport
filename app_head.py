from typing import List, Set

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template
from scipy.spatial.distance import pdist, squareform
from ultralytics import YOLO

app = Flask(__name__)

alarm = False


class Head:
    def __init__(self, bbox: List[float], depth: float = None):
        self.x1, self.y1, self.x2, self.y2 = map(int, bbox)
        self.center = (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))
        self.depth = depth

    def draw(self, image: np.ndarray) -> np.ndarray:
        image_with_bbox = cv2.rectangle(
            image, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2
        )

        return image_with_bbox

    def draw_with_score(self, image: np.ndarray, score: float) -> np.ndarray:
        image_with_score = cv2.putText(
            image,
            str(score),
            (self.x1, self.y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        image_with_bbox = cv2.rectangle(
            image_with_score, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2
        )

        return image_with_bbox


from typing import List, Set

import numpy as np
from scipy.spatial.distance import pdist, squareform


class HeadGroup:
    def __init__(
        self,
        bboxes: List[List[float]],
        image_width: int,
        thres_x: float = 0.4,
        area_ratio_min: float = 0.8,
        area_ratio_max: float = 1.2,
    ):
        self.bboxes = bboxes
        self.image_width = image_width
        self.thres_x = thres_x
        self.area_ratio_min = area_ratio_min
        self.area_ratio_max = area_ratio_max
        self.groups = self._group_heads()

    def _calculate_bbox_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _group_heads_by_distance(self, distances):
        visited = set()
        groups = []

        for i in range(len(distances)):
            if i in visited:
                continue
            group = set()
            stack = [i]
            while stack:
                idx = stack.pop()

                if idx not in visited:
                    visited.add(idx)
                    group.add(idx)

                    for j in range(len(distances)):
                        if distances[idx][j] < self.thres_x:
                            stack.append(j)

            groups.append(group)

        return groups

    def _group_heads_by_areas(self, potential_groups, areas):
        visited = set()
        groups = []
        for group in potential_groups:
            filtered_group = set()
            for idx in group:
                if idx not in visited:
                    visited.add(idx)
                    for j in group:
                        if (
                            idx != j
                            and self.area_ratio_min
                            <= areas[idx] / areas[j]
                            <= self.area_ratio_max
                        ):
                            filtered_group.add(idx)
                            filtered_group.add(j)

            if filtered_group:
                groups.append(filtered_group)

        return groups

    def _group_heads(self) -> List[Set[int]]:
        # Group by distances
        centers = [
            (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            for bbox in self.bboxes
        ]
        x_centers = [center[0] for center in centers]
        x_centers_normalized = [x / self.image_width for x in x_centers]
        distances = squareform(pdist(np.array(x_centers_normalized).reshape(-1, 1)))
        groups = self._group_heads_by_distance(distances)

        # Group by areas
        areas = [self._calculate_bbox_area(bbox) for bbox in self.bboxes]
        groups = self._group_heads_by_areas(groups, areas)

        return groups

    def draw_bounding_boxes(self, image: np.ndarray) -> np.ndarray:
        for group in self.groups:
            if len(group) > 1:  # Only draw for groups with more than one head
                x_min = min(self.bboxes[i][0] for i in group)
                y_min = min(self.bboxes[i][1] for i in group)
                x_max = max(self.bboxes[i][2] for i in group)
                y_max = max(self.bboxes[i][3] for i in group)
                cv2.rectangle(
                    image,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    image,
                    f"{len(group)}",
                    (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        return image


def setup_capture(resolution, fps):
    cap = cv2.VideoCapture(2)
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
