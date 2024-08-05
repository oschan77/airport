import argparse
import ctypes as ct
import datetime
import enum
import os
import sys
import time
from ctypes.util import find_library
from turtle import color
from typing import List, Tuple
from unittest import result

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
import torch

# import transformers
from flask import Flask, Response, jsonify, render_template
from IPython.display import Image
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

app = Flask(__name__)

alarm = False
thermal_map = None


def plot_temperature(im, box, temperature, txt_color=(255, 255, 255)):
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)
    sf = lw / 3
    tf = max(lw - 1, 1)
    """Add one xyxy box to image with label."""
    label = "Temperature: {}".format(str(int(temperature)))
    if not isinstance(box, list):
        box = box.tolist()

    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[
        0
    ]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(
        im,
        label,
        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
        0,
        sf,
        txt_color,
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return im


def plot_util(image, h, w, box, temperature, except_case=False):
    fs = int((h + w) * 0.01)  # font size
    annotator = Annotator(image, line_width=round(fs / 6), font_size=fs * 10, pil=False)
    if temperature > 100 and not except_case:
        color = (255, 10, 10)
        global alarm
        alarm = True
        send_signal()
    else:
        color = (10, 255, 10)
    annotator.box_label(
        box, "Temperature: " + str(int(temperature)), color=color, temperature=True
    )
    return annotator.im


def send_signal():
    response = requests.get(
        "http://127.0.0.1:5000/alarm"
    )  # Replace with your Flask app's URL
    if response.status_code == 200:
        print("Signal sent successfully")
    else:
        print("Failed to send signal")


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array(
        (
            kpt[:, 0].min() - ex,
            kpt[:, 1].min() - ex,
            kpt[:, 0].max() + ex,
            kpt[:, 1].max() + ex,
        )
    )


class Person:
    def __init__(self, bbox: List[float], depth: float = None):
        self.x1, self.y1, self.x2, self.y2 = map(int, bbox)
        self.center = (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))
        self.depth = depth

    def draw(self, image: np.ndarray) -> np.ndarray:
        return cv2.rectangle(
            image, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2
        )

    def draw_with_score(self, image: np.ndarray, score: float) -> np.ndarray:
        image = cv2.putText(
            image,
            str(score),
            (self.x1, self.y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return cv2.rectangle(
            image, (self.x1, self.y1), (self.x2, self.y2), (255, 0, 0), 2
        )


def init_model():
    resolution = (1920, 1080)
    # resolution = (640, 480)
    fps = 30

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

    od_model = YOLO("best_crowd_airport.pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    while True:
        ret, color_img = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # OD
        od_preds = od_model.predict(color_img, classes=[0], conf=0.3, device="cuda")
        nCrowds = len(od_preds[0].boxes)
        od_img = cv2.putText(
            color_img,
            f"Total Number of Crowds: {nCrowds}",
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
            person = Person(bbox)
            od_img = person.draw_with_score(od_img, round(float(score), 2))

        od_img = cv2.resize(od_img, (resolution[0], resolution[1]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        ret, buffer = cv2.imencode(".jpg", od_img)
        od_img = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + od_img + b"\r\n")

    # Clear resource.
    cv2.destroyAllWindows()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(init_model(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/alarm")
def update_image():
    # Your Python code logic here
    # Check the condition and send the signal when reached
    global alarm
    if alarm:
        # Perform any necessary operations
        # ...
        return jsonify(success=True)
    else:
        return jsonify(success=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
