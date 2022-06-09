import os
import cv2
import shutil
from flask import Blueprint, request, jsonify


data_bp = Blueprint('data', __name__, url_prefix='/data')

@data_bp.route("/")
def data_index():
    return "Hello"

@data_bp.route("/video/upload", methods=["POST"])
def upload_video():
    if request.method == "POST":
        video = request.files["video"]
        user_id = request.form["user_id"]
         # timestamp = request.form["timestamp"]
        timestamp = "2012-06-04 21:00:00"

        if video.filename == "":
            return jsonify(False)

        # ra
        # if os.path.isdir(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp)):
        #     shutil.rmtree(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp))

        video_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "video")
        os.makedirs(video_path, exist_ok=True)
        os.makedirs(os.path.join(video_path, "JPEGImages"), exist_ok=True)
        os.makedirs(os.path.join(video_path, "Annotations"), exist_ok=True)

        video.save(os.path.join(video_path, video.filename))

        cap = cv2.VideoCapture(os.path.join(video_path, video.filename))
        idx = 1

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imwrite(os.path.join(video_path, "JPEGImages", f"video-frame-{idx}.png"), frame)
            cv2.imwrite(os.path.join(video_path, "Annotations", f"video-frame-{idx}.png"), frame)

            idx += 1

        return jsonify(True)
    

