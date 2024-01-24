import cv2
import argparse
import os
import wandb

from typing import Optional, Iterator, Tuple
from sahi_batched import Yolov8BatchedDetectionModel
from sahi_batched import get_sliced_prediction
from filevideostream import FileVideoStream
from tqdm import tqdm
from pathlib import Path

from render import annotate_frame
from detections import Detections, save_video_detections

from artifacts import download_artifact

def detect_video(video_path, model_path, max_frames=None):
    sahi_model = Yolov8BatchedDetectionModel(
        model_path=model_path,
        confidence_threshold=0.5,
        image_size=640,
        device="cuda:0",  # or 'cuda:0'
    )

    video_dir, video_name = os.path.split(video_path)
    video_dir = video_dir or os.getcwd()
    
    if not video_name:
        print("error no video name")
        return
    elif not os.path.exists(video_path):
        print(f"video path {video_path} does not exist")
        return

    # Open the video file
    cap = FileVideoStream(path=video_path, queue_size=128, transform=None).start()

    # get the number of frames
    video_frame_count = int(cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame_num = 0
    total_frame_num = (max_frames or video_frame_count)
    pbar = tqdm(total=total_frame_num)

    frame_detections = []

    # Loop through the video frames
    while cap.more() and (current_frame_num < total_frame_num):
        # Read a frame from the video
        frame = cap.read()

        if frame is not None:
            # Getting prediction using model
            object_prediction_list = get_sliced_prediction(
                frame,
                sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                perform_standard_pred=False,
                verbose=0,
            )

            detections = Detections.from_sahi_batched(object_prediction_list)

            frame_detections.append(detections)

            pbar.update(1)
            current_frame_num = current_frame_num + 1

    # Release the video capture object and close the display window
    if cap.running():
        cap.stop()

    _, detections_path = save_video_detections(frame_detections, video_path, module="detect")

    return detections_path

def detect_frame(frame_path, model_path):
    sahi_model = Yolov8BatchedDetectionModel(
        model_path=model_path,
        confidence_threshold=0.3,
        image_size=640,
        device="cuda:0",
    )

    frame_dir, frame_name = os.path.split(frame_path)
    frame_dir = frame_dir or os.getcwd()

    if not frame_name:
        print("error no frame name")
        return
    elif not os.path.exists(frame_path):
        print(f"frame path {frame_path} does not exist")
        return

    detections_name = frame_name
    detections_dir = os.path.join(frame_dir, "detect", "")
    detections_path = os.path.join(detections_dir, detections_name)

    Path(detections_dir).mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(frame_path) # image is in BGR format

    # Getting prediction using model
    object_prediction_list = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=False,
        verbose=0,
    )

    detections = Detections.from_sahi_batched(object_prediction_list)
    
    annotate_frame(frame, detections=detections)

    cv2.imwrite(detections_path, frame)

    return detections_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to the model to use', type=str, default="latest")
    parser.add_argument('--video-frames', help='Max number of video frames to detect', type=int)
    parser.add_argument('--folder', help='Folder to detect', type=str)

    parser.add_argument('--wandb', help='Log to wandb', action='store_true')
    parser.add_argument('remainder', nargs=argparse.REMAINDER, help='All other arguments')
    args = parser.parse_args()

    video_list = []
    frame_list = []

    if args.folder:
        for file in os.listdir(args.folder):
            if file.endswith(".mp4"):
                video_list.append(os.path.join(args.folder, file))
            elif file.endswith((".jpg", ".png", ".jpeg", ".tiff")):
                frame_list.append(os.path.join(args.folder, file))

    if args.remainder:
        for file in args.remainder:
            if file.endswith(".mp4"):
                video_list.append(file)
            elif file.endswith((".jpg", ".png", ".jpeg", ".tiff")):
                frame_list.append(file)

    if args.wandb:
        logged_in = wandb.login(timeout=1)
        assert(logged_in)
        #wandb.setup().settings.update(mode="online", login_timeout=None)

        run = wandb.init(project = "YOLOv8", job_type="detect", config = {})
        detections_artifact = wandb.Artifact(f"vacocam_detect", "detect")
    else:
        run = None
        detections_artifact = None

    if os.path.exists(args.model):
        model_path = args.model
    else:
        artifact, artifact_location = download_artifact("vacocam_model", args.model, run=run)
        model_path = Path(artifact_location) / "best.pt"

    for video_path in video_list:
        detections_path = detect_video(video_path, model_path, args.video_frames)
        if detections_artifact is not None and detections_path is not None:
            detections_artifact.add_file(detections_path)
    for frame_path in frame_list:
        detections_path = detect_frame(frame_path, model_path)
        if detections_artifact is not None and detections_path is not None:
            detections_artifact.add_file(detections_path)

    if run is not None:
        if detections_artifact is not None:
            run.log_artifact(detections_artifact)
        run.finish()

    print("finnyssh detect")