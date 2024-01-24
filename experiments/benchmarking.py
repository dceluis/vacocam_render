import concurrent.futures
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm

from ultralytics import YOLO
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction


from sahi_batched import get_sliced_prediction as get_sliced_prediction_new
from sahi_batched import Yolov8BatchedDetectionModel

import torch

import multiprocessing
import subprocess
import shutil
import argparse

import cv2
from imutils.video import FileVideoStream

def run_one(video_path, model_path, max_frames=None, progress=False):
    model = YOLO(model=model_path)

    video_dir, video_name = os.path.split(video_path)
    video_dir = video_dir or os.getcwd()
    
    if not video_name:
        print("error no video name")
        return
    elif not os.path.exists(video_path):
        print(f"video path {video_path} does not exist")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # get the number of frames
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    current_frame_num = 0
    total_frame_num = (max_frames or video_frame_count)
    if progress:
        pbar = tqdm(total=total_frame_num)

    # Loop through the video frames
    while cap.isOpened() and (current_frame_num < total_frame_num):
        # Read a frame from the video
        success, frame = cap.read()
        current_frame_num = current_frame_num + 1

        if success:                        
            slice_image_result = slice_image(
                image=frame,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            
            model.predict(source=slice_image_result.images, device="cuda:0", conf=0.5, verbose=False)

            if progress:
                pbar.update(1)
        else:
            # Break the loop if the end of the video is reached
            break

    cap.stop()

from render import start_ffmpeg_process_read, read_frame
from line_profiler import LineProfiler
import itertools
import select

def run_two(video_path, model_path, max_frames=None, progress=False):
    model = YOLO(model=model_path)

    video_dir, video_name = os.path.split(video_path)
    video_dir = video_dir or os.getcwd()
    
    if not video_name:
        print("error no video name")
        return
    elif not os.path.exists(video_path):
        print(f"video path {video_path} does not exist")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # get the number of frames
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()

    current_frame_num = 0
    total_frame_num = (max_frames or video_frame_count)
    
    if progress:
        pbar = tqdm(total=total_frame_num)
        
    process = start_ffmpeg_process_read(video_path)
    
    images = []

    # Loop through the video frames
    while current_frame_num < total_frame_num:
        end_of_stream = False
        # Read a frame from the video
        frame = read_frame(process, original_width, original_height)
        current_frame_num += 1

        if frame is not None:
            slice_image_result = slice_image(
                image=frame,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            
            images.append(slice_image_result.images)

            if progress:
                pbar.update(1)
        else:
            end_of_stream = True

        if len(images) > 1 or (end_of_stream and len(images) > 0):
            source = list(itertools.chain(*images))
            model.predict(source=source, device="cuda:0", conf=0.5, verbose=False)
            images = []
            
        if end_of_stream:
            logger.info('End of input stream')
            break

    while True:
        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if ready:
            data = process.stdout.read(65536)
            if not data:
                break
        elif process.poll() is not None:
            break

    process.terminate()  # Terminate the process
    process.wait()

def run_two_imtools(video_path, model_path, max_frames=None, progress=False):
    model = YOLO(model=model_path)

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
    
    original_width = cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    original_height = cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    current_frame_num = 0
    total_frame_num = (max_frames or video_frame_count)
    
    if progress:
        pbar = tqdm(total=total_frame_num)
    
    images = []

    # Loop through the video frames
    while cap.more() and (current_frame_num < total_frame_num):
        # Read a frame from the video
        frame = cap.read()
        current_frame_num += 1

        if frame is not None:
            slice_image_result = slice_image(
                image=frame,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            
            images.append(slice_image_result.images)

            if progress:
                pbar.update(1)

        if len(images) > 3 or (not cap.more() and (len(images) > 0)):
            source = list(itertools.chain(*images))
            model.predict(source=source, device="cuda:0", conf=0.5, verbose=False)
            images = []

def run_three(video_path, model_path, max_frames=None, progress=False):
    model = YOLO(model=model_path)

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
    
    original_width = cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    original_height = cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    current_frame_num = 0
    total_frame_num = (max_frames or video_frame_count)
    if progress:
        pbar = tqdm(total=total_frame_num)

    # Loop through the video frames
    while cap.more() and (current_frame_num < total_frame_num):
        frame = cap.read()
        
        if frame is not None:
            current_frame_num = current_frame_num + 1

            slice_image_result = slice_image(
                image=frame,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )

            if current_frame_num == 1: 
                print(f"type: {type(slice_image_result.images)}")

            model.predict(source=slice_image_result.images, device="cuda:0", conf=0.5, verbose=False)

            if progress:
                pbar.update(1)

    if cap.running():
        cap.stop()
        
    del frame
    del cap.Q
    del cap
    
def run_four(video_path, model_path, max_frames=None, progress=False):
    sahi_model = Yolov8DetectionModel(
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
    cap = cv2.VideoCapture(video_path)

    # get the number of frames
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    current_frame_num = 0
    total_frame_num = (max_frames or video_frame_count)
    if progress:
        pbar = tqdm(total=total_frame_num)

    # Loop through the video frames
    while cap.isOpened() and (current_frame_num < total_frame_num):
        # Read a frame from the video
        success, frame = cap.read()
        current_frame_num = current_frame_num + 1

        if success:
            # Getting prediction using model
            results = get_sliced_prediction(
                frame,
                sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                perform_standard_pred=False,
                verbose=0,
            )

            if progress:
                pbar.update(1)
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    
def run_four_batched(video_path, model_path, max_frames=None, progress=False):
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
    cap = cv2.VideoCapture(video_path)

    # get the number of frames
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    current_frame_num = 0
    total_frame_num = (max_frames or video_frame_count)
    if progress:
        pbar = tqdm(total=total_frame_num)

    # Loop through the video frames
    while cap.isOpened() and (current_frame_num < total_frame_num):
        # Read a frame from the video
        success, frame = cap.read()
        current_frame_num = current_frame_num + 1

        if success:
            # Getting prediction using model
            results = get_sliced_prediction_new(
                frame,
                sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
            )

            if progress:
                pbar.update(1)
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()

def run_five(video_path, model_path, max_frames=None, progress=False):
    sahi_model = Yolov8DetectionModel(
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
    
    original_width = cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    original_height = cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    current_frame_num = 0
    total_frame_num = (max_frames or video_frame_count)
    if progress:
        pbar = tqdm(total=total_frame_num)

    # Loop through the video frames
    while cap.more() and (current_frame_num < total_frame_num):
        # Read a frame from the video
        frame = cap.read()
        current_frame_num = current_frame_num + 1

        if frame is not None:
            # Getting prediction using model
            results = get_sliced_prediction(
                frame,
                sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                perform_standard_pred=False,
                verbose=0,
            )

            if progress:
                pbar.update(1)

    if cap.running():
        cap.stop()
        
    del frame
    del cap.Q
    del cap
    
def run_five_batched(video_path, model_path, max_frames=None, progress=False):
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
    
    original_width = cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    original_height = cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    current_frame_num = 0
    total_frame_num = (max_frames or video_frame_count)
    if progress:
        pbar = tqdm(total=total_frame_num)

    # Loop through the video frames
    while cap.more() and (current_frame_num < total_frame_num):
        # Read a frame from the video
        frame = cap.read()
        current_frame_num = current_frame_num + 1

        if frame is not None:
            # Getting prediction using model
            results = get_sliced_prediction_new(
                frame,
                sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
            )

            if progress:
                pbar.update(1)

    if cap.running():
        cap.stop()
        
    del frame
    del cap.Q
    del cap

def run(video_path, model_path, max_frames=None, progress=False, run_function_name="run_one", profile=False):
    _, video_name = os.path.split(video_path)
    run_function = globals()[run_function_name]

    if profile:
        profiler = LineProfiler()

        profiler.add_function(run_function)
        profiler.enable()  # Start profiling
    
    run_function(video_path, model_path, max_frames, progress)

    if profile:
        profiler.disable()  # Stop profiling
        profiler.dump_stats(f"profile_run_{video_name}_{run_function.__name__}.lprof")

def run_multi(video_list, model_path, max_frames=None, profile=False, progress=False, run_function_name="run_one"):
    multiprocessing.set_start_method('spawn', force=True)

    # Use Pool with 'spawn' method and apply_async
    with multiprocessing.Pool() as pool:
        results = []
        for video in video_list:
            result = pool.apply_async(run, (video, model_path, max_frames, progress, run_function_name, profile))
            results.append(result)

        # Retrieve the results
        output = [result.get() for result in results]

    print(output)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to the model to use', type=str)
    parser.add_argument('--max-frames', help='Max number of frames to detect', type=int)
    parser.add_argument('--profile', help='Profile the runtime', type=bool)
    parser.add_argument('remainder', nargs=argparse.REMAINDER, help='All other arguments')
    args = parser.parse_args()
            
    run_multi(args.remainder, args.model, args.max_frames, args.profile, progress=True)