import cv2
import os
import subprocess
import argparse
import numpy as np
from pathlib import Path

import ffmpeg
import wandb

from core.detections import load_video_detections
from utils.annotators import annotate_frame
from utils.filevideostream import FileVideoStream

from track import augment_detections, smoothen_pois, ignore_detections_from_mask

def ffmpeg_copy_audio(video_source_path, audio_source_path, output_path, remove_source=False):
    video_input = ffmpeg.input(video_source_path)
    audio_input = ffmpeg.input(audio_source_path)

    (
        ffmpeg
        .output(video_input.video, 
                audio_input.audio,
                output_path,
                c='copy',
                shortest=None,
        )
        .overwrite_output()
        .run()
    )

    if remove_source:
        os.remove(video_source_path)
        os.remove(audio_source_path)

    return (True, output_path)

def ffmpeg_concat(source_paths, output_path, remove_source=False):
    # Write the list of files to a temporary text file
    with open('filelist.txt', 'w') as filelist:
        for path in source_paths:
            filelist.write(f"file '{path}'\n")

    (
        ffmpeg
        .input('filelist.txt', format='concat', safe=0)
        .output(output_path, c='copy', shortest=None)
        .overwrite_output()
        .run()
    )

    # Clean up the temporary file
    os.remove('filelist.txt')

    if remove_source:
        for path in source_paths:
            os.remove(path)

    return (True, output_path)

def ffmpeg_read_process(in_filename):
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def ffmpeg_write_process(out_filename, width, height, framerate=30.0, preset="slow", crf=20):
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f"{width}x{height}", framerate=framerate)
        .output(out_filename, vcodec='libx264', pix_fmt='yuv420p', crf=crf, preset=preset, format='mp4')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def ffmpeg_read_frame(process, width, height):
    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame

def ffmpeg_write_frame(process, frame):
    frame_bytes = frame.astype(np.uint8).tobytes()
    process.stdin.write(frame_bytes)

def render_video(
    video_path,
    max_frames=None,
    render_height=None,
    render_width=None,
    preview=False,
    framerate=None,
    top_padding=0,
    bottom_padding=0,
    left_padding=0,
    right_padding=0,
    min_zoom=1.4,
    max_zoom=1.8,
    min_area=50,
    max_area=80,
    detections=None,
    remove_source=False,
):
    video_dir, video_name = os.path.split(video_path)
    video_dir = video_dir or os.getcwd()
    
    if not video_name:
        print(f"{video_path} does not have a valid file name")
        return (False, None)
    elif not os.path.exists(video_path):
        print(f"video path {video_path} does not exist")
        return (False, None)

    video_name_stem = Path(video_name).stem
    
    render_dir = os.path.join(video_dir, "render", "")

    intermediate_render_name = video_name_stem + "_tracked.mp4"
    intermediate_render_path = os.path.join(render_dir, intermediate_render_name)
    
    render_name = video_name_stem + "_tracked_audio.mp4"
    render_path = os.path.join(render_dir, render_name)

    Path(render_dir).mkdir(parents=True, exist_ok=True)
        
    # TODO remove
    points_new_name = Path(video_name_stem).stem + "_detections_points.npy"
    points_new_path = os.path.join(video_dir, "detect", points_new_name)
    new_points = os.path.exists(points_new_path)
    if new_points:
        loaded_points_new = np.load(points_new_path)
        points_new_inverted = loaded_points_new[:, [1, 0]]
        loaded_pois_new = smoothen_pois(points_new_inverted)

    # Open the video file
    cap = FileVideoStream(path=video_path, queue_size=128, transform=None).start()

    # get the number of frames
    video_frame_count = int(cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    video_original_width = int(cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))   # int `width`
    video_original_height = int(cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))  # int `height`
    
    total_frame_num = max_frames if max_frames is not None else video_frame_count
    
    render_width = render_width if render_width is not None else video_original_width
    render_height = render_height if render_height is not None else video_original_height
    
    info=ffmpeg.probe(video_path)

    video_duration = float(info['format']['duration'])
    
    if framerate is None:
        framerate = video_frame_count / video_duration

    if preview:
        preset, crf = ["veryfast", 26]
    else:
        preset, crf = ["slow", 20]

    render_process = ffmpeg_write_process(intermediate_render_path, render_width, render_height, framerate=framerate, preset=preset, crf=crf)

    current_frame_num = 0

    # Loop through the video frames
    while cap.more() and (current_frame_num < total_frame_num):
        # Read a frame from the video
        raw_frame = cap.read()

        if raw_frame is not None and detections is not None:
            aoi = detections[current_frame_num].area_of_interest
            aoi_clamped = max(min(aoi, max_area), min_area)
            normalized_aoi = (aoi_clamped - min_area) / (max_area - min_area)
            zoom = max_zoom - (normalized_aoi * (max_zoom - min_zoom))

            frame_width = video_original_width
            frame_height = video_original_height

            poi = detections[current_frame_num].point_of_interest
            poi = (int(poi[0]), int(poi[1]))
            
            poi_padl = int(render_width / zoom / 2)
            poi_padt = int(render_height / zoom / 2)
            poi_padr = int(render_width / zoom - poi_padl)
            poi_padb = int(render_height /zoom - poi_padt)

            poi_clamped = (
                int(
                    min(frame_width - poi_padr - right_padding, max(poi_padl + left_padding, poi[0]))
                ),
                int(
                    min(frame_height - poi_padb - bottom_padding, max(poi_padt + top_padding, poi[1]))
                )
            )

            # TODO remove
            if new_points:
                poi_new = loaded_pois_new[current_frame_num]
                poi_new = (int(poi_new[0]), int(poi_new[1]))

            tlx, tly, brx, bry = [
                int(poi_clamped[0] - poi_padl),
                int(poi_clamped[1] - poi_padt),
                int(poi_clamped[0] + poi_padr),
                int(poi_clamped[1] + poi_padb),
            ]

            if preview:
                annotate_frame(
                    raw_frame,
                    poi=poi,
                    poi_clamped=poi_clamped,
                    poi_new=poi_new if new_points else None,
                    detections=detections[current_frame_num],
                    xyxy=[tlx, tly, brx, bry],
                )

            cropped_frame = raw_frame[tly:bry, tlx:brx]

            resized_frame = cv2.resize(cropped_frame, (render_width, render_height))
            
            if preview:
                annotate_frame(
                    resized_frame,
                    stats=[f"area (smooth): {aoi:.3f}", f"area (smooth, clamped): {aoi_clamped:.3f}", f"zoom: {zoom:.3f}"]
                )

            ffmpeg_write_frame(render_process, resized_frame)

            current_frame_num += 1

    # Release the video capture object and close the display window
    if cap.running():
        cap.stop()

    if render_process.stdin is not None:
        render_process.stdin.close()
    render_process.wait()
    
    # TODO, render the final video copying the original audio track. Not in two steps like this
    ffmpeg_copy_audio(intermediate_render_path, video_path, render_path)
    os.remove(intermediate_render_path)
    
    if remove_source:
        os.remove(video_path)

    return (True, render_path)

def augment_joint_detections(video_list_detections):
    def _split_list(elements, lengths):
        split_lists = []
        start = 0
        for length in lengths:
            end = start + length
            split_lists.append(elements[start:end])
            start = end
        return split_lists
        
    lengths = [len(video_detections) for video_detections in video_list_detections]
    flat_detections = [detections for video_detections in video_list_detections for detections in video_detections]
    
    flat_detections_smooth = augment_detections(flat_detections)

    result = _split_list(flat_detections_smooth, lengths)
    
    return result

def load_ignore(video_path):
    video_dir, _ = os.path.split(video_path)
    video_dir = video_dir or os.getcwd()

    ignore_name = "ignore.png"
    ignore_dir = video_dir
    ignore_path = os.path.join(ignore_dir, ignore_name)

    if os.path.exists(ignore_path):
        return (True, ignore_path)
    else:
        print(f"ignore path {ignore_path} does not exist")
        return (False, None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-frames', help='Max number of frames to render', type=int)
    parser.add_argument('--framerate', help='Render framerate', type=int)
    parser.add_argument('--height', help='Render height', type=int)
    parser.add_argument('--width', help='Render width', type=int)

    # corner video
    # max-zoom=2.0 
    # min-zoom=1.2
    # max-area=400
    # min-area=40

    parser.add_argument('--max-zoom', help='Render max zoom', type=float, default=2.0)
    parser.add_argument('--min-zoom', help='Render min zoom', type=float, default=1.2)
    parser.add_argument('--max-area', help='Render max area', type=float, default=300)
    parser.add_argument('--min-area', help='Render min area', type=float, default=50)

    parser.add_argument('--preview', help='Preview boxes and POIs', action='store_true')
    parser.add_argument('--concat', help='Render videos to a single output', action='store_true')

    parser.add_argument('--ignore', help='Use ignore mask', action='store_true')
    parser.add_argument('--declustered', help='Use experimental static cluster cleanup', action='store_true')
    parser.add_argument('--vacocam', help='Use experimental GPT4-V unrelated cluster removal', action='store_true')

    parser.add_argument('--wandb', help='Sync to wandb', action='store_true')
    parser.add_argument('--remove-source', help='Remove source video', action='store_true')

    parser.add_argument('videos', nargs='+', default=os.getcwd())
    args = parser.parse_args()

    video_path_list = sorted(args.videos)
    
    print(f"Rendering {video_path_list}.")
    
    if args.wandb:
        logged_in = wandb.login(timeout=1)
        assert(logged_in)
        #wandb.setup().settings.update(mode="online", login_timeout=None)
        config = {}
        
        config["preview"] = args.preview

        run = wandb.init(project = "YOLOv8", job_type="render", config = config)
        render_artifact = wandb.Artifact(f"vacocam_render", "render")
    else:
        run = None
        render_artifact = None
    
    if len(video_path_list) > 1:
        video_detections = []
        for video_path in video_path_list:
            if args.declustered:
                loaded, loaded_detections = load_video_detections(video_path, module="track", version="declustered")
                print("declustered tracking")
            elif args.vacocam:
                loaded, loaded_detections = load_video_detections(video_path, module="track", version="vacocam")
                print("vacocam deluxe tracking")
            else:
                loaded, loaded_detections = load_video_detections(video_path, module="detect")
                print("simple detection tracking")

            assert loaded

            if args.ignore:
                ignore, ignore_path = load_ignore(video_path)
                assert ignore
                if ignore_path is not None:
                    loaded_detections = ignore_detections_from_mask(loaded_detections, ignore_path)
            video_detections.append(loaded_detections)

        smooth_video_detections = augment_joint_detections(video_detections)

        rendered_paths = []

        for idx, video_path in enumerate(video_path_list):
            detections = video_detections[idx]

            rendered, rendered_path = render_video(
                video_path,
                args.max_frames,
                args.height,
                args.width,
                args.preview,
                framerate=args.framerate,
                min_zoom=args.min_zoom,
                max_zoom=args.max_zoom,
                min_area=args.min_area,
                max_area=args.max_area,
                detections=smooth_video_detections[idx],
                remove_source=args.remove_source
            )

            if rendered_path is not None:
                if render_artifact is not None:
                    render_artifact.add_file(rendered_path)
                rendered_paths.append(rendered_path)
            
        if args.concat:
            pass
            # This will not work unless all videos are the same FPS. 
            # Currently we are turning the original VFR videos into a CFR thats almost equal to the original.
            # But videos can still have slightly different frame rates between each other.
            # Fix that and then concat, or even better, render all videos to a single output.
            #concat_render_path = "render.mp4"
            #ffmpeg_concat(rendered_paths, concat_render_path)
    else:
        video_path = video_path_list[0]

        if args.declustered:
            loaded, loaded_detections = load_video_detections(video_path, module="track", version="declustered")
            print("declustered tracking")
        elif args.vacocam:
            loaded, loaded_detections = load_video_detections(video_path, module="track", version="vacocam")
            print("vacocam deluxe tracking")
        else:
            loaded, loaded_detections = load_video_detections(video_path, module="detect")
            print("simple detection tracking")

        assert loaded

        if args.ignore:
            loaded, ignore_path = load_ignore(video_path)
            assert loaded
            if ignore_path is not None:
                loaded_detections = ignore_detections_from_mask(loaded_detections, ignore_path)
            
        loaded_detections = augment_detections(loaded_detections)

        video_path = video_path_list[0]

        rendered, rendered_path = render_video(
            video_path,
            args.max_frames,
            args.height,
            args.width,
            args.preview,
            framerate=args.framerate,
            min_zoom=args.min_zoom,
            max_zoom=args.max_zoom,
            min_area=args.min_area,
            max_area=args.max_area,
            detections=loaded_detections,
            remove_source=args.remove_source
        )
        if rendered_path is not None:
            if render_artifact is not None:
                render_artifact.add_file(rendered_path)

    if run is not None:
        if render_artifact is not None:
            run.log_artifact(render_artifact)
        run.finish()