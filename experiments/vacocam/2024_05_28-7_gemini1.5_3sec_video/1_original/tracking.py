import numpy as np

import supervision as sv
import os
import cv2
import io

import base64
import json
from gpt4 import submit_image as submit_image_to_gpt4

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from typing import List, Tuple, Union, Optional
from PIL import Image

from detections import Detections

class VideoSection:
    def __init__(self, video_info, start_frame, end_frame, detections, sample=None):
        self.video_info = video_info
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.detections = detections
        self.sample = sample

def cluster_detections(detections_list: List[Detections], image_height=1, image_width=1, preset=None):
    X_Y = []
    W_H = []
    FRAME = []

    DATA_FLAT: List[Detections] = []
    detections_lengths = [0] * len(detections_list)

    for idx, frame_detections in enumerate(detections_list):
        for xyxy, _, conf, cls, *_ in frame_detections:
            det_xyxy = np.zeros((1, 4))
            det_xyxy[0] = xyxy
            det_conf = np.array([conf])
            det_cls = np.array([cls])

            x = (xyxy[0] + xyxy[2]) / 2
            y = (xyxy[1] + xyxy[3]) / 2

            X_Y.append((x / image_width, y / image_height))
            W_H.append(((xyxy[2] - xyxy[0]) / image_width, (xyxy[3] - xyxy[1]) / image_height))
            FRAME.append(idx)

            DATA_FLAT.append(Detections(det_xyxy, None, det_conf, det_cls))

        detections_lengths[idx] = len(frame_detections)

    X_Y = np.array(X_Y)
    W_H = np.array(W_H)
    FRAME = np.array(FRAME)

    if len(DATA_FLAT) == 0:
        return {-1: detections_list}
    elif len(DATA_FLAT) == 1:
        return {-1: detections_list}

    spatial_distance_factor = 1.0
    temporal_distance_factor = 1.4

    if preset == 'static':
        eps = 0.02
        min_samples = 300
    elif preset == 'play':
        # With the spatial_clustering_distance_factor, and temporal_clustering_distance_factor in place, the eps and min_samples values are not as important
        eps = 0.25
        min_samples = 5
    else:
        eps = 0.5
        min_samples = 50

    X_Y_scaled = X_Y * spatial_distance_factor
    W_H_scaled = W_H * spatial_distance_factor
    FRAME_scaled = MinMaxScaler().fit_transform(FRAME.reshape(-1, 1)) * temporal_distance_factor

    # concatenate the scaled x, y and area values
    DATA = np.hstack((X_Y_scaled, W_H_scaled, FRAME_scaled))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    X_clusters = dbscan.fit_predict(DATA)

    # if preset == 'play':
    #     min_samples_real = 15
    #     # count the number of detections in each cluster
    #     cluster_counts = np.bincount(X_clusters + 1)
    #     # get the indices of the clusters with less than min_sampbles_real detections
    #     small_clusters = np.where(cluster_counts < min_samples_real)[0]

    #     # set the labels of the small clusters to -1 (noise)
    #     for cluster in small_clusters:
    #         X_clusters[X_clusters == cluster - 1] = -1

    unique_labels = list(map(int, set(X_clusters)))

    # Initialize lists to hold clustered detections for each frame
    clustered_detections = {cluster_id: [Detections.empty()] * len(detections_list) for cluster_id in unique_labels}

    processed_detections_idx = 0
    # Assign each detections object to the appropriate cluster and frame
    for frame_index, length in enumerate(detections_lengths):
        clusters = X_clusters[processed_detections_idx:processed_detections_idx + length]
        data = DATA_FLAT[processed_detections_idx:processed_detections_idx + length]
        
        temp_clustered_detections = {cluster_id: [] for cluster_id in unique_labels}

        for idx in range(length):
            cluster_id = clusters[idx]
            detection = data[idx]
            temp_clustered_detections[cluster_id].append(detection)
        
        for cluster_id, cluster_detections in temp_clustered_detections.items():
            frame_detections = Detections.merge(cluster_detections)
            clustered_detections[cluster_id][frame_index] = frame_detections

        processed_detections_idx += length

    return clustered_detections

colors = [
    {"name":"Blue","hex":"0015ff","rgb":[0,21,255],"cmyk":[100,92,0,0],"hsb":[235,100,100],"hsl":[235,100,50],"lab":[33,76,-106]},
    {"name":"Pumpkin","hex":"ff7300","rgb":[255,115,0],"cmyk":[0,55,100,0],"hsb":[27,100,100],"hsl":[27,100,50],"lab":[65,49,73]},
    {"name":"Chartreuse","hex":"90fe00","rgb":[144,254,0],"cmyk":[43,0,100,0],"hsb":[86,100,100],"hsl":[86,100,50],"lab":[90,-63,86]},
    {"name":"Violet","hex":"8400ff","rgb":[132,0,255],"cmyk":[48,100,0,0],"hsb":[271,100,100],"hsl":[271,100,50],"lab":[41,83,-92]},
    {"name":"Red","hex":"ff0a12","rgb":[255,10,18],"cmyk":[0,96,93,0],"hsb":[358,96,100],"hsl":[358,100,52],"lab":[54,80,63]},
    {"name":"Yellow","hex":"fffb00","rgb":[255,251,0],"cmyk":[0,2,100,0],"hsb":[59,100,100],"hsl":[59,100,50],"lab":[96,-20,94]},
    {"name":"Fluorescent cyan","hex":"00fff7","rgb":[0,255,247],"cmyk":[100,0,3,0],"hsb":[178,100,100],"hsl":[178,100,50],"lab":[91,-50,-10]},
    {"name":"Persian rose","hex":"ff00a1","rgb":[255,0,161],"cmyk":[0,100,37,0],"hsb":[322,100,100],"hsl":[322,100,50],"lab":[56,87,-14]},
]

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def ask_gippity_for_primary_clusters(image_bytes: Union[bytes, list[bytes]], metadata: str, version="v3"):
    """
    Returns a list of tuples of (box_label (str), is_primary (bool)), from asking GPT-4 to select the primary clusters in the image.
    Or None if GPT-4 could not find any clusters.
    """
    if type(image_bytes) == list:
        encoded_images = [encode_image(image) for image in image_bytes]
    else:
        encoded_images = [encode_image(image_bytes)]

    response_json = submit_image_to_gpt4(encoded_images, metadata, version=version)

    return response_json

def save_gpt4_response(artifact_id, response_json):
    output_dir = os.path.join(os.path.dirname(__file__), "./track", "gpt4")

    gpt4_response_path = os.path.join(output_dir, f"{artifact_id}.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(gpt4_response_path, 'w') as f:
        json.dump(response_json, f)

    return gpt4_response_path

def load_gpt4_response(artifact_id):
    output_dir = os.path.join(os.path.dirname(__file__), "./track", "gpt4")

    gpt4_response_path = os.path.join(output_dir, f"{artifact_id}.json")

    if os.path.exists(gpt4_response_path):
        with open(gpt4_response_path, 'r') as f:
            response_json = json.load(f)

        return response_json
    else:
        return None

def parse_gemini_response(response_json):
    try:
        # response_content = response_json["candidates"][0]["content"]["parts"][0]["text"]
        response_content = response_json

        if type(response_content) != list:
            return None
        else:
            result = {item["id"]: item["primary"] for item in response_content}

            print("Parsed GPT-4 response")
            print(str(result))

            return result
    except Exception as e:
        print(f"\n==== Error parsing Gemini response. ==== \n")
        print(e)
        print("--------------------------------")
        print(response_json)
        print("==== End of Gemini response. ====")
        print("\n")

        return None

def parse_claude_response(response_json):
    try:
        response_content = response_json["content"][0]["text"]

        start = response_content.find("[")
        end = response_content.find("]")

        if start == -1 or end == -1:
            return None
        else:
            json_content = response_content[start:end+1]

            response_dict = json.loads(json_content)

            result = {item["id"]: item["primary"] for item in response_dict}

            print("Parsed GPT-4 response")
            print(str(result))

            return result
    except Exception as e:
        print(f"\n==== Error parsing GPT-4 response. ==== \n")
        print(e)
        print("--------------------------------")
        print(response_json)
        print("==== End of GPT-4 response. ====")
        print("\n")

        return None

def parse_gpt4_response(response_json):
    try:
        response_content = response_json["choices"][0]["message"]["content"]

        start = response_content.find("[")
        end = response_content.find("]")

        if start == -1 or end == -1:
            return None
        else:
            json_content = response_content[start:end+1]

            response_dict = json.loads(json_content)

            result = {item["id"]: item["primary"] for item in response_dict}

            print("Parsed GPT-4 response")
            print(str(result))

            return result
    except Exception as e:
        print(f"\n==== Error parsing GPT-4 response. ==== \n")
        print(e)
        print("--------------------------------")
        print(response_json)
        print("==== End of GPT-4 response. ====")
        print("\n")

        return None

def load_section_presentation(artifact_id) -> Union[Tuple[Union[bytes, list[bytes]], str], Tuple[None, None]]:
    output_dir = os.path.join(os.path.dirname(__file__), "./track", "presentation")

    presentation_path = os.path.join(output_dir, artifact_id)

    presentation_metadata_path = f"{presentation_path}.txt"

    presentation_image_paths = [os.path.join(output_dir, f) for f
                                 in os.listdir(output_dir)
                                   if f.startswith(artifact_id) and f.endswith(".png")]

    if len(presentation_image_paths) > 1 and os.path.exists(presentation_metadata_path):
        image_bytes = [open(presentation_image_path, 'rb').read() for presentation_image_path in presentation_image_paths]
        metadata = open(presentation_metadata_path, 'r').read()

        return image_bytes, metadata
    elif len(presentation_image_paths) == 1 and os.path.exists(presentation_metadata_path):
        image_bytes = open(presentation_image_paths[0], 'rb').read()
        metadata = open(presentation_metadata_path, 'r').read()

        return image_bytes, metadata
    else:
        print(f"Could not find presentation at {presentation_path}")
        return None, None

def save_section_presentation(artifact_id, image_bytes: Union[bytes, list[bytes]], metadata: str) -> Tuple[Union[str, list[str]], str]:
    output_dir = os.path.join(os.path.dirname(__file__), "./track", "presentation")

    presentation_path = os.path.join(output_dir, artifact_id)
    presentations_dir = os.path.dirname(presentation_path)

    if not os.path.exists(presentations_dir):
        os.makedirs(presentations_dir)

    presentation_metadata_path = f"{presentation_path}.txt"
    with open(presentation_metadata_path, 'w') as f:
        f.write(metadata)

    if isinstance(image_bytes, list):
        presentation_image_paths = []
        for idx, image in enumerate(image_bytes):
            presentation_image_path = f"{presentation_path}_{idx}.png"

            pil_image = Image.open(io.BytesIO(image))
            pil_image.save(presentation_image_path)

            presentation_image_paths.append(presentation_image_path)
        return presentation_image_paths, presentation_metadata_path
    else:
        presentation_image_path = f"{presentation_path}.png"
    
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.save(presentation_image_path)

        return presentation_image_path, presentation_metadata_path

def get_artifact_id(video_path, start_frame, end_frame, framerate):
    video_name = os.path.basename(video_path)

    start_s = f"{int(start_frame / framerate // 60):02d}-{int(start_frame / framerate % 60):02d}"
    end_s = f"{int(end_frame / framerate // 60):02d}-{int(end_frame / framerate % 60):02d}"

    presentation_file_stem = f"{video_name}_{start_s}_{end_s}"

    return presentation_file_stem

def find_place_for_box(cluster: list[Detections], box_height, box_width, image_height, image_width):
    cluster_data_list: list[tuple[int, int, int, int, float, float]] = [(xyxy[0], xyxy[1], xyxy[2], xyxy[3], (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2) for frame_detections in cluster for xyxy, *_ in frame_detections]
    cluster_data = np.array(cluster_data_list)

    cluster_mean = np.mean(cluster_data, axis=0)
    centroid_x: float = cluster_mean[4]
    centroid_y: float = cluster_mean[5]

    cluster_tlx = np.min(cluster_data[:, 0])
    cluster_tly = np.min(cluster_data[:, 1])
    cluster_brx = np.max(cluster_data[:, 2])
    cluster_bry = np.max(cluster_data[:, 3])
    
    # add padding
    cluster_tlx -= box_width
    cluster_tly -= box_height
    cluster_brx += box_width
    cluster_bry += box_height

    # clamp to image bounds
    cluster_tlx = max(0, cluster_tlx)
    cluster_tly = max(0, cluster_tly)
    cluster_brx = min(image_width, cluster_brx)
    cluster_bry = min(image_height, cluster_bry)

    # from the cluster bounds, make a list of all the possible box positions
    box_positions: list[tuple[int, int]]= []
    for y in range(int(cluster_tly), int(cluster_bry), box_height):
        for x in range(int(cluster_tlx), int(cluster_brx), box_width):
            box_positions.append((x, y))
    
    # sort the box positions by distance to the cluster centroid, ascending
    box_positions = sorted(box_positions, key=lambda pos: np.sqrt((pos[0] + box_width / 2 - centroid_x) ** 2 + (pos[1] + box_height / 2 - centroid_y) ** 2))

    # return the closest box position that does not overlap with any detection bounding box, and does not go out of bounds of the image
    for x, y in box_positions:
        if any([xyxy[0] < x + box_width and xyxy[2] > x and xyxy[1] < y + box_height and xyxy[3] > y for xyxy in cluster_data]):
            continue
        if x + box_width > image_width or y + box_height > image_height:
            continue
        if x < 0 or y < 0:
            continue
        return x, y

    # if no box position was found, return the centroid
    return (int(centroid_x), int(centroid_y))

from dataclasses import dataclass

@dataclass
class ClusterMetadata:
    box_label: str
    centroid_x: float
    centroid_y: float
    median_size: float
    cluster_size: int
    start: int
    end: int
    start_ms: int
    end_ms: int
    color: dict

def present_section(bg_image: Union[np.ndarray, list[np.ndarray]], clustered_detections: dict[int, list[Detections]], ignore_noise=True, index=None, version="v1"):
    section_metadata: dict[int, ClusterMetadata] = {}
    section_data: dict[int, np.ndarray] = {}
    section_length = len(next(iter(clustered_detections.values())))

    def calculate_section_metadata(clustered_detections: dict[int, list[Detections]]):
        for idx, (label, cluster) in enumerate(clustered_detections.items()):
            if ignore_noise and label == -1:
                continue

            cluster_data = [((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) for frame_detections in cluster for xyxy, *_ in frame_detections]
            cluster_data = np.array(cluster_data)

            if len(cluster_data) == 0:
                continue

            box_label = chr(label + 65)
            cluster_mean = np.mean(cluster_data, axis=0)
            centroid_x = cluster_mean[0]
            centroid_y = cluster_mean[1]
            median_size = cluster_mean[2]
            cluster_size = len(cluster_data)

            color = colors[idx % len(colors)]

            cluster_start_frame = 0
            cluster_end_frame = len(cluster) - 1

            # find the first non-empty frame
            for idx, frame_detections in enumerate(cluster):
                if len(frame_detections) > 0:
                    cluster_start_frame = idx
                    break
            for idx, frame_detections in enumerate(reversed(cluster)):
                if len(frame_detections) > 0:
                    cluster_end_frame = len(cluster) - idx - 1
                    break

            start_ms = cluster_start_frame * 1000 // 30
            end_ms = cluster_end_frame * 1000 // 30

            metadata = ClusterMetadata(box_label=box_label, centroid_x=centroid_x, centroid_y=centroid_y, median_size=median_size, cluster_size=cluster_size, start=cluster_start_frame, end=cluster_end_frame, start_ms=start_ms, end_ms=end_ms, color=color)
            section_metadata[label] = metadata
            section_data[label] = cluster_data

    if isinstance(bg_image, list):
        calculate_section_metadata(clustered_detections)

        images: list[bytes] = []

        split_clustered_detections_list: list[dict[int, list[Detections]]] = []
        split_k = len(bg_image)
        split_indices = [(idx * section_length // split_k, (idx + 1) * section_length // split_k) for idx in range(split_k)]

        print(f"[tracking] Splitting the section into {split_k} parts")
        print(f"[tracking] Split indices: {split_indices}")
        print(f"[tracking] Section length: {section_length}")

        for start, end in split_indices:
            split_clustered_detections = {label: cluster[start:end] for label, cluster in clustered_detections.items()}
            split_clustered_detections_list.append(split_clustered_detections)

        for idx, image in enumerate(bg_image):
            presented_image, _ = present_section(image, split_clustered_detections_list[idx], index=idx, ignore_noise=ignore_noise, version=version)

            if isinstance(presented_image, list):
                raise ValueError(f"Expected a single image, got a list of images at index {idx}")
            
            images.append(presented_image)
        
        return images, present_metadata(section_metadata, version=version)
    
    calculate_section_metadata(clustered_detections)

    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)

    # reduce the whole bg image saturation
    bg_image_hsv = cv2.cvtColor(bg_image, cv2.COLOR_BGR2HSV)
    bg_image_hsv[:, :, 1] = bg_image_hsv[:, :, 1] * 0.6
    bg_image = cv2.cvtColor(bg_image_hsv, cv2.COLOR_HSV2BGR)
    
    # Plotting the clusters
    for idx, (label, cluster_metadata) in enumerate(section_metadata.items()):
        cluster = clustered_detections[label]
        cluster_data = section_data[label]

        # ========== Plotting the detections ============
        for x, y, area in cluster_data:
            # add circle, radius is proportional to the area of the bounding box
            # cv2.circle(bg_image, (int(x), int(y)), int(np.sqrt(area) / 2), color["rgb"][::-1], -1)
            # add square
            radius = int(np.sqrt(area) / 2)
            padding = 2
            tlx, tly = int(x - radius - padding), int(y - radius - padding)
            brx, bry = int(x + radius + padding), int(y + radius + padding)

            cv2.rectangle(bg_image, (tlx, tly), (brx, bry), cluster_metadata.color["rgb"][::-1], 2)

        frame_centers = []

        # find the center of the every frame
        for idx in range(cluster_metadata.start, cluster_metadata.end + 1):
            frame_detections = cluster[idx]

            if len(frame_detections) > 0:
                centers = [((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2) for xyxy, *_ in frame_detections]
                centers = np.array(centers)
                center = np.mean(centers, axis=0)

                center = (center[0], center[1])

                frame_centers.append(center)

        frame_centers = np.array(frame_centers)
        
        # find the angles 
        cluster_angles = []
        last_center_idx = 0
        for idx in range(len(frame_centers) - 1):
            if idx == 0:
                cluster_angles.append(0)
            else:
                center = frame_centers[idx]
                last_center = frame_centers[last_center_idx]

                if last_center[0] != center[0] and last_center[1] != center[1]:
                    angle = np.arctan2(center[1] - last_center[1], center[0] - last_center[0])
                    cluster_angles.append(angle)

                    last_center_idx = idx

        cluster_angles = np.array(cluster_angles[1:])

        if len(cluster_angles) == 0:
            average_angle = 0
        else:
            median_angle = np.median(cluster_angles)
            weights = 1 / (np.abs(cluster_angles - median_angle) + 1e-10)
            average_angle = np.average(cluster_angles, weights=weights)

        # ========== Plotting the label box ============
        image_height, image_width, _ = bg_image.shape

        text_content = f"{cluster_metadata.box_label}"
        text_font = cv2.FONT_HERSHEY_DUPLEX
        text_font_scale = 0.8
        text_font_thickness = 1

        box_padding = 7
        box_alpha = 0.8

        text_size, _ = cv2.getTextSize(text_content, text_font, text_font_scale, text_font_thickness)
        box_size = (text_size[0] + box_padding * 2, text_size[1] + box_padding * 2)
        
        box_color = (255, 255, 255)
        text_color = cluster_metadata.color["rgb"][::-1]

        box_tlx, box_tly = find_place_for_box(cluster, box_size[1], box_size[0], image_height, image_width)
        box_brx, box_bry = box_tlx + box_size[0], box_tly + box_size[1]

        # clip to image bounds
        box_tlx = max(0, box_tlx)
        box_tly = max(0, box_tly)
        box_brx = min(bg_image.shape[1], box_brx)
        box_bry = min(bg_image.shape[0], box_bry)

        # HACK create a detection object for the label box, so that it can be merged with the rest of the cluster
        cluster_duplicate = [Detections.merge([frame_detections]) for frame_detections in cluster]
        temp_detection = Detections(np.array([[box_tlx, box_tly, box_brx, box_bry]]), None, np.array([1]), np.array([0]))
        cluster_duplicate[-1] = Detections.merge([cluster_duplicate[-1], temp_detection])

        # create the box pixels
        box_pixels = np.zeros((box_bry - box_tly, box_brx - box_tlx, 3), dtype=np.uint8)
        # add background
        cv2.rectangle(box_pixels, (0, 0), (box_brx - box_tlx, box_bry - box_tly), box_color, -1)
        # add border
        cv2.rectangle(box_pixels, (0, 0), (box_brx - box_tlx, box_bry - box_tly), text_color, 2)

        # merge the box pixels with the background
        merged_section = cv2.addWeighted(box_pixels, box_alpha, bg_image[box_tly:box_bry, box_tlx:box_brx], 1 - box_alpha, 0)

        # apply the merged section to the background
        bg_image[box_tly:box_bry, box_tlx:box_brx] = merged_section

        text_origin = (box_tlx + box_padding, box_bry - box_padding)
        cv2.putText(bg_image, text_content, text_origin, text_font, text_font_scale, text_color, text_font_thickness, cv2.LINE_AA)

        arrow_length = 50
        arrow_tail_width = 4
        arrow_head_length = 25
        arrow_angle = np.pi / 8
        arrow_padding = 10

        arrow_head_point = (int(arrow_length / 2 * np.cos(average_angle)), int(arrow_length / 2 * np.sin(average_angle)))
        arrow_tail_point = (int(arrow_length / 2 * np.cos(average_angle + np.pi)), int(arrow_length / 2 * np.sin(average_angle + np.pi)))

        arrow_width = max(arrow_tail_point[0], arrow_head_point[0]) - min(arrow_tail_point[0], arrow_head_point[0])
        arrow_height = max(arrow_tail_point[1], arrow_head_point[1]) - min(arrow_tail_point[1], arrow_head_point[1])

        arrow_box_size = (arrow_width + arrow_padding * 2, arrow_height + arrow_padding * 2)

        arrow_tlx, arrow_tly = find_place_for_box(cluster_duplicate, arrow_box_size[1], arrow_box_size[0], image_height, image_width)
        arrow_brx, arrow_bry = arrow_tlx + arrow_box_size[0], arrow_tly + arrow_box_size[1]

        # clip to image bounds
        arrow_tlx = max(0, arrow_tlx)
        arrow_tly = max(0, arrow_tly)
        arrow_brx = min(bg_image.shape[1], arrow_brx)
        arrow_bry = min(bg_image.shape[0], arrow_bry)

        # create the box pixels
        arrow_box_pixels = np.zeros((arrow_bry - arrow_tly, arrow_brx - arrow_tlx, 3), dtype=np.uint8)
        # add background
        cv2.rectangle(arrow_box_pixels, (0, 0), (arrow_brx - arrow_tlx, arrow_bry - arrow_tly), box_color, -1)
        # add border
        cv2.rectangle(arrow_box_pixels, (0, 0), (arrow_brx - arrow_tlx, arrow_bry - arrow_tly), text_color, 2)

        # merge the box pixels with the background
        merged_section = cv2.addWeighted(arrow_box_pixels, box_alpha, bg_image[arrow_tly:arrow_bry, arrow_tlx:arrow_brx], 1 - box_alpha, 0)

        # apply the merged section to the background
        bg_image[arrow_tly:arrow_bry, arrow_tlx:arrow_brx] = merged_section

        arrow_head_point = (arrow_tlx + arrow_padding + arrow_width // 2 + arrow_head_point[0], arrow_tly + arrow_padding + arrow_height // 2 + arrow_head_point[1])
        arrow_tail_point = (arrow_tlx + arrow_padding + arrow_width // 2 + arrow_tail_point[0], arrow_tly + arrow_padding + arrow_height // 2 + arrow_tail_point[1])

        arrow_head_points = [
            arrow_head_point,
            (int(arrow_head_point[0] - arrow_head_length * np.cos(average_angle - arrow_angle)), int(arrow_head_point[1] - arrow_head_length * np.sin(average_angle - arrow_angle))),
            (int(arrow_head_point[0] - arrow_head_length * np.cos(average_angle + arrow_angle)), int(arrow_head_point[1] - arrow_head_length * np.sin(average_angle + arrow_angle)))
        ]

        cv2.drawContours(bg_image, [np.array(arrow_head_points)], 0, text_color, -1, cv2.LINE_AA)

        arrow_head_point = (int(arrow_head_point[0] - arrow_tail_width * np.cos(average_angle)), int(arrow_head_point[1] - arrow_tail_width * np.sin(average_angle)))
        cv2.line(bg_image, arrow_head_point, arrow_tail_point, text_color, arrow_tail_width, cv2.LINE_AA)
    
    if index is not None:
        text = f"Image {index + 1}"
        text_font = cv2.FONT_HERSHEY_DUPLEX
        text_font_scale = 0.8
        text_font_thickness = 1

        text_size, _ = cv2.getTextSize(text, text_font, text_font_scale, text_font_thickness)
        text_origin = (10, 10 + text_size[1] + 10)

        box_size = (text_size[0] + 20, text_size[1] + 20)

        cv2.rectangle(bg_image, (0, 0), (box_size[0], box_size[1]), (0, 0, 0), -1)
        cv2.putText(bg_image, text, text_origin, text_font, text_font_scale, (255, 255, 255), text_font_thickness, cv2.LINE_AA)

    image_bytes = cv2.imencode('.png', bg_image)[1].tobytes()
    presented_metadata = present_metadata(section_metadata, version="v1")

    return image_bytes, presented_metadata

def present_metadata(section_metadata: dict[int, ClusterMetadata], version="v1"):
    presented_metadata = []

    if version == "v2":
        headers = "id, cluster size, start time, end time, color"
    elif version == "v3":
        headers = "id, color"
    else:
        headers = "id, centroid x, centroid y, mean detection (square pixels), cluster size, start time, end time, color"

    for m in section_metadata.values():
        formatted_start_time = f"00:{m.start_ms // 1000:02d}.{m.start_ms % 1000:03d}"
        formatted_end_time = f"00:{m.end_ms // 1000:02d}.{m.end_ms % 1000:03d}"

        if version == "v2":
            presented_metadata.append(f"{m.box_label}, {m.cluster_size}, {formatted_start_time}, {formatted_end_time}, {m.color['name']}")
        if version == "v3":
            presented_metadata.append(f"{m.box_label}, {m.color['name']}")
        else:
            presented_metadata.append(f"{m.box_label}, {m.centroid_x:.0f}, {m.centroid_y:.0f}, {m.median_size:.2f}sp, {m.cluster_size}, {formatted_start_time}, {formatted_end_time}, {m.color['name']}")

    presented_metadata = "\n".join(presented_metadata)
    presented_metadata = f"{headers}\n{presented_metadata}"

    return presented_metadata