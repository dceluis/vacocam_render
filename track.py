import numpy as np
from numpy.typing import NDArray
import copy
from PIL import Image

from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

from core.detections import Detections, load_video_detections, save_video_detections

from typing import List, Tuple, Optional

class IgnoreZone:
    def __init__(self, pixels: set[tuple[int, int]], min_area: Optional[int], max_area: Optional[int]):
        self.pixels = pixels
        self.min_area = min_area
        self.max_area = max_area

    # this takes a Detections object that has a single detection
    def __contains__(self, detections: Detections):
        if len(detections) > 1:
            print("Warning: IgnoreZone.__contains__ called with a Detections object that has more than one detection.")

        xyxy = detections.xyxy[0]

        center = (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

        if self.min_area is not None and area < self.min_area:
            return False
        if self.max_area is not None and area > self.max_area:
            return False
        
        return center in self.pixels

def skewed_gaussian_kernel(sigma, kernel_size, skew_factor):
    # Generate a standard Gaussian kernel (symmetric)
    x = np.linspace(-kernel_size, kernel_size, 2 * kernel_size + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))

    # Apply skewness: Reduce the weight of future values
    kernel[x > 0] *= skew_factor  # Reduce weights for future values (x > 0)

    # Normalize the kernel
    kernel /= np.sum(kernel)
    return kernel

def apply_skewed_filter(data, sigma, kernel_size, skew_factor):
    sigma = 60
    kernel_size = 480
    skew_factor = 0.7

    kernel = skewed_gaussian_kernel(sigma, kernel_size, skew_factor)
    smoothed_data = convolve1d(data, kernel, mode='reflect')
    return smoothed_data

def apply_gaussian_filter(data: np.ndarray):
    sigma = 40

    return gaussian_filter1d(data, sigma)

def apply_filter(data: np.ndarray):
    return apply_gaussian_filter(data)
    # return apply_skewed_filter(data)
    
def smoothen_pois(points: np.ndarray):
    x_smooth = apply_filter(points[:, 0])
    y_smooth = apply_filter(points[:, 1])

    res = [(x, y) for x, y in zip(x_smooth, y_smooth)]
    
    return res

def smoothen_aois(areas: np.ndarray):
    smooth_aois = apply_filter(areas)

    return smooth_aois

def augment_detections(detections_list: list[Detections]):
    detections_with_interests = calc_interests_new(detections_list)

    detections_with_smooth_interests = smoothen_interests(detections_with_interests)

    return detections_with_smooth_interests

# using "smoothen" to mean the action of smoothing, since "smooth" can be both an adjective and a verb
def smoothen_interests(detections_list: list[Detections]):
    pois = [detection.point_of_interest for detection in detections_list]
    aois = [detection.area_of_interest for detection in detections_list]
    pois = np.array(pois)
    aois = np.array(aois)

    smooth_pois = smoothen_pois(pois)
    smooth_aois = smoothen_aois(aois)

    result: list[Detections] = []

    for idx, frame_detections in enumerate(detections_list):
        new_detections = copy.copy(frame_detections)
        new_detections.point_of_interest = smooth_pois[idx]
        new_detections.area_of_interest = smooth_aois[idx]
        result.append(new_detections)

    return result

def calc_interests(detections_list: list[Detections]):
    pois = calc_pois(detections_list)
    aois = calc_aois(detections_list)

    result: list[Detections] = []

    for idx, frame_detections in enumerate(detections_list):
        new_detections = copy.copy(frame_detections)
        new_detections.point_of_interest = pois[idx]
        new_detections.area_of_interest = aois[idx]
        result.append(new_detections)
    
    return result

def calc_interests_new(detections_list: list[Detections]):
    chunk_size = 64
    overlap_size = 16

    X_interpolated: list[float] = []
    Y_interpolated: list[float] = []
    A_interpolated: list[float] = []

    current_index = 0
    
    while current_index < len(detections_list):
        current_overlap = overlap_size if current_index > 0 else 0

        x = X_interpolated[len(X_interpolated) - current_overlap:]
        y = Y_interpolated[len(Y_interpolated) - current_overlap:]
        a = A_interpolated[len(A_interpolated) - current_overlap:]
        weights = [1.] * current_overlap

        last_center = None if current_index == 0 else (X_interpolated[-1], Y_interpolated[-1])
        last_area = None if current_index == 0 else A_interpolated[-1]

        detections_start = current_index
        detections_end = min(current_index + chunk_size - 1, len(detections_list))

        no_detection = True

        while no_detection or ((detections_end - detections_start) < chunk_size):
            if detections_end >= len(detections_list):
                break

            frame_detection = detections_list[detections_end]

            if len(frame_detection) > 0:
                no_detection = False

            detections_end += 1

        detections_chunk = detections_list[detections_start:detections_end]

        for frame_detections in detections_chunk:
            if len(frame_detections) > 0:
                min_distance = float("inf")

                for bbox in frame_detections.xyxy:
                    center_x: float = (bbox[0] + bbox[2]) / 2
                    center_y: float = (bbox[1] + bbox[3]) / 2
                    area: float = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                    distance = 0 if last_center is None else np.linalg.norm(np.array((center_x, center_y)) - np.array(last_center))
                    distance = float(distance)

                    if distance < min_distance:
                        min_distance = distance
                        last_center = (center_x, center_y)
                        last_area = area
                
                if last_center is None or last_area is None:
                    raise Exception("Error calculating interests")
                else:
                    x.append(last_center[0])
                    y.append(last_center[1])
                    a.append(last_area)

                    weights.append(1)
            else:
                x.append(1920 / 2)
                y.append(1080 / 2)
                a.append(100)
                weights.append(1e-10)

        spl_x = UnivariateSpline(range(len(x)), x, w=weights, k=1)
        spl_y = UnivariateSpline(range(len(y)), y, w=weights, k=1)
        spl_a = UnivariateSpline(range(len(a)), a, w=weights, k=1)

        interpolated_x = spl_x(range(len(x)))
        interpolated_y = spl_y(range(len(y)))
        interpolated_a = spl_a(range(len(a)))

        X_interpolated.extend(interpolated_x[current_overlap:])
        Y_interpolated.extend(interpolated_y[current_overlap:])
        A_interpolated.extend(interpolated_a[current_overlap:])

        current_index += len(detections_chunk)
    
    result: list[Detections] = []
    
    for idx, frame_detections in enumerate(detections_list):
        new_detections = copy.copy(frame_detections)
        new_detections.point_of_interest = tuple(map(int, (X_interpolated[idx], Y_interpolated[idx])))
        new_detections.area_of_interest = int(A_interpolated[idx])
        result.append(new_detections)

    return result

def ignore_detections(detections_list: List[Detections], ignore_zone: IgnoreZone):
    ignored_count = 0
    valid_detections: list[Detections] = []

    for frame_detections in detections_list:
        filtered = []

        split_detections = frame_detections.to_split_list()
        for single_detections in split_detections:
            if single_detections not in ignore_zone:
                filtered.append(single_detections)
            else:
                ignored_count += 1

        valid_detections.append(Detections.merge(filtered))
    
    if ignored_count > 0:
        print(f"[IgnoreZone] Ignored {ignored_count} detections")

    return valid_detections

def ignore_detections_from_mask(detections_list: List[Detections], mask_path=""):
    def _generate_ignore_zone(mask_path) -> IgnoreZone:
        # Load the image
        image = Image.open(mask_path)

        # Convert the image to RGBA if it is not already in that mode
        image = image.convert("RGBA")

        # Initialize a list to hold the coordinates of transparent pixels
        transparent_pixels = []

        # Iterate over the pixels of the image
        for y in range(image.height):
            for x in range(image.width):
                r, g, b, a = image.getpixel((x, y))
                # Check if the pixel is transparent
                if a == 0:
                    transparent_pixels.append((x, y))

        ignore_zone = IgnoreZone(set(transparent_pixels), None, None)

        return ignore_zone
    
    ignore_zone = _generate_ignore_zone(mask_path)
    
    return ignore_detections(detections_list, ignore_zone)

def calc_aois(detections_list, default=100):
    def _calculate_cumulative_area(bboxes):
        if len(bboxes) == 0:
            return None
        areas = [((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) for bbox in bboxes]
        avg_area = np.mean(areas, axis=0)
        return avg_area.astype(float)

    areas = [default]
    
    # Process each frame
    for detections in detections_list:
        avg_area = _calculate_cumulative_area(detections.xyxy)

        if avg_area is None:
            avg_area = areas[-1]

        areas.append(avg_area)
    
    areas.pop(0)

    res = np.array(areas)
    
    return res

def calc_pois(detections_list, default=(int(1920 / 2), int(1440/2))):
    # Function to calculate the average center of bounding boxes
    def _calculate_cumulative_center(bboxes):
        if len(bboxes) == 0:
            return None
        centers = [(int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)) for bbox in bboxes]
        avg_center = np.mean(centers, axis=0)
        return avg_center.astype(int)

    points_of_interest = [default]

    # Process each frame
    for detections in detections_list:
        # Calculate the average center for the current frame
        avg_center = _calculate_cumulative_center(detections.xyxy)

        # If no detection in current frame, use the previous frame's point
        if avg_center is None:
            avg_center = points_of_interest[-1]

        # Append the point of interest for the current frame
        points_of_interest.append(avg_center)


    points_of_interest.pop(0)
    
    res = np.array(points_of_interest)
    
    return res

def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def decay_heatmap(heatmap, decay_lookup):
    non_zero_indices = heatmap != 0

    clipped_values = heatmap[non_zero_indices]
    
    heatmap[non_zero_indices] = decay_lookup[clipped_values]

    return heatmap

def update_heatmap(heatmap, boxes, intensity_increase):
    area_influence = 400

    h, w = heatmap.shape

    for (x1, y1, x2, y2) in boxes:
        max_side = max((x2 - x1), (y2 - y1))
        
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        pad = max_side * (area_influence ** 0.5 - 1) / 2
        
        tlx, tly, brx, bry = (
            int(max(center_x - max_side / 2 - pad, 0)),
            int(max(center_y - max_side / 2 - pad, 0)),
            int(min(center_x + max_side / 2 + pad, w)),
            int(min(center_y + max_side / 2 + pad, h))
        )
        
        # Create a grid of x, y coordinates
        x_grid, y_grid = np.meshgrid(np.arange(tlx, brx), np.arange(tly, bry))

        # Calculate distance from the center
        distance = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

        # Calculate Gaussian intensity based on distance from the center
        sigma = pad / 3.5 # Standard deviation of the Gaussian
        gaussian_intensity_x = gaussian(x_grid, center_x, sigma)
        gaussian_intensity_y = gaussian(y_grid, center_y, sigma)
        gaussian_intensity = gaussian_intensity_x * gaussian_intensity_y * intensity_increase

        # Update the heatmap
        heatmap[tly:bry, tlx:brx] += gaussian_intensity.astype(np.uint16)

    return heatmap

def find_max(heatmap):
    # Flatten the heatmap and find the index of the maximum value
    max_index_flat = np.argmax(heatmap)
    
    # Convert the index back to 2D coordinates
    max_coord = np.unravel_index(max_index_flat, heatmap.shape)
    
    # Get the actual maximum value
    max_value = heatmap[max_coord]

    return max_coord, max_value

from utils.annotators import HeatMapAnnotator

def calc_pois_new(detections_list, canvas_height = 1440, canvas_width = 1920):
    decay_resolution = 1024
    decay_rate = 256
    decay_attack = 4

    intensity_increase = decay_rate
    
    nprange = np.arange(decay_resolution)

    # Precompute cube root values for integers 0 to decay_resolution
    decay_factors = (nprange ** decay_attack) / ((len(nprange) - 1) ** decay_attack) * decay_rate
    decay_lookup = np.clip(nprange - decay_factors, 0, decay_resolution - 1).astype(np.uint16)

    # Initialize an empty list to store heatmaps
    current_heatmap = np.zeros((canvas_height, canvas_width), dtype=np.uint16)  # Use NumPy array
    
    default = (canvas_width // 2, canvas_height // 2)
    points_of_interest = [default]

    # Process detections for each frame
    for detections in detections_list:
        decay_heatmap(current_heatmap, decay_lookup)
        update_heatmap(current_heatmap, detections.xyxy, intensity_increase)
        np.clip(current_heatmap, 0, decay_resolution - 1, current_heatmap)

        max_coord, max_value = find_max(current_heatmap)
        
        if max_value == 0:
            poi = points_of_interest[-1]
        else:
            poi = max_coord

        points_of_interest.append(poi)
        
    points_of_interest.pop(0)
    
    res = np.array(points_of_interest)
    
    return res

# =====================================================================
# =====================================================================
# ============================== VacoCam ==============================
# =====================================================================
# =====================================================================

import numpy as np
import supervision as sv
import os
import cv2
import io
import argparse
import base64
import json
import random
from PIL import Image
from io import BytesIO

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from typing import List, Tuple
from dataclasses import dataclass

from core.gpt4 import submit_image as submit_image_to_gpt4
from core.detections import Detections, load_video_detections, save_video_detections

def cluster_detections(detections_list: List[Detections], preset=None):
    X_Y = []
    AREA = []
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

            X_Y.append((x, y))
            AREA.append((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            FRAME.append(idx)

            DATA_FLAT.append(Detections(det_xyxy, None, det_conf, det_cls))

        detections_lengths[idx] = len(frame_detections)

    X_Y = np.array(X_Y)
    AREA = np.array(AREA)
    FRAME = np.array(FRAME)

    if len(DATA_FLAT) == 0:
        return {-1: detections_list}
    elif len(DATA_FLAT) == 1:
        return {-1: detections_list}

    if preset == 'static':
        eps = 0.02
        min_samples = 300
    elif preset == 'play':
        eps = 0.85
        min_samples = 5
    else:
        eps = 0.5
        min_samples = 50

    X_Y_scaled = StandardScaler().fit_transform(X_Y)
    AREA_scaled = StandardScaler().fit_transform(AREA.reshape(-1, 1))
    FRAME_scaled = MinMaxScaler().fit_transform(FRAME.reshape(-1, 1)) * 4 - 2

    # concatenate the scaled x, y and area values
    DATA = np.hstack((X_Y_scaled, AREA_scaled, FRAME_scaled))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    X_clusters = dbscan.fit_predict(DATA)

    if preset == 'play':
        min_samples_real = 15
        # count the number of detections in each cluster
        cluster_counts = np.bincount(X_clusters + 1)
        # get the indices of the clusters with less than min_sampbles_real detections
        small_clusters = np.where(cluster_counts < min_samples_real)[0]

        # set the labels of the small clusters to -1 (noise)
        for cluster in small_clusters:
            X_clusters[X_clusters == cluster - 1] = -1

    unique_labels = list(map(int, set(X_clusters)))

    # Initialize lists to hold clustered detections for each frame
    clustered_detections = {cluster_id: [Detections.empty()] * len(detections_list) for cluster_id in unique_labels}

    processed_detections_idx = 0
    # Assign each detections object to the appropriate cluster and frame
    for frame_index, length in enumerate(detections_lengths):
        clusters = X_clusters[processed_detections_idx:processed_detections_idx + length]
        data = DATA_FLAT[processed_detections_idx:processed_detections_idx + length]
        
        clustered_detections_for_frame: dict[int, list[Detections]] = {cluster_id: [] for cluster_id in unique_labels}

        for idx in range(length):
            cluster_id = clusters[idx]
            detection = data[idx]

            detection.cluster_id = np.array([cluster_id])

            clustered_detections_for_frame[cluster_id].append(detection)
        
        for cluster_id, cluster_detections_list in clustered_detections_for_frame.items():
            clustered_detections[cluster_id][frame_index] = Detections.merge(cluster_detections_list)

        processed_detections_idx += length

    return clustered_detections

def track_video(video_path, detections: List[Detections], tracking="declustered"):
    if tracking == "declustered":
        clustered_detections = cluster_detections(detections, preset="static")

        if 'CLUSTER_PREVIEW' in os.environ:
            video_info = sv.VideoInfo.from_video_path(video_path)

            unique_cluster_ids = clustered_detections.keys()

            print(f"Number of clusters: {len(unique_cluster_ids)} ({', '.join(str(cluster_id) for cluster_id in unique_cluster_ids)})")

            for cluster_id, cluster in clustered_detections.items():
                # Selecting data points in the current cluster
                cluster_data = [((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2) for frame_detections in cluster for xyxy, *_ in frame_detections]
                cluster_data = np.array(cluster_data)

                # Plotting the cluster
                plt.figure(figsize=(15, 10))

                if 'STATIC_CLUSTER_BACKGROUND_IMAGE' in os.environ:
                    bg_image = mpimg.imread(os.environ['STATIC_CLUSTER_BACKGROUND_IMAGE'])
                    bg_image = np.flipud(bg_image)
                    display_shape = bg_image.shape[:2]

                    plt.imshow(bg_image, extent=(0, display_shape[1], 0, display_shape[0]))

                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], marker='o', s=10)

                # show cluster size
                plt.text(0, 0, f"Cluster size: {len(cluster_data)}", fontsize=12)
                
                plt.title(f"DBSCAN Clustering - Cluster {cluster_id}")
                plt.xlim([0, video_info.width])
                plt.ylim([0, video_info.height])
                plt.gca().invert_yaxis()
                plt.show()
                plt.close()

        filtered_detections = clustered_detections[-1]
    elif tracking == "vacocam":
        seconds = 10

        video_info = sv.VideoInfo.from_video_path(video_path)

        framerate = video_info.fps
        total_frames = video_info.total_frames or 0

        frame_indices = [(i * seconds * framerate, (i + 1) * seconds * framerate) for i in range(int(total_frames / (seconds * framerate)))]
        if frame_indices[-1][1] != total_frames:
            frame_indices.append((frame_indices[-1][1], total_frames))

        def get_video_section_sample(start, end):
            # find a frame in the middle of the section, but provided it has detections
            mid = int((start + end) / 2)

            while len(detections[mid]) == 0:
                mid += 1

            cap = cv2.VideoCapture(video_path)

            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, sample = cap.read()
            
            if not ret:
                raise Exception("Error reading video")

            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

            cap.release()
            return sample
        
        def get_ignore_zone_from_cluster(cluster: list[Detections]):
            # first, flatten the detections to a numpy array
            detections_data = [((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) for frame_detections in cluster for xyxy, *_ in frame_detections]
            detections_data = np.array(detections_data)
            
            # then, find the mean of the detections centers and areas
            mean_center = np.mean(detections_data[:, :2], axis=0)
            mean_area = np.mean(detections_data[:, 2], axis=0)

            # restrict the detections to the ones that are within 2 standard deviations of the mean
            std_center = np.std(detections_data[:, :2], axis=0)
            std_area = np.std(detections_data[:, 2], axis=0)

            min_center = mean_center - 2 * std_center
            max_center = mean_center + 2 * std_center

            min_area = mean_area - 2 * std_area
            max_area = mean_area + 2 * std_area

            filtered_detections_data = [detection for detection in detections_data if min_center[0] <= detection[0] <= max_center[0] and min_center[1] <= detection[1] <= max_center[1] and min_area <= detection[2] <= max_area]
            filtered_detections_data = np.array(filtered_detections_data)

            # get the top left and bottom right corners of the bounding box that contains all the filtered detections

            if len(filtered_detections_data) == 0:
                return IgnoreZone(set(), None, None)
            else:
                min_x = np.min(filtered_detections_data[:, 0])
                min_y = np.min(filtered_detections_data[:, 1])

                max_x = np.max(filtered_detections_data[:, 0])
                max_y = np.max(filtered_detections_data[:, 1])

                # create a set of all the pixels in the bounding box
                pixels = set()

                for x in range(int(min_x), int(max_x) + 1):
                    for y in range(int(min_y), int(max_y) + 1):
                        pixels.add((x, y))

                return IgnoreZone(pixels, min_area, max_area)

        @dataclass
        class OverlappingCluster:
            overlap: int
            clusters: dict[int, List[Detections]]
            bounds: dict[int, Tuple[int, int]]

        @dataclass
        class VideoSection:
            detections: List[Detections]
            clustered_detections: dict[int, List[Detections]]
            clustered_detections_minus_noise: dict[int, List[Detections]]
            overlapping_clusters: List[OverlappingCluster]

        video_sections: dict[Tuple[int, int], VideoSection] = {}
        overlaps_count = 0

        for start, end in frame_indices:
            section_detections = detections[start:end]
            clustered_detections = cluster_detections(section_detections, preset="play")
            clustered_detections_minus_noise = { key: detections for key, detections in clustered_detections.items() if key != -1 }

            # find clusters start and ends
            start_ends: dict[int, tuple[int, int]] = { key: (0, 0) for key in clustered_detections_minus_noise.keys() }

            for key, c_detections in clustered_detections_minus_noise.items():
                # find start of the cluster, meaning the index of the first detection that is not empty. (len(detection) > 0)
                cluster_start = 0
                for i, detection in enumerate(c_detections):
                    if len(detection) > 0:
                        cluster_start = i
                        break
                # find end of the cluster, meaning the index of the last detection that is not empty. (len(detection) > 0)
                cluster_end = 0
                for i, detection in enumerate(reversed(c_detections)):
                    if len(detection) > 0:
                        cluster_end = len(c_detections) - i
                        break
                start_ends[key] = (cluster_start, cluster_end)
            
            # now find any overlapping clusters
            overlapping_clusters: List[OverlappingCluster] = []

            for key1, (start1, end1) in start_ends.items():
                for key2, (start2, end2) in start_ends.items():
                    if key1 != key2 and start1 < end2 and end1 > start2:
                        overlap = min(end1, end2) - max(start1, start2)

                        overlapping_clusters_key_pairs = [(oc.clusters.keys(), (key1, key2)) for oc in overlapping_clusters]

                        if (key1, key2) in overlapping_clusters_key_pairs or (key2, key1) in overlapping_clusters_key_pairs:
                            pass
                            # continue
                        if overlap < min(end1 - start1, end2 - start2) * 0.5:
                            continue
                        if end1 - start1 < 15 or end2 - start2 < 15 or overlap < 15:
                            continue

                        overlaps_count += 1

                        overlapping_cluster = OverlappingCluster(
                            overlap=overlap,
                            clusters={
                                key1: clustered_detections_minus_noise[key1],
                                key2: clustered_detections_minus_noise[key2]
                            },
                            bounds={
                                key1: (start + start1, start + end1),
                                key2: (start + start2, start + end2)
                            }
                        )

                        overlapping_clusters.append(overlapping_cluster)
            
            video_section = VideoSection(
                detections=section_detections,
                clustered_detections=clustered_detections,
                clustered_detections_minus_noise=clustered_detections_minus_noise,
                overlapping_clusters=overlapping_clusters
            )

            video_sections[(start, end)] = video_section

        print("[VacomCam] Done finding overlaps")
        print("[VacomCam] Found {} overlaps".format(overlaps_count))

        ########## Save overlapping sections to disk

        # lets also make a video of the overlaps for easier viewing
        v_out = cv2.VideoWriter("overlaps.mp4", cv2.VideoWriter_fourcc(*'mp4v'), framerate, (1920, 1080))

        for (section_start, section_end), section_data in video_sections.items():
            overlapping_clusters = section_data.overlapping_clusters

            formatted_start = f"{int(section_start / framerate / 60)}:{int(section_start / framerate % 60)}"
            formatted_end = f"{int(section_end / framerate / 60)}:{int(section_end / framerate % 60)}"

            print(f"{formatted_start} - {formatted_end}")

            for overlap_data in overlapping_clusters:
                key1, key2 = overlap_data.clusters.keys()

                start1, end1 = overlap_data.bounds[key1]
                start2, end2 = overlap_data.bounds[key2]

                start = min(start1, start2)
                end = max(end1, end2)

                key1_formatted = chr(key1 + 65)
                key2_formatted = chr(key2 + 65)

                print(f"\t{key1_formatted} - {key2_formatted} ({overlap_data.overlap} frames overlap)")
                formatted_start = f"{int(start / framerate / 60)}:{int(start / framerate % 60)}.{int(start / framerate % 1 * 1000)}"
                formatted_end = f"{int(end / framerate / 60)}:{int(end / framerate % 60)}.{int(end / framerate % 1 * 1000)}"
                print(f"\t\t{formatted_start} - {formatted_end}")

                artifact_id = get_artifact_id(video_path, start, end, framerate, [key1_formatted, key2_formatted])

                sample = get_video_section_sample(start, end)

                section_img, section_metadata = present_section(sample, overlap_data.clusters)
                
                save_section_presentation(artifact_id, section_img, section_metadata)

                video_frame = np.array(Image.open(BytesIO(section_img)))
                v_out.write(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))

        v_out.release()

        # Send overlapping sections to GPT-4

        filtered_detections = []
        ignore_zones: list[IgnoreZone] = []

        for (section_start, section_end), section_data in video_sections.items():
            section_detections = section_data.detections
            clustered_detections = section_data.clustered_detections
            clustered_detections_minus_noise = section_data.clustered_detections_minus_noise
            overlapping_clusters = section_data.overlapping_clusters

            new_ignore_zone = None

            if len(overlapping_clusters) > 0:
                secondary_ids = []

                for overlap_data in overlapping_clusters:
                    key1, key2 = overlap_data.clusters.keys()

                    start1, end1 = overlap_data.bounds[key1]
                    start2, end2 = overlap_data.bounds[key2]

                    start = min(start1, start2)
                    end = max(end1, end2)

                    key1_formatted = chr(key1 + 65)
                    key2_formatted = chr(key2 + 65)

                    artifact_id = get_artifact_id(video_path, start, end, framerate, [key1_formatted, key2_formatted])

                    section_img, section_metadata = load_section_presentation(artifact_id)

                    if section_img is None or section_metadata is None:
                        raise Exception("Error loading section presentation")

                    print(f"[VacomCam] Submitting section to GPT-4 ({artifact_id})")
                    loaded_response = load_gpt4_response(artifact_id)

                    if loaded_response is not None:
                        gpt4_response = loaded_response
                    else:
                        # raise Exception("Error loading GPT-4 response")
                        gpt4_response = ask_gippity_for_primary_clusters(section_img, section_metadata)
                    
                        if gpt4_response is None:
                            print("GPT-4 response was None, SAVING EMPTY RESPONSE")

                        save_gpt4_response(artifact_id, gpt4_response)

                    gpt4_response_parsed = parse_gpt4_response(gpt4_response)

                    if gpt4_response_parsed is None:
                        print("GPT-4 response could not be parsed, skipping section")
                        continue

                    overlap_secondary_ids = [ord(id) - 65 for id, is_primary in gpt4_response_parsed.items() if not is_primary]
                    secondary_ids.extend(overlap_secondary_ids)

                # The little trick big tech doesn't want you to know about
                for secondary_id in secondary_ids:
                    new_ignore_zone = get_ignore_zone_from_cluster(clustered_detections_minus_noise[secondary_id])

                primary_detections = [clustered_detections_minus_noise[id] for id in clustered_detections_minus_noise.keys() if id not in secondary_ids]
            else:
                primary_detections = [clustered_detections_minus_noise[id] for id in clustered_detections_minus_noise.keys()]

            if len(primary_detections) == 0:
                # primary_detections.append(clustered_detections[-1])
                section_result = [Detections.empty()] * len(section_detections)
            else:
                # primary_detections.append(clustered_detections[-1])
                transposed = list(map(list, list(zip(*primary_detections))))

                section_result = [Detections.merge(data) for data in transposed]

            # VACOCAM SECRET SAUCE
            for ignore_zone in ignore_zones:
                section_result = ignore_detections(section_result, ignore_zone)

            if new_ignore_zone is not None:
                ignore_zones.append(new_ignore_zone)

            filtered_detections.extend(section_result)

        print("[VacomCam] Done processing video sections")
    else:
        print("Tracking method not implemented")
        filtered_detections = detections

    _, detections_path = save_video_detections(filtered_detections, video_path, module="track", version=tracking)

    return detections_path

@dataclass
class Color:
    name: str
    hex: str
    rgb: list[int]
    cmyk: list[int]
    hsb: list[int]
    hsl: list[int]
    lab: list[int]

colors = [
    # Color(name="Blue", hex="0015ff", rgb=[0,21,255], cmyk=[100,92,0,0], hsb=[235,100,100], hsl=[235,100,50], lab=[33,76,-106]),
    # Color(name="Pumpkin", hex="ff7300", rgb=[255,115,0], cmyk=[0,55,100,0], hsb=[27,100,100], hsl=[27,100,50], lab=[65,49,73]),
    # Color(name="Chartreuse", hex="90fe00", rgb=[144,254,0], cmyk=[43,0,100,0], hsb=[86,100,100], hsl=[86,100,50], lab=[90,-63,86]),
    # Color(name="Violet", hex="8400ff", rgb=[132,0,255], cmyk=[48,100,0,0], hsb=[271,100,100], hsl=[271,100,50], lab=[41,83,-92]),
    Color(name="Red", hex="ff0a12", rgb=[255,10,18], cmyk=[0,96,93,0], hsb=[358,96,100], hsl=[358,100,52], lab=[54,80,63]),
    Color(name="Yellow", hex="fffb00", rgb=[255,251,0], cmyk=[0,2,100,0], hsb=[59,100,100], hsl=[59,100,50], lab=[96,-20,94]),
    Color(name="Cyan", hex="00fff7", rgb=[0,255,247], cmyk=[100,0,3,0], hsb=[178,100,100], hsl=[178,100,50], lab=[91,-50,-10]),
    # Color(name="Rose", hex="ff00a1", rgb=[255,0,161], cmyk=[0,100,37,0], hsb=[322,100,100], hsl=[322,100,50], lab=[56,87,-14]),
]

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def ask_gippity_for_primary_clusters(image_bytes: bytes, metadata: str):
    """
    Returns a list of tuples of (box_label (str), is_primary (bool)), from asking GPT-4 to select the primary clusters in the image.
    Or None if GPT-4 could not find any clusters.
    """
    encoded_image = encode_image(image_bytes)

    response_json = submit_image_to_gpt4(encoded_image, metadata)

    return response_json

def save_gpt4_response(artifact_id, response_json):
    output_dir = os.path.join(os.path.dirname(__file__), ".track", "vacocam", "gpt4")

    gpt4_response_path = os.path.join(output_dir, f"{artifact_id}.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(gpt4_response_path, 'w') as f:
        json.dump(response_json, f)

    return gpt4_response_path

def load_gpt4_response(artifact_id):
    output_dir = os.path.join(os.path.dirname(__file__), ".track", "vacocam", "gpt4")

    gpt4_response_path = os.path.join(output_dir, f"{artifact_id}.json")

    if os.path.exists(gpt4_response_path):
        with open(gpt4_response_path, 'r') as f:
            response_json = json.load(f)

        return response_json
    else:
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

def load_section_presentation(artifact_id):
    output_dir = os.path.join(os.path.dirname(__file__), ".track", "vacocam", "presentation")

    presentation_path = os.path.join(output_dir, artifact_id)

    return load_presentation(presentation_path)

def save_section_presentation(artifact_id, image_bytes, metadata: str):
    output_dir = os.path.join(os.path.dirname(__file__), ".track", "vacocam", "presentation")

    presentation_path = os.path.join(output_dir, artifact_id)

    presentation_image_path, presentation_metadata_path = save_presentation(presentation_path, image_bytes, metadata)

    if "CLUSTER_PREVIEW" in os.environ:
        print("\n")
        print(f"DBSCAN Clustering - {len(metadata)} Clusters")
        print(f"Saved to {presentation_image_path} and {presentation_metadata_path}")
        print("\n")
        print(metadata)
        print("\n")
    
    return presentation_image_path, presentation_metadata_path

def get_artifact_id(video_path, start_frame, end_frame, framerate, cluster_labels):
    video_name = os.path.basename(video_path)

    start_minutes = int(start_frame / framerate // 60)
    start_seconds = int(start_frame / framerate % 60)
    start_milliseconds = int((start_frame / framerate % 60 - start_seconds) * 1000)

    end_minutes = int(end_frame / framerate // 60)
    end_seconds = int(end_frame / framerate % 60)
    end_milliseconds = int((end_frame / framerate % 60 - end_seconds) * 1000)

    start_formatted = f"{start_minutes:02d}-{start_seconds:02d}.{start_milliseconds:03d}"
    end_formatted = f"{end_minutes:02d}-{end_seconds:02d}.{end_milliseconds:03d}"

    presentation_file_stem = "_".join([video_name, start_formatted, end_formatted] + cluster_labels)

    return presentation_file_stem

def load_presentation(presentation_path):
    presentation_image_path = f"{presentation_path}.png"
    presentation_metadata_path = f"{presentation_path}.txt"

    if os.path.exists(presentation_image_path) and os.path.exists(presentation_metadata_path):
        image_bytes = open(presentation_image_path, 'rb').read()
        metadata = open(presentation_metadata_path, 'r').read()
        
        return image_bytes, metadata
    else:
        print(f"Could not find presentation at {presentation_path}")
        return None, None

def save_presentation(presentation_path, image_bytes, metadata):
    presentations_dir = os.path.dirname(presentation_path)

    presentation_image_path = f"{presentation_path}.png"
    presentation_metadata_path = f"{presentation_path}.txt"

    if not os.path.exists(presentations_dir):
        os.makedirs(presentations_dir)
    
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image.save(presentation_image_path)

    with open(presentation_metadata_path, 'w') as f:
        f.write(metadata)

    return presentation_image_path, presentation_metadata_path


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

metadata_type = List[Tuple[str, float, float, float, int, int, int, Color]]

import hashlib

def present_section(bg_image, clustered_detections: dict[int, list[Detections]], ignore_noise=True):
    metadata: metadata_type = []

    if bg_image is None:
        bg_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    else:
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)

        # reduce the whole bg image saturation
        bg_image_hsv = cv2.cvtColor(bg_image, cv2.COLOR_BGR2HSV)
        bg_image_hsv[:, :, 1] = bg_image_hsv[:, :, 1] * 0.6
        bg_image = cv2.cvtColor(bg_image_hsv, cv2.COLOR_HSV2BGR)
    
    image_as_bytes = bg_image.tobytes()
    cluster_keys = list(clustered_detections.keys())
    cluster_keys = [key % 256 for key in cluster_keys]
    keys_as_bytes = bytes(cluster_keys)

    hash_object = hashlib.sha256(image_as_bytes + keys_as_bytes)
    hash_hex = hash_object.hexdigest()
    seed_value = int(hash_hex, 16) % (2 ** 32)
    random.seed(seed_value)

    colors_shuffled = copy.deepcopy(colors)
    random.shuffle(colors_shuffled)

    for idx, (label, cluster) in enumerate(clustered_detections.items()):
        if ignore_noise and label == -1:
            continue

        cluster_data = [((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) for frame_detections in cluster for xyxy, *_ in frame_detections]
        cluster_data = np.array(cluster_data)

        box_label = chr(label + 65)
        cluster_mean = np.mean(cluster_data, axis=0)
        centroid_x = cluster_mean[0]
        centroid_y = cluster_mean[1]
        median_size = cluster_mean[2]
        cluster_size = len(cluster_data)

        color = colors_shuffled[idx % len(colors_shuffled)]

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

        metadata.append((box_label, centroid_x, centroid_y, median_size, cluster_size, start_ms, end_ms, color))

        # ========== Plotting the cluster ============
        for x, y, area in cluster_data:
            # add circle, radius is proportional to the area of the bounding box
            # cv2.circle(bg_image, (int(x), int(y)), int(np.sqrt(area) / 2), color["rgb"][::-1], -1)
            # add square
            radius = int(np.sqrt(area) / 2)
            padding = 2
            tlx, tly = int(x - radius - padding), int(y - radius - padding)
            brx, bry = int(x + radius + padding), int(y + radius + padding)

            cv2.rectangle(bg_image, (tlx, tly), (brx, bry), color.rgb[::-1], 2)

        frame_centers = []

        # find the center of the every frame
        for idx in range(cluster_start_frame, cluster_end_frame + 1):
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
            print("No cluster angles for cluster ", label)
        else:
            median_angle = np.median(cluster_angles)
            weights = 1 / (np.abs(cluster_angles - median_angle) + 1e-10)
            average_angle = np.average(cluster_angles, weights=weights)

        # ========== Plotting the label box ============
        image_height, image_width, _ = bg_image.shape

        text_content = f"{box_label}"
        text_font = cv2.FONT_HERSHEY_DUPLEX
        text_font_scale = 0.8
        text_font_thickness = 1

        box_padding = 7
        box_alpha = 0.8

        text_size, _ = cv2.getTextSize(text_content, text_font, text_font_scale, text_font_thickness)
        box_size = (text_size[0] + box_padding * 2, text_size[1] + box_padding * 2)
        
        box_color = (0,0,0)
        text_color = color.rgb[::-1]

        box_tlx, box_tly = find_place_for_box(cluster, box_size[1], box_size[0], image_height, image_width)
        box_brx, box_bry = box_tlx + box_size[0], box_tly + box_size[1]

        # clip to image bounds
        box_tlx = max(0, box_tlx)
        box_tly = max(0, box_tly)
        box_brx = min(bg_image.shape[1], box_brx)
        box_bry = min(bg_image.shape[0], box_bry)

        # HACK create a detection object for the label box, so that it can be merged with the rest of the cluster
        cluster_duplicate = [copy.copy(frame_detections) for frame_detections in cluster]
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

    image_bytes = cv2.imencode('.png', bg_image)[1].tobytes()
    presented_metadata = present_metadata(metadata)

    return image_bytes, presented_metadata

def present_metadata(metadata: metadata_type):
    presented_metadata = []
    headers = "id, color"

    for box_label, _, _, _, _, _, _, color in metadata:
        presented_metadata.append(f"{box_label}, {color.name}")

    presented_metadata = "\n".join(presented_metadata)
    presented_metadata = f"{headers}\n{presented_metadata}"

    return presented_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking', default="declustered")
    parser.add_argument('remainder', nargs=argparse.REMAINDER, help='All other arguments')
    args = parser.parse_args()

    video_path_list = []

    if args.remainder:
        for arg in args.remainder:
            if os.path.isdir(arg):
                video_path_list.extend([os.path.join(arg, file) for file in os.listdir(arg) if file.endswith(".mp4")])
            elif os.path.isfile(arg) and arg.endswith(".mp4"):
                video_path_list.append(arg)

    video_path_list = sorted(video_path_list)

    if len(video_path_list) > 1:
        for video_path in video_path_list:
            if args.tracking == "vacocam":
                _, loaded_detections = load_video_detections(video_path, module="track", version="declustered")
            else:
                _, loaded_detections = load_video_detections(video_path, module="detect")

            track_video(video_path, loaded_detections, tracking=args.tracking)
    else:
        if args.tracking == "vacocam":
            _, loaded_detections = load_video_detections(video_path_list[0], module="track", version="declustered")
        else:
            _, loaded_detections = load_video_detections(video_path_list[0], module="detect")

        track_video(video_path_list[0], loaded_detections, tracking=args.tracking)