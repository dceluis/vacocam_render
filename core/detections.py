from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
import os
from pathlib import Path
import supervision as sv

from typing import List, Any

class Detections(sv.Detections):
    point_of_interest: Optional[Tuple[int, int]] = None
    area_of_interest: Optional[int] = None

    def to_structured_array(self):
        """
        Converts the Detections object to a numpy structured array of shape `(N, 3)`.
        """

        dtype = [('bbox', float, (4,)), ('confidence', float), ('class', int)]

        structured_array = np.zeros(len(self.xyxy), dtype=dtype)
        for i in range(len(self.xyxy)):
            structured_array[i] = (self.xyxy[i], self.confidence[i] if self.confidence is not None else 0, self.class_id[i] if self.class_id is not None else 0)
        return structured_array
    
    # from structured array, class method
    @classmethod
    def from_structured_array(cls, structured_array):
        bboxes = np.zeros((len(structured_array["bbox"]), 4))
        confidences = []
        classes = []
        for i in range(len(structured_array)):
            bboxes[i] = structured_array["bbox"][i]
            confidences.append(structured_array["confidence"][i])
            classes.append(structured_array["class"][i])

        bboxes = np.array(bboxes)
        confidences = np.array(confidences)
        classes = np.array(classes)

        return cls(bboxes, None, confidences, classes)
    
    @classmethod
    def from_single_tuple(cls, tuple):
        # do not use from_structured_array, as it will not work with a single tuple
        bboxes = np.zeros((1, 4))
        mask = None
        confidences = []
        classes = []
        tracker_ids = []

        bboxes[0] = tuple[0]
        mask = tuple[1]
        confidences.append(tuple[2])
        classes.append(tuple[3])
        tracker_ids.append(tuple[4])

        bboxes = np.array(bboxes)
        confidences = np.array(confidences)
        classes = np.array(classes)
        tracker_ids = np.array(tracker_ids)

        return cls(bboxes, mask, confidences, classes, tracker_ids)


    @classmethod
    def from_sahi_batched(cls, object_prediction_list):
        bboxes = np.zeros((len(object_prediction_list), 4)) 
        confidences = []
        class_ids = []

        for idx, result in enumerate(object_prediction_list):
            bboxes[idx] = result.bbox.to_xyxy()
            confidences.append(result.score.value)
            class_ids.append(result.category.id)
        
        confidences = np.array(confidences)
        class_ids = np.array(class_ids)

        return cls(bboxes, None, confidences, class_ids, None)
    

    @classmethod
    def merge(cls, detections_list: List[Detections]) -> Detections:
        sv_detections: List[sv.Detections] = []

        for detections in detections_list:
            if isinstance(detections, Detections):
                sv_detections.append(sv.Detections(detections.xyxy, detections.mask, detections.confidence, detections.class_id, detections.tracker_id))
            else:
                sv_detections.append(detections)
        
        merged_detections = sv.Detections.merge(sv_detections)

        return cls(merged_detections.xyxy, merged_detections.mask, merged_detections.confidence, merged_detections.class_id, merged_detections.tracker_id)

    @classmethod
    def empty(cls) -> Detections:
        super_empty = super().empty()
        return cls(super_empty.xyxy, super_empty.mask, super_empty.confidence, super_empty.class_id, super_empty.tracker_id)

    def __copy__(self):
        return Detections(self.xyxy, self.mask, self.confidence, self.class_id, self.tracker_id)
    
#################
# Detections IO #
#################

def load_detections(detections_path):
    if os.path.exists(detections_path):
        loaded_detections = np.load(detections_path, allow_pickle=True)
        loaded_detections = [Detections.from_structured_array(frame_detections) for frame_detections in loaded_detections]
    
        print(f"loaded {len(loaded_detections)} detections from {detections_path}")
        return (True, loaded_detections)
    else:
        print(f"detections path {detections_path} does not exist")
        return (False, [])

def save_detections(detections_list: List[Detections], detections_path):
    detections_array = [detections.to_structured_array() for detections in detections_list]
    detections_array = np.array(detections_array, dtype=object)
    np.save(detections_path, detections_array, allow_pickle=True)

    return (True, detections_path)

def save_video_detections(video_detections, video_path, module="detections", version=None):
    video_dir, video_name = os.path.split(video_path)
    video_dir = video_dir or os.getcwd()

    assert video_name, f"{video_path} does not have a valid file name"

    video_name_stem = Path(video_name).stem

    detections_name = "_".join([item for item in [video_name_stem, "detections", version] if item is not None]) + ".npy"
    detections_dir = os.path.join(video_dir, module, "")
    detections_path = os.path.join(detections_dir, detections_name)

    Path(detections_dir).mkdir(parents=True, exist_ok=True)

    return save_detections(video_detections, detections_path)

def load_video_detections(video_path, module="detections", version=None):
    video_dir, video_name = os.path.split(video_path)
    video_dir = video_dir or os.getcwd()

    assert video_name, f"{video_path} does not have a valid file name"

    video_name_stem = Path(video_name).stem

    detections_name = "_".join([item for item in [video_name_stem, "detections", version] if item is not None]) + ".npy"
    detections_dir = os.path.join(video_dir, module, "")
    detections_path = os.path.join(detections_dir, detections_name)

    print(f"loading detections from {detections_path}")
    return load_detections(detections_path)