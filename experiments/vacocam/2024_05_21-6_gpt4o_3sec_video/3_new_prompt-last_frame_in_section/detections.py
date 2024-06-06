from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
import os
from pathlib import Path
from dataclasses import dataclass
import supervision as sv

from typing import List, Any

@dataclass
class Detections(sv.Detections):
    """
    Detections class, a subclass of supervision.Detections, with additional methods for IO, merging and presentation.
    """

    cluster_id: np.ndarray = np.array([], dtype=int)
    point_of_interest: Optional[Tuple[int, int]] = None
    area_of_interest: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()

        if self.cluster_id.size == 0 and self.xyxy.size > 0:
            # an aray of -1s for len(self.xyxy)
            self.cluster_id = np.full(len(self.xyxy), -1, dtype=int)

    def to_split_list(self) -> List[Detections]:
        detections_list = [
            Detections(
                xyxy=self.xyxy[i].reshape(1, 4),
                mask=self.mask[i] if self.mask is not None else None,
                confidence=self.confidence[i].reshape(1) if self.confidence is not None else None,
                class_id=self.class_id[i].reshape(1) if self.class_id is not None else None,
                tracker_id=self.tracker_id[i].reshape(1) if self.tracker_id is not None else None,
                cluster_id=self.cluster_id[i].reshape(1),
                point_of_interest=self.point_of_interest if self.point_of_interest else None,
                area_of_interest=self.area_of_interest if self.area_of_interest else None
            ) for i in range(len(self.xyxy))
        ]

        return detections_list

    def to_structured_array(self):
        """
        Converts the Detections object to a numpy structured array of shape `(N, 3)`.
        """

        dtype = [('bbox', float, (4,)), ('confidence', float), ('class', int), ('cluster', int)]

        structured_array = np.zeros(len(self.xyxy), dtype=dtype)
        for i in range(len(self.xyxy)):
            xyxy_r = self.xyxy[i]
            confidence_r = self.confidence[i] if self.confidence is not None else 0
            class_id_r = self.class_id[i] if self.class_id is not None else 0
            cluster_r = self.cluster_id[i]

            structured_array[i] = (xyxy_r, confidence_r, class_id_r, cluster_r)
        return structured_array
    
    # from structured array, class method
    @classmethod
    def from_structured_array(cls, structured_array):
        bboxes = np.zeros((len(structured_array["bbox"]), 4))
        confidences = []
        classes = []
        clusters = []
        
        for i in range(len(structured_array)):
            bboxes[i] = structured_array["bbox"][i]
            confidences.append(structured_array["confidence"][i])
            classes.append(structured_array["class"][i])
            
            if "cluster" in structured_array.dtype.names:
                clusters.append(structured_array["cluster"][i])

        confidences = np.array(confidences)
        classes = np.array(classes)
        clusters = np.array(clusters)

        return cls(bboxes, None, confidences, classes, cluster_id=clusters)
    
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

        return cls(
            xyxy=bboxes,
            confidence=confidences,
            class_id=class_ids
        )
    
    @classmethod
    def merge(cls, detections_list: List[Detections]) -> Detections:
        if len(detections_list) == 0:
            return cls.empty()

        attributes = sv.Detections.__annotations__.keys()
        sv_detections = [sv.Detections(**{attr: getattr(dets, attr) for attr in attributes}) for dets in detections_list]
        sv_merged = sv.Detections.merge(sv_detections)
        sv_attributes = {attr: getattr(sv_merged, attr) for attr in attributes}

        sv_attributes["cluster_id"] = np.hstack([dets.cluster_id for dets in detections_list])
        
        ## point of interest and area of interest are reset to None

        return cls(**sv_attributes)

    @classmethod
    def empty(cls) -> Detections:
        sv_empty = sv.Detections.empty().__dict__
        sv_empty["cluster_id"] = np.array([], dtype=int)
        return cls(**sv_empty)
    
    def __copy__(self):
        return Detections(self.xyxy, self.mask, self.confidence, self.class_id, self.tracker_id, cluster_id=self.cluster_id, point_of_interest=self.point_of_interest, area_of_interest=self.area_of_interest)
    
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