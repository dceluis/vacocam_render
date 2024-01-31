import numpy as np
from typing import Optional, Iterator, Tuple
import os
from pathlib import Path
import supervision as sv

from typing import List, Any
from dataclasses import astuple

class Detections(sv.Detections):
    point_of_interest: Optional[Tuple[int, int]] = None
    area_of_interest: Optional[int] = None

    def to_structured_array(self):
        """
        Converts the Detections object to a numpy structured array of shape `(N, 3)`.
        """

        dtype = [('bbox', float, (4,)), ('confidence', float), ('class', int)]

        structured_array = np.zeros(len(self.confidence), dtype=dtype)
        for i in range(len(self.confidence)):
            structured_array[i] = (self.xyxy[i], self.confidence[i], self.class_id[i])
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
    

    ## Redeclaring so that it returns our Detections class
    @classmethod
    def merge(cls, detections_list: List["Detections"]) -> "Detections":
        """
        Merge a list of Detections objects into a single Detections object.

        This method takes a list of Detections objects and combines their
        respective fields (`xyxy`, `mask`, `confidence`, `class_id`, and `tracker_id`)
        into a single Detections object. If all elements in a field are not
        `None`, the corresponding field will be stacked.
        Otherwise, the field will be set to `None`.

        Args:
            detections_list (List[Detections]): A list of Detections objects to merge.

        Returns:
            (Detections): A single Detections object containing
                the merged data from the input list.

        Example:
            ```python
            >>> from supervision import Detections

            >>> detections_1 = Detections(...)
            >>> detections_2 = Detections(...)

            >>> merged_detections = Detections.merge([detections_1, detections_2])
            ```
        """
        if len(detections_list) == 0:
            return cls.empty()

        detections_tuples_list = [astuple(detection) for detection in detections_list]
        xyxy, mask, confidence, class_id, tracker_id = [
            list(field) for field in zip(*detections_tuples_list)
        ]

        def __all_not_none(item_list: List[Any]):
            return all(x is not None for x in item_list)

        xyxy = np.vstack(xyxy)
        mask = np.vstack(mask) if __all_not_none(mask) else None
        confidence = np.hstack(confidence) if __all_not_none(confidence) else None
        class_id = np.hstack(class_id) if __all_not_none(class_id) else None
        tracker_id = np.hstack(tracker_id) if __all_not_none(tracker_id) else None

        return cls(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
        )

    @classmethod
    def empty(cls) -> "Detections":
        """
        Create an empty Detections object with no bounding boxes,
            confidences, or class IDs.

        Returns:
            (Detections): An empty Detections object.

        Example:
            ```python
            >>> from supervision import Detections

            >>> empty_detections = Detections.empty()
            ```
        """
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )

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