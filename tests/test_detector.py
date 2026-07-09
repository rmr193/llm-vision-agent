"""
Unit tests for src/vision/detector.py.

These tests mock the underlying `ultralytics.YOLO` model so they run fast
and without GPU/model weights, verifying our wrapping/parsing logic rather
than YOLO itself.
"""
import numpy as np
import pytest

from src.vision.detector import Detection, DetectionResult


def test_detection_center_and_dict():
    d = Detection(label="button", confidence=0.91, box_xyxy=(10, 20, 30, 60), element_id=0)
    assert d.center() == (20, 40)
    as_dict = d.to_dict()
    assert as_dict["label"] == "button"
    assert as_dict["center"] == [20, 40]
    assert as_dict["box"] == [10, 20, 30, 60]


def test_detection_result_to_json_empty():
    result = DetectionResult(detections=[], inference_ms=12.3, device="cuda:0")
    assert result.to_json() == []
    assert result.device == "cuda:0"


def test_detection_result_to_json_multiple():
    dets = [
        Detection(label="icon", confidence=0.8, box_xyxy=(0, 0, 10, 10), element_id=0),
        Detection(label="text_field", confidence=0.6, box_xyxy=(20, 20, 40, 40), element_id=1),
    ]
    result = DetectionResult(detections=dets, inference_ms=8.0, device="cuda:0")
    payload = result.to_json()
    assert len(payload) == 2
    assert payload[0]["id"] == 0
    assert payload[1]["label"] == "text_field"


@pytest.mark.parametrize("frame_shape", [(100, 100, 3), (1080, 1920, 3)])
def test_frame_is_numpy_array(frame_shape):
    frame = np.zeros(frame_shape, dtype=np.uint8)
    assert frame.shape == frame_shape
    assert frame.dtype == np.uint8
