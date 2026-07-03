"""Unit test luồng điều phối, không cần nạp model YOLO thật."""

import asyncio

import numpy as np

from app.schemas import Detection, to_response
from app.service import DetectionService


class FakeFetcher:
    """Trả về một ảnh đen cố định, bỏ qua HTTP thật."""

    async def fetch(self, url: str) -> np.ndarray:
        return np.zeros((640, 640, 3), dtype=np.uint8)


class FakeDetector:
    """Trả về một detection cố định để kiểm tra luồng."""

    def predict(self, image: np.ndarray):
        return [Detection(label="failure", confidence=0.9, x=1, y=2, width=3, height=4)]


def test_detect_from_url_returns_detections():
    service = DetectionService(detector=FakeDetector(), fetcher=FakeFetcher())
    detections = asyncio.run(service.detect_from_url("http://example/frame.jpg"))
    assert len(detections) == 1
    assert detections[0].confidence == 0.9


def test_obico_response_shape():
    dets = [Detection(label="failure", confidence=0.876, x=10, y=20, width=30, height=40)]
    payload = to_response(dets)
    # Bambuddy đọc đúng cấu trúc [label, conf, [x, y, w, h]]
    assert payload == {"detections": [["failure", 0.876, [10.0, 20.0, 30.0, 40.0]]]}
