"""Lớp bọc model YOLO — nạp trọng số và chạy suy luận."""

from __future__ import annotations

import logging

import numpy as np

from app.config import Settings
from app.schemas import Detection

logger = logging.getLogger(__name__)


class YoloDetector:
    """Nạp một model YOLO (ultralytics) và suy luận trên từng khung hình.

    Model được nạp một lần lúc khởi tạo (tốn kém) và tái sử dụng cho mọi request.
    Lớp này chỉ phụ thuộc `ultralytics`; muốn đổi backend khác (ONNXRuntime,
    TensorRT thuần...) chỉ cần thay phần thân của `_load` và `predict`.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = self._load(settings.model_path, settings.device)

    @staticmethod
    def _load(model_path: str, device: str):
        # Import trễ để việc thiếu ultralytics chỉ lỗi khi thực sự cần model,
        # giúp health-check và unit test không bắt buộc phải cài torch.
        from ultralytics import YOLO

        logger.info("Đang nạp model YOLO từ %s (device=%s)", model_path, device)
        model = YOLO(model_path)
        model.to(device)
        return model

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Chạy model trên một ảnh BGR, trả về danh sách Detection đã lọc ngưỡng."""
        s = self._settings
        result = self._model.predict(
            source=image,
            imgsz=s.image_size,
            conf=s.confidence_threshold,
            iou=s.iou_threshold,
            device=s.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in result.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=s.detection_label,
                    confidence=confidence,
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                )
            )
        return detections
