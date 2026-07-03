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
        self._model = None  # nạp lười ở lần predict đầu tiên

    def _ensure_loaded(self):
        """Nạp model một lần, ở lần suy luận đầu tiên.

        Nạp lười giúp `/hc/` (health check) trả lời ngay cả khi file trọng số
        chưa được đặt vào, nên container không crash lúc khởi động — Bambuddy
        vẫn Test Connection thành công trong khi bạn còn đang chuẩn bị model.
        """
        if self._model is not None:
            return self._model
        # Import trễ để việc thiếu ultralytics chỉ lỗi khi thực sự cần model,
        # giúp health-check và unit test không bắt buộc phải cài torch.
        from ultralytics import YOLO

        s = self._settings
        logger.info("Đang nạp model YOLO từ %s (device=%s)", s.model_path, s.device)
        model = YOLO(s.model_path)
        model.to(s.device)
        self._model = model
        return model

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Chạy model trên một ảnh BGR, trả về danh sách Detection đã lọc ngưỡng."""
        s = self._settings
        model = self._ensure_loaded()
        result = model.predict(
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
