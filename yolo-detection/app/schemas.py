"""Kiểu dữ liệu trao đổi giữa các lớp và định dạng phản hồi cho Bambuddy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """Một bounding box do model phát hiện.

    Toạ độ theo hệ pixel gốc của ảnh: (x, y) là góc trên-trái, (w, h) là kích
    thước box. `confidence` trong khoảng [0, 1].
    """

    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    def to_obico(self) -> list:
        """Chuyển sang định dạng mảng mà Bambuddy đọc: [label, conf, [x, y, w, h]].

        Xem `score_from_detections` trong backend Bambuddy — nó chỉ lấy cột
        confidence (chỉ số 1) nhưng vẫn cần đủ cấu trúc box để không lỗi parse.
        """
        return [
            self.label,
            round(self.confidence, 4),
            [round(self.x, 1), round(self.y, 1), round(self.width, 1), round(self.height, 1)],
        ]


def to_response(detections: list[Detection]) -> dict:
    """Đóng gói danh sách Detection thành payload JSON kiểu Obico."""
    return {"detections": [d.to_obico() for d in detections]}
