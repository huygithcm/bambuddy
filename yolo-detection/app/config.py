"""Cấu hình runtime cho sidecar YOLO.

Mọi tham số đọc từ biến môi trường (xem .env.example) nên container không cần
sửa code. `Settings` là một singleton bất biến được nạp một lần lúc khởi động.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Tham số cấu hình được nạp từ môi trường / file .env."""

    model_config = SettingsConfigDict(
        env_prefix="YOLO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Đường dẫn tới file trọng số YOLO (.pt / .onnx / .engine).
    model_path: str = Field(default="models/best.pt")

    # Ngưỡng confidence tối thiểu để giữ lại một box (lọc nhiễu ngay tại nguồn).
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    # Ngưỡng IoU cho NMS.
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    # Kích thước cạnh ảnh đưa vào model (px).
    image_size: int = Field(default=640, gt=0)

    # Thiết bị suy luận: "cpu", "cuda", "cuda:0"...
    device: str = Field(default="cpu")

    # Nhãn Bambuddy gán cho mỗi detection (model 1-class "failure").
    detection_label: str = Field(default="failure")

    # Thời gian tối đa (giây) để tải ảnh snapshot từ URL Bambuddy cung cấp.
    fetch_timeout: float = Field(default=5.0, gt=0)

    # Địa chỉ / cổng HTTP service lắng nghe.
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=3333, gt=0, lt=65536)


@lru_cache
def get_settings() -> Settings:
    """Trả về singleton Settings (nạp một lần, cache lại)."""
    return Settings()
