"""Lớp điều phối: nối ImageFetcher với YoloDetector.

Tách khỏi tầng HTTP để có thể unit-test luồng "URL -> detections" mà không
cần dựng server FastAPI.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import Settings
from app.detector import YoloDetector
from app.image_fetcher import ImageFetcher
from app.schemas import Detection

logger = logging.getLogger(__name__)


class DetectionService:
    """Nhận URL ảnh, tải về, suy luận YOLO và trả về danh sách Detection."""

    def __init__(self, detector: YoloDetector, fetcher: ImageFetcher) -> None:
        self._detector = detector
        self._fetcher = fetcher

    async def detect_from_url(self, image_url: str) -> list[Detection]:
        """Tải ảnh từ `image_url` rồi chạy phát hiện.

        Phần suy luận YOLO là đồng bộ và nặng CPU/GPU nên được đẩy sang thread
        pool để không chặn event loop của server.
        """
        image = await self._fetcher.fetch(image_url)
        detections = await asyncio.to_thread(self._detector.predict, image)
        logger.debug("Phát hiện %d box từ %s", len(detections), image_url)
        return detections

    @classmethod
    def from_settings(cls, settings: Settings) -> "DetectionService":
        """Factory: dựng service hoàn chỉnh từ cấu hình."""
        return cls(
            detector=YoloDetector(settings),
            fetcher=ImageFetcher(timeout=settings.fetch_timeout),
        )
