"""Tải và giải mã ảnh snapshot từ URL do Bambuddy cung cấp."""

from __future__ import annotations

import logging

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class ImageFetchError(RuntimeError):
    """Không tải được hoặc không giải mã được ảnh từ URL."""


class ImageFetcher:
    """Tải bytes ảnh qua HTTP rồi giải mã thành mảng BGR của OpenCV.

    Bambuddy đưa cho service một URL dạng
    `{external_url}/api/v1/obico/cached-frame/<nonce>` phục vụ JPEG một lần.
    URL này chỉ sống ~30s nên cần tải ngay với timeout ngắn.
    """

    def __init__(self, timeout: float) -> None:
        self._timeout = timeout

    async def fetch(self, url: str) -> np.ndarray:
        """Trả về ảnh dạng ndarray BGR. Ném ImageFetchError nếu thất bại."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.content
        except httpx.HTTPError as exc:
            raise ImageFetchError(f"Không tải được ảnh từ {url}: {exc}") from exc

        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageFetchError(f"Không giải mã được ảnh từ {url} ({len(data)} bytes)")
        return image
