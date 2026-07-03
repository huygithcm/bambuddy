"""Tầng HTTP (FastAPI) — expose endpoint tương thích Obico cho Bambuddy.

    GET /hc/            -> "ok"       (Bambuddy dùng cho nút Test Connection)
    GET /p/?img=<URL>   -> {"detections": [...]}

Model YOLO được nạp một lần trong lifespan và tái sử dụng cho mọi request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import get_settings
from app.image_fetcher import ImageFetchError
from app.schemas import to_response
from app.service import DetectionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Giữ tham chiếu service ở phạm vi app.state để tránh biến toàn cục rải rác.


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Khởi động sidecar YOLO trên %s:%d", settings.host, settings.port)
    app.state.service = DetectionService.from_settings(settings)
    yield


app = FastAPI(title="Bambuddy YOLO Detection Sidecar", lifespan=lifespan)


@app.get("/hc/", response_class=PlainTextResponse)
async def health_check() -> str:
    """Health check — Bambuddy chờ đúng chuỗi 'ok' với HTTP 200."""
    return "ok"


@app.get("/p/")
async def predict(img: str = Query(..., description="URL ảnh snapshot cần phân tích")):
    """Tải ảnh từ `img`, chạy YOLO, trả detections theo định dạng Obico."""
    service: DetectionService = app.state.service
    try:
        detections = await service.detect_from_url(img)
    except ImageFetchError as exc:
        logger.warning("Bỏ qua frame: %s", exc)
        # Trả mảng rỗng thay vì lỗi 5xx: một frame hỏng không nên làm Bambuddy
        # coi cả vòng poll là thất bại — nó chỉ đơn giản ghi nhận "an toàn".
        return JSONResponse(to_response([]))
    return JSONResponse(to_response(detections))
