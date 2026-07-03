FROM python:3.12-slim

# OpenCV cần libGL và libglib runtime ngay cả với bản headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Trọng số model được mount vào lúc chạy (xem docker-compose.yml)
ENV YOLO_MODEL_PATH=/models/best.pt \
    YOLO_HOST=0.0.0.0 \
    YOLO_PORT=3333

EXPOSE 3333

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:3333/hc/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3333"]
