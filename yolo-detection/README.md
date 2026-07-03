# Bambuddy — Sidecar phát hiện lỗi in bằng YOLO

Bọc **model YOLO custom** của bạn thành một HTTP service tương thích với ML API
kiểu Obico mà Bambuddy đã hỗ trợ sẵn. **Không cần sửa Bambuddy** — chỉ chạy
sidecar này rồi trỏ URL trong Settings.

## Giao thức

Bambuddy gọi service qua đúng 2 endpoint:

| Endpoint | Mô tả |
|----------|-------|
| `GET /hc/` | Health check — trả về `ok` (dùng cho nút **Test Connection**) |
| `GET /p/?img=<URL>` | Tải ảnh từ `<URL>`, chạy YOLO, trả `{"detections": [[label, conf, [x,y,w,h]], ...]}` |

Bambuddy cộng tổng **confidence** của các box rồi làm mượt theo thời gian
(EWM + rolling mean, warmup 30 frame) trước khi kích hoạt hành động. Xem
`backend/app/services/obico_smoothing.py` trong repo chính.

## Cấu trúc code (hướng đối tượng)

```
yolo-detection/
├── app/
│   ├── config.py         # Settings — nạp cấu hình từ biến môi trường
│   ├── schemas.py        # Detection + chuyển sang định dạng Obico
│   ├── image_fetcher.py  # ImageFetcher — tải & giải mã ảnh từ URL
│   ├── detector.py       # YoloDetector — nạp model & suy luận
│   ├── service.py        # DetectionService — điều phối fetch + detect
│   └── main.py           # Tầng HTTP FastAPI (/hc/, /p/)
├── tests/                # Unit test luồng (không cần model thật)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

Mỗi lớp một trách nhiệm, nối với nhau qua dependency injection (`DetectionService`
nhận `detector` + `fetcher`), nên dễ test và dễ thay thế backend model.

## Chạy nhanh (Docker)

```bash
cd yolo-detection/
mkdir -p models
cp /đường/dẫn/best.pt models/best.pt   # đặt trọng số model của bạn vào đây
cp .env.example .env                    # (tuỳ chọn) chỉnh tham số
docker compose up -d --build
```

Kiểm tra: `curl http://localhost:3333/hc/` → phải trả về `ok`.

## Chạy trực tiếp (dev, không Docker)

```bash
cd yolo-detection/
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export YOLO_MODEL_PATH=models/best.pt               # Windows: set YOLO_MODEL_PATH=...
uvicorn app.main:app --host 0.0.0.0 --port 3333
```

## Khai báo trong Bambuddy

1. **Settings → General → External URL**: đặt địa chỉ Bambuddy mà sidecar truy
   cập được (sidecar tải ảnh snapshot từ URL này). Nếu chạy khác máy, dùng IP
   LAN chứ không phải `localhost`.
2. **Settings → Notifications → AI Print-Failure Detection**:
   - **ML URL** = `http://<host-chạy-yolo>:3333`
   - Bấm **Test Connection** (mong đợi `ok`).
   - Chọn **Sensitivity** (low/medium/high) và **Action** (notify / pause / pause-and-off).

## Tinh chỉnh

| Biến môi trường | Mặc định | Ý nghĩa |
|-----------------|----------|---------|
| `YOLO_MODEL_PATH` | `/models/best.pt` | Đường dẫn trọng số (.pt/.onnx/.engine) |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.25` | Ngưỡng lọc box tại nguồn |
| `YOLO_IOU_THRESHOLD` | `0.45` | Ngưỡng IoU cho NMS |
| `YOLO_IMAGE_SIZE` | `640` | Cạnh ảnh vào model |
| `YOLO_DEVICE` | `cpu` | `cpu` \| `cuda` \| `cuda:0` |
| `YOLO_DETECTION_LABEL` | `failure` | Nhãn gán mỗi box (chỉ để log) |

> **Lưu ý ngưỡng:** Bambuddy so *tổng* confidence với `BASE_LOW=0.38` /
> `BASE_HIGH=0.78` (nhân hệ số theo sensitivity). Nếu model của bạn hay ra
> nhiều box confidence thấp, hãy nâng `YOLO_CONFIDENCE_THRESHOLD` hoặc chọn
> sensitivity "low" để tránh báo động giả.

## Chạy test

```bash
cd yolo-detection/
pip install pytest
pytest
```
