"""Sidecar phát hiện lỗi in bằng YOLO cho Bambuddy.

Bọc một model YOLO custom thành một HTTP service tương thích với giao thức
ML API kiểu Obico mà Bambuddy mong đợi:

    GET /hc/              -> "ok"                         (health check)
    GET /p/?img=<URL>     -> {"detections": [[label, conf, [x, y, w, h]], ...]}

Bambuddy cộng tổng confidence của các detection rồi làm mượt theo thời gian,
nên service chỉ cần trả về các bounding box kèm độ tin cậy.
"""

__version__ = "0.1.0"
