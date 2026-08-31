#!/usr/bin/env python3
"""简谱标注器 Web 服务（FastAPI）：批量上传图片 → 叠印简谱 → 打包下载。

运行:
    python app.py            # 或 uvicorn app:app --host 0.0.0.0 --port 8000
"""

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import annotate as an

MAX_WORKERS = 4
THUMB_MAX = 480
HTML_PATH = Path(__file__).resolve().parent / "index.html"

app = FastAPI(title="简谱批量标注", description="批量上传五线谱图片，自动叠印简谱数字")


def decode_image(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("无法解析图片（格式不支持或文件损坏）")
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return gray, binary


def _encode_b64(vis: np.ndarray, max_dim: int | None = None) -> str:
    img = vis
    if max_dim:
        h, w = vis.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            img = cv2.resize(vis, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("结果编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def process_one(data: bytes) -> dict:
    gray, binary = decode_image(data)
    vis = an.process_image(gray, binary, quiet=True)
    return {"thumb": _encode_b64(vis, THUMB_MAX), "img": _encode_b64(vis)}


@app.post("/api/batch")
async def batch_upload(files: list[UploadFile] = File(...)):
    """批量处理上传的图片，返回每张的标注结果（含缩略图与原图 dataURL）。"""
    if not files:
        raise HTTPException(status_code=400, detail="未收到任何文件")
    if len(files) > 200:
        raise HTTPException(status_code=400, detail="单次最多上传 200 个文件")

    raw = []
    for f in files:
        raw.append((f.filename or "image", await f.read()))

    def _work(item):
        name, data = item
        if not data:
            return {"filename": name, "status": "error", "error": "空文件"}
        try:
            result = process_one(data)
            return {"filename": name, "status": "ok",
                    "thumb": result["thumb"], "img": result["img"]}
        except Exception as e:  # noqa: BLE001
            return {"filename": name, "status": "error", "error": str(e)}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(raw))) as ex:
        results = list(ex.map(_work, raw))

    return JSONResponse({"results": results})


@app.get("/", response_class=HTMLResponse)
async def index():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>简谱批量标注</h1><p>上传 /api/batch</p>")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
