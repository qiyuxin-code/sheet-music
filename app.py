#!/usr/bin/env python3
"""简谱标注器 Web 服务（FastAPI）：批量上传图片 → 叠印简谱 → 打包下载。

运行:
    python app.py            # 或 uvicorn app:app --host 0.0.0.0 --port 8000
"""

from env_local import _load_local_env

_load_local_env()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
)

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

import annotate as an

MAX_WORKERS = 4
THUMB_MAX = 480
HTML_PATH = Path(__file__).resolve().parent / "index.html"
DEBUG_ROOT = Path(__file__).resolve().parent / "debug"

app = FastAPI(title="简谱批量标注", description="批量上传五线谱图片，自动叠印简谱数字")

# 微信小程序 / 本地 H5 调试可配置允许来源（生产请在网关层限制）
_cors = os.environ.get("CORS_ALLOW_ORIGINS", "*")
if _cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


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


def debug_dir_for_upload(filename: str, root: Path | None = None) -> Path:
    """每个上传文件对应 debug/<文件名>/，避免互相覆盖。"""
    if root is None:
        root = DEBUG_ROOT
    stem = Path(filename).name
    stem = Path(stem).stem or "image"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    return root / (safe or "image")


def process_one(
    data: bytes,
    filename: str = "image",
    debug_root: Path | None = None,
) -> dict:
    gray, binary = decode_image(data)
    vis, scene = an.process_image_with_scene(
        gray,
        binary,
        quiet=True,
        use_jev=None,
        debug_dir=debug_dir_for_upload(filename, debug_root),
    )
    return {
        "thumb": _encode_b64(vis, THUMB_MAX),
        "img": _encode_b64(vis),
        "scene": scene,
    }


@app.post("/api/annotate")
async def annotate_one(file: UploadFile = File(...)):
    """单张图片标注（微信小程序 wx.uploadFile 使用，表单字段名 file）。"""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    name = file.filename or "image"
    try:
        result = process_one(data, filename=name)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "filename": name,
        "status": "ok",
        "thumb": result["thumb"],
        "img": result["img"],
        "scene": result["scene"],
    }


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
            result = process_one(data, filename=name)
            return {
                "filename": name,
                "status": "ok",
                "thumb": result["thumb"],
                "img": result["img"],
                "scene": result["scene"],
            }
        except Exception as e:  # noqa: BLE001
            return {"filename": name, "status": "error", "error": str(e)}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(raw))) as ex:
        results = list(ex.map(_work, raw))

    return JSONResponse({"results": results})


@app.get("/canvas_render.js")
async def canvas_render_js():
    js_path = Path(__file__).resolve().parent / "canvas_render.js"
    if not js_path.is_file():
        raise HTTPException(status_code=404, detail="canvas_render.js missing")
    return Response(js_path.read_text(encoding="utf-8"), media_type="application/javascript")


@app.get("/canvas_editor.js")
async def canvas_editor_js():
    js_path = Path(__file__).resolve().parent / "canvas_editor.js"
    if not js_path.is_file():
        raise HTTPException(status_code=404, detail="canvas_editor.js missing")
    return Response(js_path.read_text(encoding="utf-8"), media_type="application/javascript")


@app.get("/ui.css")
async def ui_styles():
    return Response((HTML_PATH.parent / "ui.css").read_text(encoding="utf-8"), media_type="text/css")


@app.get("/", response_class=HTMLResponse)
async def index():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>简谱批量标注</h1><p>上传 /api/batch</p>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/json/version")
async def json_version_probe():
    """浏览器/IDE 探测用，避免终端刷 404。"""
    return {"version": "sheet-music-annotate"}


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_probe():
    return {}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
