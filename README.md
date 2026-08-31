# 简谱标注器

在五线谱图片上自动识别音符，并叠印对应的简谱数字（1-7）。

## 工作原理

流水线分 6 个阶段，每个阶段可单独运行并输出调试图：

| 阶段 | 名称 | 说明 |
| --- | --- | --- |
| 1 | 谱线检测 | `detect_staff_lines` 检测五线谱谱线，`group_staves` / `pair_systems` 组合成钢琴大谱表系统 |
| 2 | 音符头检测 | `detect_noteheads` 识别实心/空心音符头 |
| 3 | 音高换算 | `assign_pitches` 根据谱线位置换算音名、八度与升降号 |
| 4 | 时值分析 | `analyze_durations` 分析符干方向、符尾(beam)与附点 |
| 5 | 简谱换算 | `to_jianpu` 将音高与时值换算为简谱数字、点、升降号前缀 |
| 6 | 渲染叠印 | `render` 在图上叠印简谱并输出 |

## 环境要求

- Python 3.12+
- OpenCV (`cv2`)
- NumPy
- FastAPI + Uvicorn + python-multipart（Web 服务用）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy fastapi uvicorn python-multipart
```

## 使用

### Web 批量上传（推荐）

```bash
python app.py                     # 启动后打开 http://127.0.0.1:8000
# 或指定端口：
uvicorn app:app --host 0.0.0.0 --port 8000
```

浏览器打开后批量选择/拖拽图片，一键标注并打包下载 zip（含 `results.json` 处理清单）。
也可直接调用接口：

```bash
curl -F "files=@a.jpg" -F "files=@b.png" http://127.0.0.1:8000/api/batch -o result.zip
```

### 命令行（单张）

```bash
python annotate.py input.jpg                     # 输出 input_简谱标注.jpg
python annotate.py input.jpg -o output.jpg       # 指定输出路径
python annotate.py input.jpg --debug-dir debug   # 调试图输出目录（默认 debug）
python annotate.py input.jpg --stage 3           # 只运行到阶段 3（调试用）
```

## 目录结构

```
annotate.py   主程序（全部逻辑）
app.py        FastAPI Web 服务（批量上传 /api/batch）
index.html    上传页面
debug/        调试图输出（已 gitignore）
.venv/        Python 虚拟环境（已 gitignore）
```

## 输出示例

运行后输出图片，终端打印简谱分布统计（前 20）。
