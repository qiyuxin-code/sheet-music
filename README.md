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
pip install -r requirements.txt
```

可选：配置 [Jev](https://console.typesafe.ai)（TypeSafe System One）后，渲染阶段可辅助判断是否属于行首曲谱头（谱号、调号、拍号等）。标注方向始终遵守大谱表上下行规则，不允许模型改到相反一侧。

```bash
export TYPESAFE_API_KEY="apikey_..."   # 勿提交到 git
python annotate.py score.jpg --jev      # 显式启用；已配置 Key 时默认也会启用
python annotate.py score.jpg --no-jev   # 强制仅用 OpenCV 规则
```

## 使用

### Web 批量上传（推荐）

```bash
python app.py                     # 启动后打开 http://127.0.0.1:8000
# 或指定端口：
uvicorn app:app --host 0.0.0.0 --port 8000
```

浏览器打开后批量选择/拖拽图片，一键标注并平铺展示结果。
也可直接调用接口：

```bash
curl -F "files=@a.jpg" -F "files=@b.png" http://127.0.0.1:8000/api/batch
curl -F "file=@a.jpg" http://127.0.0.1:8000/api/annotate
```

### 微信小程序

1. 启动后端：`python app.py`
2. 用 [微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html) 打开本仓库下的 `miniprogram/` 目录
3. 在 `miniprogram/config.js` 中把 `apiBase` 改为你的服务地址（真机调试请用电脑局域网 IP，如 `http://192.168.x.x:8000`）
4. 开发者工具 → 详情 → 本地设置 → 勾选 **不校验合法域名**（仅开发环境）

小程序通过 `POST /api/annotate` 逐张上传图片，展示缩略图并支持预览、保存到相册。

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
app.py        FastAPI Web 服务（/api/batch、/api/annotate）
index.html    Web 上传页面
miniprogram/  微信小程序客户端
debug/        调试图输出（已 gitignore）
.venv/        Python 虚拟环境（已 gitignore）
```

## 输出示例

回归检查：`python -m unittest discover -s tests`；前端状态与小程序预览检查：`node tests/test_frontend.js`。`tests/test_score_feedback.py` 包含用户反馈谱面的二值裁剪，覆盖白格误检及下加线音符漏检。

谱头过滤支持浅灰谱线，并额外排除行首高音谱号的长连通字形。叠印找不到附近空白位置时会跳过该标注，避免盖住原谱。Web 与小程序使用同一识别流程，上传的逐阶段调试图位于 `debug/<文件名>/`。

谱号识别覆盖行首与正文中的高低音谱号变化，音高按当前谱号计算。花括号连接的谱表按整体分组：上行蓝字放在音符及谱表上方，下行绿字放在整条下谱表下方；颜色按行的位置决定，不随换谱号改变。和弦按音高整列排列，每颗符头只标一次，逐列避让，不强制一行统一基线。空间紧张时缩小字号，仍无安全空位才跳过。相接符头只在存在两个形状峰和收窄处时拆分，避免一音多标。

网页大图可用左右按钮或键盘方向键切换，滚轮/双指缩放、拖拽查看，返回上一张时保留其缩放位置。小程序使用微信原生多图预览，支持滑动切换、双指缩放和放大后拖动。处理过程中显示 loading 动画。

运行后输出图片，终端打印简谱分布统计（前 20）。
