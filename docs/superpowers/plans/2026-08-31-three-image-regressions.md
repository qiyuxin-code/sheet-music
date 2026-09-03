# 行首、空心和弦与排版回归 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复用户图片中的行首误标、空心和弦漏标、可用上方空间未优先使用，以及固定右侧位置受阻后不继续向右搜索的问题。

**Architecture:** 行首过滤从谱线图像估算每个 Staff 的左边界；空心洞阈值按谱距缩放；排版使用“上方候选 → 必要时下方 → 多个右侧横向候选 → 图内兼容位”的统一顺序。每类行为先由合成回归测试约束，再使用用户真实谱面验证。

**Tech Stack:** Python 3.12、标准库 `unittest`、OpenCV、NumPy

## Global Constraints

- 保留现有连续事件去重。
- 不新增强制符干过滤，不接入 AI 或外部 API。
- 每个标注盒必须完整位于图片内，且标准候选不得覆盖原谱或已有简谱。
- 行首过滤必须按 Staff 独立计算并有有限最大宽度。

---

### Task 1: Staff 行首符号区域

**Files:**
- Modify: `annotate.py`
- Modify: `tests/test_annotate.py`

**Interfaces:**
- Add: `_staff_left_edge(gray: np.ndarray, staff: Staff) -> int | None`
- Update: `_remove_leading_symbols(page: Page) -> None`

- [ ] **Step 1: Write failing tests**

用合成灰度图绘制从不同 x 开始的五条水平谱线。为每个 Staff 放置区域内的多种候选和区域外首个真实音符，断言区域内全部删除、区域外保留。增加两个 Staff 左边界不同的独立性测试。

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_annotate.LeadingSymbolZoneTests -v`

Expected: FAIL，因为当前实现只删除特定高窄字形，且遇到首个不匹配候选即停止。

- [ ] **Step 3: Implement bounded left zone**

在每条 staff line 附近对灰度图进行水平形态学开运算，取长水平线段左端点的中位数作为 Staff 左边界。行首区域右边界限制为 `left + 7.0 * staff.spacing`。先删除该有限区域内候选，再保留现有高窄字形检查作为补充。

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_annotate.LeadingSymbolZoneTests -v`

Expected: PASS；随后运行完整测试。

---

### Task 2: 谱距自适应空心音符

**Files:**
- Modify: `annotate.py`
- Modify: `tests/test_annotate.py`

**Interfaces:**
- Change: `detect_hollow_heads(black: np.ndarray, s: float) -> list[tuple[int, int]]`

- [ ] **Step 1: Write failing small-scale tests**

构造 `s=6` 的封闭椭圆空心音符，断言洞被检测；构造一个连通和弦轮廓中的两个独立白洞，断言返回两个不同中心。增加接触图像边界的开放白区不被检测。

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_annotate.HollowHeadScaleTests -v`

Expected: FAIL，因为当前函数没有 `s` 参数且使用固定尺寸。

- [ ] **Step 3: Implement spacing-relative thresholds**

将白洞宽、高、面积上下限改为由 `s` 计算的范围，并在 `detect_noteheads` 中传入当前谱距。每个符合条件的独立封闭白洞分别返回。

- [ ] **Step 4: Verify GREEN**

Run focused tests, then `.venv/bin/python -m unittest discover -s tests -v`.

---

### Task 3: 上方优先与横向右侧搜索

**Files:**
- Modify: `annotate.py`
- Modify: `tests/test_annotate.py`

**Interfaces:**
- Update internal placement search in `_render_group`.

- [ ] **Step 1: Write failing above-priority tests**

构造低音谱事件：音符下方可放置，但音符上方也存在空位。通过绿色像素盒断言标注位于音符上方。再阻塞最近上方候选、保留稍远上方空位，断言继续向上搜索而不落到下方。

- [ ] **Step 2: Write failing right-scan test**

构造上下均被阻塞的事件；阻塞固定 `1.8*s` 右侧候选，但在更右侧留出空间。断言标注完整位于更右侧，且没有使用下方兼容位。

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_annotate.PlacementPriorityTests -v`

Expected: 当前低音分支可能选择下方，且右侧只尝试固定 x 偏移。

- [ ] **Step 4: Implement ordered candidates**

对每个事件先从 `y_row` 向上扫描完整图内候选。仅上方全部失败时才进入低音谱下方逻辑。右侧阶段从 `1.8*s` 开始，以 `max(1, 0.5*s)` 为步长向右扫描到图片边界；每个横向偏移同时尝试和弦中心及有限纵向微调。使用现有 `_box_overlaps`、`occupied` 和完整 stack bounds 校验。

- [ ] **Step 5: Preserve compatibility fallback**

右侧所有位置都失败时恢复并拟合原图内兼容位置；不得改变成员顺序或产生越界。

- [ ] **Step 6: Verify GREEN**

Run focused tests and full discovery;现有 23 项测试必须继续通过。

---

### Task 4: 真实图片与服务验证

**Files:**
- Modify: none

- [ ] **Step 1: Compile and test**

Run: `.venv/bin/python -m py_compile annotate.py app.py tests/test_annotate.py && .venv/bin/python -m unittest discover -s tests -v`

- [ ] **Step 2: Process user score**

使用用户提供的真实谱面 PNG 运行 `annotate.py`，输出 `/tmp/sheet-music-three-regressions.jpg`，并用 OpenCV 确认可读取。

- [ ] **Step 3: Inspect target crops**

输出行首、空心和弦、右侧排版及上方优先相关的调试计数或坐标证据；不能只依赖“程序退出成功”。

- [ ] **Step 4: Report restart**

当前 `python app.py` 未启用自动重载，完成后提醒用户重启服务。

> 不创建 Git 提交，除非用户明确要求。
