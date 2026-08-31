# 简谱标注位置与行首调号过滤 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 当上下没有完整可用空间时安全地将简谱放到空闲右侧，并去掉每个谱表首个音符事件上由行首调号造成的 `b/#` 前缀。

**Architecture:** 保留 `annotate.py` 的现有流水线和局部排版。通过严格的矩形边界检查修复候选位置误判，并在临时记号检测后增加独立的行首升降号清理步骤；两项行为均用合成图像和数据结构进行回归测试。

**Tech Stack:** Python 3.12、标准库 `unittest`、OpenCV、NumPy

## Global Constraints

- 不新增第三方依赖。
- 代码运行时继续只使用 OpenCV 和 NumPy，不接入大模型或外部 API。
- 右侧候选必须完整位于图片内，且不覆盖原谱黑色像素或已绘制简谱。
- 第一个音符事件之后的临时升降号继续保留。

---

### Task 1: 严格候选边界与右侧回退

**Files:**
- Create: `tests/test_annotate.py`
- Modify: `annotate.py:711-738`
- Test: `tests/test_annotate.py`

**Interfaces:**
- Consumes: `_box_overlaps(black, occupied, H, W, x, right_ext, y, top_off, bot_off) -> bool`
- Produces: 对越过任一图片边界的候选返回 `True`；`_render_group` 因而在上方不可用而右侧空闲时使用右侧候选。

- [ ] **Step 1: 写出边界失败测试**

在 `tests/test_annotate.py` 中创建 `unittest.TestCase`，用全零的 `black` 和 `occupied` 验证顶部、底部、左侧、右侧越界均被视为冲突：

```python
import unittest

import cv2
import numpy as np

import annotate as an


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.black = np.zeros((40, 60), dtype=np.uint8)
        self.occupied = np.zeros((40, 60), dtype=bool)

    def test_out_of_bounds_box_is_not_clear(self):
        cases = [
            dict(x=-1, right_ext=5, y=20, top_off=3, bot_off=3),
            dict(x=56, right_ext=5, y=20, top_off=3, bot_off=3),
            dict(x=10, right_ext=5, y=2, top_off=3, bot_off=3),
            dict(x=10, right_ext=5, y=38, top_off=3, bot_off=3),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertTrue(
                    an._box_overlaps(
                        self.black, self.occupied, 40, 60, **case
                    )
                )
```

- [ ] **Step 2: 运行测试并确认正确失败**

Run: `.venv/bin/python -m unittest tests.test_annotate.PlacementTests.test_out_of_bounds_box_is_not_clear -v`

Expected: FAIL，因为当前实现会裁剪矩形，并把全零的越界区域误判为空闲。

- [ ] **Step 3: 最小化修复边界检查**

在 `_box_overlaps` 开头先计算未裁剪矩形；任一边界越界时直接返回 `True`，随后再读取数组：

```python
    x0, x1 = x - 2, x + right_ext + 2
    y0, y1 = y - top_off, y + bot_off
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return True
```

- [ ] **Step 4: 加入真实渲染回退测试**

构造顶部没有空间的高音谱表并调用真实 `_render_group`，通过绿色像素位置验证数字被画在音符右侧：

```python
    def test_uses_right_side_when_above_is_out_of_bounds(self):
        vis = np.full((90, 140, 3), 255, dtype=np.uint8)
        black = np.zeros((90, 140), dtype=np.uint8)
        occupied = np.zeros((90, 140), dtype=bool)
        staff = an.Staff(lines=[4, 14, 24, 34, 44], thickness=1, spacing=10)
        head = an.NoteHead(x=40, y=4, filled=True, staff=staff, digit=1)
        event = an.ChordEvent(x=40, heads=[head])

        an._render_group(
            vis, black, [event], staff, top=True, s=10, scale=0.7,
            bounds=(4, 44, 54, 84), occupied=occupied
        )

        green = (vis[:, :, 1] > vis[:, :, 0] + 40) & (
            vis[:, :, 1] > vis[:, :, 2] + 40
        )
        ys, xs = np.where(green)
        self.assertGreater(len(xs), 0)
        self.assertGreater(xs.min(), head.x)
```

- [ ] **Step 5: 运行 Task 1 测试**

Run: `.venv/bin/python -m unittest tests.test_annotate.PlacementTests -v`

Expected: PASS；真实渲染的绿色数字全部位于测试音符右侧。

---

### Task 2: 清除行首调号前缀

**Files:**
- Modify: `tests/test_annotate.py`
- Modify: `annotate.py:519-520, 1012-1018, 1052-1058`
- Test: `tests/test_annotate.py`

**Interfaces:**
- Produces: `_clear_leading_accidentals(page: Page) -> None`
- Side effects: 每个谱表最小横坐标附近（`0.6 * spacing`）的首个和弦事件全部设置为 `alter = 0`，并从 `page.accidentals` 删除对应记录。
- Consumes: `Page.staves`, `Page.noteheads`, `Page.accidentals`, `Staff.spacing`

- [ ] **Step 1: 写出行首清理失败测试**

在 `tests/test_annotate.py` 增加测试：首个和弦的两个音头都不显示升降前缀，后续临时升号保留：

```python
class LeadingAccidentalTests(unittest.TestCase):
    def test_clears_first_event_but_preserves_later_accidental(self):
        image = np.zeros((80, 120), dtype=np.uint8)
        staff = an.Staff(lines=[20, 30, 40, 50, 60], thickness=1, spacing=10)
        first_a = an.NoteHead(x=20, y=30, filled=True, staff=staff, alter=-1)
        first_b = an.NoteHead(x=24, y=40, filled=True, staff=staff, alter=1)
        later = an.NoteHead(x=60, y=30, filled=True, staff=staff, alter=1)
        page = an.Page(
            image=image,
            binary=image.copy(),
            staves=[staff],
            noteheads=[first_a, first_b, later],
            accidentals=[(20, 30, -1), (24, 40, 1), (60, 30, 1)],
        )

        an._clear_leading_accidentals(page)

        self.assertEqual([first_a.alter, first_b.alter, later.alter], [0, 0, 1])
        self.assertEqual(page.accidentals, [(60, 30, 1)])
```

- [ ] **Step 2: 运行测试并确认正确失败**

Run: `.venv/bin/python -m unittest tests.test_annotate.LeadingAccidentalTests -v`

Expected: ERROR，提示 `annotate` 没有 `_clear_leading_accidentals`。

- [ ] **Step 3: 实现最小清理函数**

在 `detect_accidentals` 后定义：

```python
def _clear_leading_accidentals(page: Page) -> None:
    """行首调号不生成简谱前缀，后续临时升降号保持不变。"""
    cleared_ids = set()
    for staff in page.staves:
        notes = [n for n in page.noteheads if n.staff is staff]
        if not notes:
            continue
        first_x = min(n.x for n in notes)
        limit = first_x + 0.6 * staff.spacing
        for note in notes:
            if note.x <= limit:
                note.alter = 0
                cleared_ids.add(note.id)
    page.accidentals = [
        item for item in page.accidentals
        if not any(
            n.id in cleared_ids and n.x == item[0] and n.y == item[1]
            for n in page.noteheads
        )
    ]
```

实现时不能依赖 `id` 唯一有效，因为手工构造的测试音符默认 `id=-1`；应按音符对象对应的 `(x, y)` 坐标集合过滤 `page.accidentals`：

```python
    cleared_positions = {(n.x, n.y) for n in cleared_notes}
    page.accidentals = [
        item for item in page.accidentals
        if (item[0], item[1]) not in cleared_positions
    ]
```

- [ ] **Step 4: 接入两条处理流水线**

在内存版 `process_image` 和命令行版 `run` 中，都紧跟 `detect_accidentals(page)` 调用：

```python
    _clear_leading_accidentals(page)
```

这样 `to_jianpu` 读取 `alter` 时，首个事件不会生成 `b/#` 前缀。

- [ ] **Step 5: 运行全部单元测试**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: PASS，包含候选边界、右侧回退和行首调号清理测试。

---

### Task 3: 端到端回归验证

**Files:**
- Modify: none
- Verify: `test.jpg`

**Interfaces:**
- Consumes: `annotate.py` 命令行入口
- Produces: 临时验证图片 `/tmp/sheet-music-annotation-check.jpg`

- [ ] **Step 1: 运行样例图片**

Run: `.venv/bin/python annotate.py test.jpg --output /tmp/sheet-music-annotation-check.jpg`

Expected: 退出码 0，输出包含“时值分析完成”和输出路径。

- [ ] **Step 2: 验证输出图片可读取**

Run: `.venv/bin/python -c "import cv2; p='/tmp/sheet-music-annotation-check.jpg'; im=cv2.imread(p); assert im is not None and im.size > 0; print(im.shape)"`

Expected: 退出码 0，并打印三通道图片尺寸。

- [ ] **Step 3: 运行语法与完整测试检查**

Run: `.venv/bin/python -m py_compile annotate.py app.py tests/test_annotate.py && .venv/bin/python -m unittest discover -s tests -v`

Expected: 所有命令退出码 0，所有测试 PASS。

> 本计划不包含 Git 提交步骤；只有用户明确要求时才创建提交。
