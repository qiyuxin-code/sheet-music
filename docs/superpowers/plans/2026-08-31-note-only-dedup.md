# 仅标注音符与连续重复去重 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留现有明确符号过滤，并让每个谱表行内连续且音高、升降号、八度和时值完全一致的事件只标注第一个。

**Architecture:** 继续使用现有音符头形状、谱号区和行首符号过滤，避免通过强制符干证据误删复杂和弦。渲染前按谱表行计算规范化事件特征并过滤连续重复事件，不删除底层识别与调试数据。

**Tech Stack:** Python 3.12、标准库 `unittest`、OpenCV、NumPy

## Global Constraints

- 不新增第三方依赖或外部 AI/API。
- 连续相同必须同时匹配数字、八度、升降号、时值以及和弦全部成员。
- 休止符和小节线不打断；切换谱表行时重置。
- 保留现有标注避让、行首升降号过滤和完整图片边界规则。

---

### Task 1: 保守符号过滤基线

**Files:**
- Modify: none
- Verify: `annotate.py`

- [ ] **Step 1: Preserve existing filters**

Do not add mandatory stem/beam filtering. Keep `_is_real_notehead`, `_remove_clef_zone`, `_remove_leading_symbols`, and the existing accidental filtering unchanged.

- [ ] **Step 2: Verify clean baseline**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all existing tests pass before event deduplication work.

---

### Task 2: Per-staff consecutive event deduplication

**Files:**
- Modify: `annotate.py`
- Modify: `tests/test_annotate.py`

**Interfaces:**
- Add function: `_event_signature(event: ChordEvent) -> tuple`
- Add function: `_dedupe_consecutive_events(events: list[ChordEvent]) -> list[ChordEvent]`

- [ ] **Step 1: Add failing single-note tests**

Create events with features `(digit, dots, prefix, dur)` and assert:

```python
def test_dedupes_only_consecutive_identical_events(self):
    staff = an.Staff(lines=[20, 30, 40, 50, 60], thickness=1, spacing=10)
    def event(x, digit=1, dots=0, prefix="", dur=1.0):
        head = an.NoteHead(
            x=x, y=30, filled=True, staff=staff,
            digit=digit, dots=dots, prefix=prefix, dur=dur,
        )
        return an.ChordEvent(x=x, heads=[head])

    first = event(10)
    duplicate = event(20)
    different_duration = event(30, dur=0.5)
    repeated_after_change = event(40)

    result = an._dedupe_consecutive_events(
        [first, duplicate, different_duration, repeated_after_change]
    )

    self.assertEqual(result, [first, different_duration, repeated_after_change])
```

Add cases proving changed octave dots or prefix are retained.

- [ ] **Step 2: Add failing chord and staff-isolation tests**

Verify two chords with the same normalized member features dedupe even if their head order differs. Verify events from different `Staff` objects are never deduped together.

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_annotate.EventDedupTests -v`

Expected: ERROR because `_dedupe_consecutive_events` does not exist.

- [ ] **Step 4: Implement event signatures**

Return a sorted tuple of `(digit, dots, prefix, dur)` for every event head. `_dedupe_consecutive_events` sorts by `x`, tracks the previous signature independently per `Staff` identity, and returns retained events in x order.

- [ ] **Step 5: Integrate rendering**

In `render`, collect each treble/bass staff's events and pass them through `_dedupe_consecutive_events` before `_render_group`. Do not mutate `page.chords` or `page.noteheads`.

- [ ] **Step 6: Verify GREEN**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests pass, including changed duration/octave/prefix, repeated-after-change, chord order and separate-staff cases.

---

### Task 3: Regression and sample verification

**Files:**
- Modify: none

- [ ] **Step 1: Compile and run tests**

Run: `.venv/bin/python -m py_compile annotate.py app.py tests/test_annotate.py && .venv/bin/python -m unittest discover -s tests -v`

Expected: exit code 0 with no test failures.

- [ ] **Step 2: Process the sample**

Run `annotate.py` against an available real score image and write `/tmp/sheet-music-note-only-check.jpg`.

Expected: exit code 0 and a readable output image.

- [ ] **Step 3: Restart requirement**

Report that the currently running `python app.py` process must be restarted because Uvicorn was started without reload mode.

> Do not create Git commits unless the user explicitly requests them.
