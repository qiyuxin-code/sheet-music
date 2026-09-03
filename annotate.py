#!/usr/bin/env python3
"""简谱标注器：在五线谱图片上叠印简谱数字。

流水线（每个阶段可用 --stage 单独运行并输出调试图）：
  1. 谱线检测   detect_staff_lines / group_staves / pair_systems
  2. 音符头检测  detect_noteheads
  3. 音高换算    assign_pitches
  4. 时值分析    analyze_durations
  5. 简谱换算    to_jianpu
  6. 渲染叠印    render
"""

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ---------- 数据结构 ----------

@dataclass
class Staff:
    """一组五线谱（5 条谱线）。"""
    lines: list[int]          # 5 条谱线中心的 y 坐标（从上到下）
    thickness: float
    spacing: float            # 相邻谱线间距 s
    clef: str = "treble"      # treble / bass（由系统内位置决定）

    @property
    def top(self) -> int:
        return self.lines[0]

    @property
    def bottom(self) -> int:
        return self.lines[-1]

    @property
    def center(self) -> float:
        return (self.lines[0] + self.lines[-1]) / 2


@dataclass
class System:
    """钢琴大谱表的一个系统：上方高音谱表 + 下方低音谱表。"""
    treble: Staff
    bass: Staff


@dataclass
class NoteHead:
    """一个检测到的音符头。"""
    x: int                    # 中心 x
    y: int                    # 中心 y
    filled: bool              # True=实心(四分/八分) False=空心(二分/全分)
    id: int = -1
    # 阶段 3 填充
    staff: Staff | None = None
    letter: int = -1          # 0=C 1=D 2=E 3=F 4=G 5=A 6=B
    octave: int = 0
    alter: int = 0            # -1 降 0 还原 +1 升
    # 阶段 4 填充
    stems_up: bool = True
    beams: int = 0            # 0=四分 1=八分 2=十六分
    dotted: bool = False
    # 阶段 5 填充
    digit: int = 0            # 1-7
    dots: int = 0             # 正=上方点数 负=下方点数
    dur: int = 4              # 拍数（四分=1.0 * 4）
    prefix: str = ""          # 升降号前缀 "#"/"b"


@dataclass
class ChordEvent:
    """同一 x 的一组音符头（竖向和弦）。"""
    x: int                    # 基准 x（取平均）
    heads: list[NoteHead]


@dataclass
class Page:
    image: np.ndarray
    binary: np.ndarray
    systems: list[System] = field(default_factory=list)
    staves: list[Staff] = field(default_factory=list)
    noteheads: list[NoteHead] = field(default_factory=list)
    chords: list[ChordEvent] = field(default_factory=list)
    accidentals: list[tuple[int, int, int]] = field(default_factory=list)  # (x,y,alter)


# 音符头模板基准尺寸（由线距 s 自适应）
# 实测 s=11px：实心头 ~14x11，空心头 ~18x11
NOTE_W_SOLID = 1.27
NOTE_H_SOLID = 1.0
NOTE_W_HOLLOW = 1.64
NOTE_H_HOLLOW = 1.0


# ---------- 第 1 步：谱线检测 ----------

def load_binary(path: str) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"无法读取图片: {path}")
    # 谱线为浅灰(~176)，音符头为黑(<110)。用 128 阈值保留黑色内容、滤除谱线。
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return gray, binary


def detect_staff_lines(gray: np.ndarray, min_cover_ratio: float = 0.5) -> list[tuple[int, int]]:
    h, w = gray.shape
    _, soft = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, w // 12), 1))
    opened = cv2.morphologyEx(soft, cv2.MORPH_OPEN, kernel)
    cover = (opened > 0).sum(axis=1) / w
    cand = np.flatnonzero(cover > min_cover_ratio)
    lines = []
    if not len(cand):
        return lines
    group = [cand[0]]
    for y in cand[1:]:
        if y - group[-1] <= 2:
            group.append(y)
        else:
            lines.append((int(np.mean(group)), len(group)))
            group = [y]
    if group:
        lines.append((int(np.mean(group)), len(group)))
    return lines


def group_staves(lines: list[tuple[int, int]], n_lines: int = 5) -> list[Staff]:
    staves = []
    i = 0
    while i + n_lines <= len(lines):
        chunk = lines[i:i + n_lines]
        ys = [c for c, _ in chunk]
        gaps = np.diff(ys)
        med = float(np.median(gaps))
        if all(abs(g - med) <= med * 0.35 for g in gaps):
            staves.append(Staff(lines=ys, thickness=float(np.mean([t for _, t in chunk])),
                                spacing=med))
            i += n_lines
        else:
            i += 1
    return staves


def pair_systems(staves: list[Staff]) -> list[System]:
    if len(staves) % 2 != 0:
        print(f"警告: 检测到 {len(staves)} 个谱表（奇数），最后一个将被忽略")
    systems = []
    for i in range(0, len(staves) - 1, 2):
        treble, bass = staves[i], staves[i + 1]
        treble.clef = "treble"
        bass.clef = "bass"
        systems.append(System(treble=treble, bass=bass))
    return systems


# ---------- 第 2 步：音符头检测 ----------

def make_templates(s: float) -> dict:
    """生成实心/空心头模板（含用于匹配的黑色掩膜）。"""
    tem = {}
    for filled in (True, False):
        w = int(round((NOTE_W_SOLID if filled else NOTE_W_HOLLOW) * s))
        h = int(round((NOTE_H_SOLID if filled else NOTE_H_HOLLOW) * s))
        cw, ch = w + 6, h + 6
        t = np.zeros((ch, cw), np.uint8)
        cv2.ellipse(t, (cw // 2, ch // 2), (w // 2, h // 2), 0, 0, 360, 255, -1 if filled else 3)
        tem[filled] = (t, w, h, cw, ch)
    return tem


def _match_template(black: np.ndarray, tpl: np.ndarray, th: float, nms_w: int, nms_h: int) -> list:
    res = cv2.matchTemplate(black, tpl, cv2.TM_CCOEFF_NORMED)
    pts = []
    ys, xs = np.where(res >= th)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if any(abs(x - px) < nms_w and abs(y - py) < nms_h for px, py, _ in pts):
            continue
        pts.append((x, y, float(res[y, x])))
    return pts


def detect_hollow_heads(black: np.ndarray, s: float) -> list[tuple[int, int]]:
    """空心头 = 被黑色包围的白色小洞。返回 (中心x,中心y)。"""
    H, W = black.shape
    min_w = max(2, int(round(0.35 * s)))
    max_w = max(min_w + 1, int(round(1.40 * s)))
    min_h = max(2, int(round(0.30 * s)))
    max_h = max(min_h + 1, int(round(1.10 * s)))
    min_a = max(3, int(round(0.12 * s * s)))
    max_a = max(min_a + 1, int(round(1.05 * s * s)))
    _, black_labels, black_stats, _ = cv2.connectedComponentsWithStats(
        (black > 0).astype(np.uint8) * 255, 8
    )

    def outer_labels(cx: int, cy: int) -> set[int]:
        rx = max(2, int(round(1.4 * s)))
        ry = max(2, int(round(1.0 * s)))
        x0, x1 = max(0, cx - rx), min(W, cx + rx + 1)
        y0, y1 = max(0, cy - ry), min(H, cy + ry + 1)
        labels = set(int(v) for v in np.unique(black_labels[y0:y1, x0:x1]) if v)
        enclosing: set[int] = set()
        for label in labels:
            x, y, w, h, _ = black_stats[label]
            if not (x <= cx < x + w and y <= cy < y + h):
                continue
            enclosing.add(label)
        return enclosing

    def has_notehead_outer_geometry(cx: int, cy: int, labels: set[int]) -> bool:
        rx = max(2, int(round(1.4 * s)))
        ry = max(2, int(round(1.0 * s)))
        x0, x1 = max(0, cx - rx), min(W, cx + rx + 1)
        y0, y1 = max(0, cy - ry), min(H, cy + ry + 1)
        for label in labels:
            ys, xs = np.where(black_labels[y0:y1, x0:x1] == label)
            if len(xs) == 0:
                continue
            width = int(xs.max() - xs.min() + 1)
            height = int(ys.max() - ys.min() + 1)
            if not (
                0.70 * s <= width <= 2.40 * s
                and 0.50 * s <= height <= 2.20 * s
            ):
                continue
            abs_xs = xs + x0
            abs_ys = ys + y0
            side = max(1, int(round(0.20 * s)))
            if (
                np.any(abs_xs <= cx - side)
                and np.any(abs_xs >= cx + side)
                and np.any(abs_ys <= cy - side)
                and np.any(abs_ys >= cy + side)
            ):
                return True
        return False

    def shares_chord_structure(
        recovered: tuple[int, int],
        raw: tuple[int, int],
    ) -> bool:
        cx, cy = recovered
        rx, ry = raw
        recovered_outer = outer_labels(cx, cy)
        if not recovered_outer or not has_notehead_outer_geometry(cx, cy, recovered_outer):
            return False
        raw_outer = outer_labels(rx, ry)
        if recovered_outer & raw_outer:
            return True

        # Scanned hollow chords can have heads disconnected from their shared stem.
        # Require a near-continuous side stroke belonging to the validating raw head.
        y0, y1 = sorted((cy, ry))
        span = y1 - y0 + 1
        if span <= 1:
            return False
        max_side = max(2, int(round(2.2 * s)))
        min_side = max(1, int(round(0.45 * s)))
        max_contour_gap = max(1, int(round(1.15 * s)))
        center_x = int(round((cx + rx) / 2))
        envelope_r = max(2, int(round(1.0 * s)))
        envelope_y0 = max(0, cy - envelope_r)
        envelope_y1 = min(H, cy + envelope_r + 1)
        _, contour_xs = np.where(np.isin(
            black_labels[envelope_y0:envelope_y1, :],
            list(recovered_outer),
        ))
        if len(contour_xs) == 0:
            return False
        for x in range(max(0, center_x - max_side), min(W, center_x + max_side + 1)):
            if abs(x - cx) < min_side or abs(x - rx) < min_side:
                continue
            column = black_labels[y0:y1 + 1, x]
            for label in raw_outer:
                if int((column == label).sum()) < math.ceil(0.85 * span):
                    continue
                empty_gap = max(0, int(np.min(np.abs(contour_xs - x))) - 1)
                if empty_gap <= max_contour_gap:
                    return True
        return False

    def enclosed_candidates(ink: np.ndarray) -> list[tuple[int, int, int]]:
        white = (ink == 0).astype(np.uint8) * 255
        num, _, stats, cents = cv2.connectedComponentsWithStats(white, 8)
        found: list[tuple[int, int, int]] = []
        for i in range(1, num):
            x, y, w, h, a = stats[i]
            if x == 0 or y == 0 or x + w >= W or y + h >= H:
                continue
            if min_w <= w <= max_w and min_h <= h <= max_h and min_a <= a <= max_a:
                found.append((
                    int(round(cents[i][0])),
                    int(round(cents[i][1])),
                    int(a),
                ))
        return found

    candidates = enclosed_candidates(black)
    if candidates:
        radius = max(1, int(round(0.17 * s)))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * radius + 1, 2 * radius + 1))
        closed_candidates = enclosed_candidates(cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel))
        align_x = max(2, int(round(0.35 * s)))
        min_dy = max(2, int(round(0.50 * s)))
        max_dy = max(min_dy + 1, int(round(4.0 * s)))
        raw_centers = [(cx, cy) for cx, cy, _ in candidates]
        for cx, cy, area in closed_candidates:
            if any(
                abs(cx - rx) <= align_x and min_dy <= abs(cy - ry) <= max_dy
                and shares_chord_structure((cx, cy), (rx, ry))
                for rx, ry in raw_centers
            ):
                candidates.append((cx, cy, area))

    tol = max(2, int(round(0.25 * s)))
    candidates.sort(key=lambda item: -item[2])
    holes: list[tuple[int, int]] = []
    for cx, cy, _ in candidates:
        if any(abs(cx - hx) < tol and abs(cy - hy) < tol for hx, hy in holes):
            continue
        holes.append((cx, cy))
    return holes


def _dedupe_hollow_centers(centers: list[tuple[int, int]], s: float) -> list[tuple[int, int]]:
    tol = max(2, int(round(0.25 * s)))
    deduped: list[tuple[int, int]] = []
    for cx, cy in centers:
        if any(abs(cx - hx) < tol and abs(cy - hy) < tol for hx, hy in deduped):
            continue
        deduped.append((cx, cy))
    return deduped


def _staff_vertical_band(staff: Staff) -> tuple[int, int]:
    s = staff.spacing
    return min(staff.lines) - int(2.5 * s), max(staff.lines) + int(2.5 * s)


def _collect_staff_hollow_centers(page: Page, black: np.ndarray) -> list[tuple[int, int]]:
    """Detect hollow heads per staff with provenance; merge without cross-staff max-spacing dedup."""
    if not page.staves:
        s = page.systems[0].treble.spacing if page.systems else 11.0
        y_min, y_max = 0, black.shape[0]
        return _dedupe_hollow_centers([
            (hx, hy)
            for hx, hy in detect_hollow_heads(black, s)
            if y_min <= hy <= y_max
        ], s)

    kept: list[tuple[int, int]] = []
    for st in page.staves:
        y_min, y_max = _staff_vertical_band(st)
        staff_holes = _dedupe_hollow_centers([
            (hx, hy)
            for hx, hy in detect_hollow_heads(black, st.spacing)
            if y_min <= hy <= y_max
        ], st.spacing)
        for hx, hy in staff_holes:
            nearest = _nearest_staff(hy, page.staves)
            if nearest is st:
                kept.append((hx, hy))
    return kept


def _note_spacing(n: NoteHead, page: Page) -> float:
    if page.staves:
        st = _nearest_staff(n.y, page.staves)
        if st is not None:
            return st.spacing
    return page.systems[0].treble.spacing if page.systems else 11.0


def detect_noteheads(page: Page) -> None:
    """模板匹配实心头 + 白洞法找空心头，合并去重。"""
    s = page.systems[0].treble.spacing if page.systems else 11.0
    tem = make_templates(s)
    black = page.binary  # 阈值做了 staff 线清除
    page.noteheads = []

    # 有效纵坐标范围：整页内谱表上下界，外加谱表外 2.5*s（加线音）
    if page.staves:
        all_lines = [yy for st in page.staves for yy in st.lines]
        y_min = min(all_lines) - int(2.5 * s)
        y_max = max(all_lines) + int(2.5 * s)
    else:
        y_min, y_max = 0, black.shape[0]

    # 实心头
    tpl, w, h, cw, ch = tem[True]
    pts = _match_template(black, tpl, 0.60, 10, 9)
    for x, y, sc in pts:
        cy = y + ch // 2
        if y_min <= cy <= y_max:
            page.noteheads.append(NoteHead(x=x + cw // 2, y=cy, filled=True))

    # 空心头（白洞）：按谱表独立检测，保留谱表归属后合并
    for hx, hy in _collect_staff_hollow_centers(page, black):
        page.noteheads.append(NoteHead(x=hx, y=hy, filled=False))

    page.noteheads.sort(key=lambda n: (n.y, n.x))
    page.noteheads = [
        n for n in page.noteheads
        if (
            _is_real_notehead(black, n, _note_spacing(n, page))
            or (
                not n.filled
                and _is_real_notehead(
                    black, n, _note_spacing(n, page), recover_open_hollow=True
                )
            )
        )
    ]
    for i, n in enumerate(page.noteheads):
        n.id = i


def _is_real_notehead(
    black: np.ndarray,
    n: NoteHead,
    s: float,
    recover_open_hollow: bool = False,
) -> bool:
    """剔除非音符的误检：短横线、谱号等大字形。

    - 实心：黑像素包围盒或中心连通块过扁 → 短横线；包围盒过稀疏 → 谱号。
    - 空心：不再用"白洞两侧厚墨"判别（会把真实空心音符环误删），仅保留洞的封闭性判断。
    """
    pad = int(max(12, round(1.1 * s)))
    y0, y1 = max(0, n.y - pad), min(black.shape[0], n.y + pad + 1)
    x0, x1 = max(0, n.x - pad), min(black.shape[1], n.x + pad + 1)
    p = black[y0:y1, x0:x1]
    if p.size == 0:
        return False
    H, W = p.shape
    ys, xs = np.where(p > 0)
    if len(ys) == 0:
        return False
    bh = int(ys.max() - ys.min() + 1)
    bar_h = max(6, int(round(0.75 * s)))

    if n.filled:
        if bh <= bar_h:
            return False  # 横条

        # 中心连通块过扁 → 短横线（如 (260,203)）
        num, lab, stats, _ = cv2.connectedComponentsWithStats((p > 0).astype(np.uint8) * 255, 8)
        cy, cx = n.y - y0, n.x - x0
        best = None
        for i in range(1, num):
            bx, by, bw, bh2, ba = stats[i]
            if bx <= cx <= bx + bw and by <= cy <= by + bh2:
                if best is None or ba > best[4]:
                    best = stats[i]
        if best is not None and best[3] <= bar_h:
            return False
        # 黑像素包围盒过稀疏 → 谱号等大字形（如低音谱号 dens≈0.17）
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        if w * h > 0 and (p > 0).sum() / (w * h) < 0.22:
            return False
        return True

    # 空心：默认保持原始封闭性；仅对首轮失败的恢复候选使用谱距缩放闭运算。
    hollow_ink = p
    if recover_open_hollow:
        radius = max(1, int(round(0.17 * s)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, (2 * radius + 1, 2 * radius + 1)
        )
        hollow_ink = cv2.morphologyEx(p, cv2.MORPH_CLOSE, kernel)
    white = (hollow_ink == 0).astype(np.uint8) * 255
    num, lab, stats, cents = cv2.connectedComponentsWithStats(white, 8)
    cx, cy = n.x - x0, n.y - y0
    hole_tol = max(4, int(round(0.35 * s)))
    hole = None
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        bccx, bccy = cents[i]
        if abs(bccx - cx) <= hole_tol and abs(bccy - cy) <= hole_tol:
            if hole is None or a > hole[4]:
                hole = stats[i]
    return hole is not None


def _remove_clef_zone(page: Page) -> None:
    """移除谱表最左端孤立的谱号/调号误检（如低音谱号的点）。

    保守策略：只有同时满足以下条件才判定为谱号区并移除——
      1. 最左端 1.5*s 内聚成一簇，且与下一个音符相距 ≥ 6*s；
      2. 该簇比同一系统另一谱表的最左音符还靠左 ≥ 6*s。
    条件 2 是关键：续行系统（第 2、3…行）的真实首音符紧贴谱号、其后常跟长休止，
    与另一谱表的首音符基本对齐，因此不会被误删；而谱号/调号误检孤立在两谱表
    真实内容之前，条件 2 成立。
    """
    if not page.staves:
        page.staves = [st for s in page.systems for st in (s.treble, s.bass)]
    s = page.systems[0].treble.spacing if page.systems else 11.0
    for sys in page.systems:
        first_x = {}
        for st in (sys.treble, sys.bass):
            ns = sorted([n for n in page.noteheads if n.staff is st], key=lambda n: n.x)
            first_x[id(st)] = ns[0].x if ns else None
        for st in (sys.treble, sys.bass):
            ns = sorted([n for n in page.noteheads if n.staff is st], key=lambda n: n.x)
            if len(ns) < 2:
                continue
            cluster = [ns[0]]
            for n in ns[1:]:
                if n.x - cluster[-1].x <= 1.5 * s:
                    cluster.append(n)
                else:
                    break
            if len(cluster) >= len(ns):
                continue
            if ns[len(cluster)].x - cluster[-1].x < 6 * s:
                continue
            other = sys.bass if st is sys.treble else sys.treble
            other_first = first_x.get(id(other))
            if other_first is None or cluster[0].x >= other_first - 6 * s:
                continue
            for n in cluster:
                page.noteheads.remove(n)


def _staff_left_edge(gray: np.ndarray, staff: Staff) -> int | None:
    """从谱线灰度图估算单个谱表的左边界（各谱线长水平段左端点的中位数）。"""
    h, w = gray.shape
    band_h = max(2, int(round(staff.spacing * 0.3)))
    kernel_w = max(50, w // 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    min_run = max(20, kernel_w // 3)
    lefts: list[int] = []
    for y in staff.lines:
        y0 = max(0, y - band_h)
        y1 = min(h, y + band_h + 1)
        band = gray[y0:y1, :]
        _, soft = cv2.threshold(band, 200, 255, cv2.THRESH_BINARY_INV)
        opened = cv2.morphologyEx(soft, cv2.MORPH_OPEN, kernel)
        rel_y = min(y - y0, opened.shape[0] - 1)
        row = opened[rel_y]
        xs = np.where(row > 0)[0]
        if len(xs) >= min_run:
            lefts.append(int(xs[0]))
    if not lefts:
        return None
    return int(np.median(lefts))


def _has_compact_filled_head(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """Note-sized filled blob at candidate center (not a tall-narrow symbol body)."""
    pad = int(max(6, 0.85 * s))
    y0, y1 = max(0, n.y - pad), min(black.shape[0], n.y + pad + 1)
    x0, x1 = max(0, n.x - pad), min(black.shape[1], n.x + pad + 1)
    p = black[y0:y1, x0:x1]
    num, _, stats, _ = cv2.connectedComponentsWithStats((p > 0).astype(np.uint8) * 255, 8)
    cx, cy = n.x - x0, n.y - y0
    for i in range(1, num):
        bx, by, bw, bh, _ = stats[i]
        if bx <= cx <= bx + bw and by <= cy <= by + bh:
            if 0.55 * s <= bh <= 1.45 * s and 0.7 * s <= bw <= 2.0 * s:
                return True
    return False


def _has_adjacent_stem_or_beam(black: np.ndarray, n: NoteHead, s: float) -> bool:
    stems = stem_mask(black, s)
    beams = beam_mask(black, s)
    x0, x1 = max(0, n.x - 8), min(black.shape[1], n.x + 9)
    head_half = int(max(3, 0.55 * s))
    up = int((stems[max(0, n.y - 34):n.y - head_half, x0:x1] > 0).sum())
    dn = int((stems[n.y + head_half:min(black.shape[0], n.y + 36), x0:x1] > 0).sum())
    if (up + dn) > int(s * 1.2):
        return True
    beam_above = int((beams[max(0, n.y - 40):n.y - head_half, x0:x1] > 0).sum())
    beam_below = int((beams[n.y + head_half:min(black.shape[0], n.y + 40), x0:x1] > 0).sum())
    return (beam_above + beam_below) > max(8, int(s * 0.8))


def _has_strong_filled_note_evidence(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """Only a filled notehead with local stem/beam evidence can end the zone."""
    return (
        n.filled
        and _has_compact_filled_head(black, n, s)
        and _has_adjacent_stem_or_beam(black, n, s)
    )


def _symbol_zone_remove_right(
    left: int,
    max_right: float,
    staff: Staff,
    noteheads: list[NoteHead],
    black: np.ndarray,
    s: float,
) -> float:
    """Exclusive right edge of removable symbol zone; stops at first strong note."""
    for n in sorted(noteheads, key=lambda nh: nh.x):
        if n.staff is not staff or n.x < left or n.x >= max_right:
            continue
        if _has_strong_filled_note_evidence(black, n, s):
            return float(n.x)
    return max_right


def _remove_leading_symbols(page: Page) -> None:
    """移除每行谱表最左侧符号区（谱号/调号 b#/拍号/速度记号）产生的误检音符头。

    对每个谱表，先按灰度谱线估算左边界，在不超过 ``left + 7*s`` 的有限区间内
    移除符号候选；只有具有可靠局部符头及符干/横梁证据的实心候选可以提前结束
    非空心候选的移除区间。空心候选不能结束区域，且在整个严格排他的
    ``left + 7*s`` 硬边界内始终移除。硬边界本身为排他的，边界上及之外的候选
    不受影响。
    """
    black = page.binary
    gray = page.image
    if not page.noteheads:
        return

    staff_bounds: dict[int, tuple[int, float, float]] = {}
    for st in page.staves:
        left = _staff_left_edge(gray, st)
        if left is None:
            continue
        hard_right = left + 7.0 * st.spacing
        anchor_right = _symbol_zone_remove_right(
            left, hard_right, st, page.noteheads, black, st.spacing
        )
        staff_bounds[id(st)] = (left, anchor_right, hard_right)

    remove_ids: set[int] = set()
    for st in page.staves:
        bounds = staff_bounds.get(id(st))
        if bounds is None:
            continue
        left, anchor_right, hard_right = bounds
        for n in page.noteheads:
            if n.staff is not st or n.x < left:
                continue
            if not n.filled:
                if n.x < hard_right:
                    remove_ids.add(id(n))
            elif n.x < anchor_right:
                remove_ids.add(id(n))

    if remove_ids:
        page.noteheads = [n for n in page.noteheads if id(n) not in remove_ids]


# ---------- 第 3 步：音高换算 ----------

LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
# 各谱表最下线对应的音名（treble= E4, bass= G2）
BASE_CLEF = {"treble": (2, 4), "bass": (4, 2)}  # (letter_index, octave)


def _nearest_staff(y: float, staves: list[Staff]) -> Staff | None:
    best, bd = None, float("inf")
    for st in staves:
        d = abs(y - st.center)
        if d < bd:
            best, bd = st, d
    return best


def assign_pitches(page: Page) -> None:
    if not page.staves:
        # 由 systems 收集
        page.staves = [st for s in page.systems for st in (s.treble, s.bass)]
    for n in page.noteheads:
        st = _nearest_staff(n.y, page.staves)
        if st is None:
            continue
        n.staff = st
        ref_idx, ref_oct = BASE_CLEF[st.clef]
        half = st.spacing / 2.0
        step = round((st.bottom - n.y) / half)  # 高于最下线为正
        total = ref_idx + step
        n.letter = total % 7
        n.octave = ref_oct + total // 7


# ---------- 第 3b 步：临时记号检测 ----------

def _glyph_shape(g: np.ndarray) -> tuple[float, float, float]:
    """返回 (col0连续性, 左侧质量占比, 右侧列墨量)。"""
    h, w = g.shape
    total = int((g > 0).sum()) or 1
    col0 = float((g[:, 0] > 0).mean())
    left = float((g[:, :max(1, w * 2 // 5)] > 0).sum()) / total
    colr = float((g[:, int(w * 0.55):] > 0).sum()) / total
    return col0, left, colr


def _auto_templates(black: np.ndarray, glyphs: list[tuple], s: float) -> dict:
    """从字形里自动挑选可靠模板：flat/sharp/natural 各取几个。

    启发式（依据本字体的实侧形状）：
      flat   ：左列连续（col0>0.6）且左重（left>0.5）
      natural：双竖线居中、左右边缘墨量都低（colr<0.15 且 col0<0.4）
      sharp  ：右侧有墨（colr>=0.15）且左列不连续（col0<0.6）
    """
    pad = 2
    flats, sharps, nats = [], [], []
    for g in glyphs:
        gx0, gy0, gw, gh, gi = g[2], g[3], g[4], g[5], g[7]
        raw = black[gy0:gy0 + gh, gx0:gx0 + gw]     # 无填充的原始字形
        c = black[gy0 - pad:gy0 + gh + pad, gx0 - pad:gx0 + gw + pad]
        col0, left, colr = _glyph_shape(raw)
        if col0 > 0.6 and left > 0.5:
            flats.append(c)
        elif colr < 0.15 and col0 < 0.4:
            nats.append(c)
        elif colr >= 0.15 and col0 < 0.6:
            sharps.append(c)
    T = {"flat": [], "sharp": [], "natural": []}
    for c in flats[:3]:
        T["flat"].append(c)
    for c in sharps[:3]:
        T["sharp"].append(c)
    for c in nats[:3]:
        T["natural"].append(c)
    return T


def _classify_glyph(glyph: np.ndarray, s: float, T: dict) -> int:
    """用真实字形模板匹配分类，返回 alter。"""
    best_sc, best_alter = -1.0, 0
    for kind, alter in (("flat", -1), ("sharp", 1), ("natural", 0)):
        for t in T.get(kind, []):
            t2 = cv2.resize(t, (glyph.shape[1], glyph.shape[0]))
            r = cv2.matchTemplate(glyph, t2, cv2.TM_CCOEFF_NORMED)
            sc = float(r[0][0])
            if sc > best_sc:
                best_sc, best_alter = sc, alter
    return best_alter if best_sc >= 0.50 else 0


def detect_accidentals(page: Page) -> None:
    """检测临时记号：全局找高窄字形，匹配到最近的音符头（一对一）。

    用真实字形模板匹配分类（比合成模板可靠）。
    """
    black = page.binary
    s = page.systems[0].treble.spacing if page.systems else 11.0
    if not page.noteheads:
        return

    # 1) 收集所有高窄字形（符干/连线/加线易混，稍后用"紧邻音符头左侧"过滤）
    num, labels, stats, cents = cv2.connectedComponentsWithStats(black, 8)

    # 1b) 每个谱表最左音符 x：左侧谱号/拍号/速度区的字形一律排除
    #     （真正的临时记号必然紧跟某音符头右侧 x-42..x-7，不可能在最左音符更左边）
    staff_min_x = {}
    for n in page.noteheads:
        if n.staff is None:
            continue
        st = id(n.staff)
        staff_min_x[st] = min(staff_min_x.get(st, 10**9), n.x)

    glyphs = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if int(1.6 * s) <= h <= int(3.2 * s) and 0.4 * s <= w <= 1.5 * s and a >= 40:
            cx, cy = int(cents[i][0]), int(cents[i][1])
            # 找出该字形所属谱表（按 y 就近）
            st = _nearest_staff(cy, page.staves)
            if st is None:
                continue
            if cx < staff_min_x.get(id(st), 0):
                continue
            glyphs.append((cx, cy, x, y, w, h, a, i))

    # 2) 自动提取模板
    T = _auto_templates(black, glyphs, s)

    # 3) 对每个音符头，在左侧窗口找最近的候选字形；字形全局只分配给最近的音符头
    used = set()
    assigned = []
    for n in sorted(page.noteheads, key=lambda n: n.x):
        # 窗口：x ∈ [n.x-42, n.x-7]，纵向重叠优先（±2.2*s）
        cands = []
        for gx, gy, gx0, gy0, gw, gh, ga, gi in glyphs:
            if gi in used:
                continue
            if not (n.x - 42 <= gx0 + gw // 2 <= n.x - 7):
                continue
            if abs(gy - n.y) > 1.2 * s:
                continue
            cands.append((abs(gy - n.y), gx0, gy0, gw, gh, ga, gx, gy, gi))
        if not cands:
            continue
        cands.sort()
        _, gx0, gy0, gw, gh, ga, gx, gy, gi = cands[0]
        used.add(gi)
        # 提取孤立字形
        pad = 2
        crop = black[gy0 - pad:gy0 + gh + pad, gx0 - pad:gx0 + gw + pad]
        alter = _classify_glyph(crop, s, T)
        n.alter = alter
        assigned.append((n.x, n.y, alter))
    page.accidentals = assigned


def _clear_leading_accidentals(page: Page) -> None:
    """行首调号不生成简谱前缀，后续临时升降号保持不变。"""
    cleared_notes: list[NoteHead] = []
    for staff in page.staves:
        notes = [n for n in page.noteheads if n.staff is staff]
        if not notes:
            continue
        first_x = min(n.x for n in notes)
        limit = first_x + 0.6 * staff.spacing
        for note in notes:
            if note.x <= limit:
                note.alter = 0
                cleared_notes.append(note)
    cleared_positions = {(n.x, n.y) for n in cleared_notes}
    page.accidentals = [
        item for item in page.accidentals
        if (item[0], item[1]) not in cleared_positions
    ]


# ---------- 第 4 步：时值分析 ----------

def stem_mask(black: np.ndarray, s: float) -> np.ndarray:
    """常量：竖直符干线（长度 >= 1.6*s 的竖直开运算）。"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(1.6 * s))))
    return cv2.morphologyEx(black, cv2.MORPH_OPEN, k)


def beam_mask(black: np.ndarray, s: float) -> np.ndarray:
    """常量：水平横梁（长 1.2*s 的水平开运算）。"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, int(1.2 * s)), 1))
    return cv2.morphologyEx(black, cv2.MORPH_OPEN, k)


def _vertical_bands(mask: np.ndarray, x: int, y0: int, y1: int, min_w: int = 2) -> int:
    """数 mask 在 [y0,y1) 纵向区间内的水平条带数。横向采样较宽以容纳斜梁。"""
    y0, y1 = max(0, y0), min(mask.shape[0], y1)
    if y1 - y0 <= 0:
        return 0
    proj = (mask[y0:y1, x - 8:x + 9] > 0).sum(axis=1)
    cnt, prev = 0, False
    for v in proj:
        cur = v >= min_w
        if cur and not prev:
            cnt += 1
        prev = cur
    return cnt


def _has_dot(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """头右侧小圆点。只对实心头检测；在头侧缘外找独立的小黑簇。"""
    if not n.filled:
        return False
    x0, x1 = n.x + 8, n.x + 17
    y0, y1 = n.y - 6, n.y + 7
    win = black[y0:y1, x0:x1]
    if (win > 0).sum() < 4:
        return False
    # 圆点应是紧凑小黑簇（宽高 2~5px）
    num, labels, stats, _ = cv2.connectedComponentsWithStats((win > 0).astype(np.uint8) * 255, 8)
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if 2 <= w <= 6 and 2 <= h <= 6 and a >= 3:
            return True
    return False


def analyze_durations(page: Page) -> None:
    black = page.binary
    s = page.systems[0].treble.spacing if page.systems else 11.0
    stems = stem_mask(black, s)
    beams = beam_mask(black, s)

    for n in page.noteheads:
        # 1) 符干方向：比较头上下两侧 stem 像素量（符干贴近头侧缘，x 容差 ±8px）
        up = int((stems[max(0, n.y - 34):n.y - 4, n.x - 8:n.x + 9] > 0).sum())
        dn = int((stems[n.y + 5:min(black.shape[0], n.y + 36), n.x - 8:n.x + 9] > 0).sum())
        has_stem = (up + dn) > int(s * 1.2)
        n.stems_up = up >= dn if has_stem else True

        # 2) 梁数：沿符干方向数横梁条带（最多 2 道 = 十六分）
        if has_stem:
            if n.stems_up:
                bands = _vertical_bands(beams, n.x, n.y - 40, n.y - 2)
            else:
                bands = _vertical_bands(beams, n.x, n.y + 3, n.y + 40)
            n.beams = min(2, bands)
        else:
            n.beams = 0

        n.dotted = _has_dot(black, n, s)


# ---------- 第 5 步：简谱换算 ----------

def to_jianpu(page: Page, base_octave: int = 4) -> None:
    for n in page.noteheads:
        n.digit = (n.letter % 7) + 1       # C=1 ... B=7
        n.dots = n.octave - base_octave     # 正=高八度(上加点) 负=低八度(下加点)
        if n.alter > 0:
            n.prefix = "#"
        elif n.alter < 0:
            n.prefix = "b"
        # 时值 → 拍数
        if not n.filled:
            dur = 2.0 if n.beams >= 1 else 4.0
        else:
            dur = {0: 1.0, 1: 0.5, 2: 0.25}.get(n.beams, 1.0)
        if n.dotted:
            dur *= 1.5
        n.dur = dur


def build_chords(page: Page) -> None:
    """同谱表的音符头归为和弦事件。

    第一步：同一 x（容差 0.6*s）→ 竖向和弦。
    第二步：纵向几乎同一高度（组内跨度 ≤ 1.0*s）且横向邻近（dx ≤ 2.0*s）
    的簇合并 → 近水平簇和弦（如乐曲末尾同时按下的几个键），数字排成一列。
    """
    if not page.staves:
        page.staves = [st for s in page.systems for st in (s.treble, s.bass)]
    s = page.systems[0].treble.spacing if page.systems else 11.0
    tol = 0.6 * s
    dy_tol = 1.0 * s
    dx_tol = 2.0 * s

    # 第一步：竖向和弦（同一 x）
    events: list[ChordEvent] = []
    for n in sorted(page.noteheads, key=lambda n: n.x):
        placed = False
        for ev in events:
            if abs(n.x - ev.x) <= tol and ev.heads[0].staff is n.staff:
                ev.heads.append(n)
                ev.x = float(np.mean([h.x for h in ev.heads]))
                placed = True
                break
        if not placed:
            events.append(ChordEvent(x=n.x, heads=[n]))

    # 第二步：近水平簇合并（仅合并组内纵向跨度小的簇）
    changed = True
    while changed:
        changed = False
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                ev1, ev2 = events[i], events[j]
                if ev1.heads[0].staff is not ev2.heads[0].staff:
                    continue
                ys1 = [h.y for h in ev1.heads]
                ys2 = [h.y for h in ev2.heads]
                if max(ys1) - min(ys1) > dy_tol or max(ys2) - min(ys2) > dy_tol:
                    continue
                merge = False
                for h1 in ev1.heads:
                    for h2 in ev2.heads:
                        if abs(h1.y - h2.y) <= dy_tol and abs(h1.x - h2.x) <= dx_tol:
                            merge = True
                            break
                    if merge:
                        break
                if merge:
                    ev1.heads.extend(ev2.heads)
                    ev1.x = float(np.mean([h.x for h in ev1.heads]))
                    del events[j]
                    changed = True
                    break
            if changed:
                break

    page.chords = events
    # 每个和声内按音高从高到低排序（高音 top）
    for ev in page.chords:
        ev.heads.sort(key=lambda h: h.y)


def _head_features(h: NoteHead) -> tuple:
    return (h.digit, h.dots, h.prefix, h.dur)


def _event_signature(event: ChordEvent) -> tuple:
    seen: set[tuple] = set()
    feats: list[tuple] = []
    for h in event.heads:
        feat = _head_features(h)
        if feat in seen:
            continue
        seen.add(feat)
        feats.append(feat)
    return tuple(sorted(feats))


def _event_view_for_render(event: ChordEvent) -> ChordEvent:
    """View event with duplicate member feature tuples collapsed for rendering."""
    seen: set[tuple] = set()
    heads: list[NoteHead] = []
    for h in event.heads:
        feat = _head_features(h)
        if feat in seen:
            continue
        seen.add(feat)
        heads.append(h)
    if len(heads) == len(event.heads):
        return event
    return ChordEvent(x=event.x, heads=heads)


def _dedupe_consecutive_events(events: list[ChordEvent]) -> list[ChordEvent]:
    """Per-staff consecutive dedup by normalized event signature; returns render views."""
    if not events:
        return []
    prev_sig: dict[int, tuple] = {}
    result: list[ChordEvent] = []
    for ev in sorted(events, key=lambda e: e.x):
        staff_id = id(ev.heads[0].staff)
        sig = _event_signature(ev)
        if prev_sig.get(staff_id) == sig:
            continue
        result.append(_event_view_for_render(ev))
        prev_sig[staff_id] = sig
    return result


# ---------- 第 6 步：渲染叠印 ----------

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 220, 0)          # 简谱数字（绿色，方便与原黑字区分但不过分刺眼）


def _digit_scale(s: float) -> float:
    return max(0.7, s / 16.0)


def render(page: Page, out_path: Path | None = None, debug: bool = False) -> np.ndarray:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    s = page.systems[0].treble.spacing if page.systems else 11.0
    scale = _digit_scale(s)
    occupied = np.zeros(vis.shape[:2], dtype=bool)

    for sys in page.systems:
        treble_evs = _dedupe_consecutive_events(
            [ev for ev in page.chords if ev.heads[0].staff is sys.treble]
        )
        bass_evs = _dedupe_consecutive_events(
            [ev for ev in page.chords if ev.heads[0].staff is sys.bass]
        )
        mid_y = (sys.treble.bottom + sys.bass.top) / 2.0
        bounds = (sys.treble.top, sys.treble.bottom, sys.bass.top, sys.bass.bottom)
        _render_group(vis, page.binary, treble_evs, sys.treble, top=True, s=s, scale=scale, bounds=bounds, occupied=occupied)
        _render_group(vis, page.binary, bass_evs, sys.bass, top=False, s=s, scale=scale, mid_y=mid_y, bounds=bounds, occupied=occupied)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
    return vis


def _measure(text: str, scale: float):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 2)
    return tw, th, bl

def _box_overlaps(black: np.ndarray, occupied: np.ndarray, H: int, W: int, x: int,
                  right_ext: int, y: int, top_off: int, bot_off: int) -> bool:
    """数字（含八度点/前缀）所在区域是否压到谱面符号或已绘制的数字。"""
    x0, x1 = x - 2, x + right_ext + 2
    y0, y1 = y - top_off, y + bot_off
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return True
    return bool((black[y0:y1, x0:x1] > 0).any() or occupied[y0:y1, x0:x1].any())


def _avoid_overlap(black: np.ndarray, occupied: np.ndarray, H: int, W: int, x: int,
                   right_ext: int, y: int, th: int, line_h: int, grow: int,
                   top_off: int = 0, bot_off: int = 0) -> int:
    """若数字（含八度点/前缀）区域压到谱面符号或已绘制的数字，上下平移避让。"""
    if top_off == 0:
        top_off = th // 2 + 1
    if bot_off == 0:
        bot_off = th // 2 + 1
    if not _box_overlaps(black, occupied, H, W, x, right_ext, y, top_off, bot_off):
        return y
    span = 2 * line_h  # 最多平移 ±2*line_h（2px 步进）
    for d in (grow, -grow):
        for k in range(1, span + 1):
            cand = y + d * 2 * k
            if 0 <= cand < H and not _box_overlaps(black, occupied, H, W, x, right_ext,
                                                   cand, top_off, bot_off):
                return cand
    return y


def _stack_bounds_xy(ay: int, x_off: int, dir_sign: int, members, step: int):
    """Axis-aligned bounds of the full digit stack (left, top, right, bottom)."""
    left = top = right = bottom = None
    for i, (h, x, tw, th, ptw, pth, to, bo, re) in enumerate(members):
        xx = x + x_off
        yy = ay + dir_sign * i * step
        x0, x1 = xx - 2, xx + re + 2
        y0, y1 = yy - to, yy + bo
        left = x0 if left is None else min(left, x0)
        top = y0 if top is None else min(top, y0)
        right = x1 if right is None else max(right, x1)
        bottom = y1 if bottom is None else max(bottom, y1)
    return left, top, right, bottom


def _stack_fits_image(H: int, W: int, ay: int, x_off: int, dir_sign: int, members, step: int) -> bool:
    left, top, right, bottom = _stack_bounds_xy(ay, x_off, dir_sign, members, step)
    return left >= 0 and top >= 0 and right <= W and bottom <= H


def _fit_compat_stack(H: int, W: int, prefer_y: int, prefer_x_off: int, dir_sign: int,
                      members, step: int):
    """Fit the complete annotation stack inside image bounds without reversing direction."""
    if _stack_fits_image(H, W, prefer_y, prefer_x_off, dir_sign, members, step):
        return prefer_y, prefer_x_off, dir_sign

    def score(cand_y: int, x_off: int) -> tuple[int, int]:
        return (abs(cand_y - prefer_y), abs(x_off - prefer_x_off))

    best = None
    for k in range(H):
        cand_ys = [prefer_y] if k == 0 else [c for c in (prefer_y + k, prefer_y - k) if 0 <= c < H]
        for cand_y in cand_ys:
            left, top, right, bottom = _stack_bounds_xy(cand_y, prefer_x_off, dir_sign, members, step)
            dx_opts = {0}
            if left < 0:
                dx_opts.add(-left)
            if right > W:
                dx_opts.add(W - right)
            for dx in dx_opts:
                x_off = prefer_x_off + dx
                if _stack_fits_image(H, W, cand_y, x_off, dir_sign, members, step):
                    cand = (score(cand_y, x_off), cand_y, x_off)
                    if best is None or cand[0] < best[0]:
                        best = cand
    if best is not None:
        return best[1], best[2], dir_sign

    # Wider/taller than image: deterministic clamp, direction preserved.
    y, x_off = prefer_y, prefer_x_off
    left, top, right, bottom = _stack_bounds_xy(y, x_off, dir_sign, members, step)
    if left < 0:
        x_off += -left
    elif right > W:
        x_off += W - right
    left, top, right, bottom = _stack_bounds_xy(y, x_off, dir_sign, members, step)
    if top < 0:
        y += -top
    elif bottom > H:
        y += H - bottom
    return y, x_off, dir_sign


def _render_group(vis, black, evs, staff, top, s, scale, mid_y=0, bounds=None, occupied=None):
    """在谱表上绘制一组的简谱数字。

    数字堆叠紧贴对应和弦最高音符头上方，最低音数字在下、最高音在上。
    位置优先级：
      - 所有谱表：从音符正上方开始向图像顶部搜索，不要求完全高于谱表顶线；
      - 上方全部不可用时，低音谱表再尝试谱表下方；
      - 若上下区域都不够：从音符右侧开始逐步向右搜索；
      - 标准候选全部失败时，将兼容位置拟合到图内且不反转堆叠方向。
    """
    H, W = vis.shape[:2]
    line_h = int(s * 2.2)          # 数字行高（含和弦叠放）
    dot_r = max(1, int(s * 0.12))
    head_clear = int(s * 1.25)     # 数字写在音符右侧
    if occupied is None:
        occupied = np.zeros((H, W), dtype=bool)   # 已绘制数字占用的区域

    # 对每个和弦事件：数字作为一个整体紧贴最高音符头上方，整体上下避让不拆开
    for ev in evs:
        heads = ev.heads
        cx = int(round(ev.x))
        top_head_y = heads[0].y    # 最高音（谱面最上）
        lowest_head_y = heads[-1].y  # 最低音（谱面最下）

        # 预计算成员盒子参数；绘制顺序：最低音在下、最高音在上
        members = []
        for h in reversed(heads):
            tw, th, bl = _measure(f"{h.digit}", scale)
            ptw = pth = 0
            if h.prefix:
                pscale = max(0.3, scale * 0.6)
                ptw, pth, _ = _measure(h.prefix, pscale)
            up_dots = max(0, h.dots)
            dn_dots = max(0, -h.dots)
            top_off = th // 2 + 1 + (up_dots * (2 * dot_r + 2) + dot_r + 1 if up_dots else 0)
            bot_off = th // 2 + 1 + (dn_dots * (2 * dot_r + 2) + dot_r + 2 if dn_dots else 0)
            right_ext = tw + (ptw + 3 if ptw else 0) + 4
            members.append((h, cx - tw // 2, tw, th, ptw, pth, top_off, bot_off, right_ext))

        step = max(int(th * 0.95), 1)          # 压缩的和弦数字间距
        y_row = top_head_y - int(s * 0.6) - th // 2 - 2   # 贴着头上方

        def grp_clear(ay, x_off, dir_sign):
            for i, (h, x, tw, th, ptw, pth, to, bo, re) in enumerate(members):
                if _box_overlaps(black, occupied, H, W, x + x_off, re,
                                 ay + dir_sign * i * step, to, bo):
                    return False
            return True

        def stack_bounds(ay, dir_sign):
            """整列数字（含八度点/前缀）的纵向范围 [top, bottom]。"""
            top = bottom = None
            for i, (h, x, tw, th, ptw, pth, to, bo, re) in enumerate(members):
                yy = ay + dir_sign * i * step
                t, b = yy - to, yy + bo
                top = t if top is None else min(top, t)
                bottom = b if bottom is None else max(bottom, b)
            return top, bottom

        # 整体定位优先级：
        #   所有谱表 → 从音符头正上方起，向图像顶部完整搜索空位；
        #   低音谱表上方全不可用 → 尝试谱表下方；
        #   上下区域都不够 → 从音符右侧向右搜索。
        x_off = 0
        limit = 2 * line_h
        y_best, dir_sign = y_row, -1       # dir_sign=-1：向上堆叠（最低音在下）
        above_found = None
        cand = y_row
        while True:
            _left, cand_top, _right, _bottom = _stack_bounds_xy(
                cand, x_off, dir_sign, members, step
            )
            if cand_top < 0:
                break
            if grp_clear(cand, x_off, dir_sign):
                above_found = cand
                break
            cand -= 2

        need_right = above_found is None
        if above_found is not None:
            y_best = above_found
        elif not top:
            _unused, t_bot, b_top, b_bot = bounds if bounds else (0, staff.top, staff.bottom, staff.bottom)
            gap_top, gap_bot = t_bot + int(s * 0.5), b_top - int(s * 0.5)
            if gap_top >= gap_bot:
                gap_top, gap_bot = gap_bot, gap_top - 1   # 无空隙，直接走下方

            def in_gap(ay):
                t, b = stack_bounds(ay, -1)
                return t >= gap_top and b <= gap_bot

            need_right = False
            if in_gap(y_best) and grp_clear(y_best, x_off, dir_sign):
                pass
            else:
                t, b = stack_bounds(y_best, -1)
                found = None
                if b > gap_bot:
                    # 堆叠过低 → 向区域内上方找
                    for cand in range(y_best - 2, max(gap_top - step, 0) - 1, -2):
                        if in_gap(cand) and grp_clear(cand, x_off, dir_sign):
                            found = cand
                            break
                elif t < gap_top:
                    # 堆叠过高 → 向区域内下方找
                    for cand in range(y_best + 2, gap_bot + step + 1, 2):
                        if in_gap(cand) and grp_clear(cand, x_off, dir_sign):
                            found = cand
                            break
                else:
                    # 在区域内但与谱面符号冲突 → 上下小幅找空位
                    for k in range(1, limit + 1):
                        for sgn in (1, -1):
                            cand = y_best + sgn * 2 * k
                            if 0 <= cand < H and in_gap(cand) and grp_clear(cand, x_off, dir_sign):
                                found = cand
                                break
                        if found is not None:
                            break
                if found is not None:
                    y_best = found
                else:
                    # 两谱表之间放不下 → 低音谱表下方（须完整位于谱表下缘 clearance 之下）
                    clearance = int(s * 0.6)
                    compat_y, compat_dir = y_best, dir_sign

                    def below_staff_clear(ay):
                        top, bottom = stack_bounds(ay, -1)
                        return (
                            top >= b_bot + clearance
                            and bottom <= H
                            and grp_clear(ay, x_off, -1)
                        )

                    zero_top, zero_bottom = stack_bounds(0, -1)
                    cand0 = b_bot + clearance - zero_top
                    max_base = H - zero_bottom
                    y_best, dir_sign = cand0, -1
                    placed = False
                    if cand0 <= max_base:
                        for cand in range(cand0, max_base + 1, 2):
                            if below_staff_clear(cand):
                                y_best = cand
                                placed = True
                                break
                    if not placed:
                        y_best, dir_sign = compat_y, compat_dir
                        need_right = True

        # 上下区域都不够时：放在音符右侧（要求右侧无其他谱面符号/已标注数字）
        if need_right or not grp_clear(y_best, x_off, dir_sign):
            compat_y, compat_dir, compat_x_off = y_best, dir_sign, x_off
            cy = (top_head_y + lowest_head_y) / 2.0               # 和弦纵向中心
            base0 = int(round(cy + (len(members) - 1) * step / 2.0))  # 数字列中心对齐和弦中心

            def right_clear(ay, xr):
                for i, (h, x, tw, th, ptw, pth, to, bo, re) in enumerate(members):
                    if x + xr + re + 2 > W:
                        return False
                    if _box_overlaps(black, occupied, H, W, x + xr, re,
                                     ay - i * step, to, bo):
                        return False
                return True

            placed_right = False
            first_x_off = int(s * 1.8)
            x_step = max(1, int(s * 0.5))
            for xr in range(first_x_off, W + 1, x_step):
                base = None
                if right_clear(base0, xr):
                    base = base0
                else:
                    # 每条横向候选道只做有限纵向微调，再继续向右。
                    for k in range(1, limit + 1):
                        hit = None
                        for sgn in (1, -1):
                            cand = base0 + sgn * k
                            if 0 <= cand < H and right_clear(cand, xr):
                                hit = cand
                                break
                        if hit is not None:
                            base = hit
                            break
                if base is not None:
                    x_off = xr
                    y_best = base
                    dir_sign = -1
                    placed_right = True
                    break
            if not placed_right:
                y_best, dir_sign, x_off = compat_y, compat_dir, compat_x_off
                y_best, x_off, dir_sign = _fit_compat_stack(
                    H, W, y_best, x_off, dir_sign, members, step
                )

        for i, (h, x, tw, th, ptw, pth, to, bo, re) in enumerate(members):
            x = x + x_off
            y = y_best + dir_sign * i * step
            cv2.putText(vis, f"{h.digit}", (x, y + th // 2), FONT, scale, COLOR, 2, cv2.LINE_AA)
            if h.prefix:
                pscale = max(0.3, scale * 0.6)
                px = x + tw + 3
                py = y - th // 2 + pth - 1
                cv2.putText(vis, h.prefix, (px, py), FONT, pscale, COLOR, 1, cv2.LINE_AA)
            # 八度点：高八度在数字正上方，低八度在数字正下方
            dcx = x + tw // 2
            if h.dots > 0:
                dcy = y - th // 2 - dot_r - 2
                for k in range(h.dots):
                    cv2.circle(vis, (dcx, dcy - k * (2 * dot_r + 2)), dot_r, COLOR, -1)
            elif h.dots < 0:
                dcy = y + th // 2 + dot_r + 2
                for k in range(-h.dots):
                    cv2.circle(vis, (dcx, dcy + k * (2 * dot_r + 2)), dot_r, COLOR, -1)
            # 时值记号
            _draw_duration(vis, h, x, y, tw, th, s, scale)
            # 记录占用区域（数字+八度点范围）
            x0 = max(0, x - 2)
            x1 = min(W, x + re + 2)
            y0 = max(0, y - to)
            y1 = min(H, y + bo)
            occupied[y0:y1, x0:x1] = True


def _draw_duration(vis, h, x, y, tw, th, s, scale):
    """附点：时值含 .5 且非十六分 → 数字右侧圆点。"""
    if h.dotted and h.dur >= 1.0:
        cv2.circle(vis, (x + tw + 4, y + th // 2), max(1, int(s * 0.12)), COLOR, -1)


# ---------- 调试图 ----------

def debug_draw_noteheads(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    for n in page.noteheads:
        c = (0, 0, 255) if n.filled else (255, 0, 0)
        cv2.rectangle(vis, (n.x - 9, n.y - 7), (n.x + 9, n.y + 7), c, 1)
        pf = {0: "", -1: "b", 1: "#"}.get(n.alter, "?")
        cv2.putText(vis, f"{pf}{LETTERS[n.letter]}{n.octave}", (n.x - 9, n.y - 10),
                    FONT, 0.35, (0, 128, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def debug_draw_durations(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    for n in page.noteheads:
        dur = {0: 4, 1: 8, 2: 16}.get(n.beams, 0) if n.filled else (2 if n.beams else 1)
        tag = f"q{dur}" + ("." if n.dotted else "")
        if not n.filled:
            tag = "half" if n.beams else "whole"
        cv2.putText(vis, tag, (n.x - 10, n.y - 10), FONT, 0.35, (0, 0, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def debug_draw_staffs(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    for staff in [s for sys in page.systems for s in (sys.treble, sys.bass)]:
        for y in staff.lines:
            cv2.line(vis, (0, y), (vis.shape[1], y), (0, 0, 255), 1)
        cv2.rectangle(vis, (0, staff.top - 4), (vis.shape[1] - 1, staff.bottom + 4),
                      (0, 180, 0), 1)
    for i, sys in enumerate(page.systems):
        x = 6
        y0, y1 = sys.treble.top, sys.bass.bottom
        cv2.line(vis, (x, y0), (x, y1), (255, 0, 0), 2)
        cv2.putText(vis, f"sys{i}", (x + 4, y0 - 6), FONT, 0.5, (255, 0, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


# ---------- 入口 ----------

def process_image(gray: np.ndarray, binary: np.ndarray,
                  quiet: bool = False) -> np.ndarray:
    """内存版完整流水线：输入灰度图 + 二值图，返回叠印后的 BGR 图像。"""
    page = Page(image=gray, binary=binary)

    lines = detect_staff_lines(page.image)
    staves = group_staves(lines)
    page.staves = staves
    page.systems = pair_systems(staves)
    if not quiet:
        print(f"谱线 {len(lines)} 条 → 谱表 {len(staves)} 个 → 系统 {len(page.systems)} 个")

    detect_noteheads(page)
    if not quiet:
        print(f"音符头 {len(page.noteheads)} 个（实心 {sum(n.filled for n in page.noteheads)} "
              f"/ 空心 {sum(not n.filled for n in page.noteheads)}）")
    assign_pitches(page)
    _remove_clef_zone(page)
    _remove_leading_symbols(page)
    detect_accidentals(page)
    _clear_leading_accidentals(page)
    if not quiet:
        print(f"检测到临时记号 {len(page.accidentals)} 个")

    analyze_durations(page)
    if not quiet:
        print("时值分析完成")

    to_jianpu(page)
    build_chords(page)
    if not quiet:
        counter = {}
        for n in page.noteheads:
            key = (n.prefix, n.digit, n.dots)
            counter[key] = counter.get(key, 0) + 1
        print("简谱分布(前20):", sorted(counter.items(), key=lambda x: -x[1])[:20])

    return render(page)


def run(input_path: str, output_path: str, debug_dir: Path, stage: int):
    gray, binary = load_binary(input_path)
    page = Page(image=gray, binary=binary)

    lines = detect_staff_lines(page.image)
    staves = group_staves(lines)
    page.staves = staves
    page.systems = pair_systems(staves)
    print(f"谱线 {len(lines)} 条 → 谱表 {len(staves)} 个 → 系统 {len(page.systems)} 个")

    if stage <= 1:
        debug_draw_staffs(page, debug_dir / "01_staffs.jpg")
        return

    detect_noteheads(page)
    print(f"音符头 {len(page.noteheads)} 个（实心 {sum(n.filled for n in page.noteheads)} "
          f"/ 空心 {sum(not n.filled for n in page.noteheads)}）")
    assign_pitches(page)
    _remove_clef_zone(page)
    _remove_leading_symbols(page)
    detect_accidentals(page)
    _clear_leading_accidentals(page)
    print(f"检测到临时记号 {len(page.accidentals)} 个")
    if stage == 2:
        debug_draw_noteheads(page, debug_dir / "02_noteheads.jpg")
        return

    analyze_durations(page)
    print("时值分析完成")
    if stage == 4:
        debug_draw_durations(page, debug_dir / "04_durations.jpg")
        return

    to_jianpu(page)
    build_chords(page)
    counter = {}
    for n in page.noteheads:
        key = (n.prefix, n.digit, n.dots)
        counter[key] = counter.get(key, 0) + 1
    print("简谱分布(前20):", sorted(counter.items(), key=lambda x: -x[1])[:20])

    render(page, Path(output_path))
    print(f"输出: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="在五线谱图片上叠印简谱")
    ap.add_argument("input", help="输入图片路径")
    ap.add_argument("--output", "-o", help="输出图片路径")
    ap.add_argument("--debug-dir", default="debug", help="调试图输出目录")
    ap.add_argument("--stage", type=int, default=6, help="运行到第几阶段")
    args = ap.parse_args()

    inp = args.input
    if not args.output:
        p = Path(inp)
        args.output = str(p.with_name(p.stem + "_简谱标注.jpg"))
    run(inp, args.output, Path(args.debug_dir), args.stage)


if __name__ == "__main__":
    main()
