#!/usr/bin/env python3
"""简谱标注器：在五线谱图片上叠印简谱数字。

流水线（每个阶段可用 --stage 单独运行并输出调试图）：
  1. 谱线检测   detect_staff_lines / group_staves / pair_systems
  2. 音符头检测  detect_noteheads
  3. 音高换算    assign_pitches
  4. 时值分析    analyze_durations
  5. 简谱换算    to_jianpu
  6. 渲染叠印    render（可选 Jev：位置 + 行首曲谱头过滤，见 jev_annotate.py）
"""

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from env_local import _load_local_env

_load_local_env()

import jev_annotate as jev


# ---------- 数据结构 ----------

@dataclass
class Staff:
    """一组五线谱（5 条谱线）。"""
    lines: list[int]          # 5 条谱线中心的 y 坐标（从上到下）
    thickness: float
    spacing: float            # 相邻谱线间距 s
    clef: str = "treble"      # treble / bass（由系统内位置决定）
    clef_changes: list[tuple[int, str]] = field(default_factory=list)

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
    """一个花括号系统：钢琴高低音谱表，可含人声等附加谱表。"""
    treble: Staff
    bass: Staff
    extras: list[Staff] = field(default_factory=list)

    def all_staves(self) -> list[Staff]:
        staves: list[Staff] = []
        for st in [self.treble, self.bass] + list(self.extras):
            if st not in staves:
                staves.append(st)
        return staves


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
    clef_regions: list[tuple[Staff, tuple[int, int, int, int], str]] = field(default_factory=list)
    _nolines: tuple[int, np.ndarray] | None = field(default=None, repr=False, compare=False)


def page_binary_nolines(page: Page) -> np.ndarray:
    """去掉谱线（保留穿过符头的那段）的二值图，按 (binary, staves) 缓存。"""
    key = hash((id(page.binary), tuple(id(st) for st in page.staves)))
    if page._nolines is None or page._nolines[0] != key:
        page._nolines = (key, _strip_staff_lines(page.binary, page.staves))
    return page._nolines[1]


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
    # 扫描件里个别谱线只有 ~210 的浅灰，200 会漏掉整条线进而丢掉整个谱表；
    # 阈值放宽到 230，长横向开运算足以滤掉文字和噪点。
    _, soft = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
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
    while i + n_lines - 1 <= len(lines):
        chunk = lines[i:i + n_lines]
        ys = [c for c, _ in chunk]
        gaps = np.diff(ys)
        med = float(np.median(gaps)) if len(gaps) else 0.0
        if len(chunk) == n_lines and all(abs(g - med) <= med * 0.35 for g in gaps):
            staves.append(Staff(lines=ys, thickness=float(np.mean([t for _, t in chunk])),
                                spacing=med))
            i += n_lines
            continue
        # 只找到 4 条、其中一个间距约等于两倍：中间那条太浅没检出，补上
        chunk4 = lines[i:i + n_lines - 1]
        if len(chunk4) == n_lines - 1:
            ys4 = [c for c, _ in chunk4]
            gaps4 = [float(g) for g in np.diff(ys4)]
            small = sorted(gaps4)[: n_lines - 3]
            base = float(np.median(small)) if small else 0.0
            doubles = [k for k, g in enumerate(gaps4) if abs(g - 2 * base) <= base * 0.35]
            normal = [g for g in gaps4 if abs(g - base) <= base * 0.35]
            if base > 0 and len(doubles) == 1 and len(normal) == n_lines - 3:
                k = doubles[0]
                filled = ys4[: k + 1] + [int(round((ys4[k] + ys4[k + 1]) / 2.0))] + ys4[k + 1:]
                staves.append(Staff(
                    lines=filled,
                    thickness=float(np.mean([t for _, t in chunk4])),
                    spacing=float(np.median(np.diff(filled))),
                ))
                i += n_lines - 1
                continue
        i += 1
    return staves


def group_systems(staves: list[Staff], gray: np.ndarray | None = None) -> list[System]:
    """按谱表间大空隙划分花括号系统；系统内可含 2+ 行谱表。"""
    if not staves:
        return []
    s = staves[0].spacing
    gaps = [
        staves[i + 1].lines[0] - staves[i].lines[-1]
        for i in range(len(staves) - 1)
    ]
    # 用较小间距估计同组距离；漏掉一条谱表时，大的跨系统间距不能拉高中位数。
    threshold = max(7.5 * s, float(np.percentile(gaps, 25)) * 1.45) if gaps else 7.5 * s
    # 花括号位于谱线左侧；它的纵向跨度比单纯按行距配对更可信。
    brace_links: set[int] = set()
    brace_breaks: set[int] = set()
    if gray is not None:
        lefts = [_staff_left_edge(gray, st) for st in staves]
        valid_lefts = [left for left in lefts if left is not None]
        for left in set(valid_lefts):
            x0, x1 = max(0, left-int(4*s)), max(0, left-2)
            if x1 <= x0:
                continue
            ink = (gray[:, x0:x1] < 180).astype(np.uint8)
            _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
            for x, y, w, h, area in stats[1:]:
                if h < 6*s or w > 3*s or w < 0.3*s:
                    continue
                covered = [i for i, st in enumerate(staves)
                           if y-s <= st.top and st.bottom <= y+h+s
                           and lefts[i] is not None and abs(lefts[i]-left) <= s]
                if len(covered) < 2:
                    continue
                first, last = covered[0], covered[-1]
                if abs(y-staves[first].top) > 2*s or abs(y+h-staves[last].bottom) > 2*s:
                    continue
                brace_links.update(range(first, last))
                brace_breaks.update((first-1, last))

    chunks: list[list[Staff]] = [[staves[0]]]
    for i, gap in enumerate(gaps):
        if i in brace_breaks or (i not in brace_links and gap >= threshold):
            chunks.append([staves[i + 1]])
        else:
            chunks[-1].append(staves[i + 1])

    systems: list[System] = []
    for chunk in chunks:
        if len(chunk) == 1:
            chunk[0].clef = "treble"
            systems.append(System(treble=chunk[0], bass=chunk[0]))
            continue
        chunk[0].clef = "treble"
        chunk[1].clef = "bass"
        for st in chunk[2:]:
            st.clef = "treble"
        systems.append(System(treble=chunk[0], bass=chunk[1], extras=chunk[2:]))
    return systems


def pair_systems(staves: list[Staff], gray: np.ndarray | None = None) -> list[System]:
    """兼容旧名：按花括号系统分组。"""
    return group_systems(staves, gray)


def collect_staves(page: Page) -> list[Staff]:
    """页面上的全部谱表（含人声等 extras），顺序稳定、去重。"""
    if page.staves:
        return list(page.staves)
    seen: list[Staff] = []
    for sys in page.systems:
        for st in sys.all_staves():
            if st not in seen:
                seen.append(st)
    return seen


def detect_staff_clefs(page: Page) -> None:
    """识别行首及正文里的谱号；高音谱号上部竖长白洞与低音谱号双点作证。"""
    ink = _strip_staff_lines(_soft_ink(page), page.staves)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    page.clef_regions = []
    for st in page.staves:
        st.clef_changes = []
        s = st.spacing
        found = []
        for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
            x, y, w, h = map(int, (x, y, w, h))
            if not st.top - 3*s <= y <= st.top + 1.5*s:
                continue
            clef = None
            right = x + w
            if (4.8*s <= h <= 9*s and 1.3*s <= w <= 3.5*s
                    and y <= st.top + 0.3*s and y+h >= st.bottom + 0.4*s):
                component = (labels[y:y+h, x:x+w] == label).astype(np.uint8)*255
                padded = cv2.copyMakeBorder(component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                _, _, holes, _ = cv2.connectedComponentsWithStats(255-padded, 8)
                for hx, hy, hw, hh, ha in holes[1:]:
                    if (hx > 0 and hy > 0 and hx+hw < w+2 and hy+hh < h+2
                            and hh >= 0.7*s and hh > 1.25*hw
                            and y+hy+hh <= st.center and ha >= 0.1*s*s):
                        clef = "treble"
                        break
            if clef is None and (0.9*s <= w <= 2.8*s and 1.6*s <= h <= 3.8*s
                                 and abs(y-st.top) <= 0.7*s):
                dots = []
                for dx, dy, dw, dh, da in stats[1:]:
                    if (x+w <= dx <= x+w+1.2*s and 0.15*s <= dw <= 0.65*s
                            and 0.15*s <= dh <= 0.65*s and 0.03*s*s <= da <= 0.4*s*s):
                        dots.append((dx+dw/2, dy+dh/2, dx+dw))
                for a in dots:
                    for b in dots:
                        if (abs(a[0]-b[0]) <= 0.3*s and 0.65*s <= b[1]-a[1] <= 1.35*s
                                and abs((a[1]+b[1])/2 - (st.top+s)) <= 0.6*s):
                            clef = "bass"
                            right = int(max(a[2], b[2]))
            if clef:
                found.append((x, (x, y, right-x, h), clef))
        left = _staff_left_edge(page.image, st)
        for x, box, clef in sorted(found):
            page.clef_regions.append((st, box, clef))
            if left is not None and x < left + 4.5*s:
                st.clef = clef
            else:
                st.clef_changes.append((x, clef))


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
    order = np.argsort(-res[ys, xs], kind="stable")
    for y, x in zip(ys[order].tolist(), xs[order].tolist()):
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

    # 同一颗空心头常被切成左右两个洞，中心横向差约一个符头宽、纵向几乎重合。
    # 二度音程的纵向距离大约是 0.5*s，所以纵向容差必须更紧，避免把和弦音并掉。
    dx_tol = max(3, int(round(1.40 * s)))
    dy_tol = max(2, int(round(0.34 * s)))
    candidates.sort(key=lambda item: -item[2])
    holes: list[tuple[int, int]] = []
    for cx, cy, _ in candidates:
        if any(abs(cx - hx) <= dx_tol and abs(cy - hy) <= dy_tol for hx, hy in holes):
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
    # 下加二线下的间（G3）正好在 2.5s 处，带子要再放宽一点
    return min(staff.lines) - int(3.0 * s), max(staff.lines) + int(3.0 * s)


def _row_has_long_run(black: np.ndarray, y: int, width: int, span: int) -> bool:
    if y < 0 or y >= black.shape[0]:
        return False
    row = black[y] > 0
    i = 0
    while i < width:
        if not row[i]:
            i += 1
            continue
        j = i + 1
        while j < width and row[j]:
            j += 1
        if j - i >= span:
            return True
        i = j
    return False


def _clear_staff_line_runs(black: np.ndarray, line_ys: list[int], s: float) -> np.ndarray:
    """去掉谱线上的长横线，留下符头轮廓。

    扫描件里谱线比阈值深，会从空心符头中间穿过去：洞被切成两半（重复蓝框），
    或者开口漏到背景里（和弦的另一个音检测不到）。先闭合细缝，再只删掉上下都是
    小块空白的那段谱线。全音符底边如果就是谱线，下方是页面背景，这段要留下。
    """
    if not line_ys:
        return black
    height, width = black.shape
    span = max(4, int(round(2.2 * s)))
    # 测试图和已经滤掉谱线的二值图没有长横线，不做闭运算，避免把挨得很近的空心头填死。
    if not any(_row_has_long_run(black, y, width, span) for y in line_ys):
        return black
    radius = max(1, int(round(0.17 * s)))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * radius + 1, 2 * radius + 1))
    out = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)
    height, width = out.shape
    white = (out == 0).astype(np.uint8) * 255
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    # 上下都是小块空白才说明谱线把符头内部分成了两半。贴着全音符底边的谱线，
    # 下方是整页背景，留下它，洞才不会漏出去。
    area_limit = int(round(4.0 * s * s))
    # 谱线位置可能差 1 像素、也可能有 2 像素粗：把相邻的长横线行并成一组一起处理
    line_rows = sorted({
        r
        for y in line_ys
        for r in (y - 1, y, y + 1)
        if 0 < r < height - 1 and _row_has_long_run(out, r, width, span)
    })
    groups: list[list[int]] = []
    for r in line_rows:
        if groups and r - groups[-1][-1] <= 1:
            groups[-1].append(r)
        else:
            groups.append([r])
    for grp in groups:
        r0, r1 = grp[0], grp[-1]
        if r0 <= 0 or r1 >= height - 1:
            continue
        row = np.all(out[r0:r1 + 1] > 0, axis=0)
        i = 0
        while i < width:
            if not row[i]:
                i += 1
                continue
            j = i + 1
            while j < width and row[j]:
                j += 1
            if j - i >= span:
                for x in range(i, j):
                    above = int(labels[r0 - 1, x])
                    below = int(labels[r1 + 1, x])
                    if above == 0 or below == 0:
                        continue
                    if stats[above][cv2.CC_STAT_AREA] <= area_limit and stats[below][cv2.CC_STAT_AREA] <= area_limit:
                        out[r0:r1 + 1, x] = 0
            i = j
    return out


def _strip_staff_lines(black: np.ndarray, staves: list[Staff]) -> np.ndarray:
    """经典去谱线：谱线行上、上下都是空白的像素置零，穿过符头的那段保留。

    小谱距（s≈6）的截图里谱线会从实心符头中间穿过，模板分数掉到 0.5 以下，
    符头核心也被横向开运算切成上下两片。在去掉谱线的图上找实心头更稳。
    """
    if not staves:
        return black
    out = black.copy()
    height, width = out.shape
    for st in staves:
        span = max(4, int(round(2.2 * st.spacing)))
        # 只有墨量接近谱线的行才算谱线行；谱线旁边一行若只是有根符杠，不能并进来，
        # 否则整组要求"四行都有墨"，谱线就一点也去不掉了。
        ref = max(int((out[int(y)] > 0).sum()) for y in st.lines if 0 <= int(y) < height)
        rows = sorted({
            r
            for y in st.lines
            for r in (int(y) - 1, int(y), int(y) + 1)
            if 0 < r < height - 1
            and int((out[r] > 0).sum()) >= 0.5 * ref
            and _row_has_long_run(out, r, width, span)
        })
        groups: list[list[int]] = []
        for r in rows:
            if groups and r - groups[-1][-1] <= 1:
                groups[-1].append(r)
            else:
                groups.append([r])
        for grp in groups:
            r0, r1 = grp[0], grp[-1]
            if r0 <= 0 or r1 >= height - 1:
                continue
            mask = np.all(out[r0:r1 + 1] > 0, axis=0) & (out[r0 - 1] == 0) & (out[r1 + 1] == 0)
            out[r0:r1 + 1, mask] = 0
    return out


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
        staff_black = _clear_staff_line_runs(black, [int(y) for y in st.lines], st.spacing)
        # 谱线有时是洞的一边（小节开头的符头顶靠谱线封口），去掉谱线后反而漏；
        # 两张图都找，再按同一符头合并。
        staff_holes = _dedupe_hollow_centers([
            (hx, hy)
            for hx, hy in detect_hollow_heads(staff_black, st.spacing)
            if y_min <= hy <= y_max
        ], st.spacing)
        extra = [
            (hx, hy)
            for hx, hy in detect_hollow_heads(black, st.spacing)
            if y_min <= hy <= y_max
        ] if staff_black is not black else []
        extra += _hollow_template_centers(
            staff_black, st.spacing, y_min, y_max,
            raw=black if staff_black is not black else None,
        )
        dx_tol = max(3, int(round(1.40 * st.spacing)))
        dy_tol = max(2, int(round(0.34 * st.spacing)))
        for hx, hy in extra:
            if any(abs(hx - x) <= dx_tol and abs(hy - y) <= dy_tol for x, y in staff_holes):
                continue
            staff_holes.append((hx, hy))
        for hx, hy in staff_holes:
            nearest = _nearest_staff(hy, page.staves)
            if nearest is st:
                kept.append((hx, hy))
    return kept


def _hollow_template_centers(
    black: np.ndarray, s: float, y_min: int, y_max: int,
    raw: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """在去掉谱线的图上用空心模板找回白洞法漏掉的符头。"""
    tpl, _, _, cw, ch = make_templates(s)[False]
    if black.shape[0] < tpl.shape[0] or black.shape[1] < tpl.shape[1]:
        return []
    res = cv2.matchTemplate(black, tpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= 0.20)
    if len(ys) == 0:
        return []
    order = np.argsort(-res[ys, xs])
    nms_w = max(3, int(round(0.90 * s)))
    nms_h = max(3, int(round(0.45 * s)))
    pts: list[tuple[int, int]] = []
    centers: list[tuple[int, int]] = []
    for y, x in zip(ys[order].tolist(), xs[order].tolist()):
        cx, cy = int(x) + cw // 2, int(y) + ch // 2
        if not (y_min <= cy <= y_max):
            continue
        if any(abs(int(x) - px) <= nms_w and abs(int(y) - py) <= nms_h for px, py in pts):
            continue
        cand = NoteHead(x=cx, y=cy, filled=False)
        # 细环符头模板分只有 0.2 出头，靠洞的椭圆剖面兜底；
        # 去谱线时的闭运算会把加线符头的洞糊掉，所以原图也要看一眼。
        # 靠洞剖面接受的点，纵坐标改用洞中心，免得同一颗头在相邻位置重复出现。
        if not _hollow_template_ok(black, cand, s, tpl, cw, ch, min_score=0.32):
            center = _hole_profile_center(black, cand, s)
            if center is None and raw is not None:
                center = _hole_profile_center(raw, cand, s)
            if center is None:
                continue
            cy = center
        pts.append((int(x), int(y)))
        centers.append((cx, cy))
    return centers


def _hollow_template_ok(
    black: np.ndarray,
    n: NoteHead,
    s: float,
    tpl: np.ndarray | None = None,
    cw: int | None = None,
    ch: int | None = None,
    *,
    min_score: float = 0.32,
) -> bool:
    """空心环墨量大约 0.3，不能按实心头的 0.55 去卡。"""
    if tpl is None or cw is None or ch is None:
        tpl, _, _, cw, ch = make_templates(s)[False]
    x0 = max(0, n.x - cw // 2)
    y0 = max(0, n.y - ch // 2)
    patch = black[y0:y0 + ch, x0:x0 + cw]
    if patch.shape != tpl.shape:
        return False
    score = float(cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)[0, 0])
    ink = float((patch > 0).mean())
    return score >= min_score and 0.28 <= ink <= 0.52


def _hollow_has_solid_neighbor(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """密集的八分音符段里，符头、符干、符杠之间围出的小白洞长得和空心符头一样。
    真正的二分/全音符周围只有细环和符干；旁边若紧挨着一块实墨（实心符头或符杠），
    这个洞就不是符头。"""
    H, W = black.shape
    rx = int(round(1.6 * s))
    ry = int(round(1.1 * s))
    x0, x1 = max(0, n.x - rx), min(W, n.x + rx + 1)
    y0, y1 = max(0, n.y - ry), min(H, n.y + ry + 1)
    patch = (black[y0:y1, x0:x1] > 0).astype(np.float32)
    kh = max(2, int(round(0.5 * s)))
    kw = max(3, int(round(0.6 * s)))
    if patch.shape[0] < kh or patch.shape[1] < kw:
        return False
    dens = cv2.boxFilter(patch, -1, (kw, kh), normalize=True, borderType=cv2.BORDER_CONSTANT)
    # 环本身（含符干接头）不算：中心 0.8s × 0.7s 的范围排除
    ex = int(round(0.8 * s))
    ey = int(round(1.3 * s))
    cy, cx = n.y - y0, n.x - x0
    dens[max(0, cy - ey):cy + ey + 1, max(0, cx - ex):cx + ex + 1] = 0
    # 符干和环的接头处墨也很厚，不算：竖线两侧各 0.35s 不看。
    # 最外侧那颗头的符干只到头为止，在窗口里只占一半高，所以阈值取 0.45。
    off = max(1, int(round(0.35 * s)))
    for c in np.where(patch.mean(axis=0) >= 0.45)[0]:
        dens[:, max(0, c - off):c + off + 1] = 0
    # 盒滤波在边界处会把窗外当白，只看完整窗口的位置
    hk, wk = kh // 2, kw // 2
    inner = dens[hk:dens.shape[0] - hk, wk:dens.shape[1] - wk]
    if inner.size == 0:
        return False
    return bool(inner.max() >= 0.9)


def _hole_between_vertical_lines(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """两根符干（或符干+小节线）夹着两条谱线/符杠围出的白格子也是"洞"。
    符头的洞最多一侧靠符干，另一侧是和洞一样高的弧线；两侧都是长竖线的不是符头。"""
    H, W = black.shape
    pad = int(round(1.2 * s))
    x0, x1 = max(0, n.x - pad), min(W, n.x + pad + 1)
    y0, y1 = max(0, n.y - pad), min(H, n.y + pad + 1)
    p = black[y0:y1, x0:x1]
    if p.size == 0:
        return False
    white = (p == 0).astype(np.uint8) * 255
    num, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    cy, cx = n.y - y0, n.x - x0
    if not (0 <= cy < lab.shape[0] and 0 <= cx < lab.shape[1]):
        return False
    label = int(lab[cy, cx])
    if label == 0:
        return False
    hx, hy, hw, hh, _ = stats[label]
    if hx == 0 or hx + hw >= p.shape[1]:
        return False  # 洞在窗口边缘被截断，判断不了
    span = max(2, int(round(0.8 * s)))
    r0 = max(0, y0 + hy - span)
    r1 = min(H, y0 + hy + hh + span)
    if r1 - r0 < hh + span:
        return False

    def cover(col: int) -> float:
        if not (0 <= col < W):
            return 0.0
        seg = black[r0:r1, col] > 0
        return float(seg.mean()) if seg.size else 0.0

    # 竖线要细：再往外 0.35s 的那一列不能也是通高的墨（否则是压实的符头块）
    off = max(2, int(round(0.35 * s)))
    left, right = x0 + hx - 1, x0 + hx + hw
    return (
        cover(left) >= 0.7 and cover(left - off) < 0.5
        and cover(right) >= 0.7 and cover(right + off) < 0.5
    )


def _hollow_is_plausible(black: np.ndarray, n: NoteHead, s: float, staff_black: np.ndarray | None = None) -> bool:
    if not _hollow_center_has_ink(black, n, s):
        return False
    if _hollow_has_solid_neighbor(black, n, s):
        return False
    if _hole_between_vertical_lines(black, n, s):
        return False
    if _is_real_notehead(black, n, s) or _is_real_notehead(black, n, s, recover_open_hollow=True):
        return True
    src = staff_black if staff_black is not None else black
    if staff_black is not None and _is_real_notehead(staff_black, n, s, recover_open_hollow=True):
        return True
    if _hollow_template_ok(src, n, s):
        return True
    return _hole_profile_ok(black, n, s)


def _hollow_center_has_ink(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """洞必须在符头轮廓内部，不能借窗口边缘的休止符或邻音封口。"""
    center = _compact_hole_center(black, n, s, y_tolerance=0.6)
    small_x, small_y = max(3, int(round(0.7*s))), max(2, int(round(0.55*s)))
    small = black[max(0, n.y-small_y):n.y+small_y+1,
                  max(0, n.x-small_x):n.x+small_x+1].copy()
    for i, y in enumerate(range(max(0, n.y-small_y), min(black.shape[0], n.y+small_y+1))):
        row = black[y, max(0, n.x-int(2.2*s)):n.x+int(2.2*s)+1]
        if row.size and (row > 0).mean() > 0.8:
            small[i] = 0
    if not small.size or (small > 0).mean() < 0.12:
        return False
    if center is None and 0 <= n.y < black.shape[0] and 0 <= n.x < black.shape[1] and black[n.y, n.x]:
        row = black[n.y, max(0, n.x-int(2.2*s)):n.x+int(2.2*s)+1]
        if row.size and (row > 0).mean() < 0.8 and _hollow_patch_ink(black, n, s) < 0.55:
            return False  # 延音弧/符干交点有墨却没有内部白洞。
    cx, cy = (int(round(v)) for v in center) if center else (n.x, n.y)
    rx, ry = max(3, int(round(1.1*s))), max(2, int(round(0.85*s)))
    y0, y1 = max(0, cy-ry), min(black.shape[0], cy+ry+1)
    x0, x1 = max(0, cx-rx), min(black.shape[1], cx+rx+1)
    patch = black[y0:y1, x0:x1].copy()
    for i, y in enumerate(range(y0, y1)):
        row = black[y, max(0, cx-int(2.2*s)):min(black.shape[1], cx+int(2.2*s)+1)]
        if row.size and (row > 0).mean() > 0.8:
            patch[i] = 0
    need = max(2, int(0.08*s*s))
    return bool(patch.size and (patch > 0).mean() >= 0.08
                and (patch[:cy-y0] > 0).sum() >= need
                and (patch[cy-y0+1:] > 0).sum() >= need
                and (patch[:, :cx-x0] > 0).sum() >= need
                and (patch[:, cx-x0+1:] > 0).sum() >= need)


def _compact_hole_center(black: np.ndarray, n: NoteHead, s: float, y_tolerance: float = 0.3):
    """原始墨迹中的小封闭白洞：避免小空心符头被实心模板吞掉。"""
    pad = max(5, int(round(1.2*s)))
    x0, y0 = max(0, n.x-pad), max(0, n.y-pad)
    patch = black[y0:min(black.shape[0], n.y+pad+1), x0:min(black.shape[1], n.x+pad+1)]
    _, _, stats, centers = cv2.connectedComponentsWithStats((patch == 0).astype(np.uint8), 8)
    matches = []
    for (x, y, w, h, area), (cx, cy) in zip(stats[1:], centers[1:]):
        if x == 0 or y == 0 or x+w >= patch.shape[1] or y+h >= patch.shape[0]:
            continue
        if not (2 <= w <= 1.4*s and 1 <= h <= s and area >= max(3, 0.05*s*s)):
            continue
        if h >= 0.5*s and area/(w*h) >= 0.9:
            continue
        cx, cy = cx+x0, cy+y0
        if abs(cx-n.x) <= 0.65*s and abs(cy-n.y) <= y_tolerance*s:
            matches.append((abs(cx-n.x)+abs(cy-n.y), cx, cy))
    return min(matches)[1:] if matches else None


def _hole_profile_ok(black: np.ndarray, n: NoteHead, s: float) -> bool:
    return _hole_profile_center(black, n, s) is not None


def _hole_profile_center(black: np.ndarray, n: NoteHead, s: float) -> int | None:
    """细环空心头：模板分不高，但洞的逐行宽度是一个椭圆剖面——
    最宽处约一个符头宽、上下收窄、四周都封住。返回洞的中心行，不像则 None。"""
    span = int(round(0.8 * s))
    widths: dict[int, int | None] = {}
    for r in range(n.y - span, n.y + span + 1):
        if _row_is_line(black, r, n.x, s):
            widths[r] = -1
            continue
        widths[r] = _hole_width_at_row(black, n.x, r, s)
    near = max(1, int(round(0.4 * s)))
    near_known = [
        (r, widths[r]) for r in range(n.y - near, n.y + near + 1)
        if widths.get(r) is not None and widths[r] >= 0
    ]
    if not near_known:
        return None
    max_w = max(wd for _, wd in near_known)
    if not (0.55 * s <= max_w <= 1.3 * s):
        return None
    thresh = 0.6 * max_w
    peak = min((r for r, wd in near_known if wd == max_w), key=lambda r: abs(r - n.y))
    top = peak
    while True:
        wd = widths.get(top - 1)
        if wd is None or (wd >= 0 and wd < thresh):
            break
        top -= 1
    bottom = peak
    while True:
        wd = widths.get(bottom + 1)
        if wd is None or (wd >= 0 and wd < thresh):
            break
        bottom += 1
    # 带子两端不能停在线行上（线行只是"未知"，不是洞）
    while top < peak and widths.get(top) == -1:
        top += 1
    while bottom > peak and widths.get(bottom) == -1:
        bottom -= 1
    height = bottom - top + 1
    if not (max(2, int(0.3 * s)) <= height <= 1.0 * s):
        return None
    # 洞上下必须有墨封口（不是 None 的开放区）
    above = widths.get(top - 1)
    below = widths.get(bottom + 1)
    h = black.shape[0]

    def ink_soon(r0: int, direction: int) -> bool:
        # 环的弧线可能有一两行断口，往外多看一行
        for k in range(2):
            r = r0 + direction * k
            if 0 <= r < h and black[r, n.x] > 0:
                return True
        return None

    if above is None and not ink_soon(top - 1, -1):
        return None
    if below is None and not ink_soon(bottom + 1, 1):
        return None
    center = (top + bottom) / 2.0
    if abs(center - n.y) > 0.5 * s:
        return None
    return int(round(center))


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

    # 有效纵坐标：各谱表加线区并集，但不包含页眉（曲名/作者）带
    if page.staves:
        y_min = min(
            st.top - int(3.0 * st.spacing) for st in page.staves
        )
        y_max = max(
            st.bottom + int(3.0 * st.spacing) for st in page.staves
        )
        meta = _metadata_y_cutoff(page)
        if meta is not None:
            y_min = max(y_min, meta)
    else:
        y_min, y_max = 0, black.shape[0]

    # 实心头：在去掉谱线的图上找，谱线穿过的符头才不会漏
    filled_src = page_binary_nolines(page)
    tpl, w, h, cw, ch = tem[True]
    # 非极大抑制窗口按谱距缩放：三度叠置的实心和弦两头只差一个 s，固定 9 像素会把下面那颗吃掉
    pts = _match_template(
        filled_src, tpl, 0.55,
        max(3, int(round(0.9 * s))), max(3, int(round(0.5 * s))),
    )
    for x, y, sc in pts:
        cy = y + ch // 2
        if y_min <= cy <= y_max:
            page.noteheads.append(NoteHead(x=x + cw // 2, y=cy, filled=True))

    # 空心头（白洞）：按谱表独立检测，保留谱表归属后合并
    for hx, hy in _collect_staff_hollow_centers(page, black):
        page.noteheads.append(NoteHead(x=hx, y=hy, filled=False))

    page.noteheads.sort(key=lambda n: (n.y, n.x))
    cleared: dict[int, np.ndarray] = {}

    def staff_black_for(n: NoteHead) -> np.ndarray | None:
        st = _nearest_staff(n.y, page.staves) if page.staves else None
        if st is None:
            return None
        key = id(st)
        if key not in cleared:
            cleared[key] = _clear_staff_line_runs(
                black, [int(y) for y in st.lines], st.spacing
            )
        return cleared[key]

    soft = _soft_ink(page)
    kept_heads: list[NoteHead] = []
    for n in page.noteheads:
        s_n = _note_spacing(n, page)
        if n.filled:
            n.staff = _nearest_staff(n.y, page.staves) if page.staves else None
            if not _is_real_notehead(filled_src, n, s_n):
                continue
            box = _filled_core_bbox(filled_src, n, s_n)
            # 实心头必须连着符干；谱号的圆点、全音符的环（偶尔也能匹配上实心模板）没有
            if box is not None and not _filled_has_stem(soft, box, s_n):
                continue
            _refine_filled_center(filled_src, n, s_n, box)
            kept_heads.append(n)
            continue
        if _hollow_is_plausible(black, n, s_n, staff_black_for(n)):
            kept_heads.append(n)
    page.noteheads = kept_heads
    for i, n in enumerate(page.noteheads):
        n.id = i


def _find_relaxed_holes(
    black: np.ndarray, s: float, y_min: int, y_max: int,
) -> list[tuple[int, int]]:
    """在闭运算图上找白洞，容差比 detect_hollow_heads 更宽，用于补全叠置和弦。"""
    H, W = black.shape
    radius = max(1, int(round(0.17 * s)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (2 * radius + 1, 2 * radius + 1)
    )
    closed = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)
    white = (closed == 0).astype(np.uint8) * 255
    num, _, stats, cents = cv2.connectedComponentsWithStats(white, 8)
    min_w = max(2, int(round(0.35 * s)))
    max_w = max(min_w + 1, int(round(2.20 * s)))
    min_h = max(2, int(round(0.30 * s)))
    max_h = max(min_h + 1, int(round(1.40 * s)))
    min_a = max(3, int(round(0.12 * s * s)))
    max_a = max(min_a + 1, int(round(2.00 * s * s)))
    holes: list[tuple[int, int]] = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if not (min_w <= w <= max_w and min_h <= h <= max_h and min_a <= a <= max_a):
            continue
        cx, cy = int(round(cents[i][0])), int(round(cents[i][1]))
        if y_min <= cy <= y_max:
            holes.append((cx, cy))
    return _dedupe_hollow_centers(holes, s)


def _recover_stacked_hollow_heads(page: Page) -> None:
    """沿共享符干补全竖向叠置和弦中漏检的空心头。"""
    black = page.binary
    if not page.staves:
        return
    tol = lambda s: max(3, int(round(0.30 * s)))
    for st in page.staves:
        s = st.spacing
        y_min, y_max = _staff_vertical_band(st)
        staff_black = _clear_staff_line_runs(black, [int(y) for y in st.lines], s)
        relaxed = _find_relaxed_holes(staff_black, s, y_min, y_max)
        if staff_black is not black:
            relaxed = _dedupe_hollow_centers(
                relaxed + _find_relaxed_holes(black, s, y_min, y_max), s
            )
        staff_notes = [n for n in page.noteheads if n.staff is st]
        for cx, cy in relaxed:
            if any(abs(cx - n.x) < tol(s) and abs(cy - n.y) < tol(s) for n in staff_notes):
                continue
            anchor = None
            for n in staff_notes:
                if abs(cx - n.x) <= 1.3 * s and 0.35 * s <= abs(cy - n.y) <= 4.0 * s:
                    anchor = n
                    break
            if anchor is None:
                continue
            if cx < staff_playable_min_x(page, st):
                continue
            cand = NoteHead(x=cx, y=cy, filled=False, staff=st)
            if not _hollow_is_plausible(black, cand, s, staff_black):
                continue
            page.noteheads.append(cand)
            staff_notes.append(cand)
    _recover_chord_heads_by_template(page)
    _recover_stacked_filled_heads(page)


def _stem_vertical_span(stems: np.ndarray, x: int, y: int, s: float) -> int:
    """符干列在音符附近的竖向跨度（像素）。"""
    col = int(round(x))
    y0 = max(0, int(y - 4.0 * s))
    y1 = min(stems.shape[0], int(y + 4.0 * s))
    lo, hi = None, None
    for c in range(max(0, col - 2), min(stems.shape[1], col + 3)):
        ys = np.where(stems[y0:y1, c] > 0)[0]
        if ys.size == 0:
            continue
        r0, r1 = y0 + int(ys[0]), y0 + int(ys[-1])
        lo = r0 if lo is None else min(lo, r0)
        hi = r1 if hi is None else max(hi, r1)
    return 0 if lo is None or hi is None else hi - lo


def _recover_stacked_filled_heads(page: Page) -> None:
    """沿共享符干补全叠置实心头漏检（两颗头共一根符干时模板匹配常只留一颗）。"""
    filled_src = page_binary_nolines(page)
    soft = _soft_ink(page)
    if not page.staves:
        return
    stems = stem_mask(page.binary, page.systems[0].treble.spacing if page.systems else 11.0)
    tol = lambda s: max(3, int(round(0.28 * s)))
    for st in page.staves:
        s = st.spacing
        y_min, y_max = _staff_vertical_band(st)
        notes = [n for n in page.noteheads if n.staff is st and n.filled]
        for column in _group_notes_by_x(notes, 1.0 * s):
            anchor = column[0]
            stem_col = _stem_column_near(stems, anchor, s)
            probe_x = stem_col if stem_col is not None else anchor.x
            span = _stem_vertical_span(stems, probe_x, anchor.y, s)
            if span < 2.3 * s:
                continue
            if len(column) >= max(2, int(span / (0.55 * s))):
                continue
            cols = {anchor.x, probe_x}
            for k in range(-12, 13):
                if abs(k) < 1:
                    continue
                cy = int(round(anchor.y + k * s / 2.0))
                if not (y_min <= cy <= y_max):
                    continue
                for cx in cols:
                    if any(
                        abs(cx - n.x) < tol(s) and abs(cy - n.y) < tol(s)
                        for n in page.noteheads
                    ):
                        continue
                    if cx < staff_playable_min_x(page, st):
                        continue
                    cand = NoteHead(x=int(cx), y=cy, filled=True, staff=st)
                    if not _is_real_notehead(filled_src, cand, s):
                        continue
                    box = _filled_core_bbox(filled_src, cand, s)
                    if box is None or not _filled_has_stem(soft, box, s):
                        continue
                    _refine_filled_center(filled_src, cand, s, box)
                    page.noteheads.append(cand)


def _group_notes_by_x(notes: list[NoteHead], tol: float) -> list[list[NoteHead]]:
    groups: list[list[NoteHead]] = []
    for n in sorted(notes, key=lambda nh: nh.x):
        placed = False
        for group in groups:
            gx = float(np.mean([h.x for h in group]))
            if abs(n.x - gx) <= tol:
                group.append(n)
                placed = True
                break
        if not placed:
            groups.append([n])
    return groups


def _try_recover_hollow_at(
    page: Page,
    notes: list[NoteHead],
    staff: Staff,
    s: float,
    black: np.ndarray,
    cx: int,
    cy: int,
    y_min: int,
    y_max: int,
    tpl: np.ndarray,
    cw: int,
    ch: int,
    *,
    require_closed_hole: bool,
    raw: np.ndarray | None = None,
    min_score: float = 0.28,
) -> bool:
    if cy < y_min or cy > y_max:
        return False
    if any(
        abs(cx - nh.x) <= max(3, int(round(0.55 * s)))
        and abs(cy - nh.y) <= max(3, int(round(0.55 * s)))
        for nh in notes
    ):
        return False
    if cx < staff_playable_min_x(page, staff):
        return False
    cand = NoteHead(x=cx, y=cy, filled=False, staff=staff)
    if not _hollow_center_has_ink(page.binary, cand, s):
        return False
    if _hollow_has_solid_neighbor(page.binary, cand, s) or _hole_between_vertical_lines(page.binary, cand, s):
        return False
    images = (black,) if raw is None or raw is black else (black, raw)
    for img in images:
        if _is_real_notehead(img, cand, s, recover_open_hollow=True):
            page.noteheads.append(cand)
            notes.append(cand)
            return True
    # 扫描件的空心头常是断开的弧线，中心差一两像素分数就掉很多；上下各试一行。
    for img in images:
        for dy in (0, -1, 1):
            probe = NoteHead(x=cx, y=cy + dy, filled=False, staff=staff)
            if _hollow_template_ok(img, probe, s, tpl, cw, ch, min_score=min_score):
                page.noteheads.append(probe)
                notes.append(probe)
                return True
    if not require_closed_hole:
        x0 = max(0, cx - cw // 2)
        y0p = max(0, cy - ch // 2)
        patch = black[y0p:y0p + ch, x0:x0 + cw]
        if patch.shape == tpl.shape:
            score = float(cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)[0, 0])
            if score >= 0.33 and float((patch > 0).mean()) >= 0.55:
                page.noteheads.append(cand)
                notes.append(cand)
                return True
    return False


def _recover_chord_heads_by_template(page: Page) -> None:
    """补全叠置和弦漏检空心头：先填同一竖列内外符头之间的空隙，再沿符干向外探。"""
    black = page.binary
    if not page.staves:
        return
    stems = stem_mask(black, page.systems[0].treble.spacing if page.systems else 11.0)
    for st in page.staves:
        s = st.spacing
        staff_black = _clear_staff_line_runs(black, [int(y) for y in st.lines], s)
        tpl, _, _, cw, ch = make_templates(s)[False]
        y_min, y_max = _staff_vertical_band(st)
        notes = [n for n in page.noteheads if n.staff is st]
        for column in _group_notes_by_x(notes, 1.0 * s):
            ys = [n.y for n in column]
            if max(ys) - min(ys) < 0.6 * s:
                continue
            y_lo, y_hi = min(ys), max(ys)
            cx = int(round(float(np.mean([n.x for n in column]))))
            half = s / 2.0
            steps = int(round((y_hi - y_lo) / half))
            for k in range(1, steps):
                cy = int(round(y_lo + k * half))
                _try_recover_hollow_at(
                    page, notes, st, s, staff_black, cx, cy, y_min, y_max,
                    tpl, cw, ch, require_closed_hole=False, raw=black,
                )
        # 沿符干向外探：和弦音程可以是二度到六度，所以按半个谱距走。
        # 有符干托着、位置又在音级格点上，模板门槛可以放低一点。
        for anchor in list(notes):
            if anchor.filled:
                continue
            stem_col = _stem_column_near(stems, anchor, s)
            if stem_col is None:
                continue
            for k in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
                cy = int(round(anchor.y + k * s / 2.0))
                for cx in (anchor.x, stem_col - int(round(0.75 * s))):
                    _try_recover_hollow_at(
                        page, notes, st, s, staff_black, cx, cy, y_min, y_max,
                        tpl, cw, ch, require_closed_hole=True, raw=black,
                        min_score=0.28,
                    )


def _stem_column_near(stems: np.ndarray, n: NoteHead, s: float) -> int | None:
    """找音符头附近符干列 x。"""
    y0 = max(0, n.y - int(3.5 * s))
    y1 = min(stems.shape[0], n.y + int(3.5 * s))
    best_x, best_cnt = None, 0
    for x in range(max(0, n.x - int(1.5 * s)), min(stems.shape[1], n.x + int(1.5 * s) + 1)):
        cnt = int((stems[y0:y1, x] > 0).sum())
        if cnt > best_cnt:
            best_cnt, best_x = cnt, x
    if best_cnt < int(s * 1.0):
        return None
    return best_x


def _same_hollow_head(a: NoteHead, b: NoteHead, s: float) -> bool:
    """同一颗空心头被切成左右两个洞时，横向差大约一个符头宽，纵向几乎重合。"""
    if a.filled or b.filled or a.staff is not b.staff:
        return False
    return (
        abs(a.x - b.x) <= max(3, int(round(1.40 * s)))
        and abs(a.y - b.y) <= max(2, int(round(0.34 * s)))
    )


def _dedupe_noteheads(page: Page) -> None:
    """合并坐标极近的重复音符头，以及同一空心头上的重复洞。"""
    tol = lambda s: max(3, int(round(0.28 * s)))
    # 同一个核心被模板和沿符干补全命中多次时，按真实核心去重，不能生成多个音级。
    core_seen = set()
    canonical = []
    source = page_binary_nolines(page)
    for n in page.noteheads:
        if n.filled:
            box = _filled_core_bbox(source, n, _note_spacing(n, page))
            if box is not None:
                key = (id(n.staff), box)
                if key in core_seen:
                    continue
                core_seen.add(key)
                bx, by, bw, bh = box
                n.x = int(round(bx + (bw-1)/2))
                n.y = int(round(by + (bh-1)/2))
        canonical.append(n)
    page.noteheads = canonical
    kept: list[NoteHead] = []
    for n in sorted(page.noteheads, key=lambda nh: (id(nh.staff or 0), not nh.filled, nh.x, nh.y)):
        s = _note_spacing(n, page)
        if any(
            nh.staff is n.staff
            and abs(nh.x - n.x) < tol(s)
            and abs(nh.y - n.y) < tol(s)
            for nh in kept
        ):
            continue
        if any(_same_hollow_head(nh, n, s) for nh in kept):
            continue
        if not n.filled and any(
            nh.filled and nh.staff is n.staff
            and abs(nh.x-n.x) <= 0.9*s and abs(nh.y-n.y) <= 0.4*s
            for nh in kept
        ):
            continue
        if (not n.filled) and any(
            nh.filled
            and (nh.staff is n.staff or n.staff is None or nh.staff is None)
            and abs(nh.x - n.x) <= 1.2 * s
            and 0.45 * s <= abs(nh.y - n.y) <= 4.0 * s
            for nh in kept
        ):
            continue
        kept.append(n)
    refined: list[NoteHead] = []
    for n in kept:
        if not n.filled and n.staff is not None:
            if not _refine_hollow_y(page.binary, n, _note_spacing(n, page)):
                continue
        refined.append(n)
    page.noteheads = _drop_adjacent_step_hollows(page, refined)


def _row_is_line(black: np.ndarray, y: int, x: int, s: float) -> bool:
    # 加线只比符头宽一点点，窗口不能开太大；环的上下弧最宽也不到 0.7s。
    half = max(2, int(round(0.6 * s)))
    if y < 0 or y >= black.shape[0]:
        return False
    row = black[y, max(0, x - half):x + half + 1]
    return row.size > 0 and float((row > 0).mean()) >= 0.85


def _hole_width_at_row(black: np.ndarray, x: int, y: int, s: float) -> int | None:
    """行 y 上 x 附近的洞的宽度；两侧都必须在 1.5s 内碰到墨，否则不算洞。

    被加线劈开的下半个洞常常整体偏到一侧，而紧挨符干还会有一条细缝，
    所以 x±0.25s 内的每个白段都量一下，取最宽的；但白段边缘必须离 x 很近。"""
    h, w = black.shape
    if not (0 <= y < h):
        return None
    row = black[y] > 0
    probe = max(1, int(round(0.25 * s)))
    reach = int(round(1.5 * s))
    best: int | None = None
    seen_right = -1
    for x0 in range(max(0, x - probe), min(w, x + probe + 1)):
        if row[x0] or x0 <= seen_right:
            continue
        left = x0
        while left - 1 >= 0 and not row[left - 1] and x0 - left < reach:
            left -= 1
        right = x0
        while right + 1 < w and not row[right + 1] and right - x0 < reach:
            right += 1
        seen_right = right
        if left - 1 < 0 or right + 1 >= w or not row[left - 1] or not row[right + 1]:
            continue
        width = right - left + 1
        if width > int(round(2.0 * s)):
            continue
        if max(left - x, x - right, 0) > 0.3 * s:
            continue
        if best is None or width > best:
            best = width
    return best


def _refine_hollow_y(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """按洞的逐行宽度找符头赤道：洞在赤道处最宽，被谱线/加线劈开也不受影响。

    返回 False 表示中心附近根本没有足够宽、左右都封住的白行——那不是符头的洞，
    而是加线与下方符头弧线之间的空隙。"""
    span = int(round(0.8 * s))
    widths: dict[int, int | None] = {}
    for r in range(n.y - span, n.y + span + 1):
        if _row_is_line(black, r, n.x, s):
            widths[r] = -1  # 线行：未知
            continue
        widths[r] = _hole_width_at_row(black, n.x, r, s)
    near = max(1, int(round(0.4 * s)))
    near_known = [
        (r, widths.get(r))
        for r in range(n.y - near, n.y + near + 1)
        if widths.get(r) is not None and widths.get(r) >= 0
    ]
    min_w = max(2, int(round(0.65 * s)))
    if not any(wd >= min_w for _, wd in near_known):
        compact = _compact_hole_center(black, n, s)
        if compact is not None:
            n.x, n.y = (int(round(v)) for v in compact)
            return True
        # 内部被压实的空心头没有像样的白行，但墨量会很高；那种留下不动
        return _hollow_patch_ink(black, n, s) >= 0.5
    max_w = max(wd for _, wd in near_known)
    thresh = 0.6 * max_w
    peak = min((r for r, wd in near_known if wd == max_w), key=lambda r: abs(r - n.y))
    top = peak
    while True:
        wd = widths.get(top - 1)
        if wd is None or (wd >= 0 and wd < thresh):
            break
        top -= 1
    bottom = peak
    while True:
        wd = widths.get(bottom + 1)
        if wd is None or (wd >= 0 and wd < thresh):
            break
        bottom += 1
    while top < peak and widths.get(top) == -1:
        top += 1
    while bottom > peak and widths.get(bottom) == -1:
        bottom -= 1
    new_y = int(round((top + bottom) / 2.0))
    # 洞很矮且紧贴一条线：可能是骑在线上的符头只剩半个洞（中心就在线上），
    # 也可能是间上的符头恰好贴着线。看线两侧到环弧的空白是否对称来区分。
    # 半个洞紧贴一条线、线那一侧又没有像样的白行（另半个洞被压实了）：
    # 这是骑在线上的符头，中心取线。线那边若有宽白行则是别的符头的洞，不动。
    if bottom - top + 1 <= 0.45 * s:
        for edge, direction in ((top - 1, -1), (bottom + 1, 1)):
            if widths.get(edge) != -1:
                continue
            r = edge
            while widths.get(r + direction) == -1:
                r += direction
            far = [widths.get(r + direction * k) for k in range(1, 4)]
            if any(wd is not None and wd >= thresh for wd in far):
                break
            new_y = int(round((edge + r) / 2.0))
            break
    if abs(new_y - n.y) <= max(2, int(round(0.5 * s))):
        n.y = new_y
    return True


def _hollow_patch_ink(black: np.ndarray, n: NoteHead, s: float) -> float:
    _, _, _, cw, ch = make_templates(s)[False]
    x0 = max(0, n.x - cw // 2)
    y0 = max(0, n.y - ch // 2)
    patch = black[y0:y0 + ch, x0:x0 + cw]
    return float((patch > 0).mean()) if patch.size else 0.0


def _hollow_template_score(black: np.ndarray, n: NoteHead, s: float) -> float:
    tpl, _, _, cw, ch = make_templates(s)[False]
    x0 = max(0, n.x - cw // 2)
    y0 = max(0, n.y - ch // 2)
    patch = black[y0:y0 + ch, x0:x0 + cw]
    if patch.shape != tpl.shape:
        return -1.0
    return float(cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)[0, 0])


def _drop_adjacent_step_hollows(page: Page, notes: list[NoteHead]) -> list[NoteHead]:
    """同一列相邻一度的两颗空心头在刻谱中必然左右错开；同列出现就是加线与
    下方符头弧线围出的假洞，保留模板得分高的那颗。"""
    drop: set[int] = set()
    hollows = [n for n in notes if not n.filled and n.staff is not None]
    for i, a in enumerate(hollows):
        for b in hollows[i + 1:]:
            if a.staff is not b.staff:
                continue
            s = _note_spacing(a, page)
            if abs(a.x - b.x) > max(3, int(round(0.5 * s))):
                continue
            sa = staff_step(a.staff, a.y)
            sb = staff_step(b.staff, b.y)
            # 同级 = 同一颗头被两种方法各找了一次；相邻一度 = 其中一个是假洞
            if abs(sa - sb) > 1:
                continue
            prof_a = _hole_profile_ok(page.binary, a, s)
            prof_b = _hole_profile_ok(page.binary, b, s)
            if prof_a != prof_b:
                drop.add(id(b) if prof_a else id(a))
                continue
            score_a = _hollow_template_score(page.binary, a, s)
            score_b = _hollow_template_score(page.binary, b, s)
            drop.add(id(b) if score_a >= score_b else id(a))
    return [n for n in notes if id(n) not in drop]


def _filled_core_is_notehead(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """去掉符干后，中心必须是一颗紧凑的实心符头，而不是符尾或符杠。"""
    return _filled_core_bbox(black, n, s) is not None


def _soft_ink(page: Page) -> np.ndarray:
    """宽松阈值的墨迹图：截图里 1 像素的符干和谱线是浅灰的，128 阈值下会消失。"""
    _, soft = cv2.threshold(page.image, 200, 255, cv2.THRESH_BINARY_INV)
    return soft


def _filled_has_stem(ink: np.ndarray, box: tuple[int, int, int, int], s: float) -> bool:
    """实心符头一定连着符干（或符杠）。谱号的圆点、歌词里的实心字形没有。

    在符头左右各 3 像素和符头本身的列里找一根 ≥1.7s 的细竖线，
    它要和符头上下 0.5s 的范围有交叠。ink 用宽松阈值图，细符干才不会丢。
    """
    bx, by, bw, bh = box
    H, W = ink.shape
    need = int(round(1.5 * s))
    # 从符头上沿往上、下沿往下数连续的墨：符干会一直延伸出去 ≥1.5s；
    # 叠在一起的另一颗头只有 0.9s 高，谱线只有一两行。
    for c in range(max(0, bx - 3), min(W, bx + bw + 3)):
        col = ink[:, c] > 0
        up = 0
        r = by - 1
        while r >= 0 and col[r]:
            up += 1
            r -= 1
        down = 0
        r = by + bh
        while r < H and col[r]:
            down += 1
            r += 1
        if max(up, down) >= need:
            return True
    return False


def _refine_filled_center(
    black: np.ndarray, n: NoteHead, s: float, box: tuple[int, int, int, int] | None = None,
) -> None:
    """把实心头的坐标挪到符头核心的包围盒中心。

    模板中心和真实符头中心差一两像素，在 s≈6 时就是半个音级，
    间上的音会被算到线上。
    """
    if box is None:
        box = _filled_core_bbox(black, n, s)
    if box is None:
        return
    bx, by, bw, bh = box
    cx = bx + (bw - 1) / 2.0
    cy = by + (bh - 1) / 2.0
    if abs(cx - n.x) <= 0.75 * s and abs(cy - n.y) <= 0.75 * s:
        n.x = int(round(cx))
        n.y = int(round(cy))


def _filled_lobe_rows(mask: np.ndarray, s: float) -> list[tuple[int, int]]:
    """只在墨迹有两个独立鼓包和收窄腰部时拆开相接和弦，不按音级网格硬切。"""
    widths = (mask > 0).sum(axis=1)
    peak = int(widths.max()) if widths.size else 0
    if peak < 0.55*s:
        return []
    bands = []
    for row in np.flatnonzero(widths >= 0.82*peak):
        if bands and row == bands[-1][-1]+1:
            bands[-1].append(int(row))
        else:
            bands.append([int(row)])
    peaks = [int(round(np.mean(band))) for band in bands if len(band) >= 2]
    if len(peaks) < 2 or any(b-a < 0.65*s for a, b in zip(peaks, peaks[1:])):
        return []
    cuts = [0]
    for a, b in zip(peaks, peaks[1:]):
        valley = a + int(np.argmin(widths[a:b+1]))
        if widths[valley] > 0.8*min(widths[a], widths[b]):
            return []
        cuts.append(valley+1)
    cuts.append(len(widths))
    result = []
    for start, end in zip(cuts, cuts[1:]):
        rows = np.flatnonzero(widths[start:end] >= 0.4*peak)
        if not len(rows):
            return []
        result.append((start+int(rows[0]), start+int(rows[-1])+1))
    return result


def _has_thin_ledger_crossing(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """谱表外的细加线与椭圆交叠，是实心头而非符尾的独立证据。"""
    if n.staff is None or n.staff.top-0.5*s <= n.y <= n.staff.bottom+0.5*s:
        return False
    height, width = black.shape
    for y in range(max(2, n.y-int(0.55*s)), min(height-2, n.y+int(0.55*s)+1)):
        edge = n.staff.top if y < n.staff.top else n.staff.bottom
        if abs((y-edge)/s-round((y-edge)/s)) > 0.2 or black[y, n.x] == 0:
            continue
        left = right = n.x
        while left > 0 and black[y, left-1] and n.x-left < 3*s:
            left -= 1
        while right+1 < width and black[y, right+1] and right-n.x < 3*s:
            right += 1
        length = right-left+1
        if not 1.8*s <= length <= 3*s:
            continue
        # 横梁厚且相邻行同样长，加线只有一两像素厚。
        above = int((black[y-2, left:right+1] > 0).sum())
        below = int((black[y+2, left:right+1] > 0).sum())
        if 0.4*s <= above <= 0.8*length and 0.4*s <= below <= 0.8*length:
            return True
    return False


def _filled_core_bbox(black: np.ndarray, n: NoteHead, s: float) -> tuple[int, int, int, int] | None:
    """去掉符干/符杠后包含中心的符头核心块的包围盒（绝对坐标 x,y,w,h），不像符头则 None。"""
    pad = int(max(10, round(2.0 * s)))
    # 纵向多取一些，才看得到符干另一端是什么（判断符尾用）
    pad_y = int(max(pad, round(4.5 * s)))
    y0, y1 = max(0, n.y - pad_y), min(black.shape[0], n.y + pad_y + 1)
    x0, x1 = max(0, n.x - pad), min(black.shape[1], n.x + pad + 1)
    patch = black[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    # 两个相接符头的中部也能形成 1.7s 竖条；不能把它当符干挖空。
    kh = max(5, int(round(2.4 * s)))
    if kh >= patch.shape[0]:
        kh = max(3, patch.shape[0] - 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    stems = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)
    # 三度叠置连成一块的实心头比符干高，整列会被当成符干抹掉；符干是细的，宽块不算
    wide_w = max(3, int(round(0.45 * s)))
    if wide_w < stems.shape[1]:
        wide = cv2.morphologyEx(
            stems, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (wide_w, 1))
        )
        stems[wide > 0] = 0
    beam_w = max(5, int(round(2.2 * s)))
    if beam_w >= patch.shape[1]:
        beam_w = max(3, patch.shape[1] - 1)
    beams = cv2.morphologyEx(
        patch, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (beam_w, 1))
    )
    # 斜梁也会留下椭圆状端点，不能只去水平梁。
    for slope in (-0.3, -0.15, 0.15, 0.3):
        rise = max(1, int(round(abs(slope)*beam_w)))
        kernel = np.zeros((rise+1, beam_w), np.uint8)
        cv2.line(kernel, (0, rise if slope < 0 else 0),
                 (beam_w-1, 0 if slope < 0 else rise), 1, 1)
        beams |= cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)
    # 细加线穿过真实符头时保留核心，只剔除两侧孤立的线段。
    thick = cv2.morphologyEx(beams, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, int(round(0.3*s))))))
    above = np.zeros_like(patch)
    below = np.zeros_like(patch)
    above[2:] = patch[:-2]
    below[:-2] = patch[2:]
    beams[(thick == 0) & ((above > 0) | (below > 0))] = 0
    core = patch.copy()
    core[(stems > 0) | (beams > 0)] = 0
    num, _, stats, _ = cv2.connectedComponentsWithStats((core > 0).astype(np.uint8) * 255, 8)
    cx, cy = n.x - x0, n.y - y0
    for i in range(1, num):
        bx, by, bw, bh, area = stats[i]
        if not (bx <= cx < bx + bw and by <= cy < by + bh):
            continue
        if 1.45*s < bh <= 4*s and 0.55*s <= bw <= 2.1*s:
            shape = core[by:by+bh, bx:bx+bw]
            lobes = _filled_lobe_rows(shape, s)
            if not lobes:
                # 双声部的两侧符干之间可能留下一像素桥；只保留有厚度的
                # 连续鼓包。单侧符尾不走这个恢复分支。
                left_stem = stems[by:by+bh, max(0, bx-3):bx]
                right_stem = stems[by:by+bh, bx+bw:bx+bw+3]
                if (left_stem > 0).sum() >= s and (right_stem > 0).sum() >= s:
                    dense = np.flatnonzero((shape > 0).sum(axis=1) >= max(3, 0.4*bw))
                    bands = np.split(dense, np.where(np.diff(dense) > 1)[0]+1)
                    lobes = [(int(b[0]), int(b[-1])+1) for b in bands
                             if len(b) >= 0.55*s and b[-1]-b[0]+1 <= 1.45*s]
            for start, end in lobes:
                if start <= cy-by < end:
                    by, bh = by+start, end-start
                    area = int((core[by:by+bh, bx:bx+bw] > 0).sum())
                    break
        if (_core_blob_is_flag(core, stems, (bx, by, bw, bh), s, beams)
                and not _has_thin_ledger_crossing(black, n, s)):
            return None
        # 单个鼓包不可拆分；只有上面经过双峰验证的相接符头才有各自核心。
        if not (0.55 * s <= bh <= 1.45 * s and 0.55 * s <= bw <= 2.1 * s):
            return None
        if bw * bh <= 0 or area / float(bw * bh) < 0.45:
            return None
        if bw > 2.1 * bh or bh > 1.7 * bw:
            return None
        return int(bx + x0), int(by + y0), int(bw), int(bh)
    return None


def _core_blob_is_flag(
    core: np.ndarray, stems: np.ndarray, box: tuple[int, int, int, int], s: float,
    beams: np.ndarray | None = None,
) -> bool:
    """去掉符干后的符尾钩子也是一块符头大小的墨。靠它和符头的相对位置区分：

    符尾总在符干右边、符头所在那端的对面。所以符干在块的左侧、而符干另一端
    （≥2s 外）又有一颗头时，这块就是符尾：向上的符干另一端的头在符干左边，
    向下的符干另一端的头在符干右边。
    """
    H, W = core.shape
    bx, by, bw, bh = box
    edge_pad = max(1, int(round(0.3*s)))
    rows = slice(max(0, by-edge_pad), min(H, by+bh+edge_pad))
    reach = 3
    left_cols = range(max(0, bx - reach), bx)
    right_cols = range(bx + bw, min(W, bx + bw + reach))
    left_cnt = max((int((stems[rows, c] > 0).sum()) for c in left_cols), default=0)
    right_cnt = max((int((stems[rows, c] > 0).sum()) for c in right_cols), default=0)
    if left_cnt >= 0.5*bh and right_cnt >= 0.5*bh:
        return False  # 两侧声部符干夹着真实符头，不能按单个符尾剔除。
    # 梁端落在符干左侧时看起来像正常符头；梁与该核心同高度则是梁端残片。
    if beams is not None and right_cnt >= 0.5*bh:
        margin = max(1, int(round(0.35*s)))
        beam_band = beams[max(0, by-margin):min(H, by+bh+margin), :]
        if (int((beam_band > 0).sum()) >= 0.3*s*s
                and int(np.any(beam_band > 0, axis=1).sum()) >= max(3, int(round(0.35*s)))):
            return True
    if left_cnt < 0.5 * bh or left_cnt < right_cnt:
        return False  # 符干不在左边：是正常的向上符干音符，或没符干
    sc = max(left_cols, key=lambda c: int((stems[rows, c] > 0).sum()))
    col = stems[:, sc] > 0
    mid = by + bh // 2
    top = mid
    while top - 1 >= 0 and col[top - 1]:
        top -= 1
    bottom = mid
    while bottom + 1 < H and col[bottom + 1]:
        bottom += 1
    up_len = by - top
    down_len = bottom - (by + bh - 1)
    min_len = 2.0 * s
    if up_len < min_len and down_len < min_len:
        return False
    far_row = top if up_len >= down_len else bottom
    band = max(2, int(round(0.8 * s)))
    r0, r1 = max(0, far_row - band), min(H, far_row + band + 1)
    span = int(round(1.4 * s))
    left_ink = int((core[r0:r1, max(0, sc - span):sc] > 0).sum())
    right_ink = int((core[r0:r1, sc + 1:min(W, sc + 1 + span)] > 0).sum())
    need = 0.3 * s * s
    if far_row == bottom:
        # 块在上端、符干向下：另一端左边有头 → 向上符干的符尾
        return left_ink >= need and left_ink > right_ink
    # 块在下端、符干向上：另一端右边有头 → 向下符干的符尾
    return right_ink >= need and right_ink > left_ink


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
    # 小谱距下实心头只有 0.9s 高（s=6 时 5–6 行），"横条"的上限不能固定在 6 像素
    bar_h = max(3, int(round(0.6 * s)))

    if n.filled:
        if _compact_hole_center(black, n, s) is not None:
            return False
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
        # 谱号等大字形很稀疏。和弦里的符头连着符干，包围盒密度也会降到 0.2 附近，
        # 所以这里只丢掉极稀疏的情况，形状交给符头核心判断。
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        if w * h > 0 and (p > 0).sum() / (w * h) < 0.12:
            return False
        return _filled_core_is_notehead(black, n, s)

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
    if hole is None:
        return False
    # 小节线 + 两条谱线 + 右边符头的环也能围出一个"洞"：比符头宽，而且左边界是一根长竖线
    hx, hy, hw, hh, _ = hole
    if hw > 1.8*s or hh > 1.3*s:
        return False
    if hw >= 0.8*s and hh >= 0.5*s and hole[4]/float(hw*hh) >= 0.9:
        return False  # 谱线和竖线围出的矩形白格，不是椭圆洞。
    if hw > 1.3 * s:
        col = x0 + hx - 1
        if 0 <= col < black.shape[1]:
            span = int(round(1.5 * s))
            r0 = max(0, y0 + hy - span)
            r1 = min(black.shape[0], y0 + hy + hh + span)
            seg = black[r0:r1, col] > 0
            # 竖线可能恰好终止于相邻谱线；只要求洞外有一段长直边，
            # 不要求上下两端都继续延伸（否则矩形白格会混入空心头）。
            hole_start, hole_end = y0+hy, y0+hy+hh
            up = black[r0:hole_end, col] > 0
            down = black[hole_start:r1, col] > 0
            rectangular = float(hole[4]) / max(1, hw*hh) >= 0.8
            if rectangular and ((up.size >= hh+0.7*s and up.mean() >= 0.9)
                                or (down.size >= hh+0.7*s and down.mean() >= 0.9)):
                return False
    return True


def _remove_clef_zone(page: Page) -> None:
    """移除谱表最左端孤立的谱号/调号误检（如低音谱号的点）。

    保守策略：只有同时满足以下条件才判定为谱号区并移除——
      1. 最左端 1.5*s 内聚成一簇，且与下一个音符相距 ≥ 6*s；
      2. 该簇比同一系统另一谱表的最左音符还靠左 ≥ 6*s。
    条件 2 是关键：续行系统（第 2、3…行）的真实首音符紧贴谱号、其后常跟长休止，
    与另一谱表的首音符基本对齐，因此不会被误删；而谱号/调号误检孤立在两谱表
    真实内容之前，条件 2 成立。
    """
    page.staves = collect_staves(page)
    gray = page.image
    s = page.systems[0].treble.spacing if page.systems else 11.0
    for sys in page.systems:
        first_x = {}
        for st in sys.all_staves():
            ns = sorted([n for n in page.noteheads if n.staff is st], key=lambda n: n.x)
            first_x[id(st)] = ns[0].x if ns else None
        for st in sys.all_staves():
            ns = sorted([n for n in page.noteheads if n.staff is st], key=lambda n: n.x)
            if len(ns) < 1:
                continue
            cluster = [ns[0]]
            for n in ns[1:]:
                if n.x - cluster[-1].x <= 1.5 * s:
                    cluster.append(n)
                else:
                    break
            left_edge = _staff_left_edge(gray, st) if gray is not None else None
            symbol_right = (
                left_edge + _staff_symbol_hard_right(st) if left_edge is not None else None
            )
            if len(cluster) >= 2 and symbol_right is not None:
                span_y = max(n.y for n in cluster) - min(n.y for n in cluster)
                if span_y > 3.2 * st.spacing and cluster[-1].x < symbol_right:
                    for n in cluster:
                        page.noteheads.remove(n)
                    continue
            if len(ns) < 2:
                continue
            if (
                len(cluster) == 1
                and symbol_right is not None
                and cluster[0].x < symbol_right
                and ns[len(cluster)].x - cluster[0].x >= 4.0 * st.spacing
            ):
                page.noteheads.remove(cluster[0])
                continue
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
        # 与 detect_staff_lines 一致：浅灰谱线也必须能定位谱头边界。
        _, soft = cv2.threshold(band, 230, 255, cv2.THRESH_BINARY_INV)
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


def _glyph_tall_and_narrow(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """谱号等竖长符号上的误检：连通块很高但很窄（高音谱号卷绕笔画）。"""
    pad = int(round(2.5 * s))
    y0, y1 = max(0, n.y - pad), min(black.shape[0], n.y + pad + 1)
    x0, x1 = max(0, n.x - pad), min(black.shape[1], n.x + pad + 1)
    patch = black[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    num, _, stats, _ = cv2.connectedComponentsWithStats((patch > 0).astype(np.uint8) * 255, 8)
    cx, cy = n.x - x0, n.y - y0
    for i in range(1, num):
        bx, by, bw, bh, _ = stats[i]
        if bx <= cx < bx + bw and by <= cy < by + bh:
            if bh > 3.4 * s and bw < 2.6 * s:
                return True
    return False


def _has_strong_filled_note_evidence(black: np.ndarray, n: NoteHead, s: float) -> bool:
    """Only a filled notehead with local stem/beam evidence can end the zone."""
    # 密集段落里符头和符干、符杠、谱线连成一片，连通块不再紧凑；
    # 去掉符干/符杠后的核心是符头也算。
    if _glyph_tall_and_narrow(black, n, s):
        return False
    return (
        n.filled
        and (_has_compact_filled_head(black, n, s) or _filled_core_is_notehead(black, n, s))
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
        if n.staff is not staff or n.x < left + 3.0 * s or n.x >= max_right:
            continue
        if _has_strong_filled_note_evidence(black, n, s):
            return float(n.x)
    return max_right


def staff_header_exclusive_right(page: Page, staff: Staff) -> float | None:
    """谱头区右边界（不含）：与 _remove_leading_symbols 的 7*s 一致，或第一个可靠实心音符。"""
    gray = page.image
    black = page_binary_nolines(page)
    s = staff.spacing
    left = _staff_left_edge(gray, staff)
    if left is None:
        return None
    symbol_right = left + _staff_symbol_hard_right(staff)
    for n in sorted(page.noteheads, key=lambda nh: nh.x):
        if n.staff is not staff or n.x < left + 3.0 * s:
            continue
        if n.x >= symbol_right:
            break  # 谱头区外的实心音符不能把前面的全音符/二分音符都划进谱头
        if _has_strong_filled_note_evidence(black, n, s) and not _head_in_clef_glyph(page, n, staff):
            return max(float(left), n.x - 0.45 * s)
    return symbol_right


def staff_music_start_x(page: Page, staff: Staff) -> float | None:
    """演奏区起始 x：仅在有可靠实心锚点音符时用于过滤；否则不凭 x 截断。"""
    gray = page.image
    black = page_binary_nolines(page)
    s = staff.spacing
    left = _staff_left_edge(gray, staff)
    if left is None:
        return None
    for n in sorted(page.noteheads, key=lambda nh: nh.x):
        if n.staff is not staff or n.x < left + 3.0 * s:
            continue
        if n.x >= left + _staff_symbol_hard_right(staff):
            break  # 第一个实心音符在谱头区之外：前面可能全是空心音符，不能凭它截断
        if _has_strong_filled_note_evidence(black, n, s) and not _head_in_clef_glyph(page, n, staff):
            return max(float(left), n.x - 0.45 * s)
    return None


def staff_playable_min_x(page: Page, staff: Staff) -> float:
    """简谱可标注的最小 x（谱头区右侧，或第一个可靠演奏音符之前）。"""
    hdr = staff_header_exclusive_right(page, staff)
    mus = staff_music_start_x(page, staff)
    if hdr is None and mus is None:
        return -1.0
    if hdr is None:
        return float(mus)
    if mus is None:
        return float(hdr)
    return max(float(hdr), float(mus))


def _metadata_y_cutoff(page: Page) -> int | None:
    """页眉/曲名区下缘：高于此 y 的检测一律丢弃（与谱表无关的误检）。"""
    if not page.systems:
        return None
    s = page.systems[0].treble.spacing
    return page.systems[0].treble.top - int(3.0 * s)


def purge_staff_header_noteheads(page: Page) -> None:
    """删除谱头、页眉、谱表外纵带内的误检音符头。"""
    staves = collect_staves(page)
    if not staves:
        return
    page.staves = staves
    meta_y = _metadata_y_cutoff(page)
    kept: list[NoteHead] = []
    for n in page.noteheads:
        if meta_y is not None and n.y < meta_y:
            continue
        if n.staff is None:
            continue
        st = n.staff
        s = st.spacing
        # 加线音（高音谱表下的 C4/A3、低音谱表上的 C4）要留，和检测带一致。
        y_lo = st.top - int(3.0 * s)
        y_hi = st.bottom + int(3.0 * s)
        if not (y_lo <= n.y <= y_hi):
            continue
        if _nearest_staff(n.y, staves) is not st:
            continue
        if _head_in_clef_glyph(page, n, st):
            continue
        if n.x < staff_playable_min_x(page, st):
            continue
        if _head_on_barline(page.binary, n, st, s):
            continue
        # 离谱表超过一格多的空心头必须有加线作证，否则多半是歌词里的 o/a/e
        far = n.y < st.top - int(1.1 * s) or n.y > st.bottom + int(1.1 * s)
        if far and not n.filled and not _has_ledger_near(page.binary, n, s):
            continue
        kept.append(n)
    page.noteheads = kept


def _head_in_clef_glyph(page: Page, n: NoteHead, staff: Staff) -> bool:
    """谱头处横跨五线并向两端延伸的连通字形不能作为音符锚点。

    高音谱号的卷曲中心能匹配实心符头，侧边长笔画也能被当成符干。
    检查完整字形而非仅中心的小窗口；限定在行首以保护正文中的和弦。
    """
    s = staff.spacing
    for st, (x, y, w, h), _ in page.clef_regions:
        if st is staff and x-0.3*s <= n.x <= x+w+0.3*s and y-0.3*s <= n.y <= y+h+0.3*s:
            return True
    left = _staff_left_edge(page.image, staff)
    if left is None or not left <= n.x < left + 4.5 * s:
        return False
    ink = page_binary_nolines(page)
    x0, x1 = max(0, int(n.x - 2 * s)), min(ink.shape[1], int(n.x + 2 * s + 1))
    y0, y1 = max(0, int(staff.top - 3 * s)), min(ink.shape[0], int(staff.bottom + 3 * s + 1))
    patch = ink[y0:y1, x0:x1]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(patch, 8)
    cx, cy = n.x - x0, n.y - y0
    nearby = labels[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
    for label in np.unique(nearby):
        if not label:
            continue
        bx, by, bw, bh, _ = stats[label]
        if (bh >= 6 * s and bw <= 3.5 * s
                and y0 + by < staff.top - 0.5 * s
                and y0 + by + bh > staff.bottom + 0.5 * s):
            return True
    return False


def _has_ledger_near(black: np.ndarray, n: NoteHead, s: float) -> bool:
    reach = int(round(0.7 * s))
    return any(
        _row_is_line(black, r, n.x, s)
        for r in range(n.y - reach, n.y + reach + 1)
    )


def _remove_leading_symbols(page: Page) -> None:
    """移除每行谱表最左侧符号区（谱号/调号 b#/拍号/速度记号）产生的误检音符头。

    对每个谱表，先按灰度谱线估算左边界，在不超过 ``left + 7*s`` 的有限区间内
    移除符号候选；只有具有可靠局部符头及符干/横梁证据的实心候选可以提前结束
    非空心候选的移除区间。空心候选不能结束区域，且在整个严格排他的
    ``left + 7*s`` 硬边界内始终移除。硬边界本身为排他的，边界上及之外的候选
    不受影响。
    """
    black = page_binary_nolines(page)
    gray = page.image
    if not page.noteheads:
        return

    staff_bounds: dict[int, tuple[int, float, float]] = {}
    for st in page.staves:
        left = _staff_left_edge(gray, st)
        if left is None:
            continue
        hard_right = left + _staff_symbol_hard_right(st)
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


def staff_step(st: Staff, y: float) -> int:
    """y 对应的音级（最下线为 0，向上为正，半格一级）。

    低分辨率扫描里五条线的间距并不均匀（6/7/6/7），用实际线位插值，
    不要按平均间距从最下线一路外推。"""
    lines = sorted(float(v) for v in st.lines)
    if len(lines) < 2:
        return int(round((st.bottom - y) / (st.spacing / 2.0)))
    n_lines = len(lines)
    # 谱表外按平均间距外推（单个线距只有 6/7 像素，用它外推误差太大）
    mean_gap = (lines[-1] - lines[0]) / (n_lines - 1)
    if y <= lines[0]:
        pos = (y - lines[0]) / mean_gap
    elif y >= lines[-1]:
        pos = (n_lines - 1) + (y - lines[-1]) / mean_gap
    else:
        i = max(j for j in range(n_lines - 1) if lines[j] <= y)
        pos = i + (y - lines[i]) / (lines[i + 1] - lines[i])
    return int(round(2.0 * ((n_lines - 1) - pos)))


def assign_pitches(page: Page) -> None:
    page.staves = collect_staves(page)
    for n in page.noteheads:
        st = _nearest_staff(n.y, page.staves)
        if st is None:
            continue
        n.staff = st
        clef = st.clef
        for change_x, changed_clef in st.clef_changes:
            if n.x >= change_x:
                clef = changed_clef
        ref_idx, ref_oct = BASE_CLEF[clef]
        step = staff_step(st, n.y)  # 高于最下线为正
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


def _cluster_noteheads_by_x(
    notes: list[NoteHead], tol: float,
) -> list[NoteHead]:
    """按 x 聚类，每组保留 y 最靠上的一个音符头。"""
    if not notes:
        return []
    clusters: list[list[NoteHead]] = []
    for n in sorted(notes, key=lambda nh: nh.x):
        if not clusters or n.x - clusters[-1][-1].x > tol:
            clusters.append([n])
        else:
            clusters[-1].append(n)
    return [min(c, key=lambda nh: nh.y) for c in clusters]


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

    第一步：同一 x（容差 1.0*s）→ 竖向和弦。
    第二步：只合并横向错位的同一和弦（如二度错开的符头）。
    合并后整列宽度必须 ≤ 1.3*s，避免把相邻旋律音或相邻和弦链式捏成一列。
    """
    page.staves = collect_staves(page)
    s = page.systems[0].treble.spacing if page.systems else 11.0
    tol = 1.0 * s
    dy_tol = 1.25 * s
    dx_tol = 1.3 * s
    max_x_span = 1.3 * s
    max_chord_span = 4.5 * s

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

    # 第二步：横向略有偏移的同一和弦合并（限制整列宽度，防止吞掉旋律）
    changed = True
    while changed:
        changed = False
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                ev1, ev2 = events[i], events[j]
                if ev1.heads[0].staff is not ev2.heads[0].staff:
                    continue
                xs = [h.x for h in ev1.heads] + [h.x for h in ev2.heads]
                ys = [h.y for h in ev1.heads] + [h.y for h in ev2.heads]
                if max(xs) - min(xs) > max_x_span:
                    continue
                if max(ys) - min(ys) > max_chord_span:
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

    cuts = {
        id(st): staff_music_start_x(page, st)
        for st in collect_staves(page)
    }

    def _keep_event(ev: ChordEvent) -> bool:
        if not ev.heads:
            return False
        st = ev.heads[0].staff
        if st is None:
            return True
        cut = cuts.get(id(st))
        if cut is None:
            return True
        return ev.x >= cut

    events = [ev for ev in events if _keep_event(ev)]
    for ev in events:
        ev.heads = _collapse_same_pitch(ev.heads)
    page.chords = events
    # 每个和声内按音高从高到低排序（高音 top）
    for ev in page.chords:
        ev.heads.sort(key=lambda h: h.y)


def _collapse_same_pitch(heads: list[NoteHead]) -> list[NoteHead]:
    """同一和弦里，同一音高只留一个符头，避免一个音印出重复数字。"""
    if not heads or any(h.letter < 0 for h in heads):
        return heads
    best: dict[tuple, NoteHead] = {}
    for head in heads:
        key = (head.letter, head.octave, head.alter)
        prev = best.get(key)
        if prev is None or (head.filled and not prev.filled):
            best[key] = head
    return list(best.values())


def _event_view_for_render(event: ChordEvent) -> ChordEvent:
    """渲染视图：保留和弦内每个音符头，一音一数字。"""
    return event


# ---------- 第 6 步：渲染叠印 ----------

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 220, 0)          # 简谱数字（绿色，方便与原黑字区分但不过分刺眼）
UPPER_COLOR = (220, 100, 35)  # BGR: 蓝色，按大谱表上行区分，不随谱号变化
LOWER_COLOR = (45, 165, 30)   # BGR: 绿色


def _digit_scale(s: float) -> float:
    return max(0.7, s / 16.0)


def _staff_first_music_x(page: Page, staff: Staff) -> int | None:
    """谱表演奏区起始 x（谱头之前不标注）。"""
    start = staff_music_start_x(page, staff)
    return int(round(start)) if start is not None else None


def render(
    page: Page,
    out_path: Path | None = None,
    debug: bool = False,
    use_jev: bool | None = None,
    placements_out: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    s = page.systems[0].treble.spacing if page.systems else 11.0
    scale = _digit_scale(s)
    occupied = np.zeros(vis.shape[:2], dtype=bool)

    if not page.systems:
        return vis
    if use_jev is None:
        use_jev = jev.jev_available()
    H = vis.shape[0]
    for si, sys in enumerate(page.systems):
        staves = sys.all_staves()
        bounds = (sys.treble.top, sys.treble.bottom, sys.bass.top, sys.bass.bottom)
        mid_y = (sys.treble.bottom + sys.bass.top) / 2.0
        for i, staff in enumerate(staves):
            s = staff.spacing
            scale = _digit_scale(s)
            # 人声等附加谱表只在自己上方标注，不借用钢琴两行之间的空隙。
            is_top = staff is sys.treble or i >= 2
            if i == 0 and si == 0:
                y_ceiling = 0
            elif i == 0:
                y_ceiling = page.systems[si - 1].all_staves()[-1].bottom + int(0.4 * s)
            else:
                y_ceiling = staves[i - 1].bottom + int(0.35 * s)
            if i + 1 < len(staves):
                below_limit = staves[i + 1].top - int(0.3 * s)
            elif si + 1 < len(page.systems):
                below_limit = page.systems[si + 1].treble.top - int(0.3 * s)
            else:
                below_limit = H
            evs = [ev for ev in page.chords if ev.heads[0].staff is staff]
            _render_group(
                vis, page.binary, evs, staff,
                top=is_top, s=s, scale=scale,
                mid_y=mid_y, bounds=bounds, occupied=occupied,
                gray=page.image, page=page, use_jev=use_jev,
                y_ceiling=y_ceiling, below_limit=below_limit,
                staff_key=f"{si}_{i}",
                placements_out=placements_out,
            )

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


def _fit_compat_stack(
    H: int,
    W: int,
    prefer_y: int,
    prefer_x_off: int,
    dir_sign: int,
    members,
    step: int,
    max_y_span: int | None = None,
    max_x_span: int | None = None,
):
    """Fit the complete annotation stack inside image bounds without reversing direction."""
    if _stack_fits_image(H, W, prefer_y, prefer_x_off, dir_sign, members, step):
        return prefer_y, prefer_x_off, dir_sign

    def score(cand_y: int, x_off: int) -> tuple[int, int]:
        return (abs(cand_y - prefer_y), abs(x_off - prefer_x_off))

    def within_span(cand_y: int, x_off: int) -> bool:
        if max_y_span is not None and abs(cand_y - prefer_y) > max_y_span:
            return False
        if max_x_span is not None and abs(x_off - prefer_x_off) > max_x_span:
            return False
        return True

    best = None
    y_limit = max_y_span if max_y_span is not None else H
    for k in range(y_limit + 1):
        cand_ys = [prefer_y] if k == 0 else [c for c in (prefer_y + k, prefer_y - k) if 0 <= c < H]
        for cand_y in cand_ys:
            if not within_span(cand_y, prefer_x_off):
                continue
            left, top, right, bottom = _stack_bounds_xy(cand_y, prefer_x_off, dir_sign, members, step)
            dx_opts = {0}
            if left < 0:
                dx_opts.add(-left)
            if right > W:
                dx_opts.add(W - right)
            for dx in dx_opts:
                x_off = prefer_x_off + dx
                if not within_span(cand_y, x_off):
                    continue
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


def _build_render_members(ev: ChordEvent, cx: int, scale: float, s: float):
    dot_r = max(1, int(s * 0.12))
    members = []
    for h in reversed(ev.heads):
        tw, th, bl = _measure(f"{h.digit}", scale)
        ptw = pth = 0
        if h.prefix:
            pscale = max(0.3, scale * 0.6)
            ptw, pth, _ = _measure(h.prefix, pscale)
        up_dots = max(0, h.dots)
        dn_dots = max(0, -h.dots)
        top_off = th // 2 + 1 + (up_dots * (2 * dot_r + 2) + dot_r + 1 if up_dots else 0)
        bot_off = th // 2 + 1 + (dn_dots * (2 * dot_r + 2) + dot_r + 2 if dn_dots else 0)
        right_ext = tw + (ptw + 3 if ptw else 0) + (5 if h.dotted else 0)
        members.append((h, cx - tw // 2, tw, th, ptw, pth, top_off, bot_off, right_ext))
    return members, dot_r


def _stack_bounds_from_members(ay: int, dir_sign: int, members, step: int) -> tuple[int, int]:
    top = bottom = None
    for i, (_h, _x, _tw, _th, _ptw, _pth, to, bo, _re) in enumerate(members):
        yy = ay + dir_sign * i * step
        t, b = yy - to, yy + bo
        top = t if top is None else min(top, t)
        bottom = b if bottom is None else max(bottom, b)
    return top, bottom


def _annotation_clearance(s: float) -> int:
    return int(round(0.45 * s))


def _stack_outside_staff(staff: Staff, s: float, top: int, bottom: int) -> bool:
    clearance = _annotation_clearance(s)
    return bottom <= staff.top - clearance or top >= staff.bottom + clearance


def _stack_above_staff(staff: Staff, s: float, top: int, bottom: int) -> bool:
    return bottom <= staff.top - _annotation_clearance(s)


def _stack_below_staff(staff: Staff, s: float, top: int, bottom: int) -> bool:
    return top >= staff.bottom + _annotation_clearance(s)


def _staff_symbol_hard_right(staff: Staff) -> float:
    """谱号/调号/拍号区宽度（高音谱号比低音谱号更宽）。"""
    if staff.clef == "treble":
        return 11.5 * staff.spacing
    return 8.5 * staff.spacing


def _head_on_barline(black: np.ndarray, n: NoteHead, staff: Staff, s: float) -> bool:
    """符头中心落在贯通谱表的竖线（小节线/反复线）上 → 不是音符。"""
    if n.staff is not staff:
        return False
    col = int(round(n.x))
    if col < 0 or col >= black.shape[1]:
        return False
    # 小节线应贯通五条谱线；2.2s 的任意竖段也可能是和弦内部或符干。
    y0 = staff.top
    y1 = staff.bottom + 1
    y0, y1 = max(0, y0), min(black.shape[0], y1)
    col_ink = black[y0:y1, col] > 0
    if col_ink.size == 0:
        return False
    best = cur = 0
    for hit in col_ink:
        if hit:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best >= max(1, int(0.9*(y1-y0)))


def _note_side_anchor(
    top_head_y: int,
    lowest_head_y: int,
    side: str,
    s: float,
    th: int,
) -> tuple[int, int]:
    """side: 'above' | 'below' — anchor y and stack growth sign for the chord column."""
    gap = _annotation_clearance(s)
    if side == "below":
        # 最低音贴近最低符头下方，更高音向上叠（dir_sign=-1）。
        return lowest_head_y + gap + th // 2 + 2, -1
    return top_head_y - gap - th // 2 - 2, -1


def _find_stack_y(
    y_row: int,
    dir_sign: int,
    members,
    step: int,
    staff: Staff,
    s: float,
    black,
    occupied,
    H: int,
    W: int,
    x_off: int,
    y_ceiling: int,
    below_limit: int,
    max_span: int,
    outside_only: bool,
    grp_clear,
    in_gap=None,
    grow_down: bool = False,
    must_above_staff: bool = False,
    must_below_staff: bool = False,
) -> int | None:
    """Search along dir_sign from y_row for a clear stack position."""
    step_y = 2 if grow_down else -2
    cand = y_row
    best_inside = None
    for _ in range(max_span // 2 + 1):
        t, b = _stack_bounds_from_members(cand, dir_sign, members, step)
        if grow_down and b > below_limit:
            break
        if not grow_down and t < y_ceiling:
            break
        if not grp_clear(cand, x_off, dir_sign):
            cand += step_y
            continue
        if in_gap is not None and not in_gap(cand):
            cand += step_y
            continue
        if must_above_staff and not _stack_above_staff(staff, s, t, b):
            cand += step_y
            continue
        if must_below_staff and not _stack_below_staff(staff, s, t, b):
            cand += step_y
            continue
        outside = _stack_outside_staff(staff, s, t, b)
        if outside_only and not outside:
            if best_inside is None and not must_above_staff and not must_below_staff:
                best_inside = cand
            cand += step_y
            continue
        return cand
    return best_inside if not outside_only else None




def _jev_event_payloads(
    evs: list[ChordEvent],
    staff: Staff,
    top: bool,
    s: float,
    gray: np.ndarray | None,
    page: Page | None,
) -> list[dict[str, Any]]:
    staff_left = _staff_left_edge(gray, staff) if gray is not None else None
    first_music_x = _staff_first_music_x(page, staff) if page is not None else None
    payloads: list[dict[str, Any]] = []
    for idx, ev in enumerate(evs):
        rel_x = None
        if staff_left is not None:
            rel_x = (ev.x - staff_left) / s
        head = ev.heads[0]
        payloads.append(
            {
                "id": idx,
                "x": ev.x,
                "y": head.y,
                "digit": head.digit,
                "filled": head.filled,
                "staff_clef": staff.clef,
                "is_treble_staff": top,
                "x_after_staff_left_in_spacing": rel_x,
                "first_music_x": first_music_x,
                "candidates": {
                    "above": top,
                    "gap": False,
                    "below": not top,
                    "right": False,
                },
            }
        )
    return payloads


def _render_group(
    vis,
    black,
    evs,
    staff,
    top,
    s,
    scale,
    mid_y=0,
    bounds=None,
    occupied=None,
    gray=None,
    page=None,
    use_jev: bool = False,
    y_ceiling: int = 0,
    below_limit: int | None = None,
    staff_key: str = "0_0",
    placements_out: list[dict[str, Any]] | None = None,
):
    """按大谱表位置排版：上行蓝色向上，下行绿色放在整条谱表下方。"""
    H, W = vis.shape[:2]
    below_limit = min(H, below_limit if below_limit is not None else H)
    if occupied is None:
        occupied = np.zeros((H, W), dtype=bool)
    color = UPPER_COLOR if top else LOWER_COLOR
    decisions = {}
    if use_jev and jev.jev_available():
        try:
            choices = jev.decide_events(_jev_event_payloads(evs, staff, top, s, gray, page))
            decisions = {id(ev): choice for ev, choice in zip(evs, choices)}
        except Exception:
            jev.log.exception("Jev unavailable; using geometric annotation rules")
    seen = set()
    ordered = sorted(evs, key=lambda event: event.x)
    for ev_index, ev in enumerate(ordered):
        if decisions.get(id(ev), {}).get("skip"):
            continue
        heads = []
        for head in sorted(ev.heads, key=lambda h: h.y):
            if id(head) in seen:
                continue
            seen.add(id(head))
            if page is not None and (head.x < staff_playable_min_x(page, staff)
                                     or _head_in_clef_glyph(page, head, staff)):
                continue
            heads.append(head)
        if not heads:
            continue
        # 和弦整列排版，禁止用“离另一个头更近”剔除和弦中的低音。
        column = ChordEvent(x=int(round(ev.x)), heads=heads)
        best = None
        neighbor_gaps = [abs(ev.x-ordered[j].x) for j in (ev_index-1, ev_index+1)
                         if 0 <= j < len(ordered) and abs(ev.x-ordered[j].x) > 0.5*s]
        preferred_scale = scale
        if neighbor_gaps:
            # 密集段先按相邻列间距选择共同可读字号，不让先画的大字
            # 抢占后一个音的位置。这里只限制字宽，不强制整行基线。
            while preferred_scale > 0.4 and _measure('4', preferred_scale)[0]+4 > min(neighbor_gaps):
                preferred_scale = max(0.4, preferred_scale*0.9)
        for local_scale in (preferred_scale, max(0.3, preferred_scale*0.82), max(0.3, preferred_scale*0.65), max(0.28, preferred_scale*0.5)):
            members, _ = _build_render_members(column, column.x, local_scale, s)
            step = max(m[6]+m[7] for m in members) + max(2, int(0.2*s))
            zero_top, zero_bottom = _stack_bounds_from_members(0, -1, members, step)
            gap = max(2, int(round(0.45*s)))
            if top:
                edge = min(staff.top-gap, min(h.y for h in heads)-int(0.7*s))
                base = edge-zero_bottom
                max_travel = min(int(8*s), max(0, base+zero_top-y_ceiling))
            else:
                edge = max(staff.bottom+gap, max(h.y for h in heads)+int(0.7*s))
                base = edge-zero_top
                max_travel = min(int(8*s), max(0, below_limit-base-zero_bottom))
            for distance in range(0, max_travel+1, 2):
                ay = base-distance if top else base+distance
                for dx in (0, int(0.4*s), -int(0.4*s), int(0.8*s), -int(0.8*s)):
                    if ay+zero_top < y_ceiling or ay+zero_bottom > below_limit:
                        continue
                    if any(_box_overlaps(black, occupied, H, W, m[1]+dx, m[8],
                                         ay-i*step, m[6], m[7])
                           for i, m in enumerate(members)):
                        continue
                    # 就近与可读字号共同评分：不能为了坚持最大字号，
                    # 把本可并排的相邻数字一层层推远。
                    cost = distance + 4*abs(dx) + (1-local_scale/scale)*2*s
                    if best is None or cost < best[0]:
                        best = (cost, ay, dx, members, step, local_scale)
        if best is None:
            continue
        _, ay, dx, members, step, draw_scale = best
        for i, member in enumerate(members):
            head = member[0]
            center_y = ay-i*step
            _draw_note_label(vis, occupied, [member], center_y, dx, s, draw_scale,
                             staff_key, placements_out, head, color=color)


def _draw_note_label(vis, occupied, members, y_best, x_off, s, scale,
                     staff_key, placements_out, head, color=COLOR):
    H, W = vis.shape[:2]
    dot_r = max(1, int(s*0.12))
    dir_sign, step = -1, 0
    for i, (h, x, tw, th, ptw, pth, to, bo, re) in enumerate(members):
        x = x + x_off
        y = y_best + dir_sign * i * step
        cv2.putText(vis, f"{h.digit}", (x, y + th // 2), FONT, scale, color, 2, cv2.LINE_AA)
        if h.prefix:
            pscale = max(0.3, scale * 0.6)
            px = x + tw + 3
            py = y - th // 2 + pth - 1
            cv2.putText(vis, h.prefix, (px, py), FONT, pscale, color, 1, cv2.LINE_AA)
        # 八度点：高八度在数字正上方，低八度在数字正下方
        dcx = x + tw // 2
        if h.dots > 0:
            dcy = y - th // 2 - dot_r - 2
            for k in range(h.dots):
                cv2.circle(vis, (dcx, dcy - k * (2 * dot_r + 2)), dot_r, color, -1)
        elif h.dots < 0:
            dcy = y + th // 2 + dot_r + 2
            for k in range(-h.dots):
                cv2.circle(vis, (dcx, dcy + k * (2 * dot_r + 2)), dot_r, color, -1)
        # 时值记号
        _draw_duration(vis, h, x, y, tw, th, s, scale, color=color)
        # 记录占用区域（数字+八度点范围）
        x0 = max(0, x - 2)
        x1 = min(W, x + re + 2)
        y0 = max(0, y - to)
        y1 = min(H, y + bo)
        occupied[y0:y1, x0:x1] = True
        if placements_out is not None:
            placements_out.append(
                {
                    "staff_key": staff_key,
                    "x": int(x + tw // 2),
                    "y": int(y + th // 2),
                    "digit": int(h.digit),
                    "dots": int(h.dots),
                    "prefix": h.prefix or "",
                    "top": int(y0),
                    "bottom": int(y1),
                    "head_x": int(head.x),
                    "head_y": int(head.y),
                    "color": "#%02x%02x%02x" % (color[2], color[1], color[0]),
                    "font_height": int(th),
                }
            )


def _draw_duration(vis, h, x, y, tw, th, s, scale, color=COLOR):
    """附点：时值含 .5 且非十六分 → 数字右侧圆点。"""
    if h.dotted and h.dur >= 1.0:
        cv2.circle(vis, (x + tw + 4, y + th // 2), max(1, int(s * 0.12)), color, -1)


# ---------- 调试图 ----------

DEBUG_SLICE_NAMES = (
    "00_binary",
    "01_staffs",
    "02_noteheads",
    "03_recovered",
    "04_purged",
    "05_accidentals",
    "06_durations",
    "07_jianpu",
    "08_chords",
    "09_render",
)


def _save_debug_vis(vis: np.ndarray, out_path: Path, title: str = "") -> None:
    if title:
        cv2.putText(vis, title, (8, 22), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def _dump_debug(
    debug_dir: Path | None,
    name: str,
    page: Page,
    drawer,
) -> None:
    if debug_dir is None:
        return
    drawer(page, debug_dir / f"{name}.jpg")


def debug_draw_binary(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.binary, cv2.COLOR_GRAY2BGR)
    _save_debug_vis(vis, out_path, "00 binary")


def debug_draw_noteheads(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    for n in page.noteheads:
        c = (0, 0, 255) if n.filled else (255, 0, 0)
        cv2.rectangle(vis, (n.x - 9, n.y - 7), (n.x + 9, n.y + 7), c, 1)
        pf = {0: "", -1: "b", 1: "#"}.get(n.alter, "?")
        letter = LETTERS[n.letter] if 0 <= n.letter < 7 else "?"
        cv2.putText(
            vis, f"{pf}{letter}{n.octave}", (n.x - 9, n.y - 10),
            FONT, 0.35, (0, 128, 0), 1,
        )
    _save_debug_vis(vis, out_path, out_path.stem)


def debug_draw_durations(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    for n in page.noteheads:
        dur = {0: 4, 1: 8, 2: 16}.get(n.beams, 0) if n.filled else (2 if n.beams else 1)
        tag = f"q{dur}" + ("." if n.dotted else "")
        if not n.filled:
            tag = "half" if n.beams else "whole"
        cv2.putText(vis, tag, (n.x - 10, n.y - 10), FONT, 0.35, (0, 0, 255), 1)
    _save_debug_vis(vis, out_path, "06 durations")


def debug_draw_staffs(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    staves = collect_staves(page)
    if not staves:
        staves = [st for sys in page.systems for st in sys.all_staves()]
    for staff in staves:
        for y in staff.lines:
            cv2.line(vis, (0, y), (vis.shape[1], y), (0, 0, 255), 1)
        cv2.rectangle(
            vis, (0, staff.top - 4), (vis.shape[1] - 1, staff.bottom + 4),
            (0, 180, 0), 1,
        )
        cv2.putText(
            vis, staff.clef, (10, staff.top - 8), FONT, 0.4, (0, 180, 0), 1,
        )
    for i, sys in enumerate(page.systems):
        x = 6
        y0, y1 = sys.treble.top, sys.bass.bottom
        cv2.line(vis, (x, y0), (x, y1), (255, 0, 0), 2)
        cv2.putText(vis, f"sys{i}", (x + 4, y0 - 6), FONT, 0.5, (255, 0, 0), 1)
    _save_debug_vis(vis, out_path, "01 staffs")


def debug_draw_accidentals(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    for n in page.noteheads:
        c = (0, 0, 255) if n.filled else (255, 0, 0)
        cv2.rectangle(vis, (n.x - 9, n.y - 7), (n.x + 9, n.y + 7), c, 1)
        if n.alter:
            tag = "#" if n.alter > 0 else "b"
            cv2.putText(vis, tag, (n.x - 18, n.y + 4), FONT, 0.45, (255, 0, 255), 1)
    for x, y, alter in page.accidentals:
        tag = {1: "#", -1: "b", 0: "n"}.get(alter, "?")
        cv2.circle(vis, (x, y), 6, (255, 0, 255), 1)
        cv2.putText(vis, tag, (x + 6, y + 4), FONT, 0.4, (255, 0, 255), 1)
    _save_debug_vis(vis, out_path, "05 accidentals")


def debug_draw_jianpu(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    for n in page.noteheads:
        c = (0, 0, 255) if n.filled else (255, 0, 0)
        cv2.rectangle(vis, (n.x - 9, n.y - 7), (n.x + 9, n.y + 7), c, 1)
        cv2.putText(
            vis, f"{n.prefix}{n.digit}", (n.x - 6, n.y - 10),
            FONT, 0.45, (0, 220, 0), 2,
        )
    _save_debug_vis(vis, out_path, "07 jianpu")


def debug_draw_chords(page: Page, out_path: Path) -> None:
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    for ev in page.chords:
        if not ev.heads:
            continue
        xs = [h.x for h in ev.heads]
        ys = [h.y for h in ev.heads]
        x0, x1 = min(xs) - 12, max(xs) + 12
        y0, y1 = min(ys) - 12, max(ys) + 12
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 165, 255), 1)
        digits = "".join(str(h.digit) for h in ev.heads)
        cv2.putText(vis, digits, (x0, max(12, y0 - 4)), FONT, 0.45, (0, 220, 0), 1)
        for h in ev.heads:
            cv2.circle(vis, (h.x, h.y), 4, (0, 165, 255), 1)
    _save_debug_vis(vis, out_path, "08 chords")


def debug_draw_render(page: Page, out_path: Path, vis: np.ndarray) -> None:
    _save_debug_vis(vis.copy(), out_path, "09 render")


# ---------- Canvas 场景（规范化矢量五线谱 + 简谱） ----------

CANVAS_LAYOUT = {
    "width": 960,
    "pad_x": 56,
    "pad_y": 40,
    "line_spacing": 11,
    "grand_staff_gap": 16,
    "system_gap": 40,
}


def build_canvas_scene(page: Page, placements: list[dict[str, Any]]) -> dict[str, Any]:
    """将识别结果映射为前端 Canvas 场景：统一线距的五线谱 + 简谱数字。"""
    cfg = CANVAS_LAYOUT
    w = cfg["width"]
    if not page.systems:
        return {"width": w, "height": 200, "systems": [], "labels": [], "notes": []}

    xs: list[float] = [float(p["x"]) for p in placements]
    for n in page.noteheads:
        xs.append(float(n.x))
    if not xs:
        xs = [0.0, float(page.image.shape[1])]
    x_min, x_max = min(xs), max(xs)
    x_span = max(1.0, x_max - x_min)
    inner_w = w - 2 * cfg["pad_x"]

    def map_x(x: float) -> float:
        return cfg["pad_x"] + (x - x_min) / x_span * inner_w

    staff_map: dict[str, list[int]] = {}
    systems_out: list[dict[str, Any]] = []
    y_cursor = float(cfg["pad_y"])
    sp = cfg["line_spacing"]
    staff_height = sp * 4

    for si, sys in enumerate(page.systems):
        block: dict[str, Any] = {"index": si, "staves": []}
        for i, st in enumerate(sys.all_staves()):
            key = f"{si}_{i}"
            local = [p for p in placements if p.get("staff_key") == key]
            label_top = min([st.top] + [p.get("top", p["y"]-20) for p in local])
            label_bottom = max([st.bottom] + [p.get("bottom", p["y"]+12) for p in local])
            factor = sp/st.spacing
            # Reserve each staff's actual annotation extent before normalizing;
            # tall chords must not be cropped or collide with the next system.
            y_cursor += max(0, st.top-label_top)*factor + 8
            lines = [int(round(y_cursor + k * sp)) for k in range(5)]
            staff_map[key] = lines
            block["staves"].append(
                {"key": key, "clef": st.clef, "lines": lines}
            )
            y_cursor += staff_height
            y_cursor += max(0, label_bottom-st.bottom)*factor + 8
            if st is sys.treble:
                y_cursor += cfg["grand_staff_gap"]
            else:
                y_cursor += cfg["system_gap"]
        systems_out.append(block)

    height = int(y_cursor + cfg["pad_y"])

    def map_y_on_staff(st: Staff, key: str, src_y: float) -> float:
        lines = staff_map[key]
        if st.bottom <= st.top:
            return float(lines[2])
        t = (src_y - st.top) / (st.bottom - st.top)
        return lines[0] + t * (lines[4] - lines[0])

    staff_by_key: dict[str, Staff] = {}
    for si, sys in enumerate(page.systems):
        for i, st in enumerate(sys.all_staves()):
            staff_by_key[f"{si}_{i}"] = st

    notes_out = []
    for n in page.noteheads:
        if n.staff is None:
            continue
        key = None
        for k, st in staff_by_key.items():
            if st is n.staff:
                key = k
                break
        if key is None:
            continue
        notes_out.append(
            {
                "x": map_x(n.x),
                "y": map_y_on_staff(n.staff, key, n.y),
                "filled": bool(n.filled),
            }
        )

    labels_out = []
    for p in placements:
        key = p.get("staff_key", "0_0")
        st = staff_by_key.get(key)
        if st is None:
            cy = float(p["y"])
        else:
            cy = map_y_on_staff(st, key, float(p["y"]))
        baseline_y = cy
        labels_out.append(
            {
                "x": map_x(float(p["x"])),
                "y": baseline_y,
                "digit": p["digit"],
                "dots": p.get("dots", 0),
                "prefix": p.get("prefix", ""),
                "color": p.get("color", "#1ea52d"),
                "font_size": max(8, p.get("font_height", 15)*sp/st.spacing) if st else 15,
            }
        )

    return {
        "width": w,
        "height": height,
        "layout": cfg,
        "systems": systems_out,
        "labels": labels_out,
        "notes": notes_out,
    }


def process_image_with_scene(
    gray: np.ndarray,
    binary: np.ndarray,
    quiet: bool = True,
    use_jev: bool | None = None,
    debug_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """完整流水线，并返回 Canvas 场景 JSON（同时保留位图叠印供兼容）。"""
    placements: list[dict[str, Any]] = []
    vis, page = _process_image(gray, binary, quiet, use_jev, debug_dir, placements)
    return vis, build_canvas_scene(page, placements)


# ---------- 入口 ----------

def process_image(
    gray: np.ndarray,
    binary: np.ndarray,
    quiet: bool = False,
    use_jev: bool | None = None,
    debug_dir: Path | None = None,
) -> np.ndarray:
    """内存版完整流水线：输入灰度图 + 二值图，返回叠印后的 BGR 图像。"""
    vis, _ = _process_image(gray, binary, quiet, use_jev, debug_dir)
    return vis


def _process_image(
    gray: np.ndarray,
    binary: np.ndarray,
    quiet: bool = False,
    use_jev: bool | None = None,
    debug_dir: Path | None = None,
    placements_out: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, Page]:
    """位图与 Canvas 共用同一识别流程和逐阶段调试输出。"""
    if use_jev is None:
        use_jev = jev.jev_available()
    jev.log.info(
        "process_image: use_jev=%s jev_available=%s image=%dx%d debug_dir=%s",
        use_jev,
        jev.jev_available(),
        gray.shape[1],
        gray.shape[0],
        debug_dir,
    )
    page = Page(image=gray, binary=binary)

    def dump(name: str, drawer) -> None:
        _dump_debug(debug_dir, name, page, drawer)

    dump("00_binary", debug_draw_binary)

    lines = detect_staff_lines(page.image)
    staves = group_staves(lines)
    page.staves = staves
    page.systems = pair_systems(staves, gray)
    detect_staff_clefs(page)
    if not quiet:
        print(f"谱线 {len(lines)} 条 → 谱表 {len(staves)} 个 → 系统 {len(page.systems)} 个")
    dump("01_staffs", debug_draw_staffs)

    detect_noteheads(page)
    if not quiet:
        print(f"音符头 {len(page.noteheads)} 个（实心 {sum(n.filled for n in page.noteheads)} "
              f"/ 空心 {sum(not n.filled for n in page.noteheads)}）")
    assign_pitches(page)
    dump("02_noteheads", debug_draw_noteheads)

    _recover_stacked_hollow_heads(page)
    _dedupe_noteheads(page)
    assign_pitches(page)
    dump("03_recovered", debug_draw_noteheads)

    _remove_clef_zone(page)
    _remove_leading_symbols(page)
    purge_staff_header_noteheads(page)
    dump("04_purged", debug_draw_noteheads)

    detect_accidentals(page)
    _clear_leading_accidentals(page)
    if not quiet:
        print(f"检测到临时记号 {len(page.accidentals)} 个")
    dump("05_accidentals", debug_draw_accidentals)

    analyze_durations(page)
    if not quiet:
        print("时值分析完成")
    dump("06_durations", debug_draw_durations)

    to_jianpu(page)
    purge_staff_header_noteheads(page)
    build_chords(page)
    if not quiet:
        counter = {}
        for n in page.noteheads:
            key = (n.prefix, n.digit, n.dots)
            counter[key] = counter.get(key, 0) + 1
        print("简谱分布(前20):", sorted(counter.items(), key=lambda x: -x[1])[:20])
    dump("07_jianpu", debug_draw_jianpu)
    dump("08_chords", debug_draw_chords)

    vis = render(page, use_jev=use_jev, placements_out=placements_out)
    if debug_dir is not None:
        debug_draw_render(page, debug_dir / "09_render.jpg", vis)
    return vis, page


def run(
    input_path: str,
    output_path: str,
    debug_dir: Path,
    stage: int,
    use_jev: bool | None = None,
):
    if use_jev is None:
        use_jev = jev.jev_available()
    gray, binary = load_binary(input_path)
    page = Page(image=gray, binary=binary)

    lines = detect_staff_lines(page.image)
    staves = group_staves(lines)
    page.staves = staves
    page.systems = pair_systems(staves, gray)
    detect_staff_clefs(page)
    print(f"谱线 {len(lines)} 条 → 谱表 {len(staves)} 个 → 系统 {len(page.systems)} 个")

    if stage <= 1:
        debug_draw_staffs(page, debug_dir / "01_staffs.jpg")
        return

    detect_noteheads(page)
    print(f"音符头 {len(page.noteheads)} 个（实心 {sum(n.filled for n in page.noteheads)} "
          f"/ 空心 {sum(not n.filled for n in page.noteheads)}）")
    assign_pitches(page)
    _recover_stacked_hollow_heads(page)
    _dedupe_noteheads(page)
    assign_pitches(page)
    _remove_clef_zone(page)
    _remove_leading_symbols(page)
    purge_staff_header_noteheads(page)
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
    purge_staff_header_noteheads(page)
    build_chords(page)
    counter = {}
    for n in page.noteheads:
        key = (n.prefix, n.digit, n.dots)
        counter[key] = counter.get(key, 0) + 1
    print("简谱分布(前20):", sorted(counter.items(), key=lambda x: -x[1])[:20])

    render(page, Path(output_path), use_jev=use_jev)
    print(f"输出: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="在五线谱图片上叠印简谱")
    ap.add_argument("input", help="输入图片路径")
    ap.add_argument("--output", "-o", help="输出图片路径")
    ap.add_argument("--debug-dir", default="debug", help="调试图输出目录")
    ap.add_argument("--stage", type=int, default=6, help="运行到第几阶段")
    ap.add_argument(
        "--jev",
        action="store_true",
        help="使用 Jev 判断标注位置与行首曲谱头（需 TYPESAFE_API_KEY）",
    )
    ap.add_argument(
        "--no-jev",
        action="store_true",
        help="禁用 Jev（即使已配置 API Key）",
    )
    args = ap.parse_args()

    use_jev: bool | None = None
    if args.jev:
        use_jev = True
    elif args.no_jev:
        use_jev = False

    inp = args.input
    if not args.output:
        p = Path(inp)
        args.output = str(p.with_name(p.stem + "_简谱标注.jpg"))
    run(inp, args.output, Path(args.debug_dir), args.stage, use_jev=use_jev)


if __name__ == "__main__":
    main()
