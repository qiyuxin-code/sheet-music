import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import annotate as an


def _green_mask(vis: np.ndarray) -> np.ndarray:
    # int32 avoids uint8 overflow when differencing BGR channels.
    r = vis[:, :, 0].astype(np.int32)
    g = vis[:, :, 1].astype(np.int32)
    b = vis[:, :, 2].astype(np.int32)
    # Legacy helper name: generic placement tests now cover blue AND green ink.
    return ((g > r + 40) & (g > b + 40)) | ((r > g + 40) & (r > b + 40))


class PlacementPriorityTests(unittest.TestCase):
    """Ordered placement decisions, asserted through full annotation boxes."""

    SCALE = 0.7
    S = 10

    def _members(self, event: an.ChordEvent):
        cx = int(round(event.x))
        dot_r = max(1, int(self.S * 0.12))
        members = []
        for head in reversed(event.heads):
            tw, th, _ = an._measure(f"{head.digit}", self.SCALE)
            ptw = pth = 0
            if head.prefix:
                pscale = max(0.3, self.SCALE * 0.6)
                ptw, pth, _ = an._measure(head.prefix, pscale)
            up_dots = max(0, head.dots)
            down_dots = max(0, -head.dots)
            top_off = th // 2 + 1 + (
                up_dots * (2 * dot_r + 2) + dot_r + 1 if up_dots else 0
            )
            bottom_off = th // 2 + 1 + (
                down_dots * (2 * dot_r + 2) + dot_r + 2 if down_dots else 0
            )
            right_ext = tw + (ptw + 3 if ptw else 0) + 4
            members.append(
                (head, cx - tw // 2, tw, th, ptw, pth,
                 top_off, bottom_off, right_ext)
            )
        return members

    def _annotation_bounds(self, added: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(added)
        self.assertGreater(len(xs), 0, "expected a computed annotation box")
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _render_bass(
        self,
        black: np.ndarray,
        occupied: np.ndarray,
        event: an.ChordEvent,
        staff: an.Staff,
        bounds: tuple[int, int, int, int],
    ) -> np.ndarray:
        before = occupied.copy()
        vis = np.full((*black.shape, 3), 255, dtype=np.uint8)
        an._render_group(
            vis,
            black,
            [event],
            staff,
            top=False,
            s=self.S,
            scale=self.SCALE,
            mid_y=(bounds[1] + bounds[2]) / 2,
            bounds=bounds,
            occupied=occupied,
        )
        return occupied & ~before

    def _single_bass_fixture(self):
        black = np.zeros((160, 180), dtype=np.uint8)
        occupied = np.zeros_like(black, dtype=bool)
        bass = an.Staff(
            lines=[70, 80, 90, 100, 110], thickness=1, spacing=self.S
        )
        head = an.NoteHead(
            x=60, y=90, filled=True, staff=bass, digit=2
        )
        event = an.ChordEvent(x=head.x, heads=[head])
        return black, occupied, bass, head, event, (10, 50, 70, 110)

    def test_bass_stays_below_entire_staff(self):
        black, occupied, bass, head, event, bounds = self._single_bass_fixture()

        added = self._render_bass(black, occupied, event, bass, bounds)

        _, annotation_top, _, _ = self._annotation_bounds(added)
        self.assertGreater(annotation_top, bass.bottom)

    def test_bass_uses_nearby_clear_position_when_above_is_blocked(self):
        black, occupied, bass, head, event, bounds = self._single_bass_fixture()
        members = self._members(event)
        th = members[0][3]
        y_row = head.y - int(self.S * 0.6) - th // 2 - 2
        near_left, near_top, near_right, near_bottom = an._stack_bounds_xy(
            y_row, 0, -1, members, max(int(th * 0.95), 1)
        )
        black[near_top:near_bottom, near_left:near_right] = 255

        added = self._render_bass(black, occupied, event, bass, bounds)

        self.assertFalse((added & (black > 0)).any())
        _, annotation_top, _, annotation_bottom = self._annotation_bounds(added)
        self.assertLessEqual(abs((annotation_top+annotation_bottom)/2-head.y), 3.5*self.S)

    def test_treble_prefers_clear_position_above_event(self):
        black = np.zeros((120, 160), dtype=np.uint8)
        occupied = np.zeros_like(black, dtype=bool)
        treble = an.Staff(
            lines=[40, 50, 60, 70, 80], thickness=1, spacing=self.S
        )
        head = an.NoteHead(
            x=60, y=60, filled=True, staff=treble, digit=5
        )
        event = an.ChordEvent(x=head.x, heads=[head])
        before = occupied.copy()
        vis = np.full((*black.shape, 3), 255, dtype=np.uint8)

        an._render_group(
            vis,
            black,
            [event],
            treble,
            top=True,
            s=self.S,
            scale=self.SCALE,
            bounds=(40, 80, 90, 110),
            occupied=occupied,
        )

        _, _, _, annotation_bottom = self._annotation_bounds(
            occupied & ~before
        )
        self.assertLessEqual(annotation_bottom, head.y)

    def test_bass_below_preserves_low_to_high_vertical_order_and_bounds(self):
        h, w = 170, 180
        black = np.zeros((h, w), dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        bass = an.Staff(
            lines=[70, 80, 90, 100, 110], thickness=1, spacing=self.S
        )
        high = an.NoteHead(
            x=60, y=90, filled=True, staff=bass, digit=1
        )
        low = an.NoteHead(
            x=60, y=100, filled=True, staff=bass, digit=8, prefix="#"
        )
        event = an.ChordEvent(x=60, heads=[high, low])
        members = self._members(event)
        th = members[0][3]
        step = max(int(th * 0.95), 1)
        clearance_top = bass.bottom + int(self.S * 0.6)
        center_left, _, center_right, _ = an._stack_bounds_xy(
            high.y, 0, -1, members, step
        )
        black[:clearance_top, center_left:center_right] = 255
        vis = np.full((h, w, 3), 255, dtype=np.uint8)

        an._render_group(
            vis,
            black,
            [event],
            bass,
            top=False,
            s=self.S,
            scale=self.SCALE,
            mid_y=60,
            bounds=(10, 50, 70, 110),
            occupied=occupied,
        )

        _, actual_top, _, actual_bottom = self._annotation_bounds(occupied)
        self.assertFalse((occupied & (black > 0)).any())
        self.assertLessEqual(actual_bottom, h)
        green = _green_mask(vis)
        low_prefix_x = members[0][1] + members[0][2] + 3
        low_ys, _ = np.where(green[:, low_prefix_x:])
        high_ys, _ = np.where(
            green[:, members[1][1]:members[1][1] + members[1][2]]
        )
        self.assertGreater(len(low_ys), 0, "low member must be drawn")
        self.assertGreater(len(high_ys), 0, "high member must be drawn")
        self.assertGreater(
            int(np.max(low_ys)),
            int(np.min(high_ys)),
            "lower-pitch member must be lower on the rendered page",
        )

    def test_scans_right_when_first_horizontal_offset_is_blocked(self):
        black, occupied, bass, head, event, bounds = self._single_bass_fixture()
        members = self._members(event)
        th = members[0][3]
        step = max(int(th * 0.95), 1)
        first_x_off = int(self.S * 1.8)
        first_left, _, first_right, _ = an._stack_bounds_xy(
            head.y, first_x_off, -1, members, step
        )
        center_left, _, center_right, _ = an._stack_bounds_xy(
            head.y, 0, -1, members, step
        )
        black[:, center_left:center_right] = 255
        black[:, first_left:first_right] = 255

        added = self._render_bass(black, occupied, event, bass, bounds)

        self.assertFalse(added.any(), "blocked column must not move to an unrelated note")

    def test_farther_right_candidate_can_end_exactly_at_image_edge(self):
        h, w = 120, 81
        black = np.zeros((h, w), dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        bass = an.Staff(
            lines=[40, 50, 60, 70, 80], thickness=1, spacing=self.S
        )
        head = an.NoteHead(
            x=40, y=60, filled=True, staff=bass, digit=2
        )
        event = an.ChordEvent(x=head.x, heads=[head])
        members = self._members(event)
        first_x_off = int(self.S * 1.8)
        x_step = max(1, int(self.S * 0.5))
        exact_x_off = first_x_off + 2 * x_step
        exact_left, _, exact_right, _ = an._stack_bounds_xy(
            head.y, exact_x_off, -1, members, max(int(members[0][3] * 0.95), 1)
        )
        self.assertEqual(exact_right, w)
        black[:, 49:exact_left] = 255

        added = self._render_bass(
            black, occupied, event, bass, (0, 20, 40, 80)
        )

        annotation_left, _, annotation_right, _ = self._annotation_bounds(added)
        self.assertFalse((added & (black > 0)).any())
        self.assertGreaterEqual(annotation_left, 0)
        self.assertLessEqual(annotation_right, w)

    def test_all_candidates_blocked_skips_annotation(self):
        h, w = 80, 80
        black = np.full((h, w), 255, dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        bass = an.Staff(
            lines=[30, 40, 50, 60, 70], thickness=1, spacing=self.S
        )
        head = an.NoteHead(
            x=74, y=40, filled=True, staff=bass, digit=1
        )
        event = an.ChordEvent(x=74, heads=[head])
        added = self._render_bass(
            black, occupied, event, bass, (0, 20, 30, 70)
        )

        self.assertFalse(added.any(), "blocked placement must not overwrite the score")


class BarlinePurgeTests(unittest.TestCase):
    def test_head_on_barline_detected(self):
        s = 10.0
        staff = an.Staff(
            lines=[40, 50, 60, 70, 80], thickness=1, spacing=s, clef="treble"
        )
        black = np.zeros((120, 160), dtype=np.uint8)
        bar_x = 90
        cv2.line(black, (bar_x, staff.top - 15), (bar_x, staff.bottom + 15), 255, 2)
        on_bar = an.NoteHead(x=bar_x, y=60, filled=False, staff=staff, digit=1)
        off_bar = an.NoteHead(x=60, y=60, filled=False, staff=staff, digit=3)
        self.assertTrue(an._head_on_barline(black, on_bar, staff, s))
        self.assertFalse(an._head_on_barline(black, off_bar, staff, s))


class IndependentBaselineTests(unittest.TestCase):
    def test_two_notes_follow_their_own_head_positions(self):
        s = 10
        h, w = 100, 200
        black = np.zeros((h, w), dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        staff = an.Staff(lines=[50, 60, 70, 80, 90], thickness=1, spacing=s)
        heads = [
            an.NoteHead(x=60, y=55, filled=True, staff=staff, digit=1),
            an.NoteHead(x=120, y=75, filled=True, staff=staff, digit=6),
        ]
        vis = np.full((h, w, 3), 255, dtype=np.uint8)
        an._render_group(
            vis, black,
            [an.ChordEvent(x=h.x, heads=[h]) for h in heads],
            staff, top=True, s=s, scale=0.7, occupied=occupied, y_ceiling=0,
            below_limit=h,
        )
        green = _green_mask(vis)
        bottoms = []
        for hx in (60, 120):
            ys, _ = np.where(green[:, hx - 6:hx + 6])
            self.assertGreater(len(ys), 0)
            bottoms.append(int(ys.max()))
        self.assertTrue(all(bottom < staff.top for bottom in bottoms))


class PreferOutsideStaffTests(unittest.TestCase):
    def test_high_note_places_digit_above_staff_when_headroom_exists(self):
        s = 10
        h, w = 100, 140
        black = np.zeros((h, w), dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        staff = an.Staff(lines=[50, 60, 70, 80, 90], thickness=1, spacing=s)
        head = an.NoteHead(x=70, y=55, filled=True, staff=staff, digit=3)
        vis = np.full((h, w, 3), 255, dtype=np.uint8)
        an._render_group(
            vis, black, [an.ChordEvent(x=70, heads=[head])], staff,
            top=True, s=s, scale=0.7, occupied=occupied, y_ceiling=0,
            below_limit=h,
        )
        green = _green_mask(vis)
        ys, _ = np.where(green)
        self.assertGreater(len(ys), 0)
        self.assertLessEqual(int(ys.max()), staff.top - an._annotation_clearance(s))


class StaffUniformSideTests(unittest.TestCase):
    def test_treble_events_share_above_or_below_side(self):
        s = 10
        img_h, w = 120, 200
        black = np.zeros((img_h, w), dtype=np.uint8)
        occupied = np.zeros((img_h, w), dtype=bool)
        staff = an.Staff(lines=[40, 50, 60, 70, 80], thickness=1, spacing=s)
        heads = [
            an.NoteHead(x=50, y=55, filled=True, staff=staff, digit=1),
            an.NoteHead(x=120, y=65, filled=True, staff=staff, digit=2),
        ]
        evs = [an.ChordEvent(x=nh.x, heads=[nh]) for nh in heads]
        vis = np.full((img_h, w, 3), 255, dtype=np.uint8)
        an._render_group(
            vis, black, evs, staff, top=True, s=s, scale=0.7,
            bounds=(40, 80, 90, 110), occupied=occupied, y_ceiling=0,
            below_limit=img_h,
        )
        green = _green_mask(vis)
        sides = []
        for head in heads:
            ys, _ = np.where(green[:, head.x - 8:head.x + 8])
            self.assertGreater(len(ys), 0, f"missing annotation near x={head.x}")
            sides.append("above" if np.median(ys) < head.y else "below")
        self.assertEqual(sides[0], sides[1], "one staff must not mix above and below")


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.black = np.zeros((40, 60), dtype=np.uint8)
        self.occupied = np.zeros((40, 60), dtype=bool)

    def _assert_green_fully_in_image(self, vis: np.ndarray, green: np.ndarray) -> None:
        h, w = vis.shape[:2]
        ys, xs = np.where(green)
        self.assertGreater(len(xs), 0, "expected rendered digits")
        self.assertGreaterEqual(int(ys.min()), 0)
        self.assertLess(int(ys.max()), h)
        self.assertGreaterEqual(int(xs.min()), 0)
        self.assertLess(int(xs.max()), w)

    def _bass_right_lane_setup(self, *, block_right: str | None = None):
        """Gap and below unavailable so _render_group enters the right-side path."""
        h, w = 105, 140
        vis = np.full((h, w, 3), 255, dtype=np.uint8)
        black = np.zeros((h, w), dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        s = 10
        treble = an.Staff(lines=[4, 14, 24, 34, 44], thickness=1, spacing=s)
        bass = an.Staff(lines=[54, 64, 74, 84, 94], thickness=1, spacing=s)
        black[49:55, :] = 255
        head = an.NoteHead(x=40, y=74, filled=True, staff=bass, digit=2)
        event = an.ChordEvent(x=40, heads=[head])
        xr = int(s * 1.8)
        lane_x0 = head.x - 6 + xr - 2
        lane_x1 = head.x + xr + 18
        lane_y0 = 0
        lane_y1 = h
        if block_right == "black":
            black[lane_y0:lane_y1, lane_x0:lane_x1] = 255
        elif block_right == "occupied":
            occupied[lane_y0:lane_y1, lane_x0:lane_x1] = True
        return dict(
            vis=vis,
            black=black,
            occupied=occupied,
            staff=bass,
            event=event,
            head=head,
            s=s,
            lane=(lane_y0, lane_y1, lane_x0, lane_x1),
        )

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

    def test_never_switches_sides_when_above_is_out_of_bounds(self):
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

        self.assertFalse(_green_mask(vis).any(),
                         "must not annotate inside the staff or on the wrong side")

    def test_render_shares_occupied_mask_across_systems(self):
        """Later systems must see annotations drawn by earlier systems."""
        h, w = 220, 120
        gray = np.full((h, w), 255, dtype=np.uint8)
        black = np.zeros((h, w), dtype=np.uint8)
        s = 10
        treble1 = an.Staff(lines=[10, 20, 30, 40, 50], thickness=1, spacing=s)
        bass1 = an.Staff(lines=[60, 70, 80, 90, 100], thickness=1, spacing=s)
        treble2 = an.Staff(lines=[120, 130, 140, 150, 160], thickness=1, spacing=s)
        bass2 = an.Staff(lines=[170, 180, 190, 200, 210], thickness=1, spacing=s)
        sys1 = an.System(treble=treble1, bass=bass1)
        sys2 = an.System(treble=treble2, bass=bass2)
        treble1_head = an.NoteHead(x=60, y=30, filled=True, staff=treble1, digit=3)
        treble2_head = an.NoteHead(x=60, y=130, filled=True, staff=treble2, digit=5)
        treble1_event = an.ChordEvent(x=60, heads=[treble1_head])
        page = an.Page(
            image=gray,
            binary=black,
            systems=[sys1, sys2],
            staves=[treble1, bass1, treble2, bass2],
            chords=[
                treble1_event,
                an.ChordEvent(x=60, heads=[treble2_head]),
            ],
        )

        vis1 = np.full((h, w, 3), 255, dtype=np.uint8)
        occupied1 = np.zeros((h, w), dtype=bool)
        an._render_group(
            vis1,
            black,
            [treble1_event],
            treble1,
            top=True,
            s=s,
            scale=0.7,
            bounds=(10, 50, 60, 100),
            occupied=occupied1,
            y_ceiling=0,
            below_limit=bass1.top - int(0.3 * s),
        )
        sys1_green_count = int(_green_mask(vis1).sum())

        vis = an.render(page)
        overlap_count = int((_green_mask(vis) & occupied1).sum())
        self.assertEqual(
            overlap_count,
            sys1_green_count,
            "later systems must not draw over earlier-system occupied pixels",
        )

    def test_bass_skips_when_lower_margin_is_unavailable(self):
        h, w = 105, 140
        vis = np.full((h, w, 3), 255, dtype=np.uint8)
        black = np.zeros((h, w), dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        s = 10
        treble = an.Staff(lines=[4, 14, 24, 34, 44], thickness=1, spacing=s)
        bass = an.Staff(lines=[54, 64, 74, 84, 94], thickness=1, spacing=s)
        black[49:55, :] = 255
        head = an.NoteHead(x=40, y=74, filled=True, staff=bass, digit=2)
        event = an.ChordEvent(x=40, heads=[head])
        tw, th, _ = an._measure(f"{head.digit}", 0.7)
        center_x0 = head.x - tw // 2 - 2
        center_x1 = head.x - tw // 2 + tw + 6
        black[:, center_x0:center_x1] = 255

        an._render_group(
            vis,
            black,
            [event],
            bass,
            top=False,
            s=s,
            scale=0.7,
            mid_y=49,
            bounds=(4, 44, 54, 94),
            occupied=occupied,
        )

        self.assertFalse(_green_mask(vis).any(),
                         "must not annotate inside the staff or on the wrong side")

    def test_bass_all_blocked_never_overwrites_score(self):
        ctx = self._bass_right_lane_setup(block_right="black")
        an._render_group(
            ctx["vis"],
            ctx["black"],
            [ctx["event"]],
            ctx["staff"],
            top=False,
            s=ctx["s"],
            scale=0.7,
            mid_y=49,
            bounds=(4, 44, 54, 94),
            occupied=ctx["occupied"],
        )

        self.assertFalse(_green_mask(ctx["vis"]).any(),
                         "must not annotate inside the staff or on the wrong side")

    def test_render_right_blocked_by_original_score_black(self):
        ctx = self._bass_right_lane_setup(block_right="black")
        an._render_group(
            ctx["vis"],
            ctx["black"],
            [ctx["event"]],
            ctx["staff"],
            top=False,
            s=ctx["s"],
            scale=0.7,
            mid_y=49,
            bounds=(4, 44, 54, 94),
            occupied=ctx["occupied"],
        )

        self.assertFalse(_green_mask(ctx["vis"]).any(),
                         "must not annotate inside the staff or on the wrong side")

    def test_render_right_blocked_by_occupied_simplified_score(self):
        ctx = self._bass_right_lane_setup(block_right="occupied")
        an._render_group(
            ctx["vis"],
            ctx["black"],
            [ctx["event"]],
            ctx["staff"],
            top=False,
            s=ctx["s"],
            scale=0.7,
            mid_y=49,
            bounds=(4, 44, 54, 94),
            occupied=ctx["occupied"],
        )

        self.assertFalse(_green_mask(ctx["vis"]).any(),
                         "must not annotate inside the staff or on the wrong side")

    def _synthetic_members(self, *, cx: int, count: int, digit: int = 1, scale: float = 0.7):
        members = []
        for _ in range(count):
            h = an.NoteHead(x=cx, y=0, filled=True, digit=digit)
            tw, th, _ = an._measure(f"{digit}", scale)
            top_off = th // 2 + 1
            bot_off = th // 2 + 1
            right_ext = tw + 4
            members.append((h, cx - tw // 2, tw, th, 0, 0, top_off, bot_off, right_ext))
        return members

    def test_fit_compat_stack_preserves_upward_direction(self):
        H, W = 50, 80
        members = self._synthetic_members(cx=40, count=2)
        step = max(int(members[0][3] * 0.95), 1)
        prefer_y, prefer_x_off, dir_sign = 15, 0, -1

        y, x_off, ds = an._fit_compat_stack(
            H, W, prefer_y, prefer_x_off, dir_sign, members, step
        )

        self.assertEqual(ds, -1, "must not reverse multi-note stack direction")
        self.assertTrue(an._stack_fits_image(H, W, y, x_off, ds, members, step))
        _, top, _, _ = an._stack_bounds_xy(y, x_off, ds, members, step)
        self.assertGreaterEqual(top, 0)

    def test_fit_compat_stack_shifts_horizontally_near_right_edge(self):
        H, W = 50, 48
        members = self._synthetic_members(cx=42, count=1)
        step = max(int(members[0][3] * 0.95), 1)
        prefer_y, prefer_x_off, dir_sign = 25, 0, -1

        self.assertFalse(
            an._stack_fits_image(H, W, prefer_y, prefer_x_off, dir_sign, members, step),
            "unadjusted compat anchor should clip horizontally",
        )

        y, x_off, ds = an._fit_compat_stack(
            H, W, prefer_y, prefer_x_off, dir_sign, members, step
        )
        left, top, right, bottom = an._stack_bounds_xy(y, x_off, ds, members, step)

        self.assertEqual(ds, dir_sign)
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, W)
        self.assertLessEqual(bottom, H)

    def test_render_out_of_image_note_skips_unsafe_compat_box(self):
        h, w = 50, 48
        vis = np.full((h, w, 3), 255, dtype=np.uint8)
        black = np.zeros((h, w), dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        s = 10
        bass = an.Staff(lines=[54, 64, 74, 84, 94], thickness=1, spacing=s)
        black[49:55, :] = 255
        head = an.NoteHead(x=42, y=74, filled=True, staff=bass, digit=2)
        event = an.ChordEvent(x=42, heads=[head])
        xr = int(s * 1.8)
        lane_x0 = head.x - 6 + xr - 2
        black[:, lane_x0:w] = 255

        an._render_group(
            vis,
            black,
            [event],
            bass,
            top=False,
            s=s,
            scale=0.7,
            mid_y=49,
            bounds=(4, 44, 54, 94),
            occupied=occupied,
        )

        tw, th, _ = an._measure(f"{head.digit}", 0.7)
        members = [(head, 42 - tw // 2, tw, th, 0, 0, th // 2 + 1, th // 2 + 1, tw + 4)]
        step = max(int(th * 0.95), 1)
        y_row = head.y - int(s * 0.6) - th // 2 - 2
        y, x_off, ds = an._fit_compat_stack(h, w, y_row, 0, -1, members, step)
        left, top, right, bottom = an._stack_bounds_xy(y, x_off, ds, members, step)
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, w)
        self.assertLessEqual(bottom, h)
        self.assertFalse(_green_mask(vis).any(), "unsafe clamped placement must be skipped")


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

    def test_clears_at_limit_but_preserves_just_outside(self):
        image = np.zeros((80, 120), dtype=np.uint8)
        staff = an.Staff(lines=[20, 30, 40, 50, 60], thickness=1, spacing=10)
        first_x = 20
        limit = first_x + 0.6 * staff.spacing
        anchor = an.NoteHead(x=first_x, y=20, filled=True, staff=staff, alter=-1)
        at_limit = an.NoteHead(x=int(limit), y=30, filled=True, staff=staff, alter=1)
        just_outside = an.NoteHead(x=int(limit) + 1, y=40, filled=True, staff=staff, alter=-1)
        page = an.Page(
            image=image,
            binary=image.copy(),
            staves=[staff],
            noteheads=[anchor, at_limit, just_outside],
            accidentals=[
                (first_x, 20, -1),
                (int(limit), 30, 1),
                (int(limit) + 1, 40, -1),
            ],
        )

        an._clear_leading_accidentals(page)

        self.assertEqual(anchor.alter, 0)
        self.assertEqual(at_limit.alter, 0)
        self.assertEqual(just_outside.alter, -1)
        self.assertEqual(page.accidentals, [(int(limit) + 1, 40, -1)])

    def test_clears_leading_window_independently_per_staff(self):
        image = np.zeros((160, 120), dtype=np.uint8)
        staff_a = an.Staff(lines=[20, 30, 40, 50, 60], thickness=1, spacing=10)
        staff_b = an.Staff(lines=[90, 100, 110, 120, 130], thickness=1, spacing=10)
        a_leading = an.NoteHead(x=20, y=30, filled=True, staff=staff_a, alter=-1)
        a_later = an.NoteHead(x=60, y=30, filled=True, staff=staff_a, alter=1)
        b_leading = an.NoteHead(x=30, y=100, filled=True, staff=staff_b, alter=1)
        b_later = an.NoteHead(x=70, y=100, filled=True, staff=staff_b, alter=-1)
        page = an.Page(
            image=image,
            binary=image.copy(),
            staves=[staff_a, staff_b],
            noteheads=[a_leading, a_later, b_leading, b_later],
            accidentals=[
                (20, 30, -1),
                (60, 30, 1),
                (30, 100, 1),
                (70, 100, -1),
            ],
        )

        an._clear_leading_accidentals(page)

        self.assertEqual([a_leading.alter, a_later.alter], [0, 1])
        self.assertEqual([b_leading.alter, b_later.alter], [0, -1])
        self.assertEqual(page.accidentals, [(60, 30, 1), (70, 100, -1)])


class EventDedupTests(unittest.TestCase):
    def _staff(self):
        return an.Staff(lines=[20, 30, 40, 50, 60], thickness=1, spacing=10)

    def _event(self, staff, x, digit=1, dots=0, prefix="", dur=1.0, y=30):
        head = an.NoteHead(
            x=x, y=y, filled=True, staff=staff,
            digit=digit, dots=dots, prefix=prefix, dur=dur,
        )
        return an.ChordEvent(x=x, heads=[head])

    @patch("annotate._render_group")
    def test_adjacent_identical_events_are_all_kept(self, mock_render_group):
        staff = self._staff()
        h1 = an.NoteHead(x=40, y=30, filled=True, staff=staff, digit=2, dur=1.0)
        h2 = an.NoteHead(x=60, y=30, filled=True, staff=staff, digit=2, dur=1.0)
        ev1 = an.ChordEvent(x=40, heads=[h1])
        ev2 = an.ChordEvent(x=60, heads=[h2])
        page = an.Page(
            image=np.full((80, 120), 255, dtype=np.uint8),
            binary=np.zeros((80, 120), dtype=np.uint8),
            systems=[an.System(treble=staff, bass=staff)],
            staves=[staff],
            chords=[ev1, ev2],
        )
        mock_render_group.side_effect = lambda *args, **kwargs: None

        an.render(page)

        treble_calls = [
            call for call in mock_render_group.call_args_list
            if call.args[3] is staff
            and call.kwargs.get("top", call.args[4] if len(call.args) > 4 else None) is True
        ]
        self.assertEqual(len(treble_calls), 1)
        rendered_evs = treble_calls[0].args[2]
        self.assertEqual([ev.x for ev in rendered_evs], [40, 60])

    def test_collapses_duplicate_members_within_event(self):
        staff = self._staff()
        dup1 = an.NoteHead(x=10, y=30, filled=True, staff=staff, digit=1, dur=1.0)
        dup2 = an.NoteHead(x=12, y=31, filled=True, staff=staff, digit=1, dur=1.0)
        other = an.NoteHead(x=11, y=40, filled=True, staff=staff, digit=3, dur=1.0)
        ev = an.ChordEvent(x=10, heads=[dup1, dup2, other])

        view = an._event_view_for_render(ev)
        self.assertEqual(len(view.heads), 3)
        self.assertEqual(len(ev.heads), 3, "original event must not be mutated")

    def test_render_dedup_does_not_mutate_page_chords(self):
        staff = self._staff()
        h1 = an.NoteHead(x=40, y=30, filled=True, staff=staff, digit=2, dur=1.0)
        h2 = an.NoteHead(x=60, y=30, filled=True, staff=staff, digit=2, dur=1.0)
        ev1 = an.ChordEvent(x=40, heads=[h1])
        ev2 = an.ChordEvent(x=60, heads=[h2])
        gray = np.full((80, 120), 255, dtype=np.uint8)
        page = an.Page(
            image=gray,
            binary=np.zeros((80, 120), dtype=np.uint8),
            systems=[an.System(treble=staff, bass=staff)],
            staves=[staff],
            chords=[ev1, ev2],
        )
        chords_before = list(page.chords)

        an.render(page)

        self.assertEqual(page.chords, chords_before)
        self.assertIs(page.chords[0], ev1)
        self.assertIs(page.chords[1], ev2)

    @patch("annotate._render_group")
    def test_render_passes_repeated_harmonies_at_different_positions(self, mock_render_group):
        staff = self._staff()
        h1 = an.NoteHead(x=40, y=30, filled=True, staff=staff, digit=2, dur=1.0)
        h2 = an.NoteHead(x=60, y=30, filled=True, staff=staff, digit=2, dur=1.0)
        ev1 = an.ChordEvent(x=40, heads=[h1])
        ev2 = an.ChordEvent(x=60, heads=[h2])
        gray = np.full((80, 120), 255, dtype=np.uint8)
        page = an.Page(
            image=gray,
            binary=np.zeros((80, 120), dtype=np.uint8),
            systems=[an.System(treble=staff, bass=staff)],
            staves=[staff],
            chords=[ev1, ev2],
        )
        chords_before = list(page.chords)
        mock_render_group.side_effect = lambda *args, **kwargs: None

        an.render(page)

        self.assertEqual(page.chords, chords_before)
        treble_calls = [
            call for call in mock_render_group.call_args_list
            if call.args[3] is staff and call.kwargs.get("top", call.args[4] if len(call.args) > 4 else None) is True
        ]
        self.assertEqual(len(treble_calls), 1)
        rendered_evs = treble_calls[0].args[2]
        self.assertEqual(len(rendered_evs), 2)
        self.assertEqual([ev.x for ev in rendered_evs], [40, 60])

    @patch("annotate._render_group")
    def test_render_first_system_includes_all_staves(self, mock_render_group):
        treble = self._staff()
        bass = an.Staff(lines=[90, 100, 110, 120, 130], thickness=1, spacing=10)
        h_t = an.NoteHead(x=40, y=30, filled=True, staff=treble, digit=2, dur=1.0)
        h_b = an.NoteHead(x=40, y=100, filled=True, staff=bass, digit=5, dur=1.0)
        gray = np.full((160, 120), 255, dtype=np.uint8)
        page = an.Page(
            image=gray,
            binary=np.zeros((160, 120), dtype=np.uint8),
            systems=[an.System(treble=treble, bass=bass)],
            staves=[treble, bass],
            chords=[
                an.ChordEvent(x=40, heads=[h_t]),
                an.ChordEvent(x=40, heads=[h_b]),
            ],
        )
        mock_render_group.side_effect = lambda *args, **kwargs: None

        an.render(page)

        rendered_staffs = [call.args[3] for call in mock_render_group.call_args_list]
        self.assertEqual(rendered_staffs, [treble, bass])

    @patch("annotate._render_group")
    def test_render_includes_later_systems(self, mock_render_group):
        treble1 = self._staff()
        bass1 = an.Staff(lines=[90, 100, 110, 120, 130], thickness=1, spacing=10)
        treble2 = an.Staff(lines=[150, 160, 170, 180, 190], thickness=1, spacing=10)
        bass2 = an.Staff(lines=[220, 230, 240, 250, 260], thickness=1, spacing=10)
        h2 = an.NoteHead(x=40, y=160, filled=True, staff=treble2, digit=3, dur=1.0)
        gray = np.full((280, 120), 255, dtype=np.uint8)
        page = an.Page(
            image=gray,
            binary=np.zeros((280, 120), dtype=np.uint8),
            systems=[
                an.System(treble=treble1, bass=bass1),
                an.System(treble=treble2, bass=bass2),
            ],
            staves=[treble1, bass1, treble2, bass2],
            chords=[an.ChordEvent(x=40, heads=[h2])],
        )
        mock_render_group.side_effect = lambda *args, **kwargs: None

        an.render(page)

        rendered_staffs = [call.args[3] for call in mock_render_group.call_args_list]
        self.assertEqual(rendered_staffs, [treble1, bass1, treble2, bass2])

    def test_second_system_digit_stays_on_that_system(self):
        s = 10
        treble1 = an.Staff(lines=[10, 20, 30, 40, 50], thickness=1, spacing=s)
        bass1 = an.Staff(lines=[70, 80, 90, 100, 110], thickness=1, spacing=s)
        treble2 = an.Staff(lines=[180, 190, 200, 210, 220], thickness=1, spacing=s)
        bass2 = an.Staff(lines=[240, 250, 260, 270, 280], thickness=1, spacing=s)
        head = an.NoteHead(x=70, y=200, filled=True, staff=treble2, digit=3, dur=1.0)
        h, w = 320, 200
        page = an.Page(
            image=np.full((h, w), 255, dtype=np.uint8),
            binary=np.zeros((h, w), dtype=np.uint8),
            systems=[
                an.System(treble=treble1, bass=bass1),
                an.System(treble=treble2, bass=bass2),
            ],
            staves=[treble1, bass1, treble2, bass2],
            chords=[an.ChordEvent(x=70, heads=[head])],
        )

        vis = an.render(page, use_jev=False)

        ys, xs = np.where(_green_mask(vis))
        self.assertGreater(len(xs), 0)
        self.assertGreater(int(ys.min()), bass1.bottom)
        self.assertLess(int(ys.max()), bass2.top)
        self.assertLess(int(xs.max()), head.x + 4 * s)

    def test_vocal_extra_does_not_jump_into_piano_or_line_end(self):
        s = 10
        treble = an.Staff(lines=[30, 40, 50, 60, 70], thickness=1, spacing=s)
        bass = an.Staff(lines=[100, 110, 120, 130, 140], thickness=1, spacing=s)
        vocal = an.Staff(lines=[210, 220, 230, 240, 250], thickness=1, spacing=s)
        head = an.NoteHead(x=80, y=230, filled=True, staff=vocal, digit=5, dur=1.0)
        h, w = 300, 360
        black = np.zeros((h, w), dtype=np.uint8)
        # Local column and the rest of the measure are ink; only the line end is clear.
        black[:, 50:280] = 255
        page = an.Page(
            image=np.full((h, w), 255, dtype=np.uint8),
            binary=black,
            systems=[an.System(treble=treble, bass=bass, extras=[vocal])],
            staves=[treble, bass, vocal],
            chords=[an.ChordEvent(x=80, heads=[head])],
        )

        vis = an.render(page, use_jev=False)

        self.assertFalse(_green_mask(vis).any(), "skip when no nearby blank position exists")


class LeadingSymbolZoneTests(unittest.TestCase):
    """Bounded per-staff leading symbol zone via _remove_leading_symbols."""

    STAFF_GRAY = 176

    def _draw_staff_lines(self, gray: np.ndarray, line_ys: list[int], left: int,
                          right: int | None = None) -> None:
        if right is None:
            right = gray.shape[1]
        for y in line_ys:
            gray[y, left:right] = self.STAFF_GRAY

    def _staff(self, top: int, spacing: float = 10.0) -> an.Staff:
        lines = [top + int(round(i * spacing)) for i in range(5)]
        return an.Staff(lines=lines, thickness=1, spacing=spacing)

    def _binary_from_gray(self, gray: np.ndarray) -> np.ndarray:
        return (gray < 128).astype(np.uint8) * 255

    def _draw_filled_head(self, gray: np.ndarray, x: int, y: int, s: float) -> None:
        w = max(4, int(round(an.NOTE_W_SOLID * s)))
        h = max(3, int(round(an.NOTE_H_SOLID * s)))
        cv2.ellipse(gray, (x, y), (w // 2, h // 2), 0, 0, 360, 0, -1)

    def _draw_hollow_ring(self, gray: np.ndarray, x: int, y: int, s: float) -> None:
        w = max(6, int(round(an.NOTE_W_HOLLOW * s)))
        h = max(4, int(round(an.NOTE_H_HOLLOW * s)))
        cv2.ellipse(gray, (x, y), (w // 2, h // 2), 0, 0, 360, 0, 3)

    def _draw_stem_up(self, gray: np.ndarray, x: int, y: int, s: float) -> None:
        stem_x = x + max(2, int(round(an.NOTE_W_SOLID * s)) // 2)
        y_top = y - int(2.4 * s)
        y_bot = y - max(2, int(round(an.NOTE_H_SOLID * s)) // 2)
        cv2.line(gray, (stem_x, y_top), (stem_x, y_bot), 0, max(2, int(s * 0.18)))

    def _draw_tall_narrow_glyph(self, gray: np.ndarray, x: int, y: int, s: float) -> None:
        w = max(3, int(round(0.7 * s)))
        h = max(12, int(round(2.4 * s)))
        cv2.rectangle(gray, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), 0, -1)

    def _draw_flat_like_loop(self, gray: np.ndarray, x: int, y: int, s: float) -> None:
        """Compact horizontal loop with nearby tall flat-like stroke (separate component)."""
        cv2.ellipse(gray, (x, y), (max(6, int(0.6 * s)), max(4, int(0.4 * s))),
                    0, 0, 360, 0, max(2, int(0.12 * s)))
        stroke_x = x + max(10, int(0.75 * s))
        cv2.line(
            gray,
            (stroke_x, y - int(1.9 * s)),
            (stroke_x, y + int(1.9 * s)),
            0,
            max(2, int(0.15 * s)),
        )

    def _draw_time_sig_zero(self, gray: np.ndarray, x: int, y: int, s: float) -> None:
        """Digit-like zero that can false-trigger notehead loop detection."""
        w = max(5, int(round(0.55 * s)))
        h = max(12, int(round(1.2 * s)))
        cv2.ellipse(gray, (x, y), (w // 2, h // 2), 0, 0, 360, 0, max(2, int(0.15 * s)))

    def _draw_clef_like_loop_stroke(self, gray: np.ndarray, x: int, y: int, s: float) -> None:
        """Small loop with nearby clef-like vertical stroke."""
        cv2.ellipse(gray, (x, y), (max(6, int(0.55 * s)), max(4, int(0.35 * s))),
                    0, 0, 360, 0, max(2, int(0.12 * s)))
        sx = x + max(10, int(0.75 * s))
        cv2.line(gray, (sx, y + 2), (sx, y - int(2.1 * s)), 0, max(2, int(0.15 * s)))

    def _draw_connected_flat_like_loop(self, gray: np.ndarray, x: int, y: int,
                                       s: float) -> None:
        """Narrow flat bowl whose vertical stroke touches the bowl's outer tangent."""
        rx, ry = max(5, int(0.6 * s)), max(4, int(0.4 * s))
        thickness = max(3, int(0.2 * s))
        cv2.ellipse(gray, (x, y), (rx, ry), 0, 0, 360, 0, thickness)
        cv2.line(gray, (x - rx, y - int(2.0 * s)), (x - rx, y),
                 0, thickness)

    def _draw_connected_clef_like_loop(self, gray: np.ndarray, x: int, y: int,
                                       s: float) -> None:
        """Compact clef loop connected to a long curved-looking side stroke."""
        rx, ry = max(5, int(0.62 * s)), max(4, int(0.43 * s))
        thickness = max(3, int(0.2 * s))
        cv2.ellipse(gray, (x, y), (rx, ry), 0, 0, 360, 0, thickness)
        cv2.line(gray, (x + rx, y - int(2.1 * s)), (x + rx, y + ry),
                 0, thickness)

    def _draw_connected_digit_nine(self, gray: np.ndarray, x: int, y: int,
                                   s: float) -> None:
        """A narrow 9-like loop connected to its ascending side stroke."""
        rx, ry = max(6, int(0.68 * s)), max(5, int(0.5 * s))
        thickness = max(3, int(0.2 * s))
        cv2.ellipse(gray, (x, y), (rx, ry), 0, 0, 360, 0, thickness)
        cv2.line(gray, (x + rx, y), (x + rx, y - int(2.0 * s)),
                 0, thickness)

    def _assert_one_connected_component(self, gray: np.ndarray, x: int, y: int,
                                        s: float) -> None:
        pad_x = int(1.1 * s)
        pad_y = int(2.4 * s)
        crop = self._binary_from_gray(
            gray[max(0, y - pad_y):y + pad_y + 1,
                 max(0, x - pad_x):x + pad_x + 1]
        )
        count, _ = cv2.connectedComponents((crop > 0).astype(np.uint8), 8)
        self.assertEqual(count - 1, 1, "symbol fixture must be one connected component")

    def _setup_staff_gray(self, s: float, left: int, width: int = 260):
        gray = np.full((140, width), 255, dtype=np.uint8)
        staff = self._staff(top=20, spacing=s)
        self._draw_staff_lines(gray, staff.lines, left=left)
        edge = an._staff_left_edge(gray, staff)
        self.assertIsNotNone(edge)
        return gray, staff, edge

    def _note(self, x: int, y: int, staff: an.Staff, *, filled: bool = True) -> an.NoteHead:
        return an.NoteHead(x=x, y=y, filled=filled, staff=staff)

    def _run_remove(self, gray: np.ndarray, staff: an.Staff,
                    noteheads: list[an.NoteHead],
                    *, staves: list[an.Staff] | None = None,
                    systems: list[an.System] | None = None) -> list[an.NoteHead]:
        staves = staves or [staff]
        systems = systems or [an.System(treble=staves[0], bass=staves[-1])]
        page = an.Page(
            image=gray,
            binary=self._binary_from_gray(gray),
            staves=staves,
            systems=systems,
            noteheads=list(noteheads),
        )
        an._remove_leading_symbols(page)
        return page.noteheads

    def test_staff_left_edge_from_synthetic_lines(self):
        gray = np.full((120, 200), 255, dtype=np.uint8)
        staff = self._staff(top=20, spacing=10)
        self._draw_staff_lines(gray, staff.lines, left=35)
        edge = an._staff_left_edge(gray, staff)
        self.assertIsNotNone(edge)
        self.assertGreaterEqual(edge, 33)
        self.assertLessEqual(edge, 37)

    def test_removes_symbol_shapes_keeps_outside_note(self):
        """Multiple in-zone symbol glyphs removed; boundary/outside real notes kept."""
        s = 10.0
        left = 30
        gray = np.full((120, 240), 255, dtype=np.uint8)
        staff = self._staff(top=20, spacing=s)
        self._draw_staff_lines(gray, staff.lines, left=left)
        edge = an._staff_left_edge(gray, staff)
        self.assertIsNotNone(edge)
        zone_right = edge + 7.0 * s
        cy = staff.lines[2]

        sym_a = self._note(int(edge) + 8, cy, staff)
        sym_b = self._note(int(edge) + 28, cy - 4, staff)
        sym_c = self._note(int(edge) + 48, cy + 3, staff)
        self._draw_tall_narrow_glyph(gray, sym_a.x, sym_a.y, s)
        self._draw_tall_narrow_glyph(gray, sym_b.x, sym_b.y, s)
        self._draw_tall_narrow_glyph(gray, sym_c.x, sym_c.y, s)

        at_boundary = self._note(int(zone_right), cy, staff)
        outside = self._note(int(zone_right) + 12, cy, staff)
        self._draw_filled_head(gray, at_boundary.x, at_boundary.y, s)
        self._draw_stem_up(gray, at_boundary.x, at_boundary.y, s)
        self._draw_filled_head(gray, outside.x, outside.y, s)
        self._draw_stem_up(gray, outside.x, outside.y, s)

        remaining = self._run_remove(
            gray, staff, [sym_a, sym_b, sym_c, at_boundary, outside]
        )
        xs = sorted(n.x for n in remaining)
        self.assertEqual(xs, [at_boundary.x, outside.x])

    def test_preserves_stemmed_first_note_within_zone(self):
        """Earliest strong note inside 7*s stops removal; later notes also kept."""
        s = 10.0
        left = 35
        gray = np.full((120, 240), 255, dtype=np.uint8)
        staff = self._staff(top=20, spacing=s)
        self._draw_staff_lines(gray, staff.lines, left=left)
        edge = an._staff_left_edge(gray, staff)
        self.assertIsNotNone(edge)
        cy = staff.lines[2]

        sym1 = self._note(int(edge) + 10, cy, staff)
        sym2 = self._note(int(edge) + 22, cy - 3, staff)
        self._draw_tall_narrow_glyph(gray, sym1.x, sym1.y, s)
        self._draw_tall_narrow_glyph(gray, sym2.x, sym2.y, s)

        strong = self._note(int(edge) + 45, cy, staff)
        later = self._note(int(edge) + 62, cy + 2, staff)
        self._draw_filled_head(gray, strong.x, strong.y, s)
        self._draw_stem_up(gray, strong.x, strong.y, s)
        self._draw_filled_head(gray, later.x, later.y, s)
        self._draw_stem_up(gray, later.x, later.y, s)

        remaining = self._run_remove(gray, staff, [sym1, sym2, strong, later])
        xs = sorted(n.x for n in remaining)
        self.assertEqual(xs, [strong.x, later.x])

    def test_strong_filled_anchor_still_suppresses_in_zone_hollows(self):
        """Strong filled note anchors non-hollows; hollows in hard zone still removed."""
        s = 10.0
        left = 35
        gray = np.full((120, 280), 255, dtype=np.uint8)
        staff = self._staff(top=20, spacing=s)
        self._draw_staff_lines(gray, staff.lines, left=left)
        edge = an._staff_left_edge(gray, staff)
        self.assertIsNotNone(edge)
        hard_right = edge + 7.0 * s
        cy = staff.lines[2]

        strong = self._note(int(edge) + 40, cy, staff)
        hollow_a = self._note(int(edge) + 55, cy, staff, filled=False)
        hollow_b = self._note(int(hard_right) - 3, cy + 2, staff, filled=False)
        outside = self._note(int(hard_right) + 10, cy, staff)
        self._draw_filled_head(gray, strong.x, strong.y, s)
        self._draw_stem_up(gray, strong.x, strong.y, s)
        self._draw_hollow_ring(gray, hollow_a.x, hollow_a.y, s)
        self._draw_hollow_ring(gray, hollow_b.x, hollow_b.y, s)
        self._draw_filled_head(gray, outside.x, outside.y, s)
        self._draw_stem_up(gray, outside.x, outside.y, s)

        remaining = self._run_remove(
            gray, staff, [strong, hollow_a, hollow_b, outside]
        )
        self.assertEqual([n.x for n in remaining], [strong.x, outside.x])

    def test_removes_broad_and_narrow_hollow_loops_within_zone(self):
        s = 10.0
        left = 35
        gray = np.full((120, 240), 255, dtype=np.uint8)
        staff = self._staff(top=20, spacing=s)
        self._draw_staff_lines(gray, staff.lines, left=left)
        edge = an._staff_left_edge(gray, staff)
        self.assertIsNotNone(edge)
        cy = staff.lines[2]

        broad = self._note(int(edge) + 18, cy, staff, filled=False)
        narrow = self._note(int(edge) + 48, cy, staff, filled=False)
        self._draw_hollow_ring(gray, broad.x, broad.y, s)
        cv2.ellipse(
            gray, (narrow.x, narrow.y), (max(4, int(0.45 * s)), max(4, int(0.4 * s))),
            0, 0, 360, 0, max(2, int(0.18 * s)),
        )

        remaining = self._run_remove(gray, staff, [broad, narrow])
        self.assertEqual(remaining, [])

    def test_hard_bound_does_not_remove_beyond_zone_end(self):
        """Tall-narrow glyph past left+7*s must survive even if glyph-shaped."""
        s = 10.0
        left = 30
        gray = np.full((120, 240), 255, dtype=np.uint8)
        staff = self._staff(top=20, spacing=s)
        self._draw_staff_lines(gray, staff.lines, left=left)
        edge = an._staff_left_edge(gray, staff)
        self.assertIsNotNone(edge)
        zone_right = edge + an._staff_symbol_hard_right(staff)
        cy = staff.lines[2]

        in_zone = self._note(int(edge) + 15, cy, staff)
        self._draw_tall_narrow_glyph(gray, in_zone.x, in_zone.y, s)

        beyond = self._note(int(zone_right) + 6, cy, staff)
        self._draw_tall_narrow_glyph(gray, beyond.x, beyond.y, s)

        remaining = self._run_remove(gray, staff, [in_zone, beyond])
        self.assertEqual([n.x for n in remaining], [beyond.x])

    def test_zone_removal_continues_past_non_glyph_candidate(self):
        """Symbol candidates without strong evidence are all removed before strong note."""
        s = 10.0
        left = 40
        gray = np.full((120, 240), 255, dtype=np.uint8)
        staff = self._staff(top=20, spacing=s)
        self._draw_staff_lines(gray, staff.lines, left=left)
        cy = staff.lines[2]
        edge = an._staff_left_edge(gray, staff)
        self.assertIsNotNone(edge)
        zone_right = edge + 7.0 * s

        weak = self._note(int(edge) + 10, cy, staff)
        later_sym = self._note(int(edge) + 55, cy, staff)
        outside = self._note(int(zone_right) + 8, cy, staff)
        self._draw_filled_head(gray, weak.x, weak.y, s)
        self._draw_tall_narrow_glyph(gray, later_sym.x, later_sym.y, s)
        self._draw_filled_head(gray, outside.x, outside.y, s)
        self._draw_stem_up(gray, outside.x, outside.y, s)

        remaining = self._run_remove(gray, staff, [weak, later_sym, outside])
        self.assertEqual([n.x for n in remaining], [outside.x])

    def test_independent_zones_for_two_staffs(self):
        """Each staff uses its own left edge and strong-note stop."""
        s = 10.0
        left_a, left_b = 25, 60
        gray = np.full((200, 280), 255, dtype=np.uint8)
        staff_a = self._staff(top=20, spacing=s)
        staff_b = self._staff(top=100, spacing=s)
        self._draw_staff_lines(gray, staff_a.lines, left=left_a)
        self._draw_staff_lines(gray, staff_b.lines, left=left_b)

        cy_a = staff_a.lines[2]
        cy_b = staff_b.lines[2]
        edge_a = an._staff_left_edge(gray, staff_a)
        edge_b = an._staff_left_edge(gray, staff_b)
        self.assertIsNotNone(edge_a)
        self.assertIsNotNone(edge_b)

        a_sym = self._note(int(edge_a) + 12, cy_a, staff_a)
        a_strong = self._note(int(edge_a) + 38, cy_a, staff_a)
        b_sym = self._note(int(edge_b) + 14, cy_b, staff_b)
        b_out = self._note(int(edge_b) + 95, cy_b, staff_b)
        self._draw_tall_narrow_glyph(gray, a_sym.x, a_sym.y, s)
        self._draw_filled_head(gray, a_strong.x, a_strong.y, s)
        self._draw_stem_up(gray, a_strong.x, a_strong.y, s)
        self._draw_tall_narrow_glyph(gray, b_sym.x, b_sym.y, s)
        self._draw_filled_head(gray, b_out.x, b_out.y, s)
        self._draw_stem_up(gray, b_out.x, b_out.y, s)

        page = an.Page(
            image=gray,
            binary=self._binary_from_gray(gray),
            staves=[staff_a, staff_b],
            systems=[an.System(treble=staff_a, bass=staff_b)],
            noteheads=[a_sym, a_strong, b_sym, b_out],
        )
        an._remove_leading_symbols(page)

        by_staff = {
            id(staff_a): sorted(n.x for n in page.noteheads if n.staff is staff_a),
            id(staff_b): sorted(n.x for n in page.noteheads if n.staff is staff_b),
        }
        self.assertEqual(by_staff[id(staff_a)], [a_strong.x])
        self.assertEqual(by_staff[id(staff_b)], [b_out.x])

    def test_flat_like_loop_does_not_anchor(self):
        """Flat-like loop + tall stroke must not stop the symbol zone."""
        s = 10.0
        gray, staff, edge = self._setup_staff_gray(s, left=35)
        cy = staff.lines[2]

        flat_loop = self._note(int(edge) + 18, cy, staff, filled=False)
        self._draw_flat_like_loop(gray, flat_loop.x, flat_loop.y, s)
        sym = self._note(int(edge) + 34, cy - 2, staff)
        self._draw_tall_narrow_glyph(gray, sym.x, sym.y, s)
        real = self._note(int(edge) + 58, cy, staff)
        self._draw_filled_head(gray, real.x, real.y, s)
        self._draw_stem_up(gray, real.x, real.y, s)

        remaining = self._run_remove(gray, staff, [flat_loop, sym, real])
        self.assertEqual([n.x for n in remaining], [real.x])

    def test_time_sig_zero_does_not_anchor(self):
        """Tall time-signature zero must not anchor the symbol zone."""
        s = 10.0
        gray, staff, edge = self._setup_staff_gray(s, left=35)
        cy = staff.lines[2]

        zero = self._note(int(edge) + 16, cy, staff, filled=False)
        self._draw_time_sig_zero(gray, zero.x, zero.y, s)
        sym = self._note(int(edge) + 38, cy, staff)
        self._draw_tall_narrow_glyph(gray, sym.x, sym.y, s)
        real = self._note(int(edge) + 62, cy, staff)
        self._draw_filled_head(gray, real.x, real.y, s)
        self._draw_stem_up(gray, real.x, real.y, s)

        remaining = self._run_remove(gray, staff, [zero, sym, real])
        self.assertEqual([n.x for n in remaining], [real.x])

    def test_preserves_hollow_notes_at_and_outside_exclusive_zone_edge(self):
        s = 10.0
        gray, staff, edge = self._setup_staff_gray(s, left=35)
        cy = staff.lines[2]
        zone_right = edge + an._staff_symbol_hard_right(staff)

        inside = self._note(int(zone_right) - 1, cy, staff, filled=False)
        at_boundary = self._note(int(zone_right), cy, staff, filled=False)
        outside = self._note(int(zone_right) + 12, cy, staff, filled=False)
        for note in (inside, at_boundary, outside):
            self._draw_hollow_ring(gray, note.x, note.y, s)

        remaining = self._run_remove(gray, staff, [inside, at_boundary, outside])
        xs = sorted(n.x for n in remaining)
        self.assertEqual(xs, [at_boundary.x, outside.x])

    def test_false_loop_before_real_note_removes_intervening_symbols(self):
        """Clef-like false loop must not preserve later in-zone symbol candidates."""
        s = 10.0
        gray, staff, edge = self._setup_staff_gray(s, left=35)
        cy = staff.lines[2]

        false_loop = self._note(int(edge) + 14, cy, staff, filled=False)
        self._draw_clef_like_loop_stroke(gray, false_loop.x, false_loop.y, s)
        sym = self._note(int(edge) + 30, cy - 2, staff)
        self._draw_tall_narrow_glyph(gray, sym.x, sym.y, s)
        real = self._note(int(edge) + 54, cy, staff)
        self._draw_filled_head(gray, real.x, real.y, s)
        self._draw_stem_up(gray, real.x, real.y, s)

        remaining = self._run_remove(gray, staff, [false_loop, sym, real])
        self.assertEqual([n.x for n in remaining], [real.x])

    def test_connected_flat_like_loop_does_not_anchor_or_preserve_symbol(self):
        s = 10.0
        gray, staff, edge = self._setup_staff_gray(s, left=35)
        cy = staff.lines[2]

        flat = self._note(int(edge) + 14, cy, staff, filled=False)
        self._draw_connected_flat_like_loop(gray, flat.x, flat.y, s)
        self._assert_one_connected_component(gray, flat.x, flat.y, s)
        intervening = self._note(int(edge) + 34, cy - 2, staff)
        self._draw_tall_narrow_glyph(gray, intervening.x, intervening.y, s)
        real = self._note(int(edge) + 58, cy, staff)
        self._draw_filled_head(gray, real.x, real.y, s)
        self._draw_stem_up(gray, real.x, real.y, s)

        remaining = self._run_remove(gray, staff, [flat, intervening, real])
        self.assertEqual([n.x for n in remaining], [real.x])

    def test_connected_clef_like_loop_does_not_anchor_or_preserve_symbol(self):
        s = 10.0
        gray, staff, edge = self._setup_staff_gray(s, left=35)
        cy = staff.lines[2]

        clef = self._note(int(edge) + 14, cy, staff, filled=False)
        self._draw_connected_clef_like_loop(gray, clef.x, clef.y, s)
        self._assert_one_connected_component(gray, clef.x, clef.y, s)
        intervening = self._note(int(edge) + 34, cy + 2, staff)
        self._draw_tall_narrow_glyph(gray, intervening.x, intervening.y, s)
        real = self._note(int(edge) + 58, cy, staff)
        self._draw_filled_head(gray, real.x, real.y, s)
        self._draw_stem_up(gray, real.x, real.y, s)

        remaining = self._run_remove(gray, staff, [clef, intervening, real])
        self.assertEqual([n.x for n in remaining], [real.x])

    def test_connected_digit_loop_does_not_anchor_or_preserve_symbol(self):
        s = 10.0
        gray, staff, edge = self._setup_staff_gray(s, left=35)
        cy = staff.lines[2]

        digit = self._note(int(edge) + 14, cy, staff, filled=False)
        self._draw_connected_digit_nine(gray, digit.x, digit.y, s)
        self._assert_one_connected_component(gray, digit.x, digit.y, s)
        intervening = self._note(int(edge) + 36, cy - 2, staff)
        self._draw_tall_narrow_glyph(gray, intervening.x, intervening.y, s)
        real = self._note(int(edge) + 60, cy, staff)
        self._draw_filled_head(gray, real.x, real.y, s)
        self._draw_stem_up(gray, real.x, real.y, s)

        remaining = self._run_remove(gray, staff, [digit, intervening, real])
        self.assertEqual([n.x for n in remaining], [real.x])


class HollowHeadScaleTests(unittest.TestCase):
    """Spacing-relative hollow-head white-hole detection."""

    def _black_canvas(self, h: int = 80, w: int = 120) -> np.ndarray:
        return np.zeros((h, w), dtype=np.uint8)

    def _draw_closed_ring(self, black: np.ndarray, x: int, y: int, s: float) -> None:
        """Closed elliptical hollow notehead ring on a black canvas."""
        w = max(6, int(round(an.NOTE_W_HOLLOW * s)))
        h = max(4, int(round(an.NOTE_H_HOLLOW * s)))
        thickness = max(2, int(round(0.25 * s)))
        cv2.ellipse(black, (x, y), (w // 2, h // 2), 0, 0, 360, 255, thickness)

    def _broken_upper_hollow_chord_fixture(self) -> tuple[np.ndarray, float, tuple[int, int], tuple[int, int]]:
        """Compact binary fixture: open upper ring, closed lower ring, and shared stem."""
        rows = """
#######.........................
#########.......................
........###.....................
..........##....................
...........##...................
................................
...............###..............
.............###..#.............
............###...##............
....#.......##...##.............
....#.......#...##..............
....#.......#####...............
....#...........................
....#...........................
....#...........................
....#.......#...................
....#...........................
....#...........................
....#...........................
....#...........................
....#...........................
....#.......#...................
....#...........................
....#...........................
....#...........................
....#.........#####.............
....#.......####..#.............
....#.......###...##............
....#.......#....##.............
....#.......#..###..............
....#.......#####...............
....#.......#...................
....#......##...................
....#.....###...................
....#...#####...................
....#..###......................
#########.......................
#######.........................
..###...........................
....#...........................
""".strip().splitlines()
        black = self._black_canvas(h=80, w=80)
        pad = 20
        for y, row in enumerate(rows):
            for x, pixel in enumerate(row):
                if pixel == "#":
                    black[pad + y, pad + x] = 255
        s = 6.0
        upper = (35, 28)
        lower = (35, 48)
        return black, s, upper, lower

    def _unrelated_aligned_open_symbol_fixture(
        self, kind: str
    ) -> tuple[np.ndarray, float, tuple[int, int], tuple[int, int]]:
        """Closed hollow plus a separate aligned glyph that closing can make hole-like."""
        black = self._black_canvas(h=100, w=100)
        s = 6.0
        symbol = (50, 35)
        hollow = (50, 55)
        self._draw_closed_ring(black, *hollow, s)
        cv2.ellipse(black, symbol, (5, 3), 0, 0, 360, 255, 2)
        black[symbol[1], symbol[0] - 5:symbol[0] - 3] = 0
        if kind == "rest":
            cv2.line(
                black,
                (symbol[0] + 5, symbol[1]),
                (symbol[0] + 5, symbol[1] - 15),
                255,
                2,
            )
        elif kind == "symbol":
            cv2.line(
                black,
                (symbol[0] - 12, symbol[1] - 3),
                (symbol[0] - 4, symbol[1] - 3),
                255,
                2,
            )
            cv2.line(
                black,
                (symbol[0] + 4, symbol[1] + 3),
                (symbol[0] + 12, symbol[1] + 3),
                255,
                2,
            )
        else:
            raise ValueError(kind)
        return black, s, symbol, hollow

    def _nearest_hole(self, holes: list[tuple[int, int]], x: int, y: int) -> tuple[int, int] | None:
        if not holes:
            return None
        return min(holes, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)

    def test_detects_small_spacing_closed_ring(self):
        s = 6.0
        black = self._black_canvas()
        cx, cy = 60, 40
        self._draw_closed_ring(black, cx, cy, s)

        holes = an.detect_hollow_heads(black, s)

        hit = self._nearest_hole(holes, cx, cy)
        self.assertIsNotNone(hit)
        self.assertLess(abs(hit[0] - cx), 2)
        self.assertLess(abs(hit[1] - cy), 2)

    def test_connected_chord_outline_yields_two_distinct_holes(self):
        s = 6.0
        black = self._black_canvas(h=120, w=140)
        x1, y1 = 50, 35
        x2, y2 = 50, 55
        self._draw_closed_ring(black, x1, y1, s)
        self._draw_closed_ring(black, x2, y2, s)
        stem_x = x1 + max(2, int(round(an.NOTE_W_HOLLOW * s)) // 2)
        cv2.line(black, (stem_x, y1 - 2), (stem_x, y2 + 2), 255, max(2, int(round(0.2 * s))))

        holes = an.detect_hollow_heads(black, s)

        self.assertEqual(len(holes), 2)
        near1 = self._nearest_hole(holes, x1, y1)
        near2 = self._nearest_hole(holes, x2, y2)
        self.assertIsNotNone(near1)
        self.assertIsNotNone(near2)
        self.assertNotEqual(near1, near2)
        self.assertLess(abs(near1[0] - x1), 2)
        self.assertLess(abs(near1[1] - y1), 2)
        self.assertLess(abs(near2[0] - x2), 2)
        self.assertLess(abs(near2[1] - y2), 2)

    def test_stacked_chord_recovers_open_upper_cavity_from_closed_lower_neighbor(self):
        black, s, upper, lower = self._broken_upper_hollow_chord_fixture()

        holes = an.detect_hollow_heads(black, s)

        self.assertEqual(len(holes), 2)
        for expected in (upper, lower):
            hit = self._nearest_hole(holes, *expected)
            self.assertIsNotNone(hit)
            self.assertLess(abs(hit[0] - expected[0]), 2)
            self.assertLess(abs(hit[1] - expected[1]), 2)

    def test_recovers_open_head_on_same_raw_shared_stem_component(self):
        black = self._black_canvas(h=100, w=100)
        s = 6.0
        upper = (50, 35)
        lower = (50, 55)
        for center in (upper, lower):
            cv2.ellipse(black, center, (5, 3), 0, 0, 360, 255, 2)
        cv2.line(black, (55, 32), (55, 58), 255, 2)
        black[upper[1], upper[0] - 5:upper[0] - 3] = 0

        holes = an.detect_hollow_heads(black, s)

        self.assertEqual(len(holes), 2)
        for expected in (upper, lower):
            hit = self._nearest_hole(holes, *expected)
            self.assertIsNotNone(hit)
            self.assertLess(abs(hit[0] - expected[0]), 2)
            self.assertLess(abs(hit[1] - expected[1]), 2)

    def test_recovered_stacked_chord_survives_notehead_validation(self):
        black, s, upper, lower = self._broken_upper_hollow_chord_fixture()
        staff = self._staff_from_center((upper[1] + lower[1]) / 2, s)
        page = an.Page(
            image=255 - black,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)

        hollows = [n for n in page.noteheads if not n.filled]
        self.assertEqual(len(hollows), 2)
        for expected in (upper, lower):
            hit = self._nearest_hole([(n.x, n.y) for n in hollows], *expected)
            self.assertIsNotNone(hit)
            self.assertLess(abs(hit[0] - expected[0]), 2)
            self.assertLess(abs(hit[1] - expected[1]), 2)

    def test_aligned_open_rest_is_not_recovered_from_unrelated_hollow(self):
        black, s, symbol, hollow = self._unrelated_aligned_open_symbol_fixture("rest")

        holes = an.detect_hollow_heads(black, s)

        self.assertEqual(len(holes), 1)
        self.assertLess(abs(holes[0][0] - hollow[0]), 2)
        self.assertLess(abs(holes[0][1] - hollow[1]), 2)
        self.assertGreater(abs(holes[0][1] - symbol[1]), 2)

    def test_aligned_open_symbol_is_not_recovered_from_unrelated_hollow(self):
        black, s, symbol, hollow = self._unrelated_aligned_open_symbol_fixture("symbol")

        holes = an.detect_hollow_heads(black, s)

        self.assertEqual(len(holes), 1)
        self.assertLess(abs(holes[0][0] - hollow[0]), 2)
        self.assertLess(abs(holes[0][1] - hollow[1]), 2)
        self.assertGreater(abs(holes[0][1] - symbol[1]), 2)

    def test_validator_stem_spanning_separate_open_loop_does_not_recover_it(self):
        black, s, upper, hollow = self._broken_upper_hollow_chord_fixture()
        _, labels, stats, _ = cv2.connectedComponentsWithStats(black, 8)
        upper_label = next(
            label
            for label in range(1, len(stats))
            if (
                stats[label][0] <= upper[0] < stats[label][0] + stats[label][2]
                and stats[label][1] <= upper[1] < stats[label][1] + stats[label][3]
                and stats[label][2] <= 2.0 * s
                and stats[label][3] <= 1.5 * s
            )
        )
        ys, xs = np.where(labels == upper_label)
        black[ys, xs] = 0
        black[ys, xs + 1] = 255
        symbol = (upper[0] + 1, upper[1])

        holes = an.detect_hollow_heads(black, s)

        self.assertEqual(len(holes), 1)
        self.assertLess(abs(holes[0][0] - hollow[0]), 2)
        self.assertLess(abs(holes[0][1] - hollow[1]), 2)

    def test_border_touching_white_region_not_detected(self):
        s = 6.0
        black = np.full((80, 120), 255, dtype=np.uint8)
        # Threshold-eligible 7x3 cavity touching left edge (s=6 bounds: w 2-8, h 2-7, a 4-38).
        black[38:41, 0:7] = 0
        cx_inset = 4
        cx_ring, cy = 80, 40
        w = max(6, int(round(an.NOTE_W_HOLLOW * s)))
        h = max(4, int(round(an.NOTE_H_HOLLOW * s)))
        t = max(2, int(round(0.25 * s)))
        inner_w = max(1, w - 2 * t)
        inner_h = max(1, h - 2 * t)
        cv2.ellipse(black, (cx_ring, cy), (inner_w // 2, inner_h // 2), 0, 0, 360, 0, -1)

        holes = an.detect_hollow_heads(black, s)
        self.assertEqual(len(holes), 1)
        self.assertLess(abs(holes[0][0] - cx_ring), 2)
        self.assertLess(abs(holes[0][1] - cy), 2)

        inset_only = np.full((80, 120), 255, dtype=np.uint8)
        inset_only[38:41, 1:8] = 0
        inset_holes = an.detect_hollow_heads(inset_only, s)
        self.assertEqual(len(inset_holes), 1)
        self.assertLess(abs(inset_holes[0][0] - cx_inset), 2)
        self.assertLess(abs(inset_holes[0][1] - 39), 2)

    def test_per_staff_dedup_keeps_pair_collapsed_by_global_max_spacing(self):
        centers = [(80, 36), (80, 39)]
        self.assertEqual(len(an._dedupe_hollow_centers(centers, 6.0)), 2)
        self.assertEqual(len(an._dedupe_hollow_centers(centers, 14.0)), 1)

    def test_wrong_staff_scale_candidate_rejected_in_overlap_band(self):
        s_small, s_large = 6.0, 14.0
        cy_small, cy_large = 40, 52
        gray = np.full((120, 160), 255, dtype=np.uint8)
        black = np.full((120, 160), 255, dtype=np.uint8)
        cx, cy = 80, 45
        w = max(6, int(round(an.NOTE_W_HOLLOW * s_large)))
        h = max(4, int(round(an.NOTE_H_HOLLOW * s_large)))
        t = max(2, int(round(0.25 * s_large)))
        inner_w = max(1, w - 2 * t)
        inner_h = max(1, h - 2 * t)
        cv2.ellipse(black, (cx, cy), (inner_w // 2, inner_h // 2), 0, 0, 360, 0, -1)
        gray = 255 - black
        staff_small = self._staff_from_center(cy_small, s_small)
        staff_large = self._staff_from_center(cy_large, s_large)
        page = an.Page(
            image=gray,
            binary=black,
            staves=[staff_small, staff_large],
            systems=[an.System(treble=staff_small, bass=staff_large)],
        )

        an.detect_noteheads(page)

        hollows = [n for n in page.noteheads if not n.filled]
        self.assertEqual(hollows, [], "large-scale-only cavity nearer small staff must be rejected")

    def test_close_small_staff_holes_stay_distinct_with_large_staff_present(self):
        s_small, s_large = 6.0, 14.0
        cy_small, cy_large = 40, 150
        gray = np.full((220, 160), 255, dtype=np.uint8)
        black = np.zeros((220, 160), dtype=np.uint8)
        cx = 80
        y1, y2 = 35, 40
        for y in (y1, y2):
            w = max(6, int(round(an.NOTE_W_HOLLOW * s_small)))
            h = max(4, int(round(an.NOTE_H_HOLLOW * s_small)))
            cv2.ellipse(black, (cx, y), (w // 2, h // 2), 0, 0, 360, 255,
                        max(2, int(round(0.25 * s_small))))
        stem_x = cx + max(2, int(round(an.NOTE_W_HOLLOW * s_small)) // 2)
        cv2.line(black, (stem_x, y1 - 2), (stem_x, y2 + 2), 255,
                 max(2, int(round(0.2 * s_small))))
        w = max(6, int(round(an.NOTE_W_HOLLOW * s_large)))
        h = max(4, int(round(an.NOTE_H_HOLLOW * s_large)))
        cv2.ellipse(black, (cx, cy_large), (w // 2, h // 2), 0, 0, 360, 255,
                    max(2, int(round(0.25 * s_large))))
        gray = 255 - black
        staff_small = self._staff_from_center(cy_small, s_small)
        staff_large = self._staff_from_center(cy_large, s_large)
        page = an.Page(
            image=gray,
            binary=black,
            staves=[staff_small, staff_large],
            systems=[an.System(treble=staff_small, bass=staff_large)],
        )

        an.detect_noteheads(page)

        small_hollows = sorted(
            [n for n in page.noteheads if not n.filled and abs(n.y - cy_small) < 20],
            key=lambda n: n.y,
        )
        self.assertEqual(len(small_hollows), 2)
        self.assertLess(abs(small_hollows[0].y - y1), 2)
        self.assertLess(abs(small_hollows[1].y - y2), 2)

    def _staff_page(self, gray: np.ndarray, staff: an.Staff) -> an.Page:
        return an.Page(
            image=gray,
            binary=(gray < 128).astype(np.uint8) * 255,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

    def _staff_from_center(self, cy: float, s: float) -> an.Staff:
        lines = [int(round(cy + (i - 2) * s)) for i in range(5)]
        return an.Staff(lines=lines, thickness=1, spacing=s)

    def _draw_closed_ring_gray(self, gray: np.ndarray, x: int, y: int, s: float) -> None:
        w = max(6, int(round(an.NOTE_W_HOLLOW * s)))
        h = max(4, int(round(an.NOTE_H_HOLLOW * s)))
        thickness = max(2, int(round(0.25 * s)))
        cv2.ellipse(gray, (x, y), (w // 2, h // 2), 0, 0, 360, 0, thickness)

    def test_staff_line_through_hollow_dyad_is_one_box_per_note(self):
        """谱线穿过空心和弦时，每个音一颗头，不能并成一个，也不能一音两框。"""
        s = 8.0
        black = np.zeros((90, 180), dtype=np.uint8)
        cy = 40
        lines = [int(round(cy + (i - 2) * s)) for i in range(5)]
        for y in lines:
            black[y, :] = 255
        upper = (90, lines[2])
        lower = (90, lines[2] + int(s))
        for x, y in (upper, lower):
            cv2.ellipse(black, (x, y), (8, 5), 0, 0, 360, 255, 1)
        # 左上开口：不处理谱线时白洞检测不到，谱线本身把符头切开。
        black[upper[1] - 3:upper[1] - 1, upper[0] - 8:upper[0] - 5] = 0
        staff = an.Staff(lines=lines, thickness=1, spacing=s)
        page = an.Page(
            image=255 - black,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)

        hollows = sorted([n for n in page.noteheads if not n.filled], key=lambda n: n.y)
        self.assertEqual(len(hollows), 2)
        self.assertLess(abs(hollows[0].y - upper[1]), 3)
        self.assertLess(abs(hollows[1].y - lower[1]), 3)
        self.assertLess(abs(hollows[0].x - upper[0]), 6)
        self.assertLess(abs(hollows[1].x - lower[0]), 6)

    def test_staff_line_through_hollow_triad_recovers_middle(self):
        """三音空心和弦被谱线切开时，中间音也要标上。"""
        s = 8.0
        black = np.zeros((90, 180), dtype=np.uint8)
        cy = 40
        lines = [int(round(cy + (i - 2) * s)) for i in range(5)]
        for y in lines:
            black[y, :] = 255
        heads = [
            (90, lines[1]),
            (90, lines[2]),
            (90, lines[3]),
        ]
        for x, y in heads:
            cv2.ellipse(black, (x, y), (8, 5), 0, 0, 360, 255, 1)
        black[heads[1][1] - 3:heads[1][1] - 1, heads[1][0] - 8:heads[1][0] - 5] = 0
        staff = an.Staff(lines=lines, thickness=1, spacing=s)
        page = an.Page(
            image=255 - black,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)
        an.assign_pitches(page)
        an._recover_stacked_hollow_heads(page)
        an._dedupe_noteheads(page)

        hollows = sorted([n for n in page.noteheads if not n.filled], key=lambda n: n.y)
        self.assertEqual(len(hollows), 3)
        for expected, got in zip(heads, hollows):
            self.assertLess(abs(got.x - expected[0]), 6)
            self.assertLess(abs(got.y - expected[1]), 4)

    def test_thin_ring_ledger_heads_below_staff_are_kept(self):
        """E4/C4/A3 三音和弦：下面两颗在加线上、环只有 1px 粗、右侧与符干粘连。"""
        s = 8.0
        black = np.zeros((120, 200), dtype=np.uint8)
        lines = [20 + int(round(i * s)) for i in range(5)]  # 20..52
        for y in lines:
            black[y, :] = 255
        bottom = lines[-1]
        x = 100
        heads_y = [bottom, bottom + 8, bottom + 16]  # E4, C4, A3
        for y in heads_y:
            cv2.ellipse(black, (x, y), (5, 3), 0, 0, 360, 255, 1)
        for y in (bottom + 8, bottom + 16):  # 加线
            black[y, x - 8:x + 9] = 255
        stem_x = x + 5
        cv2.line(black, (stem_x, heads_y[0] - 20), (stem_x, heads_y[-1]), 255, 1)
        staff = an.Staff(lines=lines, thickness=1, spacing=s)
        page = an.Page(
            image=255 - black,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)
        an.assign_pitches(page)
        an._recover_stacked_hollow_heads(page)
        an._dedupe_noteheads(page)
        an.assign_pitches(page)
        an.purge_staff_header_noteheads(page)

        hollows = sorted([n for n in page.noteheads if not n.filled], key=lambda n: n.y)
        self.assertEqual([n.y for n in hollows], heads_y)
        self.assertEqual([an.staff_step(staff, n.y) for n in hollows], [0, -2, -4])

    def test_faint_staff_line_still_detected(self):
        """扫描件里一条谱线只有 ~210 的浅灰，整个谱表不能因此丢掉。"""
        gray = np.full((120, 400), 255, dtype=np.uint8)
        ys = [40, 48, 56, 64, 72]
        for y in ys:
            gray[y, 20:380] = 60
        gray[56, 20:380] = 212  # 浅灰的那条
        lines = an.detect_staff_lines(gray)
        self.assertEqual([y for y, _ in lines], ys)
        self.assertEqual(len(an.group_staves(lines)), 1)

    def test_group_staves_fills_single_missing_line(self):
        lines = [(209, 1), (222, 1), (229, 1), (236, 1)]
        staves = an.group_staves(lines)
        self.assertEqual(len(staves), 1)
        self.assertEqual(staves[0].lines, [209, 216, 222, 229, 236])

    def test_staff_step_interpolates_uneven_lines(self):
        """低分辨率下线距 6/7 交错，谱表外要按平均线距外推，不能用最后一个线距。"""
        staff = an.Staff(lines=[344, 350, 357, 364, 370], thickness=1, spacing=6.5)
        self.assertEqual(an.staff_step(staff, 370), 0)
        self.assertEqual(an.staff_step(staff, 367), 1)
        self.assertEqual(an.staff_step(staff, 360), 3)  # 357 与 364 之间的间
        self.assertEqual(an.staff_step(staff, 377), -2)  # 下加一线 C4
        self.assertEqual(an.staff_step(staff, 384), -4)  # 下加二线 A3

    def test_broken_hollow_chord_with_stem_is_detected(self):
        """开口空心和弦没有封闭白洞时，仍应按模板检出。"""
        s = 8.0
        black = np.zeros((90, 200), dtype=np.uint8)
        cy = 40
        lines = [int(round(cy + (i - 2) * s)) for i in range(5)]
        for y in lines:
            black[y, :] = 255
        upper = (120, lines[2])
        lower = (120, lines[2] + int(s))
        for x, y in (upper, lower):
            cv2.ellipse(black, (x, y), (8, 5), 0, 0, 360, 255, 1)
            black[y - 1:y + 2, x - 6:x - 3] = 0
        cv2.line(black, (128, upper[1] - 4), (128, lower[1] + 4), 255, 1)
        staff = an.Staff(lines=lines, thickness=1, spacing=s)
        page = an.Page(
            image=255 - black,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)
        an.assign_pitches(page)
        an._recover_stacked_hollow_heads(page)
        an._dedupe_noteheads(page)

        hollows = sorted([n for n in page.noteheads if not n.filled], key=lambda n: n.y)
        self.assertGreaterEqual(len(hollows), 2)
        self.assertLess(abs(hollows[0].y - upper[1]), 5)
        self.assertLess(abs(hollows[-1].y - lower[1]), 5)

    def test_detect_noteheads_finds_s6_hollow_ring(self):
        s = 6.0
        gray = np.full((80, 120), 255, dtype=np.uint8)
        cx, cy = 60, 40
        self._draw_closed_ring_gray(gray, cx, cy, s)
        staff = self._staff_from_center(cy, s)
        page = self._staff_page(gray, staff)

        an.detect_noteheads(page)

        hollows = [n for n in page.noteheads if not n.filled]
        self.assertEqual(len(hollows), 1)
        self.assertLess(abs(hollows[0].x - cx), 2)
        self.assertLess(abs(hollows[0].y - cy), 2)

    def test_spacing_aware_validation_keeps_large_hollow_ring(self):
        s = 18.0
        gray = np.full((120, 200), 255, dtype=np.uint8)
        cx, cy = 100, 60
        self._draw_closed_ring_gray(gray, cx, cy, s)
        staff = self._staff_from_center(cy, s)
        page = self._staff_page(gray, staff)

        an.detect_noteheads(page)

        hollows = [n for n in page.noteheads if not n.filled]
        self.assertEqual(len(hollows), 1)
        self.assertLess(abs(hollows[0].x - cx), 3)
        self.assertLess(abs(hollows[0].y - cy), 3)

    def test_per_staff_spacing_detects_both_hollow_heads(self):
        s_small, s_large = 6.0, 14.0
        gray = np.full((220, 160), 255, dtype=np.uint8)
        cx = 80
        cy_small, cy_large = 40, 150
        self._draw_closed_ring_gray(gray, cx, cy_small, s_small)
        self._draw_closed_ring_gray(gray, cx, cy_large, s_large)
        staff_small = self._staff_from_center(cy_small, s_small)
        staff_large = self._staff_from_center(cy_large, s_large)
        page = an.Page(
            image=gray,
            binary=(gray < 128).astype(np.uint8) * 255,
            staves=[staff_small, staff_large],
            systems=[an.System(treble=staff_small, bass=staff_large)],
        )

        an.detect_noteheads(page)

        hollows = sorted([n for n in page.noteheads if not n.filled], key=lambda n: n.y)
        self.assertEqual(len(hollows), 2)
        self.assertLess(abs(hollows[0].y - cy_small), 2)
        self.assertLess(abs(hollows[1].y - cy_large), 3)

    def test_three_note_vertical_chord_merges_with_offset_x(self):
        s = 10.0
        gray = np.full((120, 120), 255, dtype=np.uint8)
        cx = 60
        y_top, y_mid, y_bot = 30, 42, 54
        for y in (y_top, y_mid, y_bot):
            self._draw_closed_ring_gray(gray, cx + (y - y_mid) // 6, y, s)
        black = (gray < 128).astype(np.uint8) * 255
        stem_x = cx + max(2, int(round(an.NOTE_W_HOLLOW * s)) // 2)
        cv2.line(black, (stem_x, y_top - 2), (stem_x, y_bot + 2), 255,
                 max(2, int(round(0.2 * s))))
        gray = 255 - black
        cy = (y_top + y_bot) / 2.0
        staff = self._staff_from_center(cy, s)
        page = an.Page(
            image=gray,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)
        an.build_chords(page)

        self.assertEqual(len(page.chords), 1)
        self.assertEqual(len(page.chords[0].heads), 3)

    def test_two_hollow_heads_merge_into_chord_event(self):
        s = 10.0
        gray = np.full((120, 120), 255, dtype=np.uint8)
        black = (gray < 128).astype(np.uint8) * 255
        cx = 60
        y1, y2 = 35, 55
        self._draw_closed_ring_gray(gray, cx, y1, s)
        self._draw_closed_ring_gray(gray, cx, y2, s)
        black = (gray < 128).astype(np.uint8) * 255
        stem_x = cx + max(2, int(round(an.NOTE_W_HOLLOW * s)) // 2)
        cv2.line(black, (stem_x, y1 - 2), (stem_x, y2 + 2), 255, max(2, int(round(0.2 * s))))
        gray = 255 - black  # keep stem in gray for consistency
        cy = (y1 + y2) / 2.0
        staff = self._staff_from_center(cy, s)
        page = an.Page(
            image=gray,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)
        an.assign_pitches(page)
        an.to_jianpu(page)
        an.build_chords(page)

        hollows = [n for n in page.noteheads if not n.filled]
        self.assertEqual(len(hollows), 2)
        self.assertEqual(len(page.chords), 1)
        self.assertEqual(len(page.chords[0].heads), 2)
        self.assertLess(abs(page.chords[0].x - cx), 2)

    def test_stemless_stack_recovers_crushed_inner_hollows(self):
        s = 10.0
        staff = an.Staff(lines=[40, 50, 60, 70, 80], thickness=1, spacing=s)
        ys = [40, 50, 60, 70]
        cx = 80
        w = max(6, int(round(an.NOTE_W_HOLLOW * s)))
        h = max(4, int(round(an.NOTE_H_HOLLOW * s)))
        t = max(2, int(round(0.25 * s)))
        gray = np.full((140, 180), 255, dtype=np.uint8)
        for y in ys:
            cv2.ellipse(gray, (cx, y), (w // 2, h // 2), 0, 0, 360, 0, t)
        for y in (50, 60):
            cv2.ellipse(
                gray, (cx, y),
                (max(1, w // 2 - t), max(1, h // 2 - t)),
                0, 0, 360, 0, -1,
            )
            gray[y, cx - 2:cx + 3] = 255
        page = an.Page(
            image=gray,
            binary=(gray < 128).astype(np.uint8) * 255,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)
        an.assign_pitches(page)
        an._recover_stacked_hollow_heads(page)
        an._dedupe_noteheads(page)

        hollows = sorted(
            [n for n in page.noteheads if not n.filled], key=lambda n: n.y
        )
        self.assertEqual(len(hollows), 4)
        for expected, got in zip(ys, hollows):
            self.assertLess(abs(got.y - expected), 4)
            self.assertLess(abs(got.x - cx), 4)

    def test_stemless_octave_does_not_invent_inner_heads(self):
        s = 10.0
        staff = an.Staff(lines=[40, 50, 60, 70, 80], thickness=1, spacing=s)
        cx, y_top, y_bot = 80, 40, 75
        gray = np.full((140, 180), 255, dtype=np.uint8)
        self._draw_closed_ring_gray(gray, cx, y_top, s)
        self._draw_closed_ring_gray(gray, cx, y_bot, s)
        page = an.Page(
            image=gray,
            binary=(gray < 128).astype(np.uint8) * 255,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)
        an.assign_pitches(page)
        an._recover_stacked_hollow_heads(page)
        an._dedupe_noteheads(page)

        hollows = [n for n in page.noteheads if not n.filled]
        self.assertEqual(len(hollows), 2)


class ChordGroupingTests(unittest.TestCase):
    def _page(self, staff, heads):
        gray = np.full((160, 400), 255, dtype=np.uint8)
        return an.Page(
            image=gray,
            binary=np.zeros_like(gray),
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
            noteheads=heads,
        )

    def test_flagged_note_is_one_digit(self):
        s = 10.0
        black = np.zeros((180, 160), dtype=np.uint8)
        cx, cy = 70, 90
        nw = int(round(an.NOTE_W_SOLID * s))
        nh = int(round(an.NOTE_H_SOLID * s))
        cv2.ellipse(black, (cx, cy), (nw // 2, nh // 2), -18, 0, 360, 255, -1)
        stem_x = cx + nw // 2 - 1
        cv2.line(black, (stem_x, cy - 2), (stem_x, cy - int(3.5 * s)), 255, 2)
        flag_top = cy - int(3.5 * s)
        pts = np.array([
            [stem_x, flag_top],
            [stem_x + int(1.3 * s), flag_top + int(0.8 * s)],
            [stem_x + int(0.4 * s), flag_top + int(1.6 * s)],
            [stem_x, flag_top + int(1.1 * s)],
        ], np.int32)
        cv2.fillPoly(black, [pts], 255)
        staff = an.Staff(lines=[50, 60, 70, 80, 90], thickness=1, spacing=s)
        page = an.Page(
            image=255 - black,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)
        an.assign_pitches(page)
        an._recover_stacked_hollow_heads(page)
        an._dedupe_noteheads(page)
        an.assign_pitches(page)
        an.to_jianpu(page)
        an.build_chords(page)

        self.assertEqual(len(page.chords), 1)
        self.assertEqual(len(page.chords[0].heads), 1)

    def test_beamed_chord_does_not_invent_notes_between_heads(self):
        s = 10.0
        black = np.zeros((160, 200), dtype=np.uint8)
        nw = int(round(an.NOTE_W_SOLID * s))
        nh = int(round(an.NOTE_H_SOLID * s))
        bx = 100
        for y in (55, 70, 85):
            cv2.ellipse(black, (bx, y), (nw // 2, nh // 2), -18, 0, 360, 255, -1)
        stem_x = bx + nw // 2 - 1
        cv2.line(black, (stem_x, 40), (stem_x, 90), 255, 2)
        cv2.rectangle(black, (stem_x - 36, 36), (stem_x + 2, 43), 255, -1)
        staff = an.Staff(lines=[40, 50, 60, 70, 80], thickness=1, spacing=s)
        page = an.Page(
            image=255 - black,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
        )

        an.detect_noteheads(page)
        an.assign_pitches(page)
        an._recover_stacked_hollow_heads(page)
        an._dedupe_noteheads(page)
        an.assign_pitches(page)
        an.to_jianpu(page)
        an.build_chords(page)

        self.assertEqual(len(page.chords), 1)
        self.assertEqual(len(page.chords[0].heads), 3)

    def test_same_pitch_in_one_chord_prints_once(self):
        s = 10.0
        staff = an.Staff(lines=[40, 50, 60, 70, 80], thickness=1, spacing=s)
        heads = [
            an.NoteHead(x=60, y=58, filled=True, staff=staff, letter=4, octave=4, digit=5),
            an.NoteHead(x=63, y=63, filled=True, staff=staff, letter=4, octave=4, digit=5),
            an.NoteHead(x=61, y=78, filled=True, staff=staff, letter=2, octave=4, digit=3),
        ]
        page = self._page(staff, heads)

        an.build_chords(page)

        self.assertEqual(len(page.chords), 1)
        self.assertEqual(sorted(h.digit for h in page.chords[0].heads), [3, 5])

    def test_sequential_melody_notes_stay_separate_events(self):
        s = 10.0
        staff = an.Staff(lines=[40, 50, 60, 70, 80], thickness=1, spacing=s)
        digits = [1, 3, 3, 5, 6, 6]
        ys = [70, 60, 60, 50, 40, 40]
        xs = [40 + i * int(1.8 * s) for i in range(len(digits))]
        heads = [
            an.NoteHead(x=x, y=y, filled=True, staff=staff, digit=d)
            for x, y, d in zip(xs, ys, digits)
        ]
        page = self._page(staff, heads)

        an.build_chords(page)

        self.assertEqual(len(page.chords), 6)
        self.assertTrue(all(len(ev.heads) == 1 for ev in page.chords))

    def test_adjacent_identical_vertical_chords_stay_separate(self):
        s = 10.0
        staff = an.Staff(lines=[40, 50, 60, 70, 80], thickness=1, spacing=s)
        ys = [40, 50, 60, 70]
        digits = [6, 4, 1, 4]
        gap = int(2.0 * s)
        heads = []
        for x0 in (60, 60 + gap):
            for y, d in zip(ys, digits):
                heads.append(
                    an.NoteHead(x=x0, y=y, filled=False, staff=staff, digit=d)
                )
        page = self._page(staff, heads)

        an.build_chords(page)

        events = sorted(page.chords, key=lambda ev: ev.x)
        self.assertEqual(len(events), 2)
        self.assertEqual([len(ev.heads) for ev in events], [4, 4])
        def _sig(ev):
            return sorted((h.digit, h.dots, h.prefix, h.dur) for h in ev.heads)

        self.assertEqual(_sig(events[0]), _sig(events[1]))


class StaffHeaderPurgeTests(unittest.TestCase):
    def test_faint_staff_header_is_removed(self):
        staff = an.Staff(lines=[40, 50, 60, 70, 80], thickness=1, spacing=10)
        gray = np.full((140, 300), 255, np.uint8)
        for y in staff.lines:
            gray[y, 30:280] = 210
        fake = an.NoteHead(x=65, y=60, filled=False, staff=staff)
        real = an.NoteHead(x=200, y=60, filled=False, staff=staff)
        page = an.Page(image=gray, binary=np.zeros_like(gray), staves=[staff],
                       noteheads=[fake, real])
        self.assertAlmostEqual(an._staff_left_edge(gray, staff), 30, delta=1)
        an.purge_staff_header_noteheads(page)
        self.assertEqual(page.noteheads, [real])

    def test_broad_clef_loop_cannot_shorten_header(self):
        staff = an.Staff(lines=[60, 70, 80, 90, 100], thickness=1, spacing=10)
        gray = np.full((160, 320), 255, np.uint8)
        for y in staff.lines:
            gray[y, 30:300] = 176
        # A broad G-clef curl with a filled center and long vertical stroke.
        cv2.ellipse(gray, (65, 85), (14, 11), 0, 0, 360, 0, 2)
        cv2.ellipse(gray, (65, 85), (6, 5), 0, 0, 360, 0, -1)
        cv2.line(gray, (71, 30), (71, 120), 0, 2)
        fake = an.NoteHead(x=65, y=85, filled=True, staff=staff)
        key = an.NoteHead(x=110, y=80, filled=False, staff=staff)
        real = an.NoteHead(x=210, y=80, filled=False, staff=staff)
        page = an.Page(image=gray, binary=(gray < 128).astype(np.uint8) * 255,
                       staves=[staff], noteheads=[fake, key, real])
        self.assertTrue(an._head_in_clef_glyph(page, fake, staff))
        self.assertAlmostEqual(an.staff_header_exclusive_right(page, staff), 145, delta=1)
        an.purge_staff_header_noteheads(page)
        self.assertEqual(page.noteheads, [real])

    def test_drops_title_zone_and_key_signature_false_heads(self):
        s = 10
        treble = an.Staff(lines=[80, 90, 100, 110, 120], thickness=1, spacing=s)
        bass = an.Staff(lines=[150, 160, 170, 180, 190], thickness=1, spacing=s)
        gray = np.full((220, 300), 220, dtype=np.uint8)
        black = np.zeros_like(gray)
        left = 40
        for y in treble.lines:
            cv2.line(gray, (left, y), (280, y), 80, 1)
        for y in bass.lines:
            cv2.line(gray, (left, y), (280, y), 80, 1)
        # 页眉误检（曲名区）
        title_fake = an.NoteHead(x=120, y=40, filled=True, staff=treble, digit=4)
        # 谱头调号误检
        key_fake = an.NoteHead(x=95, y=100, filled=False, staff=treble, digit=1)
        # 演奏区（需有符干证据才会锚定 music_start；此处仅测 x 在谱头外）
        real = an.NoteHead(x=200, y=100, filled=True, staff=treble, digit=3)
        page = an.Page(
            image=gray,
            binary=black,
            staves=[treble, bass],
            systems=[an.System(treble=treble, bass=bass)],
            noteheads=[title_fake, key_fake, real],
        )
        an.assign_pitches(page)
        an.purge_staff_header_noteheads(page)
        xs = [n.x for n in page.noteheads]
        self.assertNotIn(title_fake.x, xs)
        self.assertNotIn(key_fake.x, xs)
        self.assertIn(real.x, xs)


class JevRenderTests(unittest.TestCase):
    def test_jev_skip_suppresses_annotation(self):
        black = np.zeros((120, 200), dtype=np.uint8)
        gray = np.full_like(black, 220)
        staff = an.Staff(lines=[40, 50, 60, 70, 80], thickness=1, spacing=10)
        head = an.NoteHead(x=30, y=60, filled=False, staff=staff, digit=1)
        head2 = an.NoteHead(x=120, y=60, filled=True, staff=staff, digit=3)
        page = an.Page(
            image=gray,
            binary=black,
            staves=[staff],
            systems=[an.System(treble=staff, bass=staff)],
            noteheads=[head, head2],
            chords=[
                an.ChordEvent(x=head.x, heads=[head]),
                an.ChordEvent(x=head2.x, heads=[head2]),
            ],
        )
        decisions = [{"skip": True, "placement": None}, {"skip": False, "placement": "above"}]

        with patch("annotate.jev.jev_available", return_value=True), patch(
            "annotate.jev.decide_events", return_value=decisions
        ):
            vis = an.render(page, use_jev=True)

        green = _green_mask(vis)
        self.assertEqual(green[:, :70].sum(), 0, "skipped header event should not draw")
        self.assertGreater(green[:, 70:].sum(), 0, "non-header event should still annotate")

    def test_jev_payload_marks_gap_only_on_lower_staff(self):
        treble = an.Staff(lines=[10, 20, 30, 40, 50], thickness=1, spacing=10)
        bass = an.Staff(lines=[70, 80, 90, 100, 110], thickness=1, spacing=10)
        ev = an.ChordEvent(x=80, heads=[an.NoteHead(x=80, y=90, filled=True, staff=bass)])
        gray = np.zeros((140, 160), dtype=np.uint8)
        page = an.Page(image=gray, binary=gray, noteheads=ev.heads)
        top_payload = an._jev_event_payloads([ev], treble, True, 10, gray, page)[0]
        bass_payload = an._jev_event_payloads([ev], bass, False, 10, gray, page)[0]
        self.assertFalse(top_payload["candidates"]["gap"])
        self.assertFalse(bass_payload["candidates"]["gap"])
        self.assertTrue(bass_payload["candidates"]["below"])


class PipelineDebugDumpTests(unittest.TestCase):
    def _two_staff_page(self):
        gray = np.full((220, 300), 255, dtype=np.uint8)
        for y in (40, 50, 60, 70, 80):
            cv2.line(gray, (20, y), (280, y), 80, 1)
        for y in (130, 140, 150, 160, 170):
            cv2.line(gray, (20, y), (280, y), 80, 1)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        return gray, binary

    def test_process_image_writes_every_pipeline_slice(self):
        gray, binary = self._two_staff_page()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            an.process_image(
                gray, binary, quiet=True, use_jev=False, debug_dir=out,
            )
            for name in an.DEBUG_SLICE_NAMES:
                path = out / f"{name}.jpg"
                self.assertTrue(path.is_file(), f"missing slice {name}")
                img = cv2.imread(str(path))
                self.assertIsNotNone(img, f"unreadable slice {name}")
                self.assertEqual(img.shape[0], gray.shape[0])
                self.assertEqual(img.shape[1], gray.shape[1])

    def test_upload_saves_slices_under_debug_filename(self):
        import app as web

        gray, binary = self._two_staff_page()
        ok, buf = cv2.imencode(".png", gray)
        self.assertTrue(ok)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            web.process_one(
                bytes(buf), filename="My Score.png", debug_root=root,
            )
            folder = root / "My_Score"
            self.assertTrue(folder.is_dir())
            for name in an.DEBUG_SLICE_NAMES:
                self.assertTrue(
                    (folder / f"{name}.jpg").is_file(), f"missing {name}"
                )

    def test_build_canvas_scene_normalized_staff_and_labels(self):
        h, w = 200, 240
        gray = np.full((h, w), 255, dtype=np.uint8)
        black = np.zeros((h, w), dtype=np.uint8)
        s = 10
        treble = an.Staff(lines=[20, 30, 40, 50, 60], thickness=1, spacing=s)
        bass = an.Staff(lines=[80, 90, 100, 110, 120], thickness=1, spacing=s)
        sys = an.System(treble=treble, bass=bass)
        head = an.NoteHead(x=120, y=40, filled=True, staff=treble, digit=1)
        page = an.Page(
            image=gray,
            binary=black,
            systems=[sys],
            staves=[treble, bass],
            noteheads=[head],
        )
        placements = [
            {
                "staff_key": "0_0",
                "x": 120,
                "y": 12,
                "bottom": 14,
                "digit": 1,
                "dots": 0,
                "prefix": "",
            }
        ]
        scene = an.build_canvas_scene(page, placements)
        self.assertEqual(scene["width"], an.CANVAS_LAYOUT["width"])
        self.assertEqual(len(scene["systems"]), 1)
        lines = scene["systems"][0]["staves"][0]["lines"]
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[4] - lines[0], 4 * an.CANVAS_LAYOUT["line_spacing"])
        self.assertEqual(len(scene["labels"]), 1)
        self.assertEqual(scene["labels"][0]["digit"], 1)
        self.assertTrue(any(n["filled"] for n in scene["notes"]))
