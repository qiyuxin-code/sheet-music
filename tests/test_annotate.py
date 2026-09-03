import unittest
from unittest.mock import patch

import cv2
import numpy as np

import annotate as an


def _green_mask(vis: np.ndarray) -> np.ndarray:
    # int32 avoids uint8 overflow when differencing BGR channels.
    r = vis[:, :, 0].astype(np.int32)
    g = vis[:, :, 1].astype(np.int32)
    b = vis[:, :, 2].astype(np.int32)
    return (g > r + 40) & (g > b + 40)


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

    def test_bass_prefers_clear_position_above_event(self):
        black, occupied, bass, head, event, bounds = self._single_bass_fixture()

        added = self._render_bass(black, occupied, event, bass, bounds)

        _, _, _, annotation_bottom = self._annotation_bounds(added)
        self.assertLessEqual(
            annotation_bottom,
            head.y,
            "a clear annotation box above the bass event must beat below/right",
        )

    def test_bass_searches_farther_up_when_nearest_above_is_blocked(self):
        black, occupied, bass, head, event, bounds = self._single_bass_fixture()
        members = self._members(event)
        th = members[0][3]
        y_row = head.y - int(self.S * 0.6) - th // 2 - 2
        near_left, near_top, near_right, near_bottom = an._stack_bounds_xy(
            y_row, 0, -1, members, max(int(th * 0.95), 1)
        )
        black[near_top:near_bottom, near_left:near_right] = 255

        added = self._render_bass(black, occupied, event, bass, bounds)

        _, _, _, annotation_bottom = self._annotation_bounds(added)
        self.assertLessEqual(
            annotation_bottom,
            near_top,
            "placement must continue upward past the blocked nearest box",
        )

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
        expected_anchor = (
            clearance_top + (len(members) - 1) * step + members[-1][6]
        )
        expected = np.zeros_like(occupied)
        expected_centers = []
        for i, member in enumerate(members):
            _, x, _, _, _, _, top_off, bottom_off, right_ext = member
            center_y = expected_anchor - i * step
            expected_centers.append(center_y)
            expected[
                center_y - top_off:center_y + bottom_off,
                x - 2:x + right_ext + 2,
            ] = True
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

        np.testing.assert_array_equal(occupied, expected)
        self.assertGreater(
            expected_centers[0],
            expected_centers[1],
            "lower-pitch member must be lower on the rendered page",
        )
        _, actual_top, _, actual_bottom = self._annotation_bounds(occupied)
        self.assertGreaterEqual(actual_top, clearance_top)
        self.assertLessEqual(actual_bottom, h)
        low_prefix_x = members[0][1] + members[0][2] + 3
        low_center, high_center = expected_centers
        green = _green_mask(vis)
        self.assertTrue(
            green[low_center - th // 2:low_center + th // 2 + 1,
                  low_prefix_x:].any(),
            "the low member's prefix must be drawn at the lower center",
        )
        self.assertTrue(
            green[high_center - th // 2:high_center + th // 2 + 1,
                  members[1][1]:members[1][1] + members[1][2]].any(),
            "the high member must be drawn at the upper center",
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

        annotation_left, _, _, _ = self._annotation_bounds(added)
        self.assertGreaterEqual(
            annotation_left,
            first_right,
            "a clear farther-right annotation box must beat compatibility fallback",
        )

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
        self.assertEqual(annotation_left, exact_left)
        self.assertEqual(annotation_right, w)

    def test_all_candidates_use_in_image_direction_preserving_fallback(self):
        h, w = 80, 80
        black = np.full((h, w), 255, dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        bass = an.Staff(
            lines=[30, 40, 50, 60, 70], thickness=1, spacing=self.S
        )
        high = an.NoteHead(
            x=74, y=40, filled=True, staff=bass, digit=1
        )
        low = an.NoteHead(
            x=74, y=60, filled=True, staff=bass, digit=8, prefix="#"
        )
        event = an.ChordEvent(x=74, heads=[high, low])
        members = self._members(event)
        th = members[0][3]
        step = max(int(th * 0.95), 1)
        y_row = high.y - int(self.S * 0.6) - th // 2 - 2
        expected_y, expected_x_off, expected_direction = an._fit_compat_stack(
            h, w, y_row, 0, -1, members, step
        )
        expected = np.zeros_like(occupied)
        for i, member in enumerate(members):
            _, x, _, _, _, _, top_off, bottom_off, right_ext = member
            box_x = x + expected_x_off
            box_y = expected_y + expected_direction * i * step
            expected[
                box_y - top_off:box_y + bottom_off,
                box_x - 2:box_x + right_ext + 2,
            ] = True

        added = self._render_bass(
            black, occupied, event, bass, (0, 20, 30, 70)
        )

        self.assertEqual(expected_direction, -1)
        self.assertTrue(an._stack_fits_image(
            h, w, expected_y, expected_x_off, expected_direction, members, step
        ))
        np.testing.assert_array_equal(added, expected)


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

        green = _green_mask(vis)
        ys, xs = np.where(green)
        self.assertGreater(len(xs), 0)
        self.assertGreater(xs.min(), head.x)

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
        bass_head = an.NoteHead(x=60, y=80, filled=True, staff=bass1, digit=3)
        treble_head = an.NoteHead(x=60, y=122, filled=True, staff=treble2, digit=5)
        bass_event = an.ChordEvent(x=60, heads=[bass_head])
        page = an.Page(
            image=gray,
            binary=black,
            systems=[sys1, sys2],
            staves=[treble1, bass1, treble2, bass2],
            chords=[
                bass_event,
                an.ChordEvent(x=60, heads=[treble_head]),
            ],
        )

        vis1 = np.full((h, w, 3), 255, dtype=np.uint8)
        occupied1 = np.zeros((h, w), dtype=bool)
        an._render_group(
            vis1,
            black,
            [bass_event],
            bass1,
            top=False,
            s=s,
            scale=0.7,
            mid_y=55,
            bounds=(10, 50, 60, 100),
            occupied=occupied1,
        )
        sys1_green_count = int(_green_mask(vis1).sum())

        vis = an.render(page)
        overlap_count = int((_green_mask(vis) & occupied1).sum())
        self.assertEqual(
            overlap_count,
            sys1_green_count,
            "later systems must not draw over earlier-system occupied pixels",
        )

    def test_bass_uses_right_when_gap_and_bottom_unavailable(self):
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

        green = _green_mask(vis)
        ys, xs = np.where(green)
        self.assertGreater(len(xs), 0)
        self.assertGreater(xs.min(), head.x)

    def test_bass_all_blocked_uses_in_image_compatibility_fallback(self):
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

        green = _green_mask(ctx["vis"])
        self._assert_green_fully_in_image(ctx["vis"], green)
        lane_y0, lane_y1, lane_x0, lane_x1 = ctx["lane"]
        lane = green[lane_y0:lane_y1, lane_x0:lane_x1]
        self.assertFalse(lane.any(), "must not draw in blocked right lane")

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

        green = _green_mask(ctx["vis"])
        self._assert_green_fully_in_image(ctx["vis"], green)
        lane_y0, lane_y1, lane_x0, lane_x1 = ctx["lane"]
        self.assertFalse(green[lane_y0:lane_y1, lane_x0:lane_x1].any())
        self.assertGreater(
            int(np.where(green)[1].min()),
            ctx["head"].x + int(ctx["s"] * 1.8),
        )

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

        green = _green_mask(ctx["vis"])
        self._assert_green_fully_in_image(ctx["vis"], green)
        lane_y0, lane_y1, lane_x0, lane_x1 = ctx["lane"]
        self.assertFalse(green[lane_y0:lane_y1, lane_x0:lane_x1].any())
        self.assertGreater(
            int(np.where(green)[1].min()),
            ctx["head"].x + int(ctx["s"] * 1.8),
        )

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

    def test_render_compat_near_right_edge_uses_in_image_box(self):
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
        self.assertTrue(_green_mask(vis).any(), "expected compat render output")


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

    def test_dedupes_only_consecutive_identical_events(self):
        staff = self._staff()
        first = self._event(staff, 10)
        duplicate = self._event(staff, 20)
        different_duration = self._event(staff, 30, dur=0.5)
        repeated_after_change = self._event(staff, 40)

        result = an._dedupe_consecutive_events(
            [first, duplicate, different_duration, repeated_after_change]
        )

        self.assertEqual(result, [first, different_duration, repeated_after_change])

    def test_different_octave_retained(self):
        staff = self._staff()
        a = self._event(staff, 10, dots=0)
        b = self._event(staff, 20, dots=1)

        result = an._dedupe_consecutive_events([a, b])
        self.assertEqual(result, [a, b])

    def test_different_prefix_retained(self):
        staff = self._staff()
        a = self._event(staff, 10, prefix="")
        b = self._event(staff, 20, prefix="#")

        result = an._dedupe_consecutive_events([a, b])
        self.assertEqual(result, [a, b])

    def test_chord_dedupes_with_different_head_order(self):
        staff = self._staff()
        h1 = an.NoteHead(x=10, y=30, filled=True, staff=staff, digit=1, dur=1.0)
        h2 = an.NoteHead(x=10, y=40, filled=True, staff=staff, digit=3, dur=1.0)
        ev1 = an.ChordEvent(x=10, heads=[h1, h2])
        h3 = an.NoteHead(x=20, y=40, filled=True, staff=staff, digit=3, dur=1.0)
        h4 = an.NoteHead(x=20, y=30, filled=True, staff=staff, digit=1, dur=1.0)
        ev2 = an.ChordEvent(x=20, heads=[h3, h4])

        result = an._dedupe_consecutive_events([ev1, ev2])
        self.assertEqual(result, [ev1])

    def test_duplicate_member_events_share_signature(self):
        staff = self._staff()
        a = an.NoteHead(x=10, y=30, filled=True, staff=staff, digit=1, dur=1.0)
        dup = an.NoteHead(x=10, y=31, filled=True, staff=staff, digit=1, dur=1.0)
        b = an.NoteHead(x=10, y=40, filled=True, staff=staff, digit=3, dur=1.0)
        ev_aab = an.ChordEvent(x=10, heads=[a, dup, b])
        ev_ab = an.ChordEvent(x=20, heads=[a, b])

        self.assertEqual(an._event_signature(ev_aab), an._event_signature(ev_ab))

        result = an._dedupe_consecutive_events([ev_aab, ev_ab])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].x, ev_aab.x)
        self.assertEqual(an._event_signature(result[0]), an._event_signature(ev_ab))

    def test_staff_isolation(self):
        staff_a = self._staff()
        staff_b = an.Staff(lines=[90, 100, 110, 120, 130], thickness=1, spacing=10)
        a1 = self._event(staff_a, 10)
        b1 = self._event(staff_b, 10, y=100)
        a2 = self._event(staff_a, 20)

        result = an._dedupe_consecutive_events([a1, b1, a2])
        self.assertEqual(result, [a1, b1])

    def test_different_event_on_same_staff_breaks_run(self):
        staff = self._staff()
        first = self._event(staff, 10)
        different = self._event(staff, 20, dur=0.5)
        repeated = self._event(staff, 30)

        result = an._dedupe_consecutive_events([first, different, repeated])
        self.assertEqual(result, [first, different, repeated])

    def test_collapses_duplicate_members_within_event(self):
        staff = self._staff()
        dup1 = an.NoteHead(x=10, y=30, filled=True, staff=staff, digit=1, dur=1.0)
        dup2 = an.NoteHead(x=12, y=31, filled=True, staff=staff, digit=1, dur=1.0)
        other = an.NoteHead(x=11, y=40, filled=True, staff=staff, digit=3, dur=1.0)
        ev = an.ChordEvent(x=10, heads=[dup1, dup2, other])

        view = an._event_view_for_render(ev)
        self.assertEqual(len(view.heads), 2)
        self.assertEqual(
            {(h.digit, h.dots, h.prefix, h.dur) for h in view.heads},
            {(1, 0, "", 1.0), (3, 0, "", 1.0)},
        )
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
    def test_render_passes_only_first_duplicate_to_render_group(self, mock_render_group):
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
        self.assertEqual(len(rendered_evs), 1)
        self.assertIs(rendered_evs[0], ev1)


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
        zone_right = edge + 7.0 * s
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
        zone_right = edge + 7.0 * s

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
