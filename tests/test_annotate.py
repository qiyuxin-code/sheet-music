import unittest

import numpy as np

import annotate as an


def _green_mask(vis: np.ndarray) -> np.ndarray:
    # int32 avoids uint8 overflow when differencing BGR channels.
    r = vis[:, :, 0].astype(np.int32)
    g = vis[:, :, 1].astype(np.int32)
    b = vis[:, :, 2].astype(np.int32)
    return (g > r + 40) & (g > b + 40)


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
        self.assertLess(int(np.where(green)[1].max()), ctx["head"].x + int(ctx["s"] * 1.8))

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
        self.assertLess(int(np.where(green)[1].max()), ctx["head"].x + int(ctx["s"] * 1.8))

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
