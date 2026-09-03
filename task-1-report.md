# Task 1 implementation evidence

## 2026-08-31 — fix round 4

The report file was absent from the working tree at the start of this round, so
this entry creates it rather than replacing prior content.

### RED

Added three leading-zone negative fixtures:

- connected flat-like bowl plus tangent stem;
- connected clef-like loop plus tangent stroke;
- connected 9-like digit loop plus tangent stroke.

Each fixture calls `cv2.connectedComponents` and asserts exactly one foreground
component before exercising `_remove_leading_symbols`.

Command:

```text
.venv/bin/python -m unittest tests.test_annotate.LeadingSymbolZoneTests.test_connected_flat_like_loop_does_not_anchor_or_preserve_symbol tests.test_annotate.LeadingSymbolZoneTests.test_connected_clef_like_loop_does_not_anchor_or_preserve_symbol tests.test_annotate.LeadingSymbolZoneTests.test_connected_digit_loop_does_not_anchor_or_preserve_symbol -v
```

Observed before the production change: exit code 1, `Ran 3 tests`, three
failures. In every failure the connected false candidate and intervening symbol
survived (`[50, 70, 94]` or `[50, 72, 96]`) instead of only the genuine note
remaining (`[94]` or `[96]`).

### Geometry evidence and minimal fix

At `s=10`, the accepted connected false components had head-band size
`17x13` and enclosed-hole width `7`; genuine horizontal hollow half notes with
both up- and down-stems had head-band size `21x13` and enclosed-hole width
`11`. The existing own-component edge-stem topology check returned true for
all of them.

The production change therefore retains the component/tangent-stem topology
checks and adds only a notehead-specific cavity-width requirement:
`hw >= 0.8 * s`. It does not use nearby-component proximity.

### GREEN

Focused command:

```text
.venv/bin/python -m unittest tests.test_annotate.LeadingSymbolZoneTests -v
```

Observed after the production change: exit code 0, `Ran 17 tests`, `OK`.
This includes all three connected negatives and the genuine hollow ring,
up-stem half note, and down-stem half note positives.

Full command:

```text
.venv/bin/python -m py_compile annotate.py tests/test_annotate.py && .venv/bin/python -m unittest discover -s tests -v
```

Observed: exit code 0, `Ran 40 tests`, `OK`. Existing event deduplication and
placement tests remained green. IDE diagnostics reported no linter errors in
`annotate.py` or `tests/test_annotate.py`.

## 2026-08-31 — fix round 5 (final allowed round)

### User ruling

Inside each staff's bounded `[staff_left, staff_left + 7*s)` leading zone,
clef/key/time suppression takes priority. Hollow candidates, including
note-shaped and stemmed hollow candidates, must not stop removal. The accepted
tradeoff is that a rare hollow half/whole note very near the line start may be
omitted. A filled candidate may stop removal only when it has both a compact
local notehead and adjacent stem/beam evidence. The `staff_left + 7*s` edge is
exclusive, so candidates at or beyond it must remain untouched.

### RED

Replaced the contradictory in-zone hollow-preservation cases with policy tests
covering broad and narrow hollow loops inside the zone and hollow notes at and
outside the exclusive boundary.

Command:

```text
.venv/bin/python -m unittest tests.test_annotate.LeadingSymbolZoneTests.test_removes_broad_and_narrow_hollow_loops_within_zone tests.test_annotate.LeadingSymbolZoneTests.test_preserves_hollow_notes_at_and_outside_exclusive_zone_edge -v
```

Observed before the production change: exit code 1, `Ran 2 tests`, one failure.
Both in-zone hollow candidates remained in `page.noteheads` instead of the
expected empty list. The boundary test already passed, confirming the existing
exclusive-edge behavior before simplification.

### Minimal implementation

Removed hollow candidates from strong-anchor classification. Deleted the five
now-unused hollow-anchor geometry helpers and removed the redundant glyph
supplement pass. `_has_strong_filled_note_evidence` now requires `n.filled`,
a compact filled head, and adjacent stem/beam evidence. Each staff continues
to compute its own left edge and now also supplies its own spacing to the
evidence check. Removal remains the strict interval
`staff_left <= x < staff_left + 7*staff.spacing`.

Contradictory tests asserting that hollow rings or stemmed hollow half notes
anchor inside the zone were removed. Connected flat/clef/digit fixtures remain
as negative coverage, broad/narrow hollow loops are explicitly removed, and
hollow candidates at/outside the right boundary are explicitly preserved.

### GREEN

Focused command:

```text
.venv/bin/python -m unittest tests.test_annotate.LeadingSymbolZoneTests -v
```

Observed: exit code 0, `Ran 14 tests`, `OK`.

Full command:

```text
.venv/bin/python -m py_compile annotate.py tests/test_annotate.py && .venv/bin/python -m unittest discover -s tests -v
```

Observed: exit code 0, `Ran 37 tests`, `OK`. This includes the existing event
deduplication and placement suites. IDE diagnostics reported no linter errors
in `annotate.py` or `tests/test_annotate.py`.

## 2026-08-31 — fix round 6 (filled-anchor vs hard-boundary)

### User ruling

A strong filled note inside the `7*s` leading zone is preserved and may stop
deletion of other non-hollow candidates, but hollow candidates anywhere within
the full hard `[left, left + 7*s)` interval must still be suppressed.
Candidates at/outside the exclusive hard boundary remain untouched.

### RED

Added interaction regression
`test_strong_filled_anchor_still_suppresses_in_zone_hollows`: early strong filled
note, two in-zone hollow candidates after it, one filled note outside the hard
boundary. Expected `[strong.x, outside.x]`.

Command:

```text
.venv/bin/python -m unittest tests.test_annotate.LeadingSymbolZoneTests.test_strong_filled_anchor_still_suppresses_in_zone_hollows -v
```

Observed before the production change: exit code 1, `Ran 1 test`, one failure.
In-zone hollow candidates at `edge + 55` and `hard_right - 3` survived because
the filled anchor shortened the removal interval to `strong.x`, so only
`left <= x < anchor_right` was checked uniformly.

### Minimal implementation

`_remove_leading_symbols` now tracks per staff:

- `hard_right = left + 7*s` — exclusive end for hollow suppression;
- `anchor_right` — earliest strong filled note (or `hard_right` if none).

Removal rules:

- hollow (`not n.filled`): remove when `left <= x < hard_right`;
- non-hollow: remove when `left <= x < anchor_right`.

Per-staff left-edge and spacing behavior unchanged.

### GREEN

Focused command:

```text
.venv/bin/python -m unittest tests.test_annotate.LeadingSymbolZoneTests -v
```

Observed: exit code 0, `Ran 15 tests`, `OK`.

Full command:

```text
.venv/bin/python -m py_compile annotate.py tests/test_annotate.py && .venv/bin/python -m unittest discover -s tests -v
```

Observed: exit code 0, `Ran 38 tests`, `OK`. Event deduplication and placement
suites remained green. IDE diagnostics reported no linter errors in
`annotate.py` or `tests/test_annotate.py`.
