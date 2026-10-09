"""Jev (TypeSafe System One) decisions for jianpu placement and staff-header filtering."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from env_local import _load_local_env

_load_local_env()

log = logging.getLogger("jev")

_SKIP_THRESHOLD = 0.55

def api_key() -> str | None:
    return os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")


def jev_available() -> bool:
    return bool(api_key())


def decide_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one {skip, placement} per input event (same order).

    Each event should include:
      id, x, y, digit, filled, staff_clef, is_treble_staff,
      x_after_staff_left_in_spacing, first_music_x,
      candidates: {above, gap, below, right} booleans
    """
    if not events:
        log.debug("decide_events: no events")
        return []
    if not jev_available():
        log.info(
            "decide_events: JEV skipped (no TYPESAFE_API_KEY / JEV_API_KEY), n=%d",
            len(events),
        )
        return [{"skip": False, "placement": None} for _ in events]

    model = os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
    t0 = time.perf_counter()
    log.info(
        "decide_events: calling system_one model=%s n_events=%d questions=%d",
        model,
        len(events),
        len(events) * 2,
    )

    from typesafe_sdk import Choice, Noul

    questions: dict[str, Any] = {}
    for ev in events:
        i = ev["id"]
        questions[f"skip_{i}"] = Noul(
            instructions=(
                "Should we SKIP drawing a jianpu digit because this detection is in the "
                "staff header (clef, key signature, time signature, tempo marking) "
                "rather than playable music?"
            ),
        )
        crit = {
            "above": "Above the notehead (preferred when clear)",
            "gap": "In the gap between treble and bass staves (piano)",
            "below": "Below the bass staff line",
            "right": "To the right of the note when vertical space is tight",
        }
        allowed = {k: v for k, v in crit.items() if ev.get("candidates", {}).get(k)}
        if not allowed:
            allowed = {"above": crit["above"]}
        questions[f"place_{i}"] = Choice(
            instructions=(
                "Pick the best jianpu digit placement. Only choose among options "
                "that are marked as geometrically valid in the event candidates."
            ),
            criteria=allowed,
        )

    from typesafe_sdk import TypeSafeClient

    key = api_key()
    if not key:
        raise RuntimeError("JEV API key not configured")
    with TypeSafeClient(api_key=key) as client:
        response = client.system_one(
            model=model,
            state={"events": events},
            questions=questions,
        )

    elapsed = time.perf_counter() - t0
    out: list[dict[str, Any]] = []
    n_skip = 0
    placements: dict[str, int] = {}
    for ev in events:
        i = ev["id"]
        skip_prob = float(response.answers[f"skip_{i}"].noul)
        placement = response.answers[f"place_{i}"].choice
        skip = skip_prob >= _SKIP_THRESHOLD
        if skip:
            n_skip += 1
        placements[placement] = placements.get(placement, 0) + 1
        out.append({"skip": skip, "placement": placement})
    log.info(
        "decide_events: done in %.2fs skip=%d/%d placements=%s",
        elapsed,
        n_skip,
        len(events),
        placements,
    )
    return out
