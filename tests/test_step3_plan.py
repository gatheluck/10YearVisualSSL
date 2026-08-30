#!/usr/bin/env python3
"""The Step-3 porting plan lives on `main` and the plan is enforced here.

**This is a mechanism against a drift that actually happened.** The sequenced
Step-3 plan (Phase A1 first, then A2 ...) lived only in a document kept off
`main`, on a side branch; the declared on-`main` source of truth (`README.md`)
carried no ordering; and no test checked any of it. So across context
compactions the "next method" was reconstructed from impression: the A1
evaluation harness was skipped, `A2` was pursued as single-`linear_eval` ports,
and a Phase-B method (SigLIP) was ported before Phase A finished. Nothing went
red, because nothing was watching.

The fix is not more care. It is this: the ordered plan is a machine-readable
block inside `docs/STEP3_PORTING_PLAN.md` (on `main`, in the working tree every
session), and these tests make any future drift RED:

  - **accuracy** -- a plan item is marked `done` iff it is actually on disk, so a
    checkbox cannot lie;
  - **the next pointer** -- the plan names the single next unit of work, and it
    must be the earliest unfinished item, so "what is next" is a checked fact,
    not a memory;
  - **no new out-of-order work** -- the count of already-`done` items that sit
    *after* the next pointer (the ones ported ahead of their turn) may not grow
    past a ceiling frozen here in the test, so a new out-of-turn port cannot land
    silently;
  - **partition** -- every un-numbered directory under `methods/` is accounted
    for by the plan (as a Step-3 port or as an explicit non-Step-3 exception), so
    a ported Step-3 method cannot go unlisted.

This file **discovers** the method state from the filesystem and reads the plan
as data; it never names a method (tests/test_no_hard_coded_methods.py forbids a
shared file from doing so -- the plan document, being prose, may).
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "docs" / "STEP3_PORTING_PLAN.md"
METHODS = ROOT / "methods"

# The number of `done` items sitting *after* the `next` pointer -- ports
# completed ahead of their place in the order. It was 4 before A1 landed (the
# eval-only ports EVA-02/AIMv2/BEiT v2/SigLIP, all done before the A1 harness);
# once A1 (order 1) completed, `next` advanced past EVA-02/AIMv2/BEiT v2 (orders
# 2-4), leaving only SigLIP (order 13, a Phase-B item) genuinely out of turn --
# so the honest ceiling tightened to 1. It is a bare integer, not a method name,
# so it is allowed in a shared file. Changing it is the one way to admit (or
# retire) an out-of-order port, and that must be a deliberate, reviewed edit to
# this test -- which is exactly the point.
FROZEN_CEILING = 1


def _plan() -> dict:
    """The single machine-readable block inside the plan document."""
    text = PLAN.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.DOTALL)
    if len(blocks) != 1:
        raise AssertionError(
            f"expected exactly one ```json block in {PLAN.name}, "
            f"found {len(blocks)}")
    return json.loads(blocks[0])


def _done_on_disk(item: dict) -> bool:
    """Whether a plan item is genuinely present, measured -- not declared.

    A `method` item is done when its adapter exists; a `task` item is done when
    the artifact it names exists. A task with no artifact can never be done on
    disk, so declaring it done is caught by the accuracy test."""
    if item["kind"] == "method":
        return (METHODS / item["dir"] / "adapter" / "__init__.py").is_file()
    artifact = item.get("artifact")
    return bool(artifact) and (ROOT / artifact).exists()


def _unnumbered_dirs() -> "set[str]":
    """The method directories with no numeric prefix (the Step-3 namespace,
    plus the off-decade Step-1&2 ports the plan must list as exceptions)."""
    return {p.name for p in METHODS.iterdir()
            if p.is_dir() and not p.name.startswith(".")
            and not re.match(r"\d+_", p.name)}


class TestThePlanIsMachineReadable(unittest.TestCase):
    def test_there_is_exactly_one_json_block_and_it_parses(self):
        plan = _plan()
        self.assertIn("items", plan)
        self.assertGreater(len(plan["items"]), 10,
                           "the plan should record the whole A-F programme")

    def test_every_item_is_well_formed(self):
        items = _plan()["items"]
        ids, orders = set(), []
        for it in items:
            for key in ("id", "phase", "subphase", "order", "kind", "title",
                        "status"):
                self.assertIn(key, it, f"item {it.get('id')!r} lacks {key}")
            self.assertIn(it["status"], ("done", "todo"))
            self.assertIn(it["kind"], ("method", "task"))
            if it["kind"] == "method":
                self.assertIn("dir", it, f"method {it['id']!r} lacks a dir")
            self.assertNotIn(it["id"], ids, f"duplicate id {it['id']!r}")
            ids.add(it["id"])
            orders.append(it["order"])
        self.assertEqual(orders, sorted(orders),
                         "the `order` fields must ascend in listing order")
        self.assertEqual(len(orders), len(set(orders)), "a duplicate order")

    def test_the_plan_records_the_whole_programme(self):
        """All six Step-3 families are present, so the full roadmap (3DFM, 4DFM
        and the rest) is on main -- not just the phase in flight."""
        phases = {it["phase"] for it in _plan()["items"]}
        self.assertEqual(phases, set("ABCDEF"))


class TestACheckboxCannotLie(unittest.TestCase):
    def test_status_matches_what_is_on_disk(self):
        wrong = [it["id"] for it in _plan()["items"]
                 if (it["status"] == "done") != _done_on_disk(it)]
        self.assertEqual(
            wrong, [],
            "these items' `status` disagrees with the filesystem (a method "
            "marked done with no adapter, or built but left `todo`): "
            + ", ".join(wrong))

    def test_the_check_is_not_vacuous(self):
        """A positive control: there is at least one done and one todo item, so
        the accuracy test above is exercising both truth values."""
        statuses = {it["status"] for it in _plan()["items"]}
        self.assertEqual(statuses, {"done", "todo"})


class TestTheNextPointerIsAFact(unittest.TestCase):
    def test_next_is_the_earliest_unfinished_item(self):
        plan = _plan()
        todo = sorted((it for it in plan["items"] if it["status"] == "todo"),
                      key=lambda it: it["order"])
        self.assertTrue(todo, "the plan has no unfinished work -- Step 3 done?")
        self.assertEqual(
            plan["next"], todo[0]["id"],
            "plan['next'] must name the earliest unfinished item; work does not "
            "skip ahead")


class TestNoNewOutOfOrderWork(unittest.TestCase):
    def test_the_ceiling_is_frozen(self):
        """The plan's declared ceiling may not be bumped without editing this
        test -- the control that stops a new out-of-turn port being waved
        through by widening the allowance."""
        self.assertEqual(_plan()["grandfathered_ceiling"], FROZEN_CEILING)

    def test_done_items_after_the_next_pointer_stay_within_the_ceiling(self):
        plan = _plan()
        order_of = {it["id"]: it["order"] for it in plan["items"]}
        nxt = order_of[plan["next"]]
        late_done = [it["id"] for it in plan["items"]
                     if it["status"] == "done" and it["order"] > nxt]
        self.assertLessEqual(
            len(late_done), FROZEN_CEILING,
            "more items are done ahead of their turn than the frozen ceiling "
            f"allows -- a new out-of-order port slipped in: {late_done}")


class TestEveryPortIsAccountedFor(unittest.TestCase):
    def test_the_unnumbered_namespace_is_partitioned_by_the_plan(self):
        plan = _plan()
        non_step3 = set(plan["non_step3_unnumbered"])
        planned = {it["dir"] for it in plan["items"] if it["kind"] == "method"}
        on_disk_planned = {d for d in planned if (METHODS / d).is_dir()}
        unnumbered = _unnumbered_dirs()

        self.assertEqual(
            unnumbered, on_disk_planned | non_step3,
            "an un-numbered method directory is not accounted for: it is neither "
            "a Step-3 port in the plan nor an explicit non-Step-3 exception. "
            f"unlisted: {unnumbered - on_disk_planned - non_step3}; "
            f"stale: {(on_disk_planned | non_step3) - unnumbered}")

    def test_the_two_sets_do_not_overlap(self):
        plan = _plan()
        planned = {it["dir"] for it in plan["items"] if it["kind"] == "method"}
        self.assertEqual(
            planned & set(plan["non_step3_unnumbered"]), set(),
            "a directory is listed both as a Step-3 port and a non-Step-3 "
            "exception")


if __name__ == "__main__":
    unittest.main()
