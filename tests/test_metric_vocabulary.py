#!/usr/bin/env python3
"""`metrics.json` uses one vocabulary, and it says what may be compared.

Three ported stages produced three different names for the same kinds of
number -- `val_acc1` against `best_top1_acc` and `final_top1_acc`, `val_loss`
against `final_loss`, `global_step` against `epochs`. CONTRACT section 7 had
deferred the choice until two pilots were through; they now are, and this is
the choice.

**The important part is what may NOT share a name.** The `val_acc1` of the
first port's pretext stage is the accuracy of that method's *own* task --
eight-way patch-position classification, measured from `num_classes=8` in its
own source. The `final_top1_acc` of the linear evaluation is downstream
classification against real labels. Folding those two into one
name would let a machine build a comparison table out of numbers that are not
comparable, and the table would look right. That is the most expensive form of
"a name is a label, not evidence".

So comparability lives in the vocabulary rather than in a sentence in a
document, because a sentence in a document does not hold. A caller who wants
to put a per-method number in the same column as a comparable one has to
write that intent out.

The upstream names are kept beside the contract ones. Discarding them would
lose what the original called its own numbers, which is a silent loss.
"""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import adapterlib                                          # noqa: E402

METHODS = ROOT / "methods"


class TestTheVocabularyItself(unittest.TestCase):
    def test_it_exists_and_is_not_empty(self):
        self.assertTrue(adapterlib.METRIC_VOCABULARY)

    def test_every_name_declares_whether_it_can_be_compared(self):
        """The declaration is the whole point. A vocabulary that only lists
        names leaves the reader to guess which columns may be put side by
        side, and the guess is not recoverable from the number."""
        allowed = {adapterlib.COMPARABLE, adapterlib.PER_METHOD}
        for name, kind in adapterlib.METRIC_VOCABULARY.items():
            with self.subTest(metric=name):
                self.assertIn(kind, allowed)

    def test_both_kinds_are_actually_used(self):
        """With every entry in one bucket the distinction is decorative."""
        kinds = set(adapterlib.METRIC_VOCABULARY.values())
        self.assertEqual(kinds, {adapterlib.COMPARABLE, adapterlib.PER_METHOD})

    def test_the_pretext_numbers_are_never_comparable(self):
        """Each method's pretext task is a different task. Eight-way patch
        position and a reconstruction objective share no scale."""
        for name, kind in adapterlib.METRIC_VOCABULARY.items():
            if "pretext" in name:
                with self.subTest(metric=name):
                    self.assertEqual(kind, adapterlib.PER_METHOD)

    def test_the_linear_probe_numbers_are_comparable(self):
        """This is the number the whole project exists to compare."""
        probes = [n for n in adapterlib.METRIC_VOCABULARY
                  if "linear_probe" in n]
        self.assertTrue(probes)
        for name in probes:
            with self.subTest(metric=name):
                self.assertEqual(adapterlib.METRIC_VOCABULARY[name],
                                 adapterlib.COMPARABLE)

    def test_no_name_is_both(self):
        """A name carrying both words would sit in both buckets and the
        prefix would stop meaning anything."""
        for name in adapterlib.METRIC_VOCABULARY:
            with self.subTest(metric=name):
                self.assertFalse("pretext" in name and "linear_probe" in name)


class TestWritingMetrics(unittest.TestCase):
    def setUp(self) -> None:
        import shutil
        import tempfile
        self.out = Path(tempfile.mkdtemp(prefix="vocab-"))
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        # A contract stage, because the stage now decides which family of
        # names is reachable. "" would be refused, and rightly.
        self.ctx = adapterlib.Context(out=self.out, config={"seed": 1},
                                      stage="step1")

    def written(self) -> dict:
        return json.loads((self.out / "metrics.json").read_text())

    def test_the_contract_names_and_the_original_names_are_both_kept(self):
        self.ctx.write_metrics({"val_acc1": 12.5},
                               names={"val_acc1":
                                      "final_pretext_top1_accuracy"})
        doc = self.written()
        self.assertEqual(doc["metrics"],
                         {"final_pretext_top1_accuracy": 12.5})
        self.assertEqual(doc["metrics_raw"], {"val_acc1": 12.5})

    def test_the_schema_version_says_the_shape_changed(self):
        self.ctx.write_metrics({"val_acc1": 1.0},
                               names={"val_acc1":
                                      "final_pretext_top1_accuracy"})
        self.assertEqual(self.written()["schema_version"], 2)

    def test_a_name_outside_the_vocabulary_is_refused(self):
        with self.assertRaises(adapterlib.AdapterError) as e:
            self.ctx.write_metrics({"x": 1.0}, names={"x": "top1"})
        self.assertIn("top1", str(e.exception))

    def test_an_unmapped_original_key_is_refused_not_dropped(self):
        """Dropping it silently would lose a number the original produced,
        and nothing would say so."""
        with self.assertRaises(adapterlib.AdapterError) as e:
            self.ctx.write_metrics({"val_acc1": 1.0, "surprise": 2.0},
                                   names={"val_acc1":
                                          "final_pretext_top1_accuracy"})
        self.assertIn("surprise", str(e.exception))

    def test_a_key_deliberately_given_no_contract_name_is_still_recorded(self):
        """`None` is how a port says "this one has no contract slot". It is
        kept in the original block, so nothing is lost, and it is kept out of
        the comparable block, so nothing is invented."""
        self.ctx.write_metrics(
            {"val_acc1": 1.0, "odd": 2.0},
            names={"val_acc1": "final_pretext_top1_accuracy", "odd": None})
        doc = self.written()
        self.assertNotIn("odd", doc["metrics"])
        self.assertEqual(doc["metrics_raw"]["odd"], 2.0)

    def test_two_originals_may_not_collide_on_one_contract_name(self):
        """Whichever was written last would win, silently."""
        with self.assertRaises(adapterlib.AdapterError) as e:
            self.ctx.write_metrics(
                {"a": 1.0, "b": 2.0},
                names={"a": "final_pretext_loss", "b": "final_pretext_loss"})
        self.assertIn("final_pretext_loss", str(e.exception))

    def test_values_must_still_be_numbers(self):
        with self.assertRaises(adapterlib.AdapterError):
            self.ctx.write_metrics({"val_acc1": "12.5"},
                                   names={"val_acc1":
                                          "final_pretext_top1_accuracy"})

    def test_a_boolean_is_not_a_number(self):
        with self.assertRaises(adapterlib.AdapterError):
            self.ctx.write_metrics({"val_acc1": True},
                                   names={"val_acc1":
                                          "final_pretext_top1_accuracy"})

    def test_the_table_is_required(self):
        """Without it a port could go back to inventing names."""
        with self.assertRaises(TypeError):
            self.ctx.write_metrics({"val_acc1": 1.0})


class TestEveryPortUsesTheVocabulary(unittest.TestCase):
    """Discovered, never listed -- the adapters are found, not named.

    A list of methods here would be the mistake `test_no_hard_coded_methods`
    exists to stop, and it would go stale at the third port.
    """

    # `_reference` is not a method under study. It trains nothing and is
    # driven entirely by its config, so it has no fixed set of numbers to
    # name and cannot carry a fixed table. Same exemption, same reason, as
    # `tests/test_no_hard_coded_methods.py`.
    #
    # It is one named entry, it is checked to exist, and it is checked not to
    # swallow everything -- an exemption nobody can verify is a hole.
    EXEMPT = {"_reference"}

    @staticmethod
    def adapters() -> list[Path]:
        return sorted(METHODS.glob("*/adapter/__init__.py"))

    @classmethod
    def ports(cls) -> list[Path]:
        return [p for p in cls.adapters()
                if p.parent.parent.name not in cls.EXEMPT]

    def test_every_exempt_adapter_exists(self):
        """An exemption naming something absent is a stale hole."""
        present = {p.parent.parent.name for p in self.adapters()}
        for name in self.EXEMPT:
            with self.subTest(exempt=name):
                self.assertIn(name, present)

    def test_the_exemption_does_not_cover_everything(self):
        self.assertTrue(self.ports())

    @staticmethod
    def targets(path: Path) -> list[str]:
        """Contract names a module maps to, read without importing it.

        Importing every adapter would drag in each method's dependencies,
        which are deliberately not all installed at once.
        """
        out = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id.endswith("METRIC_NAMES")
                       for t in node.targets):
                continue
            if isinstance(node.value, ast.Dict):
                for v in node.value.values:
                    if isinstance(v, ast.Constant) and v.value is not None:
                        out.append(v.value)
        return out

    @staticmethod
    def pairs(path: Path) -> dict:
        """The whole table: original name to contract name (or None)."""
        out = {}
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id.endswith("METRIC_NAMES")
                       for t in node.targets):
                continue
            if isinstance(node.value, ast.Dict):
                for k, v in zip(node.value.keys, node.value.values):
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                        out[k.value] = v.value
        return out

    def test_the_pair_reader_finds_something(self):
        self.assertTrue(any(self.pairs(p) for p in self.adapters()))

    def test_the_adapters_were_found(self):
        self.assertGreater(len(self.adapters()), 1)

    def test_every_port_declares_a_table(self):
        for path in self.ports():
            with self.subTest(adapter=str(path.relative_to(ROOT))):
                self.assertIn("METRIC_NAMES",
                              path.read_text(encoding="utf-8"))

    def test_the_reader_finds_something(self):
        """Against an empty result the check below passes vacuously."""
        found = [n for p in self.adapters() for n in self.targets(p)]
        self.assertGreater(len(found), 3)

    def test_every_vocabulary_entry_is_used_by_some_port(self):
        """A name no port writes is a name nobody has checked against a real
        method. It reads as settled while nothing has ever produced it, and
        the first port to reach for it inherits whatever was assumed when it
        was added. Add the name and the mapping together, or not at all.

        The counters and the unavailable count are exempt: they are written
        by ports, but they are also the shape every port is entitled to use,
        so an unused one is not evidence of anything.
        """
        used = {n for p in self.adapters() for n in self.targets(p)}
        unused = sorted(set(adapterlib.METRIC_VOCABULARY) - used)
        self.assertEqual(
            unused, [],
            "in the vocabulary but written by nothing:\n"
            + "\n".join(f"  - {x}" for x in unused))

    def test_the_use_check_is_not_vacuous(self):
        """Against an empty `used` set it would fail, not pass -- but against
        a vocabulary that is somehow empty it would pass. Pin both."""
        self.assertTrue(adapterlib.METRIC_VOCABULARY)
        self.assertTrue({n for p in self.adapters() for n in self.targets(p)})

    def test_no_two_originals_across_ports_take_different_contract_names(self):
        """The same upstream name landing in two different columns.

        This is a **heuristic**, and it is written down as one. Two ports are
        independent codebases and could legitimately use `val_loss` for
        different quantities -- a pretext loss in one, a probe loss in
        another -- at which point this has to be revisited rather than
        silenced. It is here because while it holds it is cheap evidence, and
        the day it breaks is a day somebody should look.
        """
        seen: dict = {}
        clash = []
        for path in self.adapters():
            for raw, target in self.pairs(path).items():
                if target is None:
                    continue
                if seen.setdefault(raw, target) != target:
                    clash.append(f"{raw} -> {seen[raw]} and {target}")
        self.assertEqual(clash, [], "\n".join(clash))

    def test_every_mapped_name_is_in_the_vocabulary(self):
        for path in self.adapters():
            for name in self.targets(path):
                with self.subTest(adapter=path.parent.parent.name,
                                  metric=name):
                    self.assertIn(name, adapterlib.METRIC_VOCABULARY)


class TestAStageMayNotBorrowTheOtherFamilysNames(unittest.TestCase):
    """The dangerous mapping, made impossible rather than discouraged.

    Every check above is satisfied by a port that maps its pretext accuracy
    to `final_linear_probe_top1_accuracy`: the name is in the vocabulary and
    the value is a number. That single line would put eight-way patch-position
    accuracy in the same column as real classification accuracy, and the
    column would look right -- which is precisely the failure this whole
    vocabulary exists to prevent.

    A machine cannot read a number and tell which task produced it. It can
    read the **stage**, and the contract already separates them: a stage that
    trains a method's own objective is not the stage that fits a linear probe.
    So the stage decides which family of names is available, and a port cannot
    reach across by writing one word differently.

    Documenting the rule instead would not hold. This project has the counts.
    """

    def setUp(self) -> None:
        import shutil
        import tempfile
        self.out = Path(tempfile.mkdtemp(prefix="family-"))
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def ctx(self, stage: str):
        return adapterlib.Context(out=self.out, config={"seed": 1},
                                  stage=stage)

    def test_a_pretext_stage_may_not_emit_a_probe_name(self):
        with self.assertRaises(adapterlib.AdapterError) as e:
            self.ctx("step1").write_metrics(
                {"val_acc1": 12.5},
                names={"val_acc1": "final_linear_probe_top1_accuracy"})
        self.assertIn("step1", str(e.exception))

    def test_a_probe_stage_may_not_emit_a_pretext_name(self):
        with self.assertRaises(adapterlib.AdapterError) as e:
            self.ctx("linear_eval").write_metrics(
                {"acc": 12.5},
                names={"acc": "final_pretext_top1_accuracy"})
        self.assertIn("linear_eval", str(e.exception))

    def test_the_pretext_stage_may_emit_its_own(self):
        """The rule must not refuse everything."""
        self.ctx("step1").write_metrics(
            {"val_acc1": 12.5},
            names={"val_acc1": "final_pretext_top1_accuracy"})
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertIn("final_pretext_top1_accuracy", doc["metrics"])

    def test_the_probe_stage_may_emit_its_own(self):
        self.ctx("linear_eval").write_metrics(
            {"acc": 12.5},
            names={"acc": "final_linear_probe_top1_accuracy"})
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertIn("final_linear_probe_top1_accuracy", doc["metrics"])

    def test_counters_belong_to_every_stage(self):
        """A step count is neither family and both stages produce one."""
        for stage in ("step1", "linear_eval"):
            with self.subTest(stage=stage):
                self.ctx(stage).write_metrics(
                    {"n": 3}, names={"n": "steps_completed"})

    def test_an_unknown_stage_may_not_use_either_family(self):
        """Defaulting an unrecognised stage into one of the families would
        decide the question by accident."""
        with self.assertRaises(adapterlib.AdapterError) as e:
            self.ctx("something_new").write_metrics(
                {"acc": 1.0}, names={"acc": "final_pretext_top1_accuracy"})
        self.assertIn("something_new", str(e.exception))

    def test_an_unknown_stage_may_still_count_its_steps(self):
        """The rule is about the two families. A counter belongs to neither,
        so refusing it would be the rule reaching past what it is for."""
        self.ctx("something_new").write_metrics({"n": 3},
                                                names={"n": "steps_completed"})
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertEqual(doc["metrics"], {"steps_completed": 3})

    def test_every_contract_stage_is_covered(self):
        """A stage the contract defines but this table forgets would be
        refused outright, which is the listing mistake wearing a new hat."""
        self.assertEqual(set(adapterlib.STAGE_FAMILIES),
                         set(adapterlib.CONTRACT_STAGES))

    def test_knowledge_transfer_is_a_pretext_stage(self):
        """Knowledge transfer (cluster a frozen encoder's features into
        pseudo-labels, train a new network to predict them) trains a
        method's own objective, not a comparable probe -- its loss is a
        pretext number. So the stage exists in the contract and sits in the
        pretext family, and it may emit a pretext loss."""
        self.assertIn("knowledge_transfer", adapterlib.CONTRACT_STAGES)
        self.assertEqual(adapterlib.STAGE_FAMILIES["knowledge_transfer"],
                         adapterlib.PRETEXT)
        self.ctx("knowledge_transfer").write_metrics(
            {"loss": 1.0}, names={"loss": "final_pretext_loss"})
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertIn("final_pretext_loss", doc["metrics"])

    def test_a_knowledge_transfer_stage_may_not_emit_a_probe_name(self):
        """It is a pretext stage; the probe column is not its to write."""
        with self.assertRaises(adapterlib.AdapterError) as e:
            self.ctx("knowledge_transfer").write_metrics(
                {"acc": 1.0},
                names={"acc": "final_linear_probe_top1_accuracy"})
        self.assertIn("knowledge_transfer", str(e.exception))


if __name__ == "__main__":
    unittest.main()
