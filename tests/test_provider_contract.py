#!/usr/bin/env python3
"""Fleet-wide guard: every linear-probe method ships a conformant feature provider.

The BASIC5_FAIR_v1 feature dump (`bin/extract-features.py`) is meant to run
*every* method's frozen encoder over the ImageNet val split in one sweep. That
promise only holds if every method that can be linearly probed actually ships a
`feature_provider.py` exposing the extractor the driver calls -- otherwise a
method silently drops out of the dump and "all methods" quietly becomes "all but
one". Each provider is proven end to end in its own `test_method_*.py`; what was
missing was a single check that the *set* is complete and that none has drifted
off the contract. This file is that check.

It is **discover, never list** (CLAUDE.md): it names no method. The methods that
must ship a provider are discovered by a machine-readable marker -- a method
carries a linear-probe evaluator (`evaluate_linear*.py`) -- so a newly ported
method is covered the moment it lands, with no edit here. The two directories
that ship no provider by design fall out of that discovery on their own: the
`_reference` template and the sole pretrain-only method both carry no
`evaluate_linear*.py`, so neither is ever expected to.

The contract itself -- the keyword arguments the driver passes to
`extract_val_features` -- is not restated here either; it is read from the
driver's own call site, so the guard tracks the driver and cannot fall out of
step with it. The provider check is a **faithful static binding**: it asks
whether `extract_val_features` could be *called* exactly the way the driver calls
it (all six as keywords), which catches a renamed function, a dropped parameter,
a parameter turned positional-only, or a new required parameter -- not merely
whether some substring is present. It is pure AST, so it needs no torch, no GPU,
and no cross-method import (providers are run in subprocess isolation precisely
because two method stacks must not share an interpreter).

The detector carries a positive and a negative control (a detector that decides
what passes needs both), and the guard's non-vacuity is proven by
`mutations/provider-contract.json` (break one real provider's signature -> the
guard fails).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS = ROOT / "methods"
TOOL = ROOT / "bin" / "extract-features.py"
_TOOL_MOD = "extract_features_tool"

# The extractor the driver calls on every provider.
EXTRACTOR = "extract_val_features"
# The marker that makes a method a linear-probe method (and so require a
# provider): it ships a linear-probe evaluator. Matched as a glob on the
# method's own directory, never as a hard-coded method name.
EVALUATOR_GLOB = "evaluate_linear*.py"


def _load_tool():
    """Load bin/extract-features.py by path (its name is not importable).

    The driver owns provider discovery; this guard reuses that one
    implementation rather than growing a second copy of "which methods ship a
    provider" (two copies of that rule is a recorded defect here).
    """
    if _TOOL_MOD in sys.modules:
        return sys.modules[_TOOL_MOD]
    spec = importlib.util.spec_from_file_location(_TOOL_MOD, TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_TOOL_MOD] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[_TOOL_MOD]
        raise
    return mod


# -- discovery: which methods must ship a provider ---------------------------

def linear_probe_methods(methods_dir):
    """Method directories that carry a linear-probe evaluator.

    These are exactly the methods whose frozen encoder the feature dump is
    meant to cover, so these are the methods that must ship a provider. A
    directory that ships no `evaluate_linear*.py` -- the template, a
    pretrain-only method -- is not one of them and is not expected to.
    """
    out = []
    for child in sorted(Path(methods_dir).iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if any(child.glob(EVALUATOR_GLOB)):
            out.append(child.name)
    return out


# -- the contract: read from the driver's own call site ----------------------

def required_contract_params(driver_src):
    """The keyword names the driver passes to `extract_val_features`.

    Read from the driver's call site so the contract is defined in exactly one
    place. Returns the union over every such call (there should be one).
    """
    required = set()
    for node in ast.walk(ast.parse(driver_src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == EXTRACTOR):
            required |= {kw.arg for kw in node.keywords if kw.arg is not None}
    return required


# -- the provider check: a faithful static binding ---------------------------

def _extractor_def(provider_src):
    """The `extract_val_features` FunctionDef in a provider, or None."""
    for node in ast.walk(ast.parse(provider_src)):
        if isinstance(node, ast.FunctionDef) and node.name == EXTRACTOR:
            return node
    return None


def _call_binds(fn, required):
    """Whether `fn` could be called with exactly `required` as keyword args.

    A faithful model of Python's binding for an all-keyword call: a
    positional-only parameter cannot be filled by keyword; a keyword the
    function does not accept needs `**kwargs`; and no parameter without a
    default may be left unprovided. This is what "the driver can call it"
    means -- not that a name happens to appear in the file.
    """
    a = fn.args
    required = set(required)
    posonly = [x.arg for x in a.posonlyargs]
    pos = [x.arg for x in a.args]
    kwonly = [x.arg for x in a.kwonlyargs]
    has_kwargs = a.kwarg is not None

    combined = posonly + pos
    ndef = len(a.defaults)
    pos_have_default = set(combined[len(combined) - ndef:]) if ndef else set()
    kw_have_default = {name for name, d in zip(kwonly, a.kw_defaults)
                       if d is not None}

    fillable = set(pos) | set(kwonly)          # accept a keyword by this name
    # A keyword the function does not accept (and no **kwargs to absorb it).
    if not has_kwargs and not required.issubset(fillable):
        return False
    # A required keyword naming a positional-only parameter cannot bind there.
    if set(posonly) & required:
        return False
    # No parameter without a default may be left unprovided.
    for name in posonly:                       # never fillable by keyword
        if name not in pos_have_default:
            return False
    for name in pos:
        if name not in required and name not in pos_have_default:
            return False
    for name in kwonly:
        if name not in required and name not in kw_have_default:
            return False
    return True


def provider_conforms(provider_src, required):
    """Whether a provider exposes an `extract_val_features` the driver can call
    with exactly `required` as keywords."""
    fn = _extractor_def(provider_src)
    if fn is None:
        return False
    return _call_binds(fn, required)


# -- the guard ---------------------------------------------------------------

class TestEveryLinearProbeMethodShipsAConformantProvider(unittest.TestCase):
    def setUp(self):
        self.tool = _load_tool()
        self.required = required_contract_params(TOOL.read_text())
        self.probe_methods = linear_probe_methods(METHODS)
        # method -> provider path, from the driver's own discovery.
        self.providers = self.tool.discover_providers(METHODS)

    def test_discovery_is_non_vacuous(self):
        """With one method (or none) this guard could not fail."""
        self.assertGreater(len(self.probe_methods), 1)

    def test_the_contract_is_read_from_the_driver(self):
        """The required keywords come from the driver's call site, and there
        are some -- an empty set would make the binding check vacuous."""
        self.assertGreater(len(self.required), 0)

    def test_every_linear_probe_method_ships_a_provider(self):
        """No linear-probe method may silently drop out of the sweep."""
        missing = [m for m in self.probe_methods if m not in self.providers]
        self.assertEqual(
            missing, [],
            "these methods carry a linear-probe evaluator but ship no "
            "feature_provider.py, so the feature dump would silently omit "
            f"them: {missing}")

    def test_every_providers_extractor_matches_the_contract(self):
        """Every discovered provider exposes an extractor the driver can call
        with exactly the keywords it passes."""
        broken = []
        for method, path in self.providers.items():
            if not provider_conforms(Path(path).read_text(), self.required):
                broken.append(method)
        self.assertEqual(
            broken, [],
            "these providers do not expose an "
            f"{EXTRACTOR}(*, {', '.join(sorted(self.required))}) the driver "
            f"can call: {broken}")


class TestTheContractDetectorControls(unittest.TestCase):
    """The detector that decides "this provider conforms" needs a positive and
    a negative control (CLAUDE.md)."""

    REQUIRED = {"encoder_path", "data_root", "split", "device",
                "batch_size", "num_workers"}

    _CONFORMING = (
        "def extract_val_features(*, encoder_path, data_root, split, device,\n"
        "                         batch_size, num_workers):\n"
        "    return None, None, {}\n")

    _KWARGS_WILDCARD = (
        "def extract_val_features(**kw):\n"
        "    return None, None, {}\n")

    _MISSING_ONE_PARAM = (
        "def extract_val_features(*, encoder_path, data_root, device,\n"
        "                         batch_size, num_workers):\n"   # no `split`
        "    return None, None, {}\n")

    _EXTRA_REQUIRED_PARAM = (
        "def extract_val_features(*, encoder_path, data_root, split, device,\n"
        "                         batch_size, num_workers, extra):\n"
        "    return None, None, {}\n")

    _POSITIONAL_ONLY = (
        "def extract_val_features(split, /, encoder_path, data_root, device,\n"
        "                         batch_size, num_workers):\n"
        "    return None, None, {}\n")

    _NO_EXTRACTOR = "def something_else():\n    return 1\n"

    def test_positive_a_conforming_provider_is_accepted(self):
        self.assertTrue(provider_conforms(self._CONFORMING, self.REQUIRED))

    def test_positive_a_kwargs_wildcard_is_callable(self):
        # `**kw` absorbs every keyword, so the driver can call it.
        self.assertTrue(provider_conforms(self._KWARGS_WILDCARD, self.REQUIRED))

    def test_negative_a_missing_parameter_is_rejected(self):
        # Drop `split`: the driver's call would raise, so it must not pass.
        self.assertFalse(
            provider_conforms(self._MISSING_ONE_PARAM, self.REQUIRED))

    def test_negative_an_extra_required_parameter_is_rejected(self):
        # A required parameter the driver never provides -> the call is missing
        # an argument.
        self.assertFalse(
            provider_conforms(self._EXTRA_REQUIRED_PARAM, self.REQUIRED))

    def test_negative_a_positional_only_required_param_is_rejected(self):
        # `split` passed as a keyword cannot bind a positional-only parameter.
        self.assertFalse(
            provider_conforms(self._POSITIONAL_ONLY, self.REQUIRED))

    def test_negative_no_extractor_is_rejected(self):
        self.assertFalse(provider_conforms(self._NO_EXTRACTOR, self.REQUIRED))


class TestTheDiscoveryMarkerControls(unittest.TestCase):
    """The marker that decides "this method must ship a provider" needs a
    positive and a negative control. Synthetic directories with invented names
    are used, so this shared file names no real method."""

    def _method_dir(self, root, name, *, evaluator):
        d = Path(root) / name
        (d / "configs").mkdir(parents=True, exist_ok=True)
        if evaluator:
            (d / "evaluate_linear_x.py").write_text("# probe\n",
                                                    encoding="utf-8")
        return d

    def test_positive_a_method_with_an_evaluator_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._method_dir(tmp, "alpha", evaluator=True)
            self.assertIn("alpha", linear_probe_methods(tmp))

    def test_negative_a_method_without_an_evaluator_is_not_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._method_dir(tmp, "beta", evaluator=False)
            self.assertNotIn("beta", linear_probe_methods(tmp))


if __name__ == "__main__":
    unittest.main()
