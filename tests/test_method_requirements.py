#!/usr/bin/env python3
"""A method must declare exactly the packages it imports.

This exists because the first port got it wrong. `requirements.txt` came
across from the capture unchanged, and it was the *legacy* track's list:
it named `timm`, `PyYAML`, `tensorboard` and `tqdm`, none of which the ported
code imports, and it was read by nobody.

Both directions of that are harmful:

- **A package that is imported but not declared** turns into an ImportError on
  a fresh machine, after the environment was built and believed correct
- **A package that is declared but not imported** is installed for nothing and,
  worse, misleads the next reader about what the method actually needs

So the requirement is checked against the imports rather than trusted. The
same check covers every method added later, which is the point: this is one
rule in one place, not a habit each port has to remember.

Versions are a separate matter, handled by `requirements.lock.txt`. This file
only settles *which* packages, not which versions.
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS = ROOT / "methods"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _repo_files import submodule_paths  # noqa: E402


def local_modules(method: Path) -> set[str]:
    """Modules that come from the repository rather than from an index.

    **Derived, not listed.** A hand-maintained list went stale the moment a
    method gained a second file: `evaluate_linear_official` was reported as an
    undeclared third-party package, and that false positive masked the real
    one (torchvision). Anything importable from the method directory or from
    the repository root is local by construction.

    A **pinned upstream** under `third_party/<sub>` is local too: the adapter
    that consumes it imports its packages directly (`from models.mar import
    ...`), and those are vendored code an index cannot install, not third-party
    distributions. Some are namespace packages with no `__init__.py`, so a
    submodule's top-level directories all count, however they are packaged. The
    submodule roots are read from `.gitmodules`, so this cannot go stale, and
    the same list already tells the file-scan which paths are not ours.
    """
    names = set()
    for base in (method, ROOT):
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.is_dir() and (p / "__init__.py").is_file():
                names.add(p.name)
            elif p.suffix == ".py":
                names.add(p.stem)
    for sub in submodule_paths(ROOT):
        root = ROOT / sub
        if not root.is_dir():
            continue
        # A submodule holds its packages either at its root
        # (third_party/dinov2/dinov2) or one level down in a monorepo subproject
        # (third_party/ml-aim/aim-v1/aim, a PEP 420 namespace package). Both are
        # upstream code imported through PYTHONPATH, so add directory names at both
        # depths. Deeper than that is a package's own internals (aim/v1, aim/v1/
        # torch), not a top-level import, so the descent stops at depth two -- it
        # never reaches a name like `torch` that must stay a real dependency.
        for p in root.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                names.add(p.name)
                for q in p.iterdir():
                    if q.is_dir() and not q.name.startswith("."):
                        names.add(q.name)
                    elif q.suffix == ".py":
                        names.add(q.stem)
            elif p.suffix == ".py":
                names.add(p.stem)
    return names

# Import name -> distribution name, where they differ. Anything imported and
# not listed here is assumed to install under its own name; the completeness
# check below fails rather than let an unknown one slide.
DISTRIBUTION = {"PIL": "Pillow", "yaml": "PyYAML", "cv2": "opencv-python",
                "sklearn": "scikit-learn", "skimage": "scikit-image",
                "faiss": "faiss-gpu"}

# A dotted import can require a distribution that its top-level name does not
# mention. `from torch.utils.tensorboard import SummaryWriter` needs
# `tensorboard` installed, and a scan of top-level names sees only `torch` --
# so the declaration looked unused and the requirement looked absent. Found
# when the second method came across.
IMPLIED = {"torch.utils.tensorboard": "tensorboard"}


def method_dirs() -> list[Path]:
    if not METHODS.is_dir():
        return []
    return sorted(p for p in METHODS.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def imported_modules(method: Path) -> set[str]:
    """Third-party distributions the method needs, however it imports them."""
    found: set[str] = set()
    for py in sorted(method.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            dotted = []
            if isinstance(node, ast.Import):
                dotted = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                dotted = [node.module or ""]
            for name in dotted:
                found.add(name.split(".")[0])
                for prefix, dist in IMPLIED.items():
                    if name == prefix or name.startswith(prefix + "."):
                        found.add(dist)
    local = local_modules(method)
    return {m for m in found
            if m and m not in sys.stdlib_module_names and m not in local}


def _canon(name: str) -> str:
    """Distribution names compare case- and separator-insensitively."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def requirement_lines(path: Path) -> list[str]:
    """Requirement lines, with hash continuations folded back in.

    A hashed lock spreads one requirement over several lines with trailing
    backslashes. Reading it line by line makes a hash look like a requirement
    that is not pinned -- which is exactly how one of these checks first
    failed. The rule lives here once so both callers agree.
    """
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.append(line)
    return out


def declared_packages(req: Path) -> set[str]:
    """Distribution names in a requirements file, without version specifiers."""
    return {re.split(r"[<>=!~\[]", line, 1)[0].strip().lower()
            for line in requirement_lines(req)}


def gpu_only_packages(req: Path) -> set[str]:
    """Distribution names a method marks `# gpu-only` in requirements.txt.

    A gpu-only package has no cross-platform wheel and so cannot live in the CPU
    lock -- `faiss-gpu`, whose only wheel is linux-x86_64, is the first. The
    marker exempts such a package from the CPU lock **only**: it must still be
    pinned in the CUDA lock (`lock_coverage_gaps` enforces that), so the marker
    narrows where a package is locked, it does not let one go unlocked.

    Parsed from the raw lines (not `requirement_lines`, which strips comments):
    a line whose comment contains the token `gpu-only`.
    """
    out = set()
    for raw in req.read_text(encoding="utf-8").splitlines():
        if "#" not in raw:
            continue
        head, comment = raw.split("#", 1)
        if "gpu-only" not in comment:
            continue
        name = re.split(r"[<>=!~\[]", head, 1)[0].strip().lower()
        if name:
            out.add(name)
    return out


def lock_coverage_gaps(declared, gpu_only, cpu_locked, cuda_locked):
    """Declared packages that are not properly locked, as a pure function so both
    the real check and its controls read the same logic.

    Non-gpu-only packages must be in the CPU lock; gpu-only packages are exempt
    from the CPU lock but must be in the CUDA lock. Returns
    ``(missing_from_cpu, gpu_only_missing_from_cuda)`` -- both empty means every
    declared package is locked somewhere it can actually install.
    """
    declared, gpu_only = set(declared), set(gpu_only)
    missing_cpu = (declared - gpu_only) - set(cpu_locked)
    missing_cuda = gpu_only - set(cuda_locked)
    return missing_cpu, missing_cuda


class TestEveryMethodDeclaresWhatItImports(unittest.TestCase):
    def test_there_is_a_method_to_check(self):
        """With no methods this file would pass without looking at anything."""
        self.assertTrue(method_dirs(), "no methods found to check")

    def test_each_method_has_a_requirements_file(self):
        for m in method_dirs():
            with self.subTest(method=m.name):
                if not imported_modules(m):
                    continue          # a method needing nothing needs no file
                self.assertTrue((m / "requirements.txt").is_file(),
                                f"{m.name} imports packages but declares none")

    def test_nothing_imported_is_undeclared(self):
        for m in method_dirs():
            req = m / "requirements.txt"
            if not req.is_file():
                continue
            declared = declared_packages(req)
            with self.subTest(method=m.name):
                for mod in sorted(imported_modules(m)):
                    dist = DISTRIBUTION.get(mod, mod).lower()
                    self.assertIn(
                        dist, declared,
                        f"{m.name} imports {mod} but {dist} is not in "
                        "requirements.txt; it would fail on a fresh machine")

    def test_nothing_declared_is_unimported(self):
        """The failure that prompted this: four packages declared, none used."""
        for m in method_dirs():
            req = m / "requirements.txt"
            if not req.is_file():
                continue
            needed = {DISTRIBUTION.get(x, x).lower()
                      for x in imported_modules(m)}
            with self.subTest(method=m.name):
                extra = declared_packages(req) - needed
                self.assertEqual(
                    extra, set(),
                    f"{m.name} declares {sorted(extra)} but imports none of "
                    "them; the list misleads whoever builds the environment")

    def test_the_scan_finds_real_imports(self):
        """Against an empty scan both checks above would pass vacuously."""
        found = set()
        for m in method_dirs():
            found |= imported_modules(m)
        self.assertTrue(found, "no third-party imports were found at all")

    def test_the_distribution_table_is_used_where_it_must_be(self):
        """Guard the mapping itself: PIL does not install as `pil`."""
        self.assertEqual(DISTRIBUTION["PIL"], "Pillow")

    def test_a_methods_own_modules_are_recognised_as_local(self):
        """The false positive that masked a real one. A module sitting in the
        method directory is not something pip can install."""
        for m in method_dirs():
            local = local_modules(m)
            for p in m.glob("*.py"):
                with self.subTest(method=m.name, module=p.stem):
                    self.assertIn(p.stem, local)
            self.assertIn("adapterlib", local,
                          "the repository's own package is not recognised")

    def test_a_real_third_party_import_is_still_seen(self):
        """Widening what counts as local must not swallow everything."""
        found = set()
        for m in method_dirs():
            found |= imported_modules(m)
        self.assertIn("torch", found)

    def test_a_pinned_upstream_package_counts_as_local(self):
        """A method importing `from models.mar import ...` must not have
        `models` demanded in requirements.txt: it is vendored upstream code
        under third_party/, not a distribution an index can install. Namespace
        packages (no __init__.py) count too, which is how the upstream ships
        `models` and `util` -- so the check must not depend on __init__.py."""
        subs = submodule_paths(ROOT)
        if not subs:
            self.skipTest("no submodule to check")
        checked_namespace = False
        checked_nested = False
        for sub in sorted(subs):
            root = ROOT / sub
            if not root.is_dir():
                continue
            for p in sorted(root.iterdir()):
                if not (p.is_dir() and not p.name.startswith(".")):
                    continue
                for m in method_dirs():
                    self.assertIn(
                        p.name, local_modules(m),
                        f"{p.name} under {sub} is upstream code, not a package "
                        "pip installs, but it is not recognised as local")
                if not (p / "__init__.py").is_file():
                    checked_namespace = True
                # A monorepo submodule holds its importable packages one level
                # down (third_party/ml-aim/aim-v1/aim); those count as local too.
                for q in sorted(p.iterdir()):
                    if not (q.is_dir() and not q.name.startswith(".")):
                        continue
                    checked_nested = True
                    for m in method_dirs():
                        self.assertIn(
                            q.name, local_modules(m),
                            f"{q.name} under {sub}/{p.name} is upstream code, "
                            "not a package pip installs, but it is not "
                            "recognised as local")
        self.assertTrue(
            checked_namespace,
            "no namespace package (no __init__.py) among the submodules, so "
            "the branch that must not require one was never exercised")
        self.assertTrue(
            checked_nested,
            "no submodule holds a package one level down, so the nested-package "
            "branch of local_modules was never exercised")


class TestTheOptionalToolingDependency(unittest.TestCase):
    """PyYAML belongs to the tooling, not to any method.

    Found by running the README's own commands in order: the environment its
    first step builds could not carry out its second, because
    `resolve-config.py` needs PyYAML to read a YAML authoring config and no
    method declares it -- correctly, since no method imports it.
    """

    TOOLS = ROOT / "requirements-tools.txt"
    LOCK = ROOT / "requirements-tools.lock.txt"

    def test_the_tooling_requirement_is_declared_and_locked(self):
        self.assertTrue(self.TOOLS.is_file())
        self.assertTrue(self.LOCK.is_file())

    def test_it_is_pyyaml_and_nothing_else(self):
        """The core stays standard-library only. This is the one optional
        extra, and it buys exactly one thing: YAML authoring."""
        self.assertEqual(declared_packages(self.TOOLS), {"pyyaml"})

    def test_the_tooling_lock_is_exact_and_hashed(self):
        lines = requirement_lines(self.LOCK)
        self.assertTrue(lines, "the tooling lock is empty")
        for line in lines:
            self.assertIn("==", line, f"{line!r} is not pinned")
            self.assertIn("--hash=sha256:", line, f"{line!r} has no hash")

    def test_a_method_declares_it_only_if_it_imports_it(self):
        """This used to say *no* method may declare it, and that was true
        while none imported it.

        `methods/02_vae` imports `yaml` directly in its trainer, so the old
        claim is now false -- measurement, not preference. The rule that
        survives is the derived one: declaring it is allowed exactly when it
        is imported, which is what the two-directional check already enforces
        for every package. Stated here so the reasoning is not lost.
        """
        for m in method_dirs():
            req = m / "requirements.txt"
            if not req.is_file():
                continue
            declares = "pyyaml" in declared_packages(req)
            imports = "yaml" in {x.lower() for x in imported_modules(m)}
            with self.subTest(method=m.name):
                self.assertEqual(
                    declares, imports,
                    f"{m.name}: declares PyYAML={declares} but "
                    f"imports yaml={imports}")


class TestTheInterpreterIsPinnedToo(unittest.TestCase):
    """A lock over packages says nothing about the interpreter running them.

    Nothing in this repository constrained the Python version: no
    `.python-version`, no `requires-python`, nothing. The same pinned torch
    behaves differently on a different interpreter, and the lock would have
    looked satisfied either way.
    """

    VERSION_FILE = ROOT / ".python-version"

    def test_the_interpreter_version_is_declared(self):
        self.assertTrue(self.VERSION_FILE.is_file(),
                        "no .python-version: the interpreter is unconstrained")

    def test_it_is_an_exact_version(self):
        text = self.VERSION_FILE.read_text(encoding="utf-8").strip()
        self.assertRegex(text, r"^\d+\.\d+\.\d+$",
                         f"{text!r} is not an exact version")

    def test_the_declared_minor_matches_what_the_locks_were_built_for(self):
        """A lock holds wheels built for one ABI. Declaring 3.11 while the
        wheels are cp312 would install nothing."""
        declared = self.VERSION_FILE.read_text(encoding="utf-8").strip()
        major_minor = ".".join(declared.split(".")[:2])
        for m in method_dirs():
            lock = m / "requirements.lock.txt"
            if not lock.is_file():
                continue
            text = lock.read_text(encoding="utf-8")
            if "cp3" in text:
                with self.subTest(method=m.name):
                    tag = "cp" + major_minor.replace(".", "")
                    self.assertIn(tag, text,
                                  f"the lock holds no {tag} wheels")


class TestTheLockIsAClosure(unittest.TestCase):
    """A lock that names only the direct requirements is not a lock.

    Measured before this: the file pinned 3 packages while 12 were installed.
    The nine that floated included torch's own dependencies.
    """

    def test_the_lock_names_more_than_the_direct_requirements(self):
        for m in method_dirs():
            req, lock = m / "requirements.txt", m / "requirements.lock.txt"
            if not (req.is_file() and lock.is_file()):
                continue
            with self.subTest(method=m.name):
                direct, locked = declared_packages(req), declared_packages(lock)
                self.assertGreater(
                    len(locked), len(direct),
                    f"{m.name}: the lock has {len(locked)} packages and the "
                    f"direct requirements have {len(direct)}; transitive "
                    "dependencies are not pinned")

    def test_the_lock_contains_the_dependencies_of_what_it_locks(self):
        """Closure, checked against the packages' own metadata.

        Counting entries was not enough: deleting one transitive dependency
        left the count above the direct requirements and nothing failed. This
        reads each locked package's `Requires-Dist` and demands the result be
        present too.

        Only unconditional requirements are followed. Anything behind an
        environment marker or an extra may legitimately be absent, and
        evaluating markers needs a package we do not have.
        """
        import importlib.metadata as md
        for m in method_dirs():
            lock = m / "requirements.lock.txt"
            if not lock.is_file():
                continue
            locked = {_canon(x) for x in declared_packages(lock)}
            checked = 0
            for name in sorted(locked):
                try:
                    reqs = md.requires(name) or []
                except md.PackageNotFoundError:
                    continue          # not installed here; nothing to read
                checked += 1
                for raw in reqs:
                    if ";" in raw:    # marker or extra: may not apply
                        continue
                    dep = _canon(re.split(r"[<>=!~\[ (]", raw, 1)[0])
                    with self.subTest(method=m.name, package=name, needs=dep):
                        self.assertIn(
                            dep, locked,
                            f"{name} requires {dep}, which the lock omits; "
                            "the lock is not a closure")
            if checked == 0:
                self.skipTest(f"{m.name}: none of its packages are installed "
                              "here, so the closure cannot be read")

    def test_every_locked_package_carries_a_hash(self):
        """Without hashes a version can be replaced under the same name."""
        for m in method_dirs():
            lock = m / "requirements.lock.txt"
            if not lock.is_file():
                continue
            for line in requirement_lines(lock):
                with self.subTest(method=m.name, package=line.split("==")[0]):
                    self.assertIn("--hash=sha256:", line,
                                  f"{line.split('==')[0]} has no hash")


# The platforms a lock is expected to be installable on, as substrings that
# must all appear in one wheel filename. Declared here so that "we support
# linux arm64" cannot quietly stop being true.
#
# Found by building: Docker on Apple silicon builds linux/arm64, whose wheels
# are `aarch64`. The lock held x86_64 and macOS arm64 only, so pip refused --
# correctly, and loudly, but the platform should have been covered.
# Each target is satisfied by a wheel matching any one of its patterns, and a
# pattern is satisfied when every substring in it appears in the filename.
#
# `universal2` was added after grpcio and protobuf ship macOS wheels built for
# both architectures at once: `grpcio-...-macosx_11_0_universal2.whl` does
# support arm64, and reading it as a gap would have been the checker being
# wrong about the world rather than the lock being incomplete.
TARGET_PLATFORMS = {
    "linux x86_64": [("x86_64",)],
    "linux aarch64": [("aarch64",)],
    "macOS arm64": [("macosx", "arm64"), ("macosx", "universal2")],
}
UNIVERSAL = "py3-none-any"


def all_locks() -> list[Path]:
    """Every lock in the repository, not only the methods'.

    The first version of the coverage check looked at method locks alone. The
    container build then failed on the *tooling* lock, which had the same gap
    -- a rule applied to some of the things it governs is a rule with a hole
    in it.
    """
    found = [m / "requirements.lock.txt" for m in method_dirs()]
    found.append(ROOT / "requirements-tools.lock.txt")
    return [p for p in found if p.is_file()]


def wheels_per_package(lock: Path) -> dict:
    """Package name -> the wheel filenames recorded in its comments.

    The generator writes one `# <filename>` line per distinct wheel directly
    above the requirement it belongs to.
    """
    out, pending = {}, []
    for raw in lock.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            name = line[1:].strip()
            if name.endswith((".whl", ".tar.gz")):
                pending.append(name)
        elif "==" in line and not line.startswith("-"):
            out[_canon(line.split("==")[0])] = pending
            pending = []
        elif not line:
            continue
    return out


class TestTheLockCoversEveryTargetPlatform(unittest.TestCase):
    """A lock that omits a platform is not wrong, but it is not usable there.

    pip refuses an unlisted wheel rather than installing something unverified,
    which is the behaviour we want -- but the failure arrives at build time,
    far from the file that caused it. This brings it forward.
    """

    def test_there_is_more_than_one_lock_to_check(self):
        """Guards the widening: checking a single file would pass vacuously
        if the others quietly stopped being found."""
        self.assertGreater(len(all_locks()), 1)

    def test_every_package_has_a_wheel_for_every_target(self):
        for lock in all_locks():
            name = str(lock.relative_to(ROOT))
            per_package = wheels_per_package(lock)
            self.assertTrue(per_package, f"{name}: no wheels recorded")
            for pkg, wheels in sorted(per_package.items()):
                if any(UNIVERSAL in w for w in wheels):
                    continue          # pure python: one wheel serves all
                for label, patterns in sorted(TARGET_PLATFORMS.items()):
                    with self.subTest(lock=name, package=pkg,
                                      platform=label):
                        self.assertTrue(
                            any(all(n in w for n in pat)
                                for w in wheels for pat in patterns),
                            f"{pkg} has no wheel for {label}; installing "
                            "there would be refused for want of a hash")

    def test_the_number_of_hashes_matches_the_number_of_wheels(self):
        """The comments and the hashes are written together; if they drift,
        the check above is reading a filename that no hash belongs to."""
        for lock in all_locks():
            per_package = wheels_per_package(lock)
            for line in requirement_lines(lock):
                pkg = _canon(line.split("==")[0])
                with self.subTest(lock=str(lock.relative_to(ROOT)),
                                  package=pkg):
                    self.assertEqual(line.count("--hash=sha256:"),
                                     len(per_package.get(pkg, [])),
                                     "hashes and recorded wheels disagree")

    def test_the_declared_targets_are_the_ones_we_claim_to_support(self):
        """Shrinking this silently would make the check above pass by
        covering less."""
        self.assertEqual(set(TARGET_PLATFORMS),
                         {"linux x86_64", "linux aarch64", "macOS arm64"})


class TestVersionsArePinned(unittest.TestCase):
    """`torch>=2.0.0` is not a reproducible environment.

    A floor says what will import, not what ran. A run recorded against a
    floor cannot be rebuilt: any version above it satisfies the file.
    """

    def test_each_method_has_a_lock_file(self):
        for m in method_dirs():
            if not (m / "requirements.txt").is_file():
                continue
            with self.subTest(method=m.name):
                self.assertTrue(
                    (m / "requirements.lock.txt").is_file(),
                    f"{m.name} has no requirements.lock.txt, so the "
                    "environment it ran in cannot be rebuilt")

    def test_every_line_in_a_lock_file_is_an_exact_version(self):
        for m in method_dirs():
            lock = m / "requirements.lock.txt"
            if not lock.is_file():
                continue
            lines = requirement_lines(lock)
            self.assertTrue(lines, f"{m.name}: the lock is empty")
            for line in lines:
                with self.subTest(method=m.name, line=line.split()[0]):
                    self.assertIn(
                        "==", line,
                        f"{line!r} is not pinned; a lock file states exactly "
                        "one version")

    def test_the_lock_covers_everything_declared(self):
        for m in method_dirs():
            req = m / "requirements.txt"
            cpu = m / "requirements.lock.txt"
            cuda = m / "requirements.lock.cu130.txt"
            if not (req.is_file() and cpu.is_file()):
                continue
            with self.subTest(method=m.name):
                gpu_only = gpu_only_packages(req)
                if gpu_only:
                    self.assertTrue(
                        cuda.is_file(),
                        f"{m.name}: has gpu-only packages ({sorted(gpu_only)}) "
                        "but no CUDA lock to hold them")
                cuda_locked = declared_packages(cuda) if cuda.is_file() else set()
                missing_cpu, missing_cuda = lock_coverage_gaps(
                    declared_packages(req), gpu_only,
                    declared_packages(cpu), cuda_locked)
                self.assertEqual(missing_cpu, set(),
                                 f"{m.name}: declared but not in the CPU lock")
                self.assertEqual(missing_cuda, set(),
                                 f"{m.name}: marked gpu-only but not pinned in "
                                 "the CUDA lock")


class TestGpuOnlyPackages(unittest.TestCase):
    """The gpu-only marker exempts a package from the CPU lock only -- it must
    still be pinned in the CUDA lock. Proven here with positive and negative
    controls so the exemption cannot quietly let a package go unlocked."""

    def test_the_marker_is_parsed_from_requirements(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            req = Path(d) / "requirements.txt"
            req.write_text(
                "torch>=2.0.0\n"
                "numpy>=1.24.0  # a normal comment, not a marker\n"
                "faiss-gpu>=1.14.3  # gpu-only\n", encoding="utf-8")
            self.assertEqual(gpu_only_packages(req), {"faiss-gpu"})
            self.assertIn("faiss-gpu", declared_packages(req))
            self.assertIn("numpy", declared_packages(req))

    def test_a_gpu_only_package_may_skip_the_cpu_lock(self):
        # faiss-gpu is gpu-only and pinned in the CUDA lock: no gaps.
        missing_cpu, missing_cuda = lock_coverage_gaps(
            declared={"torch", "faiss-gpu"}, gpu_only={"faiss-gpu"},
            cpu_locked={"torch"}, cuda_locked={"torch", "faiss-gpu"})
        self.assertEqual((missing_cpu, missing_cuda), (set(), set()))

    def test_a_gpu_only_package_missing_from_the_cuda_lock_is_flagged(self):
        # Exempt from the CPU lock, but not from being locked at all.
        _, missing_cuda = lock_coverage_gaps(
            declared={"torch", "faiss-gpu"}, gpu_only={"faiss-gpu"},
            cpu_locked={"torch"}, cuda_locked={"torch"})
        self.assertEqual(missing_cuda, {"faiss-gpu"})

    def test_a_normal_package_missing_from_the_cpu_lock_is_still_flagged(self):
        # The marker changes nothing for unmarked packages.
        missing_cpu, _ = lock_coverage_gaps(
            declared={"torch", "numpy"}, gpu_only=set(),
            cpu_locked={"torch"}, cuda_locked={"torch", "numpy"})
        self.assertEqual(missing_cpu, {"numpy"})


if __name__ == "__main__":
    unittest.main()
