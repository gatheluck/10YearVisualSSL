#!/usr/bin/env bash
# Run everything. Standard library only; no extra dependencies.
# **Always judge by the exit status.** Grepping for a success string hides
# failures.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
echo "== syntax =="
for f in bin/*.py; do
  [ -e "$f" ] || continue
  python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "$f"; echo "  ok $f"
done
for f in bin/*.sh .githooks/*; do
  [ -e "$f" ] || continue
  bash -n "$f"; echo "  ok $f"
done
echo
# Optional dependencies are reported before the run, so that a skipped test
# is never mistaken for a passing one. The machine-checked half of this lives
# in bin/resolve-config.py, which refuses a YAML config loudly when PyYAML is
# absent (tests/test_resolve_config.py).
echo "== optional dependencies =="
python3 -c 'import importlib.util as u; print("  PyYAML: " + ("present" if u.find_spec("yaml") else "absent -- YAML authoring is unavailable and its tests will be skipped"))'
echo
echo "== unit tests =="
python3 -m unittest discover -s tests -v
echo
echo "all tests passed"
