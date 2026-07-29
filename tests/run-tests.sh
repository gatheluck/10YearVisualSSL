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
echo "== unit tests =="
python3 -m unittest discover -s tests -v
echo
echo "all tests passed"
