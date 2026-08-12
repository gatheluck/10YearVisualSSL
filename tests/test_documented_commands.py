"""Commands printed in the docs must actually run.

Two widespread doc bugs shipped because **no test ever looked at the commands
written in the method READMEs** -- they were untested prose, hand-copied per
method, and drifted from the tools' real CLIs:

- `fetch-weights.py --section ...` (the flag is `--artifact`), and
- `resolve-config.py <path> --set ... > file` -- a *positional* config and a
  stdout redirect, but the tool requires `--config <path>` and writes to `--out`.

Both fail the instant you paste them. This is exactly the CLAUDE.md rule "a
policy in a document does not hold -- make it machinery," applied to documented
commands. This test reads each tool's real argument parser and checks every
`bin/<tool>.py` invocation in the docs against it: no unknown flag, and every
required option present. **Discover, never list:** tools and their flags are read
from `bin/`, docs are globbed, and no method or tool name is hard-coded.
"""

from __future__ import annotations

import ast
import glob
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
FLAG = re.compile(r"--[a-z][a-z0-9-]*")

# Runtime "at least one of" requirements a tool enforces in main() rather than
# through argparse(required=True). Read from the tool's own refusal; kept here
# because argparse cannot express it. Each entry: tool -> list of flag-sets, one
# of each set must be present.
ONE_OF_REQUIRED = {
    "resolve-config.py": [{"--out", "--print-hash"}],
}


def tool_argspec(tool: Path) -> "tuple[set[str], set[str]]":
    """(valid long flags, required long flags) read from a tool's argparse."""
    tree = ast.parse(tool.read_text(encoding="utf-8"))
    valid, required = {"--help"}, set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        longs = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)
                 and a.value.startswith("--")]
        valid.update(longs)
        is_required = any(
            kw.arg == "required" and isinstance(kw.value, ast.Constant)
            and kw.value.value is True for kw in node.keywords)
        if is_required and longs:
            required.add(longs[0])
    return valid, required


def _tools() -> dict[str, "tuple[set[str], set[str]]"]:
    return {Path(t).name: tool_argspec(Path(t))
            for t in glob.glob(str(BIN / "*.py"))}


def _commands_in(text: str) -> list[tuple[str, str]]:
    """(tool_filename, whole-command-string) for each bin tool invoked in an
    indented code block. Command lines are joined across `\\` continuations, so a
    wrapped command is validated whole. Inline prose mentions (not indented code)
    are skipped -- they are not commands."""
    out: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        indented = line[:1] in (" ", "\t") and line.strip() != ""
        # A shell comment inside a code block is not a command to validate.
        if line.strip().startswith("#"):
            i += 1
            continue
        m = re.search(r"bin/([a-z-]+\.py)", line)
        # Only an indented code line that *starts* the invocation counts as a
        # command (the tool appears right after an optional `python`/`python3`).
        starts = re.search(r"(^|\s)(python3?\s+)?bin/[a-z-]+\.py", line)
        if indented and m and starts:
            parts = [line]
            while parts[-1].rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                parts.append(lines[i])
            out.append((m.group(1), " ".join(p.rstrip(" \\") for p in parts)))
        i += 1
    return out


def _doc_files() -> list[Path]:
    pats = ["methods/*/README.md", "docs/*.md", "README.md"]
    return [Path(x) for p in pats for x in glob.glob(str(ROOT / p))]


def command_problems(text: str, tools: dict) -> list[str]:
    problems = []
    for tool, cmd in _commands_in(text):
        if tool not in tools:
            continue
        valid, required = tools[tool]
        used = set(FLAG.findall(cmd))
        for f in sorted(used - valid):
            problems.append(f"{tool}: unknown flag {f}")
        for f in sorted(required - used):
            problems.append(f"{tool}: missing required {f}")
        for group in ONE_OF_REQUIRED.get(tool, []):
            if not (group & used):
                problems.append(f"{tool}: needs one of {sorted(group)}")
    return problems


class TestDocumentedCommandsAreWellFormed(unittest.TestCase):
    def test_the_tools_and_docs_were_found(self):
        self.assertIn("resolve-config.py", _tools())
        self.assertGreater(len(_doc_files()), 30, "docs were not found")

    def test_every_documented_command_matches_its_tool_cli(self):
        tools = _tools()
        offenders = {}
        for f in _doc_files():
            probs = command_problems(f.read_text(encoding="utf-8"), tools)
            if probs:
                offenders[str(f.relative_to(ROOT))] = sorted(set(probs))
        self.assertEqual(
            offenders, {},
            "documented commands do not match their tool's CLI (unknown flag, "
            f"or a required option missing): {offenders}")


class TestTheDetectorFires(unittest.TestCase):
    """Positive and negative controls: a guard that cannot fail is not a guard."""

    def setUp(self):
        self.tools = _tools()

    def test_a_positional_config_is_flagged(self):
        bad = "    python bin/resolve-config.py methods/x/configs/pretrain.yaml --set A=b > out.json"
        probs = command_problems(bad, self.tools)
        self.assertTrue(any("missing required --config" in p for p in probs),
                        f"did not flag the missing --config: {probs}")

    def test_an_unknown_flag_is_flagged(self):
        bad = "    python bin/fetch-weights.py --provenance p --section x --out d"
        self.assertTrue(any("unknown flag --section" in p
                            for p in command_problems(bad, self.tools)))

    def test_a_correct_command_is_clean(self):
        good = ("    python bin/resolve-config.py --config methods/x/configs/pretrain.yaml \\\n"
                "        --set A=b --out out.json")
        self.assertEqual(command_problems(good, self.tools), [])

    def test_inline_prose_mention_is_not_treated_as_a_command(self):
        prose = "fetch it with `bin/fetch-weights.py` (see below)."
        self.assertEqual(command_problems(prose, self.tools), [])


if __name__ == "__main__":
    unittest.main()
