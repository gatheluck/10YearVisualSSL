"""Method instructions describe recipes without internal experiment numbers."""
import re
import unittest
from pathlib import Path

from tests._repo_files import repository_files


ROOT = Path(__file__).resolve().parents[1]
NUMBERED = re.compile(r"\bstep[ -]?[1-4]\b", re.IGNORECASE)


def numbered_instructions(text):
    # Preserve exact config values, filenames and Markdown link destinations.
    prose = re.sub(r"`[^`\n]+`|\]\([^)]+\)", "", text)
    return [line for line in prose.splitlines() if NUMBERED.search(line)]


class MethodManuscriptTerms(unittest.TestCase):
    def test_method_instructions_explain_recipes_without_numbered_labels(self):
        guides = [path for path in repository_files(ROOT)
                  if path.suffix == ".md" and path.relative_to(ROOT).parts[0] == "methods"]
        self.assertTrue(guides)
        failures = {str(path.relative_to(ROOT)): matches[0][:100] for path in guides
                    if (matches := numbered_instructions(path.read_text()))}
        self.assertEqual(failures, {})

    def test_keep_configuration_values_and_link_destinations(self):
        self.assertEqual(numbered_instructions("Use `augmentation: step2` or [guide](step-2.md)."), [])
        self.assertTrue(numbered_instructions("Train the Step-2 backbone."))
