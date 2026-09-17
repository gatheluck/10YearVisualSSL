"""Native profile fixtures must not replace Python's process-wide importer."""
import ast
from pathlib import Path
import unittest


def global_import_patches(source):
    """Return line numbers of global importer patches, ignoring string decoys."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == 'patch'
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 'importlib.import_module'):
            found.append(node.lineno)
        elif (isinstance(func, ast.Attribute) and func.attr == 'object'
              and isinstance(func.value, ast.Name) and func.value.id == 'patch'
              and len(node.args) >= 2
              and isinstance(node.args[0], ast.Name)
              and node.args[0].id == 'importlib'
              and isinstance(node.args[1], ast.Constant)
              and node.args[1].value == 'import_module'):
            found.append(node.lineno)
    return sorted(found)


class TestNativeProfileMockIsolation(unittest.TestCase):
    def test_detector_rejects_both_global_patch_forms(self):
        self.assertEqual(global_import_patches(
            "with patch('importlib.import_module'): pass\n"
            "with patch.object(importlib, 'import_module'): pass\n"), [1, 2])

    def test_detector_accepts_provider_local_patch_and_exact_decoys(self):
        self.assertEqual(global_import_patches(
            "# patch('importlib.import_module')\n"
            "example = \"patch('importlib.import_module')\"\n"
            "with patch.object(provider, 'importlib', replacement): pass\n"
            "with patch.object(provider_support, 'import_sibling'): pass\n"), [])

    def test_native_profile_fixtures_leave_global_importer_untouched(self):
        paths = sorted(Path(__file__).parent.glob('test_*native_options.py'))
        self.assertTrue(paths, 'no native profile fixtures discovered')
        for path in paths:
            with self.subTest(path=path.name):
                self.assertEqual(global_import_patches(path.read_text()), [],
                                 'patch the provider namespace, not importlib')
