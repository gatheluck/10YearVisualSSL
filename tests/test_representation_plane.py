"""Observable rank, missingness, held-family and display contracts."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy required")
class RepresentationPlane(unittest.TestCase):
    def api(self, name):
        from downstream import representation_plane

        function = getattr(representation_plane, name, None)
        self.assertTrue(callable(function), f"missing rank-plane behavior: {name}")
        return function

    def fixture(self):
        import numpy as np

        rng = np.random.default_rng(984)
        x = rng.normal(size=28)
        y = rng.normal(size=28)
        scores = x * 0.4 + y * 0.8 + rng.normal(size=28) * 0.2
        return x, y, scores, [f"family-{i // 4}" for i in range(28)]

    def test_complex_arrays_refuse_lossy_real_projection(self):
        import numpy as np

        with self.assertRaisesRegex(ValueError, "real"):
            self.api("percentile_rank")(np.array([1 + 3j, 2 + 2j, 3 + 1j]))
        x, y, scores, families = self.fixture()
        for slot in range(3):
            vectors = [x.copy(), y.copy(), scores.copy()]
            vectors[slot] = vectors[slot].astype(complex) + 1j
            with self.subTest(slot=slot), self.assertRaisesRegex(ValueError, "real"):
                self.api("fit_task")(*vectors, families)

    def test_percentiles_use_average_ties_and_one_based_population_divisor(self):
        import numpy as np

        rank = self.api("percentile_rank")
        np.testing.assert_array_equal(rank([9, 1, 1, 5]), [1, 0.375, 0.375, 0.75])
        np.testing.assert_array_equal(rank([2, 2]), [0.75, 0.75])
        for bad in ([], [1, float("nan")], [float("inf")], [[1, 2]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                rank(bad)

    def test_lofo_uses_global_ranks_and_pooled_predictions_not_fold_correlations(self):
        import numpy as np

        def rank(values):
            return np.argsort(np.argsort(values)) + 1

        def correlation(a, b):
            return float(np.corrcoef(rank(a), rank(b))[0, 1])

        fit = self.api("fit_task")
        x, y, scores, families = self.fixture()
        result = fit(x, y, scores, families)
        design = np.column_stack([np.ones(28), rank(x) / 28, rank(y) / 28])
        target = rank(scores) / 28
        predictions = np.empty(28)
        correlations = []
        for label in sorted(set(families)):
            held = np.array(families) == label
            coefficients = np.linalg.lstsq(design[~held], target[~held], rcond=None)[0]
            predictions[held] = design[held] @ coefficients
            correlations.append(correlation(predictions[held], target[held]))
        np.testing.assert_allclose(result["predictions_xy"], predictions, atol=1e-13)
        self.assertAlmostEqual(result["rho_xy"], correlation(predictions, target))
        self.assertNotAlmostEqual(result["rho_xy"], np.mean(correlations), places=3)
        self.assertEqual([fold["family"] for fold in result["folds"]], sorted(set(families)))
        self.assertEqual([fold["held_out"] for fold in result["folds"]], [4] * 7)

    def test_missing_task_rows_are_removed_before_coordinate_and_score_ranking(self):
        import numpy as np

        fit = self.api("fit_task")
        x, y, scores, families = self.fixture()
        keep = np.ones(28, dtype=bool)
        keep[[0, 6, 10, 27]] = False
        scores[~keep] = np.nan
        actual = fit(x, y, scores, families, valid=keep)
        expected = fit(x[keep], y[keep], scores[keep], np.array(families)[keep])
        self.assertEqual(actual["n"], 24)
        self.assertEqual(actual["indices"], np.flatnonzero(keep).tolist())
        for key in ("coordinates", "target", "coefficients", "predictions_xy", "rho_xy"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-13)
        with self.assertRaises(ValueError):
            fit(x, y, scores, families)

    def test_lower_error_direction_and_fold_angles_are_explicit(self):
        import numpy as np

        fit = self.api("fit_task")
        args = self.fixture()
        high, low = fit(*args), fit(*args, lower_is_better=True)
        np.testing.assert_allclose(np.array(low["target"]) + high["target"], 29 / 28)
        np.testing.assert_allclose(low["coefficients"][1:], -np.array(high["coefficients"])[1:], atol=1e-12)
        a, b = high["coefficients"][1:]
        self.assertAlmostEqual(high["theta"], np.degrees(np.arctan2(b, a)))
        self.assertAlmostEqual(high["tmin"], min(f["theta"] for f in high["folds"]))
        self.assertAlmostEqual(high["tmax"], max(f["theta"] for f in high["folds"]))

    def test_nonfinite_constant_ambiguous_and_empty_fold_inputs_fail_closed(self):
        import numpy as np

        fit = self.api("fit_task")
        x, y, scores, families = self.fixture()
        bad = [
            ([], [], [], []),
            (x[:-1], y, scores, families),
            (np.ones(28), y, scores, families),
            (x, y, np.ones(28), families),
            (x, x, scores, families),
            (x, y, scores, ["only"] * 28),
            (x, y, scores, [""] + families[1:]),
            (x, y, scores, [1] + families[1:]),
            (np.r_[np.nan, x[1:]], y, scores, families),
        ]
        for args in bad:
            with self.subTest(args=str(args)[:50]), self.assertRaises(ValueError):
                fit(*args)
        for valid in ([1] * 28, [True] * 27, [False] * 28):
            with self.subTest(valid=valid), self.assertRaises(ValueError):
                fit(x, y, scores, families, valid=valid)
        with self.assertRaises(ValueError):
            fit(x, y, scores, families, lower_is_better="yes")

    def test_singular_held_family_and_undefined_task_direction_are_rejected(self):
        fit = self.api("fit_task")
        x, y, scores, families = self.fixture()
        y = x.copy()
        y[:4] = x[:4][::-1]
        with self.assertRaisesRegex(ValueError, "identifiable"):
            fit(x, y, scores, families)
        with self.assertRaisesRegex(ValueError, "no direction"):
            fit([-3, -2, -1, 0, 0, 1, 2, 3], [0, 1, 2, 3, -3, -2, -1, 0],
                [1, 2, 3, 4, 4, 3, 2, 1], ["a", "b", "c", "d", "d", "c", "b", "a"])

    def test_rotation_is_explicit_shared_and_never_changes_fitted_coordinates(self):
        import copy

        fit, rotate = self.api("fit_task"), self.api("rotate_directions")
        result = fit(*self.fixture())
        original = copy.deepcopy(result)
        tasks = {"image": result, "depth": fit(*self.fixture(), lower_is_better=True)}
        unchanged = rotate(tasks)
        rotated = rotate(tasks, rotation_degrees=-result["theta"])
        self.assertEqual(unchanged["display_rotation_deg"], 0)
        self.assertEqual(unchanged["tasks"]["image"]["theta"], result["theta"])
        self.assertAlmostEqual(rotated["tasks"]["image"]["theta"], 0)
        self.assertAlmostEqual(rotated["tasks"]["depth"]["theta"], tasks["depth"]["theta"] - result["theta"])
        self.assertEqual(result, original)
        self.assertEqual(rotated["tasks"]["image"]["fold_range"], [result["tmin"] - result["theta"], result["tmax"] - result["theta"]])
        for angle in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                rotate(tasks, rotation_degrees=angle)
        with self.assertRaises(ValueError):
            rotate({})
        for malformed in ({}, {"theta": 0, "tmin": 2, "tmax": 1},
                          {"theta": float("nan"), "tmin": 0, "tmax": 1}):
            with self.assertRaises(ValueError):
                rotate({"image": malformed})


if __name__ == "__main__":
    unittest.main()
