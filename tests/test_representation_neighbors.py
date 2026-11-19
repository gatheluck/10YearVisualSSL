"""Behavior of source-profile image and cross-image patch neighborhoods."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch required')
class Neighbors(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec('downstream.representation_neighbors')
        self.assertIsNotNone(spec, 'representation neighborhood API must exist')
        from downstream import representation_neighbors
        return representation_neighbors

    def test_cosine_not_magnitude_excludes_self_and_chunking_is_equivalent(self):
        import torch
        api = self.api()
        x = torch.tensor([[1., 0.], [20., 2.], [1., 1.], [-1., 0.], [0., -1.], [.1, .001]])
        got = api.nearest_neighbors(x, k=2, chunk_size=2)
        normalized = x / x.norm(dim=1, keepdim=True)
        scores = normalized @ normalized.T
        scores.fill_diagonal_(-float('inf'))
        expected = scores.topk(2, dim=1).indices.sort(dim=1).values
        self.assertTrue(torch.equal(got, expected))
        self.assertTrue(torch.equal(got, api.nearest_neighbors(x, k=2, chunk_size=5)))
        self.assertFalse((got == torch.arange(6)[:, None]).any())
        self.assertEqual(set(got[0].tolist()), {1, 5})

    def test_excludes_every_patch_of_same_image_not_just_diagonal(self):
        import torch
        api = self.api()
        x = torch.tensor([[1., .01], [1., .02], [.9, .2], [.8, .3], [-1., .2], [-1., .3]])
        ids = torch.tensor([7, 7, 11, 11, 25, 25])
        got = api.nearest_neighbors(x, k=2, image_ids=ids, chunk_size=1)
        self.assertFalse((ids[got] == ids[:, None]).any())
        self.assertEqual(set(got[0].tolist()), {2, 3})
        with self.assertRaises(ValueError): api.nearest_neighbors(x, k=5, image_ids=ids)

    def test_agreement_is_rowwise_intersection_not_reciprocal_or_jaccard(self):
        import torch
        api = self.api()
        a = torch.tensor([[1, 2], [0, 2], [0, 1], [0, 1]])
        b = torch.tensor([[2, 3], [2, 3], [0, 3], [0, 2]])
        got = api.agreement_matrix([a, b])
        self.assertEqual(got.tolist(), [[1., .5], [.5, 1.]])
        with self.assertRaises(ValueError): api.agreement_matrix([a, b[:2]])
        with self.assertRaises(ValueError): api.agreement_matrix([a, a.fill_(1)])

    def test_patch_order_centering_and_position_controls(self):
        import torch
        api = self.api()
        maps = torch.arange(1., 25.).reshape(3, 2, 2, 2)
        raw, ids = api.patch_vectors(maps, grid=2)
        flat = maps.permute(0, 2, 3, 1).reshape(12, 2)
        expected = flat / flat.norm(dim=1, keepdim=True)
        self.assertTrue(torch.equal(ids, torch.arange(3).repeat_interleave(4)))
        torch.testing.assert_close(raw, expected)
        centered, _ = api.patch_vectors(maps, grid=2, center_positions=True)
        expected = expected.reshape(3, 4, 2)
        expected = (expected - expected.mean(0, keepdim=True)).reshape(12, 2)
        expected = expected / expected.norm(dim=1, keepdim=True).clamp_min(1e-12)
        torch.testing.assert_close(centered, expected)
        idx = torch.tensor([[4 + i, 8 + i] for i in range(4)] + [[i, 8 + i] for i in range(4)] + [[i, 4 + i] for i in range(4)])
        self.assertEqual(api.position_shares(idx, grid=2), {'position': 1., 'row': 1., 'column': 1.})
        shifted = (idx + 1) % 12
        self.assertEqual(api.position_shares(shifted, grid=2)['position'], 0.)
        self.assertEqual(api.position_shares(shifted, grid=2)['row'], .5)
        self.assertEqual(api.position_shares(shifted, grid=2)['column'], 0.)
        with self.assertRaises(ValueError): api.patch_vectors(maps, grid=7)

    def test_pre_normalized_patch_vectors_are_not_normalized_twice(self):
        import numpy as np
        import torch
        api = self.api()
        rng = np.random.default_rng(1)
        for _ in range(2):
            rng.normal(size=(41, 23))
            maps = torch.from_numpy(rng.normal(size=(5, 19, 7, 7)).astype('float32'))
        vectors, ids = api.patch_vectors(maps, center_positions=True)
        expected = torch.cat([
            (vectors[start:start + 13] @ vectors.T).masked_fill(
                ids[start:start + 13, None] == ids[None, :], -float('inf')
            ).topk(10, dim=1).indices for start in range(0, len(vectors), 13)
        ]).sort(1).values
        got = api.nearest_neighbors(vectors, image_ids=ids, chunk_size=13, pre_normalized=True)
        self.assertTrue(torch.equal(got, expected))
        with self.assertRaises(ValueError): api.nearest_neighbors(vectors * 2, pre_normalized=True)

    def test_refuses_nonfinite_invalid_k_and_invalid_membership(self):
        import torch
        api = self.api()
        x = torch.eye(4)
        for kwargs in ({'k': 0}, {'k': True}, {'k': 4}, {'k': 1, 'chunk_size': 0}, {'k': 1, 'search_k': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): api.nearest_neighbors(x, **kwargs)
        bad = x.clone(); bad[0, 0] = float('nan')
        with self.assertRaises(ValueError): api.nearest_neighbors(bad, k=1)
        with self.assertRaises(ValueError): api.nearest_neighbors(torch.zeros(4, 2), k=1)
        with self.assertRaises(ValueError): api.nearest_neighbors(torch.full((21, 4), 1e30))
        with self.assertRaises(ValueError): api.nearest_neighbors(x, k=1, image_ids=torch.tensor([0, 1]))
        with self.assertRaises(ValueError): api.agreement_matrix([torch.tensor([[1.5], [0.5]])])

    def test_agreement_divides_exact_integer_intersection_once(self):
        import torch
        generator = torch.Generator().manual_seed(0)
        left = torch.stack([torch.randperm(43, generator=generator)[:10] for _ in range(43)])
        right = torch.stack([torch.randperm(43, generator=generator)[:10] for _ in range(43)])
        total = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(left, right))
        expected = total / (43 * 10)
        self.assertEqual(self.api().agreement_matrix([left, right])[0, 1], expected)

    def test_complex_features_refuse_lossy_real_projection(self):
        import torch
        api = self.api()
        features = torch.tensor([[1 + 5j, 2j], [2 + 1j, 1j], [1 + 2j, 3 + 2j]])
        with self.assertRaisesRegex(ValueError, 'real'):
            api.nearest_neighbors(features, k=1)
        with self.assertRaisesRegex(ValueError, 'real'):
            api.patch_vectors(features.reshape(3, 2, 1, 1), grid=1)

    def test_source_subset_selection_is_sorted_disjoint_and_repeatable(self):
        import numpy as np
        api = self.api()
        a = api.patch_rows(total=50, count=10, image_set='a')
        b = api.patch_rows(total=50, count=10, image_set='b')
        np.testing.assert_array_equal(a, np.sort(np.random.default_rng(2).choice(50, 10, replace=False)))
        rest = np.setdiff1d(np.arange(50), a)
        np.testing.assert_array_equal(b, np.sort(np.random.default_rng(3).choice(rest, 10, replace=False)))
        self.assertEqual(len(set(a) & set(b)), 0)
        with self.assertRaises(ValueError): api.patch_rows(total=10, count=6, image_set='b')


if __name__ == '__main__':
    unittest.main()
