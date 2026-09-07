import numpy as np

from smart_rxn_embeddings.evaluation.splitting import (
    group_holdout_split,
    stratified_group_holdout_split,
)


def _make_groups(n_groups: int, group_size: int) -> np.ndarray:
    """Return an array of length n_groups*group_size with repeated group ids."""
    return np.repeat(np.arange(n_groups), group_size)


class TestGroupHoldoutSplit:
    def test_no_group_straddles_train_test(self):
        groups = _make_groups(n_groups=50, group_size=4)
        n = len(groups)
        train_idx, test_idx = group_holdout_split(n, groups, test_size=0.2, seed=42)
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)

    def test_all_indices_accounted_for(self):
        groups = _make_groups(n_groups=30, group_size=3)
        n = len(groups)
        train_idx, test_idx = group_holdout_split(n, groups, test_size=0.2, seed=42)
        assert set(train_idx) | set(test_idx) == set(range(n))
        assert set(train_idx) & set(test_idx) == set()

    def test_test_size_is_approximately_respected(self):
        # Many small, equal-sized groups so the group constraint doesn't
        # prevent the split from approximating the requested fraction.
        groups = _make_groups(n_groups=200, group_size=2)
        n = len(groups)
        _, test_idx = group_holdout_split(n, groups, test_size=0.2, seed=42)
        assert 0.1 < len(test_idx) / n < 0.3

    def test_deterministic_given_seed(self):
        groups = _make_groups(n_groups=50, group_size=4)
        n = len(groups)
        r1 = group_holdout_split(n, groups, test_size=0.2, seed=7)
        r2 = group_holdout_split(n, groups, test_size=0.2, seed=7)
        assert np.array_equal(r1[0], r2[0])
        assert np.array_equal(r1[1], r2[1])


class TestStratifiedGroupHoldoutSplit:
    def test_no_group_straddles_train_test(self):
        rng = np.random.default_rng(0)
        groups = _make_groups(n_groups=100, group_size=4)
        # Each group is entirely one class (matches the real EC-annotation
        # invariant: every template in a reaction-group shares one EC label).
        group_labels = rng.integers(0, 3, size=100)
        y = group_labels[groups]

        train_idx, test_idx = stratified_group_holdout_split(
            y, groups, test_size=0.2, seed=42
        )
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)

    def test_all_indices_accounted_for(self):
        rng = np.random.default_rng(1)
        groups = _make_groups(n_groups=80, group_size=5)
        group_labels = rng.integers(0, 4, size=80)
        y = group_labels[groups]

        train_idx, test_idx = stratified_group_holdout_split(
            y, groups, test_size=0.2, seed=42
        )
        assert set(train_idx) | set(test_idx) == set(range(len(y)))
        assert set(train_idx) & set(test_idx) == set()

    def test_class_proportions_roughly_preserved(self):
        rng = np.random.default_rng(2)
        n_groups = 300
        groups = _make_groups(n_groups=n_groups, group_size=3)
        # Skewed but every class has enough groups to stratify.
        group_labels = rng.choice([0, 1, 2], size=n_groups, p=[0.7, 0.2, 0.1])
        y = group_labels[groups]

        train_idx, test_idx = stratified_group_holdout_split(
            y, groups, test_size=0.2, seed=42
        )
        full_props = np.bincount(y) / len(y)
        test_props = np.bincount(y[test_idx], minlength=3) / len(test_idx)
        assert np.allclose(full_props, test_props, atol=0.1)

    def test_deterministic_given_seed(self):
        rng = np.random.default_rng(3)
        groups = _make_groups(n_groups=60, group_size=4)
        group_labels = rng.integers(0, 3, size=60)
        y = group_labels[groups]

        r1 = stratified_group_holdout_split(y, groups, test_size=0.2, seed=7)
        r2 = stratified_group_holdout_split(y, groups, test_size=0.2, seed=7)
        assert np.array_equal(r1[0], r2[0])
        assert np.array_equal(r1[1], r2[1])
