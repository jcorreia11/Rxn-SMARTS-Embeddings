"""Group-aware train/test holdout splits.

RetroRules templates that share a ``reaction_group`` (see
``rxn_smarts_embeddings.evaluation.ec_labels``) are near-duplicate
radius-variants of the same underlying reaction with identical labels.
Plain (stratified) random splitting lets a template's siblings leak across
train/test, inflating every downstream metric. The helpers here keep whole
groups on one side of the split.

sklearn has ``GroupShuffleSplit`` but no stratified equivalent as a one-shot
holdout split, so ``stratified_group_holdout_split`` approximates it via
``StratifiedGroupKFold`` and takes a single fold as the held-out set.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def group_holdout_split(
    n: int,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware holdout split with no label to stratify on.

    Used for the MLM pretraining train/val split, where masked-language
    modelling has no downstream label.

    Parameters
    ----------
    n:
        Total number of samples.
    groups:
        Group id per sample, length *n*.
    test_size:
        Fraction of samples (approximately) held out for test.
    seed:
        Random seed.

    Returns
    -------
    train_idx, test_idx : numpy.ndarray
        Index arrays into the original *n* samples.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.arange(n), groups=groups))
    return train_idx, test_idx


def stratified_group_holdout_split(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware holdout split that also approximately preserves class balance.

    Implemented via ``StratifiedGroupKFold`` with ``n_splits =
    round(1 / test_size)``, taking fold 0's test indices as the held-out set.
    Used for the EC-classifier and pooling-ablation 80/20 train/test split.

    Parameters
    ----------
    y:
        Class label per sample.
    groups:
        Group id per sample, same length as *y*.
    test_size:
        Approximate fraction of samples held out for test.
    seed:
        Random seed.

    Returns
    -------
    train_idx, test_idx : numpy.ndarray
        Index arrays into the original samples.
    """
    n_splits = round(1 / test_size)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
    return train_idx, test_idx
