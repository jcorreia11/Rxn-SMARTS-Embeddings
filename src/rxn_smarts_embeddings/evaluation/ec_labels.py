"""Shared EC-label loading for downstream evaluation scripts.

Used by ``scripts/train_ec_classifier.py`` and ``scripts/ablation_pooling.py``
(previously duplicated in both). Also attaches the ``reaction_group`` column
produced by ``load_data.py``/``validate_smarts.py``: RetroRules generates
several templates per underlying reaction at different context radii, and
templates sharing a group are near-duplicates with identical EC annotations.
Any split downstream of this loader must keep whole groups together — see
``rxn_smarts_embeddings.evaluation.splitting``.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def primary_ec(ecs_field: str, depth: int) -> str | None:
    """Return the first EC number in *ecs_field*, truncated to *depth* components.

    Parameters
    ----------
    ecs_field:
        Semicolon-separated EC numbers, e.g. ``"1.14.13.39;1.14.13.40"``.
    depth:
        Number of leading components to keep (1-4).

    Returns
    -------
    str or None
        The truncated EC number, or ``None`` if *ecs_field* is blank or not a string.
    """
    if not isinstance(ecs_field, str) or not ecs_field.strip():
        return None
    first = ecs_field.split(";")[0].strip()
    if not first:
        return None
    return ".".join(first.split(".")[:depth])


def load_labelled_smarts(
    raw_files: list[str],
    validated_file: str,
    ec_depth: int,
    n_samples: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, int]:
    """Join EC labels onto validated SMARTS, dropping rare classes and unlabelled rows.

    Parameters
    ----------
    raw_files:
        RetroRules CSVs (``TEMPLATE``/``ECS``/``VALID`` columns) to source EC
        annotations from.
    validated_file:
        Path to ``validated_smarts.csv`` (output of ``validate_smarts.py``).
    ec_depth:
        EC-number truncation depth passed to :func:`primary_ec`.
    n_samples:
        Optional stratified subsample size. ``None`` keeps every labelled row.
    random_seed:
        Seed for the stratified subsample.

    Returns
    -------
    df : pandas.DataFrame
        Columns ``smarts``, ``label``, ``group``.
    n_available : int
        Labelled templates remaining after dropping rare classes but *before*
        any ``n_samples`` subsampling — callers should record this alongside
        the actual sample size used, since subsampling a large annotated pool
        down to ``n_samples`` is otherwise easy to conflate with rare-class
        filtering when reading results later.
    """
    frames = []
    for p in raw_files:
        df = pd.read_csv(p, usecols=["TEMPLATE", "ECS", "VALID"])
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["VALID"].astype(str).str.upper() == "TRUE"]
    raw["label"] = raw["ECS"].apply(lambda x: primary_ec(x, ec_depth))
    raw = raw.dropna(subset=["label"])
    raw = raw.rename(columns={"TEMPLATE": "smarts"}).drop_duplicates(subset="smarts")
    ec_map = raw.set_index("smarts")["label"].to_dict()
    logger.info("EC map built: %d labelled SMARTS", len(ec_map))

    val = pd.read_csv(validated_file)
    val = val[val["valid"].astype(str).str.upper() == "TRUE"]
    val = val.dropna(subset=["smarts"])
    val["label"] = val["smarts"].map(ec_map)
    val = val.dropna(subset=["label"])

    if "reaction_group" not in val.columns:
        logger.warning(
            "%s has no 'reaction_group' column — falling back to SMARTS-as-"
            "group (no cross-radius leakage protection). Re-run the "
            "preprocessing pipeline (load_data.py, validate_smarts.py) to "
            "regenerate validated_smarts.csv with reaction groups.",
            validated_file,
        )
        val["reaction_group"] = val["smarts"]
    val["reaction_group"] = val["reaction_group"].fillna(val["smarts"])
    logger.info("After join with validated set: %d SMARTS with EC labels", len(val))

    # Drop classes too rare to survive stratified splitting.
    min_count = max(10, int(n_samples / len(val) * 10) + 2) if n_samples else 10
    class_counts = val["label"].value_counts()
    valid_classes = class_counts[class_counts >= min_count].index
    dropped = sorted(set(class_counts.index) - set(valid_classes))
    if dropped:
        logger.info(
            "Dropping %d rare classes (<%d samples): %s",
            len(dropped),
            min_count,
            dropped,
        )
        val = val[val["label"].isin(valid_classes)].reset_index(drop=True)

    n_available = len(val)

    counts = val["label"].value_counts()
    logger.info("Class distribution:\n%s", counts.to_string())

    if n_samples and n_samples < len(val):
        total = len(val)
        sampled = [
            group.sample(
                min(len(group), max(1, int(n_samples * len(group) / total))),
                random_state=random_seed,
            )
            for _, group in val.groupby("label")
        ]
        val = (
            pd.concat(sampled)
            .sample(frac=1, random_state=random_seed)
            .reset_index(drop=True)
        )
        logger.info(
            "Sampled %d SMARTS (stratified) out of %d available", len(val), n_available
        )

    result = val[["smarts", "label", "reaction_group"]].rename(
        columns={"reaction_group": "group"}
    )
    return result.reset_index(drop=True), n_available
