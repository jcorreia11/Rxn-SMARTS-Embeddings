"""Load and clean raw RetroRules SMARTS datasets."""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_FILES = [
    "data/raw/retrorules-v3.0-metanetx.csv",
    "data/raw/retrorules-v3.0-rhea.csv",
]
OUTPUT_FILE = "data/processed/clean_smarts.txt"
GROUPS_OUTPUT_FILE = "data/processed/reaction_groups.csv"


def load_raw(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=["TEMPLATE_ID", "TEMPLATE", "VALID", "REACTIONS"])
        logger.info("Loaded %d rows from %s", len(df), p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _filtered(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the VALID / non-empty / dedup filters shared by ``clean`` and
    ``build_reaction_groups``, so both operate on the identical row set."""
    n_start = len(df)

    # Keep only rows marked valid by RetroRules itself
    df = df[df["VALID"].astype(str).str.upper() == "TRUE"]
    logger.info("After VALID filter: %d / %d rows", len(df), n_start)

    # Drop empty SMARTS
    df = df[df["TEMPLATE"].notna() & (df["TEMPLATE"].str.strip() != "")]
    logger.info("After empty filter: %d rows", len(df))

    # Drop duplicates by SMARTS string (keep first occurrence)
    df = df.drop_duplicates(subset="TEMPLATE")
    logger.info("After dedup: %d unique SMARTS", len(df))

    return df


def clean(df: pd.DataFrame) -> pd.Series:
    return _filtered(df)["TEMPLATE"].reset_index(drop=True)


def build_reaction_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Map each cleaned/deduped SMARTS to a reaction-family group id.

    RetroRules generates several templates per underlying reaction at
    different context "radii" — these share the ``REACTIONS`` field and are
    near-duplicates with identical EC annotations. Grouping on ``REACTIONS``
    lets downstream splits avoid leaking a template's radius-siblings across
    train/test. Rows with a missing/blank ``REACTIONS`` value fall back to
    the SMARTS string itself as a singleton group, so every row gets a group
    and nothing is dropped or merged incorrectly.
    """
    filtered = _filtered(df)
    reactions = filtered["REACTIONS"]
    has_reactions = reactions.notna() & (reactions.astype(str).str.strip() != "")
    group = reactions.where(has_reactions, filtered["TEMPLATE"])

    return pd.DataFrame(
        {
            "smarts": filtered["TEMPLATE"].to_numpy(),
            "reaction_group": group.to_numpy(),
        }
    ).reset_index(drop=True)


def save(smarts: pd.Series, output: str) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    smarts.to_csv(output, index=False, header=False)
    logger.info("Saved %d SMARTS to %s", len(smarts), output)


def save_groups(groups: pd.DataFrame, output: str) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    groups.to_csv(output, index=False)
    logger.info("Saved %d reaction groups to %s", len(groups), output)


def main(
    raw_files: list[str],
    output: str,
    groups_output: str = GROUPS_OUTPUT_FILE,
) -> None:
    df = load_raw(raw_files)
    smarts = clean(df)
    save(smarts, output)
    groups = build_reaction_groups(df)
    save_groups(groups, groups_output)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Load and clean raw SMARTS data.")
    parser.add_argument("--input", nargs="+", default=RAW_FILES)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--groups-output", default=GROUPS_OUTPUT_FILE)
    args = parser.parse_args()

    main(args.input, args.output, args.groups_output)
