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


def load_raw(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=["TEMPLATE_ID", "TEMPLATE", "VALID"])
        logger.info("Loaded %d rows from %s", len(df), p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.Series:
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

    return df["TEMPLATE"].reset_index(drop=True)


def save(smarts: pd.Series, output: str) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    smarts.to_csv(output, index=False, header=False)
    logger.info("Saved %d SMARTS to %s", len(smarts), output)


def main(raw_files: list[str], output: str) -> None:
    df = load_raw(raw_files)
    smarts = clean(df)
    save(smarts, output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Load and clean raw SMARTS data.")
    parser.add_argument("--input", nargs="+", default=RAW_FILES)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    main(args.input, args.output)