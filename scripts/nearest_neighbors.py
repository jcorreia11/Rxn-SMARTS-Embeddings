"""Nearest-neighbor analysis in reaction SMARTS embedding space.

For a given query reaction (by SMARTS string or index), retrieves the top-K
nearest neighbors in embedding space and prints their EC labels to demonstrate
chemical relatedness.

Usage
-----
    # Query by index (0-based):
    python scripts/nearest_neighbors.py \\
        --embeddings data/embeddings/reaction_embeddings.npy \\
        --smarts     data/embeddings/reaction_smarts.txt \\
        --query-index 0

    # Query by SMARTS string:
    python scripts/nearest_neighbors.py \\
        --embeddings data/embeddings/reaction_embeddings.npy \\
        --smarts     data/embeddings/reaction_smarts.txt \\
        --query-smarts "[C;H1:1]=[N;H0:2]>>[O;H0]=[C;H0](-[O;H1])-[C;H1:1]-[N;H0:2]-[O;H1]"

    # Pick a random query:
    python scripts/nearest_neighbors.py \\
        --embeddings data/embeddings/reaction_embeddings.npy \\
        --smarts     data/embeddings/reaction_smarts.txt \\
        --random-query

    # Save results to JSON (for report generation):
    python scripts/nearest_neighbors.py \\
        --embeddings data/embeddings/reaction_embeddings.npy \\
        --smarts     data/embeddings/reaction_smarts.txt \\
        --query-index 0 \\
        --output     results/nn_results.json
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RAW_FILES = [
    "data/raw/retrorules-v3.0-metanetx.csv",
    "data/raw/retrorules-v3.0-rhea.csv",
]

_EC_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_ec_labels(raw_files: list[str], smarts_set: set[str]) -> dict[str, str]:
    """Return {smarts: full EC string} for validated SMARTS."""
    frames = []
    for p in raw_files:
        if not Path(p).exists():
            logger.warning("Raw file not found, skipping: %s", p)
            continue
        df = pd.read_csv(p, usecols=["TEMPLATE", "ECS", "VALID"])
        df = df[df["VALID"].astype(str).str.upper() == "TRUE"]
        frames.append(df)

    if not frames:
        return {}

    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset="TEMPLATE")
    raw = raw[raw["TEMPLATE"].isin(smarts_set)]
    return dict(zip(raw["TEMPLATE"], raw["ECS"].fillna("")))


def load_reaction_groups(validated_file: str, smarts_set: set[str]) -> dict[str, str]:
    """Return {smarts: reaction_group} for SMARTS present in *smarts_set*.

    RetroRules generates several templates per underlying reaction at
    different context radii (see load_data.py); templates sharing a group
    are near-duplicate siblings, which makes them a trivial (uninteresting)
    "success" for nearest-neighbor retrieval. Returns {} if the validated
    file or its reaction_group column is unavailable, so callers degrade to
    "no sibling info" rather than crashing.
    """
    if not Path(validated_file).exists():
        logger.warning(
            "Validated file not found, skipping reaction-group tagging: %s", validated_file
        )
        return {}

    header = pd.read_csv(validated_file, nrows=0).columns
    if "reaction_group" not in header:
        logger.warning(
            "%s has no 'reaction_group' column (re-run load_data.py + "
            "validate_smarts.py) — skipping sibling tagging",
            validated_file,
        )
        return {}

    df = pd.read_csv(validated_file, usecols=["smarts", "reaction_group"])
    df = df[df["smarts"].isin(smarts_set)]
    return dict(zip(df["smarts"], df["reaction_group"]))


def top_ec_class(ecs: str) -> str:
    """Extract the top-level EC class digit from an ECS string."""
    if not ecs or not ecs.strip():
        return "?"
    first = ecs.split(";")[0].strip()
    cls = first.split(".")[0] if first else "?"
    return cls if cls.isdigit() else "?"


# ---------------------------------------------------------------------------
# Nearest-neighbor search
# ---------------------------------------------------------------------------


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and all rows in matrix."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed = matrix / norms
    return normed @ query_norm


def find_neighbors(
    query_idx: int,
    embeddings: np.ndarray,
    k: int,
    metric: str,
    exclude_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of the k nearest neighbors, excluding the
    query itself and, when given, ``exclude_indices`` (e.g. same-group
    RetroRules radius-siblings, which are a trivial "success" for retrieval)."""
    query_vec = embeddings[query_idx]
    exclude = np.fromiter(exclude_indices or (), dtype=np.int64)

    if metric == "cosine":
        scores = cosine_similarity(query_vec, embeddings)
        # Higher cosine similarity = closer; exclude query itself
        scores[query_idx] = -np.inf
        scores[exclude] = -np.inf
        top_idx = np.argsort(scores)[::-1][:k]
        return top_idx, scores[top_idx]

    # Euclidean distance: lower = closer
    diffs = embeddings - query_vec
    distances = np.linalg.norm(diffs, axis=1)
    distances[query_idx] = np.inf
    distances[exclude] = np.inf
    top_idx = np.argsort(distances)[:k]
    return top_idx, distances[top_idx]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def format_smarts(s: str, max_len: int = 80) -> str:
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def print_results(
    query_idx: int,
    smarts_list: list[str],
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    ec_map: dict[str, str],
    metric: str,
    group_map: dict[str, str] | None = None,
) -> None:
    query_smarts = smarts_list[query_idx]
    query_ecs = ec_map.get(query_smarts, "")
    query_cls = top_ec_class(query_ecs)
    query_name = _EC_NAMES.get(query_cls, "Unknown")
    query_group = (group_map or {}).get(query_smarts)

    print("\n" + "=" * 72)
    print("QUERY REACTION")
    print("=" * 72)
    print(f"  Index : {query_idx}")
    print(f"  SMARTS: {format_smarts(query_smarts)}")
    print(f"  EC    : {query_ecs or 'N/A'}  →  EC {query_cls} ({query_name})")
    print()

    score_label = "Cosine similarity" if metric == "cosine" else "Euclidean distance"
    print(f"TOP-{len(neighbor_indices)} NEAREST NEIGHBORS  ({score_label})")
    print("-" * 72)

    all_cls = [query_cls] + [top_ec_class(ec_map.get(smarts_list[i], "")) for i in neighbor_indices]
    n_same_class = sum(1 for c in all_cls[1:] if c == query_cls and query_cls != "?")
    n_same_group = 0

    for rank, (idx, score) in enumerate(zip(neighbor_indices, neighbor_scores), start=1):
        s = smarts_list[idx]
        ecs = ec_map.get(s, "")
        cls = top_ec_class(ecs)
        name = _EC_NAMES.get(cls, "Unknown")
        same_group = query_group is not None and (group_map or {}).get(s) == query_group
        n_same_group += same_group
        tags = []
        if cls == query_cls and query_cls != "?":
            tags.append("same EC class")
        if same_group:
            tags.append("RetroRules radius-sibling")
        match = f" <-- {', '.join(tags)}" if tags else ""
        print(f"  #{rank}  [{score:+.4f}]  EC {cls} ({name}){match}")
        print(f"       Index : {idx}")
        print(f"       SMARTS: {format_smarts(s)}")
        print(f"       ECS   : {ecs or 'N/A'}")
        print()

    print("-" * 72)
    if query_cls != "?":
        print(
            f"  {n_same_class} / {len(neighbor_indices)} neighbors share EC class {query_cls} "
            f"({_EC_NAMES.get(query_cls, '')})"
        )
    else:
        print("  Query has no EC label — cannot assess class agreement.")
    if group_map:
        print(
            f"  {n_same_group} / {len(neighbor_indices)} neighbors are RetroRules "
            "radius-siblings of the query (trivial near-duplicates, not "
            "evidence of learned chemistry)"
        )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def build_json_output(
    query_idx: int,
    smarts_list: list[str],
    embeddings_shape: tuple[int, int],
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    ec_map: dict[str, str],
    args: argparse.Namespace,
    search_time_s: float,
    load_time_s: float,
    group_map: dict[str, str] | None = None,
) -> dict:
    query_smarts = smarts_list[query_idx]
    query_ecs = ec_map.get(query_smarts, "")
    query_cls = top_ec_class(query_ecs)
    query_group = (group_map or {}).get(query_smarts)

    neighbors = []
    for rank, (idx, score) in enumerate(zip(neighbor_indices, neighbor_scores), start=1):
        s = smarts_list[int(idx)]
        ecs = ec_map.get(s, "")
        cls = top_ec_class(ecs)
        same_group = query_group is not None and (group_map or {}).get(s) == query_group
        neighbors.append(
            {
                "rank": rank,
                "index": int(idx),
                "smarts": s,
                "ecs": ecs,
                "ec_class": cls,
                "ec_name": _EC_NAMES.get(cls, "Unknown"),
                "score": float(score),
                "same_ec_class": cls == query_cls and query_cls != "?",
                "same_group_as_query": same_group,
            }
        )

    n_same = sum(1 for n in neighbors if n["same_ec_class"])
    n_same_group = sum(1 for n in neighbors if n["same_group_as_query"])

    return {
        "meta": {
            "embeddings_path": str(args.embeddings),
            "smarts_path": str(args.smarts),
            "n_reactions": embeddings_shape[0],
            "embedding_dim": embeddings_shape[1],
            "metric": args.metric,
            "top_k": args.top_k,
            "random_seed": args.seed,
            "load_time_s": round(load_time_s, 3),
            "search_time_s": round(search_time_s, 3),
            "exclude_same_group": bool(getattr(args, "exclude_same_group", False)),
        },
        "query": {
            "index": query_idx,
            "smarts": query_smarts,
            "ecs": query_ecs,
            "ec_class": query_cls,
            "ec_name": _EC_NAMES.get(query_cls, "Unknown"),
            "reaction_group": query_group,
        },
        "neighbors": neighbors,
        "summary": {
            "n_same_ec_class": n_same,
            "fraction_same_ec_class": n_same / len(neighbors) if neighbors else 0.0,
            "n_same_group_as_query": n_same_group,
            "fraction_same_group_as_query": n_same_group / len(neighbors) if neighbors else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Top-K nearest-neighbor retrieval in reaction embedding space."
    )
    p.add_argument(
        "--embeddings",
        default="data/embeddings/reaction_embeddings.npy",
        help="Path to embeddings .npy file (N × d)",
    )
    p.add_argument(
        "--smarts",
        default="data/embeddings/reaction_smarts.txt",
        help="Path to aligned SMARTS .txt file (one per line)",
    )
    p.add_argument(
        "--raw",
        nargs="+",
        default=RAW_FILES,
        help="Raw RetroRules CSV files with ECS column",
    )

    query_group = p.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query-index",
        type=int,
        help="0-based index into the SMARTS list to use as query",
    )
    query_group.add_argument(
        "--query-smarts",
        type=str,
        help="Exact SMARTS string to use as query (must be present in the list)",
    )
    query_group.add_argument(
        "--random-query",
        action="store_true",
        help="Pick a random reaction as query",
    )

    p.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of neighbors to retrieve (default: 5)",
    )
    p.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric (default: cosine)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random-query",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Save results to this JSON file (for report generation). Omit to skip.",
    )
    p.add_argument(
        "--validated",
        default="data/processed/validated_smarts.csv",
        help=(
            "Validated SMARTS CSV with a 'reaction_group' column, used to "
            "tag/exclude RetroRules radius-siblings of the query "
            "(default: data/processed/validated_smarts.csv)"
        ),
    )
    p.add_argument(
        "--exclude-same-group",
        action="store_true",
        help=(
            "Exclude RetroRules radius-siblings of the query from the "
            "candidate pool, so retrieved neighbors reflect genuine "
            "cross-family similarity rather than trivial near-duplicates."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    embeddings_path = Path(args.embeddings)
    smarts_path = Path(args.smarts)

    t_load_start = time.perf_counter()
    logger.info("Loading embeddings from %s ...", embeddings_path)
    embeddings = np.load(embeddings_path).astype(np.float32)
    logger.info("  shape: %s", embeddings.shape)

    logger.info("Loading SMARTS list from %s ...", smarts_path)
    smarts_list = smarts_path.read_text().splitlines()
    assert len(smarts_list) == embeddings.shape[0], (
        f"Mismatch: {len(smarts_list)} SMARTS vs {embeddings.shape[0]} embeddings"
    )
    load_time_s = time.perf_counter() - t_load_start

    # Resolve query index
    if args.random_query:
        if args.seed is not None:
            random.seed(args.seed)
        query_idx = random.randrange(len(smarts_list))
    elif args.query_smarts is not None:
        if args.query_smarts not in smarts_list:
            raise ValueError(f"SMARTS not found in list: {args.query_smarts!r}")
        query_idx = smarts_list.index(args.query_smarts)
    else:
        query_idx = args.query_index
        if not (0 <= query_idx < len(smarts_list)):
            raise ValueError(
                f"--query-index {query_idx} out of range [0, {len(smarts_list) - 1}]"
            )

    # EC labels
    logger.info("Loading EC labels ...")
    ec_map = load_ec_labels(args.raw, set(smarts_list))
    logger.info("  %d / %d SMARTS have EC labels", len(ec_map), len(smarts_list))

    # Reaction groups (RetroRules radius-siblings)
    logger.info("Loading reaction groups ...")
    group_map = load_reaction_groups(args.validated, set(smarts_list))
    logger.info("  %d / %d SMARTS have a reaction group", len(group_map), len(smarts_list))

    exclude_indices = None
    if args.exclude_same_group:
        query_reaction_group = group_map.get(smarts_list[query_idx])
        if query_reaction_group is None:
            logger.warning(
                "Query has no reaction group — --exclude-same-group has no effect"
            )
        else:
            exclude_indices = {
                i for i, s in enumerate(smarts_list)
                if group_map.get(s) == query_reaction_group
            }
            exclude_indices.discard(query_idx)
            logger.info(
                "Excluding %d same-group siblings from the candidate pool",
                len(exclude_indices),
            )

    # Search
    logger.info(
        "Finding top-%d neighbors for query index %d (metric=%s) ...",
        args.top_k,
        query_idx,
        args.metric,
    )
    t_search = time.perf_counter()
    neighbor_indices, neighbor_scores = find_neighbors(
        query_idx, embeddings, k=args.top_k, metric=args.metric,
        exclude_indices=exclude_indices,
    )
    search_time_s = time.perf_counter() - t_search
    logger.info("  Search completed in %.3f s", search_time_s)

    print_results(
        query_idx=query_idx,
        smarts_list=smarts_list,
        neighbor_indices=neighbor_indices,
        neighbor_scores=neighbor_scores,
        ec_map=ec_map,
        metric=args.metric,
        group_map=group_map,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = build_json_output(
            query_idx=query_idx,
            smarts_list=smarts_list,
            embeddings_shape=embeddings.shape,
            neighbor_indices=neighbor_indices,
            neighbor_scores=neighbor_scores,
            ec_map=ec_map,
            args=args,
            group_map=group_map,
            search_time_s=search_time_s,
            load_time_s=load_time_s,
        )
        output_path.write_text(json.dumps(result, indent=2))
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()