"""Validate reaction SMARTS with RDKit and extract structural features."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from rdkit.Chem import AllChem, MolToSmarts

logger = logging.getLogger(__name__)

INPUT_FILE = "data/processed/clean_smarts.txt"
OUTPUT_FILE = "data/processed/validated_smarts.csv"


def _extract(smarts: str) -> dict:
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
    except Exception:  # noqa: BLE001
        return {"valid": False}
    if rxn is None:
        return {"valid": False}

    try:
        rxn.Initialize()

        n_reactants = rxn.GetNumReactantTemplates()
        n_products = rxn.GetNumProductTemplates()

        reactant_mols = [rxn.GetReactantTemplate(i) for i in range(n_reactants)]
        product_mols = [rxn.GetProductTemplate(i) for i in range(n_products)]

        reactant_smarts = ".".join(MolToSmarts(m) for m in reactant_mols)
        product_smarts = ".".join(MolToSmarts(m) for m in product_mols)

        n_atoms = sum(m.GetNumAtoms() for m in reactant_mols + product_mols)
        n_bonds = sum(m.GetNumBonds() for m in reactant_mols + product_mols)

        return {
            "valid": True,
            "reactant_smarts": reactant_smarts,
            "product_smarts": product_smarts,
            "n_reactant_templates": n_reactants,
            "n_product_templates": n_products,
            "n_atoms": n_atoms,
            "n_bonds": n_bonds,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to process '%s': %s", smarts[:60], exc)
        return {"valid": False}


def validate(smarts_list: list[str]) -> pd.DataFrame:
    records = []
    n_invalid = 0
    for smarts in smarts_list:
        result = _extract(smarts)
        result["smarts"] = smarts
        records.append(result)
        if not result["valid"]:
            n_invalid += 1

    logger.info(
        "Valid: %d  |  Invalid: %d  |  Total: %d",
        len(records) - n_invalid,
        n_invalid,
        len(records),
    )
    return pd.DataFrame(records)


def main(input_file: str, output_file: str) -> None:
    smarts_list = Path(input_file).read_text().splitlines()
    smarts_list = [s.strip() for s in smarts_list if s.strip()]
    logger.info("Loaded %d SMARTS from %s", len(smarts_list), input_file)

    df = validate(smarts_list)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info("Saved validated dataset to %s", output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Validate SMARTS with RDKit.")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    main(args.input, args.output)
