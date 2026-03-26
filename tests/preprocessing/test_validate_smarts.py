import pandas as pd
import pytest

from smart_rxn_embeddings.preprocessing.validate_smarts import _extract, validate

# ---------------------------------------------------------------------------
# Fixtures — real reaction SMARTS from RetroRules
# ---------------------------------------------------------------------------

VALID_SIMPLE = "[C;H1:1]=[N;H0:2]>>[C;H1:1]-[N;H0:2]"
VALID_MULTI_REACTANT = "([O;H0:1]-[C;H2:2].[O;H0:3]-[C;H1:4])>>([O;H0]=[C;H1:2].[O;H1:3].[C;H2:4]).[O;H1:1]"
INVALID_SMARTS = "not_a_smarts"
EMPTY_SMARTS = ""


# ---------------------------------------------------------------------------
# _extract
# ---------------------------------------------------------------------------

class TestExtract:
    def test_valid_simple_reaction(self):
        result = _extract(VALID_SIMPLE)
        assert result["valid"] is True
        assert result["reactant_smarts"] != ""
        assert result["product_smarts"] != ""

    def test_counts_atoms_and_bonds(self):
        result = _extract(VALID_SIMPLE)
        assert result["n_atoms"] > 0
        assert result["n_bonds"] >= 0

    def test_template_counts(self):
        result = _extract(VALID_SIMPLE)
        assert result["n_reactant_templates"] == 1
        assert result["n_product_templates"] == 1

    def test_multi_reactant_template_count(self):
        result = _extract(VALID_MULTI_REACTANT)
        assert result["valid"] is True
        assert result["n_reactant_templates"] >= 1
        assert result["n_product_templates"] >= 1

    def test_invalid_smarts_returns_invalid(self):
        result = _extract(INVALID_SMARTS)
        assert result["valid"] is False
        assert "reactant_smarts" not in result

    def test_empty_string_returns_invalid(self):
        result = _extract(EMPTY_SMARTS)
        assert result["valid"] is False

    def test_reactant_and_product_smarts_are_strings(self):
        result = _extract(VALID_SIMPLE)
        assert isinstance(result["reactant_smarts"], str)
        assert isinstance(result["product_smarts"], str)

    def test_n_atoms_is_sum_of_reactants_and_products(self):
        result = _extract(VALID_SIMPLE)
        # VALID_SIMPLE: reactant has 2 atoms, product has 2 atoms → total 4
        assert result["n_atoms"] == 4


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_returns_dataframe(self):
        df = validate([VALID_SIMPLE])
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_required_columns(self):
        df = validate([VALID_SIMPLE])
        expected = {"smarts", "valid", "reactant_smarts", "product_smarts", "n_atoms", "n_bonds"}
        assert expected.issubset(df.columns)

    def test_smarts_column_preserved(self):
        df = validate([VALID_SIMPLE, INVALID_SMARTS])
        assert list(df["smarts"]) == [VALID_SIMPLE, INVALID_SMARTS]

    def test_invalid_row_has_false_valid(self):
        df = validate([INVALID_SMARTS])
        assert df.loc[0, "valid"] == False  # noqa: E712 — np.False_ != False with `is`

    def test_mixed_input(self):
        df = validate([VALID_SIMPLE, INVALID_SMARTS, VALID_MULTI_REACTANT])
        assert df["valid"].sum() == 2
        assert (~df["valid"]).sum() == 1

    def test_empty_input(self):
        df = validate([])
        assert len(df) == 0

    def test_all_valid(self):
        df = validate([VALID_SIMPLE, VALID_MULTI_REACTANT])
        assert df["valid"].all()