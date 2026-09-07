from unittest.mock import MagicMock, patch

import pandas as pd

from smart_rxn_embeddings.preprocessing.validate_smarts import (
    _extract,
    attach_reaction_groups,
    main,
    validate,
)

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

    def test_reaction_from_smarts_returns_none(self):
        with patch(
            "smart_rxn_embeddings.preprocessing.validate_smarts.AllChem.ReactionFromSmarts",
            return_value=None,
        ):
            result = _extract(VALID_SIMPLE)
        assert result == {"valid": False}

    def test_initialize_raises_returns_invalid(self):
        mock_rxn = MagicMock()
        mock_rxn.Initialize.side_effect = RuntimeError("boom")
        with patch(
            "smart_rxn_embeddings.preprocessing.validate_smarts.AllChem.ReactionFromSmarts",
            return_value=mock_rxn,
        ):
            result = _extract(VALID_SIMPLE)
        assert result == {"valid": False}


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_returns_dataframe(self):
        df = validate([VALID_SIMPLE])
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_required_columns(self):
        df = validate([VALID_SIMPLE])
        expected = {
            "smarts",
            "valid",
            "reactant_smarts",
            "product_smarts",
            "n_atoms",
            "n_bonds",
        }
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_produces_output_file(self, tmp_path):
        input_path = tmp_path / "smarts.txt"
        input_path.write_text(f"{VALID_SIMPLE}\n{INVALID_SMARTS}\n")
        output_path = tmp_path / "out.csv"
        groups_path = tmp_path / "groups.csv"  # deliberately absent
        main(str(input_path), str(output_path), str(groups_path))
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert "valid" in df.columns

    def test_main_attaches_reaction_groups(self, tmp_path):
        input_path = tmp_path / "smarts.txt"
        input_path.write_text(f"{VALID_SIMPLE}\n{INVALID_SMARTS}\n")
        output_path = tmp_path / "out.csv"
        groups_path = tmp_path / "groups.csv"
        pd.DataFrame(
            {
                "smarts": [VALID_SIMPLE, INVALID_SMARTS],
                "reaction_group": ["RHEA:1", "RHEA:2"],
            }
        ).to_csv(groups_path, index=False)

        main(str(input_path), str(output_path), str(groups_path))

        df = pd.read_csv(output_path)
        assert df.set_index("smarts")["reaction_group"].to_dict() == {
            VALID_SIMPLE: "RHEA:1",
            INVALID_SMARTS: "RHEA:2",
        }


# ---------------------------------------------------------------------------
# attach_reaction_groups
# ---------------------------------------------------------------------------


class TestAttachReactionGroups:
    def test_joins_matching_groups(self, tmp_path):
        groups_path = tmp_path / "groups.csv"
        pd.DataFrame({"smarts": [VALID_SIMPLE], "reaction_group": ["RHEA:1"]}).to_csv(
            groups_path, index=False
        )

        df = validate([VALID_SIMPLE])
        result = attach_reaction_groups(df, str(groups_path))
        assert result.loc[0, "reaction_group"] == "RHEA:1"

    def test_falls_back_to_smarts_when_file_missing(self, tmp_path):
        df = validate([VALID_SIMPLE])
        result = attach_reaction_groups(df, str(tmp_path / "does_not_exist.csv"))
        assert result.loc[0, "reaction_group"] == VALID_SIMPLE

    def test_falls_back_to_smarts_when_row_unmatched(self, tmp_path):
        groups_path = tmp_path / "groups.csv"
        pd.DataFrame(
            {"smarts": ["some other smarts"], "reaction_group": ["RHEA:1"]}
        ).to_csv(groups_path, index=False)

        df = validate([VALID_SIMPLE])
        result = attach_reaction_groups(df, str(groups_path))
        assert result.loc[0, "reaction_group"] == VALID_SIMPLE

    def test_reaction_group_never_null(self, tmp_path):
        df = validate([VALID_SIMPLE, INVALID_SMARTS])
        result = attach_reaction_groups(df, str(tmp_path / "missing.csv"))
        assert result["reaction_group"].notna().all()
