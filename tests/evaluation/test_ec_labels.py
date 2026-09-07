import pandas as pd

from smart_rxn_embeddings.evaluation.ec_labels import load_labelled_smarts, primary_ec

# ---------------------------------------------------------------------------
# primary_ec
# ---------------------------------------------------------------------------


class TestPrimaryEc:
    def test_truncates_to_depth(self):
        assert primary_ec("1.14.13.39", depth=1) == "1"
        assert primary_ec("1.14.13.39", depth=2) == "1.14"
        assert primary_ec("1.14.13.39", depth=3) == "1.14.13"

    def test_takes_first_of_multiple_ecs(self):
        assert primary_ec("3.1.1.122;3.5.1.137", depth=1) == "3"

    def test_none_for_blank_or_missing(self):
        assert primary_ec("", depth=1) is None
        assert primary_ec("   ", depth=1) is None
        assert primary_ec(None, depth=1) is None
        assert primary_ec(float("nan"), depth=1) is None


# ---------------------------------------------------------------------------
# load_labelled_smarts
# ---------------------------------------------------------------------------


def _write_raw(tmp_path, rows):
    path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _write_validated(tmp_path, rows):
    path = tmp_path / "validated.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _ten_per_class(class_ecs: list[str]) -> tuple[list[str], list[str]]:
    """Return (templates, ecs) with 10 distinct templates per EC value.

    ``load_labelled_smarts`` drops any class with fewer than 10 examples
    (the hard-coded floor in its rare-class filter), so fixtures need at
    least 10 rows per class that's expected to survive.
    """
    templates, ecs = [], []
    for ec in class_ecs:
        for i in range(10):
            templates.append(f"{ec}-T{i}")
            ecs.append(ec)
    return templates, ecs


class TestLoadLabelledSmarts:
    def test_returns_smarts_label_group_columns(self, tmp_path):
        templates, ecs = _ten_per_class(["1.1.1.1", "2.2.2.2"])
        raw = _write_raw(
            tmp_path,
            {"TEMPLATE": templates, "ECS": ecs, "VALID": ["True"] * len(templates)},
        )
        validated = _write_validated(
            tmp_path,
            {
                "smarts": templates,
                "valid": ["True"] * len(templates),
                "reaction_group": templates,
            },
        )
        df, n_available = load_labelled_smarts(
            [raw], validated, ec_depth=1, n_samples=None, random_seed=42
        )
        assert list(df.columns) == ["smarts", "label", "group"]
        assert n_available == len(templates)

    def test_falls_back_when_no_reaction_group_column(self, tmp_path):
        templates, ecs = _ten_per_class(["1.1.1.1"])
        raw = _write_raw(
            tmp_path,
            {"TEMPLATE": templates, "ECS": ecs, "VALID": ["True"] * len(templates)},
        )
        validated = _write_validated(
            tmp_path, {"smarts": templates, "valid": ["True"] * len(templates)}
        )
        df, _ = load_labelled_smarts(
            [raw], validated, ec_depth=1, n_samples=None, random_seed=42
        )
        assert (df["group"] == df["smarts"]).all()

    def test_drops_rare_classes(self, tmp_path):
        # 15 templates of class "1", only 2 of class "2" (< default min_count=10)
        templates = [f"T{i}" for i in range(17)]
        ecs = ["1.1.1.1"] * 15 + ["2.2.2.2"] * 2
        raw = _write_raw(
            tmp_path, {"TEMPLATE": templates, "ECS": ecs, "VALID": ["True"] * 17}
        )
        validated = _write_validated(
            tmp_path,
            {
                "smarts": templates,
                "valid": ["True"] * 17,
                "reaction_group": templates,
            },
        )
        df, n_available = load_labelled_smarts(
            [raw], validated, ec_depth=1, n_samples=None, random_seed=42
        )
        assert set(df["label"]) == {"1"}
        assert n_available == 15

    def test_n_available_reflects_pre_subsample_count(self, tmp_path):
        templates = [f"T{i}" for i in range(20)]
        ecs = ["1.1.1.1"] * 20
        raw = _write_raw(
            tmp_path, {"TEMPLATE": templates, "ECS": ecs, "VALID": ["True"] * 20}
        )
        validated = _write_validated(
            tmp_path,
            {
                "smarts": templates,
                "valid": ["True"] * 20,
                "reaction_group": templates,
            },
        )
        df, n_available = load_labelled_smarts(
            [raw], validated, ec_depth=1, n_samples=5, random_seed=42
        )
        assert n_available == 20
        assert len(df) == 5

    def test_unlabelled_templates_excluded(self, tmp_path):
        templates, ecs = _ten_per_class(["1.1.1.1"])
        templates.append("UNLABELLED")
        ecs.append("")
        raw = _write_raw(
            tmp_path,
            {"TEMPLATE": templates, "ECS": ecs, "VALID": ["True"] * len(templates)},
        )
        validated = _write_validated(
            tmp_path,
            {
                "smarts": templates,
                "valid": ["True"] * len(templates),
                "reaction_group": templates,
            },
        )
        df, _ = load_labelled_smarts(
            [raw], validated, ec_depth=1, n_samples=None, random_seed=42
        )
        assert "UNLABELLED" not in set(df["smarts"])
        assert len(df) == 10
