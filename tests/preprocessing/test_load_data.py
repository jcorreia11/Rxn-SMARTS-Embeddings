import pandas as pd

from smart_rxn_embeddings.preprocessing.load_data import (
    build_reaction_groups,
    clean,
    load_raw,
    main,
    save,
    save_groups,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMARTS = [
    "[C;H1:1]=[N;H0:2]>>[C;H1:1]-[N;H0:2]",
    "[O;H1:1]-[C;H2:2]>>[O;H0:1]=[C;H1:2]",
    "[N;H2:1]-[C;H1:2]>>[N;H1:1]=[C;H0:2]",
    "[c;H1:1]>>[C;H2:1]",
]


def _make_df(n: int = 1, **kwargs) -> pd.DataFrame:
    """Build a minimal DataFrame with the columns load_raw returns.

    ``n`` is inferred from the length of the first kwarg when overrides are
    provided, so callers can write ``_make_df(VALID=[…])`` without repeating n.
    Each row gets a unique SMARTS by default so deduplication doesn't
    silently discard rows under test.
    """
    if kwargs:
        n = len(next(iter(kwargs.values())))
    base = {
        "TEMPLATE_ID": [f"RR:0{i}" for i in range(n)],
        "TEMPLATE": [_SMARTS[i % len(_SMARTS)] for i in range(n)],
        "VALID": ["True"] * n,
        "REACTIONS": [f"RXN:{i}" for i in range(n)],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------


class TestLoadRaw:
    def test_concatenates_multiple_files(self, tmp_path):
        for name in ("a.csv", "b.csv"):
            _make_df().to_csv(tmp_path / name, index=False)
        df = load_raw([str(tmp_path / "a.csv"), str(tmp_path / "b.csv")])
        assert len(df) == 2

    def test_returns_required_columns(self, tmp_path):
        path = tmp_path / "data.csv"
        _make_df().to_csv(path, index=False)
        df = load_raw([str(path)])
        assert {"TEMPLATE_ID", "TEMPLATE", "VALID", "REACTIONS"}.issubset(df.columns)


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


class TestClean:
    def test_keeps_valid_rows(self):
        df = _make_df(VALID=["True", "False", "True"])
        result = clean(df)
        assert len(result) == 2

    def test_removes_empty_template(self):
        df = _make_df(TEMPLATE=["[C:1]>>[O:1]", "", "  "])
        result = clean(df)
        assert len(result) == 1

    def test_removes_nan_template(self):
        df = _make_df(TEMPLATE=["[C:1]>>[O:1]", None])
        result = clean(df)
        assert len(result) == 1

    def test_deduplicates(self):
        df = _make_df(TEMPLATE=["[C:1]>>[O:1]", "[C:1]>>[O:1]", "[N:1]>>[O:1]"])
        result = clean(df)
        assert len(result) == 2

    def test_valid_case_insensitive(self):
        df = _make_df(VALID=["true", "TRUE", "True", "false"])
        result = clean(df)
        assert len(result) == 3

    def test_returns_series_of_smarts_strings(self):
        df = _make_df()
        result = clean(df)
        assert list(result) == [_SMARTS[0]]

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["TEMPLATE_ID", "TEMPLATE", "VALID"])
        result = clean(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestSave:
    def test_writes_one_smarts_per_line(self, tmp_path):
        smarts = pd.Series(["[C:1]>>[O:1]", "[N:1]>>[O:1]"])
        out = tmp_path / "out.txt"
        save(smarts, str(out))
        lines = [line for line in out.read_text().splitlines() if line.strip()]
        assert lines == ["[C:1]>>[O:1]", "[N:1]>>[O:1]"]

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "out.txt"
        save(pd.Series(["[C:1]>>[O:1]"]), str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_produces_output_file(self, tmp_path):
        input_path = tmp_path / "data.csv"
        _make_df(TEMPLATE=["[C:1]>>[O:1]", "[N:1]>>[O:1]"]).to_csv(
            input_path, index=False
        )
        output_path = tmp_path / "out.txt"
        groups_path = tmp_path / "groups.csv"
        main([str(input_path)], str(output_path), str(groups_path))
        assert output_path.exists()
        lines = [ln for ln in output_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_main_produces_groups_file(self, tmp_path):
        input_path = tmp_path / "data.csv"
        _make_df(TEMPLATE=["[C:1]>>[O:1]", "[N:1]>>[O:1]"]).to_csv(
            input_path, index=False
        )
        output_path = tmp_path / "out.txt"
        groups_path = tmp_path / "groups.csv"
        main([str(input_path)], str(output_path), str(groups_path))
        assert groups_path.exists()
        groups_df = pd.read_csv(groups_path)
        assert list(groups_df.columns) == ["smarts", "reaction_group"]
        assert len(groups_df) == 2


# ---------------------------------------------------------------------------
# build_reaction_groups
# ---------------------------------------------------------------------------


class TestBuildReactionGroups:
    def test_returns_smarts_and_group_columns(self):
        df = _make_df()
        result = build_reaction_groups(df)
        assert list(result.columns) == ["smarts", "reaction_group"]

    def test_applies_same_filters_as_clean(self):
        df = _make_df(VALID=["True", "False", "True"])
        result = build_reaction_groups(df)
        assert len(result) == 2

    def test_uses_reactions_field_as_group(self):
        # Distinct TEMPLATE values (dedup key) but two rows share a REACTIONS
        # value — they should be assigned the same group.
        df = _make_df(REACTIONS=["RHEA:1;RHEA:2", "RHEA:1;RHEA:2", "RHEA:9"])
        result = build_reaction_groups(df)
        assert result["reaction_group"].tolist() == ["RHEA:1;RHEA:2", "RHEA:1;RHEA:2", "RHEA:9"]

    def test_falls_back_to_smarts_when_reactions_blank(self):
        df = _make_df(REACTIONS=[None, "  ", "RHEA:9"])
        result = build_reaction_groups(df)
        expected_smarts = result["smarts"].tolist()
        assert result["reaction_group"].tolist() == [
            expected_smarts[0],
            expected_smarts[1],
            "RHEA:9",
        ]

    def test_no_group_is_null(self):
        df = _make_df(REACTIONS=[None, "RHEA:1", None])
        result = build_reaction_groups(df)
        assert result["reaction_group"].notna().all()


# ---------------------------------------------------------------------------
# save_groups
# ---------------------------------------------------------------------------


class TestSaveGroups:
    def test_writes_csv_with_header(self, tmp_path):
        groups = pd.DataFrame(
            {"smarts": ["[C:1]>>[O:1]"], "reaction_group": ["RHEA:1"]}
        )
        out = tmp_path / "groups.csv"
        save_groups(groups, str(out))
        result = pd.read_csv(out)
        assert result.to_dict("records") == [
            {"smarts": "[C:1]>>[O:1]", "reaction_group": "RHEA:1"}
        ]

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "groups.csv"
        save_groups(
            pd.DataFrame({"smarts": ["[C:1]>>[O:1]"], "reaction_group": ["RHEA:1"]}),
            str(out),
        )
        assert out.exists()
