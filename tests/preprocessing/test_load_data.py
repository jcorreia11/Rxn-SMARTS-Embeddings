import pandas as pd

from smart_rxn_embeddings.preprocessing.load_data import clean, load_raw, save


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
        assert {"TEMPLATE_ID", "TEMPLATE", "VALID"}.issubset(df.columns)


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
