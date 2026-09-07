import pytest

from rxn_smarts_embeddings.tokenization.smarts_tokenizer import SmartsTokenizer


@pytest.fixture
def t() -> SmartsTokenizer:
    return SmartsTokenizer()


# ---------------------------------------------------------------------------
# Reconstruction invariant — joining tokens must reproduce the original string
# ---------------------------------------------------------------------------

_ROUND_TRIP_CASES = [
    "[C;H1:1]=[N;H0:2]>>[C;H1:1]-[N;H0:2]",
    "([O;H0:1]-[C;H2:2].[O;H0:3]-[C;H1:4])>>([O;H0]=[C;H1:2].[O;H1:3].[C;H2:4]).[O;H1:1]",
    "c1ccccc1>>c1cccnc1",
    "[$(C([OH]))]>>O",  # recursive SMARTS with nested brackets
    "ClCBr",  # two-letter atoms Cl and Br
    "C%10CC%10",  # two-digit ring closure
    "F/C=C/F",  # directional bonds
    "F/?C",  # directional bond /? (up-or-unspecified)
    "F\\?C",  # directional bond \? (down-or-unspecified)
    "C>N>O",  # reactant > agent > product (lone >)
    "[!C;R]>>O",  # logical NOT outside brackets? no — inside ✓
    "*~*",  # wildcard atoms with any-bond
]


class TestRoundTrip:
    @pytest.mark.parametrize("smarts", _ROUND_TRIP_CASES)
    def test_join_tokens_equals_original(self, t, smarts):
        tokens = t.tokenize(smarts)
        assert "".join(tokens) == smarts


# ---------------------------------------------------------------------------
# Bracketed atom expressions
# ---------------------------------------------------------------------------


class TestBracketedAtoms:
    def test_simple_bracket_is_single_token(self, t):
        tokens = t.tokenize("[C:1]")
        assert tokens == ["[C:1]"]

    def test_bracket_with_primitives(self, t):
        tokens = t.tokenize("[C;H1:1]")
        assert tokens == ["[C;H1:1]"]

    def test_multiple_brackets_are_separate_tokens(self, t):
        tokens = t.tokenize("[C:1][N:2]")
        assert tokens == ["[C:1]", "[N:2]"]

    def test_recursive_smarts_single_token(self, t):
        # [$(C([OH]))] must be one token despite nested brackets inside
        tokens = t.tokenize("[$(C([OH]))]")
        assert tokens == ["[$(C([OH]))]"]

    def test_complex_bracket_with_logical_ops(self, t):
        tokens = t.tokenize("[c,n;H1]")
        assert tokens == ["[c,n;H1]"]


# ---------------------------------------------------------------------------
# Reaction arrow and agent separator
# ---------------------------------------------------------------------------


class TestArrows:
    def test_reaction_arrow(self, t):
        assert ">>" in t.tokenize("C>>N")

    def test_reaction_arrow_is_single_token(self, t):
        tokens = t.tokenize(">>")
        assert tokens == [">>"]

    def test_agent_separator(self, t):
        tokens = t.tokenize("C>N>O")
        assert tokens == ["C", ">", "N", ">", "O"]

    def test_arrow_not_split_into_two_gt(self, t):
        tokens = t.tokenize("C>>N")
        assert ">>" in tokens
        assert ">" not in tokens


# ---------------------------------------------------------------------------
# Bond symbols
# ---------------------------------------------------------------------------


class TestBonds:
    @pytest.mark.parametrize(
        "smarts, expected",
        [
            ("C=C", ["C", "=", "C"]),
            ("C#N", ["C", "#", "N"]),
            ("C-O", ["C", "-", "O"]),
            ("c:c", ["c", ":", "c"]),
            ("C~N", ["C", "~", "N"]),
            ("C@C", ["C", "@", "C"]),
            ("F/C", ["F", "/", "C"]),
            ("F\\C", ["F", "\\", "C"]),
            ("F/?C", ["F", "/?", "C"]),
            ("F\\?C", ["F", "\\?", "C"]),
        ],
    )
    def test_bond(self, t, smarts, expected):
        assert t.tokenize(smarts) == expected


# ---------------------------------------------------------------------------
# Atom symbols
# ---------------------------------------------------------------------------


class TestAtoms:
    def test_two_letter_cl(self, t):
        tokens = t.tokenize("Cl")
        assert tokens == ["Cl"]

    def test_two_letter_br(self, t):
        tokens = t.tokenize("Br")
        assert tokens == ["Br"]

    def test_cl_not_split_into_c_and_l(self, t):
        # 'l' is not a valid SMARTS token on its own — Cl must stay together
        tokens = t.tokenize("ClC")
        assert tokens[0] == "Cl"

    def test_organic_subset_atoms(self, t):
        for sym in ("C", "N", "O", "P", "S", "F", "I", "B"):
            assert t.tokenize(sym) == [sym]

    def test_aromatic_atoms(self, t):
        for sym in ("c", "n", "o", "p", "s"):
            assert t.tokenize(sym) == [sym]

    def test_primitive_a_aliphatic(self, t):
        assert t.tokenize("A") == ["A"]

    def test_primitive_a_aromatic(self, t):
        assert t.tokenize("a") == ["a"]

    def test_wildcard(self, t):
        assert t.tokenize("*") == ["*"]


# ---------------------------------------------------------------------------
# Branching and disconnection
# ---------------------------------------------------------------------------


class TestStructure:
    def test_branch_open_close(self, t):
        tokens = t.tokenize("C(O)N")
        assert "(" in tokens
        assert ")" in tokens

    def test_component_grouping(self, t):
        tokens = t.tokenize("(C.N)")
        assert tokens[0] == "("
        assert tokens[-1] == ")"
        assert "." in tokens

    def test_disconnection_dot(self, t):
        tokens = t.tokenize("C.N")
        assert tokens == ["C", ".", "N"]


# ---------------------------------------------------------------------------
# Ring closures
# ---------------------------------------------------------------------------


class TestRingClosures:
    def test_single_digit(self, t):
        tokens = t.tokenize("C1CC1")
        assert tokens == ["C", "1", "C", "C", "1"]

    def test_two_digit(self, t):
        tokens = t.tokenize("C%10CC%10")
        assert tokens == ["C", "%10", "C", "C", "%10"]

    def test_two_digit_not_split_as_percent_and_digits(self, t):
        tokens = t.tokenize("C%12CC%12")
        assert "%12" in tokens
        assert "%" not in tokens


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unmatched_open_bracket_raises(self, t):
        with pytest.raises(ValueError, match="Unmatched"):
            t.tokenize("[C;H1")

    def test_unknown_character_raises(self, t):
        with pytest.raises(ValueError, match="Unexpected character"):
            t.tokenize("C?N")  # bare ? is not valid outside /?  \?

    def test_empty_string_returns_empty_list(self, t):
        assert t.tokenize("") == []
