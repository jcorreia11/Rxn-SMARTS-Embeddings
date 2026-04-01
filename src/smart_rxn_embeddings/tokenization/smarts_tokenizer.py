"""Rule-based SMARTS tokenizer following Daylight SMARTS grammar."""

import re

# Handles all tokens *except* bracketed atom expressions.
# Order matters — first alternative that matches wins.
_REST_TOKEN_RE = re.compile(
    r">>"  # reaction arrow  (before lone >)
    r"|Cl|Br"  # two-letter organic-subset atoms  (before single-letter)
    r"|[BCNOPSFIbcnopsAa*]"  # one-letter atoms: aliphatic B C N O P S F I
    #                   aromatic   b c n o p s
    #                   primitives A (aliphatic) a (aromatic)
    #                   wildcard   *
    r"|/\?|\\\?"  # directional bonds with "or unspecified": /? \?  (before / \)
    r"|[-=#~:@/\\]"  # bond symbols: - = # ~ : @ / \
    r"|\(|\)"  # branching / component grouping
    r"|\."  # disconnection / component separator
    r"|%\d{2}"  # two-digit ring closure %nn  (before single digit)
    r"|\d"  # single-digit ring closure
    r"|[!&,;]"  # logical operators that may appear outside brackets
    r"|>"  # lone > — agent separator in R>A>P notation
)


class SmartsTokenizer:
    """Tokenize reaction SMARTS strings into chemically meaningful tokens.

    Bracket expressions such as ``[C;H1:1]`` are returned as single tokens,
    preserving all SMARTS primitives and logical operators inside them.
    Recursive SMARTS with nested brackets, e.g. ``[$(C([OH]))]``, are handled
    correctly by tracking bracket depth rather than relying on a simple regex.

    Examples
    --------
    >>> t = SmartsTokenizer()
    >>> t.tokenize("[C;H1:1]=[N;H0:2]>>[C;H1:1]-[N;H0:2]")
    ['[C;H1:1]', '=', '[N;H0:2]', '>>', '[C;H1:1]', '-', '[N;H0:2]']
    """

    def tokenize(self, smarts: str) -> list[str]:
        """Split *smarts* into a list of SMARTS tokens.

        Raises
        ------
        ValueError
            If any character cannot be matched (invalid SMARTS syntax).
        """
        tokens: list[str] = []
        i = 0
        n = len(smarts)
        while i < n:
            if smarts[i] == "[":
                j = self._find_closing_bracket(smarts, i)
                tokens.append(smarts[i : j + 1])
                i = j + 1
            else:
                m = _REST_TOKEN_RE.match(smarts, i)
                if m is None:
                    raise ValueError(
                        f"Unexpected character {smarts[i]!r} at position {i} "
                        f"in SMARTS {smarts!r}"
                    )
                tokens.append(m.group())
                i = m.end()
        return tokens

    @staticmethod
    def _find_closing_bracket(smarts: str, start: int) -> int:
        """Return the index of the ``]`` that closes the ``[`` at *start*.

        Tracks bracket depth so that recursive SMARTS like ``[$(C([OH]))]``
        are handled without false early termination.
        """
        depth = 0
        for i in range(start, len(smarts)):
            if smarts[i] == "[":
                depth += 1
            elif smarts[i] == "]":
                depth -= 1
                if depth == 0:
                    return i
        raise ValueError(f"Unmatched '[' at position {start} in SMARTS {smarts!r}")
