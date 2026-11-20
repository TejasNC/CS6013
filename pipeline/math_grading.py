"""Answer matching for free-form math answers (HMMT / AIME style).

Ours, not from the TA repo. `common.py` only grades A-D multiple choice, so
the local math eval grades with this module instead. It is pure Python (no
torch / datasets imports) so it can be unit-tested anywhere.

Matching, in order (first hit wins):
  1. exact string match after light LaTeX normalization
  2. exact rational-number match (handles 0.5 == 1/2 == \\frac{1}{2})
  3. comma-separated answers compared element-wise (order-insensitive)
  4. optional: `math_verify` symbolic equivalence, only if installed
"""

from __future__ import annotations

import re
from fractions import Fraction

_FRAC_RE = re.compile(r"^(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}$")
_SLASH_RE = re.compile(r"^(-?\d+)/(\d+)$")
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def normalize(ans: str) -> str:
    s = ans.strip()
    s = s.strip("$").strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    # Drop \text{...} units that trail a number, e.g. 5\text{ cm} -> 5.
    s = re.sub(r"(?<=[\d}])\s*\\(?:text|mathrm)\{[^{}]*\}$", "", s)
    # Unwrap \text{...}, \mathrm{...}, \textbf{...} (units / words).
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^{}]*)\}", r"\1", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    # Thousands separators in plain integers, e.g. 1,000 or 12,345,678.
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    # \frac12 -> \frac{1}{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    return s


def to_fraction(s: str) -> Fraction | None:
    """Parse a normalized string as an exact rational, else None."""
    m = _FRAC_RE.match(s)
    if m:
        sign, a, b = m.groups()
        if int(b) == 0:
            return None
        v = Fraction(int(a), int(b))
        return -v if sign else v
    m = _SLASH_RE.match(s)
    if m:
        a, b = m.groups()
        return Fraction(int(a), int(b)) if int(b) else None
    if _NUM_RE.match(s):
        return Fraction(s)
    return None


def _split_top_level(s: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _single_match(p: str, g: str) -> bool:
    if p == g:
        return True
    pf, gf = to_fraction(p), to_fraction(g)
    return pf is not None and gf is not None and pf == gf


def is_equiv(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    p, g = normalize(prediction), normalize(gold)
    if not p:
        return False
    if _single_match(p, g):
        return True

    pp, gp = _split_top_level(p), _split_top_level(g)
    if len(pp) > 1 and len(pp) == len(gp):
        remaining = list(gp)
        for x in pp:
            hit = next((y for y in remaining if _single_match(x, y)), None)
            if hit is None:
                break
            remaining.remove(hit)
        else:
            return True

    try:  # optional symbolic fallback
        from math_verify import parse, verify  # type: ignore

        return bool(verify(parse(f"${g}$"), parse(f"${p}$")))
    except Exception:  # noqa: BLE001 - package missing or unparsable
        return False
