"""Offline self-test for math_grading.py. Run: python pipeline/test_math_grading.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_grading import is_equiv

CASES = [
    # (prediction, gold, expected)
    ("42", "42", True),
    ("$42$", "42", True),
    ("42.", "42", True),
    ("43", "42", False),
    ("1,000", "1000", True),
    ("\\dfrac{1}{2}", "\\frac{1}{2}", True),
    ("0.5", "\\frac{1}{2}", True),
    ("1/2", "\\frac{1}{2}", True),
    ("\\frac12", "\\frac{1}{2}", True),
    ("-\\frac{3}{4}", "\\frac{-3}{4}", True),
    ("\\frac{1}{3}", "0.33", False),
    ("90^\\circ", "90", True),
    ("5\\text{ cm}", "5", True),
    ("1,2,3", "3,2,1", True),
    ("1,2,3", "1,2,4", False),
    ("1,2", "1,2,3", False),
    ("\\left(1,2\\right)", "(1,2)", True),
    ("2\\sqrt{3}", "2\\sqrt{3}", True),
    ("2\\sqrt{3}", "3\\sqrt{2}", False),
    (None, "42", False),
    ("", "42", False),
]


def main() -> int:
    bad = 0
    for pred, gold, want in CASES:
        got = is_equiv(pred, gold)
        if got != want:
            bad += 1
            print(f"FAIL: is_equiv({pred!r}, {gold!r}) = {got}, expected {want}")
    print(f"{len(CASES) - bad}/{len(CASES)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
