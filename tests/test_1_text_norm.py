"""Sanity check 1: the shared normalizer."""

import pytest

from text_norm import is_librispeech_style, normalize

CASES = [
    # (input, expected)
    ("Hello, world!", "HELLO WORLD"),
    ("I don't know what you're saying.", "I DON'T KNOW WHAT YOU'RE SAYING"),
    ("It’s Tom’s book.", "IT'S TOM'S BOOK"),
    ("A well-known fact", "A WELL KNOWN FACT"),
    ("In 1990 he left.", "IN NINETEEN NINETY HE LEFT"),
    ("He paid 2500 coins.", "HE PAID TWO THOUSAND FIVE HUNDRED COINS"),
    ("She has 3 cats", "SHE HAS THREE CATS"),
    ("the 3rd of May", "THE THIRD OF MAY"),
    ("the 21st century", "THE TWENTY FIRST CENTURY"),
    ("50% off", "FIFTY PERCENT OFF"),
    ("Mr. Smith and Mrs. Jones met Dr. Brown.", "MISTER SMITH AND MISSUS JONES MET DOCTOR BROWN"),
    ("St. Paul's", "SAINT PAUL'S"),
    ("It costs $5.", "IT COSTS FIVE DOLLARS"),
    ("1,000 men", "ONE THOUSAND MEN"),
    ("3.5 miles", "THREE POINT FIVE MILES"),
    ("Tom & Jerry", "TOM AND JERRY"),
    ("“Quoted,” he said — twice.", "QUOTED HE SAID TWICE"),
    ("  lots   of\tspace \n", "LOTS OF SPACE"),
    ("'Tis the boys' room", "'TIS THE BOYS' ROOM"),
    ("' stray quote '", "STRAY QUOTE"),
    ("", ""),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_normalize(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize("gt", [
    "HE HOPED THERE WOULD BE STEW FOR DINNER",
    "THE BOYS' ROOM AND JAMES'S COAT",
    "'TIS SO",
    "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES",
])
def test_librispeech_identity(gt):
    assert normalize(gt) == gt
    assert is_librispeech_style(gt)


def test_output_alphabet():
    out = normalize("Café naïve résumé #1 @home 100°")
    assert is_librispeech_style(out), out
