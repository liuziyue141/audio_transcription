"""Shared text normalizer: any text -> LibriSpeech convention.

Output alphabet: uppercase A-Z, apostrophe, single spaces.

Used for (1) Whisper pseudo-labels, (2) Whisper reference transcripts,
(3) an identity assertion on every LibriSpeech ground-truth row.

Order of operations (plan order, plus abbreviation/symbol expansion that the
plan review added as step 1b, logged as a deviation):
  1.  Unicode NFKC; curly apostrophes -> '
  1b. Expand common abbreviations (Mr. -> MISTER, ...) and symbols ($, %, &)
  2.  Digits -> words (4-digit 1100-2099 = year, ordinals, decimals, cardinals)
  3.  Uppercase
  4.  Hyphens -> spaces
  5.  Drop every character except A-Z, space, apostrophe
  6.  Collapse whitespace, strip; drop tokens made only of apostrophes
"""

import re
import unicodedata

from num2words import num2words

_APOSTROPHES = {
    "‘": "'", "’": "'", "‛": "'", "ʼ": "'",
    "′": "'", "`": "'", "´": "'",
}
_DASHES = ["‐", "‑", "‒", "–", "—", "―", "−"]

# LibriSpeech spells these out. Only matched WITH a trailing period, so the
# ground-truth text (which has no periods) is never touched.
_ABBREVIATIONS = {
    "mr": "mister",
    "mrs": "missus",
    "ms": "miss",
    "dr": "doctor",
    "st": "saint",
    "jr": "junior",
    "sr": "senior",
    "mt": "mount",
    "capt": "captain",
    "col": "colonel",
    "gen": "general",
    "lt": "lieutenant",
    "sgt": "sergeant",
    "prof": "professor",
    "rev": "reverend",
    "vs": "versus",
    "etc": "et cetera",
}
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\.",
    flags=re.IGNORECASE,
)

_CURRENCY_RE = re.compile(r"([$£€])\s?(\d[\d,]*(?:\.\d+)?)")
_CURRENCY_WORD = {"$": "dollars", "£": "pounds", "€": "euros"}
_PERCENT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?%")
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", flags=re.IGNORECASE)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_DECIMAL_RE = re.compile(r"\b(\d+)\.(\d+)\b")
_NUMBER_RE = re.compile(r"\d+")


def _cardinal(n: int) -> str:
    return num2words(n)


def _number_to_words(token: str) -> str:
    """A run of digits -> words. 4-digit 1100..2099 are read as years."""
    n = int(token)
    if len(token) == 4 and 1100 <= n <= 2099:
        return num2words(n, to="year")
    return _cardinal(n)


def _decimal_to_words(m: re.Match) -> str:
    whole, frac = m.group(1), m.group(2)
    digits = " ".join(_cardinal(int(d)) for d in frac)
    return f"{_cardinal(int(whole))} point {digits}"


def _expand_numbers(text: str) -> str:
    # currency first ("$5" -> "5 dollars"), then percent, ordinals, decimals
    def currency(m):
        amount = m.group(2).replace(",", "")
        return f"{amount} {_CURRENCY_WORD[m.group(1)]}"

    text = _CURRENCY_RE.sub(currency, text)
    text = _PERCENT_RE.sub(lambda m: m.group(1) + " percent", text)
    text = _THOUSANDS_RE.sub("", text)
    text = _ORDINAL_RE.sub(lambda m: num2words(int(m.group(1)), to="ordinal"), text)
    text = _DECIMAL_RE.sub(_decimal_to_words, text)
    text = _NUMBER_RE.sub(lambda m: " " + _number_to_words(m.group(0)) + " ", text)
    return text


def normalize(text: str) -> str:
    """Map arbitrary transcript text to the LibriSpeech convention."""
    if text is None:
        return ""
    # 1. Unicode + apostrophes + dashes
    text = unicodedata.normalize("NFKC", text)
    for k, v in _APOSTROPHES.items():
        text = text.replace(k, v)
    for d in _DASHES:
        text = text.replace(d, "-")
    # 1b. abbreviations and symbols
    text = _ABBREV_RE.sub(lambda m: _ABBREVIATIONS[m.group(1).lower()], text)
    text = text.replace("&", " and ")
    # 2. numbers
    text = _expand_numbers(text)
    # 3. uppercase
    text = text.upper()
    # 4. hyphens -> spaces
    text = text.replace("-", " ")
    # 5. keep only A-Z, space, apostrophe
    text = re.sub(r"[^A-Z' ]+", " ", text)
    # 6. collapse whitespace; drop apostrophe-only tokens (stray quote marks)
    tokens = [t for t in text.split() if t.strip("'")]
    return " ".join(tokens)


def is_librispeech_style(text: str) -> bool:
    return re.fullmatch(r"[A-Z']+( [A-Z']+)*", text or "") is not None


if __name__ == "__main__":
    import sys

    for line in sys.stdin:
        print(normalize(line.rstrip("\n")))
