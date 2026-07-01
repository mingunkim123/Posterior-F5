"""WER/CER edit-distance breakdown helpers.

The dashboard and paper exports rely on these functions for comparable metrics.
Keep text normalization explicit so a run can report how WER/CER were produced.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import string
import unicodedata


_ASCII_PUNCTUATION = str.maketrans({char: " " for char in string.punctuation})
_UNICODE_PUNCTUATION_RE = re.compile(r"[\u2000-\u206F\u2E00-\u2E7F\u3000-\u303F\uFE10-\uFE1F\uFE30-\uFE4F\uFF00-\uFF65]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str | None, profile: str = "paper") -> str:
    """Normalize text before metric calculation.

    Profiles:
    - ``paper``: NFKC, casefold, remove punctuation, collapse whitespace.
    - ``lowercase``: NFKC, casefold, collapse whitespace.
    - ``none``: NFKC and whitespace cleanup only.
    """

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    if profile not in {"paper", "lowercase", "none"}:
        raise ValueError(f"Unknown normalization profile: {profile}")
    if profile in {"paper", "lowercase"}:
        normalized = normalized.casefold()
    if profile == "paper":
        normalized = normalized.translate(_ASCII_PUNCTUATION)
        normalized = _UNICODE_PUNCTUATION_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def word_tokens(text: str | None, profile: str = "paper") -> list[str]:
    normalized = normalize_text(text, profile=profile)
    return normalized.split() if normalized else []


def char_tokens(text: str | None, profile: str = "paper", *, remove_spaces: bool = True) -> list[str]:
    normalized = normalize_text(text, profile=profile)
    if remove_spaces:
        normalized = normalized.replace(" ", "")
    return list(normalized)


@dataclass
class ErrorBreakdown:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / max(self.reference_length, 1)


def edit_breakdown(reference: list[str], hypothesis: list[str]) -> ErrorBreakdown:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[None] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, cols):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                choices = [(dp[i - 1][j - 1], "ok")]
            else:
                choices = [(dp[i - 1][j - 1] + 1, "sub")]
            choices.extend([(dp[i - 1][j] + 1, "del"), (dp[i][j - 1] + 1, "ins")])
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    i, j = len(reference), len(hypothesis)
    sub = delete = insert = 0
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "ok":
            i -= 1
            j -= 1
        elif op == "sub":
            sub += 1
            i -= 1
            j -= 1
        elif op == "del":
            delete += 1
            i -= 1
        elif op == "ins":
            insert += 1
            j -= 1
        else:
            break

    return ErrorBreakdown(sub, delete, insert, len(reference))


def wer_breakdown(reference: str, hypothesis: str, profile: str = "paper") -> ErrorBreakdown:
    return edit_breakdown(word_tokens(reference, profile=profile), word_tokens(hypothesis, profile=profile))


def cer_breakdown(reference: str, hypothesis: str, profile: str = "paper", *, remove_spaces: bool = True) -> ErrorBreakdown:
    return edit_breakdown(
        char_tokens(reference, profile=profile, remove_spaces=remove_spaces),
        char_tokens(hypothesis, profile=profile, remove_spaces=remove_spaces),
    )
