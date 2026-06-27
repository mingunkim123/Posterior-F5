"""WER/CER edit-distance breakdown helpers."""

from __future__ import annotations

from dataclasses import dataclass


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


def wer_breakdown(reference: str, hypothesis: str) -> ErrorBreakdown:
    return edit_breakdown(reference.split(), hypothesis.split())


def cer_breakdown(reference: str, hypothesis: str) -> ErrorBreakdown:
    return edit_breakdown(list(reference), list(hypothesis))
