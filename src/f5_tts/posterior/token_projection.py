"""Project ASR posterior token ids into the F5-TTS text vocabulary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from f5_tts.posterior.schema import PosteriorTokenMap


def project_asr_token_to_f5_id(
    token_id: int,
    *,
    token_map: PosteriorTokenMap | None = None,
    f5_vocab: Mapping[str, int] | None = None,
    blank_value: int = -1,
    unknown_value: int = 0,
) -> int:
    """Map one ASR token id to an F5 vocab id.

    F5 uses ``-1`` as the filler/padding token before ``TextEmbedding`` shifts
    hard ids by one. CTC blank/filler and tokenizer special tokens therefore map
    to ``blank_value`` by default.
    """

    token_id = int(token_id)
    if token_map is None or f5_vocab is None:
        return token_id

    if token_id in {token_map.blank_id, token_map.filler_id}:
        return blank_value

    if token_id < 0 or token_id >= len(token_map.tokens):
        return blank_value

    token = token_map.tokens[token_id]
    if token == "|":
        token = " "
    elif token.startswith("<") and token.endswith(">"):
        return blank_value

    return f5_vocab.get(token, f5_vocab.get(token.lower(), unknown_value))


def project_topk_token_ids(
    token_id_rows: Sequence[Sequence[int]],
    *,
    token_map: PosteriorTokenMap | None = None,
    f5_vocab: Mapping[str, int] | None = None,
    blank_value: int = -1,
    unknown_value: int = 0,
) -> list[list[int]]:
    """Project a top-k posterior id matrix into F5 ids."""

    return [
        [
            project_asr_token_to_f5_id(
                token_id,
                token_map=token_map,
                f5_vocab=f5_vocab,
                blank_value=blank_value,
                unknown_value=unknown_value,
            )
            for token_id in row
        ]
        for row in token_id_rows
    ]
