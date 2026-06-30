import json

from f5_tts.eval.make_result_tables import rows_from_summary, write_markdown


def test_rows_from_summary_expands_subset_metrics():
    summary = {
        "modes": [
            {
                "mode": "hard",
                "subsets": [
                    {"subset": "clean", "num_utterances": 2, "wer": 0.1, "cer": 0.05},
                    {"subset": "noisy", "num_utterances": 2, "wer": 0.2, "cer": 0.1},
                ],
            }
        ]
    }

    rows = rows_from_summary(summary)

    assert rows[0]["mode"] == "hard"
    assert rows[0]["subset"] == "clean"
    assert rows[1]["wer"] == 0.2


def test_write_markdown_table(tmp_path):
    output = tmp_path / "table.md"
    write_markdown(output, [{"subset": "all", "mode": "hard", "num_utterances": 1, "wer": 0.125, "cer": 0.25}])

    text = output.read_text(encoding="utf-8")
    assert "| subset | mode |" in text
    assert "| all | hard | 1 | 0.1250 | 0.2500 |" in text
