import math
import struct
import wave

import pytest

from f5_tts.eval.eval_audio_metrics import evaluate_audio_metrics


def _write_wav(path, *, seconds=1.0, sample_rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(int(seconds * sample_rate)):
            sample = int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            wav.writeframes(struct.pack("<h", sample))


def test_audio_metrics_compute_duration_and_rtf(tmp_path):
    run_root = tmp_path / "run"
    wav_path = run_root / "generated" / "hard" / "utt-001.wav"
    wav_path.parent.mkdir(parents=True)
    _write_wav(wav_path, seconds=1.0)

    summary, rows = evaluate_audio_metrics(
        manifest_rows=[{"utterance_id": "utt-001", "ref_audio": str(wav_path), "text": "hello"}],
        generation_rows=[{"utterance_id": "utt-001", "output": "generated/hard/utt-001.wav", "elapsed_sec": 0.5}],
        run_root=run_root,
        mode="hard",
    )

    assert summary["num_audio"] == 1
    assert summary["audio_metric_coverage"] == pytest.approx(1.0)
    assert summary["generated_audio_duration_sec_mean"] == pytest.approx(1.0)
    assert summary["rtf_mean"] == pytest.approx(0.5)
    assert rows[0]["rtf"] == pytest.approx(0.5)
