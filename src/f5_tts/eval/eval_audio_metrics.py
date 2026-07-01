"""Evaluate generated-audio metrics for Posterior-F5 runs.

This stage owns metrics that need audio files instead of ASR text:
generated duration, real-time factor, speaker similarity, and UTMOS.
Speaker similarity and UTMOS are explicit opt-in model stages so smoke tests
can still validate the artifact contract without downloading heavyweight
models.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute generated-audio metrics for one Posterior-F5 mode.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--generation_metadata", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per_utterance_output", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--repo_root", default="")
    parser.add_argument("--manifest_base_dir", default="")
    parser.add_argument("--run_speaker_similarity", action="store_true")
    parser.add_argument("--speaker_checkpoint", default="")
    parser.add_argument("--speaker_device", default=None)
    parser.add_argument("--speaker_feat_type", default="wavlm_large")
    parser.add_argument("--run_utmos", action="store_true")
    parser.add_argument("--utmos_device", default=None)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_path(value: str | None, *, bases: list[Path]) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (bases[0] / path).resolve() if bases else path


def audio_duration_sec(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / float(info.samplerate) if info.samplerate else None
    except Exception:
        try:
            with wave.open(str(path), "rb") as wav:
                rate = wav.getframerate()
                return wav.getnframes() / float(rate) if rate else None
        except Exception:
            return None


def _numeric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else None


class SpeakerSimilarityScorer:
    def __init__(self, *, checkpoint: str, device: str | None, feat_type: str) -> None:
        if not checkpoint:
            raise ValueError("--speaker_checkpoint is required when --run_speaker_similarity is set")

        import torch
        import torch.nn.functional as F
        import torchaudio

        from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

        self.torch = torch
        self.F = F
        self.torchaudio = torchaudio
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type=feat_type, config_path=None)
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _load_audio(self, path: Path):
        wav, sr = self.torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(self.device)
        if sr != 16000:
            wav = self.torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(self.device)(wav)
        return wav

    def score(self, generated_audio: Path, ref_audio: Path) -> float:
        wav_generated = self._load_audio(generated_audio)
        wav_ref = self._load_audio(ref_audio)
        with self.torch.no_grad():
            generated_embedding = self.model(wav_generated)
            ref_embedding = self.model(wav_ref)
        return float(self.F.cosine_similarity(generated_embedding, ref_embedding)[0].item())


class UtmosScorer:
    def __init__(self, *, device: str | None) -> None:
        import librosa
        import torch

        self.librosa = librosa
        self.torch = torch
        has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "xpu" if has_xpu else "cpu"))
        self.predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(self.device)
        self.predictor.eval()

    def score(self, generated_audio: Path) -> float:
        wav, sr = self.librosa.load(generated_audio, sr=None, mono=True)
        wav_tensor = self.torch.from_numpy(wav).to(self.device).unsqueeze(0)
        with self.torch.no_grad():
            score = self.predictor(wav_tensor, sr)
        return float(score.item())


def _utterance_id(row: dict[str, Any], index: int) -> str:
    audio = row.get("ref_audio") or row.get("audio_path") or ""
    return str(row.get("utterance_id") or Path(str(audio)).stem or f"utt-{index:06d}")


def evaluate_audio_metrics(
    *,
    manifest_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    run_root: Path,
    mode: str,
    repo_root: Path | None = None,
    manifest_base_dir: Path | None = None,
    speaker_scorer: SpeakerSimilarityScorer | None = None,
    utmos_scorer: UtmosScorer | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bases = [run_root]
    if manifest_base_dir is not None:
        bases.append(manifest_base_dir)
    if repo_root is not None:
        bases.append(repo_root)

    generations = {str(row.get("utterance_id")): row for row in generation_rows if row.get("utterance_id") is not None}
    rows: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(manifest_rows):
        utterance_id = _utterance_id(manifest_row, index)
        generation_row = generations.get(utterance_id, {})
        generated_audio = resolve_path(
            generation_row.get("output") or f"generated/{mode}/{utterance_id}.wav",
            bases=[run_root],
        )
        ref_audio = resolve_path(str(manifest_row.get("ref_audio") or manifest_row.get("audio_path") or ""), bases=bases)
        elapsed_sec = generation_row.get("elapsed_sec")
        duration = audio_duration_sec(generated_audio) if generated_audio else None
        row: dict[str, Any] = {
            "utterance_id": utterance_id,
            "mode": mode,
            "subset": manifest_row.get("subset") or "all",
            "generated_audio": str(generated_audio.relative_to(run_root)) if generated_audio and generated_audio.exists() else str(generated_audio or ""),
            "ref_audio": str(ref_audio) if ref_audio else "",
            "generation_elapsed_sec": elapsed_sec,
            "generated_audio_duration_sec": duration,
            "rtf": float(elapsed_sec) / duration if isinstance(elapsed_sec, (int, float)) and duration and duration > 0 else None,
            "speaker_similarity": None,
            "spk_sim": None,
            "utmos": None,
            "status": "succeeded" if generated_audio and generated_audio.exists() else "missing_audio",
        }

        if row["status"] == "succeeded" and speaker_scorer is not None:
            if ref_audio is None or not ref_audio.exists():
                raise FileNotFoundError(f"Missing reference audio for speaker similarity: {ref_audio}")
            score = speaker_scorer.score(generated_audio, ref_audio)
            row["speaker_similarity"] = score
            row["spk_sim"] = score
        if row["status"] == "succeeded" and utmos_scorer is not None:
            row["utmos"] = utmos_scorer.score(generated_audio)

        rows.append(row)

    num_audio = sum(1 for row in rows if row.get("status") == "succeeded")
    summary = {
        "mode": mode,
        "status": "succeeded",
        "num_utterances": len(rows),
        "num_audio": num_audio,
        "audio_metric_coverage": num_audio / max(len(rows), 1),
        "generated_audio_duration_sec_mean": _numeric_mean(rows, "generated_audio_duration_sec"),
        "rtf_mean": _numeric_mean(rows, "rtf"),
        "speaker_similarity_mean": _numeric_mean(rows, "speaker_similarity"),
        "spk_sim_mean": _numeric_mean(rows, "speaker_similarity"),
        "utmos_mean": _numeric_mean(rows, "utmos"),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    speaker_scorer = (
        SpeakerSimilarityScorer(
            checkpoint=args.speaker_checkpoint,
            device=args.speaker_device,
            feat_type=args.speaker_feat_type,
        )
        if args.run_speaker_similarity
        else None
    )
    utmos_scorer = UtmosScorer(device=args.utmos_device) if args.run_utmos else None
    summary, rows = evaluate_audio_metrics(
        manifest_rows=read_jsonl(args.manifest),
        generation_rows=read_jsonl(args.generation_metadata),
        run_root=Path(args.run_root).expanduser().resolve(),
        mode=args.mode,
        repo_root=Path(args.repo_root).expanduser().resolve() if args.repo_root else None,
        manifest_base_dir=Path(args.manifest_base_dir).expanduser().resolve() if args.manifest_base_dir else None,
        speaker_scorer=speaker_scorer,
        utmos_scorer=utmos_scorer,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_utterance_path = Path(args.per_utterance_output)
    per_utterance_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, summary)
    write_jsonl(per_utterance_path, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
