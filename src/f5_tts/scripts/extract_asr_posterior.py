"""Extract ASR 1-best transcripts and optional CTC top-k posterior caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.append(str(Path(__file__).resolve().parents[2]))

from f5_tts.posterior.length import expected_text_len
from f5_tts.posterior.normalize import entropy
from f5_tts.posterior.schema import Hypothesis, PosteriorTokenMap, PosteriorUtterance, TopKPosterior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a posterior cache JSONL manifest from reference audio files.",
    )
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest or plain text audio path list.")
    parser.add_argument("--output", required=True, help="Output posterior JSONL manifest.")
    parser.add_argument("--shard_dir", help="Directory for optional CTC .npz posterior shards.")
    parser.add_argument("--language", help="Optional ASR language hint.")
    parser.add_argument("--whisper_model", default="openai/whisper-large-v3-turbo", help="Whisper ASR model id.")
    parser.add_argument("--skip_whisper", action="store_true", help="Skip Whisper 1-best transcription.")
    parser.add_argument("--ctc_model", help="Optional CTC ASR model id for frame-level top-k posterior extraction.")
    parser.add_argument("--top_k", type=int, default=8, help="Top-k posterior entries to store per CTC frame.")
    parser.add_argument("--device", default=None, help="ASR device, e.g. cuda:0, mps, or cpu.")
    parser.add_argument("--torch_dtype", choices=["float16", "float32"], default="float32")
    return parser.parse_args()


def iter_manifest(path: str | Path) -> list[dict[str, Any]]:
    entries = []
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                entry = json.loads(line)
            else:
                entry = {"audio_path": line}

            audio_path = entry["audio_path"]
            entry.setdefault("utterance_id", Path(audio_path).stem or f"utt-{index:06d}")
            entries.append(entry)

    return entries


def load_whisper_pipeline(model_id: str, device: str | None, torch_dtype: str):
    from transformers import pipeline
    import torch

    dtype = torch.float16 if torch_dtype == "float16" else torch.float32
    return pipeline("automatic-speech-recognition", model=model_id, torch_dtype=dtype, device=device)


def transcribe_with_whisper(asr_pipe, audio_path: str, language: str | None) -> str:
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language

    result = asr_pipe(
        audio_path,
        chunk_length_s=30,
        batch_size=16,
        generate_kwargs=generate_kwargs,
        return_timestamps=False,
    )
    return result["text"].strip()


def load_ctc_bundle(model_id: str, device: str | None):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(resolved_device).eval()
    return processor, model, resolved_device


def ctc_token_map(processor, model_id: str) -> PosteriorTokenMap:
    vocab = processor.tokenizer.get_vocab()
    tokens = [""] * len(vocab)
    for token, token_id in vocab.items():
        if 0 <= token_id < len(tokens):
            tokens[token_id] = token

    blank_id = getattr(processor.tokenizer, "pad_token_id", None)
    unk_id = getattr(processor.tokenizer, "unk_token_id", None)
    return PosteriorTokenMap(
        source=model_id,
        tokens=tokens,
        token_to_id=vocab,
        blank_id=blank_id,
        filler_id=blank_id,
        unk_id=unk_id,
        metadata={"tokenizer_class": processor.tokenizer.__class__.__name__},
    )


def extract_ctc_topk(audio_path: str, processor, model, device, *, top_k: int, shard_path: Path) -> TopKPosterior:
    import numpy as np
    import torch
    import torchaudio

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T)
    except Exception:
        waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sample_rate = getattr(processor.feature_extractor, "sampling_rate", sample_rate)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
        sample_rate = target_sample_rate

    duration_sec = waveform.shape[-1] / sample_rate
    inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.inference_mode():
        logits = model(input_values).logits[0]
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)

    top_ids_np = top_ids.cpu().numpy().astype(np.int64)
    top_probs_np = top_probs.cpu().numpy().astype(np.float32)
    np.savez(shard_path, token_ids=top_ids_np, probs=top_probs_np)

    frame_rate = float(top_ids_np.shape[0] / duration_sec) if duration_sec > 0 else None
    blank_id = getattr(processor.tokenizer, "pad_token_id", None)
    topk_mass = top_probs_np.sum(axis=-1)
    metadata = {
        "duration_sec": duration_sec,
        "requested_top_k": top_k,
        "stored_top_k": int(top_ids_np.shape[1]),
        "probability_source": "softmax",
        "probability_space": "raw_topk_not_renormalized",
        "topk_mass_min": float(topk_mass.min()) if topk_mass.size else None,
        "topk_mass_mean": float(topk_mass.mean()) if topk_mass.size else None,
        "topk_mass_max": float(topk_mass.max()) if topk_mass.size else None,
        "entropy_normalization": "topk_renormalized",
        "entropy_base": "e",
    }
    return TopKPosterior(
        shard_path=shard_path.name,
        num_frames=int(top_ids_np.shape[0]),
        top_k=int(top_ids_np.shape[1]),
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        blank_id=blank_id,
        source=model.config.name_or_path,
        metadata=metadata,
    )


def mean_row_entropy(probs: list[list[float]]) -> float | None:
    if not probs:
        return None
    return sum(entropy(row) for row in probs) / len(probs)


def main() -> None:
    args = parse_args()
    entries = iter_manifest(args.manifest)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shard_dir) if args.shard_dir else output_path.parent / "posterior_npz"
    if args.ctc_model:
        shard_dir.mkdir(parents=True, exist_ok=True)

    whisper_pipe = None if args.skip_whisper else load_whisper_pipeline(args.whisper_model, args.device, args.torch_dtype)
    ctc_bundle = load_ctc_bundle(args.ctc_model, args.device) if args.ctc_model else None
    token_map = ctc_token_map(ctc_bundle[0], args.ctc_model) if ctc_bundle else None

    with output_path.open("w", encoding="utf-8") as output_file:
        for entry in entries:
            audio_path = entry["audio_path"]
            one_best = entry.get("text")
            if whisper_pipe is not None:
                one_best = transcribe_with_whisper(whisper_pipe, audio_path, args.language)

            hypotheses = [Hypothesis(text=one_best or "", posterior=1.0, metadata={"confidence": 1.0})]
            frame_posteriors = None
            mean_entropy = None
            if ctc_bundle:
                processor, model, device = ctc_bundle
                shard_path = shard_dir / f"{entry['utterance_id']}.npz"
                frame_posteriors = extract_ctc_topk(audio_path, processor, model, device, top_k=args.top_k, shard_path=shard_path)
                token_ids, probs = frame_posteriors.token_ids, frame_posteriors.probs
                if token_ids is None or probs is None:
                    import numpy as np

                    with np.load(shard_path) as shard:
                        probs = shard[frame_posteriors.probs_key].tolist()
                mean_entropy = mean_row_entropy(probs)

            utterance = PosteriorUtterance(
                utterance_id=entry["utterance_id"],
                audio_path=audio_path,
                asr_source=args.ctc_model or (None if args.skip_whisper else args.whisper_model),
                one_best=one_best,
                nbest=hypotheses,
                frame_posteriors=frame_posteriors,
                token_map=token_map,
                expected_ref_len=expected_text_len(hypotheses),
                mean_entropy=mean_entropy,
                duration_sec=frame_posteriors.metadata.get("duration_sec") if frame_posteriors else None,
                language=args.language or entry.get("language"),
                metadata={
                    "input": entry,
                    "entropy_normalization": "topk_renormalized" if frame_posteriors else None,
                    "topk_probability_space": "raw_topk_not_renormalized" if frame_posteriors else None,
                },
            )
            output_file.write(json.dumps(utterance.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
