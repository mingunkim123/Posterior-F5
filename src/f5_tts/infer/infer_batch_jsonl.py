"""Batch JSONL inference runner for Posterior-F5 platform jobs.

This runner keeps the F5 model and vocoder alive across many utterances.  It is
intended for platform workers; the interactive CLI remains in ``infer_cli.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device as default_device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
)
from f5_tts.model.hybrid_reference_conditioner import HybridReferenceConditioner
from f5_tts.model.posterior_encoder import load_posterior_encoder_checkpoint
from f5_tts.model.ssl_reference_encoder import cached_ssl_condition, index_ssl_cache
from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx, list_str_to_tensor
from f5_tts.posterior.io import load_posterior_manifest, load_topk_arrays
from f5_tts.posterior.soft_embedding import expected_embedding_from_topk
from f5_tts.posterior.token_projection import project_topk_token_ids


SUPPORTED_REF_TEXT_MODES = {"hard", "length_only", "soft_ctc", "posterior_encoder", "hybrid"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F5-TTS inference for a JSONL batch.")
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Rows to synthesize. Includes ref_audio/ref_text/gen_text/output.",
    )
    parser.add_argument("--output_jsonl", required=True, help="Generation metadata JSONL to write.")
    parser.add_argument("--run_root", required=True, help="Run artifact root used to resolve relative output paths.")
    parser.add_argument("--mode", required=True, help="Platform mode name, e.g. hard or soft_ctc.")
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--model_cfg", default="")
    parser.add_argument("--ckpt_file", default="")
    parser.add_argument("--vocab_file", default="")
    parser.add_argument("--ref_text_mode", default="hard", choices=sorted(SUPPORTED_REF_TEXT_MODES))
    parser.add_argument("--posterior_file", default="")
    parser.add_argument("--posterior_encoder_ckpt", default="")
    parser.add_argument("--ssl_cache", default="")
    parser.add_argument("--vocoder_name", default=mel_spec_type, choices=["vocos", "bigvgan"])
    parser.add_argument("--target_rms", type=float, default=target_rms)
    parser.add_argument("--cross_fade_duration", type=float, default=cross_fade_duration)
    parser.add_argument("--nfe_step", type=int, default=nfe_step)
    parser.add_argument("--cfg_strength", type=float, default=cfg_strength)
    parser.add_argument("--sway_sampling_coef", type=float, default=sway_sampling_coef)
    parser.add_argument("--speed", type=float, default=speed)
    parser.add_argument("--fix_duration", type=float, default=fix_duration)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_wav_bytes", type=int, default=44)
    parser.add_argument("--remove_silence", action="store_true")
    parser.add_argument("--load_vocoder_from_local", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def output_file_metadata(path: Path, *, min_bytes: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "exists": path.exists(),
        "size_bytes": None,
        "sha256": "",
        "ok": False,
    }
    if not path.exists():
        metadata["error_message"] = f"Missing generated wav: {path}"
        return metadata

    size = path.stat().st_size
    metadata["size_bytes"] = size
    metadata["sha256"] = file_sha256(path)
    metadata["ok"] = size >= min_bytes
    if not metadata["ok"]:
        metadata["error_message"] = f"Generated wav is too small: {path} ({size} bytes < {min_bytes})"
    return metadata


def resolve_example_path(path: str) -> str:
    if "infer/examples/" in path and not Path(path).expanduser().exists():
        return str(files("f5_tts").joinpath(path))
    return path


def resolve_output_path(row: dict[str, Any], *, run_root: Path, mode: str) -> Path:
    output = row.get("output") or f"generated/{mode}/{row['utterance_id']}.wav"
    output_path = Path(str(output))
    if output_path.is_absolute():
        return output_path
    return run_root / output_path


def resolve_checkpoint(model: str, ckpt_file: str) -> tuple[str, str]:
    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
    resolved_model = model

    if model == "F5TTS_Base":
        ckpt_step = 1200000
    elif model == "F5TTS_Base_bigvgan":
        ckpt_step = 1200000
        ckpt_type = "pt"
    elif model == "E2TTS_Base":
        repo_name = "E2-TTS"
        ckpt_step = 1200000

    if not ckpt_file:
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{resolved_model}/model_{ckpt_step}.{ckpt_type}"))
    elif ckpt_file.startswith("hf://"):
        ckpt_file = str(cached_path(ckpt_file))

    return resolved_model, ckpt_file


def load_f5_runtime(args: argparse.Namespace):
    if args.vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    else:
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

    vocoder = load_vocoder(
        vocoder_name=args.vocoder_name,
        is_local=args.load_vocoder_from_local,
        local_path=vocoder_local_path,
        device=args.device,
    )

    model = args.model
    model_cfg = OmegaConf.load(
        args.model_cfg or str(files("f5_tts").joinpath(f"configs/{model}.yaml"))
    )
    if model != "F5TTS_Base":
        assert args.vocoder_name == model_cfg.model.mel_spec.mel_spec_type
    elif args.vocoder_name == "bigvgan":
        model = "F5TTS_Base_bigvgan"

    model, ckpt_file = resolve_checkpoint(model, args.ckpt_file)
    vocab_file = args.vocab_file
    if vocab_file.startswith("hf://"):
        vocab_file = str(cached_path(vocab_file))

    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    ema_model = load_model(
        model_cls,
        model_cfg.model.arch,
        ckpt_file,
        mel_spec_type=args.vocoder_name,
        vocab_file=vocab_file,
        device=args.device,
    )
    return ema_model, vocoder


def resolve_path_for_match(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError):
        return str(Path(path).expanduser())


def matches_ref_audio(utterance, ref_audio_path: str) -> bool:
    ref_path = Path(str(ref_audio_path))
    utterance_audio_path = str(utterance.audio_path)
    utterance_path = Path(utterance_audio_path)

    if utterance_audio_path == str(ref_audio_path):
        return True
    if resolve_path_for_match(utterance_audio_path) == resolve_path_for_match(str(ref_audio_path)):
        return True
    if utterance_path.name and utterance_path.name == ref_path.name:
        return True
    return utterance.utterance_id in {str(ref_audio_path), ref_path.name, ref_path.stem}


def posterior_entry_for_row(row: dict[str, Any], posterior_entries: list[Any], ref_text_mode: str):
    if ref_text_mode == "hard":
        return None

    utterance_id = str(row.get("utterance_id") or "")
    for utterance in posterior_entries:
        if utterance_id and utterance.utterance_id == utterance_id:
            return utterance

    ref_audio = str(row.get("ref_audio") or "")
    for utterance in posterior_entries:
        if matches_ref_audio(utterance, ref_audio):
            return utterance

    print(f"Warning: No posterior entry found for {utterance_id or ref_audio}. Falling back to hard length.")
    return None


def ssl_entry_for_row(row: dict[str, Any], ssl_entries: dict[str, Any], posterior_entry=None):
    if not ssl_entries:
        return None
    if posterior_entry is not None and posterior_entry.utterance_id in ssl_entries:
        return ssl_entries[posterior_entry.utterance_id]

    utterance_id = str(row.get("utterance_id") or "")
    if utterance_id in ssl_entries:
        return ssl_entries[utterance_id]

    ref_path = Path(str(row.get("ref_audio") or ""))
    for entry in ssl_entries.values():
        audio_path = str(entry.get("audio_path") or "")
        if audio_path == str(row.get("ref_audio") or ""):
            return entry
        if Path(audio_path).name == ref_path.name or Path(audio_path).stem == ref_path.stem:
            return entry
    return None


def text_tensor_from_list(model_obj, text, target_device):
    if model_obj.vocab_char_map is not None:
        return list_str_to_idx(text, model_obj.vocab_char_map).to(target_device)
    return list_str_to_tensor(text).to(target_device)


def compressed_soft_ref_embed(token_ids, probs, embedding_weight, hard_ref_embed, *, blend=0.2):
    target_len = hard_ref_embed.shape[1]
    if target_len <= 0:
        return hard_ref_embed

    num_frames = len(token_ids)
    if num_frames == 0:
        return hard_ref_embed

    soft_rows = []
    for index in range(target_len):
        start = round(index * num_frames / target_len)
        end = max(start + 1, round((index + 1) * num_frames / target_len))
        mass_by_token: dict[int, float] = {}
        for id_row, prob_row in zip(token_ids[start:end], probs[start:end]):
            for token_id, prob in zip(id_row, prob_row):
                token_id = int(token_id)
                if token_id < 0:
                    continue
                mass_by_token[token_id] = mass_by_token.get(token_id, 0.0) + max(float(prob), 0.0)

        total = sum(mass_by_token.values())
        if total <= 1e-8:
            soft_rows.append(hard_ref_embed[:, index, :])
            continue

        soft = expected_embedding_from_topk(
            [[token_id for token_id in mass_by_token]],
            [[prob / total for prob in mass_by_token.values()]],
            embedding_weight,
            normalize=False,
        )
        soft_rows.append(hard_ref_embed[:, index, :] * (1.0 - blend) + soft * blend)

    return torch.stack(soft_rows, dim=1)


def soft_ctc_builder_for_entry(
    *,
    utterance,
    ssl_entry,
    ref_text_mode: str,
    posterior_file: str,
    ssl_cache: str,
):
    if ref_text_mode not in {"soft_ctc", "hybrid"} or utterance is None or utterance.frame_posteriors is None:
        return None

    base_dir = Path(posterior_file).expanduser().resolve().parent if posterior_file else None
    hybrid_conditioner = HybridReferenceConditioner() if ref_text_mode == "hybrid" else None

    def builder(model_obj, text, duration, ref_audio_len, ref_text, gen_text, device):
        del gen_text, ref_audio_len
        token_ids, probs = load_topk_arrays(utterance.frame_posteriors, base_dir=base_dir)
        token_ids = project_topk_token_ids(token_ids, token_map=utterance.token_map, f5_vocab=model_obj.vocab_char_map)
        embedding_weight = model_obj.transformer.text_embed.text_embed.weight

        text_tensor = text_tensor_from_list(model_obj, text, embedding_weight.device)
        with torch.inference_mode():
            hard_embed = model_obj.transformer.text_embed(text_tensor, seq_len=duration, drop_text=False).clone()

        ref_token_text = convert_char_to_pinyin([ref_text])[0]
        ref_token_len = min(len(ref_token_text), hard_embed.shape[1])
        ref_soft = compressed_soft_ref_embed(token_ids, probs, embedding_weight, hard_embed[:, :ref_token_len, :])

        replace_len = min(ref_token_len, ref_soft.shape[1], hard_embed.shape[1])
        hard_embed[:, :replace_len, :] = ref_soft[:, :replace_len, :].to(
            device=hard_embed.device, dtype=hard_embed.dtype
        )
        if hybrid_conditioner is not None:
            if ssl_entry is None:
                hard_embed = hybrid_conditioner(hard_embed, hard_embed, alpha=1.0)
            else:
                ssl_condition = cached_ssl_condition(
                    ssl_entry,
                    text_dim=hard_embed.shape[-1],
                    target_len=hard_embed.shape[1],
                    base_dir=Path(ssl_cache).expanduser().resolve().parent if ssl_cache else None,
                    device=hard_embed.device,
                ).to(dtype=hard_embed.dtype)
                entropy = None
                if utterance.mean_entropy is not None:
                    entropy = torch.full(
                        (hard_embed.shape[0], hard_embed.shape[1]),
                        float(utterance.mean_entropy),
                        device=hard_embed.device,
                        dtype=hard_embed.dtype,
                    )
                hard_embed = hybrid_conditioner(hard_embed, ssl_condition, entropy=entropy)
        return hard_embed

    return builder


def posterior_encoder_builder_for_entry(*, utterance, posterior_file: str, posterior_encoder_ckpt: str):
    if utterance is None or utterance.frame_posteriors is None:
        return None
    if not posterior_encoder_ckpt:
        print("Warning: posterior_encoder mode requires --posterior_encoder_ckpt. Falling back to hard text embedding.")
        return None

    base_dir = Path(posterior_file).expanduser().resolve().parent if posterior_file else None
    encoder_cache: dict[str, Any] = {"encoder": None}

    def builder(model_obj, text, duration, ref_audio_len, ref_text, gen_text, device):
        del gen_text, ref_audio_len
        if encoder_cache["encoder"] is None:
            encoder, _ = load_posterior_encoder_checkpoint(posterior_encoder_ckpt, map_location=device)
            encoder_cache["encoder"] = encoder.to(device).eval()

        token_ids, probs = load_topk_arrays(utterance.frame_posteriors, base_dir=base_dir)
        token_ids = project_topk_token_ids(token_ids, token_map=utterance.token_map, f5_vocab=model_obj.vocab_char_map)
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
        prob_tensor = torch.tensor(probs, device=device, dtype=torch.float32).unsqueeze(0)

        text_tensor = text_tensor_from_list(model_obj, text, device)
        with torch.inference_mode():
            hard_embed = model_obj.transformer.text_embed(text_tensor, seq_len=duration, drop_text=False).clone()
            post_hidden = encoder_cache["encoder"](token_tensor, prob_tensor)

        ref_token_text = convert_char_to_pinyin([ref_text])[0]
        ref_token_len = min(len(ref_token_text), hard_embed.shape[1])
        if ref_token_len <= 0:
            return hard_embed

        if post_hidden.shape[1] != ref_token_len:
            post_hidden = torch.nn.functional.interpolate(
                post_hidden.transpose(1, 2),
                size=ref_token_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        replace_len = min(ref_token_len, post_hidden.shape[1], hard_embed.shape[1])
        hard_embed[:, :replace_len, :] = post_hidden[:, :replace_len, :].to(
            device=hard_embed.device, dtype=hard_embed.dtype
        )
        return hard_embed

    return builder


def text_embed_override_builder_for_entry(
    *,
    utterance,
    ssl_entry,
    ref_text_mode: str,
    posterior_file: str,
    posterior_encoder_ckpt: str,
    ssl_cache: str,
):
    if ref_text_mode == "posterior_encoder":
        return posterior_encoder_builder_for_entry(
            utterance=utterance,
            posterior_file=posterior_file,
            posterior_encoder_ckpt=posterior_encoder_ckpt,
        )
    return soft_ctc_builder_for_entry(
        utterance=utterance,
        ssl_entry=ssl_entry,
        ref_text_mode=ref_text_mode,
        posterior_file=posterior_file,
        ssl_cache=ssl_cache,
    )


def run_row(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_root: Path,
    ema_model,
    vocoder,
    posterior_entries: list[Any],
    ssl_entries: dict[str, Any],
) -> dict[str, Any]:
    output_path = resolve_output_path(row, run_root=run_root, mode=args.mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = dict(row)
    result["status"] = "running"
    result["started_at"] = timestamp()
    result["finished_at"] = None
    result["elapsed_sec"] = None
    result["exit_code"] = None

    started_at = datetime.now().astimezone()
    try:
        posterior_entry = posterior_entry_for_row(row, posterior_entries, args.ref_text_mode)
        ssl_entry = ssl_entry_for_row(row, ssl_entries, posterior_entry)
        expected_ref_text_len = posterior_entry.expected_ref_len if posterior_entry is not None else None
        text_embed_override_builder = text_embed_override_builder_for_entry(
            utterance=posterior_entry,
            ssl_entry=ssl_entry,
            ref_text_mode=args.ref_text_mode,
            posterior_file=args.posterior_file,
            posterior_encoder_ckpt=args.posterior_encoder_ckpt,
            ssl_cache=args.ssl_cache,
        )

        ref_audio = resolve_example_path(str(row["ref_audio"]))
        ref_text = str(row.get("ref_text") or "")
        ref_audio, ref_text = preprocess_ref_audio_text(
            ref_audio,
            ref_text,
            transcribe_if_empty=(args.ref_text_mode == "hard"),
        )
        wav, sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            str(row["gen_text"]),
            ema_model,
            vocoder,
            mel_spec_type=args.vocoder_name,
            progress=None,
            target_rms=args.target_rms,
            cross_fade_duration=args.cross_fade_duration,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            speed=args.speed,
            fix_duration=args.fix_duration,
            device=args.device,
            expected_ref_text_len=expected_ref_text_len,
            text_embed_override_builder=text_embed_override_builder,
            seed=args.seed,
        )
        if wav is None:
            raise ValueError("No audio was generated")

        sf.write(str(output_path), wav, sample_rate)
        if args.remove_silence:
            remove_silence_for_generated_wav(str(output_path))

        metadata = output_file_metadata(output_path, min_bytes=args.min_wav_bytes)
        result["output_exists"] = metadata["exists"]
        result["output_size_bytes"] = metadata["size_bytes"]
        result["output_sha256"] = metadata["sha256"]
        if metadata["ok"]:
            result["status"] = "succeeded"
            result["exit_code"] = 0
        else:
            result["status"] = "failed"
            result["exit_code"] = 1
            result["error_message"] = metadata["error_message"]
    except Exception as exc:
        result["status"] = "failed"
        result["exit_code"] = 1
        result["error_message"] = str(exc)
        metadata = output_file_metadata(output_path, min_bytes=args.min_wav_bytes)
        result["output_exists"] = metadata["exists"]
        result["output_size_bytes"] = metadata["size_bytes"]
        result["output_sha256"] = metadata["sha256"]
    finally:
        finished_at = datetime.now().astimezone()
        result["finished_at"] = finished_at.isoformat(timespec="seconds")
        result["elapsed_sec"] = max((finished_at - started_at).total_seconds(), 0.0)

    return result


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser()
    input_path = Path(args.input_jsonl).expanduser()
    output_path = Path(args.output_jsonl).expanduser()

    rows = load_jsonl(input_path)
    if not rows:
        write_jsonl(output_path, [])
        return 0

    posterior_entries = []
    if args.ref_text_mode != "hard" and args.posterior_file:
        posterior_entries = load_posterior_manifest(args.posterior_file)
    elif args.ref_text_mode != "hard":
        print("Warning: ref_text_mode is not hard, but no --posterior_file was provided. Falling back to hard length.")

    ssl_entries = index_ssl_cache(args.ssl_cache) if args.ssl_cache else {}

    print(f"Loading F5 runtime once for mode={args.mode}, rows={len(rows)}, device={args.device}")
    ema_model, vocoder = load_f5_runtime(args)
    print("Runtime loaded; starting batch inference")

    results: list[dict[str, Any]] = []
    exit_code = 0
    for row in rows:
        result = run_row(
            row,
            args=args,
            run_root=run_root,
            ema_model=ema_model,
            vocoder=vocoder,
            posterior_entries=posterior_entries,
            ssl_entries=ssl_entries,
        )
        results.append(result)
        write_jsonl(output_path, results)
        print(f"{result['utterance_id']} {result['status']} elapsed={result['elapsed_sec']}")
        if result["status"] != "succeeded":
            exit_code = 1
            break

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
