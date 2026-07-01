import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
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
from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx, list_str_to_tensor
from f5_tts.model.hybrid_reference_conditioner import HybridReferenceConditioner
from f5_tts.model.posterior_encoder import load_posterior_encoder_checkpoint
from f5_tts.model.ssl_reference_encoder import cached_ssl_condition, index_ssl_cache
from f5_tts.posterior.io import load_posterior_manifest, load_topk_arrays
from f5_tts.posterior.soft_embedding import expected_embedding_from_topk


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "--ref_text_mode",
    type=str,
    choices=["hard", "length_only", "soft_ctc", "posterior_encoder", "hybrid"],
    help="Reference text conditioning mode, default hard.",
)
parser.add_argument(
    "--posterior_file",
    type=str,
    help="JSONL posterior manifest used by length_only mode.",
)
parser.add_argument(
    "--posterior_encoder_ckpt",
    type=str,
    help="Posterior encoder checkpoint for posterior_encoder mode.",
)
parser.add_argument(
    "--ssl_cache",
    type=str,
    help="JSONL SSL feature cache for hybrid mode.",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--no_legacy_text",
    action="store_false",
    help="Not to use lossy ASCII transliterations of unicode text in saved file names.",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Optional random seed for deterministic sampling.",
)
args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "F5TTS_v1_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature, others call me mother nature.")
)
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = args.gen_file or config.get("gen_file", "")
ref_text_mode = args.ref_text_mode or config.get("ref_text_mode", "hard")
posterior_file = args.posterior_file or config.get("posterior_file", "")
posterior_encoder_ckpt = args.posterior_encoder_ckpt or config.get("posterior_encoder_ckpt", "")
ssl_cache = args.ssl_cache or config.get("ssl_cache", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)  # no_legacy_text is a store_false arg
if save_chunk and use_legacy_text:
    print(
        "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
    )

remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)
seed = args.seed if args.seed is not None else config.get("seed", None)


# patches for pip pkg user
if "infer/examples/" in ref_audio and not Path(ref_audio).expanduser().exists():
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file and not Path(gen_file).expanduser().exists():
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio and not Path(voice_ref_audio).expanduser().exists():
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))

if ref_text_mode not in {"hard", "length_only", "soft_ctc", "posterior_encoder", "hybrid"}:
    raise ValueError(f"Unsupported ref_text_mode: {ref_text_mode}")

posterior_entries = []
if ref_text_mode != "hard":
    if posterior_file:
        posterior_entries = load_posterior_manifest(posterior_file)
    else:
        print("Warning: ref_text_mode is not hard, but no --posterior_file was provided. Falling back to hard length.")

ssl_entries = index_ssl_cache(ssl_cache) if ssl_cache else {}


def _resolve_path_for_match(path):
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError):
        return str(Path(path).expanduser())


def _matches_ref_audio(utterance, ref_audio_path):
    ref_audio_path = str(ref_audio_path)
    ref_path = Path(ref_audio_path)
    utterance_audio_path = str(utterance.audio_path)
    utterance_path = Path(utterance_audio_path)

    if utterance_audio_path == ref_audio_path:
        return True
    if _resolve_path_for_match(utterance_audio_path) == _resolve_path_for_match(ref_audio_path):
        return True
    if utterance_path.name and utterance_path.name == ref_path.name:
        return True
    return utterance.utterance_id in {ref_audio_path, ref_path.name, ref_path.stem}


def _posterior_entry_for_audio(ref_audio_path):
    if ref_text_mode == "hard":
        return None

    for utterance in posterior_entries:
        if _matches_ref_audio(utterance, ref_audio_path):
            return utterance

    print(f"Warning: No posterior entry found for {ref_audio_path}. Falling back to hard length.")
    return None


def _ssl_entry_for_audio(ref_audio_path, posterior_entry=None):
    if not ssl_entries:
        return None
    if posterior_entry is not None and posterior_entry.utterance_id in ssl_entries:
        return ssl_entries[posterior_entry.utterance_id]
    ref_path = Path(str(ref_audio_path))
    for entry in ssl_entries.values():
        audio_path = str(entry.get("audio_path") or "")
        if audio_path == str(ref_audio_path):
            return entry
        if Path(audio_path).name == ref_path.name or Path(audio_path).stem == ref_path.stem:
            return entry
    return None


def _expected_ref_text_len_for_entry(utterance):
    if utterance is None:
        return None
    return utterance.expected_ref_len


def _text_tensor_from_list(model_obj, text, device):
    if model_obj.vocab_char_map is not None:
        return list_str_to_idx(text, model_obj.vocab_char_map).to(device)
    return list_str_to_tensor(text).to(device)


def _project_asr_topk_to_f5_ids(token_ids, token_map, vocab_char_map):
    if token_map is None or vocab_char_map is None:
        return token_ids

    source_tokens = token_map.tokens

    def _project_one(token_id):
        if token_id in {token_map.blank_id, token_map.filler_id}:
            return -1
        if token_id < 0 or token_id >= len(source_tokens):
            return -1

        token = source_tokens[token_id]
        if token == "|":
            token = " "
        elif token.startswith("<") and token.endswith(">"):
            return -1

        return vocab_char_map.get(token, vocab_char_map.get(token.lower(), 0))

    return [[_project_one(int(token_id)) for token_id in row] for row in token_ids]


def _compressed_soft_ref_embed(token_ids, probs, embedding_weight, hard_ref_embed, *, blend=0.2):
    """Compress frame-level CTC posterior to ref-token positions.

    F5's text embedding positions are token positions padded to the mel length,
    not mel-frame-aligned text positions. Replacing CTC frames directly would
    overwrite the target text tokens, so this keeps the generated text path
    intact and only nudges the reference-token region.
    """

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
        mass_by_token = {}
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


def _soft_ctc_builder_for_entry(utterance, ssl_entry=None):
    if ref_text_mode not in {"soft_ctc", "hybrid"} or utterance is None or utterance.frame_posteriors is None:
        return None

    base_dir = Path(posterior_file).expanduser().resolve().parent if posterior_file else None
    hybrid_conditioner = HybridReferenceConditioner() if ref_text_mode == "hybrid" else None

    def _builder(model_obj, text, duration, ref_audio_len, ref_text, gen_text, device):
        del gen_text
        token_ids, probs = load_topk_arrays(utterance.frame_posteriors, base_dir=base_dir)
        token_map = utterance.token_map
        token_ids = _project_asr_topk_to_f5_ids(token_ids, token_map, model_obj.vocab_char_map)
        embedding_weight = model_obj.transformer.text_embed.text_embed.weight

        text_tensor = _text_tensor_from_list(model_obj, text, embedding_weight.device)
        with torch.inference_mode():
            hard_embed = model_obj.transformer.text_embed(text_tensor, seq_len=duration, drop_text=False)
        hard_embed = hard_embed.clone()

        ref_token_text = convert_char_to_pinyin([ref_text])[0]
        ref_token_len = min(len(ref_token_text), hard_embed.shape[1])
        ref_soft = _compressed_soft_ref_embed(
            token_ids,
            probs,
            embedding_weight,
            hard_embed[:, :ref_token_len, :],
        )

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

    return _builder


def _posterior_encoder_builder_for_entry(utterance):
    if ref_text_mode != "posterior_encoder" or utterance is None or utterance.frame_posteriors is None:
        return None
    if not posterior_encoder_ckpt:
        print("Warning: posterior_encoder mode requires --posterior_encoder_ckpt. Falling back to hard text embedding.")
        return None

    base_dir = Path(posterior_file).expanduser().resolve().parent if posterior_file else None
    encoder_cache = {"encoder": None}

    def _builder(model_obj, text, duration, ref_audio_len, ref_text, gen_text, device):
        del gen_text, ref_audio_len
        if encoder_cache["encoder"] is None:
            encoder, _ = load_posterior_encoder_checkpoint(posterior_encoder_ckpt, map_location=device)
            encoder_cache["encoder"] = encoder.to(device).eval()

        token_ids, probs = load_topk_arrays(utterance.frame_posteriors, base_dir=base_dir)
        token_ids = _project_asr_topk_to_f5_ids(token_ids, utterance.token_map, model_obj.vocab_char_map)
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
        prob_tensor = torch.tensor(probs, device=device, dtype=torch.float32).unsqueeze(0)

        text_tensor = _text_tensor_from_list(model_obj, text, device)
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

    return _builder


def _text_embed_override_builder_for_entry(utterance, ssl_entry=None):
    if ref_text_mode == "posterior_encoder":
        return _posterior_encoder_builder_for_entry(utterance)
    return _soft_ctc_builder_for_entry(utterance, ssl_entry=ssl_entry)


# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)


# load TTS model

model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

# override for previous models
if model == "F5TTS_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    elif vocoder_name == "bigvgan":
        model = "F5TTS_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file:
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))
elif ckpt_file.startswith("hf://"):
    ckpt_file = str(cached_path(ckpt_file))

if vocab_file.startswith("hf://"):
    vocab_file = str(cached_path(vocab_file))

print(f"Using {model}...")
ema_model = load_model(
    model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
)


# inference process


def main():
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["posterior_entry"] = _posterior_entry_for_audio(voices[voice]["ref_audio"])
        voices[voice]["ssl_entry"] = _ssl_entry_for_audio(voices[voice]["ref_audio"], voices[voice]["posterior_entry"])
        voices[voice]["expected_ref_text_len"] = _expected_ref_text_len_for_entry(voices[voice]["posterior_entry"])
        voices[voice]["text_embed_override_builder"] = _text_embed_override_builder_for_entry(
            voices[voice]["posterior_entry"], voices[voice]["ssl_entry"]
        )
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"],
            voices[voice]["ref_text"],
            transcribe_if_empty=(ref_text_mode == "hard"),
        )
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        expected_ref_text_len_ = voices[voice].get("expected_ref_text_len")
        text_embed_override_builder_ = voices[voice].get("text_embed_override_builder")
        local_speed = voices[voice].get("speed", speed)
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
        audio_segment, final_sample_rate, spectrogram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=local_speed,
            fix_duration=fix_duration,
            device=device,
            expected_ref_text_len=expected_ref_text_len_,
            text_embed_override_builder=text_embed_override_builder_,
            seed=seed,
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + " ... "
            if use_legacy_text:
                gen_text_ = unidecode(gen_text_)
            sf.write(
                os.path.join(output_chunk_dir, f"{len(generated_audio_segments) - 1}_{gen_text_}.wav"),
                audio_segment,
                final_sample_rate,
            )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


if __name__ == "__main__":
    main()
