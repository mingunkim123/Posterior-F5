import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dit_and_cfm_accept_optional_text_embed_override():
    dit_source = (REPO_ROOT / "src/f5_tts/model/backbones/dit.py").read_text(encoding="utf-8")
    cfm_source = (REPO_ROOT / "src/f5_tts/model/cfm.py").read_text(encoding="utf-8")

    dit_tree = ast.parse(dit_source)
    cfm_tree = ast.parse(cfm_source)

    dit_class = next(node for node in ast.walk(dit_tree) if isinstance(node, ast.ClassDef) and node.name == "DiT")
    cfm_class = next(node for node in ast.walk(cfm_tree) if isinstance(node, ast.ClassDef) and node.name == "CFM")
    dit_forward = next(node for node in dit_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    dit_get_input_embed = next(
        node for node in dit_class.body if isinstance(node, ast.FunctionDef) and node.name == "get_input_embed"
    )
    cfm_sample = next(node for node in cfm_class.body if isinstance(node, ast.FunctionDef) and node.name == "sample")

    assert "text_embed_override" in [arg.arg for arg in dit_forward.args.args]
    assert "text_embed_override" in [arg.arg for arg in dit_get_input_embed.args.args]
    assert "text_embed_override" in [arg.arg for arg in cfm_sample.args.kwonlyargs]
    assert "text_embed_override is None" in dit_source
    assert "self.text_embed(text, seq_len=seq_len, drop_text=drop_text)" in dit_source


def test_prepare_text_embed_override_pads_and_trims_when_torch_is_available():
    torch = pytest.importorskip("torch")
    from f5_tts.model.backbones.dit import DiT

    model = DiT(dim=8, depth=1, heads=1, dim_head=8, mel_dim=4, text_num_embeds=6, text_dim=5)

    short_override = torch.ones(1, 2, 5)
    padded = model.prepare_text_embed_override(short_override, seq_len=4)
    assert padded.shape == (1, 4, 5)
    assert torch.allclose(padded[:, :2], short_override)
    assert torch.allclose(padded[:, 2:], torch.zeros(1, 2, 5))

    long_override = torch.ones(1, 6, 5)
    trimmed = model.prepare_text_embed_override(long_override, seq_len=3)
    assert trimmed.shape == (1, 3, 5)


def test_infer_cli_exposes_soft_ctc_builder_path():
    source = (REPO_ROOT / "src/f5_tts/infer/infer_cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert '"soft_ctc"' in source
    assert '"hybrid"' in source
    assert "HybridReferenceConditioner" in source
    assert "expected_embedding_from_topk" in source
    assert "load_topk_arrays" in source
    assert "text_embed_override_builder=text_embed_override_builder_" in source

    builder = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_soft_ctc_builder_for_entry"
    )
    builder_source = ast.get_source_segment(source, builder)
    assert "hard_embed[:, :replace_len, :]" in builder_source
