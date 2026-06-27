import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_tree(relative_path):
    return ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _function_def(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


def _arg_default(function_def, arg_name):
    args = function_def.args.args
    defaults = function_def.args.defaults
    offset = len(args) - len(defaults)
    for index, arg in enumerate(args):
        if arg.arg == arg_name:
            default_index = index - offset
            if default_index < 0:
                return None
            return defaults[default_index]
    raise AssertionError(f"Argument not found: {arg_name}")


def test_infer_process_keeps_expected_ref_text_len_optional():
    tree = _module_tree("src/f5_tts/infer/utils_infer.py")
    function_def = _function_def(tree, "infer_process")

    assert function_def.args.args[-1].arg == "expected_ref_text_len"
    assert isinstance(_arg_default(function_def, "expected_ref_text_len"), ast.Constant)
    assert _arg_default(function_def, "expected_ref_text_len").value is None


def test_infer_batch_process_keeps_expected_ref_text_len_optional():
    tree = _module_tree("src/f5_tts/infer/utils_infer.py")
    function_def = _function_def(tree, "infer_batch_process")

    assert function_def.args.args[-1].arg == "expected_ref_text_len"
    assert isinstance(_arg_default(function_def, "expected_ref_text_len"), ast.Constant)
    assert _arg_default(function_def, "expected_ref_text_len").value is None


def test_infer_process_forwards_expected_ref_text_len_to_batch_process():
    tree = _module_tree("src/f5_tts/infer/utils_infer.py")
    function_def = _function_def(tree, "infer_process")

    matching_calls = [
        node
        for node in ast.walk(function_def)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "infer_batch_process"
    ]

    assert matching_calls
    keywords = {keyword.arg for keyword in matching_calls[0].keywords}
    assert "expected_ref_text_len" in keywords


def test_infer_cli_exposes_length_only_without_changing_hard_default():
    tree = _module_tree("src/f5_tts/infer/infer_cli.py")
    source = (REPO_ROOT / "src/f5_tts/infer/infer_cli.py").read_text(encoding="utf-8")

    assert "--ref_text_mode" in source
    assert "--posterior_file" in source
    assert 'config.get("ref_text_mode", "hard")' in source

    infer_process_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "infer_process"
    ]
    assert infer_process_calls
    keywords = {keyword.arg for keyword in infer_process_calls[0].keywords}
    assert "expected_ref_text_len" in keywords

