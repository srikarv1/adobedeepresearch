from pathlib import Path

from adr.eval.deep_research_gym import evaluate_folder_call_args
from adr.eval.repos import find_gpt_researcher


def test_flitternie_quality_signature_uses_path_to_reports():
    def evaluate_folder_async(path_to_reports, model, num_workers=32):
        return None

    args, kwargs = evaluate_folder_call_args(
        evaluate_folder_async, Path("/tmp/exports/sys"), "gpt-5.6-sol"
    )
    assert args[0] == "/tmp/exports/sys"
    assert args[1] == "gpt-5.6-sol"
    assert kwargs == {}


def test_legacy_quality_signature_uses_subfolder_and_parent():
    def evaluate_folder_async(subfolder, model, parent_dir, key_point_dir=None):
        return None

    args, kwargs = evaluate_folder_call_args(
        evaluate_folder_async, Path("/tmp/exports/sys"), "m", "/kp"
    )
    assert args[0] == "sys"
    assert args[2] == "/tmp/exports"
    assert args[3] == "/kp"


def test_gpt_researcher_reference_resolves_when_bootstrapped():
    located = find_gpt_researcher()
    if not located.ok:
        return
    assert (located.path / "gpt_researcher" / "__init__.py").exists()
