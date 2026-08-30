"""Gym judge adapters must speak both the old and current function signatures."""

from __future__ import annotations

from pathlib import Path

from adr.eval.deep_research_gym import evaluate_folder_call_args


def _legacy_quality(subfolder, model, path_to_reports):
    pass


def _current_quality(path_to_reports, model, num_workers=32):
    pass


def _legacy_kpr(subfolder, model, path_to_reports, key_point_dir):
    pass


def _current_kpr(path_to_reports, model, key_point_dir, num_workers=32):
    pass


def test_current_quality_uses_full_export_path():
    export = Path("/tmp/exports/deep_research_gym/pilot")
    args, kwargs = evaluate_folder_call_args(_current_quality, export, "gpt-4.1-mini")
    assert args == (str(export), "gpt-4.1-mini")
    assert kwargs == {}


def test_legacy_quality_uses_subfolder_and_parent():
    export = Path("/tmp/exports/deep_research_gym/pilot")
    args, kwargs = evaluate_folder_call_args(_legacy_quality, export, "gpt-4.1-mini")
    assert args == ("pilot", "gpt-4.1-mini", str(export.parent))
    assert kwargs == {}


def test_current_kpr_passes_key_point_dir_as_kwarg():
    export = Path("/tmp/exports/deep_research_gym/pilot")
    args, kwargs = evaluate_folder_call_args(
        _current_kpr, export, "gpt-4.1-mini", "/tmp/key_point"
    )
    assert args == (str(export), "gpt-4.1-mini")
    assert kwargs == {"key_point_dir": "/tmp/key_point"}


def test_legacy_kpr_appends_key_point_dir():
    export = Path("/tmp/exports/deep_research_gym/pilot")
    args, kwargs = evaluate_folder_call_args(
        _legacy_kpr, export, "gpt-4.1-mini", "/tmp/key_point"
    )
    assert args == ("pilot", "gpt-4.1-mini", str(export.parent), "/tmp/key_point")
    assert kwargs == {}
