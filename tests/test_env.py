import subprocess

import pytest

import montagelib


def test_home_default(monkeypatch):
    monkeypatch.delenv("LEO50S_HOME", raising=False)
    assert str(montagelib.home()).endswith(".cache/leo50s-montage")


def test_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LEO50S_HOME", str(tmp_path))
    assert montagelib.home() == tmp_path


def test_ffmpeg_has_filters(ffmpeg_ok):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                         capture_output=True, text=True, check=True).stdout
    for name in "xfade zoompan minterpolate signalstats vignette curves".split():
        assert f" {name} " in out, name
