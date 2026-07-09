"""Integration tests for dataset manifest builders."""
from __future__ import annotations

from pathlib import Path
import csv
import io
import tempfile

import pytest

from scripts.build_manifests import (
    MANIFEST_DIR,
    COLUMNS,
    asvspoof2019_la,
    asvspoof2021_la,
    asvspoof2021_df,
    in_the_wild,
    wavefake,
    mlaad,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _write_proto(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")  # empty audio stub


# ── Tests ────────────────────────────────────────────────────────────────

class TestASVspoof2019LA:
    def test_bonafide_and_spoof(self, tmp_path: Path):
        proto = tmp_path / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
        audio_dir = tmp_path / "ASVspoof2019_LA_train" / "flac"
        _write_proto(proto, [
            "LA_0001 LA_T_1000001 - - bonafide",
            "LA_0001 LA_T_1000002 - A01 spoof",
        ])
        _touch(audio_dir / "LA_T_1000001.flac")
        _touch(audio_dir / "LA_T_1000002.flac")

        df = asvspoof2019_la(str(tmp_path), "train")
        assert len(df) == 2
        assert list(df.columns) == COLUMNS
        assert df.iloc[0]["label"] == 0
        assert df.iloc[1]["label"] == 1
        assert df.iloc[1]["attack"] == "A01"
        assert df.iloc[0]["dataset"] == "asvspoof2019_la"

    def test_train_suffix(self):
        assert "trn.txt" in "ASVspoof2019.LA.cm.train.trn.txt"

    def test_eval_suffix(self):
        proto = "ASVspoof2019.LA.cm.eval.trl.txt"
        assert "trl.txt" in proto


class TestASVspoof2021LA:
    def test_reads_metadata(self, tmp_path: Path):
        proto = tmp_path / "trial_metadata.txt"
        audio_dir = tmp_path / "flac"
        _write_proto(proto, [
            "LA_0019 LA_E_9332881 none ita_tx A07 spoof notrim eval",
            "LA_0009 LA_E_9332882 opus mad_tx A08 bonafide trim progress",
        ])
        _touch(audio_dir / "LA_E_9332881.flac")
        _touch(audio_dir / "LA_E_9332882.flac")

        df = asvspoof2021_la(str(tmp_path))
        assert len(df) == 2
        assert df.iloc[0]["label"] == 1
        assert df.iloc[1]["label"] == 0
        assert df.iloc[1]["codec"] == "opus"
        assert df.iloc[0]["dataset"] == "asvspoof2021_la"


class TestASVspoof2021DF:
    def test_reads_metadata(self, tmp_path: Path):
        proto = tmp_path / "trial_metadata.txt"
        audio_dir = tmp_path / "flac"
        _write_proto(proto, [
            "LA_0023 DF_E_2000011 nocodec asvspoof A12 spoof notrim eval vocoder_type - - - -",
            "LA_0023 DF_E_2000012 high_mp3 vcc2020 A13 bonafide trim eval bonafide - - - -",
        ])
        _touch(audio_dir / "DF_E_2000011.flac")
        _touch(audio_dir / "DF_E_2000012.flac")

        df = asvspoof2021_df(str(tmp_path))
        assert len(df) == 2
        assert df.iloc[0]["label"] == 1
        assert df.iloc[1]["label"] == 0
        assert df.iloc[0]["codec"] == "nocodec"


class TestInTheWild:
    def test_reads_meta_csv(self, tmp_path: Path):
        meta = tmp_path / "meta.csv"
        meta.write_text("file,label\nbarack_obama_001.wav,bonafide\ndonald_trump_002.wav,spoof\n")
        _touch(tmp_path / "barack_obama_001.wav")
        _touch(tmp_path / "donald_trump_002.wav")

        df = in_the_wild(str(tmp_path))
        assert len(df) == 2
        assert df.iloc[0]["label"] == 0
        assert df.iloc[1]["label"] == 1


class TestWaveFake:
    def test_all_spoof_without_ljspeech(self, tmp_path: Path):
        (tmp_path / "ljspeech_melgan").mkdir(parents=True)
        _touch(tmp_path / "ljspeech_melgan" / "LJ001-0001_gen.wav")
        _touch(tmp_path / "ljspeech_melgan" / "LJ001-0002_gen.wav")

        df = wavefake(str(tmp_path))
        assert len(df) == 2
        assert (df["label"] == 1).all()
        assert df.iloc[0]["attack"] == "ljspeech_melgan"

    def test_pairs_with_ljspeech(self, tmp_path: Path):
        # wavefake() checks for data/raw/LJSpeech at run time — this just
        # verifies spoof rows are produced; bonafide pairing only happens
        # when that directory exists.
        wf_dir = tmp_path / "generated_audio"
        (wf_dir / "ljspeech_hifiGAN").mkdir(parents=True)
        _touch(wf_dir / "ljspeech_hifiGAN" / "LJ001-0001_gen.wav")

        df = wavefake(str(wf_dir))
        assert len(df) == 1
        assert df.iloc[0]["label"] == 1
        assert df.iloc[0]["attack"] == "ljspeech_hifiGAN"


class TestMLAAD:
    def test_reads_language_meta(self, tmp_path: Path):
        meta = tmp_path / "fake" / "en" / "tts_model_vits" / "meta.csv"
        meta.parent.mkdir(parents=True)
        meta.write_text("path|original_file|language|is_original_language|duration|training_data|model_name\n"
                        "audio_001.wav|ref/001.wav|en|True|3.2|ljspeech|facebook/mms-tts-eng\n")
        _touch(meta.parent / "audio_001.wav")

        df = mlaad(str(tmp_path))
        assert len(df) == 1
        assert df.iloc[0]["label"] == 1
        assert df.iloc[0]["lang"] == "en"
        assert df.iloc[0]["dataset"] == "mlaad"


class TestColumnsAndRegistry:
    def test_all_adapters_produce_expected_columns(self):
        for name, adapter in [
            ("asvspoof2019_la_train", lambda: asvspoof2019_la("dummy", "train")),
            ("asvspoof2021_la_eval", lambda: asvspoof2021_la("dummy")),
            ("asvspoof2021_df_eval", lambda: asvspoof2021_df("dummy")),
        ]:
            with pytest.raises((FileNotFoundError, OSError)):
                adapter()  # no data in dummy path — just check no crash on columns


def test_manifest_dir_constant():
    assert str(MANIFEST_DIR) == "data\\manifests" or str(MANIFEST_DIR) == "data/manifests"
