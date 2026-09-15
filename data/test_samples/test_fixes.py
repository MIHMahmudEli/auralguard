"""End-to-end test for the data loading fixes."""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from scripts.build_manifests import in_the_wild, asvspoof2021_la, asvspoof2021_df, COLUMNS
from src.auralguard.data.datasets import _load_audio
import numpy as np


def test_in_the_wild_manifest():
    """Test in_the_wild manifest builder with fake/real subdir structure."""
    print('=' * 60)
    print('TEST 1: in_the_wild manifest builder with fake/real subdirs')
    print('=' * 60)

    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, 'in_the_wild')
        os.makedirs(os.path.join(root, 'release_in_the_wild', 'fake'))
        os.makedirs(os.path.join(root, 'release_in_the_wild', 'real'))

        # Create meta.csv
        meta_data = {
            'file': ['0.wav', '1.wav', '2.wav', '3.wav'],
            'speaker': ['A', 'B', 'C', 'D'],
            'label': ['spoof', 'spoof', 'bona-fide', 'bona-fide']
        }
        pd.DataFrame(meta_data).to_csv(os.path.join(root, 'meta.csv'), index=False)

        # Create actual files in fake/ and real/ dirs
        for i in range(2):
            open(os.path.join(root, 'release_in_the_wild', 'fake', f'{i}.wav'), 'w').close()
        for i in range(2, 4):
            open(os.path.join(root, 'release_in_the_wild', 'real', f'{i}.wav'), 'w').close()

        df = in_the_wild(root)
        print(f'  Manifest rows: {len(df)}')
        print(f'  Labels: {df.label.value_counts().to_dict()}')

        all_exist = all(os.path.exists(p) for p in df['path'])
        print(f'  All files exist: {all_exist}')
        assert all_exist, 'Some files could not be resolved!'
        assert len(df) == 4, f'Expected 4 rows, got {len(df)}'
        assert df.label.sum() == 2, f'Expected 2 spoof, got {df.label.sum()}'
        print('  PASSED')
    finally:
        shutil.rmtree(tmp)


def test_asvspoof2021_graceful_missing():
    """Test ASVspoof2021 adapters return empty DataFrames when data is missing."""
    print()
    print('=' * 60)
    print('TEST 2: ASVspoof2021 adapters with missing data')
    print('=' * 60)

    df_la = asvspoof2021_la('nonexistent_path')
    print(f'  LA: {len(df_la)} rows, columns={list(df_la.columns)}')
    assert len(df_la) == 0
    assert list(df_la.columns) == COLUMNS
    print('  LA PASSED')

    df_df = asvspoof2021_df('nonexistent_path')
    print(f'  DF: {len(df_df)} rows, columns={list(df_df.columns)}')
    assert len(df_df) == 0
    assert list(df_df.columns) == COLUMNS
    print('  DF PASSED')


def test_audio_loading_with_real_file():
    """Test that the updated _load_audio can load a real WAV file."""
    print()
    print('=' * 60)
    print('TEST 3: _load_audio with real WAV file')
    print('=' * 60)

    wav_path = os.path.join(os.path.dirname(__file__), '0.wav')
    if not os.path.exists(wav_path):
        print('  SKIPPED (0.wav not downloaded)')
        return

    wav = _load_audio(wav_path, sr=16000)
    print(f'  Shape: {wav.shape}, dtype: {wav.dtype}')
    print(f'  Mean: {wav.mean():.6f}, Std: {wav.std():.6f}')
    print(f'  Max abs: {np.abs(wav).max():.6f}')
    is_silent = np.abs(wav).max() < 1e-6
    print(f'  Is silent: {is_silent}')
    assert not is_silent, 'WAV file loaded as silent!'
    assert wav.shape[0] > 0, 'Empty waveform!'
    print('  PASSED')


def test_meta_csv_parsing():
    """Test that meta.csv is parsed correctly."""
    print()
    print('=' * 60)
    print('TEST 4: meta.csv parsing')
    print('=' * 60)

    meta_path = os.path.join(os.path.dirname(__file__), 'meta.csv')
    if not os.path.exists(meta_path):
        print('  SKIPPED (meta.csv not downloaded)')
        return

    df = pd.read_csv(meta_path)
    print(f'  Columns: {list(df.columns)}')
    print(f'  Shape: {df.shape}')
    print(f'  Label distribution: {df.label.value_counts().to_dict()}')
    assert 'file' in df.columns
    assert 'label' in df.columns
    assert len(df) > 0
    print('  PASSED')


if __name__ == '__main__':
    test_in_the_wild_manifest()
    test_asvspoof2021_graceful_missing()
    test_audio_loading_with_real_file()
    test_meta_csv_parsing()
    print()
    print('=' * 60)
    print('ALL TESTS PASSED')
    print('=' * 60)
