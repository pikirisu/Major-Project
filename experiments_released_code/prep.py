"""The base paper's released preprocessing, run on ds004504 (AD vs HC, 65 people).

Source: github.com/brief-city/MFE-FCGCN-model, raw_data_processing.py and
generate_graph_data.py. The maths is copied line for line; only the I/O changes
(one .npz per person instead of thousands of CSVs, then .pt files):

  Welch PSD per channel, n_fft 2500, n_overlap 500, average=None (5 s segments, 4 s step)
  window = mean of 4 consecutive segments; first 80 windows (AD) / 100 (HC)
  node features = per-channel z-score of the band's PSD vector (raw PSD, not log)
  edge set 1 ("mul-mat")    = Kraskov MI between channels, divided by its row maximum
  edge set 2 ("person-mat") = |Pearson|, off-diagonal row-normalised, diagonal kept

Only departure: mutual_info_regression gets random_state=0 (theirs is None, so
their own matrices change from run to run).

Usage: python experiments_released_code/prep.py   (writes cache_released_code/)
"""
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression

mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache_released_code"
OUT.mkdir(exist_ok=True)

CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5',
            'T6', 'Fz', 'Cz', 'Pz']
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
HZ_MIN, HZ_MAX = [0.5, 4, 8, 13, 30], [4, 8, 13, 30, 45]


def cal_mutual_information(df):                       # theirs, plus random_state
    mi = np.zeros((df.shape[1], df.shape[1]))
    for a, col1 in enumerate(df.columns):
        for b, col2 in enumerate(df.columns):
            mi[a, b] = mutual_info_regression(df[[col1]], df[col2], discrete_features='auto', random_state=0)[0]
    row_maxes = mi.max(axis=1, keepdims=True)
    return np.divide(mi, row_maxes, where=row_maxes != 0)


def create_new_adjacent_matrix(df):
    # theirs in numpy (pandas copy-on-write makes their in-place fill_diagonal fail):
    # |r|, each row's off-diagonal divided by its sum, diagonal = the original r_ii
    a = np.abs(df.to_numpy(dtype=np.float64))
    np.fill_diagonal(a, np.nan)
    row_sum = np.nansum(a, axis=1, keepdims=True)
    a = np.where(row_sum != 0, a / np.where(row_sum != 0, row_sum, 1), a)
    np.fill_diagonal(a, np.diag(df.to_numpy(dtype=np.float64)))
    return a


def one_person(i):
    sid = f"sub-{i:03d}"
    out = OUT / f"{sid}.npz"
    if out.exists():
        return sid, "cached"
    label = 1 if i <= 36 else 0
    max_sample_num = 80 if label == 1 else 100
    raw = mne.io.read_raw_eeglab(ROOT / "data" / "derivatives" / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set",
                                 preload=True)
    sfreq = raw.info['sfreq']
    xs, mis, pcs = {}, [], []
    for b, (fmin, fmax) in enumerate(zip(HZ_MIN, HZ_MAX)):
        psd_data = {}
        for name in CHANNELS:
            data, times = raw[name, :]
            psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq, fmin=fmin, fmax=fmax, n_fft=2500,
                                                             average=None, n_overlap=500)
            psd_data[name] = psds[0].T
        feats, mi_b, pc_b = [], [], []
        i_win = 0
        for _ in range(len(psd_data['Fp1'])):
            if i_win >= max_sample_num or i_win > len(psd_data['Fp1']) - 4:
                break
            vals = {}
            for ch in CHANNELS:
                m = np.mean(psd_data[ch][i_win:i_win + 4], axis=0)
                vals[ch] = (m - m.mean()) / m.std()
            df = pd.DataFrame(vals, columns=CHANNELS)
            feats.append(df.values.T)                          # (19, bins), as generate_graph_data.py
            mi_b.append(cal_mutual_information(df))
            pc_b.append(create_new_adjacent_matrix(df.corr()))
            i_win += 1
        xs[f"x_{BANDS[b]}"] = np.asarray(feats, np.float32)
        mis.append(mi_b); pcs.append(pc_b)
    n = min(len(m) for m in mis)
    np.savez_compressed(out, **{k: v[:n] for k, v in xs.items()},
                        mi=np.asarray([m[:n] for m in mis], np.float32).transpose(1, 0, 2, 3),
                        pc=np.asarray([p[:n] for p in pcs], np.float32).transpose(1, 0, 2, 3),
                        y=np.full(n, label, np.int64))
    return sid, n


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or list(range(1, 66))
    for sid, n in Parallel(n_jobs=16, verbose=5)(delayed(one_person)(i) for i in ids):
        print(sid, n)
