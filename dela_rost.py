"""dela_rost.py - delar ett sångspår i fraser och sorterar dem i N röster (tonhöjd + klangfärg).
Användning:  python dela_rost.py lead-vocal.mp3 [antal_roster=2]
Skriver: <namn>_rost1.wav, <namn>_rost2.wav, <namn>_rapport.txt bredvid filen.
"""
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

src = Path(sys.argv[1])
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2

y, sr = librosa.load(str(src), sr=None, mono=False)
if y.ndim == 1:
    y = y[None, :]
mono = y.mean(axis=0)
m22 = librosa.resample(mono, orig_sr=sr, target_sr=22050)
sr2 = 22050

# 1. dela i fraser på tystnad
iv = librosa.effects.split(m22, top_db=35, frame_length=2048, hop_length=512)
merged = []
for s, e in iv:
    if merged and (s - merged[-1][1]) < 0.35 * sr2:
        merged[-1][1] = e
    else:
        merged.append([s, e])
merged = [(s, e) for s, e in merged if (e - s) >= 0.5 * sr2]
print(f"{len(merged)} fraser hittade")
if len(merged) < K:
    sys.exit("För få fraser.")

# 2. mät varje fras
feats, info = [], []
for s, e in merged:
    seg = m22[s:e]
    f0, vf, _ = librosa.pyin(seg, fmin=70, fmax=700, sr=sr2, frame_length=2048)
    f0v = f0[~np.isnan(f0)]
    if len(f0v) < 5:
        med, sd = np.nan, np.nan
    else:
        med, sd = float(np.median(np.log2(f0v))), float(np.std(np.log2(f0v)))
    mf = librosa.feature.mfcc(y=seg, sr=sr2, n_mfcc=20)[1:]
    cen = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr2)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
    feats.append(np.concatenate([[med], mf.mean(axis=1), mf.std(axis=1)[:5], [cen, flat]]))
    info.append((s / sr2, e / sr2, 2 ** med if not np.isnan(med) else np.nan))

X = np.array(feats)
col_med = np.nanmedian(X, axis=0)
X = np.where(np.isnan(X), col_med, X)
Xs = StandardScaler().fit_transform(X)
Xs[:, 0] *= 2.5  # tonhöjd väger tyngre

km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(Xs)
lab = km.labels_
try:
    sil = silhouette_score(Xs, lab)
except Exception:
    sil = float("nan")

# sortera så röst1 = lägst tonhöjd
order = np.argsort([np.nanmedian([info[i][2] for i in range(len(info)) if lab[i] == c]) for c in range(K)])
remap = {int(c): r for r, c in enumerate(order)}
lab = np.array([remap[int(l)] for l in lab])

# 3. skriv ut ljud
out = [np.zeros_like(y) for _ in range(K)]
fade = int(0.01 * sr)
for (s, e), l in zip(merged, lab):
    a = max(0, int((s / sr2) * sr) - int(0.05 * sr))
    b = min(y.shape[1], int((e / sr2) * sr) + int(0.15 * sr))
    seg = y[:, a:b].copy()
    n = min(fade, seg.shape[1] // 2)
    seg[:, :n] *= np.linspace(0, 1, n)
    seg[:, -n:] *= np.linspace(1, 0, n)
    out[l][:, a:b] += seg

base = src.with_suffix("")
for k in range(K):
    sf.write(f"{base}_rost{k+1}.wav", out[k].T, sr, subtype="PCM_16")

lines = [f"Fil: {src.name}  röster: {K}  silhouette: {sil:.2f} (0.5+ = tydlig uppdelning, <0.25 = osäkert)", ""]
for k in range(K):
    f = [info[i][2] for i in range(len(info)) if lab[i] == k]
    t = sum(info[i][1] - info[i][0] for i in range(len(info)) if lab[i] == k)
    lines.append(f"Röst {k+1}: {sum(lab == k)} fraser, {t:.0f} s, median tonhöjd {np.nanmedian(f):.0f} Hz")
lines.append("")
lines.append("start_s  slut_s  tonhöjd_Hz  röst")
for (s, e, f), l in zip(info, lab):
    lines.append(f"{s:7.1f} {e:7.1f} {f:10.0f}  {l+1}")
Path(f"{base}_rapport.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:2 + K + 1]))
print("Klart. Lyssna på", f"{base}_rost1.wav", "och", f"{base}_rost2.wav")
