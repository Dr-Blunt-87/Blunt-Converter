"""flytta_snare.py - flyttar snare-smällar som läckt in i sångspåret över till snarespåret.

Princip: på de ställen där snaren slår (onsets från snare-stemmet) delas sångspåret i
"tonande" (sång) och "slag" (percussive, kantiga smällar). Slaget flyttas till snare.
Summan lead + snare blir exakt densamma som före.

Användning:
  python flytta_snare.py <lead.mp3> <snare.mp3> <rå_snare.mp3> [--mängd 1.0] [--ut mapp]
    lead      = 01_lead-vocal.mp3 (från albo_ut)
    snare     = 07_snare.mp3      (från albo_ut)
    rå_snare  = snare.mp3         (från raw_ut, används bara för att hitta SLAGEN i tid)
Skriver: lead_ren.wav och snare_ny.wav i --ut (default: bredvid lead-filen).
"""
import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter

ap = argparse.ArgumentParser()
ap.add_argument("lead")
ap.add_argument("snare")
ap.add_argument("raw_snare")
ap.add_argument("--mängd", type=float, default=1.0, help="0-1.5, hur mycket av slagen som flyttas")
ap.add_argument("--före", type=float, default=0.015, help="sekunder före slaget")
ap.add_argument("--efter", type=float, default=0.12, help="sekunder efter slaget")
ap.add_argument("--ut", default=None)
a = ap.parse_args()

NFFT, HOP = 2048, 512
lead, sr = librosa.load(a.lead, sr=None, mono=False)
snare, sr2 = librosa.load(a.snare, sr=None, mono=False)
raw, sr3 = librosa.load(a.raw_snare, sr=None, mono=False)
assert sr == sr2 == sr3, "olika sample rate"
if lead.ndim == 1:
    lead, snare, raw = lead[None], snare[None], raw[None]
n = min(lead.shape[1], snare.shape[1], raw.shape[1])
lead, snare, raw = lead[:, :n], snare[:, :n], raw[:, :n]

# 1. När slår snaren? onsets från det råa snare-stemmet
env = librosa.onset.onset_strength(y=raw.mean(0), sr=sr, hop_length=HOP)
on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, backtrack=False,
                                delta=0.15, wait=int(0.08 * sr / HOP))
times = librosa.frames_to_time(on, sr=sr, hop_length=HOP)
print(f"{len(times)} snare-slag hittade")

frames = 1 + n // HOP
win = np.zeros(frames, dtype=np.float32)
for t in times:
    f0 = max(0, int((t - a.före) * sr / HOP))
    f1 = min(frames, int((t + a.efter) * sr / HOP) + 1)
    win[f0:f1] = 1.0
# mjuka kanter på fönstren
k = np.hanning(7); k /= k.sum()
win = np.clip(np.convolve(win, k, mode="same") * 1.5, 0, 1)

# 2. dela sångspåret i tonande/slag med median-filter (HPSS)
moved_total = np.zeros_like(lead)
for ch in range(lead.shape[0]):
    S = librosa.stft(lead[ch], n_fft=NFFT, hop_length=HOP)
    mag = np.abs(S)
    H = median_filter(mag, size=(1, 17))   # tonande: lång i tid
    P = median_filter(mag, size=(17, 1))   # slag: lång i frekvens
    mp = P ** 2 / (H ** 2 + P ** 2 + 1e-10)
    mp = np.clip(mp * a.mängd, 0, 1)
    w = win[: S.shape[1]]
    if len(w) < S.shape[1]:
        w = np.pad(w, (0, S.shape[1] - len(w)))
    M = mp * w[None, :]
    moved = librosa.istft(S * M, hop_length=HOP, length=n)
    moved_total[ch] = moved

lead_ren = lead - moved_total
snare_ny = snare + moved_total

def db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
print(f"Flyttat från sång till snare: {db(moved_total):.1f} dB (rms)  | sång före {db(lead):.1f} dB, efter {db(lead_ren):.1f} dB")
print(f"Snare före {db(snare):.1f} dB, efter {db(snare_ny):.1f} dB")

out = Path(a.ut) if a.ut else Path(a.lead).parent
out.mkdir(parents=True, exist_ok=True)
sf.write(out / "lead_ren.wav", lead_ren.T, sr, subtype="PCM_16")
sf.write(out / "snare_ny.wav", snare_ny.T, sr, subtype="PCM_16")
sf.write(out / "flyttat.wav", moved_total.T, sr, subtype="PCM_16")
print("Klart:", out / "lead_ren.wav", "|", out / "snare_ny.wav", "|", out / "flyttat.wav")
