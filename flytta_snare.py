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
ap.add_argument("--mängd", type=float, default=0.8, help="0-1.5, hur mycket av slagen som flyttas")
ap.add_argument("--före", type=float, default=0.015, help="sekunder före slaget")
ap.add_argument("--efter", type=float, default=0.07, help="sekunder efter slaget")
ap.add_argument("--delta", type=float, default=0.06, help="känslighet för slag (lägre = fler slag, rullar)")
ap.add_argument("--vänta", type=float, default=0.025, help="minsta tid mellan slag i sekunder (rullar kräver lågt)")
ap.add_argument("--extra", nargs="*", default=[], help="fler stems att hitta slag i, t.ex. percussion.mp3 toms.mp3 (rim shots)")
ap.add_argument("--låggräns", type=float, default=1500.0, help="Hz: under denna flyttas bara det som är tydligt slag")
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

# 1. När slår snaren (och rim shots)? onsets från snare-stemmet + extra stems
def onset_env(x):
    e = librosa.onset.onset_strength(y=x.mean(0), sr=sr, hop_length=HOP)
    return e / (e.max() + 1e-9)

envs = [onset_env(raw)]
for fn in a.extra:
    x, sx = librosa.load(fn, sr=None, mono=False)
    if x.ndim == 1:
        x = x[None]
    x = x[:, :n]
    if x.shape[1] < n:
        x = np.pad(x, ((0, 0), (0, n - x.shape[1])))
    envs.append(onset_env(x))
m = min(len(e) for e in envs)
env = np.max([e[:m] for e in envs], axis=0)
on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, backtrack=False,
                                delta=a.delta, wait=max(1, int(a.vänta * sr / HOP)))
times = librosa.frames_to_time(on, sr=sr, hop_length=HOP)
print(f"{len(times)} slag hittade (snare{' + ' + str(len(a.extra)) + ' extra' if a.extra else ''})")

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
    mp_hi = P ** 2 / (H ** 2 + P ** 2 + 1e-10)                 # mjuk mask (höga frekvenser)
    mp_lo = np.clip((P - 3.0 * H) / (P + 1e-10), 0, 1)       # sträng mask: bara tydliga slag
    freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    blend = 1.0 / (1.0 + np.exp(-(freqs - a.låggräns) / 250.0))[:, None]   # 0 under gränsen, 1 över
    mp = blend * mp_hi + (1.0 - blend) * mp_lo
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
