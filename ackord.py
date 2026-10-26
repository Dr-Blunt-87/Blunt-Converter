"""ackord.py - hittar ackord i en låt (eller i ett stem som gitarr/keys/piano).

Användning:
  python ackord.py <ljudfil> [--ut mapp] [--sjuor]
    --sjuor  tillåt 7-ackord (maj7, min7, dom7). Utan den väljs bara dur/moll (stabilare).

Skriver:  <namn>_ackord.txt  (tid + ackord)   och   <namn>_ackord.mid  (ackorden som MIDI)
Tips: bäst resultat får du på gitarr/keys/piano-spåret, eller hela mixen. Trummor och sång stör.
Bygger bara på librosa + pretty_midi (fria bibliotek).
"""
import argparse
from pathlib import Path
import numpy as np
import librosa

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ap = argparse.ArgumentParser()
ap.add_argument("ljud")
ap.add_argument("--ut", default=None)
ap.add_argument("--sjuor", action="store_true")
a = ap.parse_args()

SR, HOP = 22050, 512
y, _ = librosa.load(a.ljud, sr=SR, mono=True)
yh = librosa.effects.harmonic(y, margin=4)          # ta bort trummor/slag, behåll toner

# 1. takt: slag att dela upp tiden på
tempo, bf = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, trim=False)
tempo = float(np.atleast_1d(tempo)[0])
chroma = librosa.feature.chroma_cqt(y=yh, sr=SR, hop_length=HOP, bins_per_octave=36)
nfr = chroma.shape[1]
if len(bf) < 4:                                      # inga tydliga slag: dela på 0,5 s
    bf = np.arange(0, nfr, int(0.5 * SR / HOP))
bf = librosa.util.fix_frames(bf, x_min=0, x_max=nfr)
cs = librosa.util.sync(chroma, bf, aggregate=np.median)
cs = cs / (np.linalg.norm(cs, axis=0, keepdims=True) + 1e-9)
btimes = librosa.frames_to_time(bf, sr=SR, hop_length=HOP)

# 2. ackordmallar
types = [("", [0, 4, 7], 0.0), ("m", [0, 3, 7], 0.0)]
if a.sjuor:
    types += [("7", [0, 4, 7, 10], 0.04), ("maj7", [0, 4, 7, 11], 0.04), ("m7", [0, 3, 7, 10], 0.04)]
labels, templ, pen = [], [], []
for r in range(12):
    for name, ivs, p in types:
        t = np.zeros(12)
        for i in ivs:
            t[(r + i) % 12] = 1.0
        templ.append(t / np.linalg.norm(t))
        labels.append(NOTES[r] + name)
        pen.append(p)
labels.append("N")                                    # inget ackord
templ.append(np.ones(12) / np.sqrt(12))
pen.append(0.25)
T = np.array(templ)
score = T @ cs - np.array(pen)[:, None]               # (ackord, tidssteg)
energy = chroma_energy = np.array([np.sum(chroma[:, s:e]) for s, e in zip(bf[:-1], bf[1:])] + [0.0])[: cs.shape[1]]
quiet = energy < 0.15 * np.median(energy[energy > 0]) if np.any(energy > 0) else np.zeros_like(energy, dtype=bool)
score[-1, quiet] += 1.0

# 3. jämna ut med Viterbi (ackord byts sällan)
prob = np.exp(score * 8.0)
prob = prob / prob.sum(axis=0, keepdims=True)
trans = librosa.sequence.transition_loop(len(labels), 0.8)
path = librosa.sequence.viterbi(prob, trans)

# 4. slå ihop lika ackord i följd
segs = []
for i, s in enumerate(path):
    t0 = btimes[i]
    t1 = btimes[i + 1] if i + 1 < len(btimes) else len(y) / SR
    if segs and segs[-1][2] == labels[s]:
        segs[-1][1] = t1
    else:
        segs.append([t0, t1, labels[s]])

def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"

out = Path(a.ut) if a.ut else Path(a.ljud).parent
out.mkdir(parents=True, exist_ok=True)
base = out / Path(a.ljud).stem
lines = [f"Tempo ca {tempo:.0f} BPM", ""] + [f"{fmt(s0)} - {fmt(s1)}   {c}" for s0, s1, c in segs]
Path(f"{base}_ackord.txt").write_text("\n".join(lines), encoding="utf-8")

# 5. MIDI med ackorden
import pretty_midi
pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
inst = pretty_midi.Instrument(program=0, name="Ackord")
for s0, s1, c in segs:
    if c == "N":
        continue
    root = NOTES.index(c[:2]) if len(c) > 1 and c[1] == "#" else NOTES.index(c[0])
    suffix = c[2:] if len(c) > 1 and c[1] == "#" else c[1:]
    ivs = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10]}[suffix]
    for iv in ivs:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60 + root + iv, start=s0, end=s1))
pm.instruments.append(inst)
pm.write(f"{base}_ackord.mid")

print("\n".join(lines[:40]))
print("\nKlart:", f"{base}_ackord.txt", "|", f"{base}_ackord.mid")
