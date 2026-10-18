#!/usr/bin/env python3
"""
audit_stems.py - jämför våra färdiga spår med originallåten.

Tanken: lägger man ihop alla spår ska det bli ungefär originalet. Det som
saknas (resten) är det som delningen tagit bort av misstag. Verktyget visar
  - hur mycket av originalet som finns kvar i spåren (SDR i dB, högre = bättre)
  - var det saknas: per frekvensband och per tidsstycke
  - hur stor del av originalet varje spår står för
  - om spåren tillsammans har mer ljud än originalet (dubbletter/spökkopior)
och sparar resten som  _rest.wav  (original minus spåren) så att du kan lyssna.

Exempel:
  python audit_stems.py "C:\\...\\Richie Spice - Ghetto Girl.mp3" "C:\\...\\clusters\\final"

Valfritt:
  --rest-track   skriver även  _rest.wav  in som eget spår i en mapp  med_rest/
  --exclude 05,06  hoppar över spår vars nummer/namn börjar så (t.ex. dubbletter)
"""
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, resample_poly, fftconvolve

BANDS = [(20, 60, 'sub'), (60, 250, 'bas'), (250, 500, 'låg-mitt'), (500, 2000, 'mitt'),
         (2000, 4000, 'hög-mitt'), (4000, 8000, 'diskant'), (8000, 16000, 'luft')]


def load_audio(path):
    try:
        data, sr = sf.read(path, dtype='float64', always_2d=True)
        return data, sr
    except Exception:
        tmp = Path(tempfile.mkdtemp()) / 'orig.wav'
        r = subprocess.run(['ffmpeg', '-y', '-i', str(path), '-ar', '44100', '-ac', '2', str(tmp)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            print('Kunde inte läsa originalet. Behöver ffmpeg i PATH för mp3.')
            sys.exit(1)
        return sf.read(tmp, dtype='float64', always_2d=True)


def to_stereo(x):
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def db(x):
    return 10 * np.log10(max(float(x), 1e-20))


def find_lag(orig, mix, sr, max_s=0.5, seconds=40):
    """Hur många samples mix ligger efter original (positivt = mix är försenad)."""
    n = int(min(len(orig), len(mix), seconds * sr))
    a = orig[:n].mean(axis=1)
    b = mix[:n].mean(axis=1)
    c = fftconvolve(a, b[::-1], mode='full')
    mid = len(b) - 1
    m = int(max_s * sr)
    seg = c[mid - m: mid + m + 1]
    return -(int(np.argmax(np.abs(seg))) - m)   # >0: mix ligger efter original


def shift(x, lag, n):
    """Flytta x så den passar original-längden n. lag>0: x ligger för sent => ta bort början."""
    if lag > 0:
        x = x[lag:]
    elif lag < 0:
        x = np.pad(x, ((-lag, 0), (0, 0)))
    if len(x) < n:
        x = np.pad(x, ((0, n - len(x)), (0, 0)))
    return x[:n]


def band_energies(x, sr):
    f, _, Z = stft(x.mean(axis=1), sr, nperseg=4096, noverlap=2048)
    p = np.abs(Z) ** 2
    out = []
    for lo, hi, _ in BANDS:
        m = (f >= lo) & (f < hi)
        out.append(float(p[m].sum()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('original')
    ap.add_argument('folder')
    ap.add_argument('--exclude', default='', help='kommaseparerade början på filnamn att hoppa över')
    ap.add_argument('--rest-track', action='store_true')
    ap.add_argument('--segment', type=float, default=5.0)
    args = ap.parse_args()

    folder = Path(args.folder)
    skip = [s.strip().lower() for s in args.exclude.split(',') if s.strip()]
    files = [p for p in sorted(folder.glob('*.wav'))
             if not p.name.startswith('_') and not any(p.name.lower().startswith(s) for s in skip)]
    if not files:
        print('Hittar inga wav-filer i', folder)
        sys.exit(1)

    orig, sr = load_audio(args.original)
    orig = to_stereo(orig)
    n = len(orig)
    print(f'Original: {n / sr:.1f} s, {sr} Hz. Spår att jämföra: {len(files)}')

    tracks = {}
    for p in files:
        d, s = sf.read(p, dtype='float64', always_2d=True)
        d = to_stereo(d)
        if s != sr:
            d = resample_poly(d, sr, s, axis=0)
        tracks[p.name] = d
    L = max(len(t) for t in tracks.values())
    mix = np.zeros((L, 2))
    for d in tracks.values():
        mix[:len(d)] += d

    lag = find_lag(orig, mix, sr)
    if lag != 0:
        print(f'Justerar tidsförskjutning: {lag / sr * 1000:.1f} ms')
    mix = shift(mix, lag, n)
    for k in tracks:
        tracks[k] = shift(tracks[k], lag, n)

    g = float((orig * mix).sum() / max((mix * mix).sum(), 1e-20))
    print(f'Spåren multipliceras med {g:.2f} för att matcha originalets nivå '
          f'(de sänktes ju för att inte klippa)')
    mix_g = mix * g
    res = orig - mix_g

    e_o, e_m, e_r = (orig ** 2).sum(), (mix_g ** 2).sum(), (res ** 2).sum()
    sdr = db(e_o) - db(e_r)
    print('\n=== HELA LÅTEN ===')
    print(f'Det som finns kvar i spåren (SDR): {sdr:.1f} dB')
    if sdr >= 20:
        print('  Mycket bra: spåren innehåller nästan allt i originalet.')
    elif sdr >= 12:
        print('  Bra: det mesta finns med, en del saknas.')
    elif sdr >= 6:
        print('  Måttligt: en märkbar del av låten har försvunnit eller hamnat fel.')
    else:
        print('  Lågt: mycket av låten saknas i spåren.')
    print(f'Summan av spåren har {db(e_m) - db(e_o):+.1f} dB jämfört med originalets energi'
          + ('  (över 0 = dubbletter/spökkopior)' if e_m > e_o * 1.05 else ''))

    print('\n=== PER FREKVENSBAND (hur mycket saknas) ===')
    bo, bm, br = band_energies(orig, sr), band_energies(mix_g, sr), band_energies(res, sr)
    print(f'{"band":<10}{"original":>10}{"spåren":>10}{"saknas":>10}   kvar (SDR)')
    for i, (lo, hi, name) in enumerate(BANDS):
        s_band = db(bo[i]) - db(br[i])
        flag = '  <-- saknas mycket' if s_band < 8 and bo[i] > bo.max() * 1e-3 else ''
        print(f'{name:<10}{db(bo[i]):>9.0f} {db(bm[i]):>9.0f} {db(br[i]):>9.0f}   {s_band:>6.1f} dB{flag}')

    print(f'\n=== PER {args.segment:.0f} SEKUNDER (var i låten saknas det) ===')
    seg = int(args.segment * sr)
    worst = []
    for a in range(0, n, seg):
        eo = (orig[a:a + seg] ** 2).sum()
        er = (res[a:a + seg] ** 2).sum()
        if eo < e_o * 1e-6:
            continue
        s = db(eo) - db(er)
        worst.append((s, a / sr))
        t0 = a / sr
        bar = '#' * max(0, min(30, int(s)))
        print(f'  {int(t0 // 60)}:{int(t0 % 60):02d}  {s:>5.1f} dB  {bar}')
    worst.sort()
    if worst:
        print('Sämst (mest saknas): ' + ', '.join(f'{int(t // 60)}:{int(t % 60):02d} ({s:.0f} dB)'
                                                  for s, t in worst[:3]))

    print('\n=== PER SPÅR (hur stor del av originalet varje står för) ===')
    rows = []
    for name, d in tracks.items():
        e = ((d * g) ** 2).sum()
        corr = float((orig * d).sum() / max(np.sqrt((orig ** 2).sum() * (d ** 2).sum()), 1e-20))
        rows.append((db(e) - db(e_o), corr, name))
    for share, corr, name in sorted(rows, reverse=True):
        print(f'  {share:>6.1f} dB  likhet med originalet {corr:>5.2f}   {name}')

    out = folder / '_rest.wav'
    sf.write(out, res, sr, subtype='PCM_24')
    print(f'\nResten (original minus spåren) sparad som {out}')
    print('Lyssna på den: det du hör där är det som saknas i spåren.')

    if args.rest_track:
        d2 = folder.parent / (folder.name + '_med_rest')
        d2.mkdir(exist_ok=True)
        for p in files:
            sf.write(d2 / p.name, tracks[p.name] * g, sr, subtype='PCM_24')
        sf.write(d2 / 'rest.wav', res, sr, subtype='PCM_24')
        print(f'Alla spår + resten som eget spår: {d2}')


if __name__ == '__main__':
    main()
