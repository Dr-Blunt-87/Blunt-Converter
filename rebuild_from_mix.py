#!/usr/bin/env python3
"""
rebuild_from_mix.py - delar upp ORIGINALET efter våra spår, så att inget går förlorat.

Våra färdiga spår används bara som karta över vem som äger vilken del av ljudet
(per frekvens och tidpunkt). Sedan delas originalmixen efter den kartan.
Resultat: spåren summerar exakt till originalet, och det "svajiga filterljudet"
från hårda masker blir mindre eftersom inget klipps bort.

Exempel:
  python rebuild_from_mix.py "C:\\...\\Richie Spice - Ghetto Girl.mp3" "C:\\...\\clusters\\final"

Valfritt:
  --power 3      hur skarpt en ruta tillhör ett spår (1 = väldigt mjukt, 2 = mjukt, 4 = hårt)
  --smooth-time 7  jämnar ut kartan över tid (antal rutor, ca 23 ms var). Minskar wah-wah/pumpning
  --smooth-freq 3  jämnar ut kartan över frekvens. Minskar bubblor
  --nfft 2048    fönsterstorlek: 2048 = skarpare slag (trummor), 8192 = fylligare toner (bas, gitarr, sång)
  --tag namn     sparar i  <mapp>_mix_<namn>  så att du kan jämföra flera varianter
  --keep-rest    sparar det som inget spår gör anspråk på som eget spår (rest.wav)
                 (annars fördelas det på spåren)
  --boost snare=3,kick=2  förstärker spåret kort vid varje slag (hittas med onset), så snare/kick inte hamnar i sången
  --exclude 05   hoppar över spår vars filnamn börjar så (komma-separerat)
Resultat hamnar i  <mapp>_mix/
"""
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, resample_poly, fftconvolve
from scipy.ndimage import uniform_filter

NPERSEG, NOVERLAP = 4096, 3072


def load_audio(path):
    try:
        return sf.read(path, dtype='float32', always_2d=True)
    except Exception:
        tmp = Path(tempfile.mkdtemp()) / 'orig.wav'
        r = subprocess.run(['ffmpeg', '-y', '-i', str(path), '-ar', '44100', '-ac', '2', str(tmp)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            print('Kunde inte läsa originalet. Behöver ffmpeg i PATH för mp3.')
            sys.exit(1)
        return sf.read(tmp, dtype='float32', always_2d=True)


def to_stereo(x):
    return np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x[:, :2]


def find_lag(orig, mix, sr, max_s=0.5, seconds=40):
    n = int(min(len(orig), len(mix), seconds * sr))
    a = orig[:n].mean(axis=1).astype('float64')
    b = mix[:n].mean(axis=1).astype('float64')
    c = fftconvolve(a, b[::-1], mode='full')
    mid = len(b) - 1
    m = int(max_s * sr)
    seg = c[mid - m: mid + m + 1]
    return -(int(np.argmax(np.abs(seg))) - m)


def shift(x, lag, n):
    if lag > 0:
        x = x[lag:]
    elif lag < 0:
        x = np.pad(x, ((-lag, 0), (0, 0)))
    if len(x) < n:
        x = np.pad(x, ((0, n - len(x)), (0, 0)))
    return x[:n]


def spec(x, sr):
    _, _, Z = stft(x, sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return Z.astype(np.complex64)


def onset_window(track_mono, sr, frame_times, pre=0.01, post=0.09):
    """1 i tidsrutor strax efter varje slag i spåret (annars 0)."""
    try:
        import librosa
        on = librosa.onset.onset_detect(y=track_mono.astype('float32'), sr=sr, units='time', backtrack=False)
    except Exception as e:
        print('  (onset-boost hoppas över:', e, ')')
        return np.zeros(len(frame_times), dtype='float32')
    w = np.zeros(len(frame_times), dtype='float32')
    for t0 in on:
        w[(frame_times >= t0 - pre) & (frame_times <= t0 + post)] = 1.0
    return w


def spec_t(x, sr):
    f, tt, Z = stft(x, sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return tt, Z.astype(np.complex64)


def set_nfft(nfft):
    global NPERSEG, NOVERLAP
    NPERSEG = int(nfft)
    NOVERLAP = NPERSEG * 3 // 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('original')
    ap.add_argument('folder')
    ap.add_argument('--power', type=float, default=3.0)
    ap.add_argument('--smooth-time', type=int, default=7)
    ap.add_argument('--smooth-freq', type=int, default=3)
    ap.add_argument('--nfft', type=int, default=2048)
    ap.add_argument('--tag', default='')
    ap.add_argument('--keep-rest', action='store_true')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--boost', default='')
    args = ap.parse_args()

    set_nfft(args.nfft)
    boost = {}
    for part in [x for x in args.boost.split(',') if x.strip()]:
        k, _, v = part.partition('=')
        boost[k.strip().lower()] = float(v or 3)
    folder = Path(args.folder)
    skip = [s.strip().lower() for s in args.exclude.split(',') if s.strip()]
    files = [p for p in sorted(folder.glob('*.wav'))
             if not p.name.startswith('_') and p.name != 'rest.wav'
             and not any(p.name.lower().startswith(s) for s in skip)]
    if not files:
        print('Hittar inga wav-filer i', folder)
        sys.exit(1)

    orig, sr = load_audio(args.original)
    orig = to_stereo(orig)
    n = len(orig)
    sm = (max(1, args.smooth_freq), max(1, args.smooth_time))   # (frekvens, tid)
    print(f'Original: {n / sr:.1f} s. Spår: {len(files)}. mask-power {args.power:g}, '
          f'utjämning tid {sm[1]} / frekvens {sm[0]}, fönster {args.nfft}')

    tracks = []
    for p in files:
        d, s = sf.read(p, dtype='float32', always_2d=True)
        d = to_stereo(d)
        if s != sr:
            d = resample_poly(d, sr, s, axis=0).astype('float32')
        tracks.append(d)
    mix = np.zeros((max(len(t) for t in tracks), 2), dtype='float32')
    for d in tracks:
        mix[:len(d)] += d
    lag = find_lag(orig, mix, sr)
    if lag:
        print(f'Justerar tidsförskjutning: {lag / sr * 1000:.1f} ms')
    tracks = [shift(t, lag, n) for t in tracks]
    mixs = shift(mix, lag, n)
    g = float((orig * mixs).sum() / max((mixs * mixs).sum(), 1e-20))
    print(f'Spåren multipliceras med {g:.2f} för att matcha originalets nivå')
    tracks = [t * g for t in tracks]
    del mix, mixs

    out = [np.zeros((n, 2), dtype='float32') for _ in tracks]
    rest_out = np.zeros((n, 2), dtype='float32')
    eps = 1e-30
    for ch in range(2):
        print(f'Kanal {ch + 1}/2: räknar...')
        X = spec(orig[:, ch], sr)
        ref = float(np.percentile(np.abs(X), 99.9)) or 1.0
        mags = []
        total = np.zeros(X.shape, dtype='float32')
        S_sum = np.zeros(X.shape, dtype=np.complex64)
        for ti, t in enumerate(tracks):
            ftimes, S = spec_t(t[:, ch], sr)
            S_sum += S
            m = np.abs(S) / ref
            if sm != (1, 1):
                m = uniform_filter(m, size=sm)
            f = next((v for k, v in boost.items() if k in files[ti].name.lower()), 1.0)
            if f != 1.0:
                w = onset_window(t[:, ch], sr, ftimes)
                m = m * (1.0 + (f - 1.0) * w[None, :]).astype('float32')
            m = np.maximum(m, 0) ** args.power
            mags.append(m.astype('float32'))
            total += mags[-1]
        R = X - S_sum
        rest_m = np.abs(R) / ref
        if sm != (1, 1):
            rest_m = uniform_filter(rest_m, size=sm)
        rest_m = (np.maximum(rest_m, 0) ** args.power).astype('float32')
        if args.keep_rest:
            denom = total + rest_m + eps
        else:
            denom = total + eps
        for k in range(len(tracks)):
            Y = X * (mags[k] / denom)
            _, y = istft(Y, sr, nperseg=NPERSEG, noverlap=NOVERLAP)
            y = y[:n]
            out[k][:len(y), ch] = y.astype('float32')
            mags[k] = None
        if args.keep_rest:
            Y = X * (rest_m / denom)
            _, y = istft(Y, sr, nperseg=NPERSEG, noverlap=NOVERLAP)
            y = y[:n]
            rest_out[:len(y), ch] = y.astype('float32')
        del X, total, S_sum, R, denom, mags

    allout = out + ([rest_out] if args.keep_rest else [])
    peak = max(float(np.max(np.abs(a))) for a in allout)
    scale = 1.0 if peak <= 0.99 else 0.99 / peak
    if scale < 1.0:
        print(f'(toppen låg på {peak:.2f}, sänker alla spår lika mycket med faktor {scale:.2f})')
    out_dir = folder.parent / (folder.name + '_mix' + (('_' + args.tag) if args.tag else ''))
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob('*.wav'):
        old.unlink()
    for p, a in zip(files, out):
        sf.write(out_dir / p.name, a * scale, sr, subtype='PCM_24')
        db = 20 * np.log10(max(float(np.sqrt(np.mean((a * scale) ** 2))), 1e-9))
        print(f'  ✔ {p.name}  (rms {db:.0f} dB)')
    if args.keep_rest:
        sf.write(out_dir / 'rest.wav', rest_out * scale, sr, subtype='PCM_24')
        print('  ✔ rest.wav')

    total_out = sum(out)
    err = orig - total_out - (rest_out if args.keep_rest else 0)
    sdr = 10 * np.log10(max(float((orig ** 2).sum()), 1e-20) / max(float((err ** 2).sum()), 1e-20))
    print(f'\nSumman av spåren mot originalet: {sdr:.1f} dB (högt = inget förlorat)')
    print(f'Klart! Resultat i {out_dir}')


if __name__ == '__main__':
    main()
