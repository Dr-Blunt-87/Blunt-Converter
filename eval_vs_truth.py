#!/usr/bin/env python3
"""
eval_vs_truth.py - mäter hur bra Demucs delar upp en låt jämfört med DINA RIKTIGA spår.

Du har exporterade stems (kick, snare, bas, piano ...) från Studio One.
Skriptet:
 1. summerar alla spår till en mix (mix.wav)
 2. kör Demucs (htdemucs_6s) på mixen
 3. grupperar dina riktiga spår (trummor, bas, piano, gitarr, sång, övrigt)
 4. räknar SDR (dB) per grupp: högt = nära facit, lågt/negativt = dåligt

Användning:
  python eval_vs_truth.py "D:\\TRANING\\mappen"
Valfritt:
  --skip-demucs  använd redan körd Demucs-mapp (--est <mapp med bass.wav, drums.wav ...>)
"""
import sys, argparse, subprocess
from pathlib import Path
import numpy as np
import soundfile as sf

GROUPS = {
    'drums':  ['kick', 'snare', 'hihat', 'hat', 'clap', 'crash', 'perc', 'cymbal', 'tom', 'drum', 'trum'],
    'bass':   ['bass', 'bas', '808', 'sub'],
    'piano':  ['piano', 'keys', 'rhodes'],
    'guitar': ['guitar', 'gitarr', 'gtr'],
    'vocals': ['vocal', 'vox', 'voice', 'sang', 'sång', 'rap', 'lead'],
}


def group_of(name):
    n = name.lower()
    for g, keys in GROUPS.items():
        if any(k in n for k in keys):
            return g
    return 'other'


def load(p):
    d, sr = sf.read(str(p), dtype='float32', always_2d=True)
    d = d[:, :2] if d.shape[1] >= 2 else np.repeat(d, 2, axis=1)
    return d, sr


def sdr(ref, est):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    num = float((ref ** 2).sum())
    den = float(((ref - est) ** 2).sum())
    if num < 1e-12:
        return None
    return 10 * np.log10(num / max(den, 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--est', default='')
    ap.add_argument('--skip-demucs', action='store_true')
    a = ap.parse_args()
    folder = Path(a.folder)
    files = sorted(p for p in folder.glob('*.wav') if p.stem.lower() not in ('mix',))
    if not files:
        print('Hittar inga wav-filer i', folder); sys.exit(1)

    tracks, sr = {}, None
    for p in files:
        d, s = load(p)
        sr = sr or s
        tracks[p.name] = d
    n = max(len(d) for d in tracks.values())
    mix = np.zeros((n, 2), dtype='float32')
    refs = {g: np.zeros((n, 2), dtype='float32') for g in list(GROUPS) + ['other']}
    print('\nSå här grupperar jag dina spår:')
    for name, d in tracks.items():
        g = group_of(name)
        refs[g][:len(d)] += d
        mix[:len(d)] += d
        print(f'  {g:7s} <- {name}')
    pk = float(np.abs(mix).max()) or 1.0
    k = 0.9 / pk if pk > 0.9 else 1.0
    mix *= k
    for g in refs:
        refs[g] *= k
    out = folder.parent / (folder.name + '_eval')
    out.mkdir(exist_ok=True)
    sf.write(out / 'mix.wav', mix, sr, subtype='PCM_16')

    if a.est:
        est_dir = Path(a.est)
    else:
        if not a.skip_demucs:
            print('\nKör Demucs på mixen (kan ta några minuter)...')
            subprocess.run(['demucs', '-n', 'htdemucs_6s', '--out', str(out / 'demucs'), str(out / 'mix.wav')], check=True)
        cands = list((out / 'demucs').glob('*/mix'))
        if not cands:
            print('Hittar inget Demucs-resultat.'); sys.exit(1)
        est_dir = cands[0]

    print(f'\nJämför med {est_dir}\n')
    print(f'{"grupp":8s} {"facit-nivå":>11s}   {"SDR (dB)":>9s}')
    rows = []
    for g in ['drums', 'bass', 'piano', 'guitar', 'vocals', 'other']:
        f = est_dir / f'{g}.wav'
        if not f.exists():
            continue
        e, _ = load(f)
        ref = refs[g]
        lvl = 10 * np.log10(max(float((ref ** 2).mean()), 1e-12))
        v = sdr(ref, e)
        if v is None:
            print(f'{g:8s} {"(tomt)":>11s}   {"-":>9s}   (inga facit-spår i gruppen)')
            continue
        rows.append(v)
        print(f'{g:8s} {lvl:9.1f} dB   {v:9.1f}')
    if rows:
        print(f'\nSnitt: {np.mean(rows):.1f} dB   (över 6 = hyfsat, under 3 = svagt, negativt = fel spår)')
    print(f'\nKlart. Mixen och Demucs ligger i {out}')


if __name__ == '__main__':
    main()
