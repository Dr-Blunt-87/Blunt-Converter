#!/usr/bin/env python3
"""
eval_etiketter.py - testar Demucs mot dina egna etiketter (etiketter.csv i låtmappen).

  python eval_etiketter.py "D:\\TRANING\\låt2"

Gör så här:
 1. läser etiketter.csv och sorterar spåren i lead-vocal / back-vocal / beat (tysta hoppas över)
 2. summerar alla spåren till en mix och kör Demucs (htdemucs_6s) på den
 3. mäter: hur nära är Demucs 'vocals' till dina sångspår, hur nära är resten till din beat,
    och hur mycket av ledsången respektive bakgrundssången som hamnar i Demucs sångspår
 4. sparar lead.wav, back.wav, beat.wav och mix.wav i <mapp>_eval så att du kan lyssna
Valfritt: --skip-demucs  (om Demucs redan körts)   --est <mapp>
"""
import sys, csv, argparse, subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft


def norm_label(raw):
    s = (raw or '').strip().lower()
    if not s or s in ('tyst', 'okänd'):
        return None
    if 'beat' in s:
        return 'beat'
    if any(k in s for k in ('back', 'dubb', 'ref', 'fill', 'voal')):
        return 'back-vocal'
    if any(k in s for k in ('rap', 'vers', 'lead')):
        return 'lead-vocal'
    return 'other'


def load(p):
    d, sr = sf.read(str(p), dtype='float32', always_2d=True)
    d = d[:, :2] if d.shape[1] >= 2 else np.repeat(d, 2, axis=1)
    return d, sr


def sdr(ref, est):
    n = min(len(ref), len(est)); ref, est = ref[:n], est[:n]
    num = float((ref ** 2).sum()); den = float(((ref - est) ** 2).sum())
    return None if num < 1e-12 else 10 * np.log10(num / max(den, 1e-20))


def mags(x, sr):
    _, _, Z = stft(x.mean(axis=1), sr, nperseg=2048, noverlap=1024)
    return np.abs(Z).astype('float32')


def share(truth, est, sr):
    t, e = mags(truth, sr), mags(est, sr)
    n = min(t.shape[1], e.shape[1])
    tot = float((t[:, :n] ** 2).sum())
    return 0.0 if tot < 1e-12 else float((np.minimum(t[:, :n], e[:, :n]) ** 2).sum()) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder'); ap.add_argument('--est', default=''); ap.add_argument('--skip-demucs', action='store_true')
    a = ap.parse_args()
    folder = Path(a.folder)
    csvf = folder / 'etiketter.csv'
    if not csvf.exists():
        print('Hittar ingen etiketter.csv i', folder); sys.exit(1)
    groups, sr, names = {}, None, {}
    for r in csv.reader(csvf.open(encoding='utf-8-sig'), delimiter=';'):
        if len(r) < 3 or r[0] == 'fil':
            continue
        g = norm_label(r[2])
        p = folder / r[0]
        if g is None or not p.exists():
            continue
        d, s = load(p); sr = sr or s
        n = max(len(d), len(groups[g]) if g in groups else 0)
        cur = groups.get(g, np.zeros((n, 2), 'float32'))
        cur = np.pad(cur, ((0, n - len(cur)), (0, 0))) + np.pad(d, ((0, n - len(d)), (0, 0)))
        groups[g] = cur
        names.setdefault(g, []).append(r[0])
    print('\nSå här sorterade jag dina spår:')
    for g, ns in names.items():
        print(f'  {g}: {len(ns)} spår')
    n = max(len(v) for v in groups.values())
    for g in groups:
        groups[g] = np.pad(groups[g], ((0, n - len(groups[g])), (0, 0)))
    mix = sum(groups.values())
    pk = float(np.abs(mix).max()) or 1.0
    k = 0.9 / pk if pk > 0.9 else 1.0
    mix = mix * k
    for g in groups:
        groups[g] = groups[g] * k
    out = folder.parent / (folder.name + '_eval'); out.mkdir(exist_ok=True)
    sf.write(out / 'mix.wav', mix, sr, subtype='PCM_16')
    for g, v in groups.items():
        sf.write(out / f'facit_{g}.wav', v, sr, subtype='PCM_16')

    if a.est:
        est_dir = Path(a.est)
    else:
        if not a.skip_demucs:
            print('\nKör Demucs på mixen...')
            subprocess.run(['demucs', '-n', 'htdemucs_6s', '--out', str(out / 'demucs'), str(out / 'mix.wav')], check=True)
        c = list((out / 'demucs').glob('*/mix'))
        if not c:
            print('Hittar inget Demucs-resultat.'); sys.exit(1)
        est_dir = c[0]
    est = {p.stem: load(p)[0] for p in est_dir.glob('*.wav')}
    voc = est.get('vocals', np.zeros((n, 2), 'float32'))
    rest = sum(np.pad(v, ((0, max(0, len(voc) - len(v))), (0, 0)))[:len(voc)] for k2, v in est.items() if k2 != 'vocals')
    vocals_truth = groups.get('lead-vocal', 0) + groups.get('back-vocal', 0)
    print(f'\nJämför med {est_dir}\n')
    v1 = sdr(vocals_truth, voc) if not np.isscalar(vocals_truth) else None
    v2 = sdr(groups['beat'], rest) if 'beat' in groups else None
    if v1 is not None: print(f'Sång (lead+back) mot Demucs vocals : {v1:5.1f} dB')
    if v2 is not None: print(f'Beat mot Demucs allt-utom-sång     : {v2:5.1f} dB')
    print('\nVar tog dina sångspår vägen? (andel av ljudet som finns i varje Demucs-spår)')
    for g in ('lead-vocal', 'back-vocal', 'beat'):
        if g not in groups:
            continue
        rows = sorted(((share(groups[g], v, sr), kk) for kk, v in est.items()), reverse=True)
        print(f'  {g:11s} -> ' + ', '.join(f'{kk} {100*s:.0f}%' for s, kk in rows[:3] if s > 0.02))
    print(f'\nKlart. Lyssna på facit_*.wav och mix.wav i {out}')


if __name__ == '__main__':
    main()
