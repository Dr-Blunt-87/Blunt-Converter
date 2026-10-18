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
  --chain <mapp> mäter kedjans spår (t.ex. clusters_mix_app) i stället för Demucs
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


CHAIN_MAP = {
    'drums':  ['kick', 'snare', 'hh', 'hihat', 'percussion', 'toms', 'timpani', 'congas', 'tambourine', 'drums', 'triangle'],
    'bass':   ['bass'],            # bass, double-bass
    'piano':  ['piano', 'keys', 'organ', 'harpsichord'],
    'guitar': ['guitar', 'dobro', 'banjo', 'mandolin', 'ukulele'],
    'vocals': ['vocal'],
}


def chain_group(label):
    l = label.lower()
    # blandade kluster (t.ex. double-bass-toms) -> första ordets grupp
    for g in ['vocals', 'drums', 'bass', 'piano', 'guitar']:
        if any(k in l for k in CHAIN_MAP[g]):
            return g
    return 'other'


def load_chain(folder):
    est, names = {}, {}
    for p in sorted(Path(folder).glob('*.wav')):
        label = p.stem.split('_', 2)[-1] if p.stem.startswith('cluster_') else p.stem.split('_', 1)[-1]
        d, _ = load(p)
        g = chain_group(label)
        if g in est:
            n = max(len(est[g]), len(d))
            est[g] = np.pad(est[g], ((0, n - len(est[g])), (0, 0))) + np.pad(d, ((0, n - len(d)), (0, 0)))
        else:
            est[g] = d
        names.setdefault(g, []).append(p.name)
    return est, names


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
    ap.add_argument('--chain', default='')
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

    chain_est = None
    if a.chain:
        chain_est, cn = load_chain(a.chain)
        print('\nKedjans spår grupperade så här:')
        for g, ns in cn.items():
            print(f'  {g:7s} <- ' + ', '.join(ns))
        est_dir = Path(a.chain)
        # saknas facit i en grupp (t.ex. inga gitarrer i låten) räknas kedjans spår där som 'other'
        for g in list(chain_est):
            if g != 'other' and float((refs[g] ** 2).sum()) < 1e-9:
                add = chain_est.pop(g)
                if 'other' in chain_est:
                    nn = max(len(add), len(chain_est['other']))
                    chain_est['other'] = np.pad(chain_est['other'], ((0, nn - len(chain_est['other'])), (0, 0))) + np.pad(add, ((0, nn - len(add)), (0, 0)))
                else:
                    chain_est['other'] = add
                print(f'  ({g} saknar facit -> räknas som other)')
    elif a.est:
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
        if chain_est is not None:
            if g not in chain_est:
                continue
            e = chain_est[g]
        else:
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
