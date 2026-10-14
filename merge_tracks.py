#!/usr/bin/env python3
"""
merge_tracks.py - slår ihop färdiga kluster-spår efter nummer.

Exempel (kör efter cluster_stems_clap.py):
  python merge_tracks.py "separation_results_mega/Richie Spice - Ghetto Girl/clusters" \
      "01+20=lead-vocal" "10+17=tjejkor" --drop 14

  NN+NN=namn   summerar spåren (numren från filnamnen, t.ex. cluster_10_...wav)
               och skriver dem som ett nytt spår med det namnet
  --drop NN    lämnar bort spåret
  Spår som inte nämns kopieras som de är.

Resultatet hamnar i <mapp>/final/ (originalen rörs inte).
Nivåerna ändras inte. Om summan skulle klippa sänks ALLA spår lika mycket.
"""
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def find_tracks(folder):
    tracks = {}
    for sub in (folder, folder / 'svaga'):
        if not sub.is_dir():
            continue
        for p in sorted(sub.glob('cluster_*.wav')):
            m = re.match(r'cluster_(\d+)_', p.name)
            if m:
                tracks[int(m.group(1))] = p
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('merges', nargs='*', help='t.ex. 10+17=tjejkor')
    ap.add_argument('--drop', default='', help='kommaseparerade nummer att lämna bort')
    args = ap.parse_args()

    folder = Path(args.folder)
    tracks = find_tracks(folder)
    if not tracks:
        print('Hittar inga cluster_*.wav i', folder)
        sys.exit(1)

    drop = {int(x) for x in re.split(r'[,\s]+', args.drop) if x.strip()}
    used = set(drop)
    jobs = []
    for spec in args.merges:
        if '=' not in spec:
            print('Fel format (ska vara 10+17=namn):', spec)
            sys.exit(1)
        nums, name = spec.split('=', 1)
        ids = [int(n) for n in nums.split('+') if n.strip()]
        for i in ids:
            if i not in tracks:
                print(f'Spår {i:02d} finns inte. Finns: {sorted(tracks)}')
                sys.exit(1)
        used.update(ids)
        jobs.append((ids, name.strip().replace(' ', '-')))

    out_dir = folder / 'final'
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob('*.wav'):
        old.unlink()

    results = []  # (filename, data, sr)
    for ids, name in jobs:
        total, sr = None, None
        for i in ids:
            data, s = sf.read(tracks[i], dtype='float64', always_2d=True)
            if total is None:
                total, sr = data.copy(), s
            else:
                n = max(len(total), len(data))
                if len(total) < n:
                    total = np.pad(total, ((0, n - len(total)), (0, 0)))
                if len(data) < n:
                    data = np.pad(data, ((0, n - len(data)), (0, 0)))
                total += data
        label = '+'.join(f'{i:02d}' for i in ids)
        results.append((f'{label}_{name}.wav', total, sr))
    for i, p in tracks.items():
        if i in used:
            continue
        data, s = sf.read(p, dtype='float64', always_2d=True)
        results.append((p.name, data, s))

    peak = max(float(np.max(np.abs(d))) for _, d, _ in results)
    scale = 1.0 if peak <= 0.99 else 0.99 / peak
    if scale < 1.0:
        print(f'(toppen låg på {peak:.2f}, sänker alla spår lika mycket med faktor {scale:.2f})')
    for fname, data, sr in sorted(results, key=lambda r: r[0]):
        sf.write(out_dir / fname, data * scale, sr, subtype='PCM_24')
        db = 20 * np.log10(max(float(np.sqrt(np.mean((data * scale) ** 2))), 1e-9))
        print(f'  ✔ {fname}  (rms {db:.0f} dB)')
    if drop:
        print('Lämnade bort:', ', '.join(f'{d:02d}' for d in sorted(drop)))
    print(f'\nKlart! Resultat i {out_dir}')


if __name__ == '__main__':
    main()
