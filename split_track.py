#!/usr/bin/env python3
"""
split_track.py - flyttar ett eller flera tidsstycken ur ett kluster-spår till ett eget spår.

Exempel:
  python split_track.py "<mapp med cluster_*.wav>" 03 "0:42-end" --new 21
      -> cluster_21_utdrag.wav  = 03 mellan 0:42 och slutet
         cluster_22_rest.wav    = 03 utanför det stycket (--new 21 ger rest = 22)
  Flera stycken: "1:10-1:14,2:30-2:33"
  Tid skrivs som sekunder (42), m:ss (0:42) eller m:ss.s (0:42.5). "end" = slutet.

Originalet rörs inte. Använd sedan merge_tracks.py med de nya numren
och lämna bort originalnumret (--drop). Summan utdrag + rest = originalet.
"""
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_time(text, total):
    text = text.strip().lower()
    if text in ('end', 'slut'):
        return total
    if text in ('start', ''):
        return 0.0
    parts = text.split(':')
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('track', help='spårnummer, t.ex. 03')
    ap.add_argument('ranges', help='t.ex. "0:42-end" eller "1:10-1:14,2:30-2:33"')
    ap.add_argument('--new', type=int, default=21,
                    help='nummer för utdraget (resten får numret efter), standard 21')
    ap.add_argument('--fade', type=float, default=0.03, help='fade i sekunder vid kanterna')
    args = ap.parse_args()

    folder = Path(args.folder)
    num = int(args.track)
    src = None
    for sub in (folder, folder / 'svaga'):
        if sub.is_dir():
            for p in sorted(sub.glob(f'cluster_{num:02d}_*.wav')):
                src = p
    if src is None:
        print(f'Hittar inget spår med nummer {num:02d} i {folder}')
        sys.exit(1)

    data, sr = sf.read(src, dtype='float64', always_2d=True)
    total = len(data) / sr
    mask = np.zeros(len(data))
    shown = []
    for part in args.ranges.split(','):
        if '-' not in part:
            print('Fel format på tidsstycke:', part)
            sys.exit(1)
        a, b = part.split('-', 1)
        t0, t1 = parse_time(a, total), parse_time(b, total)
        t0, t1 = max(0.0, t0), min(total, t1)
        if t1 <= t0:
            print('Tomt tidsstycke:', part)
            sys.exit(1)
        i0, i1 = int(t0 * sr), int(t1 * sr)
        mask[i0:i1] = 1.0
        shown.append(f'{t0:.1f}-{t1:.1f} s')

    # mjuka kanter så det inte knäpper
    k = max(1, int(args.fade * sr))
    kernel = np.hanning(2 * k + 1)
    kernel /= kernel.sum()
    padded = np.pad(mask, (k, k), mode='edge')
    mask = np.clip(np.convolve(padded, kernel, mode='same')[k:-k], 0.0, 1.0)

    piece = data * mask[:, None]
    rest = data * (1.0 - mask)[:, None]

    n1, n2 = args.new, args.new + 1
    out1 = folder / f'cluster_{n1:02d}_utdrag.wav'
    out2 = folder / f'cluster_{n2:02d}_rest.wav'
    sf.write(out1, piece, sr, subtype='PCM_24')
    sf.write(out2, rest, sr, subtype='PCM_24')

    def rms_db(x):
        return 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-9))

    print(f'Spår {num:02d} ({src.name}, {total:.1f} s)')
    print(f'  utdrag ({", ".join(shown)}) -> {out1.name}  ({rms_db(piece):.0f} dB)')
    print(f'  resten                  -> {out2.name}  ({rms_db(rest):.0f} dB)')
    print(f'Använd nu {n1:02d} och {n2:02d} i merge_tracks.py och lägg {num:02d} i --drop.')


if __name__ == '__main__':
    main()
