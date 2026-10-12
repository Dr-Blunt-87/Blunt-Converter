"""
cluster_stems_clap.py — slår ihop allt som LÅTER LIKA över alla stems.

BAKGRUND: AI-modellen delar upp låten i ~53 stems efter instrumentnamn, men
den gissar. I "East Of The River Nile" är det ETT munspel som modellen har
spridit över accordion, woodwind, wind, clarinet, flute, oboe, sax, trumpet,
organ, kick m.fl. De tidigare skripten (combine_families / cross_family)
jämför bara inom grupper och rensar genom att RADERA bitar, vilket ger hackigt
ljud och förlorar musik.

Det här skriptet litar INTE på stem-namnen. Det gör så här:
  1. Läser alla råa stems (hoppar över *_combined, *_clean, *_OVERLAP,
     *_crosscleaned samt 'drums' som är föräldraspåret till kick/snare/...).
  2. Delar varje stem i överlappande 1-sekundsfönster och låter CLAP lyssna på
     varje fönster (volymen jämnas ut först, så det är KLANGEN som räknas).
  3. Klustrar alla fönster från alla stems efter klang. Samma ljud hamnar i
     samma kluster oavsett vilken stem modellen lade det i.
  4. Bygger ett stereo-spår per kluster genom att lägga tillbaka varje fönster
     på sin ursprungliga tidpunkt (Hann-fönster med 50 % överlapp, så skarvarna
     är osynliga). INGET raderas: summan av alla kluster = summan av alla stems
     (skriptet kontrollerar det och skriver ut skillnaden).
  5. Gissar vad varje kluster är med CLAP:s text-förståelse (t.ex. "harmonica")
     och skriver cluster_report.txt som visar var allt kom ifrån.

ANVÄNDNING (kör på låtmappen med de RÅA stemsen, helst på GPU):
    python cluster_stems_clap.py "separation_results_mega/East Of The River Nile"

Justera vid behov (omkörning tar sekunder tack vare cluster_cache.npz):
    --threshold 0.35   lägre = fler, renare kluster. Högre = mer ihopslagning
    --clusters 12      tvinga fram exakt 12 kluster istället för tröskel
    --exclude drums    kommaseparerade stems som ska hoppas över
    --recompute        ignorera cachen och bädda in allt på nytt

Resultat läggs i undermappen  clusters/  (dina andra filer rörs inte).
"""
import argparse
import json
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import resample_poly
from scipy.signal import windows as sig_windows

# ----------------------------------------------------------------- inställningar
WINDOW_SECONDS = 1.0            # fönsterlängd; hopp = halva (50 % överlapp)
SILENCE_PEAK = 0.002            # fönster där inget ljud når över detta (~-54 dBFS) hoppas över
TARGET_RMS = 0.1                # fönstren volymjusteras till detta före CLAP (bara för analysen)
EXCLUDE_STEMS = {'drums'}       # föräldraspår som annars skulle dubbleras mot kick/snare/hh/toms
DERIVED_PARTS = ('_combined', '_clean', '_overlap', '_crosscleaned', 'cluster_')
DEFAULT_THRESHOLD = 0.35        # cosinusavstånd för att slå ihop kluster (justera efter lyssning)
MIN_CLUSTER_SECONDS = 4.0       # kluster med mindre ljud än så slås in i närmaste större
MAX_CLUSTER_SAMPLE = 4000       # max antal fönster som själva klustringen räknar på (resten tilldelas efteråt)
EMBED_BATCH = 32
OUTPUT_SUBFOLDER = 'clusters'

TEXT_LABELS = [
    'acoustic guitar', 'electric guitar', 'bass guitar', 'double bass', 'kick drum',
    'snare drum', 'hi-hat', 'tom drum', 'cymbal', 'shaker', 'tambourine',
    'congas and bongos', 'piano', 'electric piano', 'organ', 'harpsichord',
    'synthesizer pad', 'synthesizer lead', 'accordion', 'harmonica', 'flute',
    'clarinet', 'oboe', 'saxophone', 'trumpet', 'trombone', 'french horn', 'tuba',
    'violin', 'viola', 'cello', 'string section', 'harp', 'banjo', 'mandolin',
    'ukulele', 'sitar', 'marimba', 'vibraphone', 'bells', 'glockenspiel',
    'male singing voice', 'female singing voice', 'choir', 'talking',
]


# ----------------------------------------------------------------- hjälpfunktioner
def base_name(path):
    """'accordion (fanns så stämmer)' -> 'accordion' (funkar även med dina anteckningar i filnamnen)."""
    return re.split(r'[ (]', path.stem)[0]


def is_derived(path):
    n = path.stem.lower()
    return any(part in n for part in DERIVED_PARTS)


def find_stems(folder, exclude):
    stems, derived, excluded = [], [], []
    for p in sorted(folder.glob('*.wav')):
        if is_derived(p):
            derived.append(p.name)
        elif base_name(p).lower() in exclude:
            excluded.append(p.name)
        else:
            stems.append(p)
    return stems, derived, excluded


def load_stereo(path, target_sr=None):
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    if target_sr and sr != target_sr:
        x = resample_poly(x, target_sr, sr, axis=0).astype(np.float32)
        sr = target_sr
    return x, sr


def stem_length(path, sr_ref):
    info = sf.info(str(path))
    return int(round(info.frames * sr_ref / info.samplerate))


def layout(n, W):
    """Fönsterlayout så att varje riktigt sampel täcks av exakt två Hann-fönster."""
    H = W // 2
    n_win = (n + H - 1) // H + 1
    total = (n_win - 1) * H + W          # = (n_win + 1) * H
    return H, n_win, total


def pad(x, H, total):
    out = np.zeros((total, x.shape[1]), np.float32)
    out[H:H + len(x)] = x
    return out


def active_windows(xpad, n_win, H):
    peak = np.abs(xpad).max(axis=1)
    blocks = peak[:(n_win + 1) * H].reshape(n_win + 1, H).max(axis=1)
    win_peak = np.maximum(blocks[:-1], blocks[1:])
    return np.where(win_peak >= SILENCE_PEAK)[0]


def to_numpy(e):
    e = e.detach().cpu().numpy() if hasattr(e, 'detach') else np.asarray(e)
    return e.astype(np.float32)


def normalize_rows(a):
    a = np.nan_to_num(a)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)


def embed_windows(clap, xpad, act, W, H, sr, tmpdir):
    embs = []
    for b in range(0, len(act), EMBED_BATCH):
        paths = []
        for j, i in enumerate(act[b:b + EMBED_BATCH]):
            seg = xpad[i * H:i * H + W].mean(axis=1)
            rms = float(np.sqrt(np.mean(seg ** 2)))
            if rms > 1e-9:
                seg = seg * (TARGET_RMS / rms)
            p = Path(tmpdir) / f'w{j}.wav'
            sf.write(str(p), np.clip(seg, -1.0, 1.0), sr, subtype='PCM_16')
            paths.append(str(p))
        embs.append(to_numpy(clap.get_audio_embeddings(paths)))
    return normalize_rows(np.concatenate(embs))


# ----------------------------------------------------------------- cache
def signature(stems, W):
    return json.dumps({
        'W': W, 'peak': SILENCE_PEAK, 'rms': TARGET_RMS,
        'files': [[p.name, p.stat().st_size] for p in stems],
    })


def save_cache(path, sig, data):
    arrs = {'meta': np.array(sig)}
    for si, (act, E) in data.items():
        arrs[f'idx_{si}'] = act.astype(np.int32)
        arrs[f'emb_{si}'] = E
    np.savez_compressed(path, **arrs)


def load_cache(path, sig, n_stems):
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z['meta']) != sig:
                return None
            data = {}
            for si in range(n_stems):
                if f'idx_{si}' in z.files:
                    data[si] = (z[f'idx_{si}'].astype(np.int64), z[f'emb_{si}'])
            return data
    except Exception:
        return None


# ----------------------------------------------------------------- klustring
def cluster_windows(data, threshold, n_clusters, min_windows):
    """Returnerar (labels_per_stem, centroids). labels_per_stem[si] = array med klusternummer per fönster."""
    order = sorted(data)
    all_E = np.concatenate([data[si][1] for si in order])
    sizes = [len(data[si][1]) for si in order]
    offsets = np.concatenate([[0], np.cumsum(sizes)])

    rng = np.random.default_rng(0)
    cap = max(60, MAX_CLUSTER_SAMPLE // max(1, len(order)))
    sample = []
    for k, si in enumerate(order):
        n = sizes[k]
        sel = rng.choice(n, cap, replace=False) if n > cap else np.arange(n)
        sample.append(sel + offsets[k])
    sample = np.concatenate(sample)
    S = all_E[sample]

    if len(S) < 2:
        raw = np.ones(len(S), dtype=int)
    else:
        Z = linkage(S, method='average', metric='cosine')
        if n_clusters:
            raw = fcluster(Z, n_clusters, criterion='maxclust')
        else:
            raw = fcluster(Z, threshold, criterion='distance')

    ids = np.unique(raw)
    centroids = normalize_rows(np.stack([S[raw == c].mean(axis=0) for c in ids]))

    assign = np.argmax(all_E @ centroids.T, axis=1)

    # slå in pyttesmå kluster i närmaste större
    counts = np.bincount(assign, minlength=len(centroids))
    big = np.where(counts >= min_windows)[0]
    if 0 < len(big) < len(centroids):
        small_mask = np.isin(assign, np.where(counts < min_windows)[0])
        assign[small_mask] = big[np.argmax(all_E[small_mask] @ centroids[big].T, axis=1)]

    # numrera om efter storlek (störst = kluster 0) och räkna om centroider
    used = np.unique(assign)
    used = used[np.argsort(-np.bincount(assign, minlength=len(centroids))[used])]
    remap = {int(old): new for new, old in enumerate(used)}
    assign = np.array([remap[int(a)] for a in assign])
    centroids = normalize_rows(np.stack([all_E[assign == k].mean(axis=0) for k in range(len(used))]))

    labels = {si: assign[offsets[k]:offsets[k + 1]] for k, si in enumerate(order)}
    return labels, centroids


def fix_flips(idx, labels):
    """Ett ensamt fönster som hoppar till ett annat kluster mellan två likadana grannar flyttas till grannarna."""
    out = labels.copy()
    for i in range(1, len(idx) - 1):
        if (idx[i - 1] == idx[i] - 1 and idx[i + 1] == idx[i] + 1
                and labels[i - 1] == labels[i + 1] != labels[i]):
            out[i] = labels[i - 1]
    return out


def name_clusters(clap, centroids):
    try:
        prompts = [f'This is a sound of {l}.' for l in TEXT_LABELS]
        T = normalize_rows(to_numpy(clap.get_text_embeddings(prompts)))
        sims = centroids @ T.T
        result = []
        for row in sims:
            top = np.argsort(-row)[:3]
            result.append([(TEXT_LABELS[t], float(row[t])) for t in top])
        return result
    except Exception as e:
        print(f'  (kunde inte gissa namn med CLAP: {e})')
        return None


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-') or 'okand'


# ----------------------------------------------------------------- huvudprogram
def main():
    ap = argparse.ArgumentParser(description='Slå ihop allt som låter lika över alla stems.')
    ap.add_argument('folder', help='mapp med de råa stems (separation_results_mega/<låt>)')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument('--clusters', type=int, default=None)
    ap.add_argument('--window', type=float, default=WINDOW_SECONDS)
    ap.add_argument('--exclude', default=','.join(sorted(EXCLUDE_STEMS)))
    ap.add_argument('--min-seconds', type=float, default=MIN_CLUSTER_SECONDS)
    ap.add_argument('--recompute', action='store_true')
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f'Hittar inte mappen: {folder}')
        sys.exit(1)

    exclude = {e.strip().lower() for e in args.exclude.split(',') if e.strip()}
    stems, derived, excluded = find_stems(folder, exclude)
    if not stems:
        print('Hittade inga råa stems (.wav) i mappen.')
        sys.exit(1)
    print(f'Hittade {len(stems)} råa stems. Hoppar över {len(derived)} bearbetade filer'
          f' (_combined/_clean/_OVERLAP/_crosscleaned)'
          + (f' och uteslutna: {", ".join(excluded)}' if excluded else '') + '.')

    try:
        from msclap import CLAP
    except ImportError:
        print('Saknar msclap. Kör: pip install msclap')
        sys.exit(1)
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False
    if not use_cuda:
        print('OBS: hittar inget GPU - CLAP kör på CPU och blir MYCKET långsamt (kör helst på Vast.ai).')
    print('Laddar CLAP-modellen (tar en stund första gången)...')
    clap = CLAP(version='2023', use_cuda=use_cuda)

    sr_ref = sf.info(str(stems[0])).samplerate
    W = int(sr_ref * args.window) // 2 * 2
    H = W // 2
    sig = signature(stems, W)
    cache_file = folder / 'cluster_cache.npz'

    data = None
    if cache_file.exists() and not args.recompute:
        data = load_cache(cache_file, sig, len(stems))
        if data is not None:
            print(f'Använder sparade CLAP-inbäddningar ({cache_file.name}) - hoppar över lyssningen.')

    silent_stems = []
    if data is None:
        data = {}
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            for si, p in enumerate(stems):
                x, sr = load_stereo(p, sr_ref)
                _, n_win, total = layout(len(x), W)
                xpad = pad(x, H, total)
                act = active_windows(xpad, n_win, H)
                if len(act) == 0:
                    silent_stems.append(p.name)
                    print(f'  [{si + 1}/{len(stems)}] {base_name(p)}: tyst - hoppas över')
                    continue
                E = embed_windows(clap, xpad, act, W, H, sr_ref, tmp)
                data[si] = (act, E)
                print(f'  [{si + 1}/{len(stems)}] {base_name(p)}: {len(act)} fönster ({time.time() - t0:.0f}s)')
        save_cache(cache_file, sig, data)
    else:
        silent_stems = [stems[i].name for i in range(len(stems)) if i not in data]

    if sum(len(d[0]) for d in data.values()) < 2:
        print('För lite ljud att klustra.')
        sys.exit(1)

    print('\nKlustrar fönstren efter klang...')
    min_windows = max(1, int(round(args.min_seconds * sr_ref / H)))
    labels, centroids = cluster_windows(data, args.threshold, args.clusters, min_windows)
    K = len(centroids)
    for si in labels:
        labels[si] = fix_flips(data[si][0], labels[si])
    print(f'  {K} kluster')
    names = name_clusters(clap, centroids)

    # ---- bygg spåren
    n_max = max(stem_length(p, sr_ref) for p in stems)
    _, _, T_max = layout(n_max, W)
    hann = sig_windows.hann(W, sym=False).astype(np.float32)[:, None]
    buffers = [np.zeros((T_max, 2), np.float32) for _ in range(K)]
    ref = np.zeros((n_max, 2), np.float32)
    comp = [Counter() for _ in range(K)]
    per_stem = {}

    print('Bygger kluster-spåren (overlap-add)...')
    for si in sorted(data):
        p = stems[si]
        x, _ = load_stereo(p, sr_ref)
        _, _, total = layout(len(x), W)
        xpad = pad(x, H, total)
        idx, lab = data[si][0], labels[si]
        for i, k in zip(idx, lab):
            buffers[k][i * H:i * H + W] += xpad[i * H:i * H + W] * hann
            comp[k][base_name(p)] += 1
        per_stem[base_name(p) + f' [{p.name}]'] = Counter(int(k) for k in lab)
        ref[:len(x)] += x

    # kontroll: summan av klustren ska vara samma som summan av stemsen
    mixed = np.zeros((n_max, 2), np.float32)
    for b in buffers:
        mixed += b[H:H + n_max]
    ref_peak = float(np.abs(ref).max()) or 1.0
    diff_peak = float(np.abs(mixed - ref).max())
    diff_db = 20 * np.log10(max(diff_peak, 1e-12) / ref_peak)
    del mixed, ref

    peak = max(float(np.abs(b).max()) for b in buffers)
    scale = 0.99 / peak if peak > 0.99 else 1.0
    if scale < 1.0:
        print(f'  (toppen låg på {peak:.2f}, sänker alla spår lika mycket med faktor {scale:.2f} för att undvika klippning)')

    out_dir = folder / OUTPUT_SUBFOLDER
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob('cluster_*.wav'):   # städa bort spår från tidigare körning så inget räknas dubbelt
        old.unlink()
    written = []
    for k in range(K):
        if names:
            name = slug(names[k][0][0])
        else:
            name = slug(comp[k].most_common(1)[0][0]) if comp[k] else 'okand'
        fname = f'cluster_{k + 1:02d}_{name}.wav'
        sf.write(str(out_dir / fname), buffers[k][H:H + n_max] * scale, sr_ref, subtype='PCM_24')
        written.append(fname)
        print(f'  ✔ {fname}')

    # ---- rapport
    sec = H / sr_ref
    lines = [
        'CLUSTER-RAPPORT - allt som låter lika, sammanslaget över alla stems', '',
        f'Inställningar: fönster {args.window:.1f}s, '
        + (f'{args.clusters} kluster (tvingat)' if args.clusters else f'tröskel {args.threshold}'),
        f'Uteslutna stems: {", ".join(excluded) if excluded else "inga"}',
        f'Tysta stems (hoppades över): {", ".join(silent_stems) if silent_stems else "inga"}',
        f'Kontroll: skillnad mellan summan av klustren och summan av stemsen = {diff_db:.0f} dB under toppen '
        '(mycket negativt = inget har försvunnit)', '',
    ]
    for k in range(K):
        total_w = sum(comp[k].values())
        lines.append(f'KLUSTER {k + 1:02d}  ->  {written[k]}')
        if names:
            lines.append('  CLAP-gissning: ' + ', '.join(f'{l} ({s:.2f})' for l, s in names[k]))
        lines.append(f'  Ljudmängd: ca {total_w * sec:.0f}s')
        lines.append('  Kommer från: ' + ', '.join(
            f'{n} {100 * c / total_w:.0f}%' for n, c in comp[k].most_common(8)))
        lines.append('')
    lines.append('VART HAMNADE VARJE ORIGINAL-STEM?')
    for stem_name, cnt in per_stem.items():
        tot = sum(cnt.values())
        lines.append(f'  {stem_name} -> ' + ', '.join(
            f'kluster {k + 1:02d} {100 * c / tot:.0f}%' for k, c in cnt.most_common(4)))
    (out_dir / 'cluster_report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'\nKontroll: summan av klustren avviker {diff_db:.0f} dB under toppen från summan av stemsen.')
    print(f'Se {OUTPUT_SUBFOLDER}/cluster_report.txt för var allt kom ifrån.')
    print('Tips: för många/för få kluster? Kör om med --threshold (lägre = fler) eller --clusters N. Tar sekunder.')
    print('\nKlart!')


if __name__ == '__main__':
    main()
