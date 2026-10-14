"""
cluster_stems_clap.py (v5) — slår ihop allt som LÅTER LIKA över alla stems.

NYTT I v5: --pin lägger stems som modellen redan döpt rätt (sång, bas, kick, hi-hat) direkt i egna
spår utan att klustra dem. Standard: lead-vocal+vocal, back-vocal, bass, kick, hh. --pin none = av.
Eget val:  --pin "sång=lead-vocal+vocal,stämmor=back-vocal"

Från v4: --levels visar hur högt varje stem ligger mot den starkaste stemmen, och
--gate-db 30 slänger fönster som ligger 30 dB eller mer under det starkaste stemmet i samma
sekund (modellens "spöklåt": en svag, filtrerad kopia av hela låten i instrument som inte
finns). Spöklåten får annars alla stems att låta lika och hamna i ETT stort kluster.


BAKGRUND: AI-modellen delar upp låten i ~53 stems efter instrumentnamn, men
den gissar. I "East Of The River Nile" är det ETT munspel som modellen har
spridit över accordion, woodwind, wind, clarinet, flute, oboe, sax, trumpet,
organ, kick m.fl. Dessutom ligger en svag, filtrerad "spöklåt" (hela låten,
lågt) i instrument som inte ens finns i låten (banjo, bells, french-horn...).
De tidigare skripten jämför bara inom grupper och rensar genom att RADERA
bitar, vilket ger hackigt ljud och förlorar musik.

Det här skriptet litar INTE på stem-namnen. Det gör så här:
  1. Läser alla råa stems (hoppar över *_combined, *_clean, *_OVERLAP,
     *_crosscleaned samt 'drums' som är föräldraspåret till kick/snare/...).
  2. Delar varje stem i överlappande 1-sekundsfönster och låter CLAP lyssna på
     varje fönster (volymen jämnas ut först, så det är KLANGEN som räknas).
  3. Klustrar alla fönster från alla stems efter klang. Samma ljud hamnar i
     samma kluster oavsett vilken stem modellen lade det i.
  4. Bygger ett stereo-spår per kluster genom att lägga tillbaka varje fönster
     på sin ursprungliga tidpunkt (Hann-fönster med 50 % överlapp, så skarvarna
     är osynliga). Modellens stems innehåller ofta KOPIOR av samma ljud (samma
     munspel i tjugo stems), så kopior får inte adderas: i varje kluster och
     tidsfönster behålls den starkaste kopian och av de övriga läggs bara till
     det som den inte redan innehåller. Olika ljud i samma kluster (t.ex. en
     vänster- och en högergitarr) adderas som vanligt.
  5. Läckage MELLAN kluster (munspel under kick-slag, spöklåten i instrument
     som inte finns) dämpas med en mjuk frekvensmask: i varje tid/frekvens-ruta
     behåller det kluster som är STARKAST ljudet, och de andra klustrens
     svagare kopior dämpas. Kluster som blir nästan tysta efter det läggs i
     undermappen svaga/ (de är spöklåt, inte instrument).
  6. Döper klustren efter vilka stems de mest består av och skriver
     cluster_report.txt som visar var allt kom ifrån.

ANVÄNDNING (kör på låtmappen med de RÅA stemsen, helst på GPU):
    python cluster_stems_clap.py "separation_results_mega/East Of The River Nile" --threshold 0.42

Justera vid behov (omkörning tar sekunder tack vare cluster_cache.npz):
    --scan             visar hur många kluster olika trösklar ger (skriver inga filer)
    --threshold 0.35   lägre = fler, renare kluster. Högre = mer ihopslagning
    --clusters 12      tvinga fram exakt 12 kluster istället för tröskel
    --mask-power 3     hur hårt läckage mellan kluster dämpas (0 = av, 2 = mjukt, 4 = hårt)
    --ghost-db -26     kluster svagare än så (dB mot starkaste) flyttas till svaga/
    --clap-names       låt CLAP gissa instrumentnamn i rapporten (laddar CLAP, tar en halv minut)
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
from scipy.signal import istft, resample_poly, stft
from scipy.signal import windows as sig_windows

# ----------------------------------------------------------------- inställningar
WINDOW_SECONDS = 1.0            # fönsterlängd; hopp = halva (50 % överlapp)
SILENCE_PEAK = 0.002            # fönster där inget ljud når över detta (~-54 dBFS) hoppas över
TARGET_RMS = 0.1                # fönstren volymjusteras till detta före CLAP (bara för analysen)
EXCLUDE_STEMS = {'drums'}       # föräldraspår som annars skulle dubbleras mot kick/snare/hh/toms
DERIVED_PARTS = ('_combined', '_clean', '_overlap', '_crosscleaned', 'cluster_')
DEFAULT_THRESHOLD = 0.35        # cosinusavstånd för att slå ihop kluster (justera efter lyssning / --scan)
MIN_CLUSTER_SECONDS = 4.0       # kluster med mindre ljud än så slås in i närmaste större
MAX_CLUSTER_SAMPLE = 4000       # max antal fönster som själva klustringen räknar på (resten tilldelas efteråt)
EMBED_BATCH = 32
OUTPUT_SUBFOLDER = 'clusters'
GHOST_SUBFOLDER = 'svaga'
SCAN_THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
MASK_POWER = 3.0                # 2 = Wiener (mjukt), högre = skarpare val av starkaste klustret
GHOST_DB = -26.0                # kluster svagare än så mot det starkaste räknas som spöklåt
MASK_NPERSEG = 2048             # FFT-storlek för frekvensmasken (75 % överlapp)
GATE_DB = 0.0                   # 0 = av. Fönster som ligger så många dB under starkaste stemmet slängs (spöklåt)

PIN_DEFAULT = 'lead-vocal=lead-vocal+vocal,back-vocal=back-vocal,bass=bass,kick=kick,hh=hh'

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


class StemReader:
    """Läser valfritt stycke ur en stem utan att ladda hela filen (nollor utanför filen, alltid stereo)."""

    def __init__(self, path, sr_ref):
        info = sf.info(str(path))
        if info.samplerate == sr_ref:
            self.f, self.arr, self.n = sf.SoundFile(str(path)), None, info.frames
        else:                                   # ovanligt: annan samplingsfrekvens, ladda och resampla
            self.f, (self.arr, _) = None, load_stereo(path, sr_ref)
            self.n = len(self.arr)

    def get(self, start, length):
        out = np.zeros((length, 2), np.float32)
        a, b = max(start, 0), min(start + length, self.n)
        if b <= a:
            return out
        if self.arr is not None:
            seg = self.arr[a:b]
        else:
            self.f.seek(a)
            seg = self.f.read(b - a, dtype='float32', always_2d=True)
        if seg.shape[1] == 1:
            seg = np.repeat(seg, 2, axis=1)
        elif seg.shape[1] > 2:
            seg = seg[:, :2]
        out[a - start:b - start] = seg
        return out

    def close(self):
        if self.f is not None:
            self.f.close()


def dedupe(segs):
    """Slår ihop flera stems' bidrag för samma kluster och tidsfönster utan att addera kopior.

    Starkaste bidraget behålls. Av varje övrigt bidrag b läggs bara till det som INTE redan
    finns i det som behållits: b - alpha*kept, där alpha är hur stor del av b som är en kopia.
    En kopia (b = c*kept) ger 0 tillägg, ett helt annat ljud (okorrelerat) läggs till i sin helhet.
    Returnerar (resultat, energi_in, energi_ut)."""
    energies = [float(np.sum(s * s)) for s in segs]
    e_in = sum(energies)
    if len(segs) == 1:
        return segs[0], e_in, e_in
    order = np.argsort(energies)[::-1]
    kept = segs[order[0]].copy()
    for j in order[1:]:
        b = segs[j]
        aa = float(np.sum(kept * kept))
        alpha = 0.0 if aa < 1e-12 else min(1.0, max(0.0, float(np.sum(kept * b)) / aa))
        kept += b - alpha * kept
    return kept, e_in, float(np.sum(kept * kept))


def soft_mask_clusters(buffers, sr, power, chunk_s=10.0, margin_s=1.0):
    """Dämpar läckage mellan kluster.

    Varje kluster-spår delas i tid/frekvens-rutor (STFT). I varje ruta får klustret en andel
    mask = styrka^power / summan av alla klusters styrka^power. Det starkaste klustret behåller
    nästan allt, och de andras svagare kopior (läckage, spöklåt) dämpas. Samma mask används för
    båda kanalerna så stereobilden inte ändras. Körs i bitar för att spara minne."""
    K = len(buffers)
    T = buffers[0].shape[0]
    nper = MASK_NPERSEG
    nover = nper * 3 // 4
    outs = [np.zeros_like(b) for b in buffers]
    chunk, margin = int(chunk_s * sr), int(margin_s * sr)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        a, b = max(0, start - margin), min(T, end + margin)
        specs, powers = [], []
        for buf in buffers:
            _, _, Z = stft(buf[a:b].T, fs=sr, nperseg=nper, noverlap=nover,
                           window='hann', boundary='zeros', padded=True)      # (2, F, frames)
            specs.append(Z)
            mag = np.sqrt((np.abs(Z) ** 2).sum(axis=0))                        # (F, frames)
            powers.append(mag ** power)
        P = np.stack(powers)
        denom = P.sum(axis=0) + 1e-20
        for k in range(K):
            mask = (P[k] / denom)[None]
            _, y = istft(specs[k] * mask, fs=sr, nperseg=nper, noverlap=nover,
                         window='hann', boundary=True)
            y = y.T.astype(np.float32)
            if len(y) < b - a:
                y = np.concatenate([y, np.zeros((b - a - len(y), 2), np.float32)])
            lo = start - a
            outs[k][start:end] = y[lo:lo + (end - start)]
    return outs


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


_clap = None


def ensure_torchaudio():
    """Om torchaudio inte går att ladda (t.ex. CUDA-version matchar inte torch)
    läggs en minimal ersättare in. CLAP använder bara torchaudio.load och Resample."""
    try:
        import torchaudio  # noqa: F401
        return
    except Exception as e:
        print('(torchaudio går inte att ladda: %s - använder inbyggd ersättare)' % str(e).split('.')[0][:90])
    import types
    import numpy as _np
    import torch as _torch
    import soundfile as _sf
    from math import gcd as _gcd

    def _load(path, *a, **k):
        data, sr = _sf.read(path, dtype='float32', always_2d=True)
        return _torch.from_numpy(_np.ascontiguousarray(data.T)), sr

    class _Resample:
        def __init__(self, orig_freq, new_freq, *a, **k):
            self.o, self.n = int(orig_freq), int(new_freq)

        def __call__(self, x):
            from scipy.signal import resample_poly
            g = _gcd(self.o, self.n)
            y = resample_poly(x.detach().cpu().numpy(), self.n // g, self.o // g, axis=-1)
            return _torch.from_numpy(_np.ascontiguousarray(y.astype('float32')))

    ta = types.ModuleType('torchaudio')
    tr = types.ModuleType('torchaudio.transforms')
    ta.load = _load
    tr.Resample = _Resample
    ta.transforms = tr
    ta.__version__ = '0.0-shim'
    sys.modules['torchaudio'] = ta
    sys.modules['torchaudio.transforms'] = tr


def get_clap():
    """Laddar CLAP först när den behövs (så --scan och omkörningar från cache går fort)."""
    global _clap
    if _clap is None:
        ensure_torchaudio()
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
        _clap = CLAP(version='2023', use_cuda=use_cuda)
    return _clap


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


def parse_pins(text):
    """'namn=stem1+stem2,namn2=stem3' -> {'namn': {'stem1','stem2'}, ...}. 'none' = inga."""
    pins = {}
    if not text or text.strip().lower() in ('none', 'nej', 'off'):
        return pins
    for part in text.split(','):
        if '=' not in part:
            continue
        name, members = part.split('=', 1)
        pins[name.strip()] = {m.strip().lower() for m in members.split('+') if m.strip()}
    return pins


def window_rms(stems, sr_ref, W, H, n_max):
    """RMS per fönster (samma fönsterindex som i cachen) för varje stem. Returnerar {stem_index: array}."""
    _, n_win, total = layout(n_max, W)
    out = {}
    for si, p in enumerate(stems):
        x, _ = load_stereo(p, sr_ref)
        xp = pad(x, H, total)
        blocks = (xp[:(n_win + 1) * H] ** 2).reshape(n_win + 1, H, 2).mean(axis=(1, 2))
        out[si] = np.sqrt((blocks[:-1] + blocks[1:]) / 2.0).astype(np.float64)
    return out


def apply_gate(data, rms, gate_db):
    """Slänger fönster som ligger gate_db eller mer under det starkaste stemmet i samma fönster."""
    n = max(len(r) for r in rms.values())
    loud = np.zeros(n)
    for r in rms.values():
        loud[:len(r)] = np.maximum(loud[:len(r)], r)
    thr = loud * 10 ** (-gate_db / 20.0)
    kept, dropped = {}, {}
    for si, (idx, E) in data.items():
        r = rms[si]
        rr = np.where(idx < len(r), r[np.minimum(idx, len(r) - 1)], 0.0)
        m = rr >= thr[np.minimum(idx, n - 1)]
        dropped[si] = int((~m).sum())
        if m.any():
            kept[si] = (idx[m], E[m])
    return kept, dropped


def print_levels(stems, data, rms):
    n = max(len(r) for r in rms.values())
    loud = np.zeros(n)
    for r in rms.values():
        loud[:len(r)] = np.maximum(loud[:len(r)], r)
    rows = []
    for si, (idx, E) in data.items():
        rel = 20 * np.log10(np.maximum(rms[si][idx], 1e-9) / np.maximum(loud[idx], 1e-9))
        rows.append((float(np.median(rel)), base_name(stems[si]), len(idx),
                     100 * float(np.mean(rel >= -20)), 100 * float(np.mean(rel >= -30)),
                     100 * float(np.mean(rel >= -40))))
    rows.sort(reverse=True)
    print('\nNivå per stem mot det starkaste stemmet i samma sekund (0 dB = starkast):')
    print(f'  {"stem":<18}{"fönster":>8}{"median dB":>11}{">=-20dB":>9}{">=-30dB":>9}{">=-40dB":>9}')
    for med, name, cnt, a, b, c in rows:
        print(f'  {name:<18}{cnt:>8}{med:>11.0f}{a:>8.0f}%{b:>8.0f}%{c:>8.0f}%')
    print('\nStems med låg median-nivå och nästan alla fönster under -30 dB är troligen bara spöklåt.')
    print('Välj sedan t.ex.  --gate-db 30  (eller 25/35) och kör klustringen.')


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


def level_db(values):
    ref = max(max(values), 1e-9)
    return [20 * np.log10(max(v, 1e-9) / ref) for v in values]


# ----------------------------------------------------------------- huvudprogram
def main():
    ap = argparse.ArgumentParser(description='Slå ihop allt som låter lika över alla stems.')
    ap.add_argument('folder', help='mapp med de råa stems (separation_results_mega/<låt>)')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument('--clusters', type=int, default=None)
    ap.add_argument('--scan', action='store_true', help='visa antal kluster per tröskel och avsluta')
    ap.add_argument('--mask-power', type=float, default=MASK_POWER)
    ap.add_argument('--ghost-db', type=float, default=GHOST_DB)
    ap.add_argument('--clap-names', action='store_true', help='låt CLAP gissa instrumentnamn i rapporten')
    ap.add_argument('--window', type=float, default=WINDOW_SECONDS)
    ap.add_argument('--exclude', default=','.join(sorted(EXCLUDE_STEMS)))
    ap.add_argument('--min-seconds', type=float, default=MIN_CLUSTER_SECONDS)
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--gate-db', type=float, default=GATE_DB, help='släng fönster som ligger så många dB under starkaste stemmet (0 = av, prova 30)')
    ap.add_argument('--levels', action='store_true', help='visa nivå per stem och avsluta')
    ap.add_argument('--pin', default=PIN_DEFAULT, help='stems som tas direkt från stem-namnet, t.ex. "sång=lead-vocal+vocal,bas=bass" (none = av)')
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
        clap = get_clap()
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

    n_max = max(stem_length(p, sr_ref) for p in stems)
    gated_out, dropped_windows = [], 0
    if args.levels or args.gate_db > 0:
        print('Mäter nivåer per stem...')
        rms = window_rms(stems, sr_ref, W, H, n_max)
        if args.levels:
            print_levels(stems, data, rms)
            return
        before = {si: len(d[0]) for si, d in data.items()}
        data, dropped = apply_gate(data, rms, args.gate_db)
        dropped_windows = sum(dropped.values())
        gated_out = [stems[si].name for si in before if si not in data]
        print(f'Spöklåts-filter ({args.gate_db:g} dB): slängde {dropped_windows} av {sum(before.values())} stem-fönster'
              + (f'; helt bortgallrade stems: {len(gated_out)}' if gated_out else '') + '.')

    if sum(len(d[0]) for d in data.values()) < 2:
        print('För lite ljud att klustra.')
        sys.exit(1)

    min_windows = max(1, int(round(args.min_seconds * sr_ref / H)))

    pins = parse_pins(args.pin)
    pin_of = {}
    for si in data:
        bn = base_name(stems[si]).lower()
        for g, members in pins.items():
            if bn in members:
                pin_of[si] = g
                break
    group_names = [g for g in pins if g in set(pin_of.values())]
    G = len(group_names)
    rest = {si: d for si, d in data.items() if si not in pin_of}
    emb_dim = next(iter(data.values()))[1].shape[1]

    def cluster_rest(threshold, n_clusters):
        if not rest:
            return {}, np.zeros((0, emb_dim), np.float32)
        if sum(len(d[0]) for d in rest.values()) < 2:
            allE = np.concatenate([d[1] for d in rest.values()])
            return ({si: np.zeros(len(d[0]), dtype=int) for si, d in rest.items()},
                    normalize_rows(allE.mean(axis=0, keepdims=True)))
        return cluster_windows(rest, threshold, n_clusters, min_windows)

    if G:
        print('Fastlåsta grupper (tas direkt från stem-namnet): '
              + ', '.join(f'{g} ({"+".join(sorted(pins[g]))})' for g in group_names))

    if args.scan:
        print('\nAntal kluster vid olika trösklar för övriga stems (inga filer skrivs):')
        for t in SCAN_THRESHOLDS:
            _, cent = cluster_rest(t, None)
            print(f'  --threshold {t:.2f}  ->  {len(cent)} kluster' + (f' (+ {G} fastlåsta)' if G else ''))
        print('\nVälj en tröskel och kör sen utan --scan, t.ex.:  python cluster_stems_clap.py "<mapp>" --threshold 0.42')
        return

    print('\nKlustrar fönstren efter klang...')
    labels_c, centroids_c = cluster_rest(args.threshold, args.clusters)
    labels = {}
    for si, lab in labels_c.items():
        labels[si] = fix_flips(data[si][0], lab) + G
    pinned_cent = []
    for gi, g in enumerate(group_names):
        sis = [si for si in pin_of if pin_of[si] == g]
        for si in sis:
            labels[si] = np.full(len(data[si][0]), gi, dtype=int)
        pinned_cent.append(normalize_rows(np.concatenate([data[si][1] for si in sis]).mean(axis=0, keepdims=True))[0])
    centroids = np.vstack([np.array(pinned_cent), centroids_c]) if G else centroids_c
    K = len(centroids)
    kname = {gi: g for gi, g in enumerate(group_names)}
    print(f'  {K} spår ({G} fastlåsta + {len(centroids_c)} efter klang)')
    names = name_clusters(get_clap(), centroids) if args.clap_names else None

    # ---- bygg spåren
    _, _, T_max = layout(n_max, W)
    hann = sig_windows.hann(W, sym=False).astype(np.float32)[:, None]
    buffers = [np.zeros((T_max, 2), np.float32) for _ in range(K)]
    readers = {si: StemReader(stems[si], sr_ref) for si in data}

    by_window = {}                      # tidsfönster -> kluster -> vilka stems som bidrar
    comp = [Counter() for _ in range(K)]
    per_stem = {}
    for si in sorted(data):
        idx, lab = data[si][0], labels[si]
        per_stem[base_name(stems[si]) + f' [{stems[si].name}]'] = Counter(int(k) for k in lab)
        for i, k in zip(idx, lab):
            by_window.setdefault(int(i), {}).setdefault(int(k), []).append(si)
            comp[int(k)][base_name(stems[si])] += 1

    print(f'Bygger kluster-spåren ({len(by_window)} tidsfönster, kopior inom varje kluster adderas inte)...')
    e_in = e_out = 0.0
    for i in sorted(by_window):
        for k, sis in by_window[i].items():
            segs = [readers[si].get(i * H - H, W) for si in sis]
            seg, ein, eout = dedupe(segs)
            e_in += ein
            e_out += eout
            buffers[k][i * H:i * H + W] += seg * hann
    for r in readers.values():
        r.close()
    dup_ratio = e_in / max(e_out, 1e-12)

    rms_of = lambda b: float(np.sqrt(np.mean(b[H:H + n_max] ** 2)))
    lvl_before = level_db([rms_of(b) for b in buffers])
    if args.mask_power > 0 and K > 1:
        print(f'Dämpar läckage mellan klustren (mask-power {args.mask_power:g})...')
        buffers = soft_mask_clusters(buffers, sr_ref, args.mask_power)
    lvl_after = level_db([rms_of(b) for b in buffers])
    ghost = [d < args.ghost_db for d in lvl_after]

    peak = max(float(np.abs(b).max()) for b in buffers)
    scale = 0.99 / peak if peak > 0.99 else 1.0
    if scale < 1.0:
        print(f'  (toppen låg på {peak:.2f}, sänker alla spår lika mycket med faktor {scale:.2f} för att undvika klippning)')

    out_dir = folder / OUTPUT_SUBFOLDER
    ghost_dir = out_dir / GHOST_SUBFOLDER
    out_dir.mkdir(exist_ok=True)
    ghost_dir.mkdir(exist_ok=True)
    for old in list(out_dir.glob('cluster_*.wav')) + list(ghost_dir.glob('cluster_*.wav')):
        old.unlink()                    # städa bort spår från tidigare körning
    written = []
    for k in range(K):
        top = [n for n, _ in comp[k].most_common(2)] or ['okand']
        fname = f'cluster_{k + 1:02d}_{slug(kname[k]) if k in kname else slug("-".join(top))}.wav'
        target = (ghost_dir if ghost[k] else out_dir) / fname
        sf.write(str(target), buffers[k][H:H + n_max] * scale, sr_ref, subtype='PCM_24')
        written.append(f'{GHOST_SUBFOLDER}/{fname}' if ghost[k] else fname)
        print(f'  {"~" if ghost[k] else "✔"} {written[-1]}  ({lvl_after[k]:.0f} dB)')
    n_ghost = sum(ghost)
    if n_ghost:
        print(f'  ({n_ghost} kluster var nästan tysta efter läckagedämpningen och ligger i {GHOST_SUBFOLDER}/)')

    # ---- rapport
    sec = H / sr_ref
    lines = [
        'CLUSTER-RAPPORT - allt som låter lika, sammanslaget över alla stems', '',
        f'Inställningar: fönster {args.window:.1f}s, '
        + (f'{args.clusters} kluster (tvingat)' if args.clusters else f'tröskel {args.threshold}')
        + f', mask-power {args.mask_power:g}, spöklåt under {args.ghost_db:g} dB',
        f'Uteslutna stems: {", ".join(excluded) if excluded else "inga"}',
        f'Tysta stems (hoppades över): {", ".join(silent_stems) if silent_stems else "inga"}',
        (f'Spöklåts-filter: {args.gate_db:g} dB, {dropped_windows} stem-fönster slängda; helt bortgallrade stems: '
         + (", ".join(gated_out) if gated_out else "inga")) if args.gate_db > 0 else 'Spöklåts-filter: av',
        f'Dubbletter: stemsen innehöll tillsammans {dup_ratio:.1f}x mer ljudenergi än kluster-spåren; '
        'resten var kopior av samma ljud i flera stems', '',
    ]
    for k in range(K):
        total_w = sum(comp[k].values())
        lines.append(f'KLUSTER {k + 1:02d}  ->  {written[k]}' + ('   [SPÖKLÅT/läckage]' if ghost[k] else '')
                     + ('   [FASTLÅST efter stem-namn]' if k in kname else ''))
        if names:
            lines.append('  CLAP-gissning: ' + ', '.join(f'{l} ({s:.2f})' for l, s in names[k]))
        lines.append(f'  Nivå mot starkaste klustret: {lvl_after[k]:.0f} dB efter läckagedämpning '
                     f'({lvl_before[k]:.0f} dB före)')
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

    print(f'\nDubbletter: stemsen hade {dup_ratio:.1f}x mer ljudenergi än kluster-spåren (resten var kopior).')
    print(f'Se {OUTPUT_SUBFOLDER}/cluster_report.txt för var allt kom ifrån.')
    print('Tips: för hårt/mjukt? Prova --mask-power 2 (mjukare) eller 4 (hårdare). För många/få kluster? Kör --scan.')
    print('\nKlart!')


if __name__ == '__main__':
    main()
