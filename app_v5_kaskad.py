"""
Blunt Converter - Backend
Tar emot en MP4/MP3/WAV ELLER en YouTube-länk, extraherar ljud,
separerar stems (Demucs), och ger dig antingen:
  - MIDI-noter per stem (Basic Pitch), eller
  - rena ljud-stems (MP3/WAV)
Filerna sparas på servern och listas via /files så de kan laddas ner
individuellt eller dras direkt in i en DAW.

KÖR:
    python app.py
Servern startar på http://127.0.0.1:5000
"""
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import subprocess
import uuid
import shutil
import sys
import importlib.util
import zipfile
import tempfile
import threading
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, sosfiltfilt
import pretty_midi
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter, uniform_filter1d, median_filter

app = Flask(__name__)
CORS(app)  # tillåter att index.html (öppnad direkt som fil) pratar med servern

BASE = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE / 'output_midi'
OUTPUT_ROOT.mkdir(exist_ok=True)

ALL_STEMS = ['bass', 'drums', 'guitar', 'piano', 'vocals', 'other']
DEMUCS_MODEL = 'htdemucs_6s'  # 6-stems-modellen: bas, trummor, gitarr, piano, vokaler, övrigt

# Inställningar från labbet (Vast-testerna): bygg om varje stem från ORIGINALET
# så att spåren summerar till låten och inget "filterljud" blir kvar.
REBUILD_ENABLED = os.environ.get('BLUNT_REBUILD', '1') != '0'
REBUILD_POWER = 4.0      # hur skarpt en ruta tillhör ett spår
REBUILD_NFFT = 2048      # fönsterstorlek (skarpare slag)
REBUILD_SMOOTH_TIME = 5  # jämnar ut över tid
REBUILD_SMOOTH_FREQ = 3  # jämnar ut över frekvens

# v4: sången tas först av MDX-NET (bättre än Demucs på sång), sedan körs Demucs 6s på musiken som blir kvar.
# Kräver audio-separator. Hittas det inte körs Demucs som förut (inget går sönder).
MDX_ENABLED = os.environ.get('BLUNT_MDX', '1') != '0'
MDX_MODEL = 'UVR-MDX-NET-Voc_FT.onnx'
# De höga frekvenserna (över ca 5 kHz) finns annars bara i trummor/sång efter Demucs -> gitarr/piano låter burkiga.
# Med detta delas de ut igen: hållna toner -> gitarr/piano/övrigt, slag -> trummor. Sång och bas rörs inte.
HF_RESTORE = os.environ.get('BLUNT_HF', '1') != '0'
HF_START = 4000.0        # under detta rörs inget
HF_FULL = 6000.0         # över detta gäller den nya fördelningen fullt ut
HF_TONAL_BAND = (2500.0, 5000.0)   # var vi mäter vilken tonal stem som är "närvarande"
# Volymen per stem jämnades förut ut till samma nivå -> tysta stems förstärktes och svagt läckage hördes.
# Nu behålls riktiga nivåer (stems summerar till låten). Sätt BLUNT_NORMALIZE=1 för gamla beteendet.
NORMALIZE_STEMS = os.environ.get('BLUNT_NORMALIZE', '0') == '1'

# GM-trumnoter för den extra trumuppdelningen (kick/snare/hihat/cymbal)
DRUM_GM_NOTES = {
    'kick': 36,
    'snare': 38,
    'hihat_closed': 42,
    'hihat_open': 46,
    'cymbal': 49,
}

# Håller koll på pågående/klara konverteringsjobb, så frontend kan fråga /progress/<task_id>
JOBS = {}
JOBS_LOCK = threading.Lock()


def set_progress(task_id, percent=None, message=None, status=None, files=None, folder=None, bpm=None, chords=None):
    with JOBS_LOCK:
        job = JOBS.setdefault(task_id, {})
        if percent is not None:
            job['percent'] = percent
        if message is not None:
            job['message'] = message
        if status is not None:
            job['status'] = status
        if files is not None:
            job['files'] = files
        if folder is not None:
            job['folder'] = folder
        if bpm is not None:
            job['bpm'] = bpm
        if chords is not None:
            job['chords'] = chords


def detect_bpm(audio_path):
    """Räknar ut låtens tempo (BPM) med librosa. Fungerar på i princip
    vilket ljudformat som helst (mp3/wav/m4a osv), oavsett källa
    (uppladdad fil eller YouTube-nedladdning)."""
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=120)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm_value = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
        return round(bpm_value, 1)
    except Exception as e:
        print('BPM-detektering misslyckades:', e)
        return None


# ---------------------------------------------------------------------------
# ACKORD + TONART (som Fadr Plus / Orra Key Pro)
# Princip: bara de spår som spelar ackord (gitarr + piano + övrigt/stråk) analyseras,
# i bandet 30 Hz - 2,5 kHz. Bas, trummor och sång lämnas utanför. Basen används som
# facit för ackordets grundton. Stämningen räknas ut först. Tystnad/brus räknas inte
# när tonarten bestäms. Tonart väljs bland 24 dur/moll (Krumhansl-Kessler-profiler),
# skalan (dorisk, frygisk m.fl.) läses från de toner som skiljer skalorna åt.
# Ackordbyten jämnas ut med Viterbi.
# ---------------------------------------------------------------------------
CHORD_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']      # interna namn
CHORD_NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
CHORD_SR = 22050
CHORD_HOP = 512
CHORD_LOW_HZ = 30.0
CHORD_HIGH_HZ = 2500.0
CHORD_STEMS = ('guitar', 'piano', 'other')   # spår som spelar ackord (other = stråk, pads m.m.)
CHORD_BASS_WEIGHT = 0.10                     # hur mycket basens ton styr ackordets grundton
CHORD_COMPRESS = 20.0                        # logg-komprimering av tonklasserna
CHORD_LOOP = 0.85                            # hur trögt ackord byts (högre = färre byten)
CHORD_MIX_LOW_HZ = 120.0                     # vid analys av hela mixen: klipp bort bas/kick
CHORD_KEY_BONUS = 0.03                       # liten fördel för ackord som hör till tonarten
CHORD_N_PENALTY = 0.35                       # ju högre, desto svårare att välja "inget ackord"
CHORD_N_MIN_QUIET = 2.0                      # "inget ackord" via tystnad kräver minst så här många sekunder
CHORD_SEVENTH_COST = 0.15                    # 7-ackord väljs bara när sjundetonen hörs tydligt
CHORD_SCALE_MARGIN = 0.06                    # hur mycket bättre en särskild skala måste vara än dur/moll
CHORD_INTERVALS = {'': [0, 4, 7], 'm': [0, 3, 7], 'dim': [0, 3, 6],
                   '7': [0, 4, 7, 10], 'maj7': [0, 4, 7, 11], 'm7': [0, 3, 7, 10], 'm7b5': [0, 3, 6, 10]}

KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]   # Krumhansl-Kessler
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
SCALES_MAJOR = {'dur': [0, 2, 4, 5, 7, 9, 11], 'lydisk': [0, 2, 4, 6, 7, 9, 11],
                'mixolydisk': [0, 2, 4, 5, 7, 9, 10], 'frygisk dominant': [0, 1, 4, 5, 7, 8, 10]}
SCALES_MINOR = {'moll': [0, 2, 3, 5, 7, 8, 10], 'dorisk': [0, 2, 3, 5, 7, 9, 10], 'frygisk': [0, 1, 3, 5, 7, 8, 10],
                'harmonisk moll': [0, 2, 3, 5, 7, 8, 11], 'melodisk moll': [0, 2, 3, 5, 7, 9, 11]}
CAMELOT_MAJOR = {0: 8, 7: 9, 2: 10, 9: 11, 4: 12, 11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7}
CAMELOT_MINOR = {9: 8, 4: 9, 11: 10, 6: 11, 1: 12, 8: 1, 3: 2, 10: 3, 5: 4, 0: 5, 7: 6, 2: 7}
ROMAN = ['I', 'bII', 'II', 'bIII', 'III', 'IV', 'bV', 'V', 'bVI', 'VI', 'bVII', 'VII']


def _chord_load(path):
    y, _ = librosa.load(str(path), sr=CHORD_SR, mono=True)
    return y


def _chord_band(y, low, high):
    nyq = CHORD_SR / 2.0
    sos = butter(2, [low / nyq, min(high / nyq, 0.99)], btype='band', output='sos')
    return sosfiltfilt(sos, y)


def _fmt_time(t):
    return f'{int(t // 60)}:{t % 60:05.2f}'


def _split_chord(label):
    root_len = 2 if len(label) > 1 and label[1] == '#' else 1
    return CHORD_NOTES.index(label[:root_len]), label[root_len:]


def _corr(a, b):
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def estimate_key(chroma_mean):
    """Poängsätter alla 24 tonarter. Returnerar [(tonika, 'dur'|'moll', korrelation)] bäst först."""
    res = []
    for t in range(12):
        v = np.roll(chroma_mean, -t)           # v[0] = tonikans energi
        res.append((t, 'dur', _corr(v, KK_MAJOR)))
        res.append((t, 'moll', _corr(v, KK_MINOR)))
    res.sort(key=lambda x: -x[2])
    return res


def pick_scale(chroma_mean, tonic, mode):
    """Väljer skala bland de som delar tonika: läser från toner som skiljer dem åt."""
    c = chroma_mean / (np.sum(chroma_mean) + 1e-9)
    v = np.roll(c, -tonic)
    cands = SCALES_MAJOR if mode == 'dur' else SCALES_MINOR
    default = 'dur' if mode == 'dur' else 'moll'
    scores = {}
    for name, ivs in cands.items():
        m = np.zeros(12, dtype=bool)
        m[ivs] = True
        scores[name] = float(v[m].sum() - v[~m].sum())
    best = max(scores, key=scores.get)
    if scores[best] - scores[default] < CHORD_SCALE_MARGIN:
        best = default
    return best, cands[best]


def _use_flats(tonic, mode):
    flat_major = {5, 10, 3, 8, 1}        # F, Bb, Eb, Ab, Db
    flat_minor = {2, 7, 0, 5, 10, 3}     # Dm, Gm, Cm, Fm, Bbm, Ebm
    return tonic in (flat_major if mode == 'dur' else flat_minor)


def _nm(pc, flat):
    return (CHORD_NOTES_FLAT if flat else CHORD_NOTES)[pc % 12]


def display_chord(label, flat):
    if label == 'N':
        return 'N'
    root, suf = _split_chord(label)
    return _nm(root, flat) + suf


def roman_numeral(label, tonic):
    if label == 'N':
        return ''
    root, suf = _split_chord(label)
    r = ROMAN[(root - tonic) % 12]
    if suf in ('m', 'dim', 'm7', 'm7b5'):
        r = r.lower()
    return r + {'': '', 'm': '', 'dim': '°', '7': '7', 'maj7': 'maj7', 'm7': '7', 'm7b5': 'ø'}[suf]


def describe_key(tonic, mode, scale):
    flat = _use_flats(tonic, mode)
    name = _nm(tonic, flat)
    if scale in ('dur', 'moll'):
        txt = f'{name} {"dur" if mode == "dur" else "moll"}'
    else:
        txt = f'{name} {scale}'
    cam = (CAMELOT_MAJOR if mode == 'dur' else CAMELOT_MINOR)[tonic]
    cam = f'{cam}{"B" if mode == "dur" else "A"}'
    rel_t, rel_m = ((tonic + 9) % 12, 'moll') if mode == 'dur' else ((tonic + 3) % 12, 'dur')
    rel = f'{_nm(rel_t, _use_flats(rel_t, rel_m))} {rel_m}'
    return txt, cam, rel, flat


def analyze_chords(chord_wavs, bass_wav=None, mix_wav=None, sevenths=False):
    """Returnerar dict (tempo, segments, källa, tonart, stämning) eller None om inget kunde analyseras."""
    source = 'stems'
    ys = [_chord_load(p) for p in chord_wavs if p and Path(p).exists()]
    ymix = _chord_load(mix_wav) if mix_wav and Path(mix_wav).exists() else None
    if ys:
        n = min(len(v) for v in ys)
        y = np.sum([v[:n] for v in ys], axis=0)
        y = librosa.effects.harmonic(_chord_band(y, CHORD_LOW_HZ, CHORD_HIGH_HZ), margin=2)
    elif ymix is not None:
        source = 'mix'
        y = librosa.effects.harmonic(_chord_band(ymix, CHORD_MIX_LOW_HZ, CHORD_HIGH_HZ), margin=4)
    else:
        return None
    if not np.any(np.abs(y) > 1e-5):
        return None

    # 1. stämning först. Mäts i halvtoner (12 per oktav) så att hela +-50 cent syns,
    #    och omräknas till 36-delars-skalan som tonklasserna använder.
    try:
        tune_semi = float(librosa.estimate_tuning(y=y, sr=CHORD_SR, bins_per_octave=12))
    except Exception:
        tune_semi = 0.0
    tuning_cents = tune_semi * 100.0
    tuning = tune_semi * 3.0

    # 2. tonklasser (chroma) och slag
    cq = dict(sr=CHORD_SR, hop_length=CHORD_HOP, fmin=librosa.note_to_hz('C1'), bins_per_octave=36, tuning=tuning)
    chroma = librosa.feature.chroma_cqt(y=y, n_octaves=7, **cq)
    nfr = chroma.shape[1]
    y_beat = ymix if ymix is not None else y
    tempo, bf = librosa.beat.beat_track(y=y_beat, sr=CHORD_SR, hop_length=CHORD_HOP, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])

    # basens toner
    bass_raw = None
    if bass_wav and Path(bass_wav).exists():
        yb = _chord_band(_chord_load(bass_wav), CHORD_LOW_HZ, 300.0)
        if np.any(np.abs(yb) > 1e-5):
            bc = librosa.feature.chroma_cqt(y=yb, n_octaves=4, **cq)
            bass_raw = bc[:, :nfr] if bc.shape[1] >= nfr else np.pad(bc, ((0, 0), (0, nfr - bc.shape[1])))

    # 3. tonart och skala: bara toniska bitar räknas (tystnad och brus räknas inte)
    rms = librosa.feature.rms(y=y, hop_length=CHORD_HOP)[0]
    flat_m = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=CHORD_HOP)[0]
    T = min(len(rms), len(flat_m), nfr)
    pos = rms[rms > 0]
    w = np.ones(T)
    if len(pos):
        w = ((rms[:T] > 0.1 * np.median(pos)) & (flat_m[:T] < 0.5)).astype(float)
    if w.sum() < 10:
        w = np.ones(T)
    kc = (chroma[:, :T] * w).sum(axis=1)
    kc = kc / (kc.sum() + 1e-9)
    if bass_raw is not None:
        bk = (bass_raw[:, :T] * w).sum(axis=1)
        kc = kc + 0.3 * bk / (bk.sum() + 1e-9)
    ranked = estimate_key(kc)
    k_t, k_mode, k_r = ranked[0]
    scale_name, scale_ivs = pick_scale(kc, k_t, k_mode)
    scale_pcs = {(k_t + i) % 12 for i in scale_ivs}
    key_txt, camelot, relative, key_flat = describe_key(k_t, k_mode, scale_name)
    alts = []
    for t, m, r in ranked[1:]:
        if len(alts) == 2:
            break
        if (t, m) == (k_t, k_mode):
            continue
        alts.append(f'{_nm(t, _use_flats(t, m))} {m} ({max(0, r) * 100:.0f} %)')
    key_info = {'text': key_txt, 'match': max(0.0, k_r) * 100, 'camelot': camelot, 'relative': relative,
                'also': alts, 'tonic': k_t, 'mode': k_mode, 'flat': key_flat, 'scale': scale_name}

    # 4. ackord per halvt slag
    if len(bf) < 4:
        bf = np.arange(0, nfr, int(0.5 * CHORD_SR / CHORD_HOP))
    bf = librosa.util.fix_frames(bf, x_min=0, x_max=nfr)
    bf = np.unique(np.concatenate([bf, (bf[:-1] + bf[1:]) // 2]))   # halva slag = finare tidsupplösning
    cs = librosa.util.sync(chroma, bf, aggregate=np.median)
    cs = np.log1p(CHORD_COMPRESS * cs)
    cs = cs / (np.linalg.norm(cs, axis=0, keepdims=True) + 1e-9)
    btimes = librosa.frames_to_time(bf, sr=CHORD_SR, hop_length=CHORD_HOP)
    steps = cs.shape[1]
    bass_c = None
    if bass_raw is not None:
        bc2 = librosa.util.sync(bass_raw, bf, aggregate=np.median)
        bass_c = bc2 / (np.linalg.norm(bc2, axis=0, keepdims=True) + 1e-9)

    types = [('', 0.0), ('m', 0.0), ('dim', 0.05)]
    if sevenths:
        types += [('7', CHORD_SEVENTH_COST), ('maj7', CHORD_SEVENTH_COST), ('m7', CHORD_SEVENTH_COST), ('m7b5', CHORD_SEVENTH_COST + 0.02)]
    labels, templ, pen, roots = [], [], [], []
    for r in range(12):
        for name, p in types:
            t = np.zeros(12)
            for i in CHORD_INTERVALS[name]:
                t[(r + i) % 12] = 1.0
            in_key = all(((r + i) % 12) in scale_pcs for i in CHORD_INTERVALS[name])
            templ.append(t / np.linalg.norm(t))
            labels.append(CHORD_NOTES[r] + name)
            pen.append(p - (CHORD_KEY_BONUS if in_key else 0.0))
            roots.append(r)
    labels.append('N')
    templ.append(np.ones(12) / np.sqrt(12))
    pen.append(CHORD_N_PENALTY)
    roots.append(-1)
    Tm = np.array(templ)
    score = Tm @ cs - np.array(pen)[:, None]
    if bass_c is not None:
        for k, r in enumerate(roots):
            if r >= 0:
                score[k] += CHORD_BASS_WEIGHT * bass_c[r, :steps]

    energy = np.array([np.sum(chroma[:, s:e]) for s, e in zip(bf[:-1], bf[1:])] + [0.0])[:steps]
    if np.any(energy > 0):
        quiet = energy < 0.15 * np.median(energy[energy > 0])
        step_s = float(np.median(np.diff(btimes))) if len(btimes) > 1 else 0.3
        min_steps = max(1, int(CHORD_N_MIN_QUIET / max(step_s, 0.05)))
        i = 0
        while i < len(quiet):                      # bara långa tystnader räknas som "inget ackord"
            if quiet[i]:
                j = i
                while j < len(quiet) and quiet[j]:
                    j += 1
                if j - i >= min_steps:
                    score[-1, i:j] += 1.0
                i = j
            else:
                i += 1

    prob = np.exp(score * 8.0)
    prob = prob / prob.sum(axis=0, keepdims=True)
    trans = librosa.sequence.transition_loop(len(labels), CHORD_LOOP)
    path = librosa.sequence.viterbi(prob, trans)

    segs = []
    total = len(y) / CHORD_SR
    for i, s in enumerate(path):
        t0 = float(btimes[i])
        t1 = float(btimes[i + 1]) if i + 1 < len(btimes) else total
        if segs and segs[-1][2] == labels[s]:
            segs[-1][1] = t1
        else:
            segs.append([t0, t1, labels[s]])
    clean = []
    for sg in segs:                      # slå ihop pyttesmå segment (under 0,25 s) med föregående
        if clean and (sg[1] - sg[0]) < 0.25:
            clean[-1][1] = sg[1]
        else:
            clean.append(sg)
    return {'tempo': tempo, 'segments': [tuple(s) for s in clean], 'source': source,
            'key': key_info, 'tuning_cents': tuning_cents}


def write_chord_files(result, output_folder, song_name):
    """Skriver ackord.txt och ackord.mid. Returnerar (filer, textrader till rutan i appen)."""
    segs = result['segments']
    k = result['key']
    flat = k['flat']
    chord_lines = []
    for a, b, c in segs:
        rn = roman_numeral(c, k['tonic'])
        chord_lines.append(f'{_fmt_time(a)} - {_fmt_time(b)}   {display_chord(c, flat)}' + (f'   ({rn})' if rn else ''))
    key_lines = [
        f"Tonart: {k['text']}  ({k['match']:.0f} % träff)",
        f"Camelot: {k['camelot']}   Relativ: {k['relative']}",
    ]
    if k['also']:
        key_lines.append('Också troligt: ' + ', '.join(k['also']))
    key_lines.append(f"Stämning: {result['tuning_cents']:+.0f} cent mot A440   Tempo ca {result['tempo']:.0f} BPM")
    head = [f'Låt: {song_name}'] + key_lines + [
        'Källa: ' + ('gitarr + keys/piano + övrigt (stråk), bas som grundton' if result['source'] == 'stems'
                     else 'hela låten (bas och kick bortklippta)'), '']
    txt = output_folder / 'ackord.txt'
    txt.write_text('\n'.join(head + chord_lines), encoding='utf-8')

    pm = pretty_midi.PrettyMIDI(initial_tempo=max(40.0, min(240.0, result['tempo'])))
    inst = pretty_midi.Instrument(program=0, name='Ackord')
    for a, b, c in segs:
        if c == 'N':
            continue
        root, suf = _split_chord(c)
        for iv in CHORD_INTERVALS[suf]:
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=60 + root + iv, start=a, end=b))
    pm.instruments.append(inst)
    mid = output_folder / 'ackord.mid'
    pm.write(str(mid))
    return [txt, mid], key_lines + [''] + chord_lines


def run_cmd(cmd, check=True):
    print('Kör:', ' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', env=env)
    print(res.stdout)
    if check and res.returncode != 0:
        raise RuntimeError(f'Kommando misslyckades: {" ".join(cmd)}\nUtskrift:\n{res.stdout}')
    return res


def _normalize_loudness(signal, target_rms=0.1):
    """Justerar volymen efter genomsnittlig ljudstyrka (RMS) istället för bara toppen —
    annars kan t.ex. en bas med konstant hög RMS kännas mycket högre än trummor med
    korta toppar, även om de har liknande toppvärde (peak). Begränsar samtidigt så
    resultatet aldrig klipper (clip)."""
    rms = np.sqrt(np.mean(signal ** 2)) if len(signal) else 0
    if rms <= 1e-6:
        return signal
    gain = target_rms / rms
    peak = np.max(np.abs(signal)) if len(signal) else 0
    if peak > 0:
        gain = min(gain, 0.98 / peak)
    return signal * gain


def _bandpass(signal, sr, low, high, order=2):
    """Filtrerar ett ljudspår till ett frekvensband (low/high i Hz, None = öppen ände)."""
    nyq = 0.5 * sr
    if low is None:
        b, a = butter(order, min(high / nyq, 0.999), btype='low')
    elif high is None:
        b, a = butter(order, max(low / nyq, 0.001), btype='high')
    else:
        b, a = butter(order, [max(low / nyq, 0.001), min(high / nyq, 0.999)], btype='band')
    return filtfilt(b, a, signal)


def _onsets_in_band(band_signal, sr):
    """Kör transientdetektering på ETT frekvensbands egen ljudkurva istället för
    hela mixen — så en tyst kick som är maskerad av en högre snare/hihat i helmixen
    ändå fångas, eftersom den kan vara tydlig i sitt eget lågfrekventa band."""
    onset_frames = librosa.onset.onset_detect(y=band_signal, sr=sr, backtrack=True, units='frames')
    return librosa.frames_to_samples(onset_frames)


def _classify_drum_hits(drums_wav_path):
    """Läser in drums.wav och hittar trumslag PER FREKVENSBAND (inte en gång på
    hela mixen) — baserat på verkliga referensvärden för var varje digitaltrumdel
    faktiskt ligger:
      - Kick: fundament 40-100Hz ("thump"), delar tyvärr samma område som elbas
      - Snare: kropp/fundament 150-250Hz
      - Hihat/cymbal ("shimmer"): 2000-8000Hz delas av båda — skiljs sen åt på
        klingtid (cymbal ringer längre) + om det finns kropp i 300-600Hz (cymbal
        har det, hi-hat saknar det nästan helt)
    Nära-samtidiga slag i olika band (inom 30ms) slås ihop till en enda träff,
    så vi inte dubbelräknar bleed mellan banden."""
    y, sr = librosa.load(str(drums_wav_path), sr=None, mono=True)

    kick_band = _bandpass(y, sr, 40, 100)
    snare_band = _bandpass(y, sr, 150, 250)
    shimmer_band = _bandpass(y, sr, 2000, 8000)      # delad hihat+cymbal-identitet
    cymbal_body_band = _bandpass(y, sr, 300, 600)    # kropp cymbal har, hihat saknar

    candidates = []  # (sample_index, category_guess)
    for band, cat in [(kick_band, 'kick'), (snare_band, 'snare'), (shimmer_band, 'shimmer')]:
        for s in _onsets_in_band(band, sr):
            candidates.append((int(s), cat))
    candidates.sort(key=lambda c: c[0])

    # Slå ihop nära-samtidiga träffar från olika band (bleed) till en, prioritera
    # i ordningen kick > snare > shimmer eftersom kick/snare är mer diagnostiska
    merge_window = int(sr * 0.03)
    priority = {'kick': 0, 'snare': 1, 'shimmer': 2}
    merged = []
    for s, cat in candidates:
        if merged and s - merged[-1][0] < merge_window:
            if priority[cat] < priority[merged[-1][1]]:
                merged[-1] = (s, cat)
            continue
        merged.append((s, cat))

    class_window = int(sr * 0.05)
    hits = []  # lista av (tid_sek, kategori, sample_index)

    for s, cat in merged:
        end = min(s + class_window, len(y))
        if end <= s:
            continue

        if cat != 'shimmer':
            category = cat
        else:
            # Skilj hihat från cymbal: klingtid + kroppsenergi i 300-600Hz
            decay_window = int(sr * 0.4)
            decay_end = min(s + decay_window, len(y))
            shimmer_slice = np.abs(shimmer_band[s:decay_end])
            peak = shimmer_slice.max() if len(shimmer_slice) else 0
            if peak > 0:
                below = np.where(shimmer_slice < peak * 0.1)[0]
                decay_ms = (below[0] if len(below) else decay_window) / sr * 1000
            else:
                decay_ms = 0
            body_energy = np.sqrt(np.mean(cymbal_body_band[s:end] ** 2))
            shimmer_energy = np.sqrt(np.mean(shimmer_band[s:end] ** 2)) + 1e-9
            has_body = (body_energy / shimmer_energy) > 0.35

            if decay_ms > 250 or has_body:
                category = 'cymbal'
            else:
                category = 'hihat_open' if decay_ms > 120 else 'hihat_closed'

        hits.append((s / sr, category, int(s)))

    bands = {
        'kick': kick_band, 'snare': snare_band,
        'hihat_closed': shimmer_band, 'hihat_open': shimmer_band, 'cymbal': cymbal_body_band,
    }
    return y, sr, hits, bands


def split_drums(drums_wav_path, output_folder, want_audio, want_midi, audio_format='wav'):
    """Tar drums.wav (Demucs helt trumspår) och skapar antingen/både:
    - separata ljudfiler per kick/snare/hihat_closed/hihat_open/cymbal
    - en drums_all.wav som slår ihop alla träffar till en egen "hel trummor"-fil,
      byggd från originalljudet (mer tryck kvar än Demucs egna drums.wav, som ofta
      tappat lite kick-energi till bass.wav)
    - en drums_split.mid-uppsättning med rätt GM-trumnoter på rätt tidpunkt
    """
    y, sr, hits, bands = _classify_drum_hits(drums_wav_path)
    result_files = []
    categories = ['kick', 'snare', 'hihat_closed', 'hihat_open', 'cymbal']
    min_hit_len = int(sr * 0.08)   # minst 80ms, även om nästa slag kommer superfort
    max_hit_len = int(sr * 0.9)    # men klipp av senast efter 900ms så det inte ringer för evigt

    # Sortera alla träffar i tidsordning så vi vet var NÄSTA slag (oavsett kategori) kommer —
    # det ger varje slag rätt klingtid istället för ett hårt fast klipp, så det låter
    # naturligt "hackat rakt ur loopen" snarare än taggigt/gated.
    hits_sorted = sorted(hits, key=lambda h: h[2])

    def hit_window(idx):
        _, _, s = hits_sorted[idx]
        if idx + 1 < len(hits_sorted):
            next_s = hits_sorted[idx + 1][2]
            length = min(max(next_s - s, min_hit_len), max_hit_len)
        else:
            length = max_hit_len
        return s, min(s + length, len(y))

    if want_audio:
        combined = np.zeros_like(y)
        for cat in categories:
            gated = np.zeros_like(y)
            for idx, (t, category, s) in enumerate(hits_sorted):
                if category != cat:
                    continue
                start, end = hit_window(idx)
                gated[start:end] = y[start:end]
                combined[start:end] = y[start:end]
            if not np.any(gated):
                continue
            gated = _normalize_loudness(gated)
            wav_tmp = output_folder / f'{cat}_tmp.wav'
            sf.write(str(wav_tmp), gated, sr)
            if audio_format == 'mp3':
                out_file = output_folder / f'{cat}.mp3'
                run_cmd(['ffmpeg', '-y', '-i', str(wav_tmp), '-codec:a', 'libmp3lame', '-qscale:a', '2', str(out_file)])
                wav_tmp.unlink(missing_ok=True)
            else:
                out_file = output_folder / f'{cat}.wav'
                wav_tmp.rename(out_file)
            result_files.append(out_file)

        # "Hel trummor"-fil byggd av samma träffar, med mer tryck kvar än Demucs egen drums.wav
        if np.any(combined):
            combined = _normalize_loudness(combined)
            combined_tmp = output_folder / 'drums_all_tmp.wav'
            sf.write(str(combined_tmp), combined, sr)
            if audio_format == 'mp3':
                out_file = output_folder / 'drums_all.mp3'
                run_cmd(['ffmpeg', '-y', '-i', str(combined_tmp), '-codec:a', 'libmp3lame', '-qscale:a', '2', str(out_file)])
                combined_tmp.unlink(missing_ok=True)
            else:
                out_file = output_folder / 'drums_all.wav'
                combined_tmp.rename(out_file)
            result_files.append(out_file)

    if want_midi:
        for cat in categories:
            cat_hits = [h for h in hits if h[1] == cat]
            if not cat_hits:
                continue
            pm = pretty_midi.PrettyMIDI()
            drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name=cat)
            note_num = DRUM_GM_NOTES.get(cat, 38)
            for t, category, s in cat_hits:
                drum_inst.notes.append(pretty_midi.Note(velocity=100, pitch=note_num, start=t, end=t + 0.1))
            pm.instruments.append(drum_inst)
            midi_path = output_folder / f'{cat}.mid'
            pm.write(str(midi_path))
            result_files.append(midi_path)

    return result_files


def _restore_highs(shares, raw, names, X, sr, nperseg):
    """Delar om de höga frekvenserna. shares = andel per stem och ruta (summerar till 1).
    Sång och bas behåller sin andel. Resten delas: hållna toner (harmonisk del) -> tonala stems
    efter hur närvarande de är i 2.5-5 kHz, slag (percussiv del) -> trummor."""
    tonal = [i for i, nm in enumerate(names) if nm in ('guitar', 'piano', 'other')]
    if not tonal or 'drums' not in names:
        return shares
    di = names.index('drums')
    keep = [i for i, nm in enumerate(names) if nm in ('vocals', 'bass')]
    f = np.linspace(0, sr / 2, X.shape[0])
    hi = np.where(f >= HF_START)[0]
    if len(hi) == 0:
        return shares
    h0 = hi[0]
    fade = np.clip((f[h0:] - HF_START) / (HF_FULL - HF_START), 0, 1)[:, None].astype('float32')
    # harmonisk/percussiv mask från originalet, bara för höga delen (snabbare)
    mag = np.abs(X[h0:, :]).astype('float32')
    Hm = median_filter(mag, size=(1, 17))
    Pm = median_filter(mag, size=(17, 1))
    mh = (Hm ** 2 / (Hm ** 2 + Pm ** 2 + 1e-20)).astype('float32')
    del mag, Hm, Pm
    # hur närvarande är varje tonal stem i mittbandet (jämnat över tid)
    E = []
    for i in tonal:
        E.append(uniform_filter1d(raw[i], 9))
    Etot = np.sum(E, axis=0) + 1e-12
    r = np.ones_like(shares[0][h0:, :])
    for i in keep:
        r = r - shares[i][h0:, :]
    r = np.clip(r, 0, 1)
    new = {}
    for e, i in zip(E, tonal):
        target = mh * r * (e / Etot)[None, :]
        new[i] = (1 - fade) * shares[i][h0:, :] + fade * target
    new[di] = (1 - fade) * shares[di][h0:, :] + fade * ((1 - mh) * r)
    # den som ev. inte räknats (t.ex. andra stems) tar resten av sitt gamla värde
    for i, val in new.items():
        shares[i][h0:, :] = val
    return shares


def find_audio_separator():
    """Hittar audio-separator (för MDX-NET). Prioritet: miljövariabel, PATH, Alexanders sepvenv."""
    cands = []
    env = os.environ.get('BLUNT_AUDIO_SEPARATOR')
    if env:
        cands.append(env)
    w = shutil.which('audio-separator')
    if w:
        cands.append(w)
    home = Path.home()
    cands += [str(home / 'test' / 'sepvenv' / 'Scripts' / 'audio-separator.exe'),
              str(home / 'test' / 'sepvenv' / 'bin' / 'audio-separator')]
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def mdx_vocals_and_instrumental(audio_path, out_dir):
    """Kör MDX-NET på hela låten. Returnerar (vocals_wav, instrumental_wav) eller (None, None)."""
    exe = find_audio_separator()
    if not exe:
        return None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_cmd([exe, str(audio_path), '--model_filename', MDX_MODEL,
                   '--output_dir', str(out_dir), '--output_format', 'WAV'], check=False)
    if res.returncode != 0:
        return None, None
    voc = sorted(out_dir.glob('*(Vocals)*.wav'))
    ins = sorted(out_dir.glob('*(Instrumental)*.wav'))
    if not voc or not ins:
        return None, None
    return voc[0], ins[0]


def rebuild_stems_from_mix(original_path, stem_paths):
    """Tar Demucs-stems som KARTA över vem som äger vad (per frekvens och tid) och
    delar sedan ORIGINALET efter kartan. Spåren summerar då till låten.
    stem_paths: {namn: Path}. Skriver över filerna. Returnerar True om det lyckades.
    v4: över ca 5 kHz delas originalet om (hållna toner -> gitarr/piano/övrigt, slag -> trummor)."""
    nperseg = REBUILD_NFFT
    noverlap = nperseg * 3 // 4
    orig, sr = sf.read(str(original_path), dtype='float32', always_2d=True)
    orig = orig[:, :2] if orig.shape[1] >= 2 else np.repeat(orig, 2, axis=1)
    n = len(orig)
    names = list(stem_paths)
    tracks = []
    for nm in names:
        d, s2 = sf.read(str(stem_paths[nm]), dtype='float32', always_2d=True)
        d = d[:, :2] if d.shape[1] >= 2 else np.repeat(d, 2, axis=1)
        if len(d) < n:
            d = np.pad(d, ((0, n - len(d)), (0, 0)))
        tracks.append(d[:n])
    sm = (REBUILD_SMOOTH_FREQ, REBUILD_SMOOTH_TIME)
    out = [np.zeros((n, 2), dtype='float32') for _ in names]
    eps = 1e-30
    for ch in range(2):
        _, _, X = stft(orig[:, ch], sr, nperseg=nperseg, noverlap=noverlap)
        X = X.astype(np.complex64)
        ref = float(np.percentile(np.abs(X), 99.9)) or 1.0
        mags, total = [], np.zeros(X.shape, dtype='float32')
        raw = []
        for t in tracks:
            _, _, S = stft(t[:, ch], sr, nperseg=nperseg, noverlap=noverlap)
            m0 = uniform_filter(np.abs(S) / ref, size=sm)
            m0 = np.maximum(m0, 0)
            if HF_RESTORE:
                fq = np.linspace(0, sr / 2, m0.shape[0])
                rows = (fq >= HF_TONAL_BAND[0]) & (fq < HF_TONAL_BAND[1])
                raw.append(m0[rows, :].mean(axis=0).astype('float32'))   # bara bandets nivå över tid (sparar minne)
            else:
                raw.append(None)
            m = (m0 ** REBUILD_POWER).astype('float32')
            mags.append(m)
            total += m
        denom = total + eps
        shares = [mg / denom for mg in mags]          # ursprunglig fördelning per ruta
        total = None
        mags = None
        if HF_RESTORE:
            try:
                shares = _restore_highs(shares, raw, names, X, sr, nperseg)
            except Exception as e:
                print('Höga frekvenser: hoppade över (', e, ')')
        raw = None
        for k in range(len(names)):
            _, y = istft(X * shares[k], sr, nperseg=nperseg, noverlap=noverlap)
            y = y[:n]
            out[k][:len(y), ch] = y.astype('float32')
            shares[k] = None
        del X, denom, shares
    peak = max(float(np.max(np.abs(a))) for a in out)
    scale = 1.0 if peak <= 0.99 else 0.99 / peak
    for nm, a in zip(names, out):
        sf.write(str(stem_paths[nm]), a * scale, sr, subtype='PCM_24')
    return True


@app.route('/health', methods=['GET'])
def health():
    """Snabbt sätt att kolla att servern är igång och att alla verktyg finns."""
    tools = {}
    for tool in ['ffmpeg', 'demucs', 'basic-pitch', 'yt-dlp']:
        tools[tool] = shutil.which(tool) is not None
    tools['audio-separator (MDX)'] = find_audio_separator() is not None
    if not tools['yt-dlp']:
        tools['yt-dlp'] = importlib.util.find_spec('yt_dlp') is not None   # python -m yt_dlp
    return jsonify({'status': 'ok', 'tools_found': tools})


def run_conversion(task_id, youtube_url, file_bytes, file_name, output_modes, audio_format, selected_stems, split_drums_flag, sevenths=False):
    workdir = Path(tempfile.mkdtemp(prefix='blunt_'))
    try:
        set_progress(task_id, percent=5, message='Förbereder...', status='running')
        is_original_only = output_modes == ['original']

        if youtube_url:
            base_name = 'youtube_audio'
            saved_stub = workdir / base_name
            set_progress(task_id, percent=10, message='Laddar ner från YouTube...')

            if is_original_only:
                # "Bara hela låten": be yt-dlp hämta direkt i RÄTT format på en gång,
                # istället för att alltid gå via WAV och konvertera en gång till efteråt.
                run_cmd([sys.executable, '-m', 'yt_dlp', '--no-playlist', '-x', '--audio-format', audio_format,
                          '-o', str(saved_stub) + '.%(ext)s', youtube_url])
            else:
                # Stems/MIDI-läge kräver WAV som indata till Demucs, så vi hämtar WAV som vanligt.
                run_cmd([sys.executable, '-m', 'yt_dlp', '--no-playlist', '-x', '--audio-format', 'wav',
                          '-o', str(saved_stub) + '.%(ext)s', youtube_url])

            candidates = list(workdir.glob(f'{base_name}.*'))
            if not candidates:
                raise RuntimeError('yt-dlp laddade inte ner någon ljudfil från länken.')
            saved = candidates[0]
            fname = base_name
        else:
            fname_raw = file_name or 'upload'
            saved = workdir / fname_raw
            saved.write_bytes(file_bytes)
            fname = Path(fname_raw).stem

        output_folder = OUTPUT_ROOT / f'{fname}_{task_id}'
        output_folder.mkdir(parents=True, exist_ok=True)
        result_files = []

        # "Bara hela låten" - inget behov av Demucs, och inget onödigt mellansteg via WAV
        if is_original_only:
            set_progress(task_id, percent=70, message='Sparar hela låten...')
            out_file = output_folder / f'{fname}.{audio_format}'
            if saved.suffix.lower() == f'.{audio_format}':
                # Redan rätt format (t.ex. yt-dlp gav oss exakt mp3/wav direkt) - bara flytta filen
                shutil.copy(saved, out_file)
            elif audio_format == 'mp3':
                run_cmd(['ffmpeg', '-y', '-i', str(saved), '-codec:a', 'libmp3lame', '-qscale:a', '2', str(out_file)])
            else:
                run_cmd(['ffmpeg', '-y', '-i', str(saved), '-ac', '2', '-ar', '44100', str(out_file)])
            result_files.append(out_file)
            bpm = detect_bpm(out_file)
            if bpm:
                bpm_file = output_folder / 'bpm.txt'
                bpm_file.write_text(f'Tempo (BPM): {bpm}\nLåt: {fname}\n', encoding='utf-8')
                result_files.append(bpm_file)
            chord_lines = None
            try:
                set_progress(task_id, percent=85, message='Hittar ackord...')
                res = analyze_chords([], None, out_file, sevenths=sevenths)
                if res:
                    cfiles, chord_lines = write_chord_files(res, output_folder, fname)
                    result_files.extend(cfiles)
            except Exception as e:
                print('Ackord misslyckades:', e)
            set_progress(
                task_id, percent=100, message='Klart!', status='done',
                files=[f.name for f in result_files], folder=output_folder.name, bpm=bpm,
                chords=chord_lines,
            )
            return

        # Steg 1: extrahera/konvertera ljud till en ren WAV (behövs för Demucs)
        set_progress(task_id, percent=20, message='Extraherar ljud (ffmpeg)...')
        audio_path = workdir / 'audio.wav'
        run_cmd(['ffmpeg', '-y', '-i', str(saved), '-ac', '2', '-ar', '44100', str(audio_path)])
        bpm = detect_bpm(audio_path)

        # Steg 2: separera stems med Demucs (den tunga biten, tar längst tid)
        set_progress(task_id, percent=30, message='Separerar stems med Demucs... kan ta en stund')
        demucs_out = workdir / 'demucs_out'
        demucs_out.mkdir(exist_ok=True)
        mdx_voc = None
        demucs_in = audio_path
        if MDX_ENABLED:
            try:
                set_progress(task_id, percent=30, message='Tar ut sången med MDX-NET... (några minuter)')
                mdx_voc, mdx_inst = mdx_vocals_and_instrumental(audio_path, workdir / 'mdx_out')
                if mdx_voc is not None:
                    demucs_in = workdir / 'musik_utan_sang.wav'
                    shutil.copy(mdx_inst, demucs_in)
                else:
                    print('MDX-NET ej tillgänglig - kör bara Demucs (installera audio-separator för bättre sång).')
            except Exception as e:
                print('MDX-NET misslyckades, kör bara Demucs:', e)
                mdx_voc = None
                demucs_in = audio_path
        set_progress(task_id, percent=45, message='Separerar instrument med Demucs... kan ta en stund')
        run_cmd(['demucs', '-n', DEMUCS_MODEL, '--out', str(demucs_out), str(demucs_in)])
        set_progress(task_id, percent=65, message='Demucs klar, bearbetar stems...')

        subdirs = [d for d in demucs_out.iterdir() if d.is_dir()]
        if not subdirs:
            raise RuntimeError('Demucs skapade inga output-mappar.')
        model_dir = subdirs[0]
        song_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        sep_dir = song_dirs[0] if song_dirs else model_dir
        if mdx_voc is not None:
            shutil.copy(mdx_voc, sep_dir / 'vocals.wav')   # MDX-sången ersätter Demucs (nästan tomma) sångspår

        # Bygg om stems från originalet (labb-inställningarna) så de summerar till låten
        if REBUILD_ENABLED and ('stems' in output_modes or 'midi' in output_modes):
            try:
                set_progress(task_id, percent=66, message='Bygger om spåren från originalet...')
                present = {st: sep_dir / f'{st}.wav' for st in ALL_STEMS if (sep_dir / f'{st}.wav').exists()}
                if len(present) >= 2:
                    rebuild_stems_from_mix(audio_path, present)
            except Exception as e:
                print('Ombyggnad misslyckades, använder Demucs-stems som de är:', e)

        step_percent = 70

        if 'stems' in output_modes:
            for s in selected_stems:
                wavf = sep_dir / f'{s}.wav'
                if not wavf.exists():
                    continue
                set_progress(task_id, percent=step_percent, message=f'Sparar ljud-stem: {s}...')
                # Normalisera volymen så inte t.ex. bas hamnar mycket högre än övriga
                # stems bara för att Demucs råkade separera ut den på den nivån.
                norm_data, norm_sr = sf.read(str(wavf))
                if NORMALIZE_STEMS:
                    norm_data = _normalize_loudness(norm_data)
                norm_wav = output_folder / f'{s}_norm_tmp.wav'
                sf.write(str(norm_wav), norm_data, norm_sr)
                if audio_format == 'mp3':
                    out_file = output_folder / f'{s}.mp3'
                    run_cmd(['ffmpeg', '-y', '-i', str(norm_wav), '-codec:a', 'libmp3lame', '-qscale:a', '2', str(out_file)])
                    norm_wav.unlink(missing_ok=True)
                else:
                    out_file = output_folder / f'{s}.wav'
                    norm_wav.rename(out_file)
                result_files.append(out_file)
                step_percent = min(step_percent + 3, 88)

        if 'midi' in output_modes:
            for s in selected_stems:
                wavf = sep_dir / f'{s}.wav'
                if not wavf.exists():
                    continue
                set_progress(task_id, percent=step_percent, message=f'Konverterar till MIDI: {s}...')
                run_cmd(['basic-pitch', str(output_folder), str(wavf)], check=False)
                generated = output_folder / f'{s}_basic_pitch.mid'
                final = output_folder / f'{s}.mid'
                if generated.exists():
                    generated.rename(final)
                    result_files.append(final)
                step_percent = min(step_percent + 3, 92)

        # Ackord: bara spåren som spelar ackord (gitarr/piano/övrigt), bas som grundton.
        # Körs på ALLA Demucs-spår oavsett vilka du valde att spara.
        chord_lines = None
        try:
            set_progress(task_id, percent=93, message='Hittar ackord...')
            chord_wavs = [sep_dir / f'{s}.wav' for s in CHORD_STEMS if (sep_dir / f'{s}.wav').exists()]
            bass_wav = sep_dir / 'bass.wav'
            res = analyze_chords(chord_wavs, bass_wav if bass_wav.exists() else None, audio_path, sevenths=sevenths)
            if res:
                cfiles, chord_lines = write_chord_files(res, output_folder, fname)
                result_files.extend(cfiles)
        except Exception as e:
            print('Ackord misslyckades:', e)

        # Extra steg: dela upp trummor ytterligare i kick/snare/hihat/cymbal
        if split_drums_flag and 'drums' in selected_stems:
            drums_wav = sep_dir / 'drums.wav'
            if drums_wav.exists():
                set_progress(task_id, percent=95, message='Delar upp trummor i kick/snare/hihat/cymbal...')
                extra_files = split_drums(
                    drums_wav, output_folder,
                    want_audio=('stems' in output_modes),
                    want_midi=('midi' in output_modes),
                    audio_format=audio_format,
                )
                result_files.extend(extra_files)

        if not result_files:
            raise RuntimeError('Inga resultatfiler genererades. Kontrollera att alla verktyg är installerade korrekt.')

        if bpm:
            bpm_file = output_folder / 'bpm.txt'
            bpm_file.write_text(f'Tempo (BPM): {bpm}\nLåt: {fname}\n', encoding='utf-8')
            result_files.append(bpm_file)

        set_progress(
            task_id, percent=100, message='Klart!', status='done',
            files=[f.name for f in result_files], folder=output_folder.name, bpm=bpm,
            chords=chord_lines,
        )

    except Exception as e:
        set_progress(task_id, message=str(e), status='error')
    finally:
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


@app.route('/upload', methods=['POST'])
def upload():
    youtube_url = (request.form.get('youtube_url') or '').strip()
    has_file = 'file' in request.files and request.files['file'].filename

    if not has_file and not youtube_url:
        return jsonify({'error': 'Ingen fil eller YouTube-länk skickades'}), 400

    output_modes_raw = request.form.get('output_modes', 'midi')  # kommaseparerat: 'midi', 'stems', 'original'
    output_modes = [m for m in output_modes_raw.split(',') if m in ('midi', 'stems', 'original')] or ['midi']
    audio_format = request.form.get('audio_format', 'wav')  # 'mp3' eller 'wav', bara vid stems
    selected_stems_raw = request.form.get('selected_stems', '')
    selected_stems = [s for s in selected_stems_raw.split(',') if s in ALL_STEMS] or ALL_STEMS
    split_drums_flag = request.form.get('split_drums', 'false').lower() in ('1', 'true', 'yes')
    sevenths_flag = request.form.get('sevenths', 'false').lower() in ('1', 'true', 'yes')

    file_bytes = None
    file_name = None
    if has_file:
        f = request.files['file']
        file_name = f.filename
        file_bytes = f.read()

    task_id = uuid.uuid4().hex[:8]
    set_progress(task_id, percent=0, message='Startar...', status='running')

    thread = threading.Thread(
        target=run_conversion,
        args=(task_id, youtube_url, file_bytes, file_name, output_modes, audio_format, selected_stems, split_drums_flag, sevenths_flag),
        daemon=True,
    )
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/progress/<task_id>', methods=['GET'])
def progress(task_id):
    with JOBS_LOCK:
        job = JOBS.get(task_id)
    if not job:
        return jsonify({'error': 'Okänt task_id'}), 404
    return jsonify(job)


@app.route('/files/<folder>/<filename>', methods=['GET'])
def get_file(folder, filename):
    """Serverar en enskild fil ur output_midi/<folder>/ — används för
    nedladdning och för dra-ut-i-DAW-funktionen i frontend."""
    directory = OUTPUT_ROOT / folder
    return send_from_directory(directory, filename, as_attachment=False)


@app.route('/download-zip/<folder>', methods=['GET'])
def download_zip(folder):
    directory = OUTPUT_ROOT / folder
    if not directory.is_dir():
        return jsonify({'error': 'Hittar inte mappen'}), 404
    zip_path = OUTPUT_ROOT / f'{folder}.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for f in directory.iterdir():
            zf.write(f, arcname=f.name)
    return send_file(zip_path, as_attachment=True, download_name=f'{folder}.zip')


@app.route('/', methods=['GET'])
def index_page():
    """Serverar index.html, så att mobilen kan öppna http://<datorns-ip>:5000/"""
    return send_from_directory(Path(__file__).resolve().parent.parent, 'index.html')


def _lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '?'


if __name__ == '__main__':
    print('')
    print('  Från datorn:   http://127.0.0.1:5000/')
    print(f'  Från mobilen (samma wifi):   http://{_lan_ip()}:5000/')
    print('')
    # host 0.0.0.0 = nåbar från mobilen. debug AV (debuggern ska inte ligga öppen på nätet).
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
