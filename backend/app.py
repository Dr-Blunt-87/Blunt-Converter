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
import zipfile
import tempfile
import threading
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
import pretty_midi

app = Flask(__name__)
CORS(app)  # tillåter att index.html (öppnad direkt som fil) pratar med servern

BASE = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE / 'output_midi'
OUTPUT_ROOT.mkdir(exist_ok=True)

ALL_STEMS = ['bass', 'drums', 'guitar', 'piano', 'vocals', 'other']
DEMUCS_MODEL = 'htdemucs_6s'  # 6-stems-modellen: bas, trummor, gitarr, piano, vokaler, övrigt

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


def set_progress(task_id, percent=None, message=None, status=None, files=None, folder=None):
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


@app.route('/health', methods=['GET'])
def health():
    """Snabbt sätt att kolla att servern är igång och att alla verktyg finns."""
    tools = {}
    for tool in ['ffmpeg', 'demucs', 'basic-pitch', 'yt-dlp']:
        tools[tool] = shutil.which(tool) is not None
    return jsonify({'status': 'ok', 'tools_found': tools})


def run_conversion(task_id, youtube_url, file_bytes, file_name, output_modes, audio_format, selected_stems, split_drums_flag):
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
                run_cmd(['yt-dlp', '--no-playlist', '-x', '--audio-format', audio_format,
                          '-o', str(saved_stub) + '.%(ext)s', youtube_url])
            else:
                # Stems/MIDI-läge kräver WAV som indata till Demucs, så vi hämtar WAV som vanligt.
                run_cmd(['yt-dlp', '--no-playlist', '-x', '--audio-format', 'wav',
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
            set_progress(
                task_id, percent=100, message='Klart!', status='done',
                files=[f.name for f in result_files], folder=output_folder.name,
            )
            return

        # Steg 1: extrahera/konvertera ljud till en ren WAV (behövs för Demucs)
        set_progress(task_id, percent=20, message='Extraherar ljud (ffmpeg)...')
        audio_path = workdir / 'audio.wav'
        run_cmd(['ffmpeg', '-y', '-i', str(saved), '-ac', '2', '-ar', '44100', str(audio_path)])

        # Steg 2: separera stems med Demucs (den tunga biten, tar längst tid)
        set_progress(task_id, percent=30, message='Separerar stems med Demucs... kan ta en stund')
        demucs_out = workdir / 'demucs_out'
        demucs_out.mkdir(exist_ok=True)
        run_cmd(['demucs', '-n', DEMUCS_MODEL, '--out', str(demucs_out), str(audio_path)])
        set_progress(task_id, percent=65, message='Demucs klar, bearbetar stems...')

        subdirs = [d for d in demucs_out.iterdir() if d.is_dir()]
        if not subdirs:
            raise RuntimeError('Demucs skapade inga output-mappar.')
        model_dir = subdirs[0]
        song_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        sep_dir = song_dirs[0] if song_dirs else model_dir

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

        set_progress(
            task_id, percent=100, message='Klart!', status='done',
            files=[f.name for f in result_files], folder=output_folder.name,
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
        args=(task_id, youtube_url, file_bytes, file_name, output_modes, audio_format, selected_stems, split_drums_flag),
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


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
