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


def _classify_drum_hits(drums_wav_path):
    """Läser in drums.wav, hittar varje trumslag EN gång (onset) och klassar det
    som kick/snare/hihat/cymbal baserat på vilket icke-överlappande frekvensband
    som har mest energi precis efter slaget. Icke-överlappande band är viktigt —
    annars "läcker" en cymbal-smäll in i hihat-facket och tvärtom."""
    y, sr = librosa.load(str(drums_wav_path), sr=None, mono=True)

    kick_band = _bandpass(y, sr, 35, 120)      # bastrumma: djupt fundament
    snare_band = _bandpass(y, sr, 150, 500)    # virveltrumma: kropp/fundament
    cymbal_band = _bandpass(y, sr, 3000, 8000)  # crash/ride: bredare, mer sustain
    hihat_band = _bandpass(y, sr, 8000, 16000)  # hi-hat: ljusast, kortast, mest högfrekvent

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='frames')
    onset_samples = librosa.frames_to_samples(onset_frames)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    class_window = int(sr * 0.05)  # 50ms fönster för att klassa varje slag
    hits = []  # lista av (tid_sek, kategori, sample_index)

    for t, s in zip(onset_times, onset_samples):
        end = min(s + class_window, len(y))
        if end <= s:
            continue
        energies = {
            'kick': np.sqrt(np.mean(kick_band[s:end] ** 2)),
            'snare': np.sqrt(np.mean(snare_band[s:end] ** 2)),
            'cymbal': np.sqrt(np.mean(cymbal_band[s:end] ** 2)),
            'hihat': np.sqrt(np.mean(hihat_band[s:end] ** 2)),
        }
        category = max(energies, key=energies.get)

        if category == 'hihat':
            # Öppen/stängd hi-hat: mät hur länge slaget klingar av
            decay_window = int(sr * 0.3)
            decay_end = min(s + decay_window, len(y))
            hihat_slice = np.abs(hihat_band[s:decay_end])
            peak = hihat_slice.max() if len(hihat_slice) else 0
            if peak > 0:
                below = np.where(hihat_slice < peak * 0.1)[0]
                decay_ms = (below[0] if len(below) else decay_window) / sr * 1000
            else:
                decay_ms = 0
            category = 'hihat_open' if decay_ms > 120 else 'hihat_closed'

        hits.append((float(t), category, int(s)))

    bands = {
        'kick': kick_band, 'snare': snare_band,
        'hihat_closed': hihat_band, 'hihat_open': hihat_band, 'cymbal': cymbal_band,
    }
    return y, sr, hits, bands


def split_drums(drums_wav_path, output_folder, want_audio, want_midi, audio_format='wav'):
    """Tar drums.wav (Demucs helt trumspår) och skapar antingen/både:
    - separata ljudfiler per kick/snare/hihat_closed/hihat_open/cymbal
    - en drums_split.mid med rätt GM-trumnoter på rätt tidpunkt
    """
    y, sr, hits, bands = _classify_drum_hits(drums_wav_path)
    result_files = []
    categories = ['kick', 'snare', 'hihat_closed', 'hihat_open', 'cymbal']
    hit_len = int(sr * 0.15)  # 150ms isolerat ljud per slag i audio-läget

    if want_audio:
        for cat in categories:
            band_signal = bands.get(cat, y)
            gated = np.zeros_like(band_signal)
            for t, category, s in hits:
                if category != cat:
                    continue
                end = min(s + hit_len, len(band_signal))
                gated[s:end] = band_signal[s:end]
            if not np.any(gated):
                continue
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

        if youtube_url:
            base_name = 'youtube_audio'
            saved_stub = workdir / base_name
            set_progress(task_id, percent=10, message='Laddar ner från YouTube...')
            run_cmd(['yt-dlp', '-x', '--audio-format', 'wav', '-o', str(saved_stub) + '.%(ext)s', youtube_url])
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

        # Steg 1: extrahera/konvertera ljud till en ren WAV
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

        output_folder = OUTPUT_ROOT / f'{fname}_{task_id}'
        output_folder.mkdir(parents=True, exist_ok=True)

        result_files = []
        step_percent = 70

        if 'stems' in output_modes:
            for s in selected_stems:
                wavf = sep_dir / f'{s}.wav'
                if not wavf.exists():
                    continue
                set_progress(task_id, percent=step_percent, message=f'Sparar ljud-stem: {s}...')
                if audio_format == 'mp3':
                    out_file = output_folder / f'{s}.mp3'
                    run_cmd(['ffmpeg', '-y', '-i', str(wavf), '-codec:a', 'libmp3lame', '-qscale:a', '2', str(out_file)])
                else:
                    out_file = output_folder / f'{s}.wav'
                    shutil.copy(wavf, out_file)
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

    output_modes_raw = request.form.get('output_modes', 'midi')  # kommaseparerat: 'midi', 'stems' eller båda
    output_modes = [m for m in output_modes_raw.split(',') if m in ('midi', 'stems')] or ['midi']
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
