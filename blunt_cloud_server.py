"""
Blunt Converter - molnserver (körs på GPU-maskinen, inte på kundens dator).

Har samma adresser som den lokala app.py, så Blunt Converter-sidan kan peka hit
utan ändringar:  /health  /upload  /progress/<id>  /files/<mapp>/<fil>  /download-zip/<mapp>

Kedjan per låt:
  1. ffmpeg -> wav
  2. ZFTurbo 53-stem-modellen (inference.py)  -> 52 stems
  3. cluster_stems_clap.py  -> ca 12-16 spår (fasta grupper + CLAP-klustring)
  4. rebuild_from_mix.py    -> spåren byggs om från ORIGINALET (summerar till originalet)
  5. spåren sparas som wav eller mp3 och kan laddas ner

KÖR (på maskinen):
    cd /workspace/Music-Source-Separation-Training
    pip install flask flask-cors
    python blunt_cloud_server.py

Provkörning utan GPU:  BLUNT_FAKE=1 python blunt_cloud_server.py

Extra: BPM (librosa) och MIDI. Trummor -> MIDI via slagdetektering. Toner -> MIDI via
Basic Pitch om det är installerat (BASIC_PITCH_BIN eller /workspace/bpvenv/bin/basic-pitch).
"""
import os
import sys
import uuid
import shutil
import zipfile
import threading
import subprocess
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

REPO = Path(os.environ.get('BLUNT_REPO', Path(__file__).resolve().parent))
OUTPUT_ROOT = REPO / 'blunt_output'
JOBS_ROOT = REPO / 'blunt_jobs'
OUTPUT_ROOT.mkdir(exist_ok=True)
JOBS_ROOT.mkdir(exist_ok=True)
FAKE = os.environ.get('BLUNT_FAKE') == '1'

PINS = ('lead-vocal=lead-vocal,back-vocal=back-vocal+vocal,bass=bass,kick=kick,hh=hh,'
        'guitar=guitar+electric-guitar+acoustic-guitar,snare=snare,percussion=percussion+congas+tambourine')

JOBS = {}
JOBS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()   # en låt åt gången på grafikkortet


def set_progress(task_id, percent=None, message=None, status=None, files=None, folder=None, bpm=None):
    with JOBS_LOCK:
        job = JOBS.setdefault(task_id, {})
        for k, v in (('percent', percent), ('message', message), ('status', status),
                     ('files', files), ('folder', folder), ('bpm', bpm)):
            if v is not None:
                job[k] = v


def run_cmd(cmd, cwd=None):
    print('Kör:', ' '.join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['HF_HUB_DISABLE_XET'] = '1'
    res = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding='utf-8', errors='replace', env=env, cwd=cwd)
    print(res.stdout[-3000:], flush=True)
    if res.returncode != 0:
        raise RuntimeError(f'Kommando misslyckades: {" ".join(map(str, cmd))}\n{res.stdout[-1500:]}')
    return res


DRUM_NOTES = {'kick': 36, 'snare': 38, 'hh': 42, 'percussion': 39, 'toms': 45, 'timpani': 45}


def detect_bpm(audio_path):
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=120)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        v = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
        return round(v, 1)
    except Exception as e:
        print('BPM-detektering misslyckades:', e, flush=True)
        return None


def basic_pitch_bin():
    for c in (os.environ.get('BASIC_PITCH_BIN'), '/workspace/bpvenv/bin/basic-pitch', shutil.which('basic-pitch')):
        if c and Path(c).exists():
            return c
    return None


def drum_midi(wav, note, out_mid):
    """Trummor -> MIDI: ett MIDI-slag vid varje detekterat ljudslag."""
    import librosa
    import pretty_midi
    y, sr = librosa.load(str(wav), sr=None, mono=True)
    if not len(y) or float((y ** 2).mean()) < 1e-9:
        return False
    times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=False)
    if not len(times):
        return False
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for t in times:
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=note, start=float(t), end=float(t) + 0.1))
    pm.instruments.append(inst)
    pm.write(str(out_mid))
    return True


def make_midi(wav, out_folder):
    """Ett spår -> en .mid-fil. Returnerar filnamnet eller None."""
    name = wav.name.replace('cluster_', '', 1)
    stem = Path(name).stem                       # t.ex. 04_kick
    label = stem.split('_', 1)[-1]               # kick
    target = out_folder / (stem + '.mid')
    for key, note in DRUM_NOTES.items():
        if key in label:
            return target.name if drum_midi(wav, note, target) else None
    bp = basic_pitch_bin()
    if not bp:
        return None
    tmp = out_folder / '_bp_tmp'
    tmp.mkdir(exist_ok=True)
    try:
        run_cmd([bp, tmp, wav])
    except Exception as e:
        print('Basic Pitch misslyckades för', wav.name, e, flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    made = list(tmp.glob('*.mid'))
    if not made:
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    shutil.move(str(made[0]), target)
    shutil.rmtree(tmp, ignore_errors=True)
    return target.name


def fake_pipeline(audio_wav, job_dir):
    """Provkörning utan GPU: gör några tysta spår så hela flödet kan testas."""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(audio_wav), always_2d=True)
    out = job_dir / 'final'
    out.mkdir(exist_ok=True)
    for i, name in enumerate(['lead-vocal', 'bass', 'kick']):
        sf.write(out / f'cluster_{i + 1:02d}_{name}.wav', data * 0.3, sr)
    return out


def run_job(task_id, file_bytes, file_name, output_modes, audio_format):
    job_dir = JOBS_ROOT / task_id
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        set_progress(task_id, percent=3, message='Tar emot låten...', status='running')
        stem = Path(file_name or 'upload').stem or 'upload'
        raw = job_dir / ('upload' + (Path(file_name).suffix if file_name else ''))
        raw.write_bytes(file_bytes)

        # 1. ljud -> wav
        set_progress(task_id, percent=8, message='Förbereder ljudet...')
        inp = job_dir / 'input'
        inp.mkdir(exist_ok=True)
        audio = inp / 'song.wav'
        run_cmd(['ffmpeg', '-y', '-i', raw, '-ac', '2', '-ar', '44100', audio])
        bpm = detect_bpm(audio)

        if output_modes == ['original']:
            final_dir = job_dir / 'original'
            final_dir.mkdir(exist_ok=True)
            shutil.copy(audio, final_dir / f'{stem}.wav')
        elif FAKE:
            set_progress(task_id, percent=50, message='(provkörning) skapar testspår...')
            final_dir = fake_pipeline(audio, job_dir)
        else:
            with GPU_LOCK:
                # 2. dela i 52 stems
                set_progress(task_id, percent=15, message='Delar låten i 52 stems (grafikkortet jobbar)...')
                sep = job_dir / 'sep'
                run_cmd([sys.executable, 'inference.py', '--model_type', 'bs_roformer',
                         '--config_path', 'models/mvsep_mega_config.yaml',
                         '--start_check_point', 'models/mvsep_mega_model.ckpt',
                         '--input_folder', inp, '--store_dir', sep], cwd=REPO)
                song_dir = sep / 'song'
                if not song_dir.is_dir():
                    raise RuntimeError('Modellen skapade inga stems.')

                # 3. gruppera till spår
                set_progress(task_id, percent=45, message='Grupperar stems till spår (CLAP)...')
                run_cmd([sys.executable, 'cluster_stems_clap.py', song_dir, '--threshold', '0.45',
                         '--mask-power', '4', '--gate-db', '40', '--ghost-db', '-40', '--pin', PINS],
                        cwd=REPO)
                clusters = song_dir / 'clusters'

                # 4. bygg om från originalet
                set_progress(task_id, percent=75, message='Bygger spåren från originalet...')
                run_cmd([sys.executable, 'rebuild_from_mix.py', audio, clusters, '--nfft', '2048',
                         '--power', '4', '--smooth-time', '5', '--boost', 'snare=3,kick=2,hh=2,percussion=2', '--tag', 'app'], cwd=REPO)
                final_dir = clusters.parent / 'clusters_mix_app'

        # 5. spara resultat
        set_progress(task_id, percent=92, message='Sparar spåren...')
        out_folder = OUTPUT_ROOT / f'{stem}_{task_id}'
        out_folder.mkdir(parents=True, exist_ok=True)
        want_audio = ('stems' in output_modes) or output_modes == ['original']
        want_midi = ('midi' in output_modes) and output_modes != ['original']
        result = []
        for w in sorted(final_dir.glob('*.wav')):
            name = w.name.replace('cluster_', '', 1)
            if want_audio:
                if audio_format == 'mp3':
                    target = out_folder / (Path(name).stem + '.mp3')
                    run_cmd(['ffmpeg', '-y', '-i', w, '-codec:a', 'libmp3lame', '-qscale:a', '2', target])
                else:
                    target = out_folder / name
                    shutil.copy(w, target)
                result.append(target.name)
            if want_midi:
                set_progress(task_id, percent=95, message=f'Gör MIDI: {name}')
                m = make_midi(w, out_folder)
                if m:
                    result.append(m)
        # Hela trummor = summan av trumdelarna (kick, snare, hihat, percussion m.m.)
        if want_audio and not FAKE:
            try:
                import numpy as np
                import soundfile as sf
                drum_files = [w for w in sorted(final_dir.glob('*.wav'))
                              if any(k in w.name.replace('cluster_', '', 1).split('_', 1)[-1] for k in DRUM_NOTES)]
                if drum_files:
                    tot = None
                    for w in drum_files:
                        d, dsr = sf.read(str(w), dtype='float32', always_2d=True)
                        tot = d if tot is None else (tot[:len(d)] + d[:len(tot)] if len(d) >= len(tot) else tot + np.pad(d, ((0, len(tot) - len(d)), (0, 0))))
                    pk = float(np.max(np.abs(tot))) or 1.0
                    if pk > 0.99:
                        tot = tot * (0.99 / pk)
                    dw = out_folder / 'drums_all.wav'
                    sf.write(str(dw), tot, dsr, subtype='PCM_24')
                    if audio_format == 'mp3':
                        dm = out_folder / 'drums_all.mp3'
                        run_cmd(['ffmpeg', '-y', '-i', dw, '-codec:a', 'libmp3lame', '-qscale:a', '2', dm])
                        dw.unlink()
                        result.append(dm.name)
                    else:
                        result.append(dw.name)
            except Exception as e:
                print('drums_all misslyckades:', e, flush=True)
        if bpm:
            (out_folder / 'bpm.txt').write_text(f'Tempo (BPM): {bpm}\nLåt: {stem}\n', encoding='utf-8')
            result.append('bpm.txt')
        if not result:
            raise RuntimeError('Inga filer skapades.')
        set_progress(task_id, percent=100, message='Klart!', status='done', files=result,
                     folder=out_folder.name, bpm=bpm)
    except Exception as e:
        set_progress(task_id, message=str(e), status='error')
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'mode': 'fake' if FAKE else 'gpu',
                    'tools_found': {'ffmpeg': shutil.which('ffmpeg') is not None, 'basic-pitch': basic_pitch_bin() is not None}})


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'Ingen fil skickades (YouTube-länkar stöds inte i molnläget än)'}), 400
    raw = request.form.get('output_modes', 'stems')
    modes = [m for m in raw.split(',') if m in ('midi', 'stems', 'original')] or ['stems']
    audio_format = request.form.get('audio_format', 'wav')
    f = request.files['file']
    task_id = uuid.uuid4().hex[:8]
    set_progress(task_id, percent=0, message='Köar...', status='running')
    threading.Thread(target=run_job, args=(task_id, f.read(), f.filename, modes, audio_format),
                     daemon=True).start()
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
    return send_from_directory(OUTPUT_ROOT / folder, filename, as_attachment=False)


@app.route('/download-zip/<folder>', methods=['GET'])
def download_zip(folder):
    d = OUTPUT_ROOT / folder
    if not d.is_dir():
        return jsonify({'error': 'Hittar inte mappen'}), 404
    zp = OUTPUT_ROOT / f'{folder}.zip'
    with zipfile.ZipFile(zp, 'w') as zf:
        for f in d.iterdir():
            zf.write(f, arcname=f.name)
    return send_file(zp, as_attachment=True, download_name=f'{folder}.zip')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True)
