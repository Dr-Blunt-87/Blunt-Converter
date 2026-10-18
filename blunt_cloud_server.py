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


def set_progress(task_id, percent=None, message=None, status=None, files=None, folder=None):
    with JOBS_LOCK:
        job = JOBS.setdefault(task_id, {})
        for k, v in (('percent', percent), ('message', message), ('status', status),
                     ('files', files), ('folder', folder)):
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
                         '--power', '4', '--smooth-time', '5', '--tag', 'app'], cwd=REPO)
                final_dir = clusters.parent / 'clusters_mix_app'

        # 5. spara resultat
        set_progress(task_id, percent=92, message='Sparar spåren...')
        out_folder = OUTPUT_ROOT / f'{stem}_{task_id}'
        out_folder.mkdir(parents=True, exist_ok=True)
        result = []
        for w in sorted(final_dir.glob('*.wav')):
            name = w.name.replace('cluster_', '', 1)
            if audio_format == 'mp3':
                target = out_folder / (Path(name).stem + '.mp3')
                run_cmd(['ffmpeg', '-y', '-i', w, '-codec:a', 'libmp3lame', '-qscale:a', '2', target])
            else:
                target = out_folder / name
                shutil.copy(w, target)
            result.append(target.name)
        if not result:
            raise RuntimeError('Inga spår skapades.')
        set_progress(task_id, percent=100, message='Klart!', status='done', files=result, folder=out_folder.name)
    except Exception as e:
        set_progress(task_id, message=str(e), status='error')
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'mode': 'fake' if FAKE else 'gpu',
                    'tools_found': {'ffmpeg': shutil.which('ffmpeg') is not None}})


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
