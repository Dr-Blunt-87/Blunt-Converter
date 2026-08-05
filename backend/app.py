"""
Blunt Converter - Backend
Tar emot en MP4/MP3/WAV, extraherar ljud, separerar stems (Demucs),
konverterar varje stem till MIDI (Basic Pitch) och skickar tillbaka
resultatet som en ZIP-fil.

KÖR:
    python app.py
Servern startar på http://127.0.0.1:5000
"""
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import uuid
import shutil
import zipfile
import tempfile

app = Flask(__name__)
CORS(app)  # tillåter att index.html (öppnad direkt som fil) pratar med servern

BASE = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE / 'output_midi'
OUTPUT_ROOT.mkdir(exist_ok=True)

STEMS = ['bass', 'drums', 'guitar', 'vocals', 'other']


def run_cmd(cmd, check=True):
    print('Kör:', ' '.join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if check and res.returncode != 0:
        raise RuntimeError(f'Kommando misslyckades: {" ".join(cmd)}\nUtskrift:\n{res.stdout}')
    return res


@app.route('/health', methods=['GET'])
def health():
    """Snabbt sätt att kolla att servern är igång och att ffmpeg/demucs/basic-pitch finns."""
    tools = {}
    for tool in ['ffmpeg', 'demucs', 'basic-pitch']:
        tools[tool] = shutil.which(tool) is not None
    return jsonify({'status': 'ok', 'tools_found': tools})


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Ingen fil skickades'}), 400

    f = request.files['file']
    include_vocals = request.form.get('include_vocals', 'true').lower() in ('1', 'true', 'yes')
    fname = f.filename or 'upload'
    task_id = uuid.uuid4().hex[:8]
    workdir = Path(tempfile.mkdtemp(prefix='blunt_'))
    saved = workdir / fname
    f.save(saved)

    try:
        # Steg 1: extrahera/konvertera ljud till en ren WAV
        audio_path = workdir / 'audio.wav'
        run_cmd(['ffmpeg', '-y', '-i', str(saved), '-ac', '2', '-ar', '44100', str(audio_path)])

        # Steg 2: separera stems med Demucs
        demucs_out = workdir / 'demucs_out'
        demucs_out.mkdir(exist_ok=True)
        run_cmd(['demucs', '--out', str(demucs_out), str(audio_path)])

        subdirs = [d for d in demucs_out.iterdir() if d.is_dir()]
        if not subdirs:
            raise RuntimeError('Demucs skapade inga output-mappar.')
        # Demucs skapar en mapp per modell, och en undermapp per låt
        model_dir = subdirs[0]
        song_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        sep_dir = song_dirs[0] if song_dirs else model_dir

        # Steg 3: konvertera varje stem till MIDI med Basic Pitch
        output_folder = OUTPUT_ROOT / f'{Path(fname).stem}_{task_id}'
        output_folder.mkdir(parents=True, exist_ok=True)

        stems_to_process = [s for s in STEMS if s != 'vocals' or include_vocals]

        for s in stems_to_process:
            wavf = sep_dir / f'{s}.wav'
            if not wavf.exists():
                continue
            # Basic Pitch CLI: tar en OUTPUT-mapp + input-fil, skapar <namn>_basic_pitch.mid
            run_cmd(['basic-pitch', str(output_folder), str(wavf)], check=False)
            generated = output_folder / f'{s}_basic_pitch.mid'
            final = output_folder / f'{s}.mid'
            if generated.exists():
                generated.rename(final)

        midi_files = list(output_folder.glob('*.mid'))
        if not midi_files:
            raise RuntimeError('Inga MIDI-filer genererades. Kontrollera att basic-pitch är installerat korrekt.')

        # Steg 4: paketera resultatet som ZIP
        zip_path = workdir / f'results_{task_id}.zip'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for m in midi_files:
                zf.write(m, arcname=m.name)

        return send_file(zip_path, as_attachment=True, download_name=f'blunt_results_{task_id}.zip')

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


if __name__ == '__main__':
    host = os.environ.get('BLUNT_HOST', '127.0.0.1')
    port = int(os.environ.get('BLUNT_PORT', '5000'))
    print(f'Blunt Converter backend startar på http://{host}:{port}')
    print('Testa i webbläsaren: http://127.0.0.1:5000/health')
    app.run(host=host, port=port, debug=True)
