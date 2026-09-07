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

app = Flask(__name__)
CORS(app)  # tillåter att index.html (öppnad direkt som fil) pratar med servern

BASE = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE / 'output_midi'
OUTPUT_ROOT.mkdir(exist_ok=True)

ALL_STEMS = ['bass', 'drums', 'guitar', 'piano', 'vocals', 'other']
DEMUCS_MODEL = 'htdemucs_6s'  # 6-stems-modellen: bas, trummor, gitarr, piano, vokaler, övrigt


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


@app.route('/health', methods=['GET'])
def health():
    """Snabbt sätt att kolla att servern är igång och att alla verktyg finns."""
    tools = {}
    for tool in ['ffmpeg', 'demucs', 'basic-pitch', 'yt-dlp']:
        tools[tool] = shutil.which(tool) is not None
    return jsonify({'status': 'ok', 'tools_found': tools})


@app.route('/upload', methods=['POST'])
def upload():
    youtube_url = (request.form.get('youtube_url') or '').strip()
    has_file = 'file' in request.files and request.files['file'].filename

    if not has_file and not youtube_url:
        return jsonify({'error': 'Ingen fil eller YouTube-länk skickades'}), 400

    output_mode = request.form.get('output_mode', 'midi')  # 'midi' eller 'stems'
    audio_format = request.form.get('audio_format', 'wav')  # 'mp3' eller 'wav', bara vid stems
    selected_stems_raw = request.form.get('selected_stems', '')
    selected_stems = [s for s in selected_stems_raw.split(',') if s in ALL_STEMS] or ALL_STEMS

    task_id = uuid.uuid4().hex[:8]
    workdir = Path(tempfile.mkdtemp(prefix='blunt_'))

    try:
        if youtube_url:
            base_name = 'youtube_audio'
            saved_stub = workdir / base_name
            run_cmd(['yt-dlp', '-x', '--audio-format', 'wav', '-o', str(saved_stub) + '.%(ext)s', youtube_url])
            candidates = list(workdir.glob(f'{base_name}.*'))
            if not candidates:
                raise RuntimeError('yt-dlp laddade inte ner någon ljudfil från länken.')
            saved = candidates[0]
            fname = base_name
        else:
            f = request.files['file']
            fname = f.filename or 'upload'
            saved = workdir / fname
            f.save(saved)
            fname = Path(fname).stem

        # Steg 1: extrahera/konvertera ljud till en ren WAV
        audio_path = workdir / 'audio.wav'
        run_cmd(['ffmpeg', '-y', '-i', str(saved), '-ac', '2', '-ar', '44100', str(audio_path)])

        # Steg 2: separera stems med Demucs
        demucs_out = workdir / 'demucs_out'
        demucs_out.mkdir(exist_ok=True)
        run_cmd(['demucs', '-n', DEMUCS_MODEL, '--out', str(demucs_out), str(audio_path)])

        subdirs = [d for d in demucs_out.iterdir() if d.is_dir()]
        if not subdirs:
            raise RuntimeError('Demucs skapade inga output-mappar.')
        model_dir = subdirs[0]
        song_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        sep_dir = song_dirs[0] if song_dirs else model_dir

        output_folder = OUTPUT_ROOT / f'{fname}_{task_id}'
        output_folder.mkdir(parents=True, exist_ok=True)

        result_files = []

        if output_mode == 'stems':
            # Ge tillbaka rena ljud-stems, ev. konverterade till MP3
            for s in selected_stems:
                wavf = sep_dir / f'{s}.wav'
                if not wavf.exists():
                    continue
                if audio_format == 'mp3':
                    out_file = output_folder / f'{s}.mp3'
                    run_cmd(['ffmpeg', '-y', '-i', str(wavf), '-codec:a', 'libmp3lame', '-qscale:a', '2', str(out_file)])
                else:
                    out_file = output_folder / f'{s}.wav'
                    shutil.copy(wavf, out_file)
                result_files.append(out_file)
        else:
            # MIDI-läge: konvertera varje vald stem till MIDI med Basic Pitch
            for s in selected_stems:
                wavf = sep_dir / f'{s}.wav'
                if not wavf.exists():
                    continue
                run_cmd(['basic-pitch', str(output_folder), str(wavf)], check=False)
                generated = output_folder / f'{s}_basic_pitch.mid'
                final = output_folder / f'{s}.mid'
                if generated.exists():
                    generated.rename(final)
                    result_files.append(final)

        if not result_files:
            raise RuntimeError('Inga resultatfiler genererades. Kontrollera att alla verktyg är installerade korrekt.')

        return jsonify({
            'task_id': task_id,
            'folder': output_folder.name,
            'files': [f.name for f in result_files],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


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
    app.run(debug=True, port=5000)
