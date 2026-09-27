"""
clean_cross_family_clap.py — tar bort läckage MELLAN instrumentfamiljer
(t.ex. gitarrljud som läckt in i strings_combined.wav trots att gitarren
redan finns korrekt i guitar_combined.wav).

VARFÖR: combine_families_clap.py jämför bara INOM en familj (gitarr mot
gitarr, stråk mot stråk). Men ibland gör själva AI-separeringsmodellen
ett större misstag och lägger t.ex. gitarrljud i en stråk-kategori
(bowed_strings, cello) redan från början - det fångar inte det interna
familjejämförelse-skriptet, för ur dess perspektiv är allt i strings-
familjen "stråk".

Det här skriptet körs EFTER combine_families_clap.py och jämför de
färdiga *_combined.wav-filerna (och eventuella soloinstrument som bass,
piano, drums om du vill inkludera dem) MOT VARANDRA. Om en bit i t.ex.
strings_combined.wav låter mycket mer som gitarr-prototypen än sin egen
stråk-prototyp, och gitarren redan finns i guitar_combined.wav, tas
biten bort (nollställs) i strings-filen - inget dubbelt, inget läckage.

VIKTIGT: skriptet TAR BORT läckage, det flyttar det INTE till den andra
filen (till skillnad från clean_drum_bleed_clap.py) - eftersom ljudet
redan finns korrekt där i den andra familjens fil. Att också kopiera in
det där skulle bara skapa eko/dubblering.

ANVÄNDNING (kör efter combine_families_clap.py på samma mapp):
    python clean_cross_family_clap.py /sökväg/till/separation_results_mega/LåtNamn/

Skapar: <familj>_combined_crosscleaned.wav för varje familj som fick
läckage borttaget, plus cross_clean_report.txt som listar varje borttagen
bit (tidpunkt + vilken familj den egentligen hörde till).
"""
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

try:
    from msclap import CLAP
except ImportError:
    print("Saknar msclap. Kör: pip install msclap")
    sys.exit(1)

# Vilka _combined.wav-filer (eller soloinstrument) som ska jämföras mot varandra.
# Lägg till fler namn här om du vill inkludera t.ex. bass.wav, piano.wav, drums-summan osv.
FAMILY_FILES = {
    'guitar':  'guitar_combined.wav',
    'strings': 'strings_combined.wav',
    'keys':    'keys_combined.wav',
    'brass':   'brass_combined.wav',
    'vocal':   'vocal_combined.wav',
}

HIT_WINDOW_SECONDS = 0.5       # större fönster än trumskriptet - melodiska instrument behöver mer ljud för att CLAP ska "höra" klangfärg
MOVE_MARGIN = 0.12             # annan familj måste vara minst 12% bättre matchning för att vi ska ta bort biten (försiktigt tröskelvärde)
SILENCE_THRESHOLD = 0.003

print("Laddar CLAP-modellen (tar en stund första gången)...")
clap_model = CLAP(version='2023', use_cuda=True)


def cosine_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def get_embedding(chunk, sr, tmpdir, tag):
    tmp_path = Path(tmpdir) / f'chunk_{tag}.wav'
    sf.write(str(tmp_path), chunk, sr)
    emb = clap_model.get_audio_embeddings([str(tmp_path)])
    return emb[0].cpu().numpy() if hasattr(emb[0], 'cpu') else np.array(emb[0])


def build_prototype(y, sr, tmpdir, name, max_hits_for_prototype=10):
    """Bygger en klangfärgs-signatur från de starkaste sekunderna i filen
    (använder onset-detektion om möjligt, annars bara starkaste fönstren)."""
    window = int(sr * HIT_WINDOW_SECONDS)
    try:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='frames')
        starts = list(librosa.frames_to_samples(onset_frames))
    except Exception:
        starts = []
    if not starts:
        starts = list(range(0, len(y), window))

    scored = []
    for s in starts:
        end = min(s + window, len(y))
        chunk = y[s:end]
        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) else 0
        if rms > SILENCE_THRESHOLD:
            scored.append((rms, s))
    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:max_hits_for_prototype]]

    embs = []
    for s in top:
        end = min(s + window, len(y))
        embs.append(get_embedding(y[s:end], sr, tmpdir, f'proto_{name}'))
    if not embs:
        return None
    return np.mean(embs, axis=0)


def main():
    if len(sys.argv) < 2:
        print('Användning: python clean_cross_family_clap.py /sökväg/till/låtmapp/')
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f'Hittar inte mappen: {folder}')
        sys.exit(1)

    found = {}
    for name, fname in FAMILY_FILES.items():
        p = folder / fname
        if p.exists():
            found[name] = p

    if len(found) < 2:
        print('Behöver minst 2 familjefiler för att kunna jämföra. Hittade:', list(found.keys()))
        print('(Kör combine_families_clap.py först om du inte redan gjort det.)')
        sys.exit(1)

    print(f'\nHittade familjer: {list(found.keys())}\n')

    signals = {}
    sr_ref = None
    for name, path in found.items():
        y, sr = librosa.load(str(path), sr=None, mono=True)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_ref)
        signals[name] = y

    window = int(sr_ref * HIT_WINDOW_SECONDS)
    report_lines = []

    with tempfile.TemporaryDirectory() as tmpdir:
        print('Bygger klangfärgs-prototyper för varje familj...')
        prototypes = {}
        for name, y in signals.items():
            proto = build_prototype(y, sr_ref, tmpdir, name)
            if proto is not None:
                prototypes[name] = proto
                print(f'  Prototyp byggd för {name}')

        cleaned = {name: np.copy(y) for name, y in signals.items()}
        removed_counts = {name: 0 for name in signals}

        for name, y in signals.items():
            own_proto = prototypes.get(name)
            if own_proto is None:
                continue
            min_len = len(y)
            for start in range(0, min_len, window):
                end = min(start + window, min_len)
                chunk = y[start:end]
                rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) else 0
                if rms < SILENCE_THRESHOLD:
                    continue

                emb = get_embedding(chunk, sr_ref, tmpdir, 'test')
                own_sim = cosine_sim(emb, own_proto)

                best_other, best_sim = None, own_sim
                for other_name, other_proto in prototypes.items():
                    if other_name == name:
                        continue
                    sim = cosine_sim(emb, other_proto)
                    if sim > best_sim + MOVE_MARGIN:
                        best_other, best_sim = other_name, sim

                if best_other:
                    t_sec = start / sr_ref
                    cleaned[name][start:end] = 0
                    removed_counts[name] += 1
                    report_lines.append(
                        f'{t_sec:.1f}s: tog bort en bit ur {name} (matchade {best_other} '
                        f'mycket bättre: {best_sim:.2f} mot egen {own_sim:.2f}) - '
                        f'ljudet finns redan korrekt i {best_other}_combined.wav'
                    )

        any_written = False
        for name, y in cleaned.items():
            if removed_counts[name] > 0:
                out_path = folder / f'{name}_combined_crosscleaned.wav'
                sf.write(str(out_path), y, sr_ref)
                any_written = True
                print(f'  ✔ {out_path.name} ({removed_counts[name]} bitar borttagna)')
            else:
                print(f'  – {name}: inget läckage hittat, ingen ny fil skapad')

        if any_written:
            report_path = folder / 'cross_clean_report.txt'
            report_path.write_text(
                'Rapport - läckage borttaget mellan familjer\n\n' + '\n'.join(report_lines) + '\n',
                encoding='utf-8',
            )
            print(f'\nSe cross_clean_report.txt för exakt vad som togs bort var.')

    print('\nKlart!')


if __name__ == '__main__':
    main()
