"""
combine_families_clap.py — smartare version av combine_families.py.

SKILLNADEN mot den enkla varianten (som bara jämför VOLYM):
Den här använder CLAP (en AI-modell som "hör" klangfärg) för att avgöra
OM två samtidigt aktiva kandidatfiler (t.ex. acoustic-guitar och
electric-guitar) faktiskt är:
  (a) SAMMA instrument som AI-separeringen råkat sprida ut på fel fack
      -> slås ihop till en ren fil, precis som förut
  (b) TVÅ RIKTIGA, olika instrument som spelar samtidigt
      -> hålls kvar separat, inget slås ihop, du får en varning istället

Detta löser problemet med att en enkel "högst volym vinner"-metod annars
skulle kunna kasta bort en riktig akustisk gitarr bara för att en elgitarr
råkar spela lite starkare i samma sekund.

FÖRSTA GÅNGEN, installera CLAP-biblioteket (en gång, i din venv/miljö):
    pip install msclap

ANVÄNDNING:
    python combine_families_clap.py /sökväg/till/separation_results_mega/LåtNamn/

Skapar:
  - <familj>_combined.wav      när allt kunde slås ihop säkert
  - <familj>_OVERLAP_<namn>.wav  en fil per instrument när riktig
    samtidig spel upptäcktes (inget kastas bort, bara separerat kvar)
  - en textrapport, <familj>_report.txt, som förklarar vad som hände var
"""
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    from msclap import CLAP
except ImportError:
    print("Saknar msclap. Kör: pip install msclap")
    sys.exit(1)

FAMILIES = {
    'guitar_combined':  ['guitar', 'acoustic-guitar', 'electric-guitar'],
    'strings_combined': ['strings', 'violin', 'viola', 'cello', 'double-bass', 'bowed_strings'],
    'keys_combined':    ['keys', 'organ', 'harpsichord', 'digital-piano'],
    'brass_combined':   ['brass', 'trumpet', 'trombone', 'tuba', 'french-horn'],
    'vocal_combined':   ['vocal', 'lead-vocal', 'back-vocal'],
}

WINDOW_SECONDS = 2.0          # större fönster än den enkla varianten - CLAP behöver lite ljud för att "höra" klangfärg ordentligt
SILENCE_THRESHOLD = 0.005
SIMILARITY_THRESHOLD = 0.80   # över detta = "samma instrument", under = "olika riktiga instrument"

print("Laddar CLAP-modellen (tar en stund första gången)...")
clap_model = CLAP(version='2023', use_cuda=True)


def cosine_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_embedding(chunk, sr, tmpdir):
    """CLAP vill ha en fil, så vi skriver chunk:en till en tillfällig wav."""
    tmp_path = Path(tmpdir) / 'chunk_tmp.wav'
    sf.write(str(tmp_path), chunk, sr)
    emb = clap_model.get_audio_embeddings([str(tmp_path)])
    return emb[0].cpu().numpy() if hasattr(emb[0], 'cpu') else np.array(emb[0])


def combine_family_smart(folder: Path, family_name: str, candidates: list[str]):
    paths = {c: folder / f'{c}.wav' for c in candidates}
    existing = {c: p for c, p in paths.items() if p.exists()}
    if len(existing) < 2:
        return None

    names = list(existing.keys())
    signals = {}
    sr_ref = None
    for name, p in existing.items():
        y, sr = sf.read(str(p), always_2d=False)
        if sr_ref is None:
            sr_ref = sr
        signals[name] = y

    min_len = min(len(s) for s in signals.values())
    for name in names:
        signals[name] = signals[name][:min_len]

    window = int(sr_ref * WINDOW_SECONDS)
    is_stereo = signals[names[0]].ndim == 2
    combined_out = np.zeros_like(signals[names[0]])
    overlap_outs = {name: np.zeros_like(signals[names[0]]) for name in names}
    report_lines = []
    any_overlap_detected = False
    any_combined = False

    with tempfile.TemporaryDirectory() as tmpdir:
        for start in range(0, min_len, window):
            end = min(start + window, min_len)
            active = []
            for name in names:
                chunk = signals[name][start:end]
                mono = chunk.mean(axis=1) if is_stereo else chunk
                rms = np.sqrt(np.mean(mono ** 2)) if len(mono) else 0.0
                if rms > SILENCE_THRESHOLD:
                    active.append((name, rms))

            t_sec = start / sr_ref

            if not active:
                continue  # tyst just här, hoppa över

            if len(active) == 1:
                # Bara en kandidat aktiv - inget att jämföra, använd den rakt av
                name = active[0][0]
                combined_out[start:end] = signals[name][start:end]
                any_combined = True
                continue

            # Flera aktiva samtidigt - fråga CLAP om de faktiskt låter lika
            embeddings = {}
            for name, _ in active:
                mono_chunk = signals[name][start:end]
                mono_chunk = mono_chunk.mean(axis=1) if is_stereo else mono_chunk
                embeddings[name] = get_embedding(mono_chunk, sr_ref, tmpdir)

            # Jämför alla mot den starkaste (referens)
            active_sorted = sorted(active, key=lambda x: -x[1])
            ref_name = active_sorted[0][0]
            ref_emb = embeddings[ref_name]

            all_similar = True
            for name, _ in active_sorted[1:]:
                sim = cosine_sim(ref_emb, embeddings[name])
                if sim < SIMILARITY_THRESHOLD:
                    all_similar = False

            if all_similar:
                # Troligen samma instrument utspritt av misstag -> slå ihop, använd starkaste
                combined_out[start:end] = signals[ref_name][start:end]
                any_combined = True
            else:
                # Riktiga, olika instrument som spelar samtidigt -> behåll separat
                any_overlap_detected = True
                report_lines.append(
                    f'{t_sec:.1f}s: samtidigt spel upptäckt mellan {[n for n, _ in active]} - hölls separat'
                )
                for name, _ in active:
                    overlap_outs[name][start:end] = signals[name][start:end]

    results = []
    if any_combined:
        out_path = folder / f'{family_name}.wav'
        sf.write(str(out_path), combined_out, sr_ref)
        results.append(out_path)

    if any_overlap_detected:
        for name in names:
            if np.any(overlap_outs[name]):
                out_path = folder / f'{family_name}_OVERLAP_{name}.wav'
                sf.write(str(out_path), overlap_outs[name], sr_ref)
                results.append(out_path)

        report_path = folder / f'{family_name}_report.txt'
        report_path.write_text(
            f'Rapport för {family_name}\n' + '\n'.join(report_lines) + '\n',
            encoding='utf-8',
        )
        results.append(report_path)

    return results if results else None


def main():
    if len(sys.argv) < 2:
        print('Användning: python combine_families_clap.py /sökväg/till/låtmapp/')
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f'Hittar inte mappen: {folder}')
        sys.exit(1)

    print(f'\nSlår ihop familjer (CLAP-baserad, klangfärgsmedveten) i: {folder}\n')
    for family_name, candidates in FAMILIES.items():
        print(f'--- {family_name} ---')
        results = combine_family_smart(folder, family_name, candidates)
        if results:
            for r in results:
                print(f'  ✔ {r.name}')
        else:
            print('  – hoppade över (för få källfiler eller helt tyst)')
        print()

    print('Klart! Om du ser "_OVERLAP_"-filer betyder det att CLAP upptäckte riktigt '
          'samtidigt spel av olika instrument - kolla motsvarande _report.txt för detaljer.')


if __name__ == '__main__':
    main()
