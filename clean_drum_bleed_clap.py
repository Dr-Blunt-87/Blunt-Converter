"""
clean_drum_bleed_clap.py — samma verktyg (CLAP) som combine_families_clap.py,
men en annan uppgift: trumdelarna ska INTE slås ihop (kick/snare/hihat/cymbal
är redan riktiga, separata delar av ett trumset) - men de har ofta lite
LÄCKAGE mellan sig (en bit av "hihat.wav" som egentligen är en cymbal-smäll,
till exempel). Det här scriptet hittar och flyttar såna feklassade bitar till
rätt fil, så varje fil blir renare - utan att slå ihop något.

METOD:
1. Bygger en "klangfärgs-prototyp" per trumdel genom att medelvärdesbilda
   CLAP-embeddings över de starkaste/tydligaste träffarna i varje fil.
2. Går igenom varje enskilt identifierat slag i varje fil, jämför mot ALLA
   prototyper.
3. Om ett slag i t.ex. hihat.wav faktiskt liknar cymbal-prototypen mycket
   bättre än sin egen fils prototyp -> flyttas den biten dit i outputen.

FÖRSTA GÅNGEN, installera CLAP (om inte redan gjort för combine_families_clap.py):
    pip install msclap

ANVÄNDNING:
    python clean_drum_bleed_clap.py /sökväg/till/separation_results_mega/LåtNamn/

Skapar rena filer: kick_clean.wav, snare_clean.wav, hihat_clean.wav,
cymbal_clean.wav, toms_clean.wav (bara för de som faktiskt finns),
plus en clean_report.txt som listar varje flyttad bit.
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

# Namnen matchar filerna från AI-trummodellerna (kick.wav, snare.wav, hh.wav osv)
# och/eller vår egen lokala trumseparering (kick.wav, snare.wav, hihat_open.wav,
# hihat_closed.wav, cymbal.wav) - scriptet plockar upp vilka som faktiskt finns.
DRUM_CANDIDATES = ['kick', 'snare', 'hh', 'hihat_open', 'hihat_closed', 'cymbal', 'toms', 'percussion']

HIT_WINDOW_SECONDS = 0.15      # hur mycket ljud runt varje slag som analyseras
MOVE_MARGIN = 0.10             # annan kategori måste vara minst 10% bättre matchning för att vi ska flytta biten (undviker onödigt flimmer)

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


def build_prototype(y, sr, hits, tmpdir, name, max_hits_for_prototype=8):
    """Bygger en genomsnittlig klangfärgs-signatur från de starkaste slagen i filen."""
    window = int(sr * HIT_WINDOW_SECONDS)
    scored = []
    for s in hits:
        end = min(s + window, len(y))
        chunk = y[s:end]
        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) else 0
        scored.append((rms, s))
    scored.sort(key=lambda x: -x[0])
    top_hits = [s for _, s in scored[:max_hits_for_prototype]]

    embs = []
    for s in top_hits:
        end = min(s + window, len(y))
        embs.append(get_embedding(y[s:end], sr, tmpdir, f'proto_{name}'))
    if not embs:
        return None
    return np.mean(embs, axis=0)


def main():
    if len(sys.argv) < 2:
        print('Användning: python clean_drum_bleed_clap.py /sökväg/till/låtmapp/')
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f'Hittar inte mappen: {folder}')
        sys.exit(1)

    found = {name: folder / f'{name}.wav' for name in DRUM_CANDIDATES if (folder / f'{name}.wav').exists()}
    if len(found) < 2:
        print('Behöver minst 2 trumfiler i mappen för att kunna jämföra. Hittade:', list(found.keys()))
        sys.exit(1)

    print(f'\nHittade trumdelar: {list(found.keys())}\n')

    signals = {}
    sr_ref = None
    for name, path in found.items():
        y, sr = librosa.load(str(path), sr=None, mono=True)
        if sr_ref is None:
            sr_ref = sr
        signals[name] = y

    window = int(sr_ref * HIT_WINDOW_SECONDS)
    report_lines = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Steg 1: hitta slag i varje fil för sig, bygg prototyper
        all_hits = {}
        prototypes = {}
        for name, y in signals.items():
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr_ref, backtrack=True, units='frames')
            onset_samples = librosa.frames_to_samples(onset_frames)
            all_hits[name] = list(onset_samples)
            proto = build_prototype(y, sr_ref, all_hits[name], tmpdir, name)
            if proto is not None:
                prototypes[name] = proto
                print(f'  Prototyp byggd för {name} ({len(all_hits[name])} slag hittade)')

        # Steg 2: gå igenom varje slag i varje fil, kolla om det passar bättre någon annanstans
        cleaned = {name: np.copy(y) for name, y in signals.items()}
        moves = 0

        for name, hits in all_hits.items():
            own_proto = prototypes.get(name)
            if own_proto is None:
                continue
            for s in hits:
                end = min(s + window, len(signals[name]))
                chunk = signals[name][s:end]
                if not np.any(np.abs(chunk) > 0.001):
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
                    t_sec = s / sr_ref
                    # Flytta biten: nollställ i originalfilen, lägg till i rätt fil
                    cleaned[name][s:end] = 0
                    other_end = min(s + (end - s), len(cleaned[best_other]))
                    cleaned[best_other][s:other_end] = chunk[:other_end - s]
                    moves += 1
                    report_lines.append(
                        f'{t_sec:.2f}s: flyttade en bit från {name} -> {best_other} '
                        f'(egen matchning {own_sim:.2f}, {best_other}-matchning {best_sim:.2f})'
                    )

        for name, y in cleaned.items():
            out_path = folder / f'{name}_clean.wav'
            sf.write(str(out_path), y, sr_ref)

        report_path = folder / 'clean_report.txt'
        report_path.write_text(
            f'Städrapport - {moves} bitar flyttades till rätt kategori\n\n' + '\n'.join(report_lines) + '\n',
            encoding='utf-8',
        )

    print(f'\nKlart! {moves} bitar flyttades till rätt kategori.')
    print('Nya filer: *_clean.wav (samma antal filer som innan, bara renare)')
    print('Se clean_report.txt för exakt vad som flyttades var.')


if __name__ == '__main__':
    main()
