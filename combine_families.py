"""
combine_families.py — slår ihop instrumentkategorier som hör till samma
"familj" (t.ex. guitar + acoustic-guitar + electric-guitar) till EN ren fil.

VARFÖR: modellen ger 44+ separata kategorier, men flera av dem är egentligen
samma fysiska instrument uppdelat i undertyper. Ett instrument (t.ex. en
gitarr) kan under en låt vara starkast i olika av dessa kategorier vid olika
tidpunkter - problemet du hörde med reggae-gitarren.

METOD: "högst volym vinner, ruta för ruta i tid". Ingen ny AI-modell behövs.
För varje litet tidsfönster (0.5 sek) jämförs alla kandidatfiler i familjen,
och ljudet från den som är starkast just då används i den sammanslagna
outputen. Byter smidigt mellan källorna i takt med var energin faktiskt är.

ANVÄNDNING:
    python combine_families.py /sökväg/till/separation_results_mega/LåtNamn/

Skapar nya filer i samma mapp: guitar_combined.wav, strings_combined.wav osv,
en per familj där minst 2 av kategorierna faktiskt hade ljud i sig.
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

FAMILIES = {
    'guitar_combined':  ['guitar', 'acoustic-guitar', 'electric-guitar'],
    'strings_combined': ['strings', 'violin', 'viola', 'cello', 'double-bass', 'bowed_strings'],
    'keys_combined':    ['keys', 'organ', 'harpsichord', 'digital-piano'],
    'brass_combined':   ['brass', 'trumpet', 'trombone', 'tuba', 'french-horn'],
    'vocal_combined':   ['vocal', 'lead-vocal', 'back-vocal'],
}

WINDOW_SECONDS = 0.5
SILENCE_THRESHOLD = 0.005  # RMS under detta räknas som "inget ljud alls" i en fil


def combine_family(folder: Path, family_name: str, candidates: list[str]):
    paths = [folder / f'{c}.wav' for c in candidates]
    existing = [p for p in paths if p.exists()]
    if len(existing) < 2:
        return None  # inget att slå ihop om det inte finns minst 2 kandidater

    signals = []
    sr_ref = None
    for p in existing:
        y, sr = sf.read(str(p), always_2d=False)
        if sr_ref is None:
            sr_ref = sr
        signals.append(y)

    # Klipp alla till samma längd (kortaste), så fönstren linjerar upp
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]

    window = int(sr_ref * WINDOW_SECONDS)
    is_stereo = signals[0].ndim == 2
    out = np.zeros_like(signals[0])

    any_sound = False
    for start in range(0, min_len, window):
        end = min(start + window, min_len)
        chunk_energies = []
        for s in signals:
            chunk = s[start:end]
            mono = chunk.mean(axis=1) if is_stereo else chunk
            rms = np.sqrt(np.mean(mono ** 2)) if len(mono) else 0.0
            chunk_energies.append(rms)

        best_idx = int(np.argmax(chunk_energies))
        if chunk_energies[best_idx] > SILENCE_THRESHOLD:
            any_sound = True
        out[start:end] = signals[best_idx][start:end]

    if not any_sound:
        return None

    out_path = folder / f'{family_name}.wav'
    sf.write(str(out_path), out, sr_ref)
    return out_path


def main():
    if len(sys.argv) < 2:
        print('Användning: python combine_families.py /sökväg/till/låtmapp/')
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f'Hittar inte mappen: {folder}')
        sys.exit(1)

    print(f'Slår ihop familjer i: {folder}\n')
    made_any = False
    for family_name, candidates in FAMILIES.items():
        result = combine_family(folder, family_name, candidates)
        if result:
            print(f'  ✔ {family_name}.wav skapad (från {[c for c in candidates if (folder / f"{c}.wav").exists()]})')
            made_any = True
        else:
            print(f'  – {family_name}: hoppade över (för få källfiler eller helt tyst)')

    if not made_any:
        print('\nInga familjer kunde slås ihop - kontrollera att filnamnen matchar (t.ex. guitar.wav, acoustic-guitar.wav).')
    else:
        print('\nKlart! Jämför de nya _combined.wav-filerna mot originalen och hör om det låter mer sammanhängande.')


if __name__ == '__main__':
    main()
