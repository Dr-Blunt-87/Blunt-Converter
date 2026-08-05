# Blunt Converter

Video/ljud → MIDI-stems, med Demucs (stem-separering) och Basic Pitch (ljud→MIDI).

## Struktur

```
BluntConverter/
├── index.html          <- frontend, hostas på Netlify (statisk sida)
├── backend/
│   ├── app.py           <- Python-server, körs LOKALT (inte på Netlify)
│   └── requirements.txt
└── README.md
```

## Viktigt att förstå

`index.html` är en helt vanlig statisk sida (som Beatz4Pi) och kan hostas
var som helst, t.ex. Netlify. Men den pratar med en Python-server
(`backend/app.py`) som gör det tunga AI-jobbet (Demucs/Basic Pitch).
Den servern kan INTE köras på Netlify — den måste köras lokalt på en
dator, med `python app.py`, medan du använder appen.

`index.html` är hårdkodad att prata med `http://127.0.0.1:5000` —
det vill säga: din egen dator, medan backend körs där.

## Köra backend lokalt (Windows)

1. `cd backend`
2. `python -m venv venv`
3. `venv\Scripts\activate`
4. `pip install -r requirements.txt`
5. `python app.py`

Servern startar på `http://127.0.0.1:5000`. Testa i webbläsaren:
`http://127.0.0.1:5000/health`

## Öppna frontend

Öppna `index.html` direkt i webbläsaren (dubbelklick), eller besök
Netlify-adressen om den är deployad dit. Så länge backend körs lokalt
på samma dator fungerar konverteringen.

## Förutsättningar

- Python 3.10+ installerat och i PATH (`python --version`)
- ffmpeg installerat och i PATH (`ffmpeg -version`)
- Demucs och Basic Pitch installeras automatiskt via `requirements.txt`,
  men laddar ner AI-modeller (flera hundra MB) första gången de körs
