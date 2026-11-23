# Test: ein Dokument an ein Vision-LLM über OpenRouter schicken und als festes JSON zurückbekommen

import base64   # wandelt PDF und Bilddatei in Text um
import os       # prüft ob OPENROUTER_API_KEY schon gesetzt wurde + Schlüssel aus Parameter Store wird da abgelegt
import sys      # liest Modell und Dateipfad, die beim Aufruf mitgegeben werden
import time     # misst, wie lange die Anfrage dauert
from pathlib import Path    # arbeitet mit Dateipfaden (Dateiendung herausfinden, Datei einlesen, Dateiname holen)

import boto3    # holt OpenRouter-Schlüssel aus dem AWS Parameter Store
from dotenv import load_dotenv  # liest Schlüssel aus .env Datei aus
from litellm import completion  # schickt die eigentliche Anfrage über OpenRouter an Vision-LLM

from schema import Extraktion   # unser festes JSON-Schema (Pydantic)

# Lokal: Schlüssel aus der .env lesen
load_dotenv()

# In AWS Lambda: Schlüssel aus dem Parameter Store lesen (nur einmal pro Start)
if "OPENROUTER_API_KEY" not in os.environ:
    parameter_speicher = boto3.client("ssm", region_name="eu-central-1")
    parameter = parameter_speicher.get_parameter(Name="OpenRouterApiKey", WithDecryption=True)
    os.environ["OPENROUTER_API_KEY"] = parameter["Parameter"]["Value"]

# Modelle für den Vergleich (aktuelle Namen auf https://openrouter.ai/models prüfen)
# "anbieter": über welchen Anbieter OpenRouter die Anfrage schickt (None = OpenRouter entscheidet)
MODELLE = {
    "claude": {"name": "openrouter/anthropic/claude-sonnet-5", "anbieter": "anthropic"},
    "gpt":    {"name": "openrouter/openai/gpt-6-sol",          "anbieter": "azure"},
}

ANWEISUNG = """Du bekommst einen medizinischen Befund.
Extrahiere Dokumentdaten, Patientendaten, Diagnosen, Symptome/Befunde, Medikamente und Laborwerte.

Regeln:
- Jeder Eintrag enthält genau eine Sache. "ohne Fieber oder Ikterus" ergibt zwei Einträge.
- Untersuchungsbefunde und Symptome (z. B. "kein Aszites") gehören zu symptome_befunde, nicht zu diagnosen.
- Aussagen, dass etwas nicht vorliegt, sind keine Medikamente oder Diagnosen, sondern bekommen den Status "verneint".
- Bei Laborwerten im Verlauf ist jede Messung ein eigener Eintrag mit Parameter, Wert, Einheit und Datum.
- Die textstelle ist immer ein wörtliches Zitat aus dem Dokument, ohne Auslassungen.
- Erfinde keine medizinischen Codes. code_im_dokument nur, wenn der Code wörtlich im Dokument steht.
- Fehlt eine Angabe im Dokument, setze null."""

DATEITYPEN = {".pdf": "application/pdf", ".png": "image/png",
              ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# Das Schema, an das sich das Modell halten muss (Structured Outputs)
ANTWORTFORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "extraktion",
        "strict": True,
        "schema": Extraktion.model_json_schema(),
    },
}

ERGEBNIS_ORDNER = Path("ergebnisse")    # hier werden die Ergebnisse für den Vergleich gespeichert


def dokument_als_nachrichtenteil(pfad: Path) -> dict:
    """Wandelt ein PDF oder Bild in den passenden Teil der Anfrage um."""
    dateityp = DATEITYPEN.get(pfad.suffix.lower())
    if dateityp is None:
        raise ValueError(f"Dateityp {pfad.suffix} wird (noch) nicht unterstützt")

    inhalt_base64 = base64.b64encode(pfad.read_bytes()).decode()
    daten_adresse = f"data:{dateityp};base64,{inhalt_base64}"

    if dateityp == "application/pdf":
        return {"type": "file", "file": {"filename": pfad.name, "file_data": daten_adresse}}
    return {"type": "image_url", "image_url": {"url": daten_adresse}}


def analysiere_dokument(pfad: Path, modell_kuerzel: str) -> tuple[Extraktion, dict]:
    """Schickt das Dokument an das gewählte Vision-LLM und gibt Ergebnis + Messwerte zurück."""
    modell = MODELLE[modell_kuerzel]

    # Zusätzliche Einstellungen für OpenRouter
    openrouter_einstellungen = {
        # "native": das Modell sieht das PDF selbst, keine Texterkennung durch OpenRouter
        "plugins": [{"id": "file-parser", "pdf": {"engine": "native"}}],
        # mehr internes Nachdenken
        "reasoning": {"effort": "medium"},
    }
    if modell["anbieter"]:
        # nur diesen Anbieter verwenden, nicht automatisch auf einen anderen ausweichen
        openrouter_einstellungen["provider"] = {"only": [modell["anbieter"]], "allow_fallbacks": False}

    start = time.perf_counter()
    antwort = completion(
        model=modell["name"],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": ANWEISUNG},
            dokument_als_nachrichtenteil(pfad),
        ]}],
        response_format=ANTWORTFORMAT,
        extra_body=openrouter_einstellungen,
        temperature=0,          # möglichst gleichbleibende Ergebnisse
        max_tokens=32000,       # genug Platz für lange Briefe mit vielen Laborwerten
        drop_params=True,       # Einstellungen weglassen, die ein Modell nicht unterstützt
    )
    dauer = time.perf_counter() - start

    antworttext = antwort.choices[0].message.content

    # Fehlersuche: warum endet die Antwort und wie viele Tokens wurden verbraucht?
    print("Abbruchgrund:", antwort.choices[0].finish_reason)
    print("Ausgabe-Tokens gesamt:", antwort.usage.completion_tokens)
    details = getattr(antwort.usage, "completion_tokens_details", None)
    print("davon Denk-Tokens:", getattr(details, "reasoning_tokens", "unbekannt"))

    # prüft, ob die Antwort wirklich dem Schema entspricht (sonst Fehler)
    ergebnis = Extraktion.model_validate_json(antworttext)

    messwerte = {
        "modell": modell["name"],
        "dauer_sekunden": round(dauer, 1),
        "eingabe_tokens": antwort.usage.prompt_tokens,
        "ausgabe_tokens": antwort.usage.completion_tokens,
    }
    return ergebnis, messwerte


def speichere_ergebnis(pfad: Path, modell_kuerzel: str, ergebnis: Extraktion) -> Path:
    """Speichert das Ergebnis als JSON-Datei, z. B. ergebnisse/001_Sonnenwald_Lutz_claude.json."""
    ERGEBNIS_ORDNER.mkdir(exist_ok=True)
    zieldatei = ERGEBNIS_ORDNER / f"{pfad.stem}_{modell_kuerzel}.json"
    zieldatei.write_text(ergebnis.model_dump_json(indent=2), encoding="utf-8")
    return zieldatei


if __name__ == "__main__":
    # Aufruf: python src/analyse.py claude "Testdaten/KI-generated/001_Sonnenwald_Lutz.pdf"
    #         python src/analyse.py gpt    "Testdaten/KI-generated/001_Sonnenwald_Lutz.pdf"
    modell_kuerzel = sys.argv[1]
    datei = Path(sys.argv[2])

    ergebnis, messwerte = analysiere_dokument(datei, modell_kuerzel)
    zieldatei = speichere_ergebnis(datei, modell_kuerzel, ergebnis)

    print(ergebnis.model_dump_json(indent=2))
    print(f"\nGespeichert in: {zieldatei}")
    print(f"Modell: {messwerte['modell']}")
    print(f"Dauer: {messwerte['dauer_sekunden']} s")
    print(f"Tokens: {messwerte['eingabe_tokens']} Eingabe / {messwerte['ausgabe_tokens']} Ausgabe")
    print(f"Diagnosen: {len(ergebnis.diagnosen)}, Symptome/Befunde: {len(ergebnis.symptome_befunde)}, "
          f"Medikamente: {len(ergebnis.medikamente)}, Laborwerte: {len(ergebnis.laborwerte)}")