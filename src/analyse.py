# Test: ein Dokument an ein Vision-LLM über OpenRouter schicken

import base64   # wandelt PDF und Bilddatei in Text um
import os       # prüft ob OPENROUTER_API_KEY schon gesetzt wurde + Schlüssel aus Parameter Store wird da abgelegt
import sys      # ließt Dateipfad der der beim Aufruf mitgegeben wird
from pathlib import Path    # arbeitet mit Dateipfaden (Dateiendung herrausfinden, Datei einlesen, Dateiname holen)

import boto3    # holt OpenRouter-Schlüssel aus dem AWS Parameter Store
from dotenv import load_dotenv  # liest Schlüssel aus .env Datei aus
from litellm import completion  # schickt die eigentliche Anfrage über OpenRouter an Vision-LLM

# Lokal: Schlüssel aus der .env lesen
load_dotenv()

# In AWS Lambda: Schlüssel aus dem Parameter Store lesen (nur einmal pro Start)
if "OPENROUTER_API_KEY" not in os.environ:
    parameter_speicher = boto3.client("ssm", region_name="eu-central-1")
    parameter = parameter_speicher.get_parameter(Name="OpenRouterApiKey", WithDecryption=True)
    os.environ["OPENROUTER_API_KEY"] = parameter["Parameter"]["Value"]

# Aktuellen Modellnamen auf https://openrouter.ai/models prüfen
MODELL = "openrouter/anthropic/claude-sonnet-5"

ANWEISUNG = """Du bekommst einen medizinischen Befund.
Extrahiere alle Diagnosen, Medikamente und Laborwerte.
Gib zu jedem Eintrag an:
- die exakte Textstelle aus dem Dokument (wörtlich übernommen)
- die Seitennummer
- den Status: "bestätigt", "verneint" oder "Verdacht"
Erfinde keine medizinischen Codes.
Antworte ausschließlich mit gültigem JSON, ohne weiteren Text."""

DATEITYPEN = {".pdf": "application/pdf", ".png": "image/png",
              ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


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


def analysiere_dokument(pfad: Path) -> str:
    """Schickt das Dokument an das Vision-LLM und gibt die Antwort zurück."""
    antwort = completion(
        model=MODELL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": ANWEISUNG},
            dokument_als_nachrichtenteil(pfad),
        ]}],
        # "native": das Modell sieht das PDF selbst, keine Texterkennung durch OpenRouter
        extra_body={"plugins": [{"id": "file-parser", "pdf": {"engine": "native"}}]},
        temperature=0,  # möglichst gleichbleibende Ergebnisse
    )
    return antwort.choices[0].message.content


if __name__ == "__main__":
    # Aufruf: python src/analyse.py "Testdaten/KI-generated/001_Sonnenwald_Lutz.pdf"
    datei = Path(sys.argv[1])
    print(analysiere_dokument(datei))