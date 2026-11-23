# Zeigt an, wie viel Guthaben bzw. Limit auf dem OpenRouter-Schlüssel noch übrig ist

import os       # liest den Schlüssel aus den Umgebungsvariablen

import requests     # schickt die Abfrage an OpenRouter
from dotenv import load_dotenv  # liest Schlüssel aus .env Datei aus

load_dotenv()

antwort = requests.get(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    timeout=30,
)
antwort.raise_for_status()      # bricht mit Fehlermeldung ab, falls die Abfrage nicht klappt
daten = antwort.json()["data"]

limit = daten["limit"]
verbleibend = daten["limit_remaining"]

print(f"Limit des Schlüssels:   {'kein Limit gesetzt' if limit is None else f'{limit:.2f} $'}")
print(f"Noch verfügbar:         {'unbegrenzt (nur Kontoguthaben zählt)' if verbleibend is None else f'{verbleibend:.2f} $'}")
if daten.get("limit_reset"):
    print(f"Limit wird erneuert:    {daten['limit_reset']}")
print(f"Verbraucht insgesamt:   {daten['usage']:.2f} $")
print(f"Verbraucht heute:       {daten['usage_daily']:.2f} $")
print(f"Verbraucht diesen Monat: {daten['usage_monthly']:.2f} $")