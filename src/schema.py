"""Schema für die Extraktion durch das Vision-LLM (Stufe 1).

Das LLM liefert nur, was im Dokument steht. Kodierung, Position und
Confidence ergänzt später unser eigenes System (Stufe 2).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["bestätigt", "verneint", "Verdacht"]


class Basis(BaseModel):
    """Gemeinsame Einstellung: keine zusätzlichen Felder erlaubt."""
    model_config = ConfigDict(extra="forbid")


class Fundstelle(Basis):
    """Felder, die jeder extrahierte Eintrag hat."""
    textstelle: str = Field(description="Wörtliches Zitat aus dem Dokument, ohne Auslassungen oder '...'")
    seite: int = Field(description="Seitennummer, beginnend bei 1")
    status: Status = Field(description="bestätigt = liegt vor, verneint = ausgeschlossen, Verdacht = vermutet")


class Dokument(Basis):
    dokumenttyp: str | None = Field(description="z. B. Entlassungsbrief, Laborbefund, Röntgenbefund")
    datum: str | None = Field(description="Datum des Dokuments im Format JJJJ-MM-TT")
    einrichtung: str | None = Field(description="Krankenhaus, Praxis oder Labor")
    seitenanzahl: int


class Patient(Basis):
    name: str | None
    geburtsdatum: str | None = Field(description="Format JJJJ-MM-TT")
    geschlecht: Literal["männlich", "weiblich", "divers"] | None


class Diagnose(Fundstelle):
    begriff: str = Field(description="Bezeichnung der Diagnose ohne Code")
    code_im_dokument: str | None = Field(description="Code nur, wenn er wörtlich im Dokument steht, sonst null")


class SymptomBefund(Fundstelle):
    begriff: str = Field(description="Genau ein Symptom oder Untersuchungsbefund pro Eintrag")


class Medikament(Fundstelle):
    wirkstoff: str
    staerke: str | None = Field(description="z. B. '5 mg'")
    verabreichung: str | None = Field(description="z. B. oral, i.v., s.c.")
    einnahmeschema: str | None = Field(description="z. B. '1-0-0'")
    zeitpunkt: Literal["Aufnahme", "stationär", "Entlassung", "unbekannt"]


class Laborwert(Fundstelle):
    parameter: str = Field(description="Name des Laborparameters, z. B. Lipase")
    wert_original: str = Field(description="Wert genau wie im Dokument geschrieben")
    wert: float | None = Field(description="Wert als Zahl, falls numerisch")
    einheit: str | None
    datum: str | None = Field(description="Datum der Messung im Format JJJJ-MM-TT")
    referenzbereich: str | None


class Extraktion(Basis):
    """Gesamtergebnis, das das LLM liefern muss."""
    dokument: Dokument
    patient: Patient
    diagnosen: list[Diagnose]
    symptome_befunde: list[SymptomBefund]
    medikamente: list[Medikament]
    laborwerte: list[Laborwert]