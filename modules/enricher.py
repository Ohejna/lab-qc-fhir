from datetime import datetime
from typing import Dict, Optional, Tuple

# kody SNOMED CT dla płci — używamy ich przy budowaniu pola appliesTo w formacie FHIR
# to standardowe kody medyczne identyfikujące płeć biologiczną
_GENDER_SNOMED = {
    "male":   ("248153007", "Male"),
    "female": ("248152002", "Female"),
}


# oblicza wiek pacjenta w pełnych latach na dzień wykonania badania
# potrzebujemy wieku żeby wybrać właściwy zakres normy (pediatryczny vs dorosły)
# birthDate ma format "YYYY-MM" bez dnia — przyjmujemy że to 1. dzień miesiąca
def _calc_age(birth_date: str, effective_dt: str) -> Optional[int]:
    try:
        parts = birth_date.split("-")
        year, month = int(parts[0]), int(parts[1]) if len(parts) > 1 else 1
        # zakładamy 1. dzień miesiąca bo nie mamy dokładnej daty urodzenia
        birth = datetime(year, month, 1)
        # bierzemy tylko pierwsze 10 znaków daty badania (YYYY-MM-DD), reszta to czas
        eff = datetime.fromisoformat(effective_dt[:10])
        return (eff - birth).days // 365
    except Exception:
        # jeśli cokolwiek się nie zgadza w formacie dat, zwracamy None
        # wiek będzie nieznany i użyjemy zakresu default
        return None


# uzupełnia rekord o dwa pola których nie ma w surowych danych:
#   referenceRange — zakres normy dopasowany do płci i wieku pacjenta
#   interpretation — ocena czy wynik jest w normie (N), za niski (L) czy za wysoki (H)
# zwraca parę: (zaktualizowany_rekord, komunikat_ostrzegawczy)
# ostrzeżenie jest puste gdy wszystko poszło dobrze
def enrich_observation(obs: dict, ranges_config: Dict) -> Tuple[dict, str]:

    code_text = (obs.get("code") or {}).get("text", "")
    cfg = ranges_config.get(code_text)

    # nie znamy tego badania — nie możemy nic dodać, zwracamy ostrzeżenie
    if not cfg:
        return obs, f"Brak zakresów referencyjnych dla badania: {code_text}"

    # wyciągamy dane pacjenta które wcześniej dołączył loader
    patient = obs.get("_patient") or {}
    gender = (patient.get("gender") or "").lower()
    birth_date = patient.get("birthDate", "")
    effective_dt = obs.get("effectiveDateTime", "")

    # obliczamy wiek tylko jeśli mamy obie daty — bez nich nie możemy
    age = _calc_age(birth_date, effective_dt) if birth_date and effective_dt else None

    zakresy = cfg.get("zakresy", {})

    # wybieramy który zakres normy zastosować — priorytet: wiek → płeć → domyślny
    applies_to_code: Optional[str] = None
    applies_to_display: Optional[str] = None

    if age is not None and age < 18 and "pediatric" in zakresy:
        # dziecko poniżej 18 lat — używamy zakresu pediatrycznego jeśli istnieje
        zakres = zakresy["pediatric"]
    elif gender == "male" and "male" in zakresy:
        # dorosły mężczyzna — zakresy dla mężczyzn (np. hemoglobina jest wyższa niż u kobiet)
        zakres = zakresy["male"]
        applies_to_code, applies_to_display = _GENDER_SNOMED["male"]
    elif gender == "female" and "female" in zakresy:
        # dorosła kobieta
        zakres = zakresy["female"]
        applies_to_code, applies_to_display = _GENDER_SNOMED["female"]
    else:
        # płeć nieznana lub brak specyficznego zakresu — używamy wartości domyślnych
        zakres = zakresy.get("default", {})

    # jeśli nawet default nie istnieje w konfiguracji, nie możemy nic zrobić
    if not zakres:
        return obs, f"Brak zakresu normy dla badania: {code_text}"

    ref_unit = cfg.get("jednostka_referencyjna", "")
    low = zakres.get("low")
    high = zakres.get("high")

    # budujemy pole referenceRange w formacie FHIR 
    ref_range_entry: dict = {
        "low":  {"value": low,  "unit": ref_unit},
        "high": {"value": high, "unit": ref_unit},
    }
    # appliesTo informuje dla kogo jest ten zakres (np. dla kobiet) — dodajemy tylko gdy wiemy
    if applies_to_code:
        ref_range_entry["appliesTo"] = [{
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": applies_to_code,
                "display": applies_to_display,
            }]
        }]

    # dołączamy gotowy referenceRange do rekordu
    obs = {**obs, "referenceRange": [ref_range_entry]}

    # teraz porównujemy wartość z zakresem — ale tylko jeśli wartość jest liczbą
    # dla valueString lub pustego valueQuantity nie możemy porównać z normą
    value = (obs.get("valueQuantity") or {}).get("value")
    if isinstance(value, (int, float)):
        if low is not None and value < low:
            code, display = "L", "Low"     # poniżej normy
        elif high is not None and value > high:
            code, display = "H", "High"    # powyżej normy
        else:
            code, display = "N", "Normal"  # w normie

        # dodajemy interpretację jako pole FHIR z kodem z oficjalnego słownika HL7
        obs["interpretation"] = [{
            "coding": [{
                "system": "http://hl7.org/fhir/ValueSet/observation-interpretation",
                "code": code,
                "display": display,
            }]
        }]

    return obs, ""
