from typing import Tuple

# wszystkie możliwe pola w których może być zapisana wartość badania wg standardu FHIR
# sprawdzamy wszystkie bo różne laboratoria mogą używać różnych typów
_VALUE_FIELDS = (
    "valueQuantity", "valueString", "valueBoolean", "valueInteger",
    "valueRange", "valueRatio", "valueSampledData", "valueTime",
    "valueDateTime", "valuePeriod",
)


# sprawdza obserwację i przypisuje ją do jednej z trzech kategorii:
#   "red"    — błąd krytyczny, rekord do automatycznego usunięcia
#   "yellow" — błąd który trzeba ocenić ręcznie, operator decyduje co z tym zrobić
#   "green"  — rekord poprawny strukturalnie, można go dalej przetwarzać
# zwraca krotkę (kategoria, opis_powodu)
def categorize_observation(obs: dict) -> Tuple[str, str]:

    # status i code to pola obowiązkowe w standardzie FHIR — bez nich rekord jest bezużyteczny
    if not obs.get("status"):
        return "red", "Brak pola status"
    if not obs.get("code"):
        return "red", "Brak pola code"

    # subject mówi do kogo należy wynik — bez tego nie wiemy czyi to badanie
    # dajemy yellow a nie red bo to może być błąd importu, a samo badanie może być wartościowe
    subject = obs.get("subject")
    if not subject:
        return "yellow", "Brak subject"

    # subject istnieje, ale podczas wczytywania nie znaleźliśmy pacjenta o tym ID w pliku pacjentów
    # to oznacza że albo dane są niespójne, albo pacjent był w innym pliku
    patient_id = obs.get("_patient_id")
    if obs.get("_patient") is None:
        label = f"Patient/{patient_id}" if patient_id else "(brak referencji)"
        return "red", f"Rekord osierocony — brak pacjenta {label}"

    # data badania jest potrzebna do obliczenia wieku pacjenta i do interpretacji wyników w czasie
    # bez niej nie możemy wybrać właściwego zakresu normy
    if not obs.get("effectiveDateTime"):
        return "yellow", "Brak effectiveDateTime"

    # sprawdzamy czy jest jakiekolwiek pole z wartością badania
    # rekord bez wartości to rekord bez sensu — nie ma co analizować
    has_any_value = any(obs.get(f) is not None for f in _VALUE_FIELDS)
    if not has_any_value:
        return "red", "Brak wartości pomiarowej"

    # szczególny przypadek: pole valueQuantity istnieje, ale wewnątrz nie ma liczby
    # to różni się od "brak pola" — ktoś wyraźnie wpisał pustkę, to bardziej podejrzane
    if "valueQuantity" in obs:
        vq = obs["valueQuantity"]
        val = vq.get("value")
        if val is None or val == "":
            return "yellow", "Wartość w valueQuantity jest pusta lub null"

    # jeśli nic powyżej nie pasowało, rekord jest strukturalnie poprawny
    return "green", "OK"
