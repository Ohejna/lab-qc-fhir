import re
from typing import Dict, Tuple


# sprowadza zapis jednostki do wspólnego formatu żeby można było je porównywać
# problem: ten sam pomiar może być zapisany jako "G/l", "x10^9/l", "x10\^9/l" czy "g/ l"
# ta funkcja usuwa spacje i zamienia \^ na ^ żeby wszystkie warianty wyglądały tak samo
def _norm(unit: str) -> str:
    if not unit:
        return ""
    s = re.sub(r"\s+", "", unit)       # usuwa wszystkie spacje (także w środku)
    s = s.replace("\\^", "^")          # "x10\^9/l" → "x10^9/l"
    return s


# główna funkcja normalizacji — próbuje dopasować jednostkę z rekordu do konfiguracji
# i zamienia ją na jednostkę referencyjną (docelową)
# zwraca trójkę: (zaktualizowany_rekord, czy_przeliczono_wartość, opis_przeliczenia)
def standardize_observation(obs: dict, ranges_config: Dict) -> Tuple[dict, bool, str]:

    # sprawdzamy pod jaką nazwą badania szukać konfiguracji
    code_text = (obs.get("code") or {}).get("text", "")
    cfg = ranges_config.get(code_text)

    # jeśli nie znamy tego badania albo rekord nie ma wartości liczbowej — nie robimy nic
    if not cfg or "valueQuantity" not in obs:
        return obs, False, ""

    vq = obs["valueQuantity"]
    current_unit: str = vq.get("unit") or ""
    ref_unit: str = cfg.get("jednostka_referencyjna", "")  # to jest cel — jednostka do której chcemy dojść

    # normalizujemy jednostkę z danych żeby porównywać jabłka z jabłkami
    norm_current = _norm(current_unit)

    # szukamy czy jednostka z rekordu pasuje do któregoś z dozwolonych aliasów
    matched_alias: str | None = None
    for alias in cfg.get("jednostki_alias", []):
        if _norm(alias) == norm_current:
            matched_alias = alias
            break

    # jednostka nie jest rozpoznana — zostawiamy rekord bez zmian
    if matched_alias is None:
        return obs, False, ""

    # sprawdzamy czy ten alias wymaga przeliczenia wartości
    # jeśli nie ma wpisu w przeliczniki, domyślnie przyjmujemy mnożnik 1.0 (tylko zmiana tekstu)
    factor: float = cfg.get("przeliczniki", {}).get(matched_alias, 1.0)
    old_value = vq.get("value")

    # mnożymy wartość przez współczynnik (dla liczb) albo zostawiamy jak jest (dla stringów)
    if isinstance(old_value, (int, float)):
        new_value = old_value * factor
    else:
        new_value = old_value

    # was_converted jest True tylko gdy faktycznie zmieniła się liczba, nie tylko tekst jednostki
    was_converted = factor != 1.0
    info = (
        f"{old_value} {current_unit} → {new_value} {ref_unit}"
        if was_converted
        else ""
    )

    # zwracamy kopię rekordu z zaktualizowaną jednostką i wartością
    obs = {**obs, "valueQuantity": {**vq, "value": new_value, "unit": ref_unit}}
    return obs, was_converted, info
