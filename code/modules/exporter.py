import json
import csv
import io
from typing import List, Dict, Set

# zestaw nazw pól wewnętrznych których nie chcemy w pliku eksportu
# te pola dodaliśmy sami podczas przetwarzania — nie należą do standardu FHIR
_INTERNAL = frozenset({
    "_patient", "_patient_id", "_category", "_category_reason",
    "_warning", "_outlier", "_outlier_reason", "_converted", "_conversion_info",
})


# usuwa z rekordu wszystkie nasze wewnętrzne pola przed eksportem
def _clean(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if k not in _INTERNAL}


# tworzy plik JSONL z zatwierdzonymi obserwacjami
# pomija rekordy czerwone (zawsze) i te które operator ręcznie usunął (deleted_ids)
def export_jsonl(observations: List[dict], deleted_ids: Set[str]) -> bytes:
    lines = []
    for obs in observations:
        # rekordy czerwone są zawsze wykluczone — mają błąd krytyczny
        if obs.get("_category") == "red":
            continue
        # rekordy ręcznie usunięte przez operatora (żółte i inne zaznaczone)
        if obs.get("id") in deleted_ids:
            continue
        # czyścimy pola wewnętrzne i serializujemy do JSON
        # ensure_ascii=False żeby polskie znaki nie były zamieniane na \uXXXX
        lines.append(json.dumps(_clean(obs), ensure_ascii=False))
    return "\n".join(lines).encode("utf-8")


# tworzy raport jakości w formacie CSV z dwiema sekcjami
# sekcja 1: ogólne statystyki (ile rekordów, ile usuniętych itp.)
# sekcja 2: lista konkretnych problemów z ID obserwacji i opisem
def export_report_csv(
    observations: List[dict],
    deleted_ids: Set[str],
    summary_stats: Dict,
) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)

    # sekcja 1 — podsumowanie liczbowe
    w.writerow(["Metryka", "Wartość"])
    for key, val in summary_stats.items():
        w.writerow([key, val])

    # pusta linia oddziela sekcje — żeby w Excelu było czytelnie
    w.writerow([])

    # sekcja 2 — lista problemów: tylko rekordy które mają jakiś problem
    w.writerow(["ID obserwacji", "ID pacjenta", "Nazwa badania", "Problem", "Wartość", "Jednostka"])
    for obs in observations:
        cat = obs.get("_category", "")
        reason = obs.get("_category_reason", "")
        warning = obs.get("_warning", "")
        outlier_reason = obs.get("_outlier_reason", "")

        # wyciągamy kod interpretacji żeby wiedzieć czy wynik jest poza normą
        interp_code = ""
        if obs.get("interpretation"):
            interp_code = (obs["interpretation"][0].get("coding") or [{}])[0].get("code", "")

        # ustalamy jaki opis problemu wpisać do raportu — priorytet: błąd > outlier > poza normą > ostrzeżenie
        problem = ""
        if cat in ("red", "yellow"):
            problem = reason
        elif outlier_reason:
            problem = outlier_reason
        elif interp_code in ("L", "H"):
            problem = f"Wynik poza normą ({interp_code})"
        elif warning:
            problem = warning

        # rekordy bez żadnego problemu pomijamy — raport ma pokazywać tylko przypadki wymagające uwagi
        if not problem:
            continue

        vq = obs.get("valueQuantity") or {}
        w.writerow([
            obs.get("id", ""),
            obs.get("_patient_id", ""),
            (obs.get("code") or {}).get("text", ""),
            problem,
            vq.get("value", ""),
            vq.get("unit", ""),
        ])

    # BOM (znak specjalny na początku) mówi Excelowi że plik jest w UTF-8
    # bez tego Excel na Windows wyświetla polskie znaki jako krzaczki
    return ("﻿" + buf.getvalue()).encode("utf-8")
